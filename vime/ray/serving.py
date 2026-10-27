"""Keep serving alive independently of trainers and rollout managers.

This actor owns persistent topology: placements, routers, engines, the queue
controller and the weight-update lock. Managers receive snapshots of engine
handles; health checks and recovery mutate only this owner's topology.
"""

import copy
import logging
import multiprocessing
from dataclasses import dataclass

import ray

from vime.ray.training_recovery import retained_rollout_configuration
from vime.ray.utils import Lock
from vime.utils.cleanup import Cleanup
from vime.utils.health_monitor import RolloutHealthMonitor


@dataclass
class ServingDeployment:
    """Startup handoff; ordinary health checks return only the engine snapshot."""

    placements: dict
    servers: dict
    controller: object
    restore_plan: object
    routers: dict
    engine_lock: object
    reused: bool = False


@ray.remote
class ServingCluster:
    """A named, detached owner for routers, engines and their GPU placements."""

    def __init__(self, args, restore_plan):
        from vime.backends.vllm_utils.deployment import start_rollout_servers
        from vime.observability.logging_utils import configure_logger
        from vime.ray.placement_group import create_placement_groups

        configure_logger()
        self.args = args
        self.restore_plan = restore_plan
        self.configuration = retained_rollout_configuration(args)
        self.placements = {}
        self.servers = {}
        self.router_processes = []
        self.controller = None
        self.lock = None
        self._policy_locks = {}
        self._policy_binding = None
        self.driver_job_id = None
        self.training_actors = {}
        self.attachments = 0
        self._health_monitors = []
        self._ci_fault_injection_pending = args.ci_test
        existing_children = {process.pid for process in multiprocessing.active_children()}
        try:
            # Constructor failure must release partial topology too. A detached
            # actor can fail before any driver obtains a usable deployment.
            self.placements = create_placement_groups(args)
            try:
                self.servers, handles = (
                    ({}, []) if args.debug_train_only else start_rollout_servers(args, self.placements["rollout"])
                )
            finally:
                self.router_processes = [
                    process for process in multiprocessing.active_children() if process.pid not in existing_children
                ]
            ray.get(handles)
            self.lock = Lock.options(num_cpus=0, num_gpus=0).remote()
            self._attach_policy_config()
        except BaseException:
            try:
                self.dispose()
            except Exception:
                logging.getLogger(__name__).exception("Failed to dispose partial serving deployment")
            raise

    def get_queue_controller(self):
        """Create only when the built-in source or batch builder needs a queue."""
        if self.controller is None:
            from vime.data.queue_data_source import create_queue_controller

            # Managers borrow this controller; recreating a manager must not
            # create a second queue or lose accepted generation results.
            self.controller = create_queue_controller(self.args, restore_plan=self.restore_plan)
        return self.controller

    def deployment(self, reused=False):
        routers = {
            name: getattr(self.args, name)
            for name in ("vllm_router_ip", "vllm_router_port", "vllm_model_routers")
            if hasattr(self.args, name)
        }
        return ServingDeployment(
            placements=self.placements,
            servers=self.servers,
            controller=self.controller,
            restore_plan=self.restore_plan,
            routers=routers,
            engine_lock=self.lock,
            reused=reused,
        )

    def validate_attachment(self, args):
        configuration = retained_rollout_configuration(args)
        changed = [
            name
            for name in self.configuration.keys() | configuration.keys()
            if name not in self.configuration
            or name not in configuration
            or self.configuration[name] != configuration[name]
        ]
        if changed:
            raise ValueError("Retained serving requires unchanged rollout/model configuration: " + ", ".join(changed))
        if self.driver_job_id is not None:
            # Actor existence identifies the session, not whether its previous
            # trainer is dead. Fence live/unknown drivers before taking ownership.
            from ray._private.state import jobs

            previous = next((job for job in jobs() if job["JobID"] == self.driver_job_id), None)
            if previous is None or not previous["IsDead"]:
                raise RuntimeError(f"Serving session is still owned by training job {self.driver_job_id}")

    def attach_training(self, args, job_id):
        from ray.util.placement_group import remove_placement_group

        from vime.backends.vllm_utils.engine_group import reset_weights_update_groups
        from vime.ray.placement_group import _create_placement_group

        self.validate_attachment(args)
        args = copy.deepcopy(args)
        # Only a validated successor may touch the old trainer's resources.
        # Stop health checks before replacing connections or GPU allocations.
        self.health_monitoring_pause()
        self.release_trainers()
        reused = self.attachments > 0
        count = args.actor_num_nodes * args.actor_num_gpus_per_node
        actor_pg = self.placements["actor"]
        if args.colocate:
            # This placement is shared with live serving, so it cannot be resized.
            if count > len(actor_pg[1]):
                raise ValueError("Restarted colocated trainer exceeds the retained GPU placement")
        elif not args.debug_rollout_only and count != len(actor_pg[1]):
            if actor_pg[0] is not None:
                remove_placement_group(actor_pg[0])
            self.placements["actor"] = _create_placement_group(count)
            self.placements["critic"] = self.placements["actor"] if args.use_critic else None
        if reused:
            # NCCL groups and a possibly held lock belong to the old trainer.
            # Engines survive; their next update reconnects to the new trainer.
            reset_weights_update_groups(
                [group for server in self.servers.values() for group in server.server_groups],
                timeout=args.rollout_health_check_timeout,
            )
            self._recreate_locks()
            server = self._updatable_server()
            if server:
                # Healthy engines are also new peers from the trainer's point
                # of view; mark them for connection on the next weight update.
                for group in server.server_groups:
                    group.num_new_engines = len([engine for engine in group.engines if engine is not None])
        # Internal serving is always monitored, independently of the legacy flag.
        # Rebuild paused monitors using this attempt's health-check settings. The
        # manager resumes them only when generation can safely use the engines.
        for monitor in self._health_monitors:
            monitor.stop()
        self._health_monitors = []
        for server in self.servers.values():
            for group in server.server_groups:
                monitor = RolloutHealthMonitor(group, args)
                monitor.start()
                self._health_monitors.append(monitor)
        for name, value in self.deployment().routers.items():
            setattr(args, name, value)
        self.args = args
        self._attach_policy_config()
        self.driver_job_id = job_id
        self.attachments += 1
        return self.deployment(reused)

    def try_ci_fault_injection(self):
        server = self._updatable_server()
        if not self._ci_fault_injection_pending or server is None:
            return self.servers
        self._ci_fault_injection_pending = False
        engines = [engine for engine in server.all_engines if engine is not None]
        if engines:
            ray.get(engines[0].simulate_crash.remote(), timeout=self.args.rollout_health_check_timeout)
            for monitor in self._health_monitors:
                monitor.check_once()
        return self.servers

    def register_trainers(self, role, actors):
        # Keep cleanup handles outside the manager, which may itself crash.
        self.training_actors[role] = actors

    def release_trainers(self):
        with Cleanup() as cleanup:
            for actors in self.training_actors.values():
                for actor in actors[:]:
                    # Retain failed handles so a repeated detach can retry them.
                    def release(actor=actor, actors=actors):
                        ray.kill(actor, no_restart=True)
                        actors.remove(actor)

                    cleanup.run("terminate training actor", release)

    def detach_training(self, job_id, timeout=None):
        # A late cleanup from an old driver must not detach its successor.
        if self.driver_job_id is None:
            return  # A repeated cleanup after a lost RPC reply is harmless.
        if self.driver_job_id != job_id:
            raise RuntimeError("Only the owning driver can detach its serving session")
        with Cleanup(getattr(self.args, "rollout_cleanup_timeout", 60) if timeout is None else timeout) as cleanup:
            cleanup.run("pause serving health checks", self.health_monitoring_pause, timeout=cleanup.remaining)
            cleanup.run("release trainers", self.release_trainers)
        # Keep engines, routers, placements and the queue for the next attempt.
        self.driver_job_id = None

    def _recreate_locks(self):
        """Kill and recreate the global lock and every per-policy lock.

        A possibly held lock belongs to the old trainer; the next update
        reconnects through fresh actors. Server engine_lock references are
        reassigned by the _attach_policy_config() call that follows a reattach.
        """
        ray.kill(self.lock, no_restart=True)
        self.lock = Lock.options(num_cpus=0, num_gpus=0).remote()
        for policy_lock in self._policy_locks.values():
            ray.kill(policy_lock, no_restart=True)
        self._policy_locks = {}

    def _attach_policy_config(self):
        """Validate the policy->server binding ONCE and hand each server its lock.

        Every RolloutServer holds an ``engine_lock``: policy servers get a
        dedicated per-policy lock, everything else shares the global
        ``self.lock``. Called from __init__ and whenever training re-attaches
        (locks and possibly the config are recreated there).
        """
        from vime.utils.multi_policy.serving_config import bind_policy_servers

        config = getattr(self.args, "multi_policy", None)
        self._policy_binding = bind_policy_servers(self.servers, config) if config is not None else None
        policy_server_names = set(self._policy_binding.values()) if self._policy_binding else set()
        for name, server in self.servers.items():
            if name in policy_server_names:
                if name not in self._policy_locks:
                    self._policy_locks[name] = Lock.options(num_cpus=0, num_gpus=0).remote()
                server.engine_lock = self._policy_locks[name]
            else:
                server.engine_lock = self.lock

    def _updatable_server(self, model_id=None):
        from vime.utils.multi_policy.serving_config import resolve_updatable_server

        return resolve_updatable_server(self.servers, model_id, self._policy_binding)

    def get_engine_lock(self, model_id=None):
        """The lock the policy's server itself holds; global lock for single-policy runs."""
        if model_id is None:
            return self.lock
        server = self._updatable_server(model_id)
        return server.engine_lock if server is not None else self.lock

    def get_updatable_engines_and_lock(self, model_id=None):
        server = self._updatable_server(model_id)
        if server is None:
            return [], self.get_engine_lock(model_id), 0, [], [], []
        return (
            server.engines,
            self.get_engine_lock(model_id),
            server.num_new_engines,
            server.engine_gpu_counts,
            server.engine_gpu_offsets,
            server.engine_parallel_configs,
        )

    def get_weight_version(self, model_id=None, *, allow_inconsistent=False):
        from vime.utils.multi_policy.serving_config import consistent_weight_version

        server = self._updatable_server(model_id)
        engines = [engine for engine in server.engines if engine is not None] if server else []
        versions = ray.get(
            [engine.get_weight_version.remote() for engine in engines], timeout=self.args.rollout_health_check_timeout
        )
        return consistent_weight_version(versions, allow_inconsistent=allow_inconsistent)

    def recover_updatable_engines(self):
        # Recovery waits until the trainer can immediately install its weights.
        # Rollout completion only prunes failed engines; it never starts them.
        self.health_monitoring_pause()
        server = self._updatable_server()
        # Fresh and replacement trainers must connect to every retained engine.
        # A no-op recover() would clear num_new_engines and lose those markers,
        # regardless of whether the rollout manager was replaced or survived.
        if server and any(engine is None for engine in server.all_engines):
            server.recover()
        return self.servers

    def clear_updatable_num_new_engines(self):
        server = self._updatable_server()
        if server:
            server.num_new_engines = 0

    def health_monitoring_pause(self, timeout=None):
        if not self._health_monitors:
            return
        with Cleanup(self.args.rollout_health_check_timeout if timeout is None else timeout) as cleanup:
            for monitor in self._health_monitors:
                cleanup.run("pause health monitor", monitor.pause, timeout=cleanup.remaining)

    def finish_rollout(self):
        """Remove failed engines before the trainer sends control requests."""
        self.health_monitoring_pause()
        # A short rollout may finish before any periodic check. Check once even
        # during warmup grace, while weights/KV are still resident for probing.
        for monitor in self._health_monitors:
            monitor.check_once()
        return self.servers

    def health_monitoring_resume(self):
        for monitor in self._health_monitors:
            monitor.resume()
        return self.servers

    def dispose(self, timeout=None):
        from ray.util.placement_group import remove_placement_group

        # Full teardown is for successful completion. Failure cleanup uses
        # detach_training instead, preserving the resources needed to resume.
        with Cleanup(getattr(self.args, "rollout_cleanup_timeout", 60) if timeout is None else timeout) as cleanup:
            cleanup.run("release trainers", self.release_trainers)
            for monitor in self._health_monitors:
                cleanup.run("stop health monitor", monitor.stop, timeout=cleanup.remaining)
            engines = [
                engine for server in self.servers.values() for engine in server.all_engines if engine is not None
            ]
            shutdowns = []
            for engine in engines:
                ref = cleanup.run("request engine shutdown", engine.shutdown.remote)
                if ref is not None:
                    shutdowns.append(ref)
            if shutdowns:
                cleanup.run(
                    "wait for engines", ray.wait, shutdowns, num_returns=len(shutdowns), timeout=cleanup.remaining / 2
                )
            for engine in engines:
                cleanup.run("terminate engine", ray.kill, engine, no_restart=True)
            for router in self.router_processes:
                if router.is_alive():
                    cleanup.run("terminate router", router.terminate)
                    cleanup.run("join router", router.join, timeout=min(10, cleanup.remaining / 2))
                    if router.is_alive():
                        cleanup.run("kill router", router.kill)
                        cleanup.run("reap router", router.join, timeout=cleanup.remaining)
            if self.controller is not None:
                cleanup.run(
                    "close queue controller",
                    lambda: ray.get(self.controller.close.remote(), timeout=cleanup.remaining),
                )
                cleanup.run("terminate queue controller", ray.kill, self.controller, no_restart=True)
            if self.lock is not None:
                cleanup.run("terminate weight lock", ray.kill, self.lock, no_restart=True)
            for policy_lock in self._policy_locks.values():
                cleanup.run("terminate policy weight lock", ray.kill, policy_lock, no_restart=True)
            self._policy_locks = {}
            # Colocated actor/critic/rollout entries can share the same placement.
            for group in {placement[0] for placement in self.placements.values() if placement and placement[0]}:
                cleanup.run("remove serving placement", remove_placement_group, group)
