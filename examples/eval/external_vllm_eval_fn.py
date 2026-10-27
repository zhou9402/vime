"""Reference CheckpointEvalFn: eval snapshots on an external vLLM server.

Port of miles@ba28991834 examples/infra_features/fully_async/external_eval_fn.py,
with the sglang endpoints replaced by vLLM ones.

Attach to a running server:

    export VIME_EXTERNAL_EVAL_URL=http://host:8000
    --eval-function-path examples.eval.external_vllm_eval_fn.ExternalVllmEvalFn

or let it launch one itself (``vllm serve <first snapshot> --port ...``). Each
eval point pins the server: ``reload_weights`` collective RPC, then a
read-back loop on ``/weight_info`` until the served version matches, so a slow
load never silently evals stale weights. Generation then reuses the stock
eval datasets through ``retarget_args``.
"""

import asyncio
import logging
import os
import subprocess
import time

import requests

from vime.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnEvalOutput
from vime.rollout.checkpoint_eval import CheckpointEvalFn, EvalSkip, retarget_args

logger = logging.getLogger(__name__)

READY_TIMEOUT_SECS = 180.0
WEIGHT_LOAD_TIMEOUT_SECS = 600.0


class ExternalVllmEvalFn(CheckpointEvalFn):
    def __init__(self, input: RolloutFnConstructorInput):
        super().__init__(input)
        args = input.args
        self._args = args
        self._server: subprocess.Popen | None = None
        self._url = os.environ.get("VIME_EXTERNAL_EVAL_URL")
        if self._url is None:
            port = int(os.environ.get("VIME_EXTERNAL_EVAL_PORT", "8192"))
            self._url = f"http://127.0.0.1:{port}"
            self._server = subprocess.Popen(
                ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", args.hf_checkpoint, "--port", str(port)],
            )
        self._wait_ready()

    def _wait_ready(self):
        deadline = time.time() + READY_TIMEOUT_SECS
        while time.time() < deadline:
            try:
                if requests.get(f"{self._url}/health_generate", timeout=5).ok:
                    return
            except requests.RequestException:
                time.sleep(2)
        raise EvalSkip("unhealthy")

    def _pin(self, checkpoint_dir: str, weight_version: str):
        deadline = time.time() + WEIGHT_LOAD_TIMEOUT_SECS
        response = requests.post(
            f"{self._url}/collective_rpc",
            json={"method": "reload_weights", "kwargs": {"weights_path": checkpoint_dir, "is_checkpoint_format": True}},
            timeout=WEIGHT_LOAD_TIMEOUT_SECS,
        )
        response.raise_for_status()
        requests.post(f"{self._url}/update_weight_version", json={"new_version": str(weight_version)}, timeout=30)
        while time.time() < deadline:
            served = requests.get(f"{self._url}/weight_info", timeout=30).json().get("weight_version")
            if str(served) == str(weight_version):
                return
            time.sleep(5)
        raise EvalSkip("pin_violation")

    async def evaluate_checkpoint(self, checkpoint_dir: str, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        from vime.rollout.vllm_rollout import eval_rollout

        await asyncio.to_thread(self._pin, checkpoint_dir, input.weight_version or str(input.rollout_id))
        eval_args = retarget_args(
            args=self._args,
            router_ip=self._url.split("://", 1)[1].rsplit(":", 1)[0],
            router_port=int(self._url.rsplit(":", 1)[1]),
            engine_gpu_counts=[1],
        )
        output, _ = await eval_rollout(eval_args, input.rollout_id)
        return output

    async def dispose(self):
        if self._server is not None:
            self._server.terminate()
            self._server.wait(timeout=30)
