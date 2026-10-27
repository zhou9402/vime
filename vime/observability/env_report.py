"""Startup (and optional periodic) environment report: argv, env vars, pip packages, and git state
of editable installs, with all secrets hash-redacted. Degraded to plain structured logging (no
audit/event-logger subsystem). Ported from miles.utils.env_report."""

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import socket
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LAUNCHER_REPORT_ENV_VAR = "VIME_SCRIPT_ENV_REPORT"


# ---------------------------------------------------------------------------
# models (frozen dataclasses replacing miles' pydantic event-logger models)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvReportArgsDump:
    values: Any
    skipped_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EnvReportProcessFacts:
    hostname: str
    argv: list[str]
    args: EnvReportArgsDump
    env_vars: dict[str, Any]
    launcher_env_report: dict[str, Any] | None


@dataclass(frozen=True)
class EnvReportEditablePackageInfo:
    name: str
    version: str
    location: str


@dataclass(frozen=True)
class EnvReportGitRepoInfo:
    package_name: str
    location: str
    commit: str
    dirty: bool
    diff_stat: str
    uncommitted_hash: str | None
    untracked_paths: list[str]
    untracked_paths_truncated: bool
    untracked_unhashed_paths: list[str]


@dataclass(frozen=True)
class EnvReport:
    process: EnvReportProcessFacts
    key_versions: dict[str, str]
    editable_packages: list[EnvReportEditablePackageInfo]
    git_repos: list[EnvReportGitRepoInfo]
    full_pip_list: list[dict[str, str]]
    packages_probed: bool


# ---------------------------------------------------------------------------
# redaction
# ---------------------------------------------------------------------------

_SECRET_ENV_VAR_PATTERN = re.compile(
    r"(^|_)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|DATABASE_URL)(_|$)", re.IGNORECASE
)
_SECRET_ARG_NAMES = frozenset(
    {
        "wandb_key",
        "router_api_key",
        "router_oracle_password",
        "router_control_plane_api_keys",
        "oracle_password",
        "control_plane_api_keys",
    }
)
_ENV_VAR_ARG_NAMES = frozenset({"train_env_vars"})
_URL_ARG_NAMES = frozenset({"http_proxy"})
_REDACTED_PREFIX = "redacted-sha256:"
_REDACTED_HASH_CHARS = 16


def _redact(value: str) -> str:
    digest = hashlib.sha256(value.encode()).hexdigest()[:_REDACTED_HASH_CHARS]
    return f"{_REDACTED_PREFIX}{digest}"


def _redact_secret_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact(value)
    if isinstance(value, list):
        return [_redact(item) if isinstance(item, str) else item for item in value]
    return value


def redact_env_vars(env_vars: dict[str, Any]) -> dict[str, Any]:
    redacted = {}
    for name, value in sorted(env_vars.items()):
        if _SECRET_ENV_VAR_PATTERN.search(name):
            value = _redact_secret_value(value)
        redacted[name] = value
    return redacted


def _redact_url_userinfo(value: str) -> str:
    scheme, separator, rest = value.partition("://")
    if not separator:
        return value

    userinfo, at_sign, host = rest.rpartition("@")
    if not at_sign:
        return value
    return f"{scheme}://{_redact(userinfo)}@{host}"


def _redact_env_var_json(value: str) -> str:
    try:
        env_vars = json.loads(value)
    except json.JSONDecodeError:
        return _redact(value)
    if not isinstance(env_vars, dict):
        return _redact(value)
    return json.dumps(redact_env_vars({name: str(item) for name, item in env_vars.items()}))


def redact_argv(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_value: Callable[[str], str] | None = None
    for item in argv:
        if redact_value is not None and not item.startswith("-"):
            redacted.append(redact_value(item))
            continue

        redact_value = None
        flag, separator, value = item.partition("=")
        if (rewrite := _ARGV_VALUE_REDACTORS.get(flag)) is None:
            redacted.append(item)
            continue

        redacted.append(f"{flag}={rewrite(value)}" if separator else item)
        redact_value = None if separator else rewrite

    return redacted


def redact_arg(name: str, value: Any) -> Any:
    if name in _ENV_VAR_ARG_NAMES and isinstance(value, dict):
        return redact_env_vars({key: str(item) for key, item in value.items()})
    if name in _URL_ARG_NAMES and isinstance(value, str):
        return _redact_url_userinfo(value)
    if name not in _SECRET_ARG_NAMES:
        return value
    return _redact_secret_value(value)


def redact_config_values(value: Any) -> Any:
    if isinstance(value, list):
        return [redact_config_values(item) for item in value]
    if not isinstance(value, dict):
        return value

    return {
        name: redact_config_values(
            redact_env_vars(item)
            if name in {"env", "env_vars", "train_env_vars"} and isinstance(item, dict)
            else redact_arg(name, item)
        )
        for name, item in value.items()
    }


def _argv_value_redactors() -> dict[str, Callable[[str], str]]:
    def flag(name: str) -> str:
        return f"--{name.replace('_', '-')}"

    return {
        "--config": _redact_env_var_json,
        **{flag(name): _redact for name in _SECRET_ARG_NAMES},
        **{flag(name): _redact_env_var_json for name in _ENV_VAR_ARG_NAMES},
        **{flag(name): _redact_url_userinfo for name in _URL_ARG_NAMES},
    }


_ARGV_VALUE_REDACTORS = _argv_value_redactors()


# ---------------------------------------------------------------------------
# git state
# ---------------------------------------------------------------------------

_HEAD_MOVE_ATTEMPTS = 2
_UNTRACKED_MAX_FILES = 1000
_UNTRACKED_MAX_FILE_BYTES = 10 * 1024 * 1024
_PATCH_SECTION_STARTS = (b"diff --git ", b"diff --cc ", b"diff --combined ", b"* Unmerged path ")


class _HeadMoved(Exception):
    pass


@dataclass(frozen=True)
class _GitDiff:
    stat: str
    patch: bytes | None


@dataclass(frozen=True)
class _UntrackedFiles:
    paths: list[str]
    hash_entries: list[bytes]
    paths_truncated: bool
    total_count: int
    unhashed_paths: list[str]


def _run_git(*, args: list[str], location: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        timeout=10,
        cwd=location,
    )


def _decode(raw: bytes) -> str:
    return raw.decode(errors="replace")


def collect_git_info(*, package_name: str, location: str) -> EnvReportGitRepoInfo | None:
    if not location or not os.path.isdir(location):
        return None
    for _ in range(_HEAD_MOVE_ATTEMPTS):
        try:
            return _collect_stable_git_info(package_name=package_name, location=location)
        except _HeadMoved:
            continue
    logger.warning("Gave up collecting the git state of %s at %s, its HEAD keeps moving", package_name, location)
    return None


def _collect_stable_git_info(*, package_name: str, location: str) -> EnvReportGitRepoInfo | None:
    try:
        commit_result = _run_git(args=["rev-parse", "HEAD"], location=location)
        if commit_result.returncode != 0:
            return None

        diff = _collect_diff(location=location)
        untracked = _collect_untracked(location=location)

        if _run_git(args=["rev-parse", "HEAD"], location=location).stdout != commit_result.stdout:
            raise _HeadMoved

        return EnvReportGitRepoInfo(
            package_name=package_name,
            location=location,
            commit=_decode(commit_result.stdout).strip(),
            dirty=bool(diff.stat) or bool(untracked.paths),
            diff_stat=diff.stat,
            uncommitted_hash=_hash_uncommitted(diff_patch=diff.patch, untracked=untracked),
            untracked_paths=untracked.paths,
            untracked_paths_truncated=untracked.paths_truncated,
            untracked_unhashed_paths=untracked.unhashed_paths,
        )
    except _HeadMoved:
        raise
    except Exception:
        logger.warning("Failed to collect git info for %s at %s", package_name, location, exc_info=True)
        return None


def _collect_diff(*, location: str) -> _GitDiff:
    result = _run_git(args=["diff", "--patch-with-stat", "HEAD"], location=location)
    if result.returncode != 0:
        return _GitDiff(stat="", patch=None)

    if (patch_start := _find_patch_start(stdout=result.stdout)) is None:
        return _GitDiff(stat=_decode(result.stdout).strip(), patch=b"")

    return _GitDiff(stat=_decode(result.stdout[:patch_start]).strip(), patch=result.stdout[patch_start:])


def _find_patch_start(*, stdout: bytes) -> int | None:
    sentinelled = b"\n" + stdout
    starts = [found for start in _PATCH_SECTION_STARTS if (found := sentinelled.find(b"\n" + start)) >= 0]
    return min(starts) if starts else None


def _hash_uncommitted(*, diff_patch: bytes | None, untracked: _UntrackedFiles) -> str | None:
    if diff_patch is None:
        return None

    digest = hashlib.sha256()
    digest.update(b"diff\0%d\0" % len(diff_patch))
    digest.update(diff_patch)
    for entry in untracked.hash_entries:
        digest.update(b"untracked\0" + entry + b"\0")
    if untracked.paths_truncated:
        digest.update(b"untracked-truncated\0%d\0" % untracked.total_count)

    return digest.hexdigest()


def _collect_untracked(*, location: str) -> _UntrackedFiles:
    root = _resolve_repo_root(location=location)

    result = _run_git(args=["ls-files", "--others", "--exclude-standard", "-z"], location=str(root))
    if result.returncode != 0:
        return _UntrackedFiles(paths=[], hash_entries=[], paths_truncated=False, total_count=0, unhashed_paths=[])

    all_paths = sorted(path for path in result.stdout.split(b"\0") if path)
    selected = all_paths[:_UNTRACKED_MAX_FILES]

    paths: list[str] = []
    hash_entries: list[bytes] = []
    unhashed_paths: list[str] = []
    for raw_path in selected:
        entry, hashed = _untracked_hash_entry(root=root, raw_path=raw_path)
        paths.append(_decode(raw_path))
        hash_entries.append(entry)
        if not hashed:
            unhashed_paths.append(_decode(raw_path))

    return _UntrackedFiles(
        paths=paths,
        hash_entries=hash_entries,
        paths_truncated=len(all_paths) > len(selected),
        total_count=len(all_paths),
        unhashed_paths=unhashed_paths,
    )


def _resolve_repo_root(*, location: str) -> Path:
    result = _run_git(args=["rev-parse", "--show-toplevel"], location=location)
    if result.returncode != 0 or not (toplevel := _decode(result.stdout).strip()):
        return Path(location)
    return Path(toplevel)


def _untracked_hash_entry(*, root: Path, raw_path: bytes) -> tuple[bytes, bool]:
    full_path = root / os.fsdecode(raw_path)
    try:
        if (size := full_path.stat().st_size) > _UNTRACKED_MAX_FILE_BYTES:
            return b"%s\0size:%d" % (raw_path, size), False
        content_digest = hashlib.sha256(full_path.read_bytes()).hexdigest()
        return b"%s\0sha256:%s" % (raw_path, content_digest.encode()), True
    except OSError:
        logger.warning("Failed to hash the untracked file %s", full_path, exc_info=True)
        return b"%s\0unreadable" % raw_path, False


# ---------------------------------------------------------------------------
# launcher report
# ---------------------------------------------------------------------------


def read_launcher_report(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        return _as_report_object(json.loads(Path(path).read_text()))
    except Exception:
        logger.warning("Failed to read the launcher report at %s", path, exc_info=True)
        return None


def _as_report_object(decoded: Any) -> dict[str, Any] | None:
    if isinstance(decoded, dict):
        return decoded
    logger.warning("The launcher report is valid json but not an object, so it names nothing: %r", decoded)
    return None


# ---------------------------------------------------------------------------
# collector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvReportSnapshot:
    facts: EnvReportProcessFacts
    probe_env: dict[str, str]


@dataclass(frozen=True)
class _PipInfo:
    probed: bool
    editable_packages: list[EnvReportEditablePackageInfo] = field(default_factory=list)
    full_pip_list: list[dict[str, str]] = field(default_factory=list)


_KEY_PACKAGE_NAMES = (
    "vime",
    "vllm",
    "megatron-core",
    "transformers",
    "ray",
    "flashinfer-python",
)


def _config_values(args: Any) -> Any:
    if isinstance(args, argparse.Namespace):
        return vars(args)
    return args


def collect_env_report_snapshot(args: Any) -> EnvReportSnapshot:
    environ = dict(os.environ)
    env_vars = {name: value for name, value in environ.items() if name != LAUNCHER_REPORT_ENV_VAR}

    facts = EnvReportProcessFacts(
        hostname=socket.gethostname(),
        argv=redact_argv(sys.argv),
        args=_dump_args(args),
        env_vars=redact_env_vars(env_vars),
        launcher_env_report=read_launcher_report(
            getattr(args, "env_report", None) or environ.get(LAUNCHER_REPORT_ENV_VAR, "")
        ),
    )
    return EnvReportSnapshot(facts=facts, probe_env={k: v for k, v in environ.items() if k != "PYTHONPATH"})


def collect_env_report(*, snapshot: EnvReportSnapshot) -> EnvReport:
    pip_info = _collect_pip_info(snapshot.probe_env)

    git_repos = [
        info
        for pkg in pip_info.editable_packages
        if (info := collect_git_info(package_name=pkg.name, location=pkg.location))
    ]

    return EnvReport(
        process=snapshot.facts,
        key_versions=_collect_key_versions(pip_info.full_pip_list),
        editable_packages=pip_info.editable_packages,
        git_repos=git_repos,
        full_pip_list=pip_info.full_pip_list,
        packages_probed=pip_info.probed,
    )


def collect_unprobed_env_report(*, snapshot: EnvReportSnapshot) -> EnvReport:
    return EnvReport(
        process=snapshot.facts,
        key_versions=_collect_key_versions([]),
        editable_packages=[],
        git_repos=[],
        full_pip_list=[],
        packages_probed=False,
    )


def _collect_key_versions(full_pip_list: list[dict[str, str]]) -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    installed = {entry["name"].lower(): entry["version"] for entry in full_pip_list}
    versions.update({name: installed[name] for name in _KEY_PACKAGE_NAMES if name in installed})

    if (torch := sys.modules.get("torch")) is not None:
        versions["torch"] = torch.__version__
        versions["torch_cuda"] = torch.version.cuda or ""

    return versions


def _dump_args(args: Any) -> EnvReportArgsDump:
    values = _json_snapshot(_config_values(args), ancestors=frozenset())
    return EnvReportArgsDump(values=redact_config_values(values), skipped_names=[])


def _json_safe(value: Any) -> Any:
    """Best-effort JSON-safe rendering for non-JSON-native values."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        return repr(value)


def _json_snapshot(value: Any, *, ancestors: frozenset[int]) -> Any:
    try:
        if id(value) in ancestors:
            raise ValueError("Circular reference detected")
        ancestors = ancestors | {id(value)}
        if isinstance(value, argparse.Namespace):
            value = _config_values(value)
        elif is_dataclass(value) and not isinstance(value, type):
            value = {item.name: value.__getattribute__(item.name) for item in fields(value)}

        if isinstance(value, Mapping):
            return {
                (name if isinstance(name, str) else json.loads(json.dumps({name: None})).popitem()[0]): _json_snapshot(
                    item, ancestors=ancestors
                )
                for name, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [_json_snapshot(item, ancestors=ancestors) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return json.loads(json.dumps(value, allow_nan=False))
        return _json_safe(value)
    except (TypeError, ValueError, RecursionError) as error:
        return {
            "$serialization_error": f"{type(error).__name__}: Cannot serialize {type(value).__module__}.{type(value).__qualname__}"
        }


def _collect_pip_info(env: dict[str, str]) -> _PipInfo:
    """Collect all pip info in a single `pip inspect` call."""
    try:
        result = subprocess.run(
            ["pip", "inspect"],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        if result.returncode != 0:
            logger.warning("pip inspect failed: %s", result.stderr)
            return _PipInfo(probed=False)

        data = json.loads(result.stdout)
        installed: list[dict[str, Any]] = data.get("installed", [])

        full_pip_list = [_parse_pip_entry(pkg) for pkg in installed]
        editable_packages = [
            EnvReportEditablePackageInfo(
                name=entry["name"],
                version=entry["version"],
                location=pkg["direct_url"]["url"].removeprefix("file://"),
            )
            for pkg, entry in zip(installed, full_pip_list, strict=True)
            if _is_editable(pkg)
        ]

        return _PipInfo(editable_packages=editable_packages, full_pip_list=full_pip_list, probed=True)
    except Exception:
        logger.warning("Failed to collect pip info", exc_info=True)
        return _PipInfo(probed=False)


def _parse_pip_entry(pkg: dict[str, Any]) -> dict[str, str]:
    metadata = pkg.get("metadata", {})
    return {"name": metadata.get("name", ""), "version": metadata.get("version", "")}


def _is_editable(pkg: dict[str, Any]) -> bool:
    direct_url = pkg.get("direct_url")
    return bool(direct_url and direct_url.get("dir_info", {}).get("editable"))


# ---------------------------------------------------------------------------
# reporter
# ---------------------------------------------------------------------------

_REPORTER_THREAD_NAME = "env-report"
_INTERVAL_JITTER_RATIO = 0.5
_STOP_TIMEOUT_SECONDS = 5.0
_SUMMARY_HASH_CHARS = 12
SETTLED_DELAY_SECONDS = 300.0


def start_env_reporting(args: Any) -> "EnvReporter | None":
    try:
        reporter = EnvReporter(args=args, interval_seconds=args.env_report_interval_seconds)
        reporter.start()
        return reporter
    except Exception:
        logger.warning("Failed to start the env reporting of this process", exc_info=True)
        return None


class EnvReporter:
    def __init__(self, *, args: Any, interval_seconds: float) -> None:
        assert math.isfinite(interval_seconds), (
            f"--env-report-interval-seconds is {interval_seconds}, which is neither a delay nor a way to say "
            f"'only at startup'; pass a finite number"
        )

        self._interval_seconds = interval_seconds
        self._jitter = random.Random()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name=_REPORTER_THREAD_NAME, daemon=True)
        self.rebind(args)

    def rebind(self, args: Any) -> None:
        self._args = args
        try:
            _log_env_report(report=collect_unprobed_env_report(snapshot=collect_env_report_snapshot(args)))
        except Exception:
            logger.warning("Failed to log the startup env report", exc_info=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=_STOP_TIMEOUT_SECONDS)

    def _run(self) -> None:
        delay = min(SETTLED_DELAY_SECONDS, self._interval_seconds) if self._interval_seconds > 0 else 0.0
        while True:
            self._report_once()
            if self._interval_seconds <= 0 or self._stopped.wait(delay):
                return
            delay = self._next_delay_seconds()

    def _report_once(self) -> None:
        try:
            snapshot = collect_env_report_snapshot(self._args)
            _log_env_report(report=collect_env_report(snapshot=snapshot))
        except Exception:
            logger.warning("Failed to log the env report", exc_info=True)

    def _next_delay_seconds(self) -> float:
        jittered = self._interval_seconds * (1.0 + self._jitter.random() * _INTERVAL_JITTER_RATIO)
        return min(jittered, threading.TIMEOUT_MAX)


def _log_env_report(*, report: EnvReport) -> EnvReport:
    logger.info("env_report %s", json.dumps(asdict(report), default=repr))
    _log_report_summary(report)
    return report


def _log_report_summary(report: EnvReport) -> None:
    logger.info(
        "env_report_summary hostname=%s versions=%s repos=%s num_packages=%d num_env_vars=%d packages_probed=%s",
        report.process.hostname,
        report.key_versions,
        {repo.package_name: _summarise_repo(repo) for repo in report.git_repos},
        len(report.full_pip_list),
        len(report.process.env_vars),
        report.packages_probed,
    )


def _summarise_repo(repo: EnvReportGitRepoInfo) -> str:
    if not repo.dirty:
        return repo.commit
    return f"{repo.commit}-dirty-{repo.uncommitted_hash[:_SUMMARY_HASH_CHARS] if repo.uncommitted_hash else 'unknown'}"
