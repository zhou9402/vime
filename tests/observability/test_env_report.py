"""env_report: argv/pip/git snapshot with secret redaction (CPU-only)."""

import argparse
import json
import re
import subprocess
import sys
import threading

import pytest

from vime.observability import env_report as er


def _make_git_repo(tmp_path, dirty=True):
    repo = tmp_path / "pkg"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("v1")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    if dirty:
        (repo / "tracked.txt").write_text("v2")
    return repo, commit


def test_report_contains_git_pip_argv(tmp_path, monkeypatch):
    repo, commit = _make_git_repo(tmp_path)
    monkeypatch.setattr(
        er,
        "_collect_pip_info",
        lambda env: er._PipInfo(
            probed=True,
            editable_packages=[er.EnvReportEditablePackageInfo(name="pkg", version="0.1", location=str(repo))],
            full_pip_list=[{"name": "pkg", "version": "0.1"}, {"name": "ray", "version": "2.0"}],
        ),
    )
    monkeypatch.setattr(sys, "argv", ["prog", "--lr", "1e-5"])
    args = argparse.Namespace(env_report="", lr=1e-5)

    report = er.collect_env_report(snapshot=er.collect_env_report_snapshot(args))

    assert report.process.argv == ["prog", "--lr", "1e-5"]
    assert report.process.args.values["lr"] == 1e-5
    assert report.git_repos[0].commit == commit
    assert report.git_repos[0].dirty is True
    assert report.git_repos[0].uncommitted_hash
    assert report.key_versions["python"]
    assert report.key_versions["ray"] == "2.0"
    assert report.packages_probed is True


def test_env_var_secret_redaction(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "wandb-secret")
    monkeypatch.setenv("MY_DB_PASSWORD", "db-secret")
    monkeypatch.setenv("PLAIN_VAR", "visible")
    redacted = er.redact_env_vars(dict(__import__("os").environ))
    assert re.fullmatch(r"redacted-sha256:[0-9a-f]{16}", redacted["WANDB_API_KEY"])
    assert re.fullmatch(r"redacted-sha256:[0-9a-f]{16}", redacted["MY_DB_PASSWORD"])
    assert redacted["PLAIN_VAR"] == "visible"


def test_secret_word_any_segment_redacted(monkeypatch):
    """KEY/TOKEN/SECRET/PASSWORD/PASSWD/CREDENTIAL(S) count in ANY _-delimited segment."""
    monkeypatch.setenv("HF_TOKEN_PATH", "t1")
    monkeypatch.setenv("OPENAI_API_KEY_BACKUP", "t2")
    monkeypatch.setenv("DATABASE_URL", "t3")
    monkeypatch.setenv("MONKEY", "not-a-secret")  # KEY is not a separate segment here
    monkeypatch.setenv("KEYCLOAK_URL", "not-a-secret")
    redacted = er.redact_env_vars(dict(__import__("os").environ))
    for name in ["HF_TOKEN_PATH", "OPENAI_API_KEY_BACKUP", "DATABASE_URL"]:
        assert re.fullmatch(r"redacted-sha256:[0-9a-f]{16}", redacted[name]), name
    assert redacted["MONKEY"] == "not-a-secret"
    assert redacted["KEYCLOAK_URL"] == "not-a-secret"


def test_argv_secret_redaction():
    out = er.redact_argv(["prog", "--wandb-key", "secret", "--wandb-key=secret2"])
    assert out[0] == "prog"
    assert re.fullmatch(r"redacted-sha256:[0-9a-f]{16}", out[2])
    flag, _, value = out[3].partition("=")
    assert flag == "--wandb-key"
    assert re.fullmatch(r"redacted-sha256:[0-9a-f]{16}", value)


def test_redaction_hash_stable():
    assert er._redact("same") == er._redact("same")
    assert er._redact("same") != er._redact("different")


def test_proxy_userinfo_redacted():
    out = er.redact_argv(["prog", "--http-proxy", "http://user:pass@host:3128"])
    scheme, _, rest = out[2].partition("://")
    userinfo, _, host = rest.rpartition("@")
    assert re.fullmatch(r"redacted-sha256:[0-9a-f]{16}", userinfo)
    assert host == "host:3128"


def test_reporter_startup_once():
    args = argparse.Namespace(env_report="", env_report_interval_seconds=0.0)
    reporter = er.EnvReporter(args=args, interval_seconds=0.0)
    reporter.start()
    reporter._thread.join(timeout=10)
    assert not reporter._thread.is_alive()


def test_pip_inspect_failure_degrades(monkeypatch):
    def fake_run(*a, **k):
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(er.subprocess, "run", fake_run)
    info = er._collect_pip_info({})
    assert info.probed is False
    assert info.full_pip_list == []


def test_periodic_thread_stops_cleanly():
    args = argparse.Namespace(env_report="", env_report_interval_seconds=0.05)
    reporter = er.EnvReporter(args=args, interval_seconds=0.05)
    reporter.start()
    assert any(t.name == "env-report" for t in threading.enumerate())
    reporter.stop()
    assert not reporter._thread.is_alive()
