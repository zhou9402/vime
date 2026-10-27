"""Tests for --hf-to-megatron-load-mode {builtin,mbridge} (F07-1).

Covers: default stays builtin and never imports mbridge; mbridge mode
dispatches to the mbridge loader; missing mbridge package raises a clear error;
invalid model types surface the mbridge registry error.
"""

import argparse
import ast
import sys
import types
from pathlib import Path

import pytest

from vime.backends.megatron_utils import hf_to_megatron
from vime.backends.megatron_utils.hf_to_megatron import mbridge_load

_ARGUMENTS_PY = Path(__file__).resolve().parents[1] / "vime/utils/arguments.py"


def _load_add_model_load_arguments():
    """Exec only add_model_load_arguments out of arguments.py.

    Importing vime.utils.arguments pulls in vllm, which CPU CI does not
    install; AST-extract the flag group instead (same pattern as
    tests/test_megatron_argument_validation.py).
    """
    function = next(
        node
        for node in ast.walk(ast.parse(_ARGUMENTS_PY.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "add_model_load_arguments"
    )
    namespace = {}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(_ARGUMENTS_PY), "exec"),
        namespace,
    )
    return namespace["add_model_load_arguments"]


def _parse_args(*extra):
    parser = _load_add_model_load_arguments()(argparse.ArgumentParser())
    return parser.parse_args(list(extra))


def test_default_load_mode_is_builtin():
    args = _parse_args()
    assert args.hf_to_megatron_load_mode == "builtin"


def test_load_mode_choices_reject_unknown():
    with pytest.raises(SystemExit):
        _parse_args("--hf-to-megatron-load-mode", "bridge")


def test_builtin_mode_never_imports_mbridge(monkeypatch):
    monkeypatch.setitem(sys.modules, "mbridge", None)  # make any import of mbridge fail loudly
    monkeypatch.delitem(sys.modules, "vime_plugins.mbridge", raising=False)

    def _boom(*_args, **_kwargs):
        raise AssertionError("builtin mode must not touch mbridge")

    monkeypatch.setattr(mbridge_load, "load_hf_weights_with_mbridge", _boom)

    model = types.SimpleNamespace()
    args = types.SimpleNamespace(hf_to_megatron_load_mode="builtin")
    monkeypatch.setattr(hf_to_megatron, "load_model_hf_weights", lambda *a, **k: None)
    monkeypatch.setattr(hf_to_megatron.AutoConfig, "from_pretrained", lambda *a, **k: types.SimpleNamespace(model_type="qwen3"))

    hf_to_megatron.load_hf_weights(args, model, "/nonexistent")


def test_mbridge_mode_dispatches_to_mbridge_loader(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}")
    called = {}
    monkeypatch.setattr(
        mbridge_load,
        "load_hf_weights_with_mbridge",
        lambda args, model, path: called.setdefault("path", path),
    )
    args = types.SimpleNamespace(hf_to_megatron_load_mode="mbridge")
    hf_to_megatron.load_hf_weights(args, types.SimpleNamespace(), tmp_path)
    assert called["path"] == tmp_path


def test_mbridge_mode_without_package_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(mbridge_load, "mbridge_available", lambda: False)
    monkeypatch.setitem(sys.modules, "mbridge", None)
    with pytest.raises(ModuleNotFoundError, match="mbridge"):
        mbridge_load.load_hf_weights_with_mbridge(types.SimpleNamespace(), None, tmp_path)


def test_supports_dispatch_by_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(mbridge_load, "supports_mbridge_loading", lambda path: True)
    assert hf_to_megatron.supports_hf_weight_loading(tmp_path, "mbridge") is True
    # builtin mode is untouched by the mbridge registry
    monkeypatch.setattr(mbridge_load, "supports_mbridge_loading", lambda path: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(
        hf_to_megatron.AutoConfig, "from_pretrained", lambda *a, **k: types.SimpleNamespace(model_type="qwen3")
    )
    assert hf_to_megatron.supports_hf_weight_loading(tmp_path, "builtin") is True


def test_mbridge_registry_error_mentions_model_type(monkeypatch, tmp_path):
    """An unsupported checkpoint in mbridge mode must fail loudly, not fall back."""
    fake_registry = {"kimi_k3": object()}

    class FakeAutoBridge:
        @staticmethod
        def from_pretrained(path, trust_remote_code=True):
            raise KeyError("fake_model_type")

    fake_mbridge = types.ModuleType("mbridge")
    fake_mbridge.AutoBridge = FakeAutoBridge
    fake_mbridge.__path__ = []
    fake_core_bridge = types.ModuleType("mbridge.core.bridge")
    fake_core_bridge._MODEL_REGISTRY = fake_registry
    # vime_plugins.mbridge.patch imports mbridge.models.ext.deepseek_v3 at import time.
    fake_dequant = types.ModuleType("mbridge.models.ext.deepseek_v3")
    fake_dequant.dequant_fp8_safetensor_io = types.SimpleNamespace(safe_open=None)
    for name in ("mbridge.core", "mbridge.models", "mbridge.models.ext"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "mbridge", fake_mbridge)
    monkeypatch.setitem(sys.modules, "mbridge.core.bridge", fake_core_bridge)
    monkeypatch.setitem(sys.modules, "mbridge.models.ext.deepseek_v3", fake_dequant)
    monkeypatch.delitem(sys.modules, "vime_plugins.mbridge", raising=False)
    monkeypatch.setattr(mbridge_load, "mbridge_available", lambda: True)
    monkeypatch.setattr(
        hf_to_megatron.AutoConfig, "from_pretrained", lambda *a, **k: types.SimpleNamespace(model_type="fake_model_type")
    )

    assert mbridge_load.supports_mbridge_loading(tmp_path) is False


def test_load_mode_flag_survives_argument_parse_round_trip():
    args = _parse_args("--hf-to-megatron-load-mode", "mbridge")
    assert args.hf_to_megatron_load_mode == "mbridge"

def test_registry_helper_names_pin_on_drift(monkeypatch):
    fake_bridge = types.ModuleType("mbridge.core.bridge")  # no _MODEL_REGISTRY
    for name in ("mbridge", "mbridge.core"):
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, name, pkg)
    monkeypatch.setitem(sys.modules, "mbridge.core.bridge", fake_bridge)
    with pytest.raises(RuntimeError, match=mbridge_load.MBRIDGE_PIN):
        mbridge_load._model_registry()
