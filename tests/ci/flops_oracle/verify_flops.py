"""FLOPs oracle verifier (V5): compares a FLOPs-model implementation against the
vendored miles flops_utils.py oracle over the configs in configs.json.

Exit 0 = pass (or SKIP when the implementation module does not exist yet);
exit 1 = mismatch (relative error >= 1e-6 on any config x function).
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ORACLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = ORACLE_DIR.parents[2]
DEFAULT_ORACLE = ORACLE_DIR / "flops_utils_miles_ba28991834.py"
DEFAULT_IMPL = REPO_ROOT / "vime" / "utils" / "flops_model.py"
CONFIGS = ORACLE_DIR / "configs.json"
REL_TOL = 1e-6
SEQ_LEN_SETS = ([128, 256], [1024], [1, 7, 4096])


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # register before exec: dataclasses resolve string annotations via sys.modules[__module__]
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", type=Path, default=DEFAULT_IMPL)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    args = parser.parse_args()

    if not args.impl.exists():
        print(f"SKIP impl missing: {args.impl}")
        return 0
    oracle = load_module(args.oracle, "flops_oracle")
    impl = load_module(args.impl, "flops_impl")

    configs = json.loads(CONFIGS.read_text())
    failures = []
    checks = 0
    for entry in configs:
        name, config = entry["name"], entry["config"]
        oracle_model = oracle.model_flops_from_hf_config(dict(config))
        impl_model = impl.model_flops_from_hf_config(dict(config))
        for seq_lens in SEQ_LEN_SETS:
            for fn, kwargs in (
                ("fwd_flops", {}),
                ("train_flops", {}),
                ("train_flops", {"trains_indexers": True}),
            ):
                expected = getattr(oracle_model, fn)(seq_lens, **kwargs)
                actual = getattr(impl_model, fn)(seq_lens, **kwargs)
                checks += 1
                rel = abs(actual - expected) / max(abs(expected), 1.0)
                if rel >= REL_TOL:
                    failures.append(
                        f"{name} {fn}{kwargs} seq_lens={seq_lens}: impl={actual} oracle={expected} rel={rel:.3e}"
                    )
    for failure in failures:
        print(f"FAIL {failure}")
    print(f"{'FAIL' if failures else 'PASS'} {checks} checks, {len(failures)} failures over {len(configs)} configs")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
