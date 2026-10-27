"""CLI flags for P2P/RDMA weight transfer (F01). Registered from arguments.py."""


def add_p2p_weight_transfer_arguments(parser):
    """Flags for --update-weight-transport=p2p (disaggregated Mooncake RDMA weight sync)."""
    parser.add_argument(
        "--p2p-transfer-timeout",
        type=float,
        default=30.0,
        help=(
            "Timeout in seconds for one Mooncake RDMA write to finish before the engine is marked "
            "errored. Only used with --update-weight-transport=p2p."
        ),
    )
    # NOTE: --check-weight-transfer-checksum is owned by F02's
    # add_weight_check_arguments (arguments.py); p2p consumes args.check_weight_transfer_checksum.
    parser.add_argument(
        "--p2p-on-engine-error",
        choices=["fail", "continue"],
        default="fail",
        help=(
            "'fail' (default): a weight-update error on any engine fails the update. "
            "'continue': quarantine + retire the errored engine and keep training on the "
            "healthy ones (disaggregated multi-engine deployments only; the engine stays "
            "quarantined until the health monitor retires it and recover() replaces + "
            "reconnects it). Only used with --update-weight-transport=p2p."
        ),
    )
    parser.add_argument(
        "--p2p-min-healthy-engines",
        type=int,
        default=1,
        help=(
            "With --p2p-on-engine-error=continue, the update still fails if fewer engines "
            "than this remain healthy after an error."
        ),
    )
    return parser


def validate_p2p_spec_decode_args(args) -> None:
    """Speculative-decoding validation for --update-weight-transport=p2p.

    Mirrors the distributed path's draft-sync condition
    (update_weight_from_distributed.py): mtp speculative decoding is supported
    (the target buckets are re-sent in a draft session, so trained MTP layers
    reach the draft runner); dspark is not — its draft weights need
    `export_dspark_model_weights`, which P2P does not iterate.
    """
    if getattr(args, "update_weight_transport", None) != "p2p":
        return
    method = (args.vllm_speculative_config or {}).get("method")
    if method == "dspark":
        raise ValueError(
            "P2P weight transfer does not support dspark speculative decoding: P2P "
            "sends the target HF weight iterator only, while dspark draft weights "
            "need export_dspark_model_weights. Use --update-weight-transport=nccl "
            "for dspark, or mtp speculative decoding with p2p."
        )


def validate_p2p_degrade_args(args) -> None:
    """Validation for --p2p-on-engine-error (F01-P6b). The caller is inside the p2p block.

    continue degrades the rollout fleet, which is only meaningful with more
    engines than the healthy floor (a single engine can never degrade).
    """
    from vime.utils.http_utils import get_rollout_num_engines

    num_engines = get_rollout_num_engines(args)
    if not 1 <= args.p2p_min_healthy_engines <= num_engines:
        raise ValueError(
            f"--p2p-min-healthy-engines must be within [1, {num_engines}] "
            f"(the rollout fleet size), got {args.p2p_min_healthy_engines}."
        )
    if args.p2p_on_engine_error == "continue" and num_engines <= 1:
        raise ValueError(
            "--p2p-on-engine-error=continue is meaningless with a single rollout engine "
            "(no healthy engine would remain after an error); use the default 'fail'."
        )
