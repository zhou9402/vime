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
