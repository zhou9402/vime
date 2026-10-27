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
