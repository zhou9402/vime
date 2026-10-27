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
    parser.add_argument(
        "--check-weight-transfer-checksum",
        action="store_true",
        default=False,
        help=(
            "Compute per-tensor SHA-256 checksums on the sender and have each engine verify the staged "
            "bytes before loading them (p2p transport). Costs a CPU pass over every bucket."
        ),
    )
    return parser
