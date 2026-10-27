"""Agent-related CLI arguments (--custom-agent-function-path and friends)."""

from vime.agent.agent_function import AGENT_FUNCTION_MODES


def add_agent_arguments(parser):
    parser.add_argument(
        "--custom-agent-function-path",
        type=str,
        default=None,
        help=(
            "Import path (module.fn) to a custom agent function driving one sample's rollout. "
            "Unlike --custom-generate-function-path (which always runs in the rollout process), "
            "this is invoked through call_agent_function under --custom-agent-function-mode."
        ),
    )
    parser.add_argument(
        "--custom-agent-function-mode",
        type=str,
        choices=list(AGENT_FUNCTION_MODES),
        default="inline",
        help=(
            "How --custom-agent-function-path is invoked: 'subproc' runs every call in a fresh "
            "child process so a blocking step, crash, sys.exit or leaked thread cannot stall or "
            "kill the rollout loop or sibling episodes; 'inline' awaits it on the rollout event loop. "
            "Default 'inline' preserves in-process behavior; --custom-generate-function-path is "
            "unaffected by this flag."
        ),
    )
    return parser
