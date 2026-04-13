# Full GRPO training loop with Qwen3-8B (CLIENT-08)
# TODO: implement in CLIENT-08

from __future__ import annotations

import os
import sys


def _require_api_key() -> str:
    api_key = os.getenv("MC_MCP_API_KEY", "").strip()
    if not api_key:
        print("Set MC_MCP_API_KEY environment variable", file=sys.stderr)
        raise SystemExit(1)
    return api_key


def main() -> None:
    _require_api_key()
    # TODO: implement in CLIENT-08
    pass


if __name__ == "__main__":
    main()
