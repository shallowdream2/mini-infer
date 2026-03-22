"""Compatibility entrypoint for the advanced chat CLI."""

from __future__ import annotations

import sys

from mini_infer.clients.chat_client import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
