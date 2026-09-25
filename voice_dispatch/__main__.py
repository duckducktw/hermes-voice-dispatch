"""讓 `python -m voice_dispatch` 可執行。"""

from .cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
