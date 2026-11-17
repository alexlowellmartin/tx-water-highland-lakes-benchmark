"""`hlw` command: thin dispatcher over the pipeline modules."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "aoi": "hlw.aoi", "levels": "hlw.levels", "scenes": "hlw.scenes",
    "select": "hlw.selection", "download": "hlw.download", "train": "hlw.train",
    "teacher": "hlw.teacher",
    "label": "hlw.handlabel", "evaluate": "hlw.evaluate", "timeline": "hlw.timeline",
    "analyze": "hlw.analyze", "web": "hlw.webexport", "infer": "hlw.infer",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("usage: hlw {" + ",".join(COMMANDS) + "} [args]")
        raise SystemExit(2)
    cmd = sys.argv.pop(1)
    sys.argv[0] = f"hlw {cmd}"
    importlib.import_module(COMMANDS[cmd]).main()


if __name__ == "__main__":
    main()
