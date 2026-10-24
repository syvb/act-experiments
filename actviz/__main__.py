"""Export a trace for the viewer.

    python -m actviz                                  # GPT-2 small on the default prompt
    python -m actviz --model gpt2 --prompt "..." --id my-trace
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from .models import MODELS
from .writer import write_trace

DEFAULT_PROMPT = "When Mary and John went to the store, John gave a drink to"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "viewer" / "data"


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m actviz", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2", choices=sorted(MODELS), help="model to run (default: gpt2)")
    ap.add_argument("--repo", help="Hugging Face repo to load weights from (default: the model's usual repo)")
    ap.add_argument("--revision", help="repo revision (branch, tag or commit)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="input text")
    ap.add_argument("--no-bos", action="store_true", help="don't prepend the beginning-of-text token")
    ap.add_argument("--id", help="trace id, used as the output folder name (default: derived from model and prompt)")
    ap.add_argument("--title", help="title shown in the viewer's trace picker")
    ap.add_argument("--query-pos", type=int, help="token selected when the page opens (default: the last)")
    ap.add_argument("--key-pos", type=int, help="key token selected when the page opens (default: same as the query)")
    ap.add_argument("--head", type=int, default=0, help="attention head selected when the page opens")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"data directory (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    cls, default_repo = MODELS[args.model]
    t0 = time.time()
    model = cls.load(args.repo or default_repo, args.revision)
    ids = model.tokenize(args.prompt, bos=not args.no_bos)
    trace = model.trace(ids, args.prompt)
    q = args.query_pos if args.query_pos is not None else len(ids) - 1
    trace.extra["defaults"] = {"pos": q, "kpos": args.key_pos if args.key_pos is not None else q, "head": args.head}
    slug = re.sub(r"[^a-z0-9]+", "-", args.prompt.lower()).strip("-")[:40]
    trace_id = args.id or f"{args.model}-{slug}"
    title = args.title or f"{trace.model['display_name']}: {args.prompt}"
    out = write_trace(trace, args.out, trace_id, title)
    size = sum(f.stat().st_size for f in out.glob("*.js"))
    print(f"wrote {len(trace.tensors)} tensors for {len(ids)} tokens to {out} ({size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
