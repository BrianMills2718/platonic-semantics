#!/usr/bin/env python3
"""Arm 3: have a model write the political corpus, so the same instrument can read it.

Arm 2 measures a semantic space from what people wrote. This writes the matched
corpus from a model, and `text_space.py` reads both the same way. Because the
instrument is identical, any difference in the resulting space is attributable to
the writer rather than to the elicitation -- which is exactly what arm 1's
magnitude estimation could not achieve.

Two design constraints exist to keep the comparison meaningful:

* **Topics are taken from the human corpus, wording is not.** The instrument
  needs a shared vocabulary across the corpora being compared. If the model wrote
  about different subjects, that vocabulary would be empty and any "difference"
  would only report that the two corpora concerned different news.
* **Stance and voice are sampled, not left free.** Asked repeatedly for the same
  thing, a model collapses onto one register. That would show up as unusually
  tight semantic structure and be mistaken for a finding about model
  representation, when it is an artefact of how narrowly the model was sampled.

Writes JSONL incrementally and resumes, because a run is a few hundred calls and
losing it to one transport error would be expensive.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/brian/code/llm_client")

from llm_client import call_llm_json_schema  # noqa: E402
from llm_client.prompts import render_prompt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
PROMPT = ROOT / "prompts" / "social_posts.yaml"

# Taken from the Exorde Politics shard by document frequency, after stopword
# removal -- these ARE the concepts the instrument will end up measuring, so the
# model has to write about them for the shared vocabulary to exist at all.
TOPICS = [
    "Trump", "the election", "the government", "voting and turnout",
    "the state of the country", "war and foreign intervention", "Biden",
    "the president and presidential power", "state versus federal authority",
    "Israel and Gaza", "Ukraine", "the two parties", "Russia",
    "the Democrats", "America's place in the world", "who really holds power",
]

STANCES = [
    "mostly critical", "mostly supportive", "mixed -- some agree, some disagree",
    "cynical about all sides", "anxious and uncertain", "angry",
]

VOICES = [
    "blunt and short", "long-winded and argumentative", "sarcastic",
    "earnest and sincere", "posting a hot take", "replying to something unseen",
    "quoting a statistic or claim", "personal, tying it to their own life",
]


def schema_for(n: int) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["posts"],
        "properties": {
            "posts": {
                "type": "array",
                # Deliberately loose. A tight maxItems made the provider-accepted
                # response fail client-side validation and burn two paid retries
                # per call; the count is trimmed below instead.
                "minItems": 1,
                "maxItems": n * 3,
                "items": {"type": "string", "description": "One standalone post."},
            }
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-5.6-terra")
    ap.add_argument("--target", type=int, default=8000, help="posts to produce")
    ap.add_argument("--per-call", type=int, default=40)
    ap.add_argument("--budget", type=float, default=2.00)
    ap.add_argument("--out", default="results/llm_posts.jsonl")
    ap.add_argument("--workers", type=int, default=12,
                    help="concurrent calls; a few hundred sequential calls takes hours")
    ap.add_argument("--calibrate", action="store_true",
                    help="one call only; report length and cost, then stop")
    args = ap.parse_args()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    have = 0
    seen_cells = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                have += 1
                seen_cells.add(r["cell"])
        print(f"resuming: {have:,} posts already on disk")

    cells = [f"{t}|{s}|{v}" for t, s, v in itertools.product(TOPICS, STANCES, VOICES)]
    rng = random.Random(20260904)
    rng.shuffle(cells)
    trace = f"galileo-arm3-{int(time.time())}"

    todo = [c for c in cells if c not in seen_cells]
    need_calls = max(1, -(-(args.target - have) // args.per_call))
    todo = todo[:need_calls] if not args.calibrate else todo[:1]
    if not todo:
        raise RuntimeError("no unused topic/stance/voice cells left; widen TOPICS to go further")

    lock = threading.Lock()
    fh = out.open("a", encoding="utf-8")
    state = {"have": have, "spent": 0.0, "calls": 0}

    def one(cell):
        topic, stance, voice = cell.split("|")
        messages = render_prompt(PROMPT, n=args.per_call, topic=topic,
                                 stance=stance, voice=voice)
        data, result = call_llm_json_schema(
            args.model, messages, schema_for(args.per_call),
            schema_name="social_posts",
            task="galileo-arm3-post-generation",
            trace_id=trace,
            max_budget=args.budget,
            reasoning_effort="none",
            model_justification=(
                "Arm 3 needs a model to write ~18k posts. The OpenRouter balance is "
                "exhausted, and arm 3's claim is about how a language model organises "
                "concepts, not about one specific vendor, so any capable writer serves."),
            max_tokens=args.per_call * 130,   # short posts; the default reserves 64k and OpenRouter refuses
        )
        posts = [p.strip() for p in data["posts"] if p and p.strip()][:args.per_call]
        with lock:
            for p in posts:
                fh.write(json.dumps({"text": p, "cell": cell}) + "\n")
            fh.flush()
            state["have"] += len(posts)
            state["calls"] += 1
            state["spent"] += getattr(result, "cost", None) or 0.0
            if state["calls"] % 20 == 0:
                print(f"  {state['have']:,}/{args.target:,} posts   "
                      f"{state['calls']}/{len(todo)} calls   ${state['spent']:.3f}", flush=True)
        return posts

    failures = []
    with ThreadPoolExecutor(max_workers=1 if args.calibrate else args.workers) as ex:
        futs = {ex.submit(one, c): c for c in todo}
        for f in as_completed(futs):
            try:
                posts = f.result()
            except Exception as e:                    # one bad call must not lose the run
                failures.append((futs[f], repr(e)[:160]))
                continue
            if args.calibrate:
                words = [len(p.split()) for p in posts]
                print(f"\ncell: {futs[f]}")
                print(f"asked {args.per_call}, got {len(posts)}; "
                      f"mean {sum(words)/len(words):.1f} words")
                print(f"cost ${state['spent']:.5f}  -> {args.target:,} posts "
                      f"~${state['spent']*args.target/max(len(posts),1):.2f}")
                for p in posts[:4]:
                    print(f"  - {p}")

    have, spent, calls = state["have"], state["spent"], state["calls"]
    if failures:
        print(f"\n{len(failures)} of {len(todo)} calls failed:")
        for c, e in failures[:5]:
            print(f"  {c}: {e}")
        if len(failures) > len(todo) * 0.2:
            raise RuntimeError(f"{len(failures)}/{len(todo)} calls failed; not a transient error")
    fh.close()
    if not args.calibrate:
        print(f"\n{have:,} posts, {calls} calls this run, ${spent:.3f}")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
