"""Run eval scenarios twice (memory ON vs OFF), score in Langfuse, print comparison table.

Usage: uv run python -m eval.run_eval
"""

import statistics
import time
import uuid

import yaml
from langfuse import Langfuse
from langfuse.api.resources.dataset_run_items.types import CreateDatasetRunItemRequest

from memhero.service import ChatService, store

DATASET = "memhero-eval"
COLLEAGUES = [
    ("Aarav", "chai"), ("Diya", "green tea"), ("Rohan", "espresso"), ("Isha", "latte"),
    ("Kabir", "lemonade"), ("Ananya", "filter coffee"), ("Vivaan", "masala chai"),
    ("Myra", "hot chocolate"), ("Arjun", "americano"), ("Saanvi", "iced tea"),
    ("Reyansh", "cappuccino"), ("Aadhya", "matcha"), ("Krish", "mocha"),
    ("Pari", "buttermilk"), ("Advik", "black coffee"), ("Zara", "coconut water"),
    ("Yash", "cold brew"), ("Navya", "rose tea"), ("Dev", "protein shake"),
    ("Kiara", "smoothie"), ("Aryan", "drip coffee"), ("Riya", "jasmine tea"),
    ("Om", "soda lime"), ("Anvi", "almond milk latte"), ("Vihaan", "nitro cold brew"),
    ("Tara", "oolong"), ("Sam", "flat white"), ("Meera", "cold brew"),  # Meera probed
    ("Raj", "espresso tonic"), ("Nisha", "cardamom chai"),
]
PROBED_COLLEAGUE = "Meera"
PROBED_DRINK = "cold brew"


def _build_scenarios() -> list[dict]:
    raw = yaml.safe_load(open("eval/scenarios.yaml"))["scenarios"]
    by_name = {s["name"]: s for s in raw}
    convos = [[] for _ in range(15)]
    for i, (name, drink) in enumerate(COLLEAGUES):
        convos[i % 15].append(f"FYI: my colleague {name} always drinks {drink} in meetings.")
    by_name["growth-stress"]["seeds"] = convos
    return list(by_name.values())


def ensure_dataset(lf: Langfuse, scenarios: list[dict]) -> list:
    try:
        lf.get_dataset(DATASET)
    except Exception:
        lf.create_dataset(name=DATASET, description="memhero conversational-memory eval")
    for s in scenarios:
        lf.create_dataset_item(
            dataset_name=DATASET,
            id=f"memhero-eval-{s['name']}",
            input={"probe": s["probe"]},
            expected_output={
                "expect_contains": s.get("expect_contains", []),
                "expect_not_contains": s.get("expect_not_contains", []),
            },
            metadata=s,
        )
    lf.flush()
    return lf.get_dataset(DATASET).items


def _llm_judge(answer: str, rubric: str) -> bool:
    from memhero import llm
    out = llm.aux_chat(
        [
            {"role": "system", "content":
                "You grade an assistant's answer against a rubric. Reply with exactly "
                "'PASS' or 'FAIL' and nothing else."},
            {"role": "user", "content": f"ANSWER:\n{answer}\n\nRUBRIC:\n{rubric}"},
        ],
        temperature=0,
    )
    return out.strip().upper().startswith("PASS")


def run_scenario(mode: str, meta: dict) -> dict:
    use_memory = mode == "memory-on"
    uid = f"eval-{meta['name']}-{mode}-{uuid.uuid4().hex[:6]}"
    svc = ChatService()
    store().clear_user(uid)
    latencies = []
    for ci, convo in enumerate(meta["seeds"]):
        for ti, line in enumerate(convo):
            t0 = time.perf_counter()
            out = svc.turn(uid, line, use_memory=use_memory, background=False)
            latencies.append(out["inline_ms"])
            print(f"  [{mode}] {meta['name']} seed {ci}.{ti} {out['inline_ms']:.0f}ms", flush=True)

    probe_out = svc.turn(uid, meta["probe"], use_memory=use_memory, background=False)
    answer = probe_out["reply"]
    latencies.append(probe_out["inline_ms"])

    forbidden_hit = any(
        frag.lower() in " ".join(m.content.lower() for m in store().list_memories(uid))
        for frag in meta.get("forbidden_in_memory", [])
    )
    return {
        "reply": answer,
        "recalled_count": len(probe_out["retrieved"]),
        "forbidden_violation": forbidden_hit,
        "p50_inline_ms": round(statistics.median(latencies), 1),
        "user_id": uid,
    }


def judge(meta: dict, exp: dict, out: dict) -> tuple[bool, str]:
    answer = out["reply"]
    ok = all(c.lower() in answer.lower() for c in exp.get("expect_contains", []))
    if not ok:
        reason = f"missing one of {exp.get('expect_contains')}"
    elif exp.get("expect_not_contains") or meta.get("judge_rubric"):
        ok = _llm_judge(answer, meta["judge_rubric"])
        reason = "rubric judge"
    else:
        reason = "substring"
    if out["forbidden_violation"]:
        ok, reason = False, "secret stored in DB"
    return ok, reason


def run(lf: Langfuse, mode: str, items: list) -> dict:
    run_name = f"{mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    passed = 0
    p50s, answers = [], []
    for item in items:
        meta = item.metadata
        exp = item.expected_output or {}
        print(f"[{mode}] {meta['name']} ...", flush=True)
        out = run_scenario(mode, meta)
        ok, reason = judge(meta, exp, out)
        print(f"[{mode}] {meta['name']}: {'PASS' if ok else 'FAIL'} ({reason})", flush=True)

        tid = str(uuid.uuid4())
        lf.trace(
            id=tid, name=f"eval-{meta['name']}", metadata={"mode": mode, "reason": reason},
            input={"probe": meta["probe"]}, output=out,
            user_id=out["user_id"],
        )
        lf.score(trace_id=tid, name="correct", value=1.0 if ok else 0.0,
                 comment=f"[{reason}] {out['reply'][:300]}")
        lf.api.dataset_run_items.create(request=CreateDatasetRunItemRequest(
            run_name=run_name, run_description=f"memhero eval {mode}",
            dataset_item_id=item.id, trace_id=tid,
        ))
        passed += int(ok)
        p50s.append(out["p50_inline_ms"])
        answers.append((meta["name"], ok))

    lf.flush()
    return {
        "mode": mode, "passed": passed, "total": len(items), "run_name": run_name,
        "p50_ms": round(statistics.median(p50s), 1) if p50s else None,
        "per_scenario": answers,
    }


def main():
    lf = Langfuse()
    items = ensure_dataset(lf, _build_scenarios())
    rows = [run(lf, m, items) for m in ("memory-on", "memory-off")]

    print("\n=== memhero eval ===")
    for r in rows:
        print(f"\n{r['mode']}: {r['passed']}/{r['total']} passed | median turn {r['p50_ms']} ms | run: {r['run_name']}")
        for name, ok in r["per_scenario"]:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\nView in Langfuse -> Datasets -> {DATASET} -> Runs")
    lf.flush()


if __name__ == "__main__":
    main()
