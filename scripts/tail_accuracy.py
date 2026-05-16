"""Direct tail-accuracy analysis (no LLM judge).

Compares a self-hosted service (default: `nemotron_local`) and a control
service (default: cloud `nemotron`) against the ground-truth references in
results.db, focusing on whether the END of utterances is consistently lost
(a finalization-path signature). The control service hits the same audio and
the same GT, so a tail-loss gap between the two isolates the serving/
finalization path from model/GT artifacts.

Usage:
    uv run python scripts/tail_accuracy.py [RESULTS_DB] [SUBJECT] [CONTROL]

Defaults: RESULTS_DB=stt_benchmark_data/results.db
          SUBJECT=nemotron_local  CONTROL=nemotron
"""

import re
import sqlite3
import sys
from collections import Counter
from difflib import SequenceMatcher

DB = sys.argv[1] if len(sys.argv) > 1 else "stt_benchmark_data/results.db"
SUBJECT = sys.argv[2] if len(sys.argv) > 2 else "nemotron_local"
CONTROL = sys.argv[3] if len(sys.argv) > 3 else "nemotron"

_punct = re.compile(r"[^\w\s]")
_ws = re.compile(r"\s+")


def norm(t: str) -> list[str]:
    if not t:
        return []
    t = _punct.sub(" ", t.lower())
    return _ws.sub(" ", t).strip().split()


def latest_by_sample(con, service: str) -> dict[str, str]:
    # Most recent row per sample for this service (max rowid = latest insert).
    q = """
    select r.sample_id, r.transcription
    from results r
    join (select sample_id, max(rowid) mx from results
          where service_name=? group by sample_id) m
      on r.sample_id=m.sample_id and r.rowid=m.mx
    where r.service_name=?
    """
    return {sid: (tx or "") for sid, tx in con.execute(q, (service, service))}


def analyze(ref: list[str], hyp: list[str]) -> dict:
    """Word-level alignment metrics, tail-focused."""
    sm = SequenceMatcher(None, ref, hyp, autojunk=False)
    S = D = I = 0
    matched_ref = set()
    del_positions = []  # relative position in ref of each unmatched ref word
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                matched_ref.add(i)
        elif tag == "replace":
            S += max(i2 - i1, j2 - j1)
            for i in range(i1, i2):
                del_positions.append(i)
        elif tag == "delete":
            D += i2 - i1
            for i in range(i1, i2):
                del_positions.append(i)
        elif tag == "insert":
            I += j2 - j1
    n = len(ref) or 1
    wer = (S + D + I) / n
    # trailing ref words missing: longest suffix of ref not in matched_ref
    tail_missing = 0
    for i in range(len(ref) - 1, -1, -1):
        if i in matched_ref:
            break
        tail_missing += 1
    strict_prefix = len(hyp) < len(ref) and ref[: len(hyp)] == hyp and len(hyp) > 0
    last_k = {k: (len(ref) >= k and ref[-k:] == hyp[-k:]) for k in (1, 2, 3)}
    return dict(
        wer=wer, tail_missing=tail_missing, strict_prefix=strict_prefix,
        last_k=last_k, del_positions=[p / n for p in del_positions],
        n_ref=len(ref), n_hyp=len(hyp),
    )


def summarize(rows: list[dict], label: str):
    k = len(rows)
    if not k:
        print(f"  {label}: no data")
        return

    def mean(xs):
        return sum(xs) / len(xs)

    wer = mean([r["wer"] for r in rows])
    any_tail = sum(r["tail_missing"] > 0 for r in rows) / k
    mean_tail = mean([r["tail_missing"] for r in rows])
    pref = sum(r["strict_prefix"] for r in rows) / k
    rec = {kk: sum(r["last_k"][kk] for r in rows) / k for kk in (1, 2, 3)}
    deciles = Counter()
    total_del = 0
    for r in rows:
        for p in r["del_positions"]:
            deciles[min(int(p * 10), 9)] += 1
            total_del += 1
    hist = " ".join(
        f"{d*10}-{d*10+10}%:{(deciles[d]/total_del*100 if total_del else 0):4.1f}"
        for d in range(10)
    )
    print(f"  {label}  (n={k})")
    print(f"    plain WER (edit)         : {wer*100:5.1f}%")
    print(f"    samples missing >=1 tail : {any_tail*100:5.1f}%")
    print(f"    mean trailing words lost : {mean_tail:5.2f}")
    print(f"    strict prefix-truncation : {pref*100:5.1f}%  (hyp == ref[:n], tail cut)")
    print(f"    final-word recall        : {rec[1]*100:5.1f}%   last-2: {rec[2]*100:4.1f}%   last-3: {rec[3]*100:4.1f}%")
    print(f"    deletion position (share by ref decile): {hist}")


def main():
    con = sqlite3.connect(DB)
    gt = {sid: tx for sid, tx in con.execute("select sample_id, text from ground_truth")}
    subj = latest_by_sample(con, SUBJECT)
    ctrl = latest_by_sample(con, CONTROL)
    print(f"db={DB}")
    print(f"ground_truth={len(gt)}  {SUBJECT}={len(subj)}  {CONTROL}={len(ctrl)}")

    common = [s for s in gt if s in subj and s in ctrl]
    print(f"samples with GT + {SUBJECT} + {CONTROL}: {len(common)}\n")
    if not common:
        return

    L = [analyze(norm(gt[s]), norm(subj[s])) for s in common]
    C = [analyze(norm(gt[s]), norm(ctrl[s])) for s in common]

    print("=== vs ground truth ===")
    summarize(L, f"{SUBJECT} (subject)")
    summarize(C, f"{CONTROL} (control)")

    lc = [analyze(norm(ctrl[s]), norm(subj[s])) for s in common]  # ref=control
    worse = sum(a["tail_missing"] > b["tail_missing"] for a, b in zip(L, C))
    better = sum(a["tail_missing"] < b["tail_missing"] for a, b in zip(L, C))
    same = len(common) - worse - better
    print(f"\n=== {SUBJECT} vs {CONTROL} (head-to-head, same audio) ===")
    print(f"  trailing-words-lost vs GT:  subject worse:{worse}  equal:{same}  subject better:{better}")
    print(f"  subject is a strict prefix of control (cut tail control kept): "
          f"{sum(x['strict_prefix'] for x in lc)}/{len(common)}")
    print(f"  mean trailing control-words missing from subject: "
          f"{sum(x['tail_missing'] for x in lc)/len(common):.2f}")

    print(f"\n=== worst {SUBJECT} tail-loss examples (ref tail vs subject tail) ===")
    for sid, m in sorted(zip(common, L), key=lambda z: -z[1]["tail_missing"])[:6]:
        if m["tail_missing"] == 0:
            break
        r, h = norm(gt[sid]), norm(subj[sid])
        print(f"  -{m['tail_missing']:>2} | GT ...{' '.join(r[-8:])!r}")
        print(f"      | HY ...{' '.join(h[-8:])!r}")


if __name__ == "__main__":
    main()
