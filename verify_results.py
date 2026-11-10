#!/usr/bin/env python
"""Check every number the paper claims against the artifact that produced it.

Three things this verifies, in order of what they cost to fix:

1. **The ledger is honest.** Every claim in ``expected/claims.json`` is recomputed
   from the artifact it names and compared with the value printed in the
   manuscript. A mismatch means either the manuscript or the artifact drifted.

2. **The folds are the published ones.** The subject assignment recorded inside
   the expected run artifacts must equal ``folds/cis_folds.json``.

3. **A fresh run reproduces the numbers** (only with ``--runs DIR``). Artifacts
   from a new run are matched to the expected ones by filename and the same
   claims are recomputed, with a stated tolerance.

Usage
-----
    python verify_results.py                     # ledger vs expected/
    python verify_results.py --paths             # what the path layer resolves
    python verify_results.py --runs runs/paper   # plus a fresh run
    python verify_results.py --tol 0.3 --quiet

Exit status is 0 when everything passes, 1 otherwise, so it works as a CI gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EXPECTED = Path(__file__).resolve().parent / "expected"
DEFAULT_TOL = 0.1


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def _get(obj, dotted):
    """Walk a dotted path: ``a.b.0`` -> obj['a']['b'][0]."""
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            if part not in cur:
                raise KeyError(f"{part!r} not in {sorted(cur)[:8]}")
            cur = cur[part]
        else:
            raise TypeError(f"cannot descend into {type(cur).__name__} at {part!r}")
    return cur


def extract(payload, spec):
    """Return (value, std_or_None) from a parsed artifact per an extract spec."""
    kind = spec["kind"]
    scale = spec.get("scale", 1.0)

    if kind == "run_mean_std":
        if isinstance(payload, list) and payload and isinstance(payload[0], list):
            rows = [r for part in payload for r in part]   # merged artifacts
        else:
            rows = payload["runs"] if isinstance(payload, dict) else payload
        vals = [float(r[spec["field"]]) for r in rows]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return mean * scale, var ** 0.5 * scale

    if kind == "summary_pair":
        obj = _get(payload, spec["prefix"]) if spec.get("prefix") else payload
        return float(obj[spec["mean"]]) * scale, float(obj[spec["std"]]) * scale

    if kind == "dotted":
        return float(_get(payload, spec["path"])) * scale, None

    if kind == "nested_pair":
        obj = _get(payload, spec["path"])
        return float(obj["mean"]) * scale, float(obj["std"]) * scale

    if kind == "table_row":
        rows = payload["table"]
        for row in rows:
            if all(row.get(k) == v for k, v in spec["match"].items()):
                return float(row[spec["field"]]) * scale, None
        raise KeyError(f"no table row matching {spec['match']}")

    if kind == "named_entry":
        for row in payload:
            if row.get("name") == spec["name"]:
                return float(row[spec["field"]]) * scale, None
        raise KeyError(f"no entry named {spec['name']!r}")

    if kind == "curve":
        point = _get(payload, f"curve.{spec['key']}")
        if isinstance(point, dict):
            return float(point["mean"]) * scale, float(point.get("std", 0.0)) * scale
        return float(point[spec["which"]]) * scale, float(point[1]) * scale

    raise ValueError(f"unknown extract kind {kind!r}")


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def resolve(artifact, runs_dir):
    """Find the artifact for this claim inside ``runs_dir`` (by filename)."""
    if runs_dir is None:
        return EXPECTED / artifact
    name = Path(artifact).name
    candidates = [p for p in Path(runs_dir).rglob(name) if p.is_file()]
    if not candidates:
        return None
    # prefer the shallowest hit: a per-experiment directory over a nested copy
    return min(candidates, key=lambda p: len(p.parts))


def check_claims(ledger, runs_dir, tol, quiet):
    """Compare every claim with its artifact.

    In ``--runs`` mode a claim whose artifact is not present in that directory is
    reported as *skipped*, not failed: reproducing the whole paper at once is not
    the normal workflow, and a partial re-run should still give a usable verdict.
    """
    rows, failures, skipped = [], 0, 0
    cache: dict[Path, dict] = {}
    for claim in ledger["claims"]:
        paper = claim["paper"]
        wanted = claim.get("artifacts") or [claim["artifact"]]
        paths = [resolve(a, runs_dir) for a in wanted]
        absent = [p is None or not p.is_file() for p in paths]
        if any(absent):
            missing = [a for a, a_ in zip(wanted, absent) if a_]
            if runs_dir is not None:
                rows.append((claim["id"], claim["where"], "SKIP", "", "", "not in this run"))
                skipped += 1
                continue
            rows.append((claim["id"], claim["where"], "MISSING", "", "", f"absent: {missing[0]}"))
            failures += 1
            continue
        payloads = []
        for p in paths:
            if p not in cache:
                cache[p] = load_json(p)
            payloads.append(cache[p])
        payload = payloads[0] if len(payloads) == 1 else payloads
        try:
            value, std = extract(payload, claim["extract"])
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            rows.append((claim["id"], claim["where"], "ERROR", "", f"{type(exc).__name__}", str(exc)[:44]))
            failures += 1
            continue

        want = float(paper[0])
        got_std = paper[1] if len(paper) > 1 else None
        d_value = abs(value - want)
        ok = d_value <= tol
        if ok and got_std is not None and std is not None:
            ok = abs(std - float(got_std)) <= tol

        shown = f"{value:.2f}" + (f"±{std:.2f}" if std is not None else "")
        target = f"{want:.2f}" + (f"±{float(got_std):.2f}" if got_std is not None else "")
        rows.append((claim["id"], claim["where"], "ok" if ok else "MISMATCH",
                     shown, target, "" if ok else f"Δ={d_value:.2f}"))
        failures += 0 if ok else 1
    return rows, failures, skipped


def check_folds(folds_path, expected_folds, quiet):
    """The subject assignment inside the run artifacts must match the frozen file."""
    problem = None
    frozen = load_json(folds_path)
    by_fold = {i: set(f["test"]) for i, f in enumerate(frozen["folds"])}
    for rel in expected_folds:
        p = EXPECTED / rel
        if not p.is_file():
            problem = f"{rel} missing"
            break
        payload = load_json(p)
        rows = payload["runs"] if isinstance(payload, dict) else payload
        if not rows or "test_subj" not in rows[0]:
            continue
        for row in rows:
            want = by_fold.get(int(row["fold"]))
            if want and set(row["test_subj"]) != want:
                problem = (f"{rel} fold {row['fold']}: artifact tests "
                           f"{sorted(row['test_subj'])} but cis_folds.json says {sorted(want)}")
                break
        if problem:
            break
    return problem


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None,
                    help="artifacts from a fresh run; enables the reproduction check")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help=f"tolerance in points (default {DEFAULT_TOL})")
    ap.add_argument("--paths", action="store_true", help="print resolved paths and exit")
    ap.add_argument("--quiet", action="store_true", help="only show failures")
    args = ap.parse_args()

    if args.paths:
        from src.paths import describe
        print(describe())
        return 0

    ledger = load_json(EXPECTED / "claims.json")
    mode = "REPRODUCE" if args.runs else "LEDGER"
    print(f"verify_results — mode: {mode}, tolerance ±{args.tol} points")
    print(f"  source commit {ledger['provenance']['source_repo_commit'][:12]}, "
          f"{len(ledger['claims'])} claims\n")

    rows, failures, skipped = check_claims(ledger, args.runs, args.tol, args.quiet)

    w_id = max(len(r[0]) for r in rows) + 1
    w_where = max(len(r[1]) for r in rows) + 1
    for cid, where, status, got, want, note in rows:
        if args.quiet and status in ("ok", "SKIP"):
            continue
        mark = {"ok": "  ok  ", "MISMATCH": " FAIL ", "MISSING": " MISS ",
                "SKIP": " skip ", "ERROR": " ERR  "}[status]
        line = f"{mark} {cid:<{w_id}} {where:<{w_where}} got {got:>10s}  paper {want:>10s}"
        if note:
            line += f"  {note}"
        print(line)

    folds = check_folds(EXPECTED.parent / "folds" / "cis_folds.json",
                        ["cis/exp1/ema_only_results.json", "cis/exp1/full_results.json"],
                        args.quiet)
    print()
    if folds:
        print(f"FAIL  folds: {folds}")
        failures += 1
    else:
        print("  ok   folds: subject assignment matches folds/cis_folds.json")

    total = len(rows) - skipped
    print(f"\n{total - failures}/{total} checks passed"
          + (f", {skipped} skipped (no artifact in this run)" if skipped else "")
          + ("" if failures == 0 else f" — {failures} failure(s)"))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
