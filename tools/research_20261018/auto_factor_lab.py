"""Offline factor-proposal gate; never trains or alters the 2026-10-17 model.

An optional caller can provide a *local* LLM callable that receives only allowed
column names. The CLI consumes a proposal JSON file; it never contacts a model.
Selection uses mature training and validation labels. The later test is read only
after selection and never changes an acceptance decision.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/research_20261011"))
from factor_dsl import evaluate  # noqa: E402  (existing audited arithmetic DSL)

PANEL = ROOT / "training_data/research_20261011/model_panel.parquet"
MAX_PROPOSALS = 20
MAX_EXPR_CHARS = 160
MAX_AST_NODES = 31
NAME = re.compile(r"^[a-z][a-z0-9_]{2,50}$")


@dataclass(frozen=True)
class Protocol:
    train_start: str
    valid_start: str
    test_start: str
    test_end: str  # exclusive; an answer must finish strictly before this date
    min_stocks_per_day: int = 10
    min_days: int = 15
    min_coverage: float = 0.70
    min_train_abs_ic: float = 0.02
    min_valid_signed_ic: float = 0.01
    max_parent_abs_corr: float = 0.995

    def dates(self) -> tuple[pd.Timestamp, ...]:
        values = tuple(pd.Timestamp(getattr(self, k)) for k in
                       ("train_start", "valid_start", "test_start", "test_end"))
        if not all(a < b for a, b in zip(values[:-1], values[1:])):
            raise ValueError("Require train_start < valid_start < test_start < test_end")
        return values


def allowed_inputs(columns: list[str] | pd.Index) -> list[str]:
    """Existing, precomputed signal-time features only; never labels/answers."""
    return sorted(c for c in columns if c.startswith("x_") and not c.startswith("x_llm_")
                  and c.isidentifier() and not any(t in c.lower() for t in
                                                   ("target", "label", "exit", "future")))


def build_local_prompt(columns: list[str], count: int = 5) -> str:
    """No stock rows, prices, returns, reports or private documents enter prompt."""
    if not 1 <= count <= MAX_PROPOSALS:
        raise ValueError("count out of bounds")
    names = allowed_inputs(columns)
    return ("只提出可检验的股票因子假说，不声称收益。只输出 JSON 数组，每项含 id、"
            "expression、hypothesis、proposed_by。最多 " + str(count) + " 项。"
            "表达式只能用列名、有限数字、+ - * / 和括号；不得用函数、索引、未来值。"
            "可用列名：" + ", ".join(names))


def proposal_from_local_llm(local_callable: Callable[[str], str],
                            columns: list[str], count: int = 5) -> list[dict]:
    """Dependency-injected local model. This project provides no network client."""
    return parse_proposals(json.loads(local_callable(build_local_prompt(columns, count))),
                           allowed_inputs(columns))


def parse_proposals(payload: object, inputs: list[str]) -> list[dict]:
    if not isinstance(payload, list) or not 1 <= len(payload) <= MAX_PROPOSALS:
        raise ValueError("Expected a nonempty JSON array of at most 20 proposals")
    seen_ids: set[str] = set()
    seen_formulas: set[str] = set()
    result: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each proposal must be an object")
        pid, expression, hypothesis = (item.get(k) for k in
                                       ("id", "expression", "hypothesis"))
        if not isinstance(pid, str) or not NAME.fullmatch(pid) or pid in seen_ids:
            raise ValueError("Invalid or duplicate proposal id")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError("A concrete hypothesis is required")
        parents, key = inspect_expression(expression, inputs)
        if key in seen_formulas:
            raise ValueError("Duplicate formula")
        seen_ids.add(pid)
        seen_formulas.add(key)
        result.append({"id": pid, "expression": expression,
                       "hypothesis": hypothesis.strip(),
                       "proposed_by": str(item.get("proposed_by", "unrecorded")),
                       "parents": parents, "ast_key": key})
    return result


def partition_proposals(payload: object, inputs: list[str]) -> tuple[list[dict], list[dict]]:
    """Keep a complete rejection ledger even if one proposed formula is invalid."""
    if not isinstance(payload, list) or not 1 <= len(payload) <= MAX_PROPOSALS:
        raise ValueError("Expected a nonempty JSON array of at most 20 proposals")
    good: list[dict] = []
    rejected: list[dict] = []
    seen_ids: set[str] = set()
    seen_formulas: set[str] = set()
    for position, item in enumerate(payload, 1):
        pid = item.get("id", f"invalid_{position}") if isinstance(item, dict) else f"invalid_{position}"
        try:
            parsed = parse_proposals([item], inputs)[0]
            if parsed["id"] in seen_ids or parsed["ast_key"] in seen_formulas:
                raise ValueError("Duplicate id or formula in this batch")
        except ValueError as exc:
            rejected.append({"id": str(pid),
                             "expression": str(item.get("expression", "")) if isinstance(item, dict) else "",
                             "hypothesis": str(item.get("hypothesis", "")) if isinstance(item, dict) else "",
                             "proposed_by": str(item.get("proposed_by", "unrecorded")) if isinstance(item, dict) else "unrecorded",
                             "parents": "", "coverage_train": np.nan,
                             "train_days": 0, "train_rank_ic": np.nan,
                             "orientation_from_train": np.nan,
                             "max_parent_abs_corr_train": np.nan,
                             "valid_days": 0, "valid_rank_ic_after_train_orientation": np.nan,
                             "accepted_before_independent_test": False,
                             "rejection_reasons": "syntax_or_contract:" + str(exc)})
            continue
        good.append(parsed)
        seen_ids.add(parsed["id"])
        seen_formulas.add(parsed["ast_key"])
    return good, rejected


def inspect_expression(expression: object, inputs: list[str]) -> tuple[list[str], str]:
    if not isinstance(expression, str) or not 1 <= len(expression) <= MAX_EXPR_CHARS:
        raise ValueError("Expression empty or too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid expression syntax") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise ValueError("Expression too complex")
    accepted = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load,
                ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub)
    if any(not isinstance(node, accepted) for node in nodes):
        raise ValueError("Expression contains disallowed operation")
    for node in nodes:
        if isinstance(node, ast.Constant) and (type(node.value) not in (int, float)
                or not np.isfinite(node.value) or abs(node.value) > 1_000):
            raise ValueError("Constant must be finite and at most 1000 in magnitude")
    parents = sorted({n.id for n in nodes if isinstance(n, ast.Name)})
    if not parents or any(n not in inputs for n in parents):
        raise ValueError("Only allowlisted x_* signal-time columns may be used")
    return parents, ast.dump(tree, include_attributes=False)


def _score(expression: str, frame: pd.DataFrame, parents: list[str]) -> pd.Series:
    value = evaluate(expression, frame[parents])
    if np.isscalar(value):
        result = pd.Series(value, index=frame.index, dtype=float)
    else:
        result = pd.Series(value, index=frame.index, dtype=float)
    return result.replace([np.inf, -np.inf], np.nan)


def daily_rank_ic(frame: pd.DataFrame, scores: pd.Series,
                  min_stocks: int) -> pd.DataFrame:
    x = frame[["date", "label_5"]].copy()
    x["score"] = scores
    rows = []
    for day, group in x.groupby("date", sort=True):
        group = group.dropna(subset=["score", "label_5"])
        if len(group) < min_stocks:
            continue
        a = group.score.rank().to_numpy(dtype=float)
        b = group.label_5.rank().to_numpy(dtype=float)
        if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
            continue
        rows.append({"date": day, "rank_ic": float(np.corrcoef(a, b)[0, 1]),
                     "stocks": len(group)})
    return pd.DataFrame(rows, columns=["date", "rank_ic", "stocks"])


def _mean_ic(daily: pd.DataFrame) -> float:
    return float(daily.rank_ic.mean()) if len(daily) else float("nan")


def _max_parent_correlation(frame: pd.DataFrame, scores: pd.Series,
                            parents: list[str]) -> float:
    maxima = []
    for parent in parents:
        pairs = pd.DataFrame({"a": scores, "b": frame[parent]}).dropna()
        if len(pairs) >= 3 and pairs.a.std() > 1e-12 and pairs.b.std() > 1e-12:
            maxima.append(abs(float(pairs.a.corr(pairs.b))))
    return max(maxima, default=float("nan"))


def split_mature(panel: pd.DataFrame, protocol: Protocol) -> tuple[pd.DataFrame, ...]:
    start, valid, test, end = protocol.dates()
    if not {"date", "stockid", "label_5", "label_exit_5"}.issubset(panel.columns):
        raise ValueError("Panel needs date, stockid, label_5 and label_exit_5")
    p = panel.copy()
    p["date"] = pd.to_datetime(p.date)
    p["label_exit_5"] = pd.to_datetime(p.label_exit_5)
    if p.duplicated(["date", "stockid"]).any():
        raise ValueError("Duplicate date-stockid rows")
    train = p.loc[(p.date >= start) & (p.date < valid) &
                  (p.label_exit_5 < valid) & p.label_5.notna()].copy()
    validation = p.loc[(p.date >= valid) & (p.date < test) &
                       (p.label_exit_5 < test) & p.label_5.notna()].copy()
    independent = p.loc[(p.date >= test) & (p.date < end) &
                        (p.label_exit_5 < end) & p.label_5.notna()].copy()
    return train, validation, independent


def run_lab(panel: pd.DataFrame, proposals: list[dict],
            protocol: Protocol, *, evaluate_test: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    inputs = allowed_inputs(panel.columns)
    clean, rejected = partition_proposals(proposals, inputs)
    # Feedback rounds are physically restricted to pre-test rows. They do not
    # calculate test IC, coverage or other diagnostics used to revise formulas.
    selection_panel = panel if evaluate_test else panel.loc[pd.to_datetime(panel.date) < pd.Timestamp(protocol.test_start)]
    train, valid, independent = split_mature(selection_panel, protocol)
    required = (train, valid, independent) if evaluate_test else (train, valid)
    if min(f.date.nunique() for f in required) < protocol.min_days:
        raise ValueError("At least min_days mature dates are required in each segment")
    screen_rows = list(rejected)
    test_rows = []
    for proposal in clean:
        parents = proposal["parents"]
        train_score = _score(proposal["expression"], train, parents)
        coverage = float(train_score.notna().mean())
        train_daily = daily_rank_ic(train, train_score, protocol.min_stocks_per_day)
        train_ic = _mean_ic(train_daily)
        sign = 1 if np.isfinite(train_ic) and train_ic >= 0 else -1
        parent_corr = _max_parent_correlation(train, train_score, parents)
        valid_score = _score(proposal["expression"], valid, parents)
        valid_daily = daily_rank_ic(valid, valid_score, protocol.min_stocks_per_day)
        valid_signed_ic = sign * _mean_ic(valid_daily)
        reasons = []
        if coverage < protocol.min_coverage:
            reasons.append("train_coverage_low")
        if len(train_daily) < protocol.min_days:
            reasons.append("train_too_few_valid_days")
        if not np.isfinite(train_ic) or abs(train_ic) < protocol.min_train_abs_ic:
            reasons.append("train_ic_weak")
        if np.isfinite(parent_corr) and parent_corr >= protocol.max_parent_abs_corr:
            reasons.append("duplicate_parent_feature")
        if len(valid_daily) < protocol.min_days:
            reasons.append("validation_too_few_valid_days")
        if not np.isfinite(valid_signed_ic) or valid_signed_ic < protocol.min_valid_signed_ic:
            reasons.append("validation_ic_weak_or_reversed")
        accepted = not reasons
        screen_rows.append({"id": proposal["id"], "expression": proposal["expression"],
                            "hypothesis": proposal["hypothesis"],
                            "proposed_by": proposal["proposed_by"],
                            "parents": ",".join(parents), "coverage_train": coverage,
                            "train_days": len(train_daily), "train_rank_ic": train_ic,
                            "orientation_from_train": sign,
                            "max_parent_abs_corr_train": parent_corr,
                            "valid_days": len(valid_daily),
                            "valid_rank_ic_after_train_orientation": valid_signed_ic,
                            "accepted_before_independent_test": accepted,
                            "rejection_reasons": ";".join(reasons)})
        if accepted and evaluate_test:
            test_score = _score(proposal["expression"], independent, parents)
            test_daily = daily_rank_ic(independent, test_score, protocol.min_stocks_per_day)
            test_rows.append({"id": proposal["id"], "test_days": len(test_daily),
                              "test_rank_ic_after_train_orientation": sign * _mean_ic(test_daily),
                              "test_coverage": float(test_score.notna().mean())})
    counts = {name: {"rows": len(frame), "days": int(frame.date.nunique()),
                     "first_signal": str(frame.date.min().date()) if len(frame) else None,
                     "last_signal": str(frame.date.max().date()) if len(frame) else None,
                     "last_label_exit": str(frame.label_exit_5.max().date()) if len(frame) else None}
              for name, frame in (("train", train), ("validation", valid),
                                  ("independent_test", independent))}
    return pd.DataFrame(screen_rows), pd.DataFrame(test_rows), counts


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--valid-start", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--screen-only", action="store_true", help="Feedback round: do not read or evaluate test rows")
    args = parser.parse_args()
    protocol = Protocol(args.train_start, args.valid_start, args.test_start, args.test_end)
    proposals = json.loads(args.proposals.read_text(encoding="utf-8"))
    import pyarrow.parquet as pq
    schema = pq.read_schema(args.panel)
    inputs = allowed_inputs(schema.names)
    clean, _ = partition_proposals(proposals, inputs)
    parents = sorted({p for item in clean for p in item["parents"]})
    columns = ["date", "stockid", "label_5", "label_exit_5", *parents]
    filters = [("date", "<", pd.Timestamp(args.test_start))] if args.screen_only else None
    panel = pd.read_parquet(args.panel, columns=columns, filters=filters)
    screening, independent, counts = run_lab(panel, proposals, protocol, evaluate_test=not args.screen_only)
    args.out.mkdir(parents=True, exist_ok=True)
    screening.to_csv(args.out / "screening.csv", index=False, encoding="utf-8-sig")
    independent.to_csv(args.out / "independent_test.csv", index=False, encoding="utf-8-sig")
    manifest = {"protocol": asdict(protocol), "segment_counts": counts,
                "proposals_sha256": _sha(args.proposals), "panel_sha256": _sha(args.panel),
                "proposal_count": len(proposals), "valid_formula_count": len(clean),
                "accepted_before_independent_test": int(screening.accepted_before_independent_test.sum()),
                "llm_invocation_in_this_cli": False,
                "independent_test_evaluated": not args.screen_only,
                "scope": "factor-screening research; no trading or LightGBM retrain"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(json.dumps({"proposals": len(clean), "accepted":
                      manifest["accepted_before_independent_test"], "out": str(args.out)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
