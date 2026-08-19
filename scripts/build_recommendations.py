#!/usr/bin/env python3
"""M3→M4→M5 파이프라인을 전 패턴에 대해 돌려 앱이 즉시 읽을 아티팩트를 만든다.

왜 사전 계산하나 (설계서 §4.6): 화면에서 SHAP을 매번 계산하면 패턴 하나에 20~30초가
걸린다. 사용자가 슬라이더를 움직일 때마다 그만큼 기다리게 만들 수는 없다.
무거운 계산은 여기서 한 번 하고, 앱은 결과만 읽어 ROI만 실시간으로 다시 계산한다.

사용법:
    python scripts/build_recommendations.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

warnings.filterwarnings("ignore", message=".*binary classifier with TreeExplainer.*")

import numpy as np  # noqa: E402

from wafermap.analysis import attribution, commonality, recommend, spc  # noqa: E402
from wafermap.config import CAUSE_RULES, STEPS_BY_ID, processed_dir  # noqa: E402
from wafermap.data import loader  # noqa: E402
from wafermap.models import cause_model  # noqa: E402

ARTIFACT_NAME = "recommendations.json"


def build_pattern(wm, fdc, pattern: str) -> dict | None:
    """패턴 하나에 대해 M3→M4→M5를 이어 돌리고 직렬화 가능한 dict로 만든다."""
    rule = CAUSE_RULES.get(pattern)
    if rule is None or rule.step_id is None:
        return None

    split = spc.split_by_pattern(wm, pattern)
    ranking = commonality.analyze(
        fdc, split.case_ids, split.control_ids, wafer_master=wm, unit="lot"
    )
    same_step = ranking[ranking["step_id"] == rule.step_id]
    chamber = str(same_step.iloc[0]["chamber_id"]) if not same_step.empty else None

    result = cause_model.fit(
        fdc, rule.step_id, pattern, split.case_ids, split.control_ids, chamber_id=chamber
    )
    evidence = attribution.build_evidence(
        result, fdc, split.case_ids, split.control_ids, chamber_id=chamber
    )
    actions = recommend.build_actions(evidence, result, chamber_id=chamber)

    # 챔버로 층화했다면 그 챔버의 물량 비중을 구해 둔다 — ROI를 라인 기준으로 환산할 때 쓴다
    step_wafers = fdc[fdc["step_id"] == rule.step_id]
    scope = 1.0
    if chamber and len(step_wafers):
        scope = float((step_wafers["chamber_id"] == chamber).mean())
    cf = recommend.counterfactual_from_actions(result, actions, scope_fraction=scope)

    step = STEPS_BY_ID[result.step_id]
    payload = {
        "pattern": pattern,
        "step_id": result.step_id,
        "step_name": step.name_ko,
        "step_name_en": step.name_en,
        "mechanism": rule.mechanism,
        "chamber_id": chamber,
        "auc": result.auc,
        "n_case": result.n_case,
        "n_control": result.n_control,
        "truth_params": [p.param for p in rule.perturbations],
        "actions": [asdict(a) for a in actions],
        "evidence": [
            {
                "column": e.column, "param": e.param, "role": e.role,
                "shap_rank": e.shap_rank, "shap_importance": e.shap_importance,
                "direction": e.direction, "delta_sigma": e.delta_sigma,
                "cliffs_delta": e.cliffs_delta, "ks_pvalue": e.ks_pvalue,
                "spec_violation_rate": e.spec_violation_rate,
                "is_controllable": e.is_controllable,
                "has_distribution_evidence": e.has_distribution_evidence,
            }
            for e in evidence[:8]
        ],
        "counterfactual": None,
        "curves": {},
    }

    if cf is not None:
        payload["counterfactual"] = {
            "baseline_score": cf.baseline_score,
            "improved_score": cf.improved_score,
            "relative_reduction": cf.relative_reduction,
            "observed_defect_rate": cf.observed_defect_rate,
            "expected_defect_rate": cf.expected_defect_rate,
            "yield_gain_pp": cf.yield_gain_pp,
            "line_yield_gain_pp": cf.line_yield_gain_pp,
            "scope_fraction": cf.scope_fraction,
            "n_wafers": cf.n_wafers,
            "n_clipped": cf.n_clipped,
            "move_fraction": cf.move_fraction,
            "feasibility": cf.feasibility,
            "extrapolated": cf.extrapolated,
            "caveats": cf.caveats(),
        }

    # dependence 곡선 — 개선안 근거 그래프용
    for action in actions:
        if action.spec is None or action.column not in result.X.columns:
            continue
        curve = cause_model.dependence_curve(result, action.column)
        if not curve.empty:
            payload["curves"][action.column] = curve.to_dict(orient="list")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="M5 아티팩트 생성")
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    args = parser.parse_args()

    if not loader.is_built(args.source):
        print(f"❌ 데이터가 없습니다. python scripts/build_dataset.py --source {args.source}",
              file=sys.stderr)
        return 1

    wm = loader.load_wafer_master(args.source)
    fdc = loader.load_fdc_summary(args.source)
    patterns = sorted(p for p, r in CAUSE_RULES.items() if r.step_id)

    out: dict[str, object] = {"source": args.source, "patterns": {}}
    for pattern in patterns:
        print(f"  · {pattern} …", end="", flush=True)
        try:
            payload = build_pattern(wm, fdc, pattern)
        except ValueError as exc:
            print(f" 건너뜀 ({exc})")
            continue
        if payload is None:
            print(" 건너뜀 (원인 스텝 미정의)")
            continue
        out["patterns"][pattern] = payload
        cf = payload["counterfactual"]
        gain = (
            f"챔버 {cf['yield_gain_pp']:+.2f}%p · 라인 {cf['line_yield_gain_pp']:+.2f}%p"
            if cf else "—"
        )
        print(f" AUC {payload['auc']:.3f} · 조치 {len(payload['actions'])}건 · {gain}")

    path = processed_dir(args.source) / ARTIFACT_NAME
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"\n✅ 저장: {path} ({path.stat().st_size / 1024:.0f} KB)")
    return 0


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"직렬화할 수 없는 타입: {type(obj)}")


if __name__ == "__main__":
    raise SystemExit(main())
