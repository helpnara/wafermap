#!/usr/bin/env python3
"""원인 분석 화면이 즉시 읽을 아티팩트를 만든다 (M6-3).

왜 사전 계산하나 (설계서 §4.6): 커미널리티·SHAP·교호작용을 화면에서 매번 돌리면
패턴 하나에 30초가 넘는다. 무거운 계산은 여기서 한 번 하고, 앱은 결과만 읽는다.

사용법:
    python scripts/build_rootcause.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

warnings.filterwarnings("ignore", message=".*binary classifier with TreeExplainer.*")

import numpy as np  # noqa: E402

from wafermap.analysis import attribution, commonality, interaction, spc  # noqa: E402
from wafermap.config import (  # noqa: E402
    CAUSE_RULES,
    INTERACTION_CAUSE_RULES,
    STEPS_BY_ID,
    TEST_CAUSE_RULES,
    TEST_STEP_ID,
    processed_dir,
)
from wafermap.data import loader  # noqa: E402
from wafermap.models import cause_model  # noqa: E402

ARTIFACT_NAME = "rootcause.json"
TOP_RANKING = 12


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"직렬화할 수 없는 타입: {type(obj)}")


def _clean(value: float) -> float | None:
    """NaN/Inf는 JSON에 담을 수 없다 — None으로 바꾼다."""
    return None if value is None or not np.isfinite(value) else float(value)


def build_pattern(wm, fdc, gt, pattern: str) -> dict | None:
    rule = CAUSE_RULES.get(pattern)
    if rule is None or rule.step_id is None:
        return None

    split = spc.split_by_pattern(wm, pattern)
    sub = gt[gt["pattern_label"] == pattern]

    payload: dict = {
        "pattern": pattern,
        "n_case": len(split.case_ids),
        "n_control": len(split.control_ids),
        "mechanisms": sub["cause_mechanism"].value_counts().to_dict(),
        "truth": {
            "process_step": rule.step_id,
            "process_step_name": STEPS_BY_ID[rule.step_id].name_ko,
            "mechanism": rule.mechanism,
            "params": [p.param for p in rule.perturbations],
            "equip": sorted(
                {str(e) for e in sub["true_root_equip"].dropna().unique()}
            ),
        },
    }
    if pattern in TEST_CAUSE_RULES:
        payload["truth"]["test_mechanism"] = TEST_CAUSE_RULES[pattern].mechanism

    # ── 커미널리티 랭킹 ─────────────────────────────────────────────────
    ranking = commonality.analyze(
        fdc, split.case_ids, split.control_ids, wafer_master=wm, unit="lot"
    ).head(TOP_RANKING)
    payload["ranking"] = [
        {
            "rank": i + 1,
            "step_id": str(row["step_id"]),
            "step_name": STEPS_BY_ID[str(row["step_id"])].name_ko,
            "chamber_id": str(row["chamber_id"]),
            "odds_ratio": _clean(row["odds_ratio"]),
            "p_adj": _clean(row["p_adj"]),
            "case_rate": _clean(row["case_rate"]),
            "control_rate": _clean(row["control_rate"]),
            "significant": bool(row["significant"]),
            "is_test": str(row["step_id"]) == TEST_STEP_ID,
        }
        for i, (_, row) in enumerate(ranking.iterrows())
    ]

    # ── 공정 축 vs 검사 축 ──────────────────────────────────────────────
    axes = commonality.by_axis(
        fdc, split.case_ids, split.control_ids, wafer_master=wm, unit="lot"
    )
    payload["axes"] = [
        {
            "axis": str(row["axis"]),
            "axis_ko": str(row["axis_ko"]),
            "step_id": str(row["step_id"]),
            "chamber_id": str(row["chamber_id"]),
            "odds_ratio": _clean(row["odds_ratio"]),
            "rank_overall": int(row["rank_overall"]),
        }
        for _, row in axes.iterrows()
    ]

    # ── 껍질 벗기기 ─────────────────────────────────────────────────────
    payload["layers"] = [
        {
            "rank": layer.rank,
            "step_id": layer.step_id,
            "chamber_id": layer.chamber_id,
            "odds_ratio": _clean(layer.odds_ratio),
            "n_case_before": layer.n_case_before,
            "n_explained": layer.n_explained,
            "is_test": layer.is_test_equipment,
        }
        for layer in commonality.peel(
            fdc, split.case_ids, split.control_ids, wafer_master=wm
        )
    ]

    # ── 원인 파라미터 (SHAP) ────────────────────────────────────────────
    # ★ 층화 챔버는 **원인 스텝 안에서** 골라야 한다.
    #   공정 축 1위는 전체 공정 스텝을 통틀어 뽑은 것이라 다른 스텝의 챔버일 수 있다.
    #   실제로 Edge-Ring에서 세정(P060) 챔버가 뽑혀 식각(P020) 행을 거르자 표본이 0이 됐다.
    same_step = [r for r in payload["ranking"] if r["step_id"] == rule.step_id]
    chamber = same_step[0]["chamber_id"] if same_step else None
    try:
        result = cause_model.fit(
            fdc, rule.step_id, pattern, split.case_ids, split.control_ids,
            chamber_id=chamber, suffixes=cause_model.TRACE_SUFFIXES,
        )
        evidence = attribution.build_evidence(
            result, fdc, split.case_ids, split.control_ids, chamber_id=chamber, top_n=8
        )
        payload["model"] = {"auc": _clean(result.auc), "chamber_id": chamber}
        payload["evidence"] = [
            {
                "column": e.column, "param": e.param, "role": e.role,
                "shap_rank": e.shap_rank,
                "shap_importance": _clean(e.shap_importance),
                "direction": e.direction,
                "delta_sigma": _clean(e.delta_sigma),
                "cliffs_delta": _clean(e.cliffs_delta),
                "ks_pvalue": _clean(e.ks_pvalue),
                "spec_violation_rate": _clean(e.spec_violation_rate),
                "is_controllable": e.is_controllable,
                "has_distribution_evidence": e.has_distribution_evidence,
            }
            for e in evidence
        ]
    except ValueError as exc:
        payload["model"] = {"auc": None, "chamber_id": chamber, "error": str(exc)}
        payload["evidence"] = []

    # ── 교호작용 ────────────────────────────────────────────────────────
    payload["interaction"] = None
    irule = INTERACTION_CAUSE_RULES.get(pattern)
    if irule is not None:
        case = list(sub.loc[sub["cause_mechanism"] == "interaction", "wafer_id"])
        if len(case) >= 15:
            ids = case + list(split.control_ids)
            matrix = interaction.cross_step_matrix(
                fdc, ids, steps=(irule.step_a, irule.step_b)
            )
            matrix = matrix.loc[[w for w in ids if w in matrix.index]]
            is_case = matrix.index.isin(set(case))
            pairs = interaction.screen_pairs(matrix, is_case, top_n=100)
            hidden = [p for p in pairs if p.hidden_from_main_effects][:5]
            equips = interaction.equipment_pairs(
                fdc, case, list(gt["wafer_id"]),
                step_a=irule.step_a, step_b=irule.step_b, top_n=5,
            )
            payload["interaction"] = {
                "n_case": len(case),
                "mechanism": irule.mechanism,
                "truth_pair": [
                    f"{irule.step_a}.{irule.param_a}", f"{irule.step_b}.{irule.param_b}"
                ],
                "truth_equip_pair": list(irule.equip_pair),
                "pairs": [
                    {
                        "param_a": p.param_a, "param_b": p.param_b,
                        "contrast": _clean(p.contrast),
                        "main_a": _clean(p.main_a), "main_b": _clean(p.main_b),
                        "p_adj": _clean(p.p_adj),
                        "cells": {
                            "ll": [_clean(p.rate_ll), p.n_ll],
                            "lh": [_clean(p.rate_lh), p.n_lh],
                            "hl": [_clean(p.rate_hl), p.n_hl],
                            "hh": [_clean(p.rate_hh), p.n_hh],
                        },
                    }
                    for p in hidden
                ],
                "equip_pairs": [
                    {
                        "chamber_a": e.chamber_a, "chamber_b": e.chamber_b,
                        "n_pair": e.n_pair,
                        "rate_pair": _clean(e.rate_pair),
                        "rate_a_only": _clean(e.rate_a_only),
                        "rate_b_only": _clean(e.rate_b_only),
                        "excess": _clean(e.excess),
                    }
                    for e in equips
                ],
            }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="원인 분석 아티팩트 생성")
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    args = parser.parse_args()

    if not loader.is_built(args.source):
        print(f"❌ 데이터가 없습니다. python scripts/build_dataset.py --source {args.source}",
              file=sys.stderr)
        return 1

    wm = loader.load_wafer_master(args.source)
    fdc = loader.load_fdc_summary(args.source)
    gt = loader.load_ground_truth(args.source)

    out: dict = {"source": args.source, "patterns": {}}
    for pattern in sorted(p for p, r in CAUSE_RULES.items() if r.step_id):
        print(f"  · {pattern} …", end="", flush=True)
        try:
            payload = build_pattern(wm, fdc, gt, pattern)
        except ValueError as exc:
            print(f" 건너뜀 ({exc})")
            continue
        if payload is None:
            print(" 건너뜀")
            continue
        out["patterns"][pattern] = payload
        auc = payload["model"].get("auc")
        print(
            f" 랭킹 {len(payload['ranking'])} · AUC {auc:.3f}" if auc
            else f" 랭킹 {len(payload['ranking'])} · 모델 없음"
        )

    path = processed_dir(args.source) / ARTIFACT_NAME
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"\n✅ 저장: {path} ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
