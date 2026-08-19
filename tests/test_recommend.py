"""개선안 도출·기대효과·ROI 검증 (M5).

핵심 질문:
  1. **조치안이 파라미터 성격에 맞는가?** — 카운터는 PM, 연속값은 규격
  2. **실행 가능한 안을 내는가?** — 표본 대부분을 옮기라는 안은 개선안이 아니다
  3. **기대효과가 과대평가되지 않는가?** — 보정된 확률의 절대값을 쓰면 안 된다
  4. **가정치를 바꾸면 결과가 따라 바뀌는가?** — ROI는 가정의 함수다
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap", reason="shap 미설치 시 건너뜀")

from wafermap.analysis import attribution, recommend  # noqa: E402
from wafermap.models import cause_model  # noqa: E402
from tests.test_cause_model import _make_fdc  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    """P020(식각) 스텝에 chamber_pressure 이탈을 심은 모델."""
    fdc, case, control = _make_fdc(culprit="chamber_pressure", shift=3.0, seed=21)
    result = cause_model.fit(fdc, "P020", "Edge-Ring", case, control)
    evidence = attribution.build_evidence(result, fdc, case, control, top_n=5)
    return result, evidence


# ── 모델이 반사실 계산에 필요한 것을 들고 있는가 ─────────────────────────


def test_result_keeps_booster_and_population(fitted):
    """M5가 재예측하려면 학습된 모델과 전체 표본이 필요하다."""
    result, _ = fitted
    assert result.booster is not None
    X, y = result.population
    assert len(X) == len(y) == result.n_case + result.n_control


def test_population_differs_from_shap_sample():
    """SHAP용 표본은 불량을 과대 표집한다 — 모집단과 구분되어야 한다 ★.

    왜 중요한가: 여기서 불량률을 재면 실제보다 훨씬 높게 나온다.
        기대효과 추정의 분모가 틀리면 ROI 자릿수가 통째로 틀린다.
    """
    fdc, case, control = _make_fdc(n_case=60, n_control=900, seed=22)
    result = cause_model.fit(fdc, "P020", "P", case, control, max_background=200)

    X_pop, y_pop = result.population
    assert len(X_pop) > len(result.X), "모집단이 SHAP 표본으로 대체됐다"
    assert y_pop.mean() < result.y.mean(), "SHAP 표본이 불량을 과대 표집하지 않았다"


# ── 권고 구간 ────────────────────────────────────────────────────────────


def test_safe_band_respects_move_budget(fitted):
    """옮겨야 하는 웨이퍼 비율이 예산 안에 들어와야 한다."""
    result, _ = fitted
    band = recommend.safe_band(result, "chamber_pressure_mean", max_move_fraction=0.2)

    assert band is not None
    assert band["move_fraction"] <= 0.2 + 1e-9, "이동 예산을 초과했다"
    assert band["coverage"] + band["move_fraction"] == pytest.approx(1.0)


def test_safe_band_is_wider_than_shap_optimum(fitted):
    """M4의 recommend_spec보다 넓어야 한다 (실행 가능성을 위해 양보한 것) ★."""
    result, _ = fitted
    strict = cause_model.recommend_spec(result, "chamber_pressure_mean")
    feasible = recommend.safe_band(result, "chamber_pressure_mean", max_move_fraction=0.25)

    assert strict is not None and feasible is not None
    assert feasible["coverage"] >= strict["coverage"]


def test_safe_band_stays_within_observed_range(fitted):
    """관측 범위를 벗어나면 외삽이다."""
    result, _ = fitted
    band = recommend.safe_band(result, "chamber_pressure_mean")
    assert band["low"] >= band["current_min"] - 1e-6
    assert band["high"] <= band["current_max"] + 1e-6


def test_safe_band_rejects_unknown_column(fitted):
    result, _ = fitted
    with pytest.raises(KeyError):
        recommend.safe_band(result, "not_a_param")


# ── 조치안 ───────────────────────────────────────────────────────────────


def test_actions_exclude_measurements(fitted):
    """계측값은 조치안이 될 수 없다 (M4 발견 1)."""
    result, evidence = fitted
    actions = recommend.build_actions(evidence, result)
    for action in actions:
        if action.param != "-":
            from wafermap.config import is_controllable
            assert is_controllable(action.param), f"계측값이 조치안에 들어갔다: {action.param}"


def test_counter_param_gets_pm_action():
    """카운터 파라미터는 규격 강화가 아니라 PM 주기 단축으로 나와야 한다 ★.

    왜: `pad_life`(누적 처리량)를 "1000~1100으로 유지하시오"라고 하면 말이 안 된다.
        카운터는 계속 증가하는 값이고, 조작 손잡이는 **언제 리셋하느냐**뿐이다.
    """
    fdc, case, control = _make_fdc(culprit="chamber_pressure", shift=2.5, seed=23)
    # CMP 스텝의 실제 카운터 파라미터 이름으로 바꿔 단다 (P050의 pad_life)
    fdc = fdc.rename(columns={"chamber_pressure_mean": "pad_life_mean",
                              "chamber_pressure_std": "pad_life_std"})
    fdc["step_id"] = "P050"
    result = cause_model.fit(fdc, "P050", "Center", case, control)
    evidence = attribution.build_evidence(result, fdc, case, control, top_n=5)
    actions = recommend.build_actions(evidence, result)

    kinds = {a.param: a.kind for a in actions}
    assert kinds.get("pad_life") == "pm_shortening"


def test_weak_evidence_is_labeled(fitted):
    """분포 증거가 없는 조치안은 '약함'으로 표시되어야 한다."""
    result, evidence = fitted
    actions = recommend.build_actions(evidence, result, top_n=5)
    labels = {a.evidence for a in actions if a.param != "-"}
    assert labels, "조치안이 하나도 없다"
    assert all(label in {"강함", "보통"} or label.startswith("약함") for label in labels)


def test_low_auc_triggers_sampling_action():
    """모델이 신호를 못 잡으면 규격을 조이는 대신 데이터를 더 모으라고 해야 한다 ★★.

    왜 이게 중요한가: 근거가 약할 때 개선안을 내는 것은 **근거 없는 규제**다.
        현장에서 지켜지지도 않고, 분석의 신뢰만 잃는다. "모른다"고 말할 수 있어야 한다.
    """
    fdc, case, control = _make_fdc(culprit="none", shift=0.0, seed=24)
    result = cause_model.fit(fdc, "P020", "P", case, control)
    actions = recommend.build_actions(
        attribution.build_evidence(result, fdc, case, control), result
    )
    assert any(a.kind == "sampling" for a in actions)


def test_chamber_matching_only_when_stratified(fitted):
    result, evidence = fitted
    without = recommend.build_actions(evidence, result)
    with_chamber = recommend.build_actions(evidence, result, chamber_id="ETCH-A/ch1")

    assert not any(a.kind == "chamber_matching" for a in without)
    assert any(a.kind == "chamber_matching" for a in with_chamber)


def test_move_budget_is_shared_across_parameters(fitted):
    """파라미터 3개를 동시에 조여도 합계 이동 비율이 예산을 크게 넘지 않아야 한다 ★.

    왜: 개별로 25%씩 허용하면 3개 합쳐 58%가 움직인다. 개별로는 온건해 보이는 안이
        합치면 공정 재설계가 되는 것이다.
    """
    result, evidence = fitted
    actions = recommend.build_actions(evidence, result, top_n=3, move_budget=0.25)
    cf = recommend.counterfactual_from_actions(result, actions)

    assert cf is not None
    assert cf.move_fraction <= 0.35, f"합계 이동 비율이 너무 크다 ({cf.move_fraction:.0%})"


# ── 반사실 시뮬레이션 ────────────────────────────────────────────────────


def test_counterfactual_reduces_predicted_rate(fitted):
    """원인 파라미터를 안전 구간으로 옮기면 예측 불량률이 내려가야 한다."""
    result, _ = fitted
    band = recommend.safe_band(result, "chamber_pressure_mean")
    cf = recommend.counterfactual(result, {"chamber_pressure_mean": (band["low"], band["high"])})

    assert cf.improved_score < cf.baseline_score
    assert 0.0 < cf.relative_reduction <= 1.0
    assert cf.yield_gain_pp > 0


def test_counterfactual_uses_observed_rate_not_model_probability(fitted):
    """예측 확률의 절대값이 아니라 관측 불량률을 기준으로 삼아야 한다 ★★.

    왜: 이 모델은 scale_pos_weight로 불균형을 보정해 학습했다. 예측 확률의 절대값은
        실제 불량률보다 훨씬 크다. 그대로 "예상 불량률"이라고 보고하면 완전히 틀린
        숫자가 나간다. 절대값은 버리고 감소 **비율**만 실측 불량률에 적용한다.
    """
    result, _ = fitted
    band = recommend.safe_band(result, "chamber_pressure_mean")
    cf = recommend.counterfactual(result, {"chamber_pressure_mean": (band["low"], band["high"])})

    _, y = result.population
    assert cf.observed_defect_rate == pytest.approx(float(y.mean()))
    # 보정된 모델의 예측 평균은 실제 불량률보다 크다 — 그래서 절대값을 못 쓴다
    assert cf.baseline_score > cf.observed_defect_rate
    assert cf.expected_defect_rate <= cf.observed_defect_rate


def test_no_change_when_band_covers_everything(fitted):
    """모든 값이 이미 구간 안이면 변화가 없어야 한다 (0 나눗셈·환각 금지)."""
    result, _ = fitted
    values = result.population[0]["chamber_pressure_mean"]
    cf = recommend.counterfactual(
        result, {"chamber_pressure_mean": (float(values.min()), float(values.max()))}
    )
    assert cf.n_clipped == 0
    assert cf.yield_gain_pp == pytest.approx(0.0, abs=1e-9)
    assert any("변화가 계산되지 않았다" in note for note in cf.caveats())


def test_counterfactual_flags_infeasible_plan(fitted):
    """표본 대부분을 옮기라는 안은 실행 가능성 경고를 달아야 한다 ★."""
    result, _ = fitted
    values = result.population[0]["chamber_pressure_mean"]
    tiny = (float(values.quantile(0.0)), float(values.quantile(0.2)))
    cf = recommend.counterfactual(result, {"chamber_pressure_mean": tiny})

    assert cf.move_fraction > 0.35
    assert cf.feasibility.startswith("낮음")
    assert any("공정 재설계" in note for note in cf.caveats())


def test_counterfactual_rejects_unknown_column(fitted):
    result, _ = fitted
    with pytest.raises(KeyError):
        recommend.counterfactual(result, {"nope_mean": (0.0, 1.0)})


def test_counterfactual_rejects_empty_clips(fitted):
    result, _ = fitted
    with pytest.raises(ValueError, match="권고 구간이 없습니다"):
        recommend.counterfactual(result, {})


def test_counterfactual_requires_booster(fitted):
    from dataclasses import replace as dc_replace
    result, _ = fitted
    stripped = dc_replace(result, booster=None)
    with pytest.raises(ValueError, match="학습된 모델이 없습니다"):
        recommend.counterfactual(stripped, {"chamber_pressure_mean": (0.0, 1.0)})


# ── ROI ──────────────────────────────────────────────────────────────────


def test_roi_formula_matches_design():
    """설계서 §3 M5.4의 계산식과 일치해야 한다."""
    assumptions = recommend.RoiAssumptions(
        wafer_value_krw=5_000_000, monthly_wafers=3_000, adoption_rate=1.0,
        annual_running_cost_krw=0.0,
    )
    roi = recommend.estimate_roi(1.0, assumptions)

    # 3,000 × 12 × 1% = 360장, 360 × 500만원 = 18억원
    assert roi.recovered_wafers_year == pytest.approx(360.0)
    assert roi.gross_saving_krw == pytest.approx(1_800_000_000.0)


def test_adoption_rate_discounts_gain():
    """적용률은 기대효과를 깎아야 한다 (부풀림 방지)."""
    full = recommend.estimate_roi(1.0, recommend.RoiAssumptions(adoption_rate=1.0))
    half = recommend.estimate_roi(1.0, recommend.RoiAssumptions(adoption_rate=0.5))
    assert half.recovered_wafers_year == pytest.approx(full.recovered_wafers_year / 2)


def test_negative_gain_is_floored():
    """수율이 나빠지는 시뮬레이션 결과를 이익으로 환산하면 안 된다."""
    roi = recommend.estimate_roi(-3.0)
    assert roi.yield_gain_pp == 0.0
    assert roi.recovered_wafers_year == 0.0
    assert roi.payback_months is None


def test_payback_is_none_when_cost_exceeds_benefit():
    """운영비를 못 넘으면 회수기간이 계산되지 않아야 한다 ★.

    왜: 여기서 억지로 숫자를 만들면 '적자 조치'가 '회수 가능'으로 보고된다.
    """
    roi = recommend.estimate_roi(
        0.01, recommend.RoiAssumptions(annual_running_cost_krw=10_000_000_000.0)
    )
    assert roi.net_saving_krw < 0
    assert roi.payback_months is None


def test_assumptions_reject_impossible_values():
    with pytest.raises(ValueError):
        recommend.RoiAssumptions(wafer_value_krw=0)
    with pytest.raises(ValueError):
        recommend.RoiAssumptions(monthly_wafers=-1)
    with pytest.raises(ValueError):
        recommend.RoiAssumptions(adoption_rate=1.5)


def test_every_assumption_has_a_badge():
    """가정치는 전부 '가정인지 데이터인지' 표시가 있어야 한다 (설계서 §3 M5.4)."""
    from dataclasses import fields
    for f in fields(recommend.RoiAssumptions):
        assert f.name in recommend.ASSUMPTION_BADGES, f"{f.name}에 출처 배지가 없다"


def test_sensitivity_scales_monotonically():
    """가정치를 키우면 순효익도 커져야 한다."""
    table = recommend.roi_sensitivity(1.0)
    assert list(table["net_saving_krw"]) == sorted(table["net_saving_krw"])
    assert len(table) == 5


def test_sensitivity_rejects_non_numeric_field():
    with pytest.raises((TypeError, AttributeError)):
        recommend.roi_sensitivity(1.0, field_name="not_a_field")


def test_format_krw_reads_naturally():
    roi = recommend.estimate_roi(1.0)
    assert roi.format_krw(1_850_000_000) == "18.5억원"
    assert roi.format_krw(3_400_000) == "340만원"
    assert roi.format_krw(500) == "500원"
