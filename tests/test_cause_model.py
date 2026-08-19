"""원인 파라미터 규명 모델과 기여도 해석 검증.

핵심 질문:
  1. **심어 둔 원인 파라미터를 찾아내는가?**
  2. **계측값과 조작 가능한 파라미터를 구분하는가?** — 개선안의 대상이 달라진다
  3. **AUC가 낮을 때 그 사실이 드러나는가?** — 신뢰할 수 없는 SHAP을 걸러야 한다
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap", reason="shap 미설치 시 건너뜀")

from wafermap.analysis import attribution  # noqa: E402
from wafermap.models import cause_model  # noqa: E402


def _make_fdc(
    n_case: int = 60,
    n_control: int = 300,
    culprit: str = "chamber_pressure",
    shift: float = 3.0,
    seed: int = 0,
):
    """인공 FDC 데이터. culprit 파라미터만 불량군에서 이동시킨다."""
    rng = np.random.default_rng(seed)
    n = n_case + n_control
    is_case = np.array([True] * n_case + [False] * n_control)

    params = {
        "chamber_pressure": (45.0, 0.8),
        "rf_power": (1500.0, 12.0),
        "o2_flow": (20.0, 0.6),
        "cf4_flow": (120.0, 1.5),
        "electrode_temp": (60.0, 0.7),
    }
    data = {
        "wafer_id": [f"W{i:04d}" for i in range(n)],
        "step_id": "P020",
        "step_name": "Etch",
        "equip_id": "ETCH-A",
        "chamber_id": "ETCH-A/ch1",
        "recipe_id": "RCP-STD-01",
        "run_time": pd.Timestamp("2026-01-01"),
    }
    for name, (nominal, sigma) in params.items():
        values = rng.normal(nominal, sigma, n)
        if name == culprit:
            values = values + is_case * shift * sigma
        data[f"{name}_mean"] = values
        data[f"{name}_std"] = np.abs(rng.normal(sigma * 0.3, sigma * 0.05, n))

    fdc = pd.DataFrame(data)
    case_ids = list(fdc.loc[is_case, "wafer_id"])
    control_ids = list(fdc.loc[~is_case, "wafer_id"])
    return fdc, case_ids, control_ids


# ── 파라미터 선택 ────────────────────────────────────────────────────────


def test_parameter_columns_excludes_identifiers():
    """식별 컬럼이 피처로 들어가면 안 된다."""
    fdc, _, _ = _make_fdc()
    cols = cause_model.parameter_columns(fdc)

    for forbidden in cause_model.ID_COLUMNS:
        assert forbidden not in cols
    assert all(c.endswith(("_mean", "_std")) for c in cols)


def test_parameter_columns_drops_constant():
    """값이 하나뿐인 컬럼은 정보가 없어 제외해야 한다."""
    fdc, _, _ = _make_fdc()
    fdc["constant_mean"] = 1.0
    assert "constant_mean" not in cause_model.parameter_columns(fdc)


def test_min_max_excluded_to_avoid_split_importance():
    """_min/_max를 넣지 않아야 한다 (기여도가 분산되는 것을 막는다)."""
    fdc, _, _ = _make_fdc()
    fdc["chamber_pressure_min"] = fdc["chamber_pressure_mean"] - 1
    fdc["chamber_pressure_max"] = fdc["chamber_pressure_mean"] + 1

    cols = cause_model.parameter_columns(fdc)
    assert "chamber_pressure_min" not in cols
    assert "chamber_pressure_max" not in cols


# ── 원인 탐지 ────────────────────────────────────────────────────────────


def test_finds_injected_culprit_parameter():
    """인공적으로 이동시킨 파라미터가 SHAP 1위여야 한다."""
    fdc, case, control = _make_fdc(culprit="chamber_pressure", shift=3.0, seed=1)
    result = cause_model.fit(fdc, "P020", "TestPattern", case, control)

    assert result.auc > 0.8, f"모델이 신호를 못 잡음 (AUC {result.auc:.3f})"
    assert result.shap_importance.index[0] == "chamber_pressure_mean"


def test_direction_matches_injected_shift():
    """이탈 방향이 실제 주입 방향과 일치해야 한다."""
    up, case, control = _make_fdc(culprit="rf_power", shift=+3.0, seed=2)
    result_up = cause_model.fit(up, "P020", "P", case, control)
    assert result_up.direction["rf_power_mean"] > 0

    down, case2, control2 = _make_fdc(culprit="rf_power", shift=-3.0, seed=3)
    result_down = cause_model.fit(down, "P020", "P", case2, control2)
    assert result_down.direction["rf_power_mean"] < 0


def test_low_auc_when_no_signal():
    """신호가 없으면 AUC가 0.5 근처여야 한다 ★.

    왜 중요한가: AUC가 낮은데도 SHAP을 그대로 해석하면 노이즈를 원인으로
        보고하게 된다. 모델이 먼저 유효해야 해석이 의미를 갖는다.
    """
    fdc, case, control = _make_fdc(culprit="none_of_them", shift=0.0, seed=4)
    result = cause_model.fit(fdc, "P020", "P", case, control)
    assert result.auc < 0.75, f"신호가 없는데 AUC가 높다 ({result.auc:.3f})"


def test_rejects_insufficient_samples():
    fdc, case, control = _make_fdc(n_case=5, n_control=5)
    with pytest.raises(ValueError, match="표본이 부족"):
        cause_model.fit(fdc, "P020", "P", case, control)


def test_chamber_filter_reduces_samples():
    """챔버를 지정하면 그 챔버 웨이퍼만 남아야 한다 (층화)."""
    fdc, case, control = _make_fdc(seed=5)
    fdc.loc[fdc.index[:100], "chamber_id"] = "ETCH-A/ch2"

    X_all, _ = cause_model.build_matrix(fdc, "P020", case, control)
    X_one, _ = cause_model.build_matrix(fdc, "P020", case, control, chamber_id="ETCH-A/ch1")
    assert len(X_one) < len(X_all)


# ── dependence 곡선 ─────────────────────────────────────────────────────


def test_dependence_curve_is_monotonic_for_shifted_param():
    """이동시킨 파라미터는 값이 커질수록 기여도가 올라가야 한다."""
    fdc, case, control = _make_fdc(culprit="chamber_pressure", shift=3.0, seed=6)
    result = cause_model.fit(fdc, "P020", "P", case, control)
    curve = cause_model.dependence_curve(result, "chamber_pressure_mean")

    assert not curve.empty
    # 가장 높은 구간의 기여도가 가장 낮은 구간보다 커야 한다
    assert curve.iloc[-1]["mean_shap"] > curve.iloc[0]["mean_shap"]
    # 불량률도 같이 올라가야 한다
    assert curve.iloc[-1]["case_rate"] > curve.iloc[0]["case_rate"]


def test_dependence_curve_rejects_unknown_feature():
    fdc, case, control = _make_fdc(seed=7)
    result = cause_model.fit(fdc, "P020", "P", case, control)
    with pytest.raises(KeyError):
        cause_model.dependence_curve(result, "not_a_param")


def test_recommend_spec_within_observed_range():
    """권고 구간이 관측 범위를 벗어나면 안 된다 (외삽 방지)."""
    fdc, case, control = _make_fdc(culprit="chamber_pressure", shift=3.0, seed=8)
    result = cause_model.fit(fdc, "P020", "P", case, control)
    spec = cause_model.recommend_spec(result, "chamber_pressure_mean")

    assert spec is not None
    assert spec["low"] >= spec["current_min"] - 1e-6
    assert spec["high"] <= spec["current_max"] + 1e-6
    assert 0 < spec["coverage"] <= 1


# ── 개별 웨이퍼 설명 ─────────────────────────────────────────────────────


def test_explain_wafer_returns_top_contributions():
    fdc, case, control = _make_fdc(seed=9)
    result = cause_model.fit(fdc, "P020", "P", case, control)
    frame = cause_model.explain_wafer(result, 0, top_n=3)

    assert len(frame) == 3
    assert set(frame.columns) == {"parameter", "value", "shap", "direction"}
    # 기여도 절댓값 내림차순이어야 한다
    assert (frame["shap"].abs().diff().dropna() <= 1e-9).all()


def test_explain_wafer_rejects_bad_index():
    fdc, case, control = _make_fdc(seed=10)
    result = cause_model.fit(fdc, "P020", "P", case, control)
    with pytest.raises(IndexError):
        cause_model.explain_wafer(result, 99_999)


# ── 조치 가능성 구분 ★ ──────────────────────────────────────────────────


def test_measurement_params_are_flagged():
    """계측값이 '조작 불가'로 분류되어야 한다."""
    from wafermap.config import is_controllable

    assert not is_controllable("thickness_sigma_mean")
    assert not is_controllable("particle_count_mean")
    assert not is_controllable("cd_mean_mean")
    assert is_controllable("dep_temp_mean")
    assert is_controllable("chamber_pressure_mean")


def test_actionable_ranking_removes_measurements():
    """조치 가능 필터가 계측값을 제거해야 한다."""
    evidence = [
        attribution.ParamEvidence(
            column="thickness_sigma_mean", param="thickness_sigma", role="measurement",
            shap_importance=2.0, shap_rank=1, direction=1,
            case_mean=10.0, control_mean=8.0, delta_sigma=1.5,
            cliffs_delta=0.5, ks_pvalue=0.001, spec_violation_rate=0.1,
        ),
        attribution.ParamEvidence(
            column="dep_temp_mean", param="dep_temp", role="control",
            shap_importance=1.0, shap_rank=2, direction=1,
            case_mean=625.0, control_mean=620.0, delta_sigma=2.0,
            cliffs_delta=0.6, ks_pvalue=0.001, spec_violation_rate=0.2,
        ),
    ]
    actionable = attribution.actionable_ranking(evidence)

    assert len(actionable) == 1
    assert actionable[0].param == "dep_temp"


def test_base_param_name_strips_suffix():
    assert attribution.base_param_name("chamber_pressure_mean") == "chamber_pressure"
    assert attribution.base_param_name("rf_power_std") == "rf_power"
    assert attribution.base_param_name("pad_life") == "pad_life"


# ── 분포 비교 (모델과 독립적인 증거) ─────────────────────────────────────


def test_cliffs_delta_detects_separation():
    """완전히 분리된 두 분포는 |delta|가 1에 가까워야 한다."""
    assert attribution.cliffs_delta(np.arange(100, 200), np.arange(0, 100)) == pytest.approx(1.0)
    assert attribution.cliffs_delta(np.arange(0, 100), np.arange(100, 200)) == pytest.approx(-1.0)


def test_cliffs_delta_zero_for_identical():
    values = np.arange(100, dtype=float)
    assert abs(attribution.cliffs_delta(values, values)) < 0.05


def test_cliffs_delta_handles_empty():
    assert attribution.cliffs_delta(np.array([]), np.arange(10)) == 0.0


def test_compare_distributions_finds_shifted_param():
    """분포 비교가 이동된 파라미터를 잡아내야 한다 (모델 없이)."""
    fdc, case, control = _make_fdc(culprit="o2_flow", shift=3.0, seed=11)
    dist = attribution.compare_distributions(
        fdc, "P020", case, control, ["o2_flow_mean", "rf_power_mean"]
    ).set_index("column")

    assert abs(dist.loc["o2_flow_mean", "delta_sigma"]) > 2.0
    assert abs(dist.loc["o2_flow_mean", "cliffs_delta"]) > 0.5
    assert dist.loc["o2_flow_mean", "ks_pvalue"] < 0.01
    # 이동시키지 않은 파라미터는 차이가 없어야 한다
    assert abs(dist.loc["rf_power_mean", "cliffs_delta"]) < 0.3


def test_build_evidence_combines_shap_and_distribution():
    """SHAP 결과와 분포 증거가 함께 담겨야 한다."""
    fdc, case, control = _make_fdc(culprit="chamber_pressure", shift=3.0, seed=12)
    result = cause_model.fit(fdc, "P020", "P", case, control)
    evidence = attribution.build_evidence(result, fdc, case, control, top_n=5)

    assert evidence
    top = evidence[0]
    assert top.column == "chamber_pressure_mean"
    assert top.shap_rank == 1
    assert top.has_distribution_evidence, "분포 증거가 SHAP과 일치하지 않는다"
