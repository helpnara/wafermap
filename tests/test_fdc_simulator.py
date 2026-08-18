"""FDC 시뮬레이터 검증.

핵심 질문 세 가지:
  1. 라벨 분포가 WM-811K 실측 비율을 재현하는가?
  2. 인과 신호가 **실제로 존재하는가** — 원인 파라미터가 정말 이탈했는가?
  3. 인과 신호가 **너무 쉽지는 않은가** — 교락·위양성·위음성이 섞여 있는가?

2번이 없으면 M3/M4가 찾을 것이 없고, 3번이 없으면 분석 역량을 증명하지 못한다.
둘 다 데이터 생성 단계에서만 보장할 수 있으므로 여기서 검사한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from wafermap.config import CAUSE_RULES, PATTERN_LABELS, STEPS_BY_ID, WM811K_LABEL_COUNTS
from wafermap.data import schema
from wafermap.data.fdc_simulator import simulate


# ── 구조·스키마 ──────────────────────────────────────────────────────────


def test_output_passes_schema(sim):
    """생성 직후 스키마를 만족해야 한다 (wafer_master의 맵 통계 컬럼 제외)."""
    schema.validate(sim.fdc_summary, schema.FDC_SUMMARY)
    schema.validate(sim.fdc_trace, schema.FDC_TRACE)
    schema.validate(sim.ground_truth, schema.GROUND_TRUTH)


def test_every_wafer_visits_every_step(sim):
    n_steps = sim.fdc_summary["step_id"].nunique()
    per_wafer = sim.fdc_summary.groupby("wafer_id").size()
    assert per_wafer.eq(n_steps).all(), "일부 웨이퍼가 스텝을 건너뜀"


def test_lot_uses_single_chamber_per_step(sim):
    """같은 lot의 웨이퍼는 한 스텝에서 같은 챔버를 쓴다 (팹의 lot 단위 배정)."""
    df = sim.fdc_summary.merge(
        sim.wafer_master[["wafer_id", "lot_id"]], on="wafer_id"
    )
    n_chambers = df.groupby(["lot_id", "step_id"])["chamber_id"].nunique()
    assert n_chambers.eq(1).all(), "한 lot이 같은 스텝에서 여러 챔버를 사용함"


def test_wafer_ids_are_consistent_across_tables(sim):
    wm = set(sim.wafer_master["wafer_id"])
    assert set(sim.ground_truth["wafer_id"]) == wm
    assert set(sim.fdc_summary["wafer_id"]) == wm
    assert set(sim.fdc_trace["wafer_id"]) <= wm


def test_slot_numbers_are_within_lot_size(sim):
    assert sim.wafer_master["slot_no"].between(1, 25).all()


# ── 라벨 분포 ────────────────────────────────────────────────────────────


def test_label_distribution_matches_wm811k_ratio(sim):
    """합성 라벨 비율이 WM-811K 실측 비율과 맞아야 한다.

    왜: 합성에서 얻은 macro-F1이 실데이터로 옮겨 갈 수 있으려면 클래스 불균형이
        같아야 한다. 특히 Near-full(0.086%)의 희소성이 모델링 난이도의 핵심이다.
    """
    counts = sim.wafer_master["pattern_label"].value_counts()
    n = len(sim.wafer_master)
    total_real = sum(WM811K_LABEL_COUNTS.values())

    assert set(counts.index) <= set(PATTERN_LABELS)
    # 'none'이 압도적 다수여야 한다
    assert counts["none"] / n > 0.75

    for label in ("Edge-Ring", "Edge-Loc", "Center"):
        expected = WM811K_LABEL_COUNTS[label] / total_real
        actual = counts.get(label, 0) / n
        assert actual == pytest.approx(expected, abs=0.015), f"{label} 비율 이탈"


def test_rare_classes_are_present(sim):
    """희소 클래스도 최소 1장은 있어야 학습이 가능하다."""
    counts = sim.wafer_master["pattern_label"].value_counts()
    for label in ("Near-full", "Donut", "Random"):
        assert counts.get(label, 0) >= 1, f"{label}이 한 장도 생성되지 않음"


# ── 인과 신호 (있어야 한다) ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "pattern", ["Center", "Edge-Ring", "Edge-Loc", "Loc", "Scratch"]
)
def test_causal_parameters_are_shifted(sim, pattern):
    """원인 파라미터가 정상군 대비 설계된 방향으로 이탈해야 한다."""
    rule = CAUSE_RULES[pattern]
    step = STEPS_BY_ID[rule.step_id]

    df = sim.fdc_summary[sim.fdc_summary["step_id"] == rule.step_id].merge(
        sim.wafer_master[["wafer_id", "pattern_label"]], on="wafer_id"
    )
    case = df[df["pattern_label"] == pattern]
    ctrl = df[df["pattern_label"] == "none"]
    if len(case) < 10:
        pytest.skip(f"{pattern} 표본 부족 ({len(case)}장)")

    for pert in rule.perturbations:
        col = f"{pert.param}_mean"
        delta = (case[col].mean() - ctrl[col].mean()) / ctrl[col].std()

        spec = step.param(pert.param)
        if spec.kind == "counter":
            expected_sign = 1.0  # counter는 항상 '많이 썼다' 방향
        else:
            expected_sign = np.sign(pert.shift_sigma)

        assert np.sign(delta) == expected_sign, (
            f"{pattern}.{pert.param}: 이탈 방향이 반대 ({delta:+.2f}σ)"
        )
        assert abs(delta) > 0.5, (
            f"{pattern}.{pert.param}: 신호가 너무 약함 ({delta:+.2f}σ)"
        )


def test_defect_concentrates_on_root_chamber(sim):
    """불량이 원인 챔버에 집중되어야 커미널리티 분석(M3)이 성립한다."""
    pattern = "Edge-Ring"
    rule = CAUSE_RULES[pattern]

    df = sim.fdc_summary[sim.fdc_summary["step_id"] == rule.step_id].merge(
        sim.wafer_master[["wafer_id", "pattern_label"]], on="wafer_id"
    )
    rate = df.groupby("chamber_id")["pattern_label"].apply(lambda s: (s == pattern).mean())

    assert rate.max() > rate.median() * 3, "특정 챔버로의 집중이 관찰되지 않음"


def test_excursions_are_time_bounded(sim):
    """설비 원인이 있는 이상 사건은 시간 구간을 가져야 한다 (SPC 탐지 대상)."""
    equip_excursions = [e for e in sim.excursions if e.step_id is not None]
    assert equip_excursions, "설비 원인 이상 사건이 하나도 없음"
    for e in equip_excursions:
        assert e.t_start < e.t_end
        assert 0.0 < e.attack_rate <= 1.0
        assert e.chamber_id in STEPS_BY_ID[e.step_id].chamber_ids


# ── 난이도 (너무 쉬우면 안 된다) ─────────────────────────────────────────


def test_confounding_is_injected(sim):
    """교락 쌍이 실제로 주입되어야 CMH 층화 검정(M3)이 의미를 갖는다."""
    assert sim.affinities, "교락 쌍이 주입되지 않음"
    assert sim.ground_truth["is_confounded"].sum() > 0


def test_unexplained_and_false_positive_exist(sim):
    """위음성·위양성이 존재해야 원인 분석 정확도에 현실적 상한이 생긴다."""
    gt = sim.ground_truth
    n_defect = (gt["pattern_label"] != "none").sum()

    assert gt["is_unexplained"].sum() > 0, "위음성(신호 없는 불량)이 없음"
    assert gt["is_false_positive"].sum() > 0, "위양성(정상인데 이탈)이 없음"
    # 위음성은 불량 중 일부여야 한다 (전부이거나 0이면 설정이 잘못된 것)
    assert 0 < gt["is_unexplained"].sum() < n_defect


def test_unexplained_wafers_lack_signal(sim):
    """위음성으로 표시된 웨이퍼는 실제로 FDC 신호가 약해야 한다.

    왜 이 검사가 필요한가: 플래그만 세우고 값은 그대로 흔들어 놨다면, 정답지는
    '설명 불가'라는데 데이터에는 신호가 남아 있는 모순이 생긴다. 그러면 M4의
    검증 지표가 실제보다 나쁘게 나온다.
    """
    pattern = "Edge-Ring"
    rule = CAUSE_RULES[pattern]
    param = f"{rule.perturbations[1].param}_mean"  # chamber_pressure

    df = sim.fdc_summary[sim.fdc_summary["step_id"] == rule.step_id].merge(
        sim.ground_truth[["wafer_id", "pattern_label", "is_unexplained"]], on="wafer_id"
    )
    case = df[df["pattern_label"] == pattern]
    ctrl = df[df["pattern_label"] == "none"]
    if case["is_unexplained"].sum() < 3:
        pytest.skip("위음성 표본 부족")

    baseline = ctrl[param].mean()
    signal = case[~case["is_unexplained"]][param].mean()
    silent = case[case["is_unexplained"]][param].mean()

    assert abs(signal - baseline) > abs(silent - baseline), (
        "위음성 웨이퍼가 정상 웨이퍼보다 더 크게 이탈함"
    )


def test_distractor_step_has_no_true_cause(sim):
    """P045는 어떤 웨이퍼의 정답 원인 스텝으로도 기록되지 않아야 한다."""
    assert (sim.ground_truth["true_root_step"] == "P045").sum() == 0


def test_equipment_baseline_differs_between_chambers(sim):
    """정상 웨이퍼만 봐도 챔버 간 baseline 차이가 있어야 한다.

    왜: 이 차이가 있어야 분석이 '절대값이 규격을 벗어났는가'가 아니라
        '평소 대비 변했는가'를 봐야 하는 실제 문제가 된다.
    """
    df = sim.fdc_summary[sim.fdc_summary["step_id"] == "P020"].merge(
        sim.wafer_master[["wafer_id", "pattern_label"]], on="wafer_id"
    )
    normal = df[df["pattern_label"] == "none"]
    means = normal.groupby("chamber_id")["rf_power_mean"].mean()
    assert means.std() > 0, "챔버 간 baseline이 완전히 동일함"


# ── 정답지 ───────────────────────────────────────────────────────────────


def test_ground_truth_root_matches_cause_rules(sim):
    """정답지의 원인 스텝이 CAUSE_RULES와 일치해야 한다."""
    for _, row in sim.ground_truth.sample(200, random_state=0).iterrows():
        expected = CAUSE_RULES[row["pattern_label"]].step_id
        actual = row["true_root_step"]
        assert (actual == expected) or (expected is None and actual is None)


def test_normal_wafers_have_no_root_cause(sim):
    gt = sim.ground_truth
    normal = gt[gt["pattern_label"] == "none"]
    assert normal["true_root_step"].isna().all()
    assert normal["severity"].isna().all()


def test_root_equipment_matches_actual_routing(sim):
    """정답 원인 설비가 그 웨이퍼의 실제 라우팅과 일치해야 한다."""
    gt = sim.ground_truth[sim.ground_truth["true_root_equip"].notna()]
    if not len(gt):
        pytest.skip("원인 설비가 기록된 웨이퍼 없음")

    merged = gt.merge(sim.fdc_summary, left_on=["wafer_id", "true_root_step"],
                      right_on=["wafer_id", "step_id"])
    assert (merged["true_root_equip"] == merged["chamber_id"]).all(), (
        "정답 원인 설비가 실제 통과 챔버와 다름"
    )


def test_simulation_is_reproducible():
    a = simulate(n_wafers=300, n_days=30, seed=99)
    b = simulate(n_wafers=300, n_days=30, seed=99)
    assert a.wafer_master["pattern_label"].tolist() == b.wafer_master["pattern_label"].tolist()
    np.testing.assert_allclose(
        a.fdc_summary["rf_power_mean"].to_numpy(),
        b.fdc_summary["rf_power_mean"].to_numpy(),
    )
