"""공정 간 교호작용 분석 검증 (M5.5-③).

핵심 질문:
  1. **주효과가 0인 교호작용을 만들어 낼 수 있는가?** — 없으면 실험 자체가 성립 안 한다
  2. **2×2 표가 그것을 잡는가?**
  3. **설비 조합을 찾는가?** — "B가 문제"가 아니라 "A→B 조합"
  4. **다중검정을 보정하는가?** — 쌍이 제곱으로 늘어나면 가짜가 반드시 나온다
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wafermap.analysis import interaction
from wafermap.config import INTERACTION_CAUSE_RULES, STEPS_BY_ID


@pytest.fixture
def pure_interaction():
    """주효과가 정확히 0이고 교호작용만 있는 인공 데이터.

    같은 방향으로 치우친 웨이퍼만 불량이 된다(둘 다 높거나 둘 다 낮거나).
    그러면 A의 주변부 평균은 불량군과 정상군이 같다 — 설계상 그렇게 만든다.
    """
    rng = np.random.default_rng(0)
    n = 1200
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    same_direction = (a > 0.8) & (b > 0.8) | (a < -0.8) & (b < -0.8)
    is_case = same_direction & (rng.random(n) < 0.8)

    matrix = pd.DataFrame(
        {"P030.alpha": a, "P050.beta": b, "P030.noise": rng.normal(0, 1, n)},
        index=[f"W{i:04d}" for i in range(n)],
    )
    return matrix, is_case


# ── 설정 ─────────────────────────────────────────────────────────────────


def test_interaction_shift_stays_within_spec():
    """주입 크기가 규격 안이어야 한다 ★.

    왜 못 박나: 규격을 벗어나면 단변량 규격 검사로 잡히고, 그러면
        "각각은 정상인데 조합이 문제"라는 이 마일스톤의 전제가 무너진다.
    """
    for rule in INTERACTION_CAUSE_RULES.values():
        for step, param in ((rule.step_a, rule.param_a), (rule.step_b, rule.param_b)):
            spec = STEPS_BY_ID[step].param(param)
            headroom = min(
                (spec.nominal - spec.spec_lo) / spec.sigma,
                (spec.spec_hi - spec.nominal) / spec.sigma,
            )
            assert rule.shift_range[1] < headroom, (
                f"{step}.{param}: 주입 {rule.shift_range[1]}σ가 규격 폭 {headroom:.1f}σ를 넘는다"
            )


def test_interaction_spans_two_steps():
    """교호작용은 **서로 다른 스텝** 사이여야 한다 (공정 간 상호작용)."""
    for rule in INTERACTION_CAUSE_RULES.values():
        assert rule.step_a != rule.step_b


# ── 2×2 분할표 ───────────────────────────────────────────────────────────


def test_pair_table_finds_pure_interaction(pure_interaction):
    """주효과 0, 교호작용만 있는 경우를 잡아야 한다."""
    matrix, is_case = pure_interaction
    result = interaction.pair_table(matrix, is_case, "P030.alpha", "P050.beta")

    assert result is not None
    assert abs(result.contrast) > 0.1, "교호작용 대비가 너무 작다"
    assert abs(result.main_a) < abs(result.contrast) / 2
    assert abs(result.main_b) < abs(result.contrast) / 2
    assert result.hidden_from_main_effects


def test_diagonal_cells_are_the_high_ones(pure_interaction):
    """같은 방향 조합(대각선)만 불량률이 높아야 한다."""
    matrix, is_case = pure_interaction
    result = interaction.pair_table(matrix, is_case, "P030.alpha", "P050.beta")
    assert min(result.rate_hh, result.rate_ll) > max(result.rate_hl, result.rate_lh)


def test_pair_table_rejects_thin_cells(pure_interaction):
    """칸 표본이 적으면 계산하지 않아야 한다 (0% 아니면 100%로 튄다)."""
    matrix, is_case = pure_interaction
    assert interaction.pair_table(
        matrix.head(10), is_case[:10], "P030.alpha", "P050.beta", min_cell=8
    ) is None


def test_unrelated_pair_has_small_contrast(pure_interaction):
    """무관한 파라미터 쌍은 대비가 작아야 한다."""
    matrix, is_case = pure_interaction
    result = interaction.pair_table(matrix, is_case, "P030.noise", "P050.beta")
    assert result is not None
    assert abs(result.contrast) < 0.1


def test_min_contrast_must_suit_the_defect_rate():
    """대비 문턱이 불량률보다 크면 진짜 교호작용도 걸러진다 ★.

    왜 못 박나: 처음에 문턱을 5%p로 잡았더니 불량률 0.87% 데이터에서 대비
        1.9%p인 진짜 교호작용이 통째로 사라졌다. 기본값이 다시 커지면
        같은 실수가 반복된다.
    """
    assert interaction.MIN_CONTRAST <= 0.01


def test_table_renders_four_cells(pure_interaction):
    matrix, is_case = pure_interaction
    result = interaction.pair_table(matrix, is_case, "P030.alpha", "P050.beta")
    rendered = result.table()
    assert rendered.count("%") >= 4
    assert "alpha" in rendered and "beta" in rendered


# ── 쌍 탐색 ──────────────────────────────────────────────────────────────


def test_screen_pairs_ranks_true_pair_first(pure_interaction):
    matrix, is_case = pure_interaction
    results = interaction.screen_pairs(matrix, is_case, cross_step_only=True)
    assert results
    top = results[0]
    assert {top.param_a, top.param_b} == {"P030.alpha", "P050.beta"}


def test_screen_pairs_applies_multiple_testing_correction(pure_interaction):
    """보정된 p값이 원래 p값보다 크거나 같아야 한다 ★.

    왜 필요한가: 파라미터 40개면 쌍이 780개다. 보정 없이 5% 유의수준을 쓰면
        신호가 전혀 없어도 39개가 유의하게 나온다.
    """
    matrix, is_case = pure_interaction
    results = interaction.screen_pairs(matrix, is_case, cross_step_only=False)
    assert results
    for result in results:
        assert result.p_adj >= result.p_value - 1e-12


def test_cross_step_only_excludes_same_step_pairs(pure_interaction):
    matrix, is_case = pure_interaction
    results = interaction.screen_pairs(matrix, is_case, cross_step_only=True)
    for result in results:
        assert result.param_a.split(".")[0] != result.param_b.split(".")[0]


# ── 넓은 행렬 ────────────────────────────────────────────────────────────


def test_cross_step_matrix_prefixes_step_id(sim):
    """스텝이 다르면 같은 이름의 파라미터가 있으므로 접두가 필요하다."""
    ids = list(sim.wafer_master["wafer_id"].head(300))
    matrix = interaction.cross_step_matrix(sim.fdc_summary, ids, steps=("P020", "P030"))

    assert not matrix.empty
    assert all("." in c for c in matrix.columns)
    # chamber_pressure는 식각·증착 양쪽에 있다 — 접두 덕분에 충돌하지 않는다
    pressure = [c for c in matrix.columns if c.endswith("chamber_pressure")]
    assert len(pressure) == 2


def test_cross_step_matrix_drops_measurements(sim):
    """계측값은 조작할 수 없어 조합 조치로 이어지지 않는다."""
    ids = list(sim.wafer_master["wafer_id"].head(300))
    matrix = interaction.cross_step_matrix(
        sim.fdc_summary, ids, steps=("P030",), controllable_only=True
    )
    assert not any(c.endswith("thickness_mean") for c in matrix.columns)


# ── 설비 조합 ★ ─────────────────────────────────────────────────────────


def test_equipment_pairs_finds_injected_combination(sim):
    """심어 둔 앞뒤 챔버 조합이 1위여야 한다 ★.

    첨부 문서 §15 예시 3: "Equipment B가 문제다"가 아니라
    "Equipment A → Equipment B 조합에서 문제가 증가한다"일 수 있다.
    """
    rule = INTERACTION_CAUSE_RULES["Center"]
    gt = sim.ground_truth
    case = list(gt.loc[gt["cause_mechanism"] == "interaction", "wafer_id"])
    if len(case) < 15:
        pytest.skip("축소 시뮬레이션에 교호작용 표본이 부족")

    results = interaction.equipment_pairs(
        sim.fdc_summary, case, list(gt["wafer_id"]),
        step_a=rule.step_a, step_b=rule.step_b, min_wafers=10,
    )
    assert results
    assert (results[0].chamber_a, results[0].chamber_b) == rule.equip_pair
    assert results[0].excess > 0.05


def test_equipment_pairs_excess_beats_solo(sim):
    """조합의 불량률이 각각 단독보다 높아야 '조합 문제'라고 부를 수 있다."""
    rule = INTERACTION_CAUSE_RULES["Center"]
    gt = sim.ground_truth
    case = list(gt.loc[gt["cause_mechanism"] == "interaction", "wafer_id"])
    if len(case) < 15:
        pytest.skip("축소 시뮬레이션에 교호작용 표본이 부족")

    top = interaction.equipment_pairs(
        sim.fdc_summary, case, list(gt["wafer_id"]),
        step_a=rule.step_a, step_b=rule.step_b, min_wafers=10,
    )[0]
    assert top.rate_pair > top.rate_a_only
    assert top.rate_pair > top.rate_b_only


def test_equipment_pairs_returns_empty_for_missing_step(sim):
    results = interaction.equipment_pairs(
        sim.fdc_summary, [], list(sim.ground_truth["wafer_id"]),
        step_a="P010", step_b="NOPE",
    )
    assert results == []


# ── 시뮬레이터 산출물 ────────────────────────────────────────────────────


def test_interaction_marginals_are_nearly_unshifted(sim):
    """교호작용군의 파라미터 **평균**이 정상군과 크게 다르지 않아야 한다 ★★.

    이것이 이 마일스톤의 전제다. 평균이 밀려 있으면 단변량 분석으로도 잡히고,
    "주효과로는 안 보인다"는 주장이 성립하지 않는다.
    """
    rule = INTERACTION_CAUSE_RULES["Center"]
    gt = sim.ground_truth
    case = list(gt.loc[gt["cause_mechanism"] == "interaction", "wafer_id"])
    if len(case) < 15:
        pytest.skip("축소 시뮬레이션에 교호작용 표본이 부족")
    control = list(gt.loc[gt["pattern_label"] == "none", "wafer_id"])

    step = sim.fdc_summary[sim.fdc_summary["step_id"] == rule.step_a]
    values = step.set_index("wafer_id")[f"{rule.param_a}_mean"]
    spec = STEPS_BY_ID[rule.step_a].param(rule.param_a)
    gap = abs(
        values[case].mean() - values[[w for w in control if w in values.index]].mean()
    ) / spec.sigma
    assert gap < 1.0, f"평균이 {gap:.2f}σ 밀려 있다 — 주효과로 보인다"
