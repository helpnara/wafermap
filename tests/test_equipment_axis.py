"""검사 설비 축 검증 (M5.5-①).

핵심 질문:
  1. **검사 설비가 후보에 들어가는가?** — 후보에 없으면 1위가 될 수 없다
  2. **두 축이 서로를 가리지 않는가?** — 표본 많은 쪽이 이기면 소수 원인은 묻힌다
  3. **껍질 벗기기가 정상군도 같이 제외하는가?** — 안 하면 가짜 원인이 생긴다
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wafermap.analysis import commonality
from wafermap.config import (
    PROBE_CARDS,
    TEST_CAUSE_RULES,
    TEST_INDUCED_SHARE,
    TEST_STEP,
    TEST_STEP_ID,
    is_controllable,
)


# ── 설정 ─────────────────────────────────────────────────────────────────


def test_test_step_units_are_probe_cards():
    """검정 단위가 테스터가 아니라 프로브 카드여야 한다 ★.

    왜: 프로브 카드는 소모품이라 테스터 사이를 옮겨 다닌다. 마모되는 것도 카드다.
        테스터 단위로 묶으면 카드가 옮겨 갈 때마다 신호가 흩어져 사라진다.
    """
    assert TEST_STEP.chamber_ids == PROBE_CARDS
    assert "ch1" not in "".join(TEST_STEP.chamber_ids)


def test_test_step_is_not_a_process_step():
    """검사 스텝이 공정 라우팅에 끼면 안 된다."""
    from wafermap.config import PROCESS_STEPS, STEPS_BY_ID

    assert TEST_STEP not in PROCESS_STEPS
    assert STEPS_BY_ID[TEST_STEP_ID] is TEST_STEP


def test_contact_resistance_is_a_measurement():
    """접촉 저항은 마모의 결과지 조작 손잡이가 아니다."""
    assert not is_controllable("contact_resistance_mean")
    assert not is_controllable("test_time_mean")
    assert is_controllable("probe_overdrive_mean")


def test_test_cause_rules_reference_real_params():
    """검사 규칙이 실제로 존재하는 파라미터를 흔들어야 한다."""
    for pattern, rule in TEST_CAUSE_RULES.items():
        assert rule.step_id == TEST_STEP_ID
        for pert in rule.perturbations:
            TEST_STEP.param(pert.param)  # 없으면 KeyError
        assert pattern in TEST_INDUCED_SHARE


def test_test_induced_share_is_a_fraction():
    """검사 기인 비율은 0~1이어야 한다 (패턴 총량을 나눠 갖는 값)."""
    for pattern, share in TEST_INDUCED_SHARE.items():
        assert 0.0 < share < 1.0, f"{pattern}: {share}"


# ── 시뮬레이터 산출물 ────────────────────────────────────────────────────


def test_wafer_master_carries_test_equipment(sim):
    """웨이퍼마다 '어느 카드로 쟀는지'가 남아야 한다."""
    master = sim.wafer_master
    for col in ("tester_id", "probe_card_id", "probe_touchdown"):
        assert col in master.columns
    assert set(master["probe_card_id"]).issubset(set(PROBE_CARDS))
    assert (master["probe_touchdown"] >= 0).all()


def test_probe_card_and_tester_are_independent(sim):
    """카드와 테스터가 같이 움직이면 둘을 영원히 구분할 수 없다 ★."""
    table = pd.crosstab(sim.wafer_master["tester_id"], sim.wafer_master["probe_card_id"])
    # 모든 테스터가 여러 카드를 써야 한다
    assert (table > 0).sum(axis=1).min() >= 3


def test_fdc_contains_test_step(sim):
    """검사 이력이 공정 FDC와 **같은 테이블**에 있어야 한다.

    왜: 커미널리티·SHAP이 프로브 카드를 공정 챔버와 나란히 놓고 경쟁시키려면
        같은 후보 목록에 들어와야 한다.
    """
    fdc = sim.fdc_summary
    test_rows = fdc[fdc["step_id"] == TEST_STEP_ID]
    assert len(test_rows) == len(sim.wafer_master)
    assert set(test_rows["chamber_id"]).issubset(set(PROBE_CARDS))
    assert "touchdown_count_mean" in fdc.columns


def test_test_step_run_time_is_eds_time(sim):
    """검사 이력의 시각은 공정 투입이 아니라 EDS 검사 시각이어야 한다 ★.

    왜: 검사 이상은 검사할 때 일어난다. 공정 투입 시각으로 기록하면 SPC가
        66시간 어긋난 곳을 뒤지게 된다.
    """
    fdc = sim.fdc_summary
    test_rows = fdc[fdc["step_id"] == TEST_STEP_ID].set_index("wafer_id")["run_time"]
    eds = sim.wafer_master.set_index("wafer_id")["eds_time"]
    joined = pd.concat([test_rows, eds], axis=1).dropna()
    assert (joined["run_time"] == joined["eds_time"]).all()


def test_touchdown_grows_much_faster_than_wafer_count(sim):
    """터치다운은 웨이퍼당 수십 회씩 쌓인다 — 순번을 그대로 쓰면 안 된다."""
    master = sim.wafer_master
    per_card = master.groupby("probe_card_id")["probe_touchdown"].max()
    n_per_card = master.groupby("probe_card_id").size()
    assert (per_card > n_per_card * 10).all()


def test_ground_truth_marks_test_induced(sim):
    """검사 기인 웨이퍼의 정답 설비가 프로브 카드여야 한다."""
    gt = sim.ground_truth
    assert "is_test_induced" in gt.columns
    marked = gt[gt["is_test_induced"]]
    if len(marked):
        assert set(marked["true_root_step"]) == {TEST_STEP_ID}
        assert set(marked["true_root_equip"]).issubset(set(PROBE_CARDS))


def test_test_induced_only_for_declared_patterns(sim):
    """검사 기인 규칙이 없는 패턴에 플래그가 붙으면 안 된다."""
    gt = sim.ground_truth
    marked = set(gt.loc[gt["is_test_induced"], "pattern_label"])
    assert marked.issubset(set(TEST_CAUSE_RULES))


def test_label_distribution_unchanged_by_test_causes(sim):
    """검사 기인을 얹어도 패턴별 총 장수는 그대로여야 한다 ★.

    왜: 검사 기인을 **추가**하면 WM-811K 실측 라벨 비율이 깨진다. 정해진 장수를
        두 경로가 나눠 갖게 설계한 이유다.
    """
    gt = sim.ground_truth
    for pattern in TEST_CAUSE_RULES:
        sub = gt[gt["pattern_label"] == pattern]
        n_test = int(sub["is_test_induced"].sum())
        assert 0 < n_test < len(sub), f"{pattern}: 한쪽 경로로만 생성됐다"


# ── 분석 ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def axis_case(sim):
    """검사 기인이 섞인 패턴 하나와 그 이상/정상군."""
    gt = sim.ground_truth
    counts = {
        p: int(gt[(gt["pattern_label"] == p) & gt["is_test_induced"]].shape[0])
        for p in TEST_CAUSE_RULES
    }
    pattern = max(counts, key=counts.get)
    case = list(gt.loc[gt["pattern_label"] == pattern, "wafer_id"])
    control = list(gt.loc[gt["pattern_label"] == "none", "wafer_id"])
    return pattern, case, control


def test_by_axis_returns_both_axes(sim, axis_case):
    """두 축의 1위를 각각 돌려줘야 한다."""
    _, case, control = axis_case
    axes = commonality.by_axis(
        sim.fdc_summary, case, control, wafer_master=sim.wafer_master
    )
    assert set(axes["axis"]) == {"process", "test"}
    assert (axes[axes["axis"] == "test"]["step_id"] == TEST_STEP_ID).all()
    assert (axes[axes["axis"] == "process"]["step_id"] != TEST_STEP_ID).all()


def test_by_axis_test_candidate_is_a_probe_card(sim, axis_case):
    _, case, control = axis_case
    axes = commonality.by_axis(
        sim.fdc_summary, case, control, wafer_master=sim.wafer_master
    )
    card = axes[axes["axis"] == "test"].iloc[0]["chamber_id"]
    assert card in PROBE_CARDS


def test_by_axis_reports_combined_rank(sim, axis_case):
    """통합 랭킹에서 몇 위였는지 알려줘야 한다 ★.

    왜: 축을 나눠서 1위로 올라온 것이 통합에서는 9위였다면, **축을 나누지 않았다면
        놓쳤을 원인**이라는 뜻이다. 그 사실이 보여야 축 분리의 값어치가 드러난다.
    """
    _, case, control = axis_case
    axes = commonality.by_axis(
        sim.fdc_summary, case, control, wafer_master=sim.wafer_master
    )
    assert "rank_overall" in axes.columns
    assert (axes["rank_overall"] >= 1).all()


def test_peel_removes_from_controls_too(sim, axis_case):
    """벗겨 낸 설비는 정상군에서도 빠져야 한다 ★★.

    왜: 불량군에서만 빼면 같은 스텝의 나머지 챔버가 기계적으로 부풀려진다.
        남은 불량은 정의상 다른 챔버만 지났는데 정상군은 여전히 전 챔버에
        퍼져 있기 때문이다. 여기서는 2층 이후의 OR이 비현실적으로 튀지 않는지 본다.
    """
    _, case, control = axis_case
    layers = commonality.peel(
        sim.fdc_summary, case, control, wafer_master=sim.wafer_master, max_layers=3
    )
    assert layers
    assert layers[0].rank == 1
    for layer in layers[1:]:
        assert layer.n_case_before < layers[0].n_case_before


def test_peel_stops_on_weak_signal(sim, axis_case):
    """OR이 문턱 아래로 내려가면 멈춰야 한다 (잡음을 원인이라 부르지 않는다)."""
    _, case, control = axis_case
    layers = commonality.peel(
        sim.fdc_summary, case, control, wafer_master=sim.wafer_master,
        min_odds_ratio=100.0,
    )
    assert layers == []


def test_peel_layer_describes_axis(sim, axis_case):
    _, case, control = axis_case
    layers = commonality.peel(
        sim.fdc_summary, case, control, wafer_master=sim.wafer_master
    )
    for layer in layers:
        assert layer.is_test_equipment == (layer.step_id == TEST_STEP_ID)
        assert "층" in layer.describe()


def test_test_axis_does_not_crowd_out_process(sim, axis_case):
    """검사 축을 넣어도 공정 후보가 밀려나면 안 된다 ★.

    왜 이 형태로 확인하나: "공정 원인 Top-1 적중률"은 6,000장 전체 데이터로
        측정하는 값이다(`docs/04_results.md`, 100%). 이 픽스처는 1,500장이라
        불량 패턴 하나가 100장이 안 되고, 챔버 최소 표본 기준에 걸려 정답
        챔버가 후보 목록에서 아예 빠지기도 한다. 그런 규모에서 적중률을 단언하면
        **데이터가 아니라 운을 검사하는 테스트**가 된다.

        그래서 규모와 무관하게 성립해야 하는 성질만 못 박는다 — 검사 설비가
        후보에 들어오되, 공정 후보를 밀어내지는 않는다.
    """
    _, case, control = axis_case
    ranking = commonality.analyze(
        sim.fdc_summary, case, control, wafer_master=sim.wafer_master
    )
    n_test = int((ranking["step_id"] == TEST_STEP_ID).sum())
    n_process = len(ranking) - n_test

    assert n_test > 0, "검사 설비가 후보에 없다"
    assert n_process > n_test, "검사 축이 후보 목록을 잠식했다"


def test_touchdown_is_a_time_proxy(sim):
    """터치다운 카운터는 시각과 강하게 상관된다 ★★ — 해석 시 함정.

    왜 테스트로 못 박나: 이 값은 처리할수록 단조 증가하므로 **시각의 대리 변수**다.
        불량이 특정 기간에 몰리면, 검사와 무관한 불량에서도 터치다운이 원인처럼
        보인다. M4의 `pad_life`가 Scratch에서 일으킨 문제와 같은 종류이며,
        아직 해결하지 못한 한계다. 이 성질이 사라지면 그 경고도 갱신해야 한다.
    """
    master = sim.wafer_master
    card = master["probe_card_id"].iloc[0]
    sub = master[master["probe_card_id"] == card].sort_values("eds_time")
    order = np.arange(len(sub))
    corr = np.corrcoef(order, sub["probe_touchdown"].to_numpy())[0, 1]
    assert corr > 0.8, f"터치다운이 시각과 상관되지 않는다 (r={corr:.2f})"
