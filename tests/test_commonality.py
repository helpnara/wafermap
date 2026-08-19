"""커미널리티 분석 검증.

핵심 질문 세 가지:
  1. **진범을 찾아내는가** — 인공적으로 심은 원인 챔버가 1위로 나오는가?
  2. **표본 단위가 올바른가** — lot 단위가 웨이퍼 단위보다 보수적인가?
  3. **다중검정을 보정하는가** — 원인이 없는 데이터에서 침묵하는가?

3번이 특히 중요하다. 챔버 37개를 검정하면 원인이 없어도 우연히 유의한 것이
나온다. 보정이 제대로 걸리지 않으면 멀쩡한 장비를 세우게 된다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wafermap.analysis import commonality as cm


def _make_fab(
    n_lots: int = 200,
    wafers_per_lot: int = 25,
    n_chambers: int = 6,
    culprit: str | None = "CH-2",
    attack_rate: float = 0.8,
    base_rate: float = 0.02,
    seed: int = 0,
):
    """인공 팹 데이터를 만든다.

    culprit 챔버를 거친 lot은 attack_rate 확률로 불량이 된다.
    Returns:
        (wafer_master, fdc_summary, case_ids, control_ids)
    """
    rng = np.random.default_rng(seed)
    chambers = [f"CH-{i}" for i in range(n_chambers)]

    wafers, fdc = [], []
    case_ids, control_ids = [], []

    for lot_i in range(n_lots):
        lot_id = f"LOT{lot_i:04d}"
        # 두 스텝: P010(원인 후보), P020(무관)
        ch_a = str(rng.choice(chambers))
        ch_b = str(rng.choice(chambers))

        rate = attack_rate if (culprit and ch_a == culprit) else base_rate
        lot_is_bad = rng.random() < rate

        for w in range(wafers_per_lot):
            wafer_id = f"{lot_id}-W{w:02d}"
            is_defect = lot_is_bad and rng.random() < 0.9
            wafers.append(
                {
                    "wafer_id": wafer_id,
                    "lot_id": lot_id,
                    "pattern_label": "Bad" if is_defect else "none",
                }
            )
            (case_ids if is_defect else control_ids).append(wafer_id)
            for step, ch in (("P010", ch_a), ("P020", ch_b)):
                fdc.append(
                    {
                        "wafer_id": wafer_id,
                        "step_id": step,
                        "chamber_id": f"{step}/{ch}",
                        "equip_id": ch,
                    }
                )

    return pd.DataFrame(wafers), pd.DataFrame(fdc), case_ids, control_ids


# ── 진범 탐지 ────────────────────────────────────────────────────────────


def test_finds_the_culprit_chamber():
    """인공적으로 심은 원인 챔버가 1위로 나와야 한다."""
    wm, fdc, case, control = _make_fab(culprit="CH-2", seed=1)
    ranking = cm.analyze(fdc, case, control, wafer_master=wm, unit="lot")

    assert not ranking.empty
    assert ranking.iloc[0]["chamber_id"] == "P010/CH-2", (
        f"진범을 못 찾음. 1위={ranking.iloc[0]['chamber_id']}"
    )
    assert ranking.iloc[0]["odds_ratio"] > 2.0
    assert ranking.iloc[0]["significant"]


def test_innocent_step_ranks_lower():
    """원인과 무관한 스텝(P020)의 챔버는 상위에 오면 안 된다."""
    wm, fdc, case, control = _make_fab(culprit="CH-2", seed=3)
    ranking = cm.analyze(fdc, case, control, wafer_master=wm, unit="lot")

    top = ranking.iloc[0]
    assert top["step_id"] == "P010", f"무관한 스텝이 1위: {top['step_id']}"


def test_silent_when_no_cause_exists():
    """원인이 없으면 유의 판정이 거의 없어야 한다 ★.

    왜 중요한가: 챔버가 많으면 우연히 유의해 보이는 것이 반드시 나온다.
        BH-FDR 보정이 제대로 걸리지 않으면 멀쩡한 장비를 범인으로 지목한다.
    """
    wm, fdc, case, control = _make_fab(culprit=None, base_rate=0.1, seed=7)
    if len(case) < 50:
        pytest.skip("이상군 표본 부족")

    ranking = cm.analyze(fdc, case, control, wafer_master=wm, unit="lot")
    n_significant = int(ranking["significant"].sum())
    assert n_significant <= 1, (
        f"원인이 없는데 {n_significant}개 챔버가 유의 판정 — 다중검정 보정 실패"
    )


def test_bh_correction_is_applied():
    """보정된 p_adj가 원래 p_value보다 크거나 같아야 한다."""
    wm, fdc, case, control = _make_fab(seed=2)
    ranking = cm.analyze(fdc, case, control, wafer_master=wm, unit="lot")

    assert (ranking["p_adj"] >= ranking["p_value"] - 1e-12).all(), (
        "p_adj가 p_value보다 작다 — 보정이 잘못됨"
    )
    assert (ranking["p_adj"] <= 1.0).all()


def test_significant_flag_uses_adjusted_p():
    """유의 판정은 반드시 보정된 p를 기준으로 해야 한다."""
    wm, fdc, case, control = _make_fab(seed=4)
    ranking = cm.analyze(fdc, case, control, wafer_master=wm, unit="lot")

    for _, row in ranking.iterrows():
        assert row["significant"] == (row["p_adj"] < cm.DEFAULT_ALPHA)


# ── 표본 단위 ★ ─────────────────────────────────────────────────────────


def test_lot_unit_reduces_sample_size():
    """lot 단위는 웨이퍼 단위보다 표본이 훨씬 작아야 한다."""
    wm, fdc, case, control = _make_fab(seed=5)
    by_wafer = cm.analyze(fdc, case, control, wafer_master=wm, unit="wafer")
    by_lot = cm.analyze(fdc, case, control, wafer_master=wm, unit="lot")

    assert by_lot.iloc[0]["case_total"] < by_wafer.iloc[0]["case_total"] / 5, (
        "lot 단위인데 표본이 거의 줄지 않았다"
    )


def test_lot_unit_is_more_conservative():
    """lot 단위의 p-value가 웨이퍼 단위보다 커야 한다 (덜 확신).

    왜: 웨이퍼 단위는 같은 lot의 25장을 독립 관측으로 세어 표본을 부풀린다.
        그 결과 사소한 차이도 압도적으로 유의해 보인다.
    """
    wm, fdc, case, control = _make_fab(seed=6)
    by_wafer = cm.analyze(fdc, case, control, wafer_master=wm, unit="wafer")
    by_lot = cm.analyze(fdc, case, control, wafer_master=wm, unit="lot")

    assert by_lot.iloc[0]["p_adj"] > by_wafer.iloc[0]["p_adj"], (
        "lot 단위가 웨이퍼 단위보다 확신이 강하다 — 표본 축소가 반영되지 않음"
    )


def test_lot_unit_requires_wafer_master():
    wm, fdc, case, control = _make_fab(seed=8)
    with pytest.raises(ValueError, match="wafer_master"):
        cm.analyze(fdc, case, control, unit="lot")


def test_rejects_invalid_unit():
    wm, fdc, case, control = _make_fab(seed=9)
    with pytest.raises(ValueError, match="unit"):
        cm.analyze(fdc, case, control, wafer_master=wm, unit="bogus")


# ── 입력 검증 ────────────────────────────────────────────────────────────


def test_rejects_overlapping_groups():
    """같은 웨이퍼가 양쪽 집단에 들어가면 거부해야 한다."""
    wm, fdc, case, control = _make_fab(seed=10)
    with pytest.raises(ValueError, match="겹칩니다"):
        cm.analyze(fdc, case, list(control) + list(case[:5]), wafer_master=wm)


def test_rejects_empty_group():
    wm, fdc, case, control = _make_fab(seed=11)
    with pytest.raises(ValueError, match="비어 있지"):
        cm.analyze(fdc, [], control, wafer_master=wm)


# ── 오즈비 ───────────────────────────────────────────────────────────────


def test_odds_ratio_handles_zero_cells():
    """셀에 0이 있어도 오즈비가 발산하면 안 된다 (Haldane 보정)."""
    odds, low, high = cm._odds_ratio_ci(10, 0, 0, 10)
    assert np.isfinite(odds) and odds > 1
    assert np.isfinite(low) and np.isfinite(high)
    assert low < odds < high


def test_odds_ratio_one_when_no_difference():
    """두 집단이 같으면 오즈비가 1 근처여야 한다."""
    odds, low, high = cm._odds_ratio_ci(50, 50, 50, 50)
    assert 0.9 < odds < 1.1
    assert low < 1 < high


def test_confidence_interval_ordering():
    odds, low, high = cm._odds_ratio_ci(30, 10, 10, 30)
    assert low < odds < high


# ── 층화 검정 ────────────────────────────────────────────────────────────


def test_stratified_test_keeps_true_cause():
    """진범은 다른 설비를 고정해도 효과가 남아야 한다."""
    wm, fdc, case, control = _make_fab(culprit="CH-2", seed=12)
    result = cm.stratified_test(fdc, case, control, "P010", "P010/CH-2", "P020")

    assert result.verdict in ("원인 유지", "판정 불가")
    if result.verdict == "원인 유지":
        assert result.adjusted_or > 1.5


def test_stratified_test_reports_insufficient_strata():
    """층이 부족하면 '판정 불가'를 명확히 알려야 한다."""
    wm, fdc, case, control = _make_fab(n_lots=12, n_chambers=2, seed=13)
    result = cm.stratified_test(
        fdc, case, control, "P010", "P010/CH-0", "P020", min_per_stratum=100
    )
    assert result.verdict == "판정 불가"
    assert result.n_strata < 2


def test_screen_confounders_returns_verdicts():
    wm, fdc, case, control = _make_fab(seed=14)
    ranking = cm.analyze(fdc, case, control, wafer_master=wm, unit="lot")
    if int(ranking["significant"].sum()) < 2:
        ranking = ranking.assign(significant=ranking.index < 3)

    screen = cm.screen_confounders(fdc, case, control, ranking, top_n=3)
    assert not screen.empty
    assert set(screen["verdict"]) <= {"원인 유지", "교락 의심", "부분 유지", "판정 불가"}


# ── lot 변환 ─────────────────────────────────────────────────────────────


def test_to_lot_groups_excludes_contaminated_control():
    """이상군 웨이퍼를 포함한 lot은 정상군에서 빠져야 한다."""
    wm = pd.DataFrame(
        {
            "wafer_id": ["A1", "A2", "B1", "B2"],
            "lot_id": ["LOT-A", "LOT-A", "LOT-B", "LOT-B"],
            "pattern_label": ["Bad", "none", "none", "none"],
        }
    )
    case_lots, control_lots = cm.to_lot_groups(wm, ["A1"], ["A2", "B1", "B2"])

    assert case_lots == ("LOT-A",)
    assert control_lots == ("LOT-B",), "이상군 웨이퍼가 섞인 lot이 정상군에 남았다"
