"""SPC 이상 탐지 검증.

핵심 질문: **관리도가 실제로 이상을 잡아내고, 정상에는 침묵하는가?**

성능 수치를 고정하지는 않는다(데이터가 바뀌면 깨진다). 대신 관리도가 성립하기
위한 구조적 성질을 검사한다 — 지표 성격에 맞는 관리도를 쓰는가, 표본 수를
반영하는가, 인공적으로 만든 이상을 잡는가.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wafermap.analysis import spc


def _make_master(
    n_days: int = 60,
    wafers_per_day: int = 50,
    defect_rate: float = 0.05,
    spike_days: tuple[int, int] | None = None,
    spike_rate: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    """인공 wafer_master를 만든다. spike_days 구간에만 불량률을 올린다."""
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2026-01-01")

    for day in range(n_days):
        rate = defect_rate
        if spike_days and spike_days[0] <= day <= spike_days[1]:
            rate = spike_rate
        for i in range(wafers_per_day):
            is_defect = rng.random() < rate
            rows.append(
                {
                    "wafer_id": f"W{day:03d}-{i:03d}",
                    "lot_id": f"LOT{day:03d}-{i // 25:01d}",
                    "eds_time": start + pd.Timedelta(days=day, hours=i % 24),
                    "yield_pct": 80.0 if is_defect else 97.0,
                    "pattern_label": "Edge-Ring" if is_defect else "none",
                }
            )
    return pd.DataFrame(rows)


# ── 집계 ─────────────────────────────────────────────────────────────────


def test_aggregate_returns_value_and_sample_size():
    """p-chart에 필요한 표본 수와 불량 수를 함께 돌려줘야 한다."""
    master = _make_master(n_days=30)
    out = spc.aggregate(master, metric="Edge-Ring_rate", freq="D")

    assert list(out.columns) == ["value", "n", "defects"]
    assert (out["n"] > 0).all()
    assert (out["defects"] <= out["n"]).all()
    # value는 백분율이어야 한다
    np.testing.assert_allclose(out["value"], out["defects"] / out["n"] * 100, rtol=1e-9)


def test_aggregate_drops_sparse_timepoints():
    """표본이 적은 시점은 버려야 한다 (관리한계가 왜곡되는 것을 막는다)."""
    master = _make_master(n_days=20, wafers_per_day=3)
    out = spc.aggregate(master, metric="yield_pct", freq="D", min_count=5)
    assert len(out) == 0, "표본 부족 시점이 걸러지지 않음"


def test_aggregate_rejects_unknown_metric():
    master = _make_master(n_days=20)
    with pytest.raises(ValueError, match="알 수 없는 지표"):
        spc.aggregate(master, metric="bogus")


# ── 관리도 유형 선택 ★ ──────────────────────────────────────────────────


def test_rate_metric_uses_p_chart_by_default():
    """비율 지표는 자동으로 p-chart가 선택되어야 한다."""
    master = _make_master(n_days=40)
    data = spc.aggregate(master, metric="Edge-Ring_rate")
    chart = spc.control_chart(data, metric="Edge-Ring_rate")
    assert chart.chart_type == "p"


def test_yield_metric_uses_individual_chart_by_default():
    master = _make_master(n_days=40)
    data = spc.aggregate(master, metric="yield_pct")
    chart = spc.control_chart(data, metric="yield_pct")
    assert chart.chart_type == "individual"


def test_p_chart_limits_vary_with_sample_size():
    """p-chart의 관리한계는 표본 수에 따라 달라야 한다.

    왜: 웨이퍼 10장으로 잰 불량률과 500장으로 잰 불량률은 신뢰도가 다르다.
        표본이 적은 시점은 한계가 넓어야 한다.
    """
    rng = np.random.default_rng(1)
    n = np.where(np.arange(40) < 20, 20, 400)  # 앞 20시점은 표본이 적다
    defects = rng.binomial(n, 0.05)
    data = pd.DataFrame(
        {"value": defects / n * 100, "n": n, "defects": defects},
        index=pd.date_range("2026-01-01", periods=40, freq="D"),
    )
    chart = spc.control_chart(data, metric="x_rate", chart_type="p")

    small_sigma = chart.sigma_series.iloc[:20].mean()
    large_sigma = chart.sigma_series.iloc[20:].mean()
    assert small_sigma > large_sigma * 2, "표본 수가 관리한계에 반영되지 않음"


def test_p_chart_requires_sample_counts():
    """표본 수 없이 p-chart를 요청하면 명확히 거부해야 한다."""
    series = pd.Series(
        np.random.rand(20) * 10, index=pd.date_range("2026-01-01", periods=20, freq="D")
    )
    with pytest.raises(ValueError, match="표본 수"):
        spc.control_chart(series, metric="x_rate", chart_type="p")


def test_overdispersion_detected_for_clustered_data():
    """lot 단위로 뭉친 데이터는 과분산 계수가 1보다 커야 한다.

    왜: 웨이퍼가 lot 단위로 함께 감염되면 실제 변동이 이항분포보다 크다.
        Laney 보정이 이를 잡아내지 못하면 관리한계가 너무 좁아 위양성이 폭증한다.
    """
    rng = np.random.default_rng(2)
    n = np.full(40, 100)
    # lot 단위 감염: 0 또는 50이 번갈아 — 이항분포보다 훨씬 큰 변동
    defects = np.where(rng.random(40) < 0.3, 50, 0)
    data = pd.DataFrame(
        {"value": defects / n * 100, "n": n, "defects": defects},
        index=pd.date_range("2026-01-01", periods=40, freq="D"),
    )
    chart = spc.control_chart(data, metric="x_rate", chart_type="p", laney=True)
    assert chart.overdispersion > 1.5, (
        f"과분산이 감지되지 않음 (계수 {chart.overdispersion:.2f})"
    )


# ── 검출 능력 ────────────────────────────────────────────────────────────


def test_detects_injected_spike():
    """인공적으로 만든 이상 구간을 잡아내야 한다."""
    master = _make_master(n_days=60, spike_days=(30, 40), spike_rate=0.4)
    data = spc.aggregate(master, metric="Edge-Ring_rate")
    chart = spc.control_chart(data, metric="Edge-Ring_rate")

    assert chart.alarms, "명백한 이상 구간을 검출하지 못함"
    spike_start = pd.Timestamp("2026-01-01") + pd.Timedelta(days=30)
    spike_end = pd.Timestamp("2026-01-01") + pd.Timedelta(days=41)
    assert any(
        not (a.end < spike_start or a.start > spike_end) for a in chart.alarms
    ), "검출된 구간이 실제 이상 구간과 겹치지 않음"


def test_stays_silent_on_stable_process():
    """안정된 공정에서는 경보가 거의 없어야 한다 (위양성 억제)."""
    master = _make_master(n_days=80, defect_rate=0.05, seed=5)
    data = spc.aggregate(master, metric="Edge-Ring_rate")
    chart = spc.control_chart(data, metric="Edge-Ring_rate")

    assert len(chart.alarms) <= 2, (
        f"안정 공정에서 경보가 {len(chart.alarms)}건 — 위양성이 너무 많다"
    )


def test_yield_drop_detected_downward():
    """수율 지표는 '떨어지는' 방향을 감시해야 한다."""
    master = _make_master(n_days=60, spike_days=(35, 45), spike_rate=0.6)
    data = spc.aggregate(master, metric="yield_pct")
    chart = spc.control_chart(data, metric="yield_pct")

    assert all(a.direction == "down" for a in chart.alarms), (
        "수율 관리도가 상승 방향 경보를 냈다"
    )


def test_short_series_is_rejected():
    master = _make_master(n_days=5)
    data = spc.aggregate(master, metric="yield_pct")
    with pytest.raises(ValueError, match="너무 짧"):
        spc.control_chart(data, metric="yield_pct")


def test_min_run_filters_single_point_noise():
    """min_run을 올리면 단발성 이탈이 걸러져야 한다."""
    master = _make_master(n_days=60, spike_days=(30, 30), spike_rate=0.8)
    data = spc.aggregate(master, metric="Edge-Ring_rate")

    loose = spc.control_chart(data, metric="Edge-Ring_rate", min_run=1)
    strict = spc.control_chart(data, metric="Edge-Ring_rate", min_run=5)
    assert len(strict.alarms) <= len(loose.alarms)


# ── 사이클 타임 보정 ★ ──────────────────────────────────────────────────


def test_cause_window_shifts_backward():
    """원인 시점은 검출 시점보다 **앞서야** 한다.

    왜 중요한가: 불량은 EDS에서 발견되지만 원인은 그보다 전에 발생했다.
        이 보정을 빼먹으면 엉뚱한 기간의 설비 이력을 뒤지게 된다.
    """
    alarm = spc.Alarm(
        alarm_id="ALM-001", method="EWMA",
        start=pd.Timestamp("2026-03-05"), end=pd.Timestamp("2026-03-10"),
        n_points=6, peak_deviation=3.2, direction="up", metric="x_rate",
    )
    cycle = pd.Timedelta(hours=66)
    cause_start, cause_end = spc.to_cause_window(alarm, cycle)

    assert cause_start < alarm.start
    assert alarm.start - cause_start == cycle
    assert alarm.end - cause_end == cycle


# ── 이상군 / 정상군 분할 ────────────────────────────────────────────────


def test_split_case_control_does_not_overlap():
    """두 집단이 겹치면 검정이 무의미해진다."""
    master = _make_master(n_days=60, spike_days=(30, 40), spike_rate=0.4)
    data = spc.aggregate(master, metric="Edge-Ring_rate")
    chart = spc.control_chart(data, metric="Edge-Ring_rate")
    if not chart.alarms:
        pytest.skip("경보가 검출되지 않음")

    split = spc.split_case_control(master, chart.alarms[0])
    assert not (set(split.case_ids) & set(split.control_ids))
    assert split.case_ids and split.control_ids


def test_split_control_precedes_case():
    """대조군은 이상 구간 **직전**이어야 한다 (시간 요인 상쇄)."""
    master = _make_master(n_days=60, spike_days=(30, 40), spike_rate=0.4)
    data = spc.aggregate(master, metric="Edge-Ring_rate")
    chart = spc.control_chart(data, metric="Edge-Ring_rate")
    if not chart.alarms:
        pytest.skip("경보가 검출되지 않음")

    split = spc.split_case_control(master, chart.alarms[0])
    assert split.control_window[1] <= split.case_window[0]


def test_split_by_pattern_selects_correct_labels():
    master = _make_master(n_days=40, defect_rate=0.2)
    split = spc.split_by_pattern(master, "Edge-Ring")

    labels = master.set_index("wafer_id")["pattern_label"]
    assert all(labels[w] == "Edge-Ring" for w in split.case_ids)
    assert all(labels[w] == "none" for w in split.control_ids)
