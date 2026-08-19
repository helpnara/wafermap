"""시계열 파생 피처 검증 (M5.5-②).

핵심 질문:
  1. **순간 스파이크를 잡는가?** — 평균은 거의 안 움직이는데 피처는 반응하는가
  2. **추세와 스파이크를 구분하는가?** — 조치가 완전히 다르다
  3. **정상 웨이퍼에서 오탐이 없는가?** — 노이즈를 이상이라 부르면 못 쓴다
"""

from __future__ import annotations

import numpy as np
import pytest

from wafermap.features.trace_feat import (
    FEATURE_SUFFIXES,
    STEADY_START_FRACTION,
    is_trace_feature,
    trace_features,
)

SIGMA = 0.6
NOISE = 0.3
N_POINTS = 60
T = np.linspace(0.0, 60.0, N_POINTS)


def _flat(n: int, rng: np.random.Generator, base: float = 65.0) -> np.ndarray:
    return base + rng.normal(0.0, NOISE, (n, N_POINTS))


def _with_spike(
    n: int, rng: np.random.Generator, height: float = 7.0, width: int = 3
) -> np.ndarray:
    values = _flat(n, rng)
    for i in range(n):
        start = int(rng.integers(N_POINTS // 3, N_POINTS - width))
        values[i, start:start + width] += height * SIGMA
    return values


def _with_drift(n: int, rng: np.random.Generator, span: float = 8.6) -> np.ndarray:
    """서서히 밀리는 시계열.

    **정상 구간의 평균**이 0이 되도록 맞춘다. 전체 60초를 기준으로 중심을 잡으면
    피처가 보는 구간(뒤쪽 60%)에서는 평균이 +1.2σ 올라가 버려, "평균은 같은데
    모양만 다르다"는 실험 전제가 깨진다.
    """
    ramp = np.linspace(-span * SIGMA / 2, span * SIGMA / 2, N_POINTS)
    ramp = ramp - ramp[int(N_POINTS * STEADY_START_FRACTION):].mean()
    return _flat(n, rng) + ramp[None, :]


# ── 기본 동작 ────────────────────────────────────────────────────────────


def test_returns_all_declared_features(rng):
    out = trace_features(_flat(5, rng), T, sigma=SIGMA)
    assert set(out) == set(FEATURE_SUFFIXES)
    for values in out.values():
        assert values.shape == (5,)


def test_rejects_too_few_points(rng):
    with pytest.raises(ValueError, match="시점이 너무 적습니다"):
        trace_features(np.zeros((3, 2)), np.array([0.0, 1.0]), sigma=SIGMA)


def test_rejects_length_mismatch(rng):
    with pytest.raises(ValueError, match="시각 길이"):
        trace_features(_flat(2, rng), T[:10], sigma=SIGMA)


def test_accepts_single_wafer(rng):
    out = trace_features(_flat(1, rng)[0], T, sigma=SIGMA)
    assert out["_time_above"].shape == (1,)


def test_is_trace_feature_labels_columns():
    assert is_trace_feature("bath_temp_time_above")
    assert is_trace_feature("bath_temp_slope")
    assert not is_trace_feature("bath_temp_mean")
    assert not is_trace_feature("bath_temp_std")


# ── 정상 웨이퍼에서 오탐이 없어야 한다 ★ ─────────────────────────────────


def test_no_false_alarm_on_flat_trace(rng):
    """노이즈만 있는 시계열에서 임계 초과가 나오면 안 된다.

    왜 중요한가: `time_above`가 정상에서도 계속 0이 아니면 이상 신호와 구분되지
        않는다. 실제로 detrend 이전 구현에서는 안정화 구간의 완만한 상승 때문에
        정상 웨이퍼도 6초씩 임계를 넘었다.
    """
    out = trace_features(_flat(50, rng), T, sigma=SIGMA)
    assert out["_time_above"].mean() < 0.5
    assert out["_n_excursions"].mean() < 0.3


def test_slope_near_zero_for_flat_trace(rng):
    out = trace_features(_flat(50, rng), T, sigma=SIGMA)
    assert abs(out["_slope"].mean()) < 0.01


# ── 스파이크 검출 ────────────────────────────────────────────────────────


def test_spike_barely_moves_the_mean(rng):
    """3초 스파이크는 60초 평균을 거의 못 움직인다 ★★ — 이 프로젝트의 전제.

    첨부 도메인 문서 §12의 주장을 수치로 확인한다.
    """
    normal, spiked = _flat(60, rng), _with_spike(60, rng)
    shift = (spiked.mean() - normal.mean()) / SIGMA
    assert shift < 0.6, f"평균이 {shift:.2f}σ나 움직였다 — 전제가 성립하지 않는다"


def test_time_above_detects_spike(rng):
    """`time_above`는 스파이크에서 뚜렷이 반응해야 한다."""
    normal = trace_features(_flat(60, rng), T, sigma=SIGMA)
    spiked = trace_features(_with_spike(60, rng), T, sigma=SIGMA)
    assert spiked["_time_above"].mean() > normal["_time_above"].mean() + 1.0
    assert spiked["_n_excursions"].mean() > 0.7


def test_peak_dev_scales_with_spike_height(rng):
    small = trace_features(_with_spike(40, rng, height=4.0), T, sigma=SIGMA)
    large = trace_features(_with_spike(40, rng, height=10.0), T, sigma=SIGMA)
    assert large["_peak_dev"].mean() > small["_peak_dev"].mean()


def test_time_above_scales_with_spike_width(rng):
    narrow = trace_features(_with_spike(40, rng, width=2), T, sigma=SIGMA)
    wide = trace_features(_with_spike(40, rng, width=8), T, sigma=SIGMA)
    assert wide["_time_above"].mean() > narrow["_time_above"].mean() * 1.5


# ── 스파이크 vs 드리프트 ★★ ─────────────────────────────────────────────


def test_spike_and_drift_look_alike_in_summary_stats(rng):
    """두 기전은 요약통계에서 비슷해 보인다 — 이것이 시계열 피처가 필요한 이유.

    왜 이 테스트가 중요한가: 이 전제가 깨지면(예: 산포가 크게 달라지면) 뒤따르는
        "시계열 피처라야 구분된다"는 주장이 근거를 잃는다. 실험 설계 자체를
        지키는 테스트다.
    """
    spike, drift = _with_spike(80, rng), _with_drift(80, rng)
    for values in (spike, drift):
        assert values.shape == (80, N_POINTS)

    steady = slice(int(N_POINTS * 0.4), N_POINTS)
    std_gap = abs(spike[:, steady].std(axis=1).mean() - drift[:, steady].std(axis=1).mean())
    mean_gap = abs(spike[:, steady].mean() - drift[:, steady].mean())
    assert std_gap / SIGMA < 0.5, f"산포가 {std_gap / SIGMA:.2f}σ나 다르다"
    assert mean_gap / SIGMA < 1.0, f"평균이 {mean_gap / SIGMA:.2f}σ나 다르다"


def test_slope_separates_drift_from_spike(rng):
    """기울기는 드리프트에서만 커야 한다."""
    spike = trace_features(_with_spike(80, rng), T, sigma=SIGMA)
    drift = trace_features(_with_drift(80, rng), T, sigma=SIGMA)
    assert abs(drift["_slope"].mean()) > abs(spike["_slope"].mean()) * 5


def test_time_above_separates_spike_from_drift(rng):
    """임계 초과 시간은 스파이크에서만 커야 한다 ★.

    드리프트는 추세를 뺀 잔차가 작으므로 `time_above`가 거의 0이어야 한다.
    이 대비가 두 기전을 갈라내는 핵심이다.
    """
    spike = trace_features(_with_spike(80, rng), T, sigma=SIGMA)
    drift = trace_features(_with_drift(80, rng), T, sigma=SIGMA)
    assert spike["_time_above"].mean() > 1.0
    assert drift["_time_above"].mean() < 0.5


def test_detrending_is_what_makes_the_separation_work(rng):
    """추세 제거가 없으면 드리프트도 임계를 넘는다 ★★.

    왜 못 박나: `deviation`을 중앙값 기준으로 되돌리면 드리프트의 상승분이
        그대로 이탈로 잡혀 `time_above`가 커진다. 그러면 스파이크와 구분되지
        않는다. 이 테스트는 detrend를 빼는 리팩터링을 막는다.
    """
    drift = _with_drift(60, rng)
    steady = slice(int(N_POINTS * 0.4), N_POINTS)
    body = drift[:, steady]

    naive_over = np.abs(body - np.median(body, axis=1, keepdims=True)) > 3.0 * SIGMA
    detrended = trace_features(drift, T, sigma=SIGMA)["_time_above"]

    assert naive_over.sum(axis=1).mean() > 0.5, "실험 전제: 추세 제거 없으면 임계를 넘는다"
    assert detrended.mean() < 0.2, "추세를 뺐는데도 임계를 넘는다"
