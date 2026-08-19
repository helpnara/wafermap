"""센서 시계열(trace)에서 요약통계가 놓치는 피처를 뽑는다 — M5.5-②.

무엇을: 웨이퍼 1장을 처리하는 동안의 파라미터 시계열에서 기울기·피크·임계 초과 시간
        같은 **모양 피처**를 계산한다.
어떻게: (웨이퍼 × 시점) 행렬을 통째로 받아 벡터 연산으로 처리한다.
왜:     mean/std/min/max는 **언제 어떻게 변했는지**를 버린다. 60초 중 3초만 튄
        이상은 평균을 0.35σ밖에 못 움직여 요약통계 분석에서 구조적으로 사라진다.

첨부 도메인 문서 §12의 예가 정확히 이 문제다.

    정상 Temperature = 500℃
      0~30 sec   500℃
     30~31 sec   540℃      ← 이 1초가 품질을 망쳤는데
     31~60 sec   500℃
    → 평균 500.7℃. 관리한계 안이다.

왜 min/max로는 부족한가 ★: `max`는 스파이크를 잡을 것 같지만 **노이즈의 최댓값과
    구분되지 않는다.** 정상 웨이퍼도 60개 시점 중 하나는 우연히 높다. 반면
    "임계를 넘은 **시간**"은 노이즈로는 잘 안 만들어진다 — 연속으로 여러 시점이
    같이 높아야 하기 때문이다. 그래서 스파이크 검출력의 핵심은 max가 아니라
    `time_above`다.
"""

from __future__ import annotations

import numpy as np

#: 임계 초과 판정 기준 — 정상 산포의 몇 배까지를 정상으로 볼 것인가
DEFAULT_K_SIGMA = 3.0

#: 정상 구간으로 볼 시작 시점 비율. 초반은 안정화(ramp-up) 구간이라 제외한다.
STEADY_START_FRACTION = 0.4

#: 이 모듈이 만드는 피처 접미사 — 모델이 쓸 컬럼을 한곳에서 관리한다
FEATURE_SUFFIXES: tuple[str, ...] = (
    "_slope",
    "_peak_dev",
    "_time_above",
    "_n_excursions",
    "_settle_time",
    "_range",
)


def _steady_slice(n_points: int) -> slice:
    """안정화 구간을 뺀 정상 구간.

    왜 빼나: 처리 초반의 상승 구간은 모든 웨이퍼에서 크게 변한다. 여기를 포함하면
        기울기·피크가 전부 ramp-up에 지배돼 정작 이상 신호가 묻힌다.
    """
    return slice(int(n_points * STEADY_START_FRACTION), n_points)


def trace_features(
    values: np.ndarray,
    t_sec: np.ndarray,
    *,
    sigma: float,
    k_sigma: float = DEFAULT_K_SIGMA,
) -> dict[str, np.ndarray]:
    """(웨이퍼 × 시점) 행렬에서 모양 피처를 계산한다.

    Args:
        values: shape (n_wafers, n_points) 시계열 값
        t_sec: shape (n_points,) 시각(초)
        sigma: 이 파라미터의 정상 산포 — 임계 계산 기준
        k_sigma: 정상 구간 중앙값에서 몇 σ를 넘으면 이상으로 볼 것인가

    Returns:
        피처명(접미사) → shape (n_wafers,) 배열

    피처 설명:
        slope        정상 구간 선형 추세. 처리 중 계속 밀리고 있는가(드리프트)
        peak_dev     **추세를 뺀 뒤의** 최대 이탈(절댓값). 얼마나 크게 튀었나
        time_above   임계를 넘은 총 시간(초). **스파이크 검출의 핵심**
        n_excursions 임계를 넘은 구간의 **개수**. 1회성인가 반복인가
        settle_time  처음으로 정상 대역에 들어온 시각. 안정화가 느렸나
        range        정상 구간 최대−최소
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    t_sec = np.asarray(t_sec, dtype=float)
    n_wafers, n_points = values.shape
    if n_points < 4:
        raise ValueError(f"시점이 너무 적습니다 ({n_points}개). 최소 4개 필요.")
    if len(t_sec) != n_points:
        raise ValueError(f"시각 길이 {len(t_sec)} != 값 길이 {n_points}")

    steady = _steady_slice(n_points)
    body = values[:, steady]
    t_body = t_sec[steady]
    dt = float(np.median(np.diff(t_sec))) if n_points > 1 else 1.0

    threshold = k_sigma * max(sigma, 1e-9)

    # 기울기 — 최소제곱 1차 적합을 벡터로
    t_centered = t_body - t_body.mean()
    denom = float(np.sum(t_centered**2)) or 1.0
    slope = (body - body.mean(axis=1, keepdims=True)) @ t_centered / denom

    # ── 추세를 뺀 잔차에서 스파이크를 본다 ★ ────────────────────────────
    # 왜: 장비가 안정화되며 값이 천천히 오르는 구간이 남아 있으면, 그 상승만으로도
    #     중앙값 대비 편차가 커져 **정상 웨이퍼도 임계를 계속 넘는다.** 실제로
    #     detrend 없이는 정상군의 `time_above`가 6초나 나왔다.
    #     느린 추세는 `slope`가 따로 담당하므로, 여기서는 빼고 봐야 스파이크만 남는다.
    deviation = body - (
        body.mean(axis=1, keepdims=True) + slope[:, None] * t_centered[None, :]
    )
    over = np.abs(deviation) > threshold

    # 임계 초과 구간의 개수 — False→True 전이 횟수
    padded = np.concatenate([np.zeros((n_wafers, 1), dtype=bool), over], axis=1)
    n_excursions = np.sum(padded[:, 1:] & ~padded[:, :-1], axis=1)

    # 안정화 시간 — 처음으로 대역 안에 들어온 시각 (전 구간 기준)
    full_center = np.median(body, axis=1, keepdims=True)
    inside = np.abs(values - full_center) <= threshold
    first_inside = np.argmax(inside, axis=1)
    never = ~inside.any(axis=1)
    settle = t_sec[np.clip(first_inside, 0, n_points - 1)]
    settle[never] = t_sec[-1]

    return {
        "_slope": slope,
        "_peak_dev": np.max(np.abs(deviation), axis=1),
        "_time_above": over.sum(axis=1).astype(float) * dt,
        "_n_excursions": n_excursions.astype(float),
        "_settle_time": settle,
        "_range": body.max(axis=1) - body.min(axis=1),
    }


def is_trace_feature(column: str) -> bool:
    """컬럼명이 trace 파생 피처인가."""
    return column.endswith(FEATURE_SUFFIXES)
