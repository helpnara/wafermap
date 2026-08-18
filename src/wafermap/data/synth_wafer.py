"""합성 wafer map 생성기 — WM-811K와 동일 스키마의 die-level bin map을 만든다.

무엇을: 불량 패턴명(Center/Donut/Edge-Ring/…)과 심각도를 받아, 웨이퍼 위 die별
        pass/fail 2차원 배열(0=die 없음, 1=pass, 2=fail)을 생성한다.

어떻게: 패턴마다 **불량 발생 확률장(probability field)** 을 물리 좌표 위에 그린 뒤,
        각 die에서 베르누이 시행으로 pass/fail을 뽑는다. 확률장을 거쳐 가는 이유는
        같은 패턴이라도 웨이퍼마다 조금씩 다르게 나와야 실제 데이터와 닮기 때문이다.

왜 물리 좌표인가: 웨이퍼는 원형인데 die는 직사각형(가상 DRAM 4.5×9.0mm)이라
        **행/열 인덱스 거리와 실제 거리가 다르다.** 인덱스 기준으로 원을 그리면
        세로로 찌그러진 타원이 되어, Center/Donut/Edge-Ring 같은 반경 기반 패턴이
        물리적으로 틀린 모양이 된다. 그래서 die 중심의 mm 좌표로 반경을 계산한다.

참고: docs/00_design.md §2.2(합성 전략), §2.4(패턴-원인 매핑)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wafermap.config import DIE_FAIL, DIE_NONE, DIE_PASS, WAFER_SIZE_MM

# ── 가상 DRAM die 규격 (설계서 §2.5.1) ──────────────────────────────────
DIE_WIDTH_MM = 4.5
DIE_HEIGHT_MM = 9.0
EDGE_EXCLUSION_MM = 3.0  # 웨이퍼 최외곽 배제영역 — 여기엔 die를 놓지 않는다

#: 패턴이 없는 웨이퍼에도 존재하는 기본 랜덤 불량률 범위
BASELINE_FAIL_RATE = (0.005, 0.030)


@dataclass(frozen=True)
class WaferGeometry:
    """웨이퍼 die 배치 기하 — 한 번 계산해 모든 웨이퍼가 재사용한다.

    Attributes:
        mask: die가 존재하는 위치 (rows, cols) bool 배열
        r_norm: die 중심의 정규화 반경 (0=중심, 1=사용 가능 최외곽). mask 밖은 NaN
        theta: die 중심의 각도(radian, -π~π). mask 밖은 NaN
        x_mm, y_mm: die 중심의 물리 좌표 (웨이퍼 중심이 원점)
        n_die: 유효 die 개수
    """

    mask: np.ndarray
    r_norm: np.ndarray
    theta: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape

    @property
    def n_die(self) -> int:
        return int(self.mask.sum())


def build_geometry(
    die_width_mm: float = DIE_WIDTH_MM,
    die_height_mm: float = DIE_HEIGHT_MM,
    wafer_dia_mm: float = WAFER_SIZE_MM,
    edge_exclusion_mm: float = EDGE_EXCLUSION_MM,
) -> WaferGeometry:
    """die 격자를 웨이퍼 원판 위에 배치하고 각 die의 극좌표를 계산한다.

    무엇을: 직사각형 die를 격자로 깔고, **네 모서리가 모두** 사용 가능 반경 안에 들어오는
            die만 유효(mask=True)로 남긴다.
    왜 네 모서리인가: die 중심만 보면 웨이퍼 경계에 걸쳐 절반만 걸린 die까지 유효로 잡힌다.
            실제 팹은 완전한 die만 노광·측정하므로 "full die" 기준을 쓴다.

    Returns:
        WaferGeometry — mask/r_norm/theta 등 사전 계산된 기하 정보
    """
    r_usable = wafer_dia_mm / 2.0 - edge_exclusion_mm

    # die 개수는 웨이퍼 지름을 die 크기로 나눈 값(격자를 넉넉히 깔고 마스크로 잘라낸다)
    n_cols = int(np.ceil(wafer_dia_mm / die_width_mm))
    n_rows = int(np.ceil(wafer_dia_mm / die_height_mm))

    # die 중심 좌표 — 격자를 웨이퍼 중심에 맞춰 정렬
    cx = (np.arange(n_cols) - (n_cols - 1) / 2.0) * die_width_mm
    cy = (np.arange(n_rows) - (n_rows - 1) / 2.0) * die_height_mm
    x_mm, y_mm = np.meshgrid(cx, cy)

    # 네 모서리 중 가장 먼 지점이 사용 가능 반경 안에 있어야 유효 die
    corner_r = np.sqrt(
        (np.abs(x_mm) + die_width_mm / 2.0) ** 2
        + (np.abs(y_mm) + die_height_mm / 2.0) ** 2
    )
    mask = corner_r <= r_usable

    with np.errstate(invalid="ignore"):
        r_center = np.sqrt(x_mm**2 + y_mm**2)
        r_norm = np.where(mask, r_center / r_usable, np.nan)
        theta = np.where(mask, np.arctan2(y_mm, x_mm), np.nan)

    return WaferGeometry(mask=mask, r_norm=r_norm, theta=theta, x_mm=x_mm, y_mm=y_mm)


# ──────────────────────────────────────────────────────────────────────────
# 패턴별 불량 확률장
# ──────────────────────────────────────────────────────────────────────────


def _prob_center(geom: WaferGeometry, rng: np.random.Generator, sev: float) -> np.ndarray:
    """중심부 집중 — 가우시안 형태로 중심에서 멀어질수록 감소."""
    width = rng.uniform(0.22, 0.40)
    amp = np.clip(rng.uniform(0.55, 0.85) * sev, 0.0, 0.97)
    return amp * np.exp(-((geom.r_norm / width) ** 2))


def _prob_donut(geom: WaferGeometry, rng: np.random.Generator, sev: float) -> np.ndarray:
    """중심을 비운 링 — 특정 반경대에서 최대가 되는 가우시안 링."""
    r0 = rng.uniform(0.42, 0.66)
    width = rng.uniform(0.09, 0.16)
    amp = np.clip(rng.uniform(0.55, 0.85) * sev, 0.0, 0.97)
    return amp * np.exp(-(((geom.r_norm - r0) / width) ** 2))


def _prob_edge_ring(geom: WaferGeometry, rng: np.random.Generator, sev: float) -> np.ndarray:
    """최외곽 전체 링 — 임계 반경 바깥에서 급격히 상승(시그모이드)."""
    r_thresh = rng.uniform(0.82, 0.92)
    sharpness = rng.uniform(18.0, 40.0)
    amp = np.clip(rng.uniform(0.60, 0.90) * sev, 0.0, 0.97)
    return amp / (1.0 + np.exp(-sharpness * (geom.r_norm - r_thresh)))


def _prob_edge_loc(geom: WaferGeometry, rng: np.random.Generator, sev: float) -> np.ndarray:
    """가장자리 국부 — Edge-Ring을 각도 구간으로 잘라낸 형태.

    왜 Edge-Ring을 재사용하나: 물리적으로 둘은 '엣지에서 발생'이라는 같은 계열이고,
    차이는 원주 전체냐 일부냐다. 실제로 두 클래스는 분류기가 가장 자주 혼동하는 쌍이며,
    이 구조로 생성해야 그 혼동이 데이터에도 자연스럽게 재현된다.
    """
    ring = _prob_edge_ring(geom, rng, sev)

    theta0 = rng.uniform(-np.pi, np.pi)
    half_width = rng.uniform(np.pi / 9, np.pi / 3)  # 20°~60° 반각

    # 각도 차이를 -π~π로 감싸서 경계(±π)에서 끊기지 않게 한다
    dtheta = np.abs(np.mod(geom.theta - theta0 + np.pi, 2 * np.pi) - np.pi)
    angular = np.exp(-((dtheta / half_width) ** 2))
    return ring * angular


def _prob_loc(geom: WaferGeometry, rng: np.random.Generator, sev: float) -> np.ndarray:
    """국부 클러스터 — 웨이퍼 내부 임의 위치의 가우시안 블롭 1~2개."""
    prob = np.zeros_like(geom.r_norm)
    r_usable = WAFER_SIZE_MM / 2.0 - EDGE_EXCLUSION_MM

    for _ in range(rng.integers(1, 3)):
        # 엣지 계열과 구분되도록 블롭 중심은 r<0.75 안쪽에 둔다
        r0 = rng.uniform(0.05, 0.72) * r_usable
        t0 = rng.uniform(-np.pi, np.pi)
        x0, y0 = r0 * np.cos(t0), r0 * np.sin(t0)

        sigma_mm = rng.uniform(0.06, 0.15) * r_usable
        d2 = (geom.x_mm - x0) ** 2 + (geom.y_mm - y0) ** 2
        amp = np.clip(rng.uniform(0.55, 0.88) * sev, 0.0, 0.97)
        prob = np.maximum(prob, amp * np.exp(-d2 / (2 * sigma_mm**2)))

    return np.where(geom.mask, prob, np.nan)


def _prob_scratch(geom: WaferGeometry, rng: np.random.Generator, sev: float) -> np.ndarray:
    """선형 긁힘 — 웨이퍼를 가로지르는 가늘고 긴 곡선.

    어떻게: 시작점과 방향을 뽑아 여러 개의 짧은 선분을 이어 붙이고(약간의 곡률),
            각 die에서 그 경로까지의 최단 거리가 가까울수록 불량 확률을 높인다.
    왜 곡선인가: CMP 슬러리 이물에 의한 스크래치는 회전하는 패드를 따라가므로
            완전한 직선이 아니라 완만한 호를 그리는 경우가 많다.
    """
    r_usable = WAFER_SIZE_MM / 2.0 - EDGE_EXCLUSION_MM

    # 시작점을 웨이퍼 가장자리 근처에 두고 안쪽을 향해 진행시킨다
    t_start = rng.uniform(-np.pi, np.pi)
    r_start = rng.uniform(0.55, 0.98) * r_usable
    x, y = r_start * np.cos(t_start), r_start * np.sin(t_start)

    heading = t_start + np.pi + rng.uniform(-0.6, 0.6)  # 대체로 중심 방향
    curvature = rng.uniform(-0.06, 0.06)                # 스텝마다 방향이 조금씩 휜다
    n_seg = int(rng.integers(14, 34))
    step_mm = rng.uniform(0.05, 0.09) * r_usable

    pts_x, pts_y = [x], [y]
    for _ in range(n_seg):
        heading += curvature
        x += step_mm * np.cos(heading)
        y += step_mm * np.sin(heading)
        if np.hypot(x, y) > r_usable * 1.05:  # 웨이퍼를 벗어나면 중단
            break
        pts_x.append(x)
        pts_y.append(y)

    path_x = np.asarray(pts_x)
    path_y = np.asarray(pts_y)

    # 각 die에서 경로 위 모든 점까지의 거리 중 최솟값
    d2 = (geom.x_mm[..., None] - path_x) ** 2 + (geom.y_mm[..., None] - path_y) ** 2
    dist = np.sqrt(d2.min(axis=-1))

    width_mm = rng.uniform(0.9, 2.2) * DIE_WIDTH_MM
    amp = np.clip(rng.uniform(0.60, 0.92) * sev, 0.0, 0.97)
    prob = amp * np.exp(-((dist / width_mm) ** 2))
    return np.where(geom.mask, prob, np.nan)


def _prob_random(geom: WaferGeometry, rng: np.random.Generator, sev: float) -> np.ndarray:
    """산발성 — 공간 구조 없이 전면에 균일하게 상승한 불량률."""
    level = np.clip(rng.uniform(0.10, 0.28) * sev, 0.0, 0.6)
    return np.where(geom.mask, level, np.nan)


def _prob_near_full(geom: WaferGeometry, rng: np.random.Generator, sev: float) -> np.ndarray:
    """전면 불량 — 설비 major fault 수준. 약간의 반경 기울기를 남긴다."""
    base = np.clip(rng.uniform(0.72, 0.93) * sev, 0.0, 0.98)
    tilt = rng.uniform(-0.06, 0.06) * geom.r_norm
    return np.clip(base + tilt, 0.0, 0.99)


def _prob_none(geom: WaferGeometry, rng: np.random.Generator, sev: float) -> np.ndarray:
    """정상 — 패턴 성분 없음(기본 불량률만 남는다)."""
    return np.where(geom.mask, 0.0, np.nan)


_PATTERN_FUNCS = {
    "Center": _prob_center,
    "Donut": _prob_donut,
    "Edge-Ring": _prob_edge_ring,
    "Edge-Loc": _prob_edge_loc,
    "Loc": _prob_loc,
    "Scratch": _prob_scratch,
    "Random": _prob_random,
    "Near-full": _prob_near_full,
    "none": _prob_none,
}


# ──────────────────────────────────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────────────────────────────────


def generate_wafer_map(
    pattern: str,
    rng: np.random.Generator,
    geom: WaferGeometry | None = None,
    severity: float = 1.0,
) -> np.ndarray:
    """불량 패턴 1종에 해당하는 wafer map 배열을 생성한다.

    Args:
        pattern: PATTERN_LABELS 중 하나
        rng: 난수 생성기 (재현성을 위해 호출자가 주입)
        geom: 웨이퍼 기하. None이면 기본 규격으로 생성(반복 호출 시 미리 만들어 넘길 것)
        severity: 심각도 배수. 1.0이 기준이며 클수록 불량이 강하게 나타난다.

    Returns:
        (rows, cols) int8 배열 — 0=die 없음, 1=pass, 2=fail
    """
    if pattern not in _PATTERN_FUNCS:
        raise ValueError(f"알 수 없는 패턴: {pattern!r}")
    if geom is None:
        geom = build_geometry()

    prob = _PATTERN_FUNCS[pattern](geom, rng, severity)

    # 패턴과 무관하게 항상 존재하는 기본 불량률을 더한다.
    # 왜: 실제 웨이퍼는 정상이어도 die 몇 개는 떨어진다. 이게 없으면 'none' 클래스가
    #     전부 완벽한 웨이퍼가 되어 분류가 비현실적으로 쉬워진다.
    baseline = rng.uniform(*BASELINE_FAIL_RATE)
    prob = np.clip(np.nan_to_num(prob, nan=0.0) + baseline, 0.0, 0.995)

    draw = rng.random(geom.shape)
    wafer = np.full(geom.shape, DIE_NONE, dtype=np.int8)
    wafer[geom.mask] = np.where(draw[geom.mask] < prob[geom.mask], DIE_FAIL, DIE_PASS)
    return wafer


def map_stats(wafer: np.ndarray) -> tuple[int, int, float]:
    """wafer map에서 (die_total, die_pass, yield_pct)를 계산한다."""
    die_total = int(np.count_nonzero(wafer != DIE_NONE))
    die_pass = int(np.count_nonzero(wafer == DIE_PASS))
    yield_pct = 100.0 * die_pass / die_total if die_total else 0.0
    return die_total, die_pass, yield_pct
