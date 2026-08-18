"""기하 피처 — 웨이퍼 맵의 공간 구조를 숫자로 요약한다.

무엇을: wafer map 2차원 배열에서 반경 프로파일, 각도 프로파일, 13-zone 밀도,
        Hu 모멘트 등을 뽑아 낸다. 이 숫자들이 LightGBM의 입력이 된다.

어떻게 — 맵에서 **타원을 직접 유도한다**:
        die 마스크(`wafer != 0`)의 경계 상자에서 반축 a, b를 구하고
        `r = sqrt(((x-cx)/a)² + ((y-cy)/b)²)` 로 정규화 반경을 계산한다.

왜 타원을 유도하나 ★: 실측 WM-811K는 맵 크기가 제각각이고(26×26부터 300×200까지)
        die의 물리 치수도 공개되어 있지 않다. 합성 데이터처럼 mm 좌표를 쓸 수가 없다.
        그렇다고 행/열 인덱스 거리를 그냥 쓰면, 가로세로 비가 2:1인 맵에서 Edge-Ring이
        타원으로 일그러져 반경 피처가 전부 틀어진다.
        마스크에서 타원을 유도하면 **맵 크기와 종횡비에 무관하게** 같은 의미의 반경이
        나온다. 합성/실측 어느 쪽이든 같은 코드가 동작하는 근거가 여기에 있다.

참고: docs/00_design.md §M1(피처 목록), Wu et al. 2015 (13-zone 밀도 피처)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wafermap.config import DIE_FAIL, DIE_NONE

#: 반경 프로파일을 나눌 동심 링 개수
N_RINGS = 10
#: 각도 프로파일을 나눌 섹터 개수
N_SECTORS = 12


@dataclass(frozen=True)
class MapGeometry:
    """맵 하나에서 유도한 좌표계.

    Attributes:
        mask: die가 존재하는 위치
        r_norm: 정규화 반경 (0=중심, 약 1=최외곽). mask 밖은 NaN
        theta: 각도 (-π~π). mask 밖은 NaN
        n_die: 유효 die 수
    """

    mask: np.ndarray
    r_norm: np.ndarray
    theta: np.ndarray

    @property
    def n_die(self) -> int:
        return int(self.mask.sum())


def derive_geometry(wafer: np.ndarray) -> MapGeometry:
    """맵에서 정규화 극좌표계를 유도한다.

    어떻게: die 마스크의 경계 상자로 타원의 중심과 반축을 정한 뒤,
            타원 좌표계에서 반경과 각도를 계산한다.

    Args:
        wafer: (rows, cols) 배열. 0=die 없음, 1=pass, 2=fail

    Returns:
        MapGeometry — 이후 모든 반경/각도 피처가 이 좌표계를 공유한다
    """
    mask = wafer != DIE_NONE
    rows, cols = np.nonzero(mask)

    if rows.size == 0:
        nan = np.full(wafer.shape, np.nan)
        return MapGeometry(mask=mask, r_norm=nan, theta=nan)

    # 경계 상자 → 타원 중심과 반축. +0.5는 셀 중심 보정.
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    cy, cx = (r0 + r1) / 2.0, (c0 + c1) / 2.0
    a = max((c1 - c0) / 2.0, 0.5)  # 가로 반축
    b = max((r1 - r0) / 2.0, 0.5)  # 세로 반축

    yy, xx = np.mgrid[0 : wafer.shape[0], 0 : wafer.shape[1]]
    u = (xx - cx) / a  # 타원 좌표로 정규화 → 종횡비 제거
    v = (yy - cy) / b

    r = np.sqrt(u**2 + v**2)
    return MapGeometry(
        mask=mask,
        r_norm=np.where(mask, r, np.nan),
        theta=np.where(mask, np.arctan2(v, u), np.nan),
    )


# ──────────────────────────────────────────────────────────────────────────
# 피처 그룹별 계산
# ──────────────────────────────────────────────────────────────────────────


def basic_features(wafer: np.ndarray, geom: MapGeometry) -> dict[str, float]:
    """기본 불량률 지표.

    ⚠️ 여기에 맵 크기(rows/cols/aspect)나 die 총수를 **넣지 않는다.**

    왜 ★: 실측 WM-811K는 맵 크기가 26×26부터 300×200까지 제각각이고, 크기가 곧
        제품군을 식별한다. 그리고 제품군마다 불량 패턴 분포가 다르다. 크기를 피처로
        주면 모델이 웨이퍼의 **형상**이 아니라 "이 크기의 맵은 보통 Edge-Ring"이라는
        지름길을 학습한다. 검증 성능은 올라가지만 새 제품에서는 무너진다.
        합성 데이터는 맵 크기가 모두 같아 이 문제가 드러나지 않으므로, 실측으로
        넘어가기 전에 구조적으로 막아 둔다.

    die_fill_ratio는 크기가 아니라 **모양** 지표라 남긴다(원판을 얼마나 채웠는가,
    완전한 원이면 약 π/4 ≈ 0.785).
    """
    n_die = geom.n_die
    n_fail = int(np.count_nonzero(wafer == DIE_FAIL))
    rows, cols = wafer.shape
    return {
        "fail_ratio": n_fail / n_die if n_die else 0.0,
        "die_fill_ratio": n_die / (rows * cols) if rows * cols else 0.0,
    }


def radial_features(wafer: np.ndarray, geom: MapGeometry) -> dict[str, float]:
    """반경 프로파일 — Center / Donut / Edge-Ring 판별의 핵심.

    무엇을: 중심에서 최외곽까지 10개 동심 링으로 나눠 각 링의 불량률을 잰다.
    왜:    세 패턴 모두 "불량이 어느 반경대에 몰려 있는가"로 구분된다.
           Center는 링0이 최대, Donut은 중간 링이 최대, Edge-Ring은 링9가 최대다.
           이 10개 숫자만으로 셋을 대부분 가를 수 있다.
    """
    fail = wafer == DIE_FAIL
    edges = np.linspace(0.0, 1.0, N_RINGS + 1)

    profile = np.zeros(N_RINGS)
    for i in range(N_RINGS):
        lo, hi = edges[i], edges[i + 1]
        # 마지막 링은 r>1인 모서리 die까지 포함한다 (타원 근사의 오차 흡수)
        sel = geom.mask & (geom.r_norm >= lo) & (
            (geom.r_norm < hi) if i < N_RINGS - 1 else True
        )
        profile[i] = fail[sel].mean() if sel.any() else 0.0

    out = {f"ring{i}_fail_rate": float(profile[i]) for i in range(N_RINGS)}

    inner = profile[:5].mean()
    outer = profile[5:].mean()
    out.update(
        {
            "radial_peak_ring": float(np.argmax(profile)),
            "radial_peak_value": float(profile.max()),
            "radial_std": float(profile.std()),
            "radial_slope": float(np.polyfit(np.arange(N_RINGS), profile, 1)[0]),
            "edge_inner_ratio": float(outer / (inner + 1e-6)),
            # 최외곽 2링 vs 나머지 — Edge-Ring 전용 지표
            "outer2_ratio": float(profile[-2:].mean() / (profile[:-2].mean() + 1e-6)),
            # 중심 2링이 비었는지 — Donut을 Center와 가르는 지표
            "center2_ratio": float(profile[:2].mean() / (profile.mean() + 1e-6)),
        }
    )
    return out


def angular_features(wafer: np.ndarray, geom: MapGeometry) -> dict[str, float]:
    """각도 프로파일 — Edge-Loc과 Edge-Ring을 가른다.

    왜: 둘 다 엣지에 불량이 몰리지만, Edge-Ring은 원주 **전체**이고
        Edge-Loc은 **일부 방향**에만 나타난다. 반경 프로파일만으로는 구분되지 않고,
        각도 방향의 쏠림을 재야 갈라진다.
    """
    fail = wafer == DIE_FAIL
    edges = np.linspace(-np.pi, np.pi, N_SECTORS + 1)

    profile = np.zeros(N_SECTORS)
    for i in range(N_SECTORS):
        sel = geom.mask & (geom.theta >= edges[i]) & (geom.theta < edges[i + 1])
        profile[i] = fail[sel].mean() if sel.any() else 0.0

    out = {f"sector{i}_fail_rate": float(profile[i]) for i in range(N_SECTORS)}

    mean = profile.mean()
    # 원형 통계: 불량을 각도 벡터의 합으로 보고 그 크기를 잰다.
    # 고르게 퍼지면 벡터들이 상쇄되어 0에 가깝고, 한쪽으로 쏠리면 1에 가까워진다.
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight = profile / (profile.sum() + 1e-9)
    resultant = np.hypot((weight * np.cos(centers)).sum(), (weight * np.sin(centers)).sum())

    out.update(
        {
            "angular_std": float(profile.std()),
            "angular_max_ratio": float(profile.max() / (mean + 1e-6)),
            "angular_concentration": float(resultant),
            # 불량이 존재하는 섹터의 비율 — 전체 원주냐 일부냐
            "angular_active_ratio": float((profile > mean * 0.5).mean()),
        }
    )
    return out


def zone13_features(wafer: np.ndarray, geom: MapGeometry) -> dict[str, float]:
    """13-zone 밀도 — Wu et al. 2015의 표준 피처.

    어떻게 13개로 나누나: 맵을 5×5로 자른 뒤
        · 상/하/좌/우 가장자리 띠 4개
        · 가운데 3×3 격자 9개
        = 13개 영역의 불량 밀도를 잰다.

    왜 반경 피처가 있는데 또 필요한가: 반경/각도 프로파일은 회전 대칭 구조에 강하지만
        Loc처럼 **특정 위치**에 뭉친 불량은 잘 표현하지 못한다. 13-zone은 위치를
        직접 인코딩해서 그 빈틈을 메운다. 문헌에서 표준으로 쓰이는 이유이기도 하다.
    """
    fail = (wafer == DIE_FAIL).astype(float)
    die = (wafer != DIE_NONE).astype(float)
    rows, cols = wafer.shape

    ri = np.linspace(0, rows, 6).astype(int)
    ci = np.linspace(0, cols, 6).astype(int)

    def density(r_slice: slice, c_slice: slice) -> float:
        n_die = die[r_slice, c_slice].sum()
        return float(fail[r_slice, c_slice].sum() / n_die) if n_die > 0 else 0.0

    zones = [
        density(slice(ri[0], ri[1]), slice(None)),  # 상단 띠
        density(slice(ri[4], ri[5]), slice(None)),  # 하단 띠
        density(slice(None), slice(ci[0], ci[1])),  # 좌측 띠
        density(slice(None), slice(ci[4], ci[5])),  # 우측 띠
    ]
    for i in range(1, 4):  # 가운데 3×3
        for j in range(1, 4):
            zones.append(density(slice(ri[i], ri[i + 1]), slice(ci[j], ci[j + 1])))

    out = {f"zone{i}_density": z for i, z in enumerate(zones)}
    out["zone_max"] = float(max(zones))
    out["zone_std"] = float(np.std(zones))
    return out


def moment_features(wafer: np.ndarray, geom: MapGeometry) -> dict[str, float]:
    """Hu 모멘트와 형상 지표 — 불량 분포의 '모양'을 요약한다.

    Hu 모멘트는 이동·회전·크기에 불변인 7개 값이다. 웨이퍼는 회전 방향이
    고정되어 있지 않으므로(같은 스크래치가 어느 각도로든 날 수 있다) 회전 불변량이
    유용하다. 값의 범위가 매우 넓어 log 스케일로 압축해 넣는다.
    """
    from skimage.measure import moments_central, moments_hu, moments_normalized

    fail = (wafer == DIE_FAIL).astype(float)
    out: dict[str, float] = {}

    if fail.sum() < 3:
        # 불량이 거의 없으면 모멘트가 정의되지 않는다 — 0으로 채운다
        out.update({f"hu{i}": 0.0 for i in range(7)})
        out.update({"fail_centroid_r": 0.0, "fail_spread": 0.0, "fail_anisotropy": 0.0})
        return out

    mu = moments_central(fail)
    try:
        hu = moments_hu(moments_normalized(mu))
    except (ValueError, ZeroDivisionError):
        hu = np.zeros(7)

    # log 압축: 부호를 보존하면서 자릿수 차이를 줄인다
    for i, h in enumerate(hu):
        out[f"hu{i}"] = float(np.sign(h) * np.log10(np.abs(h) + 1e-30))

    # 불량 무게중심이 웨이퍼 중심에서 얼마나 떨어졌는가 (0=중심, 1=엣지)
    r_vals = geom.r_norm[geom.mask]
    fail_mask = (wafer == DIE_FAIL)[geom.mask]
    if fail_mask.any():
        out["fail_centroid_r"] = float(r_vals[fail_mask].mean())
        out["fail_spread"] = float(r_vals[fail_mask].std())
    else:
        out["fail_centroid_r"] = 0.0
        out["fail_spread"] = 0.0

    # 2차 중심모멘트의 이방성 — 길쭉함(Scratch)을 잡는다
    m20, m02, m11 = mu[2, 0], mu[0, 2], mu[1, 1]
    denom = m20 + m02
    out["fail_anisotropy"] = (
        float(np.sqrt((m20 - m02) ** 2 + 4 * m11**2) / denom) if denom > 0 else 0.0
    )
    return out


def extract(wafer: np.ndarray) -> dict[str, float]:
    """맵 하나에서 기하 피처 전체를 뽑는다.

    Returns:
        피처명 → 값 딕셔너리 (약 55개)
    """
    geom = derive_geometry(wafer)
    features: dict[str, float] = {}
    features.update(basic_features(wafer, geom))
    features.update(radial_features(wafer, geom))
    features.update(angular_features(wafer, geom))
    features.update(zone13_features(wafer, geom))
    features.update(moment_features(wafer, geom))
    return features
