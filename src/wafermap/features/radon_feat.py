"""Radon 변환 피처 — 선형 구조(Scratch)와 링 구조(Donut)를 잡는다.

무엇을: 웨이퍼 맵을 여러 각도에서 '투영'해 만든 Radon 변환에서 통계를 뽑는다.

Radon 변환이 뭔가: 이미지를 특정 각도에서 빛을 비춰 그림자를 만든다고 생각하면 된다.
        각도 θ마다 그 방향의 직선들을 따라 픽셀 값을 더한 1차원 그림자가 나오고,
        모든 각도(0~180°)의 그림자를 쌓으면 2차원 sinogram이 된다.

        ┌──────────┐
        │    ╱     │   ← 대각선 스크래치를 45°에서 투영하면
        │   ╱      │      그림자가 한 점에 뾰족하게 뭉친다
        │  ╱       │      (선과 투영 방향이 나란하므로)
        └──────────┘      다른 각도에서는 넓게 퍼진다

왜 필요한가 ★: 이 "각도에 따라 뾰족함이 달라지는" 성질이 선형 구조의 지문이다.
        연결성분의 이심률도 길쭉함을 재지만, 불량이 끊겨서 여러 덩어리로 나뉘면
        각 덩어리는 작고 둥글어 보인다. Radon은 **끊어진 선도 하나로 이어서** 보므로
        더 강건하다. Wu et al. 2015가 WM-811K 분류에 이 피처를 쓴 이유다.

        Donut/Center 같은 동심 구조도 Radon에서 특징적인 모양이 나온다 —
        모든 각도에서 그림자가 비슷하되(회전 대칭) 가운데가 눌린 형태다.

참고: docs/00_design.md §M1(피처 목록), Wu et al. 2015
"""

from __future__ import annotations

import numpy as np

from wafermap.config import DIE_FAIL

#: sinogram을 몇 개 각도에서 계산할지 (많을수록 정밀하지만 느리다)
N_ANGLES = 36
#: 각도별 통계를 몇 개 점으로 리샘플링할지 (맵 크기가 달라도 차원을 맞추기 위함)
N_INTERP = 20


def _resample(values: np.ndarray, n: int) -> np.ndarray:
    """길이가 다른 1차원 배열을 고정 길이 n으로 선형 보간한다.

    왜 필요한가: 맵 크기가 26×26이든 300×200이든 **피처 차원은 같아야** 모델에
        넣을 수 있다. 실측 WM-811K는 맵 크기가 제각각이라 이 정규화가 필수다.
    """
    if values.size == 0:
        return np.zeros(n)
    if values.size == 1:
        return np.full(n, float(values[0]))
    src = np.linspace(0.0, 1.0, values.size)
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, values)


def extract(wafer: np.ndarray) -> dict[str, float]:
    """Radon 변환 기반 피처를 뽑는다.

    Args:
        wafer: (rows, cols) 배열. 0=die 없음, 1=pass, 2=fail

    Returns:
        피처명 → 값 딕셔너리 (약 46개: 평균 20 + 표준편차 20 + 요약 6)
    """
    from skimage.transform import radon

    fail = (wafer == DIE_FAIL).astype(np.float64)

    empty = {}
    empty.update({f"radon_mean_{i}": 0.0 for i in range(N_INTERP)})
    empty.update({f"radon_std_{i}": 0.0 for i in range(N_INTERP)})
    empty.update(
        {
            "radon_peak_angle": 0.0,
            "radon_peak_sharpness": 0.0,
            "radon_angular_contrast": 0.0,
            "radon_mean_range": 0.0,
            "radon_std_range": 0.0,
            "radon_peak_to_median": 0.0,
            "radon_isotropy": 0.0,
        }
    )
    if fail.sum() < 3:
        return empty

    theta = np.linspace(0.0, 180.0, N_ANGLES, endpoint=False)
    # circle=False로 두면 맵이 정사각이 아니어도 잘리지 않는다
    sino = radon(fail, theta=theta, circle=False, preserve_range=True)

    # 각 각도(열)별로 그림자의 평균/표준편차를 낸다
    col_mean = sino.mean(axis=0)
    col_std = sino.std(axis=0)
    col_max = sino.max(axis=0)

    total = fail.sum()

    # ⚠️ 불량 개수로 나눠 정규화한다.
    # 왜: Radon 투영은 픽셀 값의 '합'이라 불량이 많을수록, 맵이 클수록 값이 커진다.
    #     정규화하지 않으면 이 피처들이 사실상 "불량 개수"와 "맵 크기"를 인코딩하게 되어,
    #     모델이 형상 대신 그 지름길을 학습한다(geometry.basic_features의 경고와 같은 문제).
    col_mean = col_mean / total
    col_std = col_std / total
    col_max_n = col_max / total

    out: dict[str, float] = {}
    # 각도축을 따라 정렬한 뒤 리샘플링 — 회전에 불변인 표현이 된다.
    # (웨이퍼는 스크래치가 어느 방향으로든 날 수 있으므로 절대 각도는 의미가 없다)
    for i, v in enumerate(_resample(np.sort(col_mean), N_INTERP)):
        out[f"radon_mean_{i}"] = float(v)
    for i, v in enumerate(_resample(np.sort(col_std), N_INTERP)):
        out[f"radon_std_{i}"] = float(v)

    peak_idx = int(np.argmax(col_max))

    out.update(
        {
            # 투영이 가장 뾰족해지는 각도 (0~1로 정규화)
            "radon_peak_angle": float(peak_idx / N_ANGLES),
            # 그 각도에서 얼마나 뭉쳤나 — 선형 구조일수록 크다
            "radon_peak_sharpness": float(col_max.max() / (total + 1e-6)),
            # 각도 간 최대/최소 대비 — 방향성이 있으면 크고, 등방적이면 1에 가깝다
            "radon_angular_contrast": float(col_max.max() / (col_max.min() + 1e-6)),
            "radon_mean_range": float(col_mean.max() - col_mean.min()),
            "radon_std_range": float(col_std.max() - col_std.min()),
            # 상위/하위 각도 대비 (정규화 값 기준)
            "radon_peak_to_median": float(col_max_n.max() / (np.median(col_max_n) + 1e-9)),
            # 등방성: 각도에 따른 변동이 작을수록 1에 가깝다 (Center/Donut 계열)
            "radon_isotropy": float(1.0 - col_max.std() / (col_max.mean() + 1e-6)),
        }
    )
    return out
