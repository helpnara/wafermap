"""연결성분 피처 — 불량이 '뭉쳐 있는가, 흩어져 있는가'를 잰다.

무엇을: 인접한 불량 die를 하나의 덩어리(cluster)로 묶고, 그 개수·크기·모양을 잰다.

왜 필요한가 ★: 반경/각도 프로파일은 **공간 구조가 회전 대칭일 때** 잘 통한다.
        그런데 Loc(국부 뭉침), Scratch(선형), Random(산발)은 셋 다 반경 프로파일이
        비슷하게 밋밋하다. 실제로 M1 검증에서 Loc과 Scratch의 반경 프로파일이
        거의 같게 나왔다.
        이 셋을 가르는 것은 반경이 아니라 **덩어리의 모양**이다.
          · Loc     → 큰 덩어리 1~2개, 둥글다 (이심률 낮음)
          · Scratch → 큰 덩어리 1개, 길쭉하다 (이심률 높음)
          · Random  → 작은 덩어리 여러 개
        연결성분 피처가 이 구분을 담당한다.

참고: docs/00_design.md §M1(피처 목록)
"""

from __future__ import annotations

import numpy as np

from wafermap.config import DIE_FAIL, DIE_NONE

#: 잡음으로 볼 최소 덩어리 크기 (die 개수)
MIN_CLUSTER_SIZE = 2


def extract(wafer: np.ndarray) -> dict[str, float]:
    """연결성분 기반 피처를 뽑는다.

    Args:
        wafer: (rows, cols) 배열. 0=die 없음, 1=pass, 2=fail

    Returns:
        피처명 → 값 딕셔너리 (약 14개)
    """
    from skimage.measure import label, regionprops

    fail = wafer == DIE_FAIL
    n_fail = int(fail.sum())

    empty = {
        "cluster_count_norm": 0.0,
        "cluster_count_significant_norm": 0.0,
        "largest_cluster_die_ratio": 0.0,
        "largest_cluster_ratio": 0.0,
        "largest_cluster_eccentricity": 0.0,
        "largest_cluster_solidity": 0.0,
        "largest_cluster_extent": 0.0,
        "largest_cluster_elongation": 0.0,
        "largest_cluster_perimeter_ratio": 0.0,
        "mean_cluster_size_norm": 0.0,
        "cluster_size_cv": 0.0,
        "top3_cluster_ratio": 0.0,
        "isolated_fail_ratio": 0.0,
        "fail_compactness": 0.0,
    }
    if n_fail < MIN_CLUSTER_SIZE:
        return empty

    # 8-연결(대각선 포함) — 스크래치가 대각선으로 지나가면 4-연결로는 끊어진다
    labels = label(fail, connectivity=2)
    regions = regionprops(labels)
    if not regions:
        return empty

    sizes = np.array([r.area for r in regions], dtype=float)
    biggest = max(regions, key=lambda r: r.area)

    minor = float(biggest.axis_minor_length)
    major = float(biggest.axis_major_length)

    n_die = int(np.count_nonzero(wafer != DIE_NONE)) or 1

    # ⚠️ 모든 개수·크기 피처를 die 수 또는 불량 수로 나눈다.
    # 왜: 절대 개수를 그대로 쓰면 큰 맵일수록 값이 커져, 모델이 형상이 아니라
    #     맵 크기를 학습한다(geometry.basic_features의 경고와 같은 이유).
    out = {
        "cluster_count_norm": float(len(regions) / n_die),
        # 크기 2 이상만 — 단독 die는 대개 랜덤 불량이라 덩어리로 세지 않는다
        "cluster_count_significant_norm": float((sizes >= MIN_CLUSTER_SIZE).sum() / n_die),
        "largest_cluster_die_ratio": float(sizes.max() / n_die),
        "largest_cluster_ratio": float(sizes.max() / n_fail),
        "largest_cluster_eccentricity": float(biggest.eccentricity),
        "largest_cluster_solidity": float(biggest.solidity),
        "largest_cluster_extent": float(biggest.extent),
        # 장축/단축 비 — Scratch 판별의 직접 지표
        "largest_cluster_elongation": float(major / minor) if minor > 0.5 else float(major),
        # 둘레²/면적 — 길쭉하거나 너덜너덜할수록 커진다
        "largest_cluster_perimeter_ratio": float(
            biggest.perimeter**2 / biggest.area if biggest.area > 0 else 0.0
        ),
        "mean_cluster_size_norm": float(sizes.mean() / n_die),
        # 변동계수(표준편차/평균) — 크기에 무관한 산포 지표
        "cluster_size_cv": float(sizes.std() / (sizes.mean() + 1e-9)),
        "top3_cluster_ratio": float(np.sort(sizes)[-3:].sum() / n_fail),
        # 단독으로 떨어진 die 비율 — Random에서 높다
        "isolated_fail_ratio": float((sizes == 1).sum() / n_fail),
        # 불량 die 수 대비 덩어리 수 — 1에 가까울수록 완전히 흩어진 상태
        "fail_compactness": float(len(regions) / n_fail),
    }
    return out
