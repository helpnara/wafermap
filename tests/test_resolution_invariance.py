"""피처가 '맵 크기'를 지름길로 학습하지 않는지 (실측 전환 준비 — 설계서 M7).

왜 이 테스트가 필요한가 ★:
    합성 데이터는 맵이 전부 34×67로 **크기가 고정**이다. 그래서 크기에만 반응하는
    피처가 섞여 있어도 그 값은 상수가 되고, 모델은 상수를 무시하므로 성능에 아무
    영향이 없다 — 즉 **합성 데이터로는 이 결함을 볼 수 없다.**

    실측 WM-811K는 맵 크기가 제각각이다. 같은 피처가 거기서는 "맵 크기"라는
    강력한 지름길로 바뀐다. 실제로 `radon_mean_*` 20개가 그랬다. 정규화를 했는데도
    남아 있었는데, 어떤 각도에서 보든 투영의 합은 불량 개수와 같기 때문에
    열평균이 정의상 `불량수 / 검출기길이` 였고, 불량수로 나누면 `1/검출기길이`,
    곧 **맵 크기 하나만** 남았던 것이다.

    이 테스트는 그 부류의 결함을 실측 데이터를 받기 전에 잡는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wafermap.data import synth_wafer as sw
from wafermap.features import build

#: die 크기를 바꿔 격자 해상도를 바꾼다. 실측 맵 크기가 제각각인 상황을 흉내 낸다.
GRIDS = ((4.5, 9.0), (6.0, 6.0), (3.5, 3.5), (9.0, 9.0))
#: 해상도별 평균을 낼 표본 수. 적으면 뽑기 잡음이 해상도 효과를 덮는다.
N_PER_GRID = 12

#: 패턴 신호가 해상도 민감도의 몇 배는 되어야 하는가.
#: 1.0이면 "해상도가 패턴만큼 값을 흔든다"는 뜻이라 지름길 위험이 크다.
MIN_SIGNAL_RATIO = 1.0

#: 해상도 의존이 남아 있는 것을 **알고 두는** 피처들.
#:
#: 정규화로 고칠 수 없는 부류다 — 래스터 격자 위에서 연결성분과 둘레를 세는 이상,
#: 격자가 성기면 이웃 덩어리가 붙어 개수가 줄고 둘레는 계단처럼 각져 커진다.
#: 이것은 계산식의 결함이 아니라 이산화 자체의 성질이다.
#: 그래서 지우지 않고 **실측에서 확인할 목록**으로 남긴다 — WM-811K로 재측정할 때
#: 이 피처들의 중요도가 튀면 맵 크기를 학습한 것으로 의심해야 한다.
KNOWN_RESOLUTION_SENSITIVE = {
    "hu5": "고차 모멘트라 수치적으로 불안정하다",
    "largest_cluster_perimeter_ratio": "래스터 둘레는 격자가 성길수록 계단만큼 길어진다",
    "mean_cluster_size_norm": "격자가 성기면 이웃 덩어리가 하나로 붙는다",
    "cluster_count_norm": "같은 이유로 덩어리 개수가 줄어든다",
    "radon_peak_angle": "각도 자체라 원래 분산이 크다",
}


def _mean_features(pattern: str, grid: tuple[float, float]) -> pd.Series:
    geom = sw.build_geometry(die_width_mm=grid[0], die_height_mm=grid[1])
    rows = [
        build.extract_one(
            sw.generate_wafer_map(pattern, np.random.default_rng(2000 + i), geom, 0.8)
        )
        for i in range(N_PER_GRID)
    ]
    return pd.DataFrame(rows).select_dtypes("number").mean()


@pytest.fixture(scope="module")
def signal_ratio() -> pd.Series:
    """피처별 (패턴에 따른 변동) ÷ (해상도에 따른 변동).

    분자는 "이 피처가 패턴을 구분하는 힘", 분모는 "맵 크기에 휘둘리는 정도"다.
    """
    by_grid = pd.DataFrame({f"{g[0]}x{g[1]}": _mean_features("Edge-Ring", g) for g in GRIDS})
    by_pattern = pd.DataFrame(
        {p: _mean_features(p, GRIDS[0]) for p in ("Edge-Ring", "Center", "Scratch", "none")}
    )
    resolution_span = (by_grid.max(axis=1) - by_grid.min(axis=1)).abs()
    pattern_span = (by_pattern.max(axis=1) - by_pattern.min(axis=1)).abs()
    return (pattern_span / resolution_span.replace(0.0, np.nan)).dropna()


def test_radon_mean_features_are_gone():
    """정보가 0인 것으로 밝혀진 피처군이 되살아나지 않는지 못 박는다.

    `radon_mean_i` 는 정규화로 고칠 수 있는 값이 아니었다 — 어느 각도에서 보든
    같은 값이 나오도록 정의돼 있어 애초에 패턴 정보가 없다. 지운 것이 맞다.
    """
    geom = sw.build_geometry()
    feats = build.extract_one(sw.generate_wafer_map("Scratch", np.random.default_rng(1), geom, 0.8))
    revived = [k for k in feats if k.startswith("radon_mean_")]
    assert not revived, f"정보가 없는 피처가 되살아났다: {revived}"


def test_no_new_feature_is_dominated_by_map_size(signal_ratio):
    """어떤 피처도 패턴보다 맵 크기에 더 크게 반응하면 안 된다.

    이미 알고 있는 이산화 잔여분은 위 목록에서 빼고 본다. 목록에 없는 피처가
    걸리면 그것은 새로 생긴 지름길이므로 고쳐야 한다.
    """
    weak = signal_ratio[signal_ratio < MIN_SIGNAL_RATIO]
    unexpected = weak.drop(labels=[k for k in KNOWN_RESOLUTION_SENSITIVE if k in weak.index])
    assert unexpected.empty, (
        "맵 크기에 휘둘리는 새 피처 — 실측에서 지름길이 된다:\n"
        + unexpected.sort_values().round(3).to_string()
    )


def test_overall_features_are_mostly_resolution_robust(signal_ratio):
    """전체 경향을 하나의 숫자로 못 박는다.

    개별 피처는 흔들려도 전체가 건강하면 모델은 버틴다. 반대로 이 중앙값이 떨어지면
    피처 층 어딘가에 크기 성분이 새로 섞인 것이다.
    """
    assert signal_ratio.median() > 3.0, (
        f"패턴신호/해상도민감도 중앙값이 {signal_ratio.median():.2f}로 낮다"
    )


def test_known_sensitive_list_stays_accurate(signal_ratio):
    """목록이 현실과 어긋나지 않게 한다.

    고쳐져서 더 이상 민감하지 않은 피처가 목록에 남아 있으면, 다음 사람이 "이건
    원래 그런 것"이라고 넘길 근거가 된다. 목록은 살아 있어야 쓸모가 있다.
    """
    stale = [
        name for name, _ in KNOWN_RESOLUTION_SENSITIVE.items()
        if name in signal_ratio.index and signal_ratio[name] >= 3.0
    ]
    assert not stale, f"이제 안정적이므로 목록에서 빼야 한다: {stale}"


def test_feature_count_is_pinned():
    """피처 수를 고정한다.

    피처가 늘거나 줄면 저장된 모델과 어긋나 예측이 조용히 틀어진다. 의도한 변경이면
    이 숫자를 함께 고치면 된다 — 그 순간 '왜 바뀌었나'를 커밋에 적게 된다.
    """
    geom = sw.build_geometry()
    feats = build.extract_one(sw.generate_wafer_map("Center", np.random.default_rng(1), geom, 0.8))
    assert len(feats) == 99, f"피처 수가 {len(feats)}개로 바뀌었다"
