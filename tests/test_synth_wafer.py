"""합성 wafer map 생성기 검증.

핵심 질문: **생성된 맵이 패턴 이름에 걸맞은 물리적 형상을 갖는가?**
'Edge-Ring인데 중심에 불량이 몰려 있다'면 이후 모든 모델링이 무의미해진다.
그래서 눈으로 보는 대신 반경 프로파일 같은 정량 지표로 검사한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from wafermap.config import DIE_FAIL, DIE_NONE, DIE_PASS, PATTERN_LABELS
from wafermap.data.synth_wafer import (
    build_geometry,
    generate_wafer_map,
    map_stats,
)


def _fail_rate_by_ring(wafer, geom, n_rings=8):
    """중심→엣지 방향 링별 불량률을 계산한다."""
    edges = np.linspace(0, 1, n_rings + 1)
    fail = wafer == DIE_FAIL
    out = []
    for i in range(n_rings):
        sel = geom.mask & (geom.r_norm >= edges[i]) & (geom.r_norm < edges[i + 1])
        out.append(fail[sel].mean() if sel.sum() else 0.0)
    return np.asarray(out)


def _mean_profile(pattern, geom, n=30, seed=5):
    rng = np.random.default_rng(seed)
    return np.mean(
        [_fail_rate_by_ring(generate_wafer_map(pattern, rng, geom), geom) for _ in range(n)],
        axis=0,
    )


# ── 기하 ─────────────────────────────────────────────────────────────────


def test_geometry_is_circular(geom):
    """die 마스크가 원형이어야 한다 (인덱스 기준 타원이 아니라)."""
    r = geom.r_norm[geom.mask]
    assert np.all(r <= 1.0), "사용 가능 반경을 넘는 die가 존재"
    assert r.max() > 0.9, "최외곽까지 die가 채워지지 않음"
    assert geom.n_die > 1_000, f"die 수가 비현실적으로 적음: {geom.n_die}"


def test_geometry_mask_excludes_corners(geom):
    """격자 네 모서리는 웨이퍼 밖이므로 die가 없어야 한다."""
    assert not geom.mask[0, 0]
    assert not geom.mask[0, -1]
    assert not geom.mask[-1, 0]
    assert not geom.mask[-1, -1]


def test_geometry_is_symmetric(geom):
    """웨이퍼는 원형이므로 die 배치가 상하·좌우 대칭이어야 한다."""
    assert np.array_equal(geom.mask, geom.mask[::-1, :])
    assert np.array_equal(geom.mask, geom.mask[:, ::-1])


# ── 맵 생성 기본 ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("pattern", PATTERN_LABELS)
def test_map_values_are_valid(pattern, geom, rng):
    wafer = generate_wafer_map(pattern, rng, geom)
    assert wafer.shape == geom.shape
    assert set(np.unique(wafer)) <= {DIE_NONE, DIE_PASS, DIE_FAIL}
    # 마스크 밖은 반드시 die 없음
    assert np.all(wafer[~geom.mask] == DIE_NONE)
    assert np.all(wafer[geom.mask] != DIE_NONE)


def test_generation_is_reproducible(geom):
    """같은 시드는 같은 맵을 만들어야 한다 (실험 재현성의 전제)."""
    a = generate_wafer_map("Center", np.random.default_rng(7), geom)
    b = generate_wafer_map("Center", np.random.default_rng(7), geom)
    assert np.array_equal(a, b)


def test_map_stats_consistent(geom, rng):
    wafer = generate_wafer_map("Loc", rng, geom)
    die_total, die_pass, yield_pct = map_stats(wafer)
    assert die_total == geom.n_die
    assert 0 <= die_pass <= die_total
    assert yield_pct == pytest.approx(100 * die_pass / die_total)


# ── 패턴별 물리 형상 ─────────────────────────────────────────────────────


def test_center_concentrates_at_center(geom):
    p = _mean_profile("Center", geom)
    assert p[0] > p[-1] * 5, "중심 불량률이 엣지보다 충분히 높지 않음"
    # 중심에서 바깥으로 대체로 감소해야 한다
    assert p[0] > p[2] > p[5]


def test_edge_ring_concentrates_at_edge(geom):
    p = _mean_profile("Edge-Ring", geom)
    assert p[-1] > p[0] * 5, "엣지 불량률이 중심보다 충분히 높지 않음"
    assert p[-1] == p.max()


def test_donut_peaks_at_middle_radius(geom):
    """Donut은 중심이 비고 중간 반경대가 최대여야 한다."""
    p = _mean_profile("Donut", geom)
    peak = int(np.argmax(p))
    assert 2 <= peak <= 5, f"정점이 중간 반경대가 아님 (ring {peak})"
    assert p[peak] > p[0] * 5, "중심이 충분히 비어 있지 않음"
    assert p[peak] > p[-1] * 3, "최외곽이 충분히 비어 있지 않음"


def test_near_full_is_uniformly_high(geom):
    p = _mean_profile("Near-full", geom)
    assert p.min() > 0.6, "전면 불량이 아님"
    assert p.std() < 0.06, "Near-full은 공간 구조가 거의 없어야 함"


def test_none_is_low_and_flat(geom):
    p = _mean_profile("none", geom)
    assert p.max() < 0.06, "정상 웨이퍼의 기본 불량률이 너무 높음"
    assert p.std() < 0.02


def test_random_is_flat_but_elevated(geom):
    """Random은 'none'보다 높지만 공간 구조가 없어야 한다."""
    p_random = _mean_profile("Random", geom)
    p_none = _mean_profile("none", geom)
    assert p_random.mean() > p_none.mean() * 3
    assert p_random.std() < 0.04, "Random에 공간 구조가 생김"


def test_edge_loc_is_edge_biased_but_partial(geom):
    """Edge-Loc은 엣지 편향이되, Edge-Ring보다 전체 불량률이 낮아야 한다.

    왜: 둘 다 엣지 계열이지만 Edge-Loc은 원주 일부만 차지한다. 이 차이가
        데이터에 재현되어야 두 클래스의 혼동이 자연스럽게 발생한다.
    """
    p_loc = _mean_profile("Edge-Loc", geom)
    p_ring = _mean_profile("Edge-Ring", geom)
    assert p_loc[-1] > p_loc[0], "엣지 편향이 없음"
    assert p_loc.sum() < p_ring.sum(), "Edge-Loc이 Edge-Ring보다 넓게 퍼짐"


def test_scratch_is_sparse_and_elongated(geom):
    """Scratch는 불량 die 수가 적고, 형상이 길쭉해야 한다."""
    from skimage.measure import label, regionprops

    rng = np.random.default_rng(3)
    elongations = []
    for _ in range(20):
        wafer = generate_wafer_map("Scratch", rng, geom)
        fail = wafer == DIE_FAIL
        assert fail.mean() < 0.25, "Scratch 불량률이 지나치게 높음"

        regions = regionprops(label(fail))
        if regions:
            big = max(regions, key=lambda r: r.area)
            if big.axis_minor_length > 0:
                elongations.append(big.axis_major_length / big.axis_minor_length)

    assert elongations, "연결성분을 찾지 못함"
    assert np.median(elongations) > 2.0, "가장 큰 클러스터가 길쭉하지 않음"


def test_severity_increases_defect_rate(geom):
    """severity가 클수록 불량이 심해져야 한다 — FDC와 맵을 잇는 연결고리."""
    rng = np.random.default_rng(11)
    low = np.mean([
        (generate_wafer_map("Center", rng, geom, severity=0.6) == DIE_FAIL).mean()
        for _ in range(25)
    ])
    high = np.mean([
        (generate_wafer_map("Center", rng, geom, severity=1.4) == DIE_FAIL).mean()
        for _ in range(25)
    ])
    assert high > low * 1.2, f"severity가 불량률에 반영되지 않음 ({low:.3f} → {high:.3f})"


def test_unknown_pattern_raises(geom, rng):
    with pytest.raises(ValueError, match="알 수 없는 패턴"):
        generate_wafer_map("Bogus", rng, geom)


def test_build_geometry_scales_with_die_size():
    """die가 작아지면 die 개수가 늘어야 한다 (기하 계산의 정합성)."""
    coarse = build_geometry(die_width_mm=9.0, die_height_mm=12.0)
    fine = build_geometry(die_width_mm=4.5, die_height_mm=6.0)
    assert fine.n_die > coarse.n_die * 2
