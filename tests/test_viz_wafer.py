"""웨이퍼 맵 그리기 검증 (M6).

핵심 질문:
  1. **웨이퍼가 원으로 보이는가?** — die가 직사각형이라 그냥 그리면 타원이 된다
  2. **반경 프로파일이 패턴을 반영하는가?** — 이게 M2 피처의 핵심이다
"""

from __future__ import annotations

import numpy as np
import pytest

from wafermap.config import DIE_FAIL, DIE_NONE, DIE_PASS
from wafermap.data.synth_wafer import DIE_HEIGHT_MM, DIE_WIDTH_MM
from wafermap.viz import wafer as wviz


@pytest.fixture
def edge_ring(geom):
    """최외곽 반경대만 불량인 맵."""
    die_map = np.where(geom.mask, DIE_PASS, DIE_NONE).astype(np.int8)
    die_map[geom.mask & (geom.r_norm > 0.85)] = DIE_FAIL
    return die_map


@pytest.fixture
def center_blob(geom):
    """중심부만 불량인 맵."""
    die_map = np.where(geom.mask, DIE_PASS, DIE_NONE).astype(np.int8)
    die_map[geom.mask & (geom.r_norm < 0.25)] = DIE_FAIL
    return die_map


def test_thumbnail_is_nearly_square(edge_ring):
    """die 종횡비를 반영해 썸네일이 정사각에 가까워야 한다 ★.

    왜: DRAM die는 가로 4.5mm × 세로 9.0mm다. 격자를 그대로 그리면 웨이퍼가
        **납작한 타원**으로 보여 반경 방향 패턴을 오해하게 된다.
    """
    image = wviz.thumbnail(edge_ring)
    height, width = image.shape[:2]
    assert 0.85 < height / width < 1.2, f"종횡비 {height / width:.2f} — 타원으로 보인다"


def test_die_aspect_matches_config():
    assert wviz.DIE_ASPECT == pytest.approx(DIE_HEIGHT_MM / DIE_WIDTH_MM)


def test_outside_wafer_is_transparent(edge_ring):
    """웨이퍼 밖은 투명해야 한다 (배경과 섞이면 원형이 뭉개진다)."""
    rgba = wviz.to_rgba(edge_ring)
    outside = rgba[edge_ring == DIE_NONE]
    assert (outside[:, 3] == 0).all()


def test_pass_and_fail_use_different_colors(edge_ring):
    rgba = wviz.to_rgba(edge_ring)
    assert not np.array_equal(
        rgba[edge_ring == DIE_PASS][0], rgba[edge_ring == DIE_FAIL][0]
    )


# ── 반경 프로파일 ────────────────────────────────────────────────────────


def test_radial_profile_peaks_at_edge(edge_ring):
    profile = wviz.radial_profile(edge_ring)
    assert profile.argmax() >= 8, f"최외곽이 아니라 {profile.argmax()}번 반경대가 최고"
    assert profile[0] == 0.0


def test_radial_profile_peaks_at_center(center_blob):
    profile = wviz.radial_profile(center_blob)
    assert profile.argmax() <= 2, f"중심이 아니라 {profile.argmax()}번 반경대가 최고"


def test_radial_profile_length_is_configurable(edge_ring):
    assert len(wviz.radial_profile(edge_ring, n_rings=5)) == 5


def test_radial_profile_ignores_outside_die(geom):
    """웨이퍼 밖 셀이 분모에 들어가면 불량률이 희석된다."""
    die_map = np.where(geom.mask, DIE_FAIL, DIE_NONE).astype(np.int8)
    profile = wviz.radial_profile(die_map)
    assert (profile[profile > 0] == 1.0).all()


# ── 확대 뷰 ──────────────────────────────────────────────────────────────


def test_figure_locks_axis_scale(edge_ring):
    """x·y 축척을 묶어야 웨이퍼가 정원으로 보인다."""
    fig = wviz.wafer_figure(edge_ring)
    assert fig.layout.xaxis.scaleanchor == "y"
    assert fig.layout.xaxis.scaleratio == 1


def test_figure_has_no_title_by_default(edge_ring):
    """제목을 비우면 화면에 'undefined'가 찍히던 문제를 막는다."""
    fig = wviz.wafer_figure(edge_ring)
    assert fig.layout.title.text is None


def test_overlays_are_optional(edge_ring):
    assert len(wviz.wafer_figure(edge_ring).layout.shapes) == 0
    with_rings = wviz.wafer_figure(edge_ring, show_rings=True, n_rings=5)
    assert len(with_rings.layout.shapes) == 5
    with_both = wviz.wafer_figure(
        edge_ring, show_rings=True, show_sectors=True, n_rings=5, n_sectors=8
    )
    assert len(with_both.layout.shapes) == 13
