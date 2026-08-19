"""웨이퍼 맵 그리기 — M6.

무엇을: die별 합격/불합격 배열을 화면에 띄울 수 있는 그림으로 만든다.
왜 별도 모듈인가: 갤러리 썸네일과 확대 뷰가 **같은 규칙**으로 그려져야 한다.
    화면마다 색과 종횡비를 따로 정하면 같은 웨이퍼가 다르게 보인다.
"""

from __future__ import annotations

import numpy as np

from wafermap.config import DIE_FAIL, DIE_NONE, DIE_PASS
from wafermap.data.synth_wafer import DIE_HEIGHT_MM, DIE_WIDTH_MM
from wafermap.ui import theme

#: 셀 값 → 색. 파랑(합격) ↔ 주황(불합격) 축이라 적록 색약에서도 구분된다(§4.8).
CELL_COLORS = {
    DIE_NONE: "rgba(0,0,0,0)",   # 웨이퍼 밖 — 투명
    DIE_PASS: "#c7dbf5",
    DIE_FAIL: theme.BAD,
}

#: die 종횡비. 격자를 그대로 그리면 웨이퍼가 타원이 된다.
DIE_ASPECT = DIE_HEIGHT_MM / DIE_WIDTH_MM


def _rgb(hex_or_rgba: str) -> tuple[int, int, int, int]:
    if hex_or_rgba.startswith("rgba"):
        return (0, 0, 0, 0)
    value = hex_or_rgba.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), 255)


def to_rgba(die_map: np.ndarray) -> np.ndarray:
    """맵을 RGBA 이미지 배열로 바꾼다 (썸네일용).

    왜 이미지인가: 썸네일을 20장 띄울 때 Plotly 그림 20개는 브라우저가 버겁다.
        이미지 한 장이면 훨씬 가볍고, 작게 그릴 때 화질 차이도 없다.
    """
    out = np.zeros((*die_map.shape, 4), dtype=np.uint8)
    for value, color in CELL_COLORS.items():
        out[die_map == value] = _rgb(color)
    return out


def thumbnail(die_map: np.ndarray, *, scale: int = 3) -> np.ndarray:
    """썸네일 이미지 — die 종횡비를 반영해 웨이퍼가 원형으로 보이게 한다.

    어떻게: 세로 방향으로 die 종횡비만큼 더 늘려 픽셀을 반복한다.
    왜: DRAM die는 정사각형이 아니다(가로 8mm × 세로 16mm 가정). 격자를 그대로
        그리면 웨이퍼가 **납작한 타원**으로 보여, 반경 방향 패턴을 오해하게 된다.
    """
    rgba = to_rgba(die_map)
    return np.repeat(np.repeat(rgba, int(round(scale * DIE_ASPECT)), axis=0), scale, axis=1)


def wafer_figure(
    die_map: np.ndarray,
    *,
    title: str = "",
    show_rings: bool = False,
    show_sectors: bool = False,
    n_rings: int = 5,
    n_sectors: int = 8,
    height: int = 460,
):
    """확대 뷰 — 반경/섹터 오버레이를 켤 수 있다.

    왜 오버레이가 필요한가: M2의 피처가 **반경대(ring)와 섹터**로 맵을 나눠 계산한다.
        화면에서 같은 선을 그려 주지 않으면, 피처 값이 무엇을 잰 것인지 알 수 없다.
        "ring3의 불량률이 높다"는 말이 그림과 연결되어야 이해가 된다.
    """
    import plotly.graph_objects as go

    rows, cols = die_map.shape
    # 물리 좌표(mm)로 그린다 — 그래야 원이 원으로 보이고 오버레이도 정확하다
    x = (np.arange(cols) - (cols - 1) / 2.0) * DIE_WIDTH_MM
    y = (np.arange(rows) - (rows - 1) / 2.0) * DIE_HEIGHT_MM

    display = np.where(die_map == DIE_NONE, np.nan, die_map)
    fig = go.Figure(
        go.Heatmap(
            z=display, x=x, y=y,
            colorscale=[[0.0, CELL_COLORS[DIE_PASS]], [1.0, CELL_COLORS[DIE_FAIL]]],
            zmin=DIE_PASS, zmax=DIE_FAIL,
            showscale=False, hoverongaps=False,
            hovertemplate="x %{x:.0f}mm · y %{y:.0f}mm<br>%{customdata}<extra></extra>",
            customdata=np.where(die_map == DIE_FAIL, "불합격", "합격"),
            xgap=0.4, ygap=0.4,
        )
    )

    r_max = float(np.nanmax(np.abs(x))) + DIE_WIDTH_MM
    if show_rings:
        for i in range(1, n_rings + 1):
            radius = r_max * i / n_rings
            fig.add_shape(
                type="circle", xref="x", yref="y",
                x0=-radius, y0=-radius, x1=radius, y1=radius,
                line=dict(color=theme.MUTED, width=1, dash="dot"),
            )
    if show_sectors:
        for i in range(n_sectors):
            angle = 2 * np.pi * i / n_sectors
            fig.add_shape(
                type="line", xref="x", yref="y",
                x0=0, y0=0,
                x1=r_max * np.cos(angle), y1=r_max * np.sin(angle),
                line=dict(color=theme.MUTED, width=1, dash="dot"),
            )

    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=34 if title else 8, b=8),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    # 제목이 없을 때 title=None을 넘기면 화면에 "undefined"가 찍힌다.
    # 아예 설정하지 않는 것이 맞다.
    if title:
        fig.update_layout(title=title)
    # scaleanchor로 x·y 축의 물리 축척을 묶는다 → 웨이퍼가 정원으로 보인다
    fig.update_xaxes(visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    return fig


def radial_profile(die_map: np.ndarray, n_rings: int = 10) -> np.ndarray:
    """반경대별 불량률 — 확대 뷰 옆에 붙여 패턴을 수치로 확인한다."""
    rows, cols = die_map.shape
    x = (np.arange(cols) - (cols - 1) / 2.0) * DIE_WIDTH_MM
    y = (np.arange(rows) - (rows - 1) / 2.0) * DIE_HEIGHT_MM
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx**2 + yy**2)

    valid = die_map != DIE_NONE
    r_norm = np.zeros_like(radius)
    if valid.any():
        r_norm = radius / (radius[valid].max() + 1e-9)

    out = np.zeros(n_rings, dtype=float)
    for i in range(n_rings):
        lo, hi = i / n_rings, (i + 1) / n_rings
        band = valid & (r_norm >= lo) & (r_norm < hi if i < n_rings - 1 else r_norm <= hi)
        if band.any():
            out[i] = float((die_map[band] == DIE_FAIL).mean())
    return out
