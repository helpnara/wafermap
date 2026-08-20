"""화면 2 — 웨이퍼맵 탐색 (설계서 §4.5).

이 화면이 답하는 것:
    1. 불량 패턴이 실제로 어떻게 생겼나 — 갤러리
    2. 이 웨이퍼는 무엇을 겪었나 — 확대 뷰 + 이력
    3. 피처가 무엇을 재는가 — 반경/섹터 오버레이
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from wafermap.config import PATTERN_LABELS
from wafermap.data import loader
from wafermap.ui import layout, theme
from wafermap.viz import wafer as wviz

SOURCE = "synthetic"

#: 한 페이지에 띄울 썸네일 수 (설계서 §4A.4: 데스크탑 20장 / 모바일 6장)
PAGE_SIZE_DESKTOP = 20
PAGE_SIZE_MOBILE = 6

GALLERY_CSS = """
<style>
  .wm-gallery { display:grid; grid-template-columns:repeat(2,1fr); gap:.6rem; margin:.4rem 0 .8rem 0; }
  .wm-gallery figure { margin:0; }
  .wm-gallery img { width:100%; image-rendering:pixelated; border-radius:6px; }
  .wm-gallery figcaption { font-size:.74rem; color:#6b7280; line-height:1.35; margin-top:.2rem; }
</style>
"""

MECHANISM_LABELS = {
    "none": "정상",
    "process": "공정 기인",
    "test": "검사 기인",
    "spike": "순간 스파이크",
    "drift": "드리프트",
    "interaction": "교호작용",
}


@st.cache_data(show_spinner="웨이퍼 목록을 불러오는 중…")
def _catalog(source: str) -> pd.DataFrame:
    """탐색에 필요한 열만 합쳐 둔다 (맵 자체는 필요할 때만 읽는다)."""
    master = loader.load_wafer_master(source)
    truth = loader.load_ground_truth(source)
    cols = ["wafer_id", "cause_mechanism", "true_root_step", "true_root_equip"]
    return master.merge(truth[cols], on="wafer_id", how="left")


@st.cache_data(show_spinner=False, max_entries=64)
def _die_map(wafer_id: str, source: str) -> np.ndarray:
    return loader.load_die_map(wafer_id, source)


def _data_uri(image: np.ndarray) -> str:
    """RGBA 배열을 PNG data URI로 — HTML 격자에 바로 넣기 위해."""
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGBA").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@st.cache_data(show_spinner=False, max_entries=8)
def _thumbnails(wafer_ids: tuple[str, ...], source: str) -> list[np.ndarray]:
    """썸네일 한 페이지분을 한꺼번에 만든다.

    왜 튜플을 받나: `st.cache_data`의 캐시 키는 인자를 해시한다. 리스트는 해시가
        안 되므로 튜플로 받아야 페이지 단위 캐시가 걸린다.
    """
    return [wviz.thumbnail(_die_map(w, source)) for w in wafer_ids]


def _filters(catalog: pd.DataFrame) -> pd.DataFrame:
    """필터 바 — 모바일에서는 세로로 쌓인다."""
    mobile = layout.is_mobile()
    cols = st.columns(1 if mobile else 3)

    with cols[0]:
        patterns = [p for p in PATTERN_LABELS if (catalog["pattern_label"] == p).any()]
        pattern = st.selectbox(
            "불량 패턴", ["(전체)"] + patterns,
            index=(patterns.index("Edge-Ring") + 1) if "Edge-Ring" in patterns else 0,
        )
    with cols[1 if not mobile else 0]:
        available = [
            m for m in MECHANISM_LABELS if (catalog["cause_mechanism"] == m).any()
        ]
        mechanism = st.selectbox(
            "원인 경로", ["(전체)"] + [MECHANISM_LABELS[m] for m in available],
            help="같은 패턴이라도 원인이 다르다. 맵으로는 구분되지 않는다는 점을 직접 확인해 보라.",
        )
    with cols[2 if not mobile else 0]:
        only_defect = st.toggle("불량만 보기", value=True)

    view = catalog
    if pattern != "(전체)":
        view = view[view["pattern_label"] == pattern]
    if mechanism != "(전체)":
        key = next(k for k, v in MECHANISM_LABELS.items() if v == mechanism)
        view = view[view["cause_mechanism"] == key]
    if only_defect:
        view = view[view["pattern_label"] != "none"]
    return view.sort_values("eds_time")


def _gallery(view: pd.DataFrame) -> None:
    """썸네일 갤러리 — 페이지네이션 필수 (전체 로드 금지, §4.5)."""
    mobile = layout.is_mobile()
    page_size = PAGE_SIZE_MOBILE if mobile else PAGE_SIZE_DESKTOP
    n_pages = max(1, int(np.ceil(len(view) / page_size)))

    head, nav = st.columns([3, 2]) if not mobile else (st.container(), st.container())
    with head:
        st.caption(f"조건에 맞는 웨이퍼 {len(view):,}장 · {n_pages}페이지")
    with nav:
        page = st.number_input(
            "페이지", min_value=1, max_value=n_pages, value=1, step=1,
            label_visibility="collapsed" if not mobile else "visible",
        )

    chunk = view.iloc[(page - 1) * page_size: page * page_size]
    if chunk.empty:
        st.info("조건에 맞는 웨이퍼가 없습니다. 필터를 넓혀 보세요.")
        return

    images = _thumbnails(tuple(chunk["wafer_id"]), SOURCE)

    if mobile:
        # ★ Streamlit의 st.columns는 좁은 화면에서 **자동으로 1열로 무너진다.**
        #   썸네일이 한 줄에 하나씩 쌓이면 6장을 보는 데 화면 여섯 번을 넘겨야 한다.
        #   격자를 직접 그려야 설계서의 2×3(§4A.4)을 지킬 수 있다.
        cells = "".join(
            f'<figure><img src="{_data_uri(image)}" alt="{row["wafer_id"]}"/>'
            f'<figcaption>{row["wafer_id"]}<br>'
            f'{row["pattern_label"]} · {row["yield_pct"]:.1f}%</figcaption></figure>'
            for (_, row), image in zip(chunk.iterrows(), images)
        )
        st.markdown(f'<div class="wm-gallery">{cells}</div>', unsafe_allow_html=True)
        picked = st.selectbox(
            "자세히 볼 웨이퍼", list(chunk["wafer_id"]),
            help="좁은 화면에서는 작은 버튼보다 목록에서 고르는 편이 정확하다",
        )
        if st.button("확대 보기로", width="stretch"):
            st.session_state["selected_wafer_id"] = picked
            st.rerun()
        return

    n_cols = 5
    for start in range(0, len(chunk), n_cols):
        block = chunk.iloc[start:start + n_cols]
        for col, (_, row), image in zip(
            st.columns(n_cols), block.iterrows(), images[start:start + n_cols]
        ):
            with col:
                st.image(image, use_container_width=True)
                st.caption(
                    f"{row['wafer_id']}\n\n"
                    f"{row['pattern_label']} · 수율 {row['yield_pct']:.1f}%"
                )
                if st.button("자세히", key=f"pick_{row['wafer_id']}", width="stretch"):
                    st.session_state["selected_wafer_id"] = row["wafer_id"]
                    st.rerun()


def _detail(catalog: pd.DataFrame, wafer_id: str) -> None:
    """확대 뷰 + 그 웨이퍼가 겪은 일."""
    row = catalog[catalog["wafer_id"] == wafer_id]
    if row.empty:
        st.warning(f"{wafer_id} 를 찾을 수 없습니다.")
        return
    row = row.iloc[0]
    die_map = _die_map(wafer_id, SOURCE)

    st.subheader(f"{wafer_id}")
    mechanism = MECHANISM_LABELS.get(str(row["cause_mechanism"]), "—")
    layout.metric_grid([
        ("패턴", str(row["pattern_label"]), None),
        ("수율", f"{row['yield_pct']:.2f}%", f"{row['die_pass']:,}/{row['die_total']:,} die"),
        ("원인 경로", mechanism, "정답지 — 분석에는 쓰지 않는다"),
        ("원인 설비", str(row["true_root_equip"] or "—"), str(row["true_root_step"] or "")),
    ])

    with layout.responsive_split(("웨이퍼 맵", "반경 프로파일"), ratio=(1.3, 1.0)) as (left, right):
        with left:
            toggles = st.columns(2)
            rings = toggles[0].toggle("반경대 표시", value=False)
            sectors = toggles[1].toggle("섹터 표시", value=False)
            st.plotly_chart(
                wviz.wafer_figure(
                    die_map, show_rings=rings, show_sectors=sectors,
                    height=420 if not layout.is_mobile() else 340,
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption(
                "주황이 불합격 die다. 오버레이를 켜면 **M2 피처가 맵을 어떻게 나눠 재는지** "
                "볼 수 있다 — 반경대 10개, 섹터 12개로 나눠 각 구역의 불량률을 피처로 쓴다."
            )
        with right:
            _radial_chart(die_map)

    st.markdown(
        '<div class="wm-note"><b>맵만 봐서는 원인을 알 수 없다.</b> '
        'Edge-Ring은 식각 챔버 마모로도, 프로브 카드 니들 마모로도 똑같이 생긴다. '
        '위의 "원인 경로"는 시뮬레이터의 <b>정답지</b>이며, 분석 모델에는 절대 넣지 않는다. '
        '채점할 때만 쓴다.</div>',
        unsafe_allow_html=True,
    )

    with st.expander(f"같은 lot 25장 보기 — {row['lot_id']}"):
        _lot_view(catalog, str(row["lot_id"]), wafer_id)


def _radial_chart(die_map: np.ndarray) -> None:
    import plotly.graph_objects as go

    profile = wviz.radial_profile(die_map) * 100
    labels = [f"{i}" for i in range(len(profile))]
    colors = [
        theme.BAD if value > profile.mean() + 1e-9 else theme.GOOD for value in profile
    ]
    fig = go.Figure(
        go.Bar(
            x=labels, y=profile, marker_color=colors,
            hovertemplate="반경대 %{x}<br>불량률 %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="white", showlegend=False,
        xaxis_title="반경대 (0=중심 → 9=최외곽)", yaxis_title="불량률 (%)",
    )
    fig.update_yaxes(gridcolor=theme.BORDER, ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "중심에 몰리면 Center, 최외곽만 높으면 Edge-Ring, 중간 반경대만 솟으면 Donut이다. "
        "**이 프로파일이 M2 피처의 핵심**이며, 사람이 눈으로 보는 것과 같은 정보를 쓴다."
    )


def _lot_view(catalog: pd.DataFrame, lot_id: str, highlight: str) -> None:
    """lot 25장을 슬롯 순서로 — 로트 전체가 감염됐는지 한 장만인지 보인다."""
    lot = catalog[catalog["lot_id"] == lot_id].sort_values("slot_no")
    st.caption(
        f"{lot_id} · {len(lot)}장. "
        "**lot 전체가 같은 패턴이면 설비 이상**, 한두 장만이면 산발 불량일 가능성이 크다. "
        "커미널리티 분석의 검정 단위가 웨이퍼가 아니라 lot인 이유이기도 하다."
    )
    images = _thumbnails(tuple(lot["wafer_id"]), SOURCE)
    n_cols = 4 if layout.is_mobile() else 9
    for start in range(0, len(lot), n_cols):
        block = lot.iloc[start:start + n_cols]
        for col, (_, row), image in zip(
            st.columns(n_cols), block.iterrows(), images[start:start + n_cols]
        ):
            with col:
                st.image(image, use_container_width=True)
                mark = "▶ " if row["wafer_id"] == highlight else ""
                col.caption(f"{mark}#{row['slot_no']:02d} {row['pattern_label']}")


def render() -> None:
    st.markdown(theme.CSS + GALLERY_CSS, unsafe_allow_html=True)
    st.title("웨이퍼맵 탐색")

    try:
        catalog = _catalog(SOURCE)
    except FileNotFoundError as exc:
        layout.missing_artifact(exc, what="웨이퍼맵 탐색")
        return

    st.markdown(
        '<div class="wm-why"><b>왜 맵부터 보는가</b><br>'
        'EDS는 die 하나하나의 합격/불합격을 낸다. 그걸 <b>웨이퍼 위 좌표에 그리면</b> '
        '숫자로는 안 보이던 형상이 나타난다. 링, 중심, 긁힘 — 이 형상이 '
        '어느 공정을 의심할지 알려주는 첫 단서다.</div>',
        unsafe_allow_html=True,
    )

    view = _filters(catalog)
    st.divider()

    selected = st.session_state.get("selected_wafer_id")
    if selected is None and not view.empty:
        selected = str(view.iloc[0]["wafer_id"])
        st.session_state["selected_wafer_id"] = selected

    if layout.is_mobile():
        # 좁은 화면에서 갤러리와 상세를 같이 쌓으면 스크롤이 끝없이 길어진다
        tabs = st.tabs(["갤러리", "확대 보기"])
        with tabs[0]:
            _gallery(view)
        with tabs[1]:
            if selected:
                _detail(catalog, selected)
    else:
        _gallery(view)
        st.divider()
        if selected:
            _detail(catalog, selected)


render()
