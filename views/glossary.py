"""화면 9 — 용어사전 (설계서 §5-A).

반도체 용어는 비전문가에게 진입 장벽이다. 분석 화면에서 "커미널리티"나 "프로브 카드"가
나올 때 뜻을 모르면 그 화면 전체가 읽히지 않는다. 여기서 한곳에 모아 둔다.

각 용어는 세 층으로 설명한다.
    short       한 줄 정의
    plain       비유를 쓴 쉬운 설명
    why_matters **왜 이것이 현업에서 중요한가** — 돈·시간·품질과의 연결
"""

from __future__ import annotations

import streamlit as st

from wafermap.learning import glossary
from wafermap.ui import layout, theme

#: 분류별 색 — 화면에서 종류를 빠르게 구분하기 위한 것이지, 의미를 담지는 않는다
CATEGORY_COLORS = {
    "제품": "#0f766e", "테스트": "#1f6feb", "공정": "#7c3aed",
    "설비": "#b45309", "불량패턴": "#c2410c", "통계": "#0369a1",
    "머신러닝": "#4338ca", "데이터": "#525252",
}


def _term_card(term: glossary.Term, *, expanded: bool = False) -> None:
    color = CATEGORY_COLORS.get(term.category, theme.MUTED)
    with st.expander(f"{term.term} — {term.short}", expanded=expanded):
        st.markdown(theme.badge(term.category, color), unsafe_allow_html=True)
        st.markdown(term.plain)

        if term.detail:
            st.markdown(f"**더 정확히는**\n\n{term.detail}")

        if term.why_matters:
            st.markdown(
                f'<div class="wm-why"><b>왜 중요한가</b><br>'
                f'{term.why_matters.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )

        meta = []
        if term.aliases:
            meta.append(f"**다른 이름** {' · '.join(term.aliases)}")
        if term.related:
            meta.append(f"**관련 용어** {' · '.join(term.related)}")
        if term.used_in:
            files = " · ".join(f"`{path}`" for path in term.used_in)
            meta.append(f"**코드에서** {files}")
        if meta:
            st.caption("　　".join(meta) if not layout.is_mobile() else "\n\n".join(meta))


def render() -> None:
    st.markdown(theme.CSS, unsafe_allow_html=True)
    st.title("용어사전")

    counts = glossary.stats()
    st.caption(
        f"{counts['합계']}개 용어 · "
        "분석 화면에 나오는 용어는 모두 여기에 있다"
    )

    st.markdown(
        '<div class="wm-why"><b>왜 용어사전이 필요한가</b><br>'
        '반도체 용어는 비전문가에게 진입 장벽이다. "커미널리티 분석에서 오즈비가 높은 챔버"라는 '
        '한 문장에 모르는 말이 세 개 들어 있으면 그 화면 전체가 읽히지 않는다.<br>'
        '<span style="color:#4b5563">각 용어는 <b>쉬운 설명</b>과 '
        '<b>왜 현업에서 중요한가</b>를 함께 담았다. 뜻만 알고 이유를 모르면 오래 남지 않는다.</span></div>',
        unsafe_allow_html=True,
    )

    mobile = layout.is_mobile()
    cols = st.columns(1 if mobile else [3, 2])
    with cols[0]:
        query = st.text_input(
            "검색", placeholder="용어·영문 표기·약어로 찾기 (예: OR, 커미널리티, probe card)",
            label_visibility="collapsed" if not mobile else "visible",
        )
    with cols[-1]:
        category = st.selectbox(
            "분류", ["(전체)"] + list(glossary.categories()),
            label_visibility="collapsed" if not mobile else "visible",
        )

    if query:
        results = glossary.search(query, limit=40)
        st.caption(f"'{query}' 검색 결과 {len(results)}건")
        if not results:
            st.info(
                "찾지 못했습니다. 영문 표기나 약어로도 검색됩니다 — "
                "예: `commonality`, `OR`, `FDC`."
            )
        for term in results:
            _term_card(term, expanded=len(results) <= 3)
        return

    if category != "(전체)":
        terms = glossary.by_category(category)
        st.caption(f"{category} · {len(terms)}개 — {glossary.categories()[category]}")
        for term in terms:
            _term_card(term)
        return

    # ── 전체 보기: 분류별로 접어 둔다 ──────────────────────────────────
    st.divider()
    layout.metric_grid(
        [
            (name, f"{counts.get(name, 0)}개", None)
            for name in glossary.categories()
        ],
        desktop_cols=4,
    )
    st.caption("분류를 고르거나 검색하면 해당 용어만 볼 수 있다.")

    for name, description in glossary.categories().items():
        terms = glossary.by_category(name)
        if not terms:
            continue
        with st.expander(f"{name} ({len(terms)}개) — {description}"):
            for term in terms:
                st.markdown(f"**{term.term}** — {term.short}")


render()
