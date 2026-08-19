"""EDS Wafer Map 분석 시스템 — Streamlit 진입점 (설계서 §4).

실행: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# src 레이아웃을 쓰므로 설치 없이도 임포트되도록 경로를 넣는다
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="EDS Wafer Map 분석",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",  # 모바일에서 표시 영역을 넓게 (§4A.1)
)

from wafermap.ui import layout  # noqa: E402

PAGES = [
    st.Page("views/improvement.py", title="개선방안·기대효과", icon="🎯", default=True),
]

# 모바일에서는 사이드바 대신 상단 바 — 화면 폭을 최대한 쓴다 (§4A.1)
position = "top" if layout.is_mobile() else "sidebar"
st.navigation(PAGES, position=position).run()
