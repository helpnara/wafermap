"""EDS Wafer Map 분석 시스템 — Streamlit 진입점 (설계서 §4).

실행: streamlit run app.py

화면 확인용 URL 파라미터
    ?view=mobile   모바일 레이아웃 강제
    ?nav=rail      내비게이션을 아이콘만 남기는 모드로
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
    # auto = 넓은 화면에서는 펼치고 좁은 화면에서는 접는다.
    # collapsed로 고정하면 데스크탑에서도 메뉴가 숨어 이동 방법이 안 보인다 (§4A.1)
    initial_sidebar_state="auto",
)

from wafermap.ui import layout  # noqa: E402

#: (파일, 제목, 아이콘) — rail 모드에서는 제목을 아이콘으로 대체한다
PAGE_SPECS = [
    ("views/overview.py", "개요", "🏠"),
    ("views/explorer.py", "웨이퍼맵 탐색", "🗺️"),
    ("views/detection.py", "이상공정 탐지", "📉"),
    ("views/rootcause.py", "원인 분석", "🔬"),
    ("views/improvement.py", "개선방안·기대효과", "🎯"),
    ("views/glossary.py", "용어사전", "📖"),
    ("views/learning.py", "도움말·학습", "🎓"),
]

mode = layout.nav_mode()

pages = [
    st.Page(
        path,
        # rail 모드는 아이콘만 남겨 본문 폭을 확보한다. Streamlit이 제목을 반드시
        # 요구하므로 제목 자리에 아이콘을 넣고 icon은 비운다.
        title=icon if mode == "rail" else title,
        icon=None if mode == "rail" else icon,
        default=(index == 0),
    )
    for index, (path, title, icon) in enumerate(PAGE_SPECS)
]

st.navigation(pages, position="top" if mode == "top" else "sidebar").run()
