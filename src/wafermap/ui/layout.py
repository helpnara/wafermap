"""반응형 레이아웃 도우미 (설계서 §4A).

왜 필요한가: Streamlit의 `st.columns`는 모바일에서도 가로 분할을 유지해 각 칸이
    손톱만 해진다. 좁은 화면에서는 **열을 탭으로 바꿔야** 읽을 수 있다.
    이 모듈은 그 판단을 한 곳에 모아 화면마다 다르게 처리되는 일을 막는다.
"""

from __future__ import annotations

from contextlib import contextmanager

import streamlit as st

#: 이 폭 미만이면 모바일 레이아웃 (설계서 §4A.1 브레이크포인트)
MOBILE_MAX_PX = 640

#: 내비게이션 3-모드 (설계서 §4A.1)
#:   sidebar — 넓은 화면. 메뉴 이름을 모두 보여준다
#:   rail    — 중간 화면. 아이콘만 남겨 본문 폭을 확보한다
#:   top     — 좁은 화면. 사이드바를 없애고 상단 바로 올린다
NAV_MODES = ("sidebar", "rail", "top")


def nav_mode() -> str:
    """어떤 내비게이션 모드로 그릴지 결정한다.

    왜 3단계인가: 사이드바는 넓은 화면에서는 편하지만 좁아지면 본문을 잡아먹는다.
        그렇다고 바로 상단 바로 보내면 태블릿 폭에서 메뉴가 한 줄에 안 들어간다.
        중간에 **아이콘만 남기는 단계**를 두면 본문 폭을 지키면서 이동은 유지된다.

    URL 파라미터 `?nav=rail` 로 강제할 수 있다. 실기기 없이 확인하려면 필요하다.
    """
    forced = st.query_params.get("nav")
    if forced in NAV_MODES:
        return forced
    if is_mobile():
        return "top"
    return "sidebar"


def is_mobile() -> bool:
    """모바일 폭인지 판단한다.

    어떻게: Streamlit은 서버 사이드라 화면 폭을 직접 알 수 없다. 그래서
        ① URL 쿼리 파라미터(`?view=mobile`)를 우선 보고,
        ② 없으면 User-Agent로 추정한다.

    왜 쿼리 파라미터를 먼저 보나: 데스크탑에서도 모바일 레이아웃을 **확인**할 수
        있어야 한다. 실기기 없이 검증할 방법이 없으면 모바일 화면은 방치된다.
    """
    view = st.query_params.get("view")
    if view in ("mobile", "desktop"):
        return view == "mobile"

    try:
        agent = str(st.context.headers.get("User-Agent", "")).lower()
    except Exception:  # 헤더를 못 읽는 환경(테스트 등)
        return False
    return any(k in agent for k in ("mobile", "android", "iphone", "ipad"))


@contextmanager
def responsive_split(labels: tuple[str, str], *, ratio: tuple[float, float] = (1.0, 1.0)):
    """데스크탑은 2열, 모바일은 2탭으로 나눈다.

    사용:
        with responsive_split(("조치안", "기대효과")) as (left, right):
            with left: ...
            with right: ...
    """
    if is_mobile():
        tabs = st.tabs(list(labels))
        yield (tabs[0], tabs[1])
    else:
        cols = st.columns(list(ratio), gap="large")
        yield (cols[0], cols[1])


def _escape(text: str) -> str:
    return (
        str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def metric_grid(items: list[tuple[str, str, str | None]], *, desktop_cols: int = 4) -> None:
    """지표 카드를 화면 폭에 맞춰 배치한다 (모바일 2열, 데스크탑은 지정한 열 수).

    왜 `st.metric`을 안 쓰나 ★: Streamlit의 컬럼은 좁은 화면에서 **자동으로 1열로
        무너진다.** 지표 4개가 세로로 늘어서면 모바일에서 첫 화면에 본문이 하나도
        안 들어온다. 격자를 직접 그려야 2열을 유지할 수 있다. 그리고 `st.metric`은
        긴 값을 말줄임표로 잘라버려 "불량 19 / 정…" 처럼 정보가 사라진다.
    """
    n_cols = 2 if is_mobile() else desktop_cols
    cells = []
    for label, value, help_text in items:
        hint = f'<div class="h">{_escape(help_text)}</div>' if help_text else ""
        cells.append(
            f'<div><div class="k">{_escape(label)}</div>'
            f'<div class="v">{_escape(value)}</div>{hint}</div>'
        )
    st.markdown(
        f'<div class="wm-kpi" style="--wm-cols:{n_cols}">{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )
