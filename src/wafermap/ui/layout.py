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


def missing_artifact(exc: Exception, *, what: str = "") -> None:
    """필요한 데이터·아티팩트가 없을 때의 화면을 한 곳에서 그린다.

    왜 공통으로 두나 ★: 화면마다 제각각 문구를 쓰면 같은 상황인데 다르게 보인다.
        실제로 "데이터가 없습니다", "데이터셋이 없습니다", 그리고 예외 메시지를
        그대로 흘리는 것 — 세 가지가 섞여 있었다. 더 나쁜 것은 안내하는 명령이
        틀리는 경우다. 원인 분석 화면에 필요한 것은 `build_rootcause.py`인데
        `build_dataset.py`를 안내하면 사용자는 시키는 대로 하고도 같은 화면을 본다.

    로더가 던지는 예외 메시지에 이미 정확한 명령이 들어 있으므로, 여기서는 그것을
    **파싱해서 명령만 코드 블록으로 떼어 낸다.** 화면이 명령을 따로 적어 두면
    스크립트 이름이 바뀔 때 어긋난다.

    Args:
        exc: 로더/아티팩트가 던진 예외
        what: 이 화면이 무엇을 하려 했는지 (한 줄). 비우면 일반 문구를 쓴다.
    """
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    commands = [line for line in lines if "python " in line]
    reason = next((line for line in lines if line not in commands), str(exc))

    st.warning(
        f"{what or '이 화면'}에 필요한 데이터가 아직 없습니다.\n\n{reason}"
    )
    for line in commands:
        # "먼저 실행하세요: python ..." 형태에서 명령만 떼어 복사하기 쉽게 둔다.
        st.code(line.split(":", 1)[-1].strip() if ":" in line else line, language="bash")
    st.caption(
        "전체 파이프라인 순서는 **도움말·학습** 화면의 단계별 재현 명령에서 볼 수 있습니다."
    )
