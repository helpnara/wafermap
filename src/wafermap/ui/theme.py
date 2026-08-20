"""화면 공통 스타일과 표시 규칙 (설계서 §4.8 접근성).

왜 색을 코드로 고정하나: 반도체 불량 분석 화면은 "좋다/나쁘다"를 색으로 말한다.
    적록 색약(남성 약 8%)에게 빨강-초록 조합은 구분이 안 된다. 그래서 이 프로젝트는
    **파랑-주황** 축을 쓰고, 색만으로 정보를 전달하지 않도록 항상 기호나 숫자를 함께 둔다.
"""

from __future__ import annotations

import re

# 파랑(양호) ↔ 주황(위험) — 적록 색약에서도 구분되는 축
GOOD = "#1f6feb"
WARN = "#d97706"
BAD = "#c2410c"
MUTED = "#6b7280"

# 채우기용 색과 글자용 색을 나눈다 ★
#   WCAG는 그래픽(막대·선)에 3:1, 본문 글자에 4.5:1을 요구한다. 기준이 다르므로
#   같은 "주황"을 둘 다에 쓰면 한쪽이 반드시 어긋난다. 실제로 WARN(#d97706)은
#   흰 배경에서 3.19:1 — 막대로는 충분하지만 글자로는 미달이었다.
#   그래서 차트 채우기는 WARN을, 글자·배지는 아래 값을 쓴다.
WARN_TEXT = "#9a3412"   # 흰 배경 7.31:1
MID_TEXT = "#2563eb"    # 흰 배경 5.17:1 (연한 파랑 #3b82f6은 3.68:1로 미달이었다)
TABLE_HEAD_TEXT = "#4b5563"  # 표 머리 배경(#f3f4f6) 위 6.87:1
SURFACE = "#f4f6fa"
BORDER = "#dfe3ea"

#: 근거 강도 → (색, 기호). 색만으로 판단하지 않도록 기호를 함께 준다.
EVIDENCE_STYLE: dict[str, tuple[str, str]] = {
    "강함": (GOOD, "●●●"),
    "보통": (MID_TEXT, "●●○"),
    "—": (MUTED, "○○○"),
}


def evidence_style(label: str) -> tuple[str, str]:
    """근거 강도 라벨에 맞는 (색, 기호)를 돌려준다."""
    for key, style in EVIDENCE_STYLE.items():
        if label.startswith(key):
            return style
    return (WARN_TEXT, "●○○")  # "약함 — …"


CSS = f"""
<style>
  /* 여백 축소 — 분석 화면은 정보 밀도가 중요하다 */
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }}

  .wm-card {{
    border: 1px solid {BORDER}; border-radius: 10px; padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem; background: #fff;
  }}
  .wm-card h4 {{ margin: 0 0 .35rem 0; font-size: 1.02rem; line-height: 1.4; }}
  .wm-row {{ font-size: .87rem; color: #374151; margin: .18rem 0; }}
  .wm-row b {{ color: #111827; }}
  .wm-badge {{
    display: inline-block; padding: .12rem .5rem; border-radius: 999px;
    font-size: .75rem; font-weight: 600; margin-right: .35rem;
  }}
  .wm-note {{
    border-left: 3px solid {WARN}; background: #fffbeb; padding: .55rem .85rem;
    font-size: .84rem; border-radius: 0 6px 6px 0; margin: .35rem 0;
  }}
  .wm-why {{
    border-left: 3px solid {GOOD}; background: #eff6ff; padding: .6rem .9rem;
    font-size: .87rem; border-radius: 0 6px 6px 0; margin: .5rem 0 .9rem 0;
  }}
  .wm-assume {{ font-size: .73rem; color: {MUTED}; margin: -.5rem 0 .5rem 0; }}

  /* KPI 그리드 — st.metric은 좁은 화면에서 1열로 무너지므로 직접 그린다 */
  /* 데이터 사전 — st.dataframe은 셀을 줄바꿈하지 않고 잘라 버린다.
     여기서는 설명문 자체가 내용이라 잘리면 안 되므로 직접 표를 그린다. */
  .wm-dict {{ width: 100%; border-collapse: collapse; font-size: .84rem; }}
  .wm-dict th {{
    text-align: left; padding: .4rem .6rem; background: #f3f4f6;
    color: {TABLE_HEAD_TEXT}; font-weight: 600; font-size: .78rem; white-space: nowrap;
  }}
  .wm-dict td {{
    padding: .45rem .6rem; border-top: 1px solid #e5e7eb;
    vertical-align: top; line-height: 1.5;
  }}
  .wm-dict td.n {{ white-space: nowrap; font-family: ui-monospace, monospace; color: #111827; }}
  .wm-dict td.s {{ white-space: nowrap; color: {MUTED}; font-size: .78rem; }}
  .wm-dict tr:hover td {{ background: #fafafa; }}
  @media (max-width: 640px) {{
    /* 좁은 화면에서는 규격 열을 접고 이름과 설명만 남긴다 */
    .wm-dict th:nth-child(4), .wm-dict td:nth-child(4) {{ display: none; }}
  }}
  .wm-kpi {{
    display: grid; grid-template-columns: repeat(var(--wm-cols, 4), 1fr);
    gap: .55rem; margin: .2rem 0 .9rem 0;
  }}
  .wm-kpi > div {{
    border: 1px solid {BORDER}; border-radius: 9px; padding: .55rem .7rem;
    background: {SURFACE}; min-width: 0;
  }}
  .wm-kpi .k {{ font-size: .74rem; color: {MUTED}; margin-bottom: .18rem; }}
  .wm-kpi .v {{
    font-size: 1.28rem; font-weight: 700; color: #111827; line-height: 1.25;
    overflow-wrap: anywhere;
  }}
  .wm-kpi .h {{ font-size: .7rem; color: {MUTED}; margin-top: .2rem; line-height: 1.35; }}

  /* 모바일 — 제목을 줄이고 여백을 좁혀 첫 화면에 내용이 들어오게 한다 */
  @media (max-width: 640px) {{
    .block-container {{ padding-top: 1rem; padding-left: .7rem; padding-right: .7rem; }}
    h1 {{ font-size: 1.5rem !important; line-height: 1.3 !important; }}
    h2, .stSubheader {{ font-size: 1.15rem !important; }}
    .wm-card {{ padding: .75rem .85rem; }}
    .wm-row {{ font-size: .88rem; }}
    .wm-kpi .v {{ font-size: 1.05rem; }}
  }}
</style>
"""


def badge(text: str, color: str, *, bg: str | None = None) -> str:
    """색 배지 HTML."""
    background = bg or f"{color}18"
    return f'<span class="wm-badge" style="color:{color};background:{background}">{text}</span>'


def html(text: str) -> str:
    """평문 문구를 `unsafe_allow_html` 블록 안에 넣을 수 있는 HTML로 바꾼다.

    세 가지를 한다:
        1. `<`, `>`, `&` 를 이스케이프한다. 안 하면 `<lot_id>` 같은 문구가 브라우저에
           태그로 먹혀 통째로 사라진다 — 데이터 사전에서 실제로 그랬다.
        2. 마크다운 굵게(`**`)를 `<b>` 로 바꾼다.
        3. 백틱 코드(`` ` ``)를 `<code>` 로 바꾼다.

    왜 필요한가: 분석 모듈이 만드는 문구는 콘솔·문서에서도 그대로 쓰이므로 마크다운
        으로 강조를 넣는다. 그런데 `unsafe_allow_html` 로 감싼 div 안에서는 마크다운이
        해석되지 않아 별표와 백틱이 그대로 노출된다. 변환은 표시 계층의 책임이다.
    """
    out = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
