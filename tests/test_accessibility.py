"""접근성 — 색 대비와 '색만으로 정보를 주지 않는가' (설계서 §4.8).

왜 테스트로 두나 ★: 접근성은 한 번 점검하고 끝나는 것이 아니다. 새 화면에서 색을
    하나 고르는 순간 조용히 깨진다. 눈으로는 "잘 보이는데?" 싶은 색이 실제로는
    기준 미달인 경우가 많아서(WARN #d97706 이 3.19:1 이었다) 숫자로 못 박아 둔다.
"""

from __future__ import annotations

import pytest

from wafermap.ui import theme

WHITE = "#ffffff"
#: 표 머리 배경 — 데이터 사전에서 쓴다
TABLE_HEAD_BG = "#f3f4f6"

#: WCAG 2.1 AA. 본문 글자 4.5:1, 큰 글자와 그래픽 요소 3:1.
AA_TEXT = 4.5
AA_GRAPHIC = 3.0


def _relative_luminance(hex_color: str) -> float:
    raw = hex_color.lstrip("#")
    channels = []
    for start in (0, 2, 4):
        value = int(raw[start:start + 2], 16) / 255
        channels.append(
            value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
        )
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast(foreground: str, background: str) -> float:
    """WCAG 대비비. 1(같은 색) ~ 21(검정 대 흰색)."""
    a, b = _relative_luminance(foreground), _relative_luminance(background)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


def test_contrast_helper_matches_known_values():
    """계산식 자체가 맞는지 — 검정/흰색은 정확히 21:1 이다."""
    assert contrast("#000000", "#ffffff") == pytest.approx(21.0, abs=0.01)
    assert contrast("#ffffff", "#ffffff") == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("color", [theme.GOOD, theme.BAD, theme.MUTED,
                                   theme.WARN_TEXT, theme.MID_TEXT])
def test_text_colors_meet_aa_on_white(color):
    assert contrast(color, WHITE) >= AA_TEXT, f"{color} 는 흰 배경에서 본문 기준 미달"


def test_table_head_text_meets_aa_on_its_own_background():
    assert contrast(theme.TABLE_HEAD_TEXT, TABLE_HEAD_BG) >= AA_TEXT


def test_fill_color_is_not_used_as_text():
    """WARN 은 채우기 전용이다.

    글자로 쓰면 3.19:1 이라 미달인데, 막대 색으로는 필요하다. 두 쓰임을 색으로
    갈라 두었으므로 WARN 이 본문 기준을 넘지 '않는' 것이 오히려 정상이다.
    이 테스트는 누가 WARN 을 글자에 쓰려다 기준을 낮추는 것을 막는다.
    """
    assert contrast(theme.WARN, WHITE) >= AA_GRAPHIC   # 막대로는 충분하고
    assert contrast(theme.WARN, WHITE) < AA_TEXT       # 글자로는 부족하다
    assert contrast(theme.WARN_TEXT, WHITE) >= AA_TEXT  # 그래서 글자용이 따로 있다


def test_good_and_warn_are_distinguishable_without_hue():
    """적록 색약을 넘어, 흑백 인쇄에서도 구분되도록 명도 차이를 요구한다."""
    good = _relative_luminance(theme.GOOD)
    warn = _relative_luminance(theme.WARN)
    assert abs(good - warn) > 0.03, "파랑과 주황의 명도가 너무 비슷하다"


def test_evidence_strength_carries_a_symbol_not_only_color():
    """근거 강도는 색 배지로 보여 주지만, 기호가 항상 함께 붙어야 한다."""
    for label in ("강함", "보통", "—", "약함 — SHAP 단독"):
        color, symbol = theme.evidence_style(label)
        assert symbol.strip(), f"{label} 에 기호가 없다"
        assert set(symbol) <= {"●", "○"}, f"{label} 의 기호가 예상 밖이다: {symbol}"
        assert contrast(color, WHITE) >= AA_TEXT, f"{label} 배지 색이 본문 기준 미달"
