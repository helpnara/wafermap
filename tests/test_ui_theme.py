"""표시 계층의 문구 변환 — 화면에 글자가 사라지거나 별표가 새는 사고를 막는다."""

from __future__ import annotations

from wafermap.ui import theme


def test_bold_becomes_tag():
    assert theme.html("이것은 **중요**하다") == "이것은 <b>중요</b>하다"


def test_backtick_becomes_code():
    assert theme.html("`scripts/run.py` 를 돌려라") == "<code>scripts/run.py</code> 를 돌려라"


def test_angle_brackets_survive():
    """`<lot_id>` 가 태그로 먹혀 통째로 사라진 사고가 실제로 있었다."""
    out = theme.html("`<lot_id>-W<슬롯>` 형식")
    assert "lot_id" in out and "슬롯" in out
    assert "<lot_id>" not in out  # 살아 있되 태그가 아닌 문자로


def test_ampersand_is_escaped():
    assert theme.html("A & B") == "A &amp; B"


def test_plain_text_is_unchanged():
    assert theme.html("그냥 문장이다") == "그냥 문장이다"
