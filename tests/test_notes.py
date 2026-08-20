"""학습노트 읽기 모듈 검증 (M6-5).

핵심 질문:
  1. **frontmatter를 제대로 읽는가?** — 화면이 난이도·시간·선수 노트를 쓴다
  2. **읽기 순서가 선수 관계를 지키는가?** — 순서가 틀리면 학습이 안 된다
  3. **확인 문제와 정답을 분리하는가?** — 화면에서 접이식으로 만들어야 한다
"""

from __future__ import annotations

import pytest

from wafermap.learning import notes


def test_all_notes_load():
    loaded = notes.all_notes()
    assert loaded, "학습노트를 하나도 읽지 못했다"
    for note in loaded:
        assert note.title, f"{note.slug}: 제목이 없다"
        assert note.milestone, f"{note.slug}: 마일스톤이 없다"
        assert note.module, f"{note.slug}: 대상 모듈이 없다"


def test_frontmatter_is_stripped_from_body():
    """frontmatter가 본문에 남으면 화면에 `module: ...` 이 그대로 찍힌다."""
    note = notes.load("spc")
    assert not note.body.startswith("---")
    assert "estimated_minutes" not in note.body


def test_title_is_not_duplicated_in_body():
    """제목을 본문에서 떼어 내야 화면에 두 번 나오지 않는다."""
    note = notes.load("spc")
    assert not note.body.lstrip().startswith("# ")


def test_required_sections_are_parsed():
    note = notes.load("spc")
    for name in notes.REQUIRED_SECTIONS:
        assert note.section(name), f"'{name}' 섹션을 못 읽었다"


def test_missing_note_raises():
    with pytest.raises(FileNotFoundError):
        notes.load("nope")


# ── 확인 문제 ────────────────────────────────────────────────────────────


def test_quiz_splits_question_and_answer():
    note = notes.load("spc")
    quiz = note.quiz
    assert len(quiz) >= 2
    for question, answer in quiz:
        assert question, "문제가 비어 있다"
        assert answer, "정답이 비어 있다"
        assert "<details>" not in question
        assert "<details>" not in answer


def test_quiz_count_matches_every_note():
    """모든 노트가 확인 문제를 갖춰야 한다 (§5.3)."""
    for note in notes.all_notes():
        assert len(note.quiz) >= 2, f"{note.slug}: 확인 문제가 {len(note.quiz)}개뿐"


# ── 읽기 순서 ★ ─────────────────────────────────────────────────────────


def test_reading_order_respects_prerequisites():
    """선수 노트가 반드시 앞에 와야 한다 ★.

    왜 중요한가: 순서를 건너뛰면 앞 노트에서 설명한 개념을 모른 채 다음 노트를
        만난다. 학습 화면의 존재 이유가 사라진다.
    """
    order = notes.reading_order()
    position = {note.slug: i for i, note in enumerate(order)}
    for note in order:
        for prerequisite in note.prerequisites:
            if prerequisite in position:
                assert position[prerequisite] < position[note.slug], (
                    f"{note.slug} 가 선수 노트 {prerequisite} 보다 앞에 있다"
                )


def test_reading_order_includes_every_note():
    assert len(notes.reading_order()) == len(notes.all_notes())


def test_prerequisites_point_to_real_notes():
    """없는 노트를 선수로 지정하면 순서가 조용히 깨진다."""
    slugs = {note.slug for note in notes.all_notes()}
    for note in notes.all_notes():
        for prerequisite in note.prerequisites:
            assert prerequisite in slugs, f"{note.slug}: 없는 선수 노트 {prerequisite}"


# ── 소스 연결 ────────────────────────────────────────────────────────────


def test_source_code_is_readable():
    """화면이 원본 코드를 함께 보여주려면 실제로 읽혀야 한다."""
    note = notes.load("spc")
    sources = notes.source_code(note)
    assert sources
    for path, code in sources.items():
        assert code.strip(), f"{path}: 내용이 비었다"


def test_multi_module_note_reports_all_sources():
    """`also_covers`가 있는 노트는 보조 모듈까지 돌려줘야 한다."""
    note = notes.load("attribution")
    assert len(note.modules) > 1
    assert len(notes.source_code(note)) == len(note.modules)


def test_by_milestone_groups_all():
    grouped = notes.by_milestone()
    assert sum(len(v) for v in grouped.values()) == len(notes.all_notes())
