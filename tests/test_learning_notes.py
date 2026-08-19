"""학습노트가 실제 코드와 어긋나지 않는지 검사한다 (설계서 §5.5 드리프트 방지).

왜 이 테스트가 필요한가: 학습노트는 코드를 설명하는 문서다. 코드를 리팩터링했는데
노트가 옛 코드를 그대로 인용하고 있으면, 공부하는 사람이 **존재하지 않는 코드를
배우게 된다.** 문서와 코드의 불일치는 조용히 벌어지고 아무도 알려주지 않으므로,
테스트로 강제하는 것 외에는 막을 방법이 없다.

검사 내용:
  1. frontmatter의 `module` 경로가 실제로 존재하는가
  2. ```python 블록에 인용된 코드가 실제 소스에 **글자 그대로** 있는가
  3. §5.3 템플릿의 필수 섹션을 갖추었는가
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTES_DIR = PROJECT_ROOT / "docs" / "learning_notes"

#: §5.3에서 정한 학습노트 필수 섹션
REQUIRED_SECTIONS = ("## 이 코드는 무엇을 하나", "## 핵심 로직", "## 왜 이렇게 만들었나", "## 확인 문제")

#: 인용 블록에서 검사를 건너뛸 줄 (생략 표시, 주석, 의사코드)
_SKIP_PREFIXES = ("#", "...", ">>>")


def _note_files() -> list[Path]:
    return sorted(NOTES_DIR.glob("*.md"))


def _frontmatter(text: str) -> dict[str, str]:
    """--- 로 감싼 YAML frontmatter를 간단히 파싱한다 (의존성 없이)."""
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not match:
        return {}
    out = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip()
    return out


def _python_blocks(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, re.S)


def test_learning_notes_exist():
    assert _note_files(), "학습노트가 하나도 없습니다"


def _module_paths(meta: dict[str, str]) -> list[Path]:
    """노트가 다루는 소스 파일 목록.

    하나의 학습노트가 여러 모듈을 함께 설명하는 경우가 있다(예: M4는 모델과
    해석 모듈을 같이 다룬다). `module`은 대표 모듈, `also_covers`는 함께 인용하는
    보조 모듈을 쉼표로 나열한다.
    """
    paths = [PROJECT_ROOT / meta["module"]]
    for extra in meta.get("also_covers", "").split(","):
        extra = extra.strip().strip("[]")
        if extra:
            paths.append(PROJECT_ROOT / extra)
    return paths


@pytest.mark.parametrize("note", _note_files(), ids=lambda p: p.name)
def test_frontmatter_module_exists(note: Path):
    """frontmatter가 가리키는 소스 파일이 실제로 존재해야 한다."""
    meta = _frontmatter(note.read_text(encoding="utf-8"))
    assert meta, f"{note.name}: frontmatter가 없습니다"
    assert "module" in meta, f"{note.name}: module 필드가 없습니다"
    assert "milestone" in meta, f"{note.name}: milestone 필드가 없습니다"

    for path in _module_paths(meta):
        assert path.exists(), f"{note.name}: 없는 모듈을 가리킴 → {path}"


@pytest.mark.parametrize("note", _note_files(), ids=lambda p: p.name)
def test_required_sections_present(note: Path):
    """§5.3 템플릿의 필수 섹션을 갖추어야 한다."""
    text = note.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in text, f"{note.name}: '{section}' 섹션이 없습니다"


@pytest.mark.parametrize("note", _note_files(), ids=lambda p: p.name)
def test_quoted_code_matches_source(note: Path):
    """인용된 파이썬 코드가 실제 소스에 글자 그대로 존재해야 한다.

    ⚠️ 이 테스트가 실패하면, **노트를 소스에 맞춰 고쳐라.** 반대로 하지 말 것.
       소스가 진실의 원천이고 노트는 그 설명이다.
    """
    text = note.read_text(encoding="utf-8")
    meta = _frontmatter(text)
    source = "\n".join(p.read_text(encoding="utf-8") for p in _module_paths(meta))

    mismatches: list[str] = []
    for block in _python_blocks(text):
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(_SKIP_PREFIXES):
                continue
            if stripped not in source:
                mismatches.append(stripped)

    assert not mismatches, (
        f"{note.name}: 소스에 없는 코드를 인용하고 있습니다 "
        f"(노트를 소스에 맞춰 수정하세요)\n  - " + "\n  - ".join(mismatches[:8])
    )


@pytest.mark.parametrize("note", _note_files(), ids=lambda p: p.name)
def test_self_check_questions_have_answers(note: Path):
    """확인 문제에는 접이식 정답이 달려 있어야 한다."""
    text = note.read_text(encoding="utf-8")
    section = text.split("## 확인 문제", 1)[1]
    n_questions = len(re.findall(r"^\*\*\d+\.\*\*", section, re.M))
    n_answers = section.count("<details>")

    assert n_questions >= 2, f"{note.name}: 확인 문제가 2개 미만입니다"
    assert n_answers == n_questions, (
        f"{note.name}: 문제 {n_questions}개 vs 정답 {n_answers}개 — 개수가 맞지 않습니다"
    )
