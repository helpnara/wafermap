#!/usr/bin/env python3
"""용어사전 터미널 조회 — Streamlit 앱 없이도 바로 찾아볼 수 있다.

사용법:
    python scripts/glossary.py                    # 전체 목록 (분류별)
    python scripts/glossary.py 커미널리티            # 검색
    python scripts/glossary.py EDS --full          # 전체 설명 보기
    python scripts/glossary.py --category 통계      # 분류별 목록
    python scripts/glossary.py --categories        # 분류 목록
    python scripts/glossary.py --stats             # 규모 요약
"""

from __future__ import annotations

import argparse
import shutil
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wafermap.learning import glossary as G  # noqa: E402

WIDTH = min(shutil.get_terminal_size((90, 20)).columns, 92)


def _wrap(text: str, indent: str = "  ") -> str:
    """마크다운 강조를 걷어 내고 터미널 폭에 맞춰 줄바꿈한다."""
    plain = text.replace("**", "").replace("> ", "")
    blocks = []
    for block in plain.split("\n\n"):
        block = " ".join(line.strip() for line in block.splitlines() if line.strip())
        if block:
            blocks.append(textwrap.fill(block, width=WIDTH - len(indent),
                                        initial_indent=indent, subsequent_indent=indent))
    return "\n\n".join(blocks)


def print_term(term: G.Term, *, full: bool = False) -> None:
    """용어 하나를 출력한다."""
    print(f"\n{'━' * WIDTH}")
    print(f"  {term.term}   [{term.category}]")
    print(f"{'━' * WIDTH}")
    print(_wrap(term.short))

    print(f"\n  ── 쉽게 말하면 {'─' * (WIDTH - 18)}")
    print(_wrap(term.plain))

    if term.why_matters:
        print(f"\n  ── 왜 중요한가 {'─' * (WIDTH - 18)}")
        print(_wrap(term.why_matters))

    if full and term.detail:
        print(f"\n  ── 더 정확히는 {'─' * (WIDTH - 18)}")
        print(_wrap(term.detail))

    if term.aliases:
        print(f"\n  다른 이름: {', '.join(term.aliases)}")
    if term.related:
        print(f"  관련 용어: {', '.join(term.related)}")
    if full and term.used_in:
        print(f"  등장 위치: {', '.join(term.used_in)}")


def print_index() -> None:
    """분류별 용어 목록을 한눈에 보여 준다."""
    print(f"\n{'=' * WIDTH}")
    print(f"  용어사전 — 총 {len(G.all_terms())}개")
    print(f"{'=' * WIDTH}")
    for category, desc in G.categories().items():
        terms = G.by_category(category)
        print(f"\n  [{category}] {desc}  ({len(terms)}개)")
        line = "    "
        for term in terms:
            if len(line) + len(term.term) + 3 > WIDTH:
                print(line)
                line = "    "
            line += f"{term.term} · "
        print(line.rstrip(" ·"))
    print(f"\n  자세히 보려면: python scripts/glossary.py <용어>")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="반도체·통계·머신러닝 용어사전",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("query", nargs="*", help="검색할 용어")
    parser.add_argument("--full", action="store_true", help="상세 설명까지 출력")
    parser.add_argument("--category", help="특정 분류의 용어만 출력")
    parser.add_argument("--categories", action="store_true", help="분류 목록 출력")
    parser.add_argument("--stats", action="store_true", help="사전 규모 요약")
    args = parser.parse_args()

    if args.categories:
        print()
        for category, desc in G.categories().items():
            print(f"  {category:<8} {desc}")
        return 0

    if args.stats:
        print()
        for name, count in G.stats().items():
            print(f"  {name:<10} {count:>4}개")
        return 0

    if args.category:
        try:
            terms = G.by_category(args.category)
        except KeyError as exc:
            print(f"❌ {exc}", file=sys.stderr)
            return 1
        for term in terms:
            print_term(term, full=args.full)
        return 0

    if not args.query:
        print_index()
        return 0

    query = " ".join(args.query)
    results = G.search(query)

    if not results:
        print(f"\n  '{query}'에 해당하는 용어를 찾지 못했습니다.")
        print(f"  전체 목록: python scripts/glossary.py")
        return 1

    # 정확히 하나면 상세히, 여러 개면 목록 먼저
    if len(results) == 1 or G.get(query) is not None:
        print_term(G.get(query) or results[0], full=True)
        if len(results) > 1:
            others = [t.term for t in results if t.term != (G.get(query) or results[0]).term]
            print(f"\n  관련 검색 결과: {', '.join(others)}")
    else:
        print(f"\n  '{query}' 검색 결과 {len(results)}건:")
        for term in results:
            print(f"    · {term.term:<18} [{term.category}] {term.short}")
        print(f"\n  자세히 보려면: python scripts/glossary.py <용어>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
