#!/usr/bin/env python3
"""WM-811K 원본(LSWMD.pkl) 다운로드 도우미 — **사용자 로컬 PC에서 실행한다.**

무엇을: Kaggle API로 WM-811K 데이터셋을 받아 `data/raw/LSWMD.pkl`에 배치한다.

왜 로컬에서만: 이 프로젝트의 개발 컨테이너는 네트워크 정책상 pypi/npm만 허용되고
        Kaggle은 차단되어 있다(§2.3). 또한 원본이 약 2GB라 클라우드 환경에 두기에도
        부담이 크다. 그래서 개발은 합성 데이터로 진행하고, 실측 검증만 로컬에서 한다.

사용법:
    1) Kaggle 계정에서 API 토큰 발급 → ~/.kaggle/kaggle.json 배치
       (Kaggle → Account → Create New API Token)
    2) pip install kaggle
    3) python scripts/download_wm811k.py

    자동 다운로드가 막히면 --manual 로 수동 안내를 확인한다.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wafermap.config import RAW_DIR, WM811K_FILENAME  # noqa: E402

KAGGLE_DATASET = "qingyi/wm811k-wafer-map"

MANUAL_GUIDE = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 WM-811K 수동 준비 안내
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1. 아래 중 한 곳에서 데이터셋을 내려받습니다.
      · Kaggle : https://www.kaggle.com/datasets/{KAGGLE_DATASET}
      · MIR Lab: http://mirlab.org/dataSet/public/   (원 배포처)

 2. 압축을 풀면 나오는 `{WM811K_FILENAME}` 파일(약 2GB)을 아래 경로에 둡니다.
      {RAW_DIR / WM811K_FILENAME}

 3. 준비 확인:
      python -c "import sys; sys.path.insert(0,'src'); \\
                 from wafermap.data import wm811k; print(wm811k.is_available())"

 4. 실측 기준으로 데이터셋 생성:
      python scripts/build_dataset.py --source real
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def _kaggle_available() -> bool:
    """kaggle CLI와 인증 파일이 모두 준비되었는지 확인한다."""
    if shutil.which("kaggle") is None:
        return False
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def download() -> Path:
    """Kaggle에서 데이터셋을 받아 data/raw에 배치한다.

    Returns:
        배치된 LSWMD.pkl 경로

    Raises:
        RuntimeError: kaggle CLI/인증이 없거나 다운로드에 실패한 경우
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / WM811K_FILENAME

    if target.exists():
        size_gb = target.stat().st_size / 1024**3
        print(f"✅ 이미 준비되어 있습니다: {target} ({size_gb:.2f} GB)")
        return target

    if not _kaggle_available():
        raise RuntimeError(
            "kaggle CLI 또는 ~/.kaggle/kaggle.json 이 없습니다.\n"
            "  pip install kaggle 후 API 토큰을 배치하거나, --manual 안내를 따르세요."
        )

    print(f"⬇️  Kaggle에서 다운로드 중: {KAGGLE_DATASET}  (약 2GB, 수 분 소요)")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(RAW_DIR)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle 다운로드 실패:\n{result.stderr}")

    # 받은 zip에서 LSWMD.pkl만 꺼낸다
    for zip_path in RAW_DIR.glob("*.zip"):
        print(f"📦 압축 해제: {zip_path.name}")
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.endswith(WM811K_FILENAME):
                    with zf.open(name) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    break
        zip_path.unlink()

    if not target.exists():
        raise RuntimeError(
            f"압축 안에서 {WM811K_FILENAME}을 찾지 못했습니다. --manual 안내를 따르세요."
        )

    size_gb = target.stat().st_size / 1024**3
    print(f"✅ 완료: {target} ({size_gb:.2f} GB)")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="WM-811K 원본 다운로드")
    parser.add_argument(
        "--manual", action="store_true", help="자동 다운로드 없이 수동 준비 안내만 출력"
    )
    args = parser.parse_args()

    if args.manual:
        print(MANUAL_GUIDE)
        return 0

    try:
        download()
    except RuntimeError as exc:
        print(f"❌ {exc}\n{MANUAL_GUIDE}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
