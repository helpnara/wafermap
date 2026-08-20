# 로컬 환경 구축 — VS Code + GitHub

이 문서 하나만 따라 하면 로컬 PC에서 프로젝트를 실행하고 화면을 띄울 수 있습니다.
**Windows / macOS 둘 다** 다룹니다.

---

## 0. 미리 확인할 것

| 항목 | 필요 버전 | 확인 명령 | 없으면 |
|---|---|---|---|
| Python | **3.11 이상** | `python --version` | [python.org](https://www.python.org/downloads/) |
| Git | 2.x | `git --version` | [git-scm.com](https://git-scm.com/downloads) |
| VS Code | 최신 | — | [code.visualstudio.com](https://code.visualstudio.com/) |

> ⚠️ **Python 3.10 이하는 안 됩니다.** 코드가 `X | None` 형태의 타입 표기와
> 3.11부터 안정화된 문법을 씁니다. 3.12·3.13도 정상 동작합니다.

**Windows 사용자 주의**: python.org 설치 시 첫 화면의
**"Add python.exe to PATH"** 체크박스를 반드시 켜세요. 이걸 놓치면
터미널에서 `python`이 인식되지 않습니다.

---

## 1. GitHub 인증 준비

로컬에서 `git push`를 하려면 인증이 필요합니다. **셋 중 하나**를 고르세요.

### 방법 A — VS Code 내장 인증 (가장 쉬움, 추천)

VS Code를 열고 좌측 하단 **계정 아이콘 → "Sign in with GitHub"**.
브라우저가 열리면 승인하면 끝입니다. 이후 push/pull이 자동으로 인증됩니다.

### 방법 B — GitHub CLI

```bash
# macOS
brew install gh
# Windows
winget install --id GitHub.cli

gh auth login
# → GitHub.com → HTTPS → Login with a web browser 선택
```

### 방법 C — Personal Access Token (PAT)

1. GitHub → 우측 상단 프로필 → **Settings**
2. 좌측 맨 아래 **Developer settings**
3. **Personal access tokens → Tokens (classic) → Generate new token (classic)**
4. **Note**: `wafermap-local` / **Expiration**: 90 days
5. **Scopes**: `repo` 체크 (이것만 있으면 됩니다)
6. **Generate token** → 화면에 나온 문자열을 복사

> 🔑 토큰은 **이 화면을 벗어나면 다시 볼 수 없습니다.** 비밀번호 관리자에 저장하세요.
> 토큰은 비밀번호와 같습니다 — 채팅·이메일·코드에 절대 붙여넣지 마세요.

push할 때 아이디/비밀번호를 물으면 **비밀번호 자리에 이 토큰**을 붙여넣습니다.
매번 묻지 않게 하려면:

```bash
git config --global credential.helper store      # macOS/Linux
git config --global credential.helper manager    # Windows
```

---

## 2. 저장소 클론

### VS Code에서 (마우스로)

1. VS Code 실행 → `Ctrl+Shift+P` (macOS: `Cmd+Shift+P`)
2. `Git: Clone` 입력 후 선택
3. URL 입력: `https://github.com/helpnara/wafermap.git`
4. 저장할 폴더 선택 → 완료되면 **"Open"** 클릭

### 터미널에서

```bash
cd ~/projects            # 원하는 상위 폴더로 이동 (없으면 mkdir -p ~/projects)
git clone https://github.com/helpnara/wafermap.git
cd wafermap
code .                   # VS Code로 열기
```

### 작업 브랜치로 이동 ★

`main`에는 아직 아무것도 없습니다. **개발 내용은 전부 작업 브랜치에 있습니다.**

```bash
git fetch origin
git checkout claude/semiconductor-eds-model-design-1hfzrk
```

VS Code에서는 **좌측 하단 상태바의 브랜치 이름**을 클릭해 목록에서 고르면 됩니다.

확인:

```bash
git log --oneline -5
# 7ee6c0a M5 화면: 개선방안·기대효과 (데스크탑/모바일 반응형)
# 67b23b4 M5: 개선안 도출 · 반사실 시뮬레이션 · ROI 추정
# ...
```

---

## 3. VS Code 확장 설치

프로젝트를 열면 VS Code가 **"이 저장소가 권장하는 확장이 있습니다"** 라고 물어봅니다.
**Install All**을 누르면 끝입니다. (`.vscode/extensions.json`에 정의돼 있습니다.)

저장소에 `.vscode/settings.json`(인터프리터·테스트 설정), `extensions.json`(권장 확장),
`launch.json`(실행 구성)이 커밋돼 있어서 **클론하면 바로 적용**됩니다.

수동으로 설치한다면 `Ctrl+Shift+X` → 아래를 검색:

| 확장 | 왜 필요한가 |
|---|---|
| **Python** (ms-python.python) | 인터프리터 선택, 실행, 디버깅 |
| **Pylance** | 자동완성·타입 검사 |
| **Ruff** (charliermarsh.ruff) | 린터 — 저장할 때 자동 정리 |
| **Even Better TOML** | `.streamlit/config.toml` 편집 |
| **GitLens** | 코드 줄마다 "누가 왜 바꿨나" 표시 — 코드 공부에 유용 |

---

## 4. 가상환경과 패키지 설치

### 4-1. 가상환경 만들기

VS Code 터미널을 엽니다 (``Ctrl+` ``).

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

프롬프트 앞에 `(.venv)`가 붙으면 성공입니다.

> **Windows에서 "이 시스템에서 스크립트를 실행할 수 없으므로" 오류가 나면**
> PowerShell을 관리자로 열고 한 번만 실행:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

### 4-2. VS Code에 인터프리터 알려주기 ★

`Ctrl+Shift+P` → **`Python: Select Interpreter`** →
`./.venv/bin/python` (Windows: `.\.venv\Scripts\python.exe`) 선택.

**이걸 안 하면** VS Code가 시스템 파이썬을 보면서 "모듈을 찾을 수 없음" 밑줄을
계속 긋습니다. 코드는 멀쩡한데 에디터만 빨간 줄이 뜨는 상황이 여기서 나옵니다.

> 🪟 **Windows 사용자**: `.vscode/settings.json`의 `python.defaultInterpreterPath`는
> macOS/Linux 경로(`.venv/bin/python`)로 적혀 있습니다. Windows에서는 위 방법으로
> **직접 `.venv\Scripts\python.exe`를 골라 주세요.** 한 번 고르면 VS Code가
> 워크스페이스에 기억하므로 이후에는 자동입니다.
> (이 설정을 OS별로 분기하는 기능이 VS Code에 없어서, 문서로 안내합니다.)

### 4-3. 패키지 설치

```bash
pip install --upgrade pip
pip install -r requirements-dev.txt
```

약 3~5분 걸립니다.

> 💡 **torch를 CPU 전용으로 줄이기 (권장, 5GB → 200MB)**
>
> `requirements.txt`의 기본 torch는 CUDA 의존성까지 끌어와 약 5GB입니다.
> GPU를 안 쓴다면 먼저 CPU 휠을 설치하세요:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements-dev.txt
> ```
> torch는 **CNN 비교 모델(M2)에만** 쓰입니다. LightGBM 본선 모델과 앱 실행에는
> 필요 없으므로, 급하면 나중에 설치해도 됩니다.

---

## 5. 데이터와 모델 만들기 ★

**저장소에는 데이터가 들어 있지 않습니다.** `.gitignore`가 `data/processed/`를
제외하기 때문입니다. 로컬에서 직접 생성해야 합니다.

> **왜 데이터를 커밋하지 않나**: 18MB짜리 parquet을 매번 커밋하면 저장소 히스토리가
> 급격히 무거워지고, 코드 변경 diff가 바이너리 덩어리에 묻힙니다. 생성 스크립트가
> **시드 고정으로 결정적(deterministic)** 이므로 누가 돌려도 같은 데이터가 나옵니다.
> 데이터 대신 **데이터를 만드는 방법**을 버전 관리하는 것이 옳습니다.

터미널에서 순서대로 실행하세요. (괄호는 이 환경에서 실측한 소요 시간)

```bash
python scripts/build_dataset.py           # 5초    — 웨이퍼 맵 6,000장 + FDC 54,000행
python scripts/build_features.py          # 83초   — 기하·Radon 피처 추출
python scripts/train_pattern_model.py     # 23초   — LightGBM 패턴 분류 모델
python scripts/build_recommendations.py   # 12초   — M3→M4→M5 파이프라인 결과
python scripts/build_rootcause.py         # 40초   — 원인 분석 화면용 결과
```

**총 3분 남짓**입니다. PC 성능에 따라 다소 차이가 납니다.

### 선택 — CNN 비교 모델 (오래 걸림)

```bash
python scripts/train_pattern_cnn.py       # 약 10~15분 (CPU 기준)
```

CNN은 **LightGBM과 비교해 보여주기 위한 대조군**입니다(결과: LightGBM 0.974 vs CNN 0.946).
앱 실행에는 필요 없으니 건너뛰어도 됩니다.

### 잘 만들어졌는지 확인

```bash
ls data/processed/synthetic/
# die_map.npz  excursions.parquet  fdc_summary.parquet  fdc_trace.parquet
# features.parquet  ground_truth.parquet  recommendations.json  wafer_master.parquet
```

---

## 6. 테스트 실행

```bash
pytest -q
# 297 passed
```

**297개가 모두 통과해야 정상입니다.** 하나라도 실패하면 아래 [문제 해결](#9-문제-해결)을 보세요.

VS Code에서 마우스로 돌리려면: 좌측 **플라스크 아이콘(Testing)** →
`Configure Python Tests` → `pytest` → `tests` 폴더 선택 → 재생 버튼.

---

## 7. 앱 실행 ★

```bash
streamlit run app.py
```

브라우저가 자동으로 열리고 `http://localhost:8501` 로 접속됩니다.

### 모바일 화면 확인하기

실기기 없이도 확인할 수 있게 URL 파라미터를 넣어 뒀습니다.

```
http://localhost:8501/?view=mobile     ← 모바일 레이아웃 (2열 KPI, 탭 전환)
http://localhost:8501/?view=desktop    ← 데스크탑 레이아웃 (2열 분할)
```

브라우저 개발자 도구(`F12`) → 좌상단 **기기 툴바 아이콘**(`Ctrl+Shift+M`)으로
iPhone 크기를 골라 보면 실제 모바일과 거의 같습니다.

### 같은 와이파이의 폰에서 열기 ★

**틈틈이 공부하려면 이 경로를 씁니다.** 원래는 클라우드에 배포해 공유 URL로 여는
것이 주 경로였는데 배포를 범위에서 뺐으므로(`00_design.md` §4.3), 폰으로 보는
방법은 이것입니다.

`streamlit run` 실행 시 터미널에 **Network URL**이 표시됩니다
(예: `http://192.168.0.12:8501`). 폰 브라우저에 그 주소를 치면 됩니다.
방화벽이 막으면 PC 방화벽에서 8501 포트를 허용하세요.

PC가 켜져 있고 같은 와이파이에 있어야 합니다. 도움말·학습 화면의 진도는 URL의
`?done=` 파라미터에 저장되므로, 폰에서 보던 주소를 그대로 북마크해 두면 이어서 볼 수 있습니다.

### 종료

터미널에서 `Ctrl+C`.

### VS Code 실행 버튼으로 돌리기

좌측 **실행 및 디버그**(`Ctrl+Shift+D`) → 상단 드롭다운에서 고르고 ▶ 클릭:

| 구성 | 하는 일 |
|---|---|
| **Streamlit 앱 실행** | 앱을 디버거에 붙여 실행 (중단점 사용 가능) |
| **현재 파일 실행** | 열려 있는 파일을 그대로 실행 |
| **M5 개선안 (Donut 상세)** | 개선안 도출 과정을 콘솔에서 단계별로 확인 |
| **M4 원인 규명 (Donut 상세)** | SHAP 원인 규명 과정을 확인 |

`justMyCode: false`로 설정해 두어서 **라이브러리 내부까지 들어가며 디버깅**할 수
있습니다. 코드를 공부할 때 `src/wafermap/analysis/recommend.py`의
`counterfactual()` 같은 함수에 중단점을 찍고 M5 구성을 실행해 보면,
값이 어떻게 흘러가는지 눈으로 따라갈 수 있습니다.

---

## 8. 코드 수정하고 GitHub에 올리기

### 8-1. 반드시 작업 브랜치에서

```bash
git branch --show-current
# claude/semiconductor-eds-model-design-1hfzrk   ← 이게 나와야 함
```

### 8-2. 커밋과 푸시

**VS Code에서 (마우스)**
1. 좌측 **소스 제어 아이콘**(`Ctrl+Shift+G`)
2. 변경된 파일 옆 **+** 를 눌러 스테이징
3. 위쪽 입력창에 커밋 메시지 작성 → **✓ Commit**
4. **Sync Changes** 버튼 클릭

**터미널에서**
```bash
git add -A
git commit -m "무엇을 왜 바꿨는지 한 줄"
git push -u origin claude/semiconductor-eds-model-design-1hfzrk
```

### 8-3. 원격 변경사항 받아오기

제가 이 세션에서 작업을 이어가면 원격 브랜치가 앞서 나갑니다. 로컬에서 작업을
시작하기 **전에** 항상 먼저 받아오세요.

```bash
git pull origin claude/semiconductor-eds-model-design-1hfzrk
```

> ⚠️ 양쪽에서 같은 파일을 동시에 고치면 충돌이 납니다.
> **로컬에서 실험하실 땐 별도 브랜치를 파시는 것을 권합니다:**
> ```bash
> git checkout -b my-experiment
> ```
> 이러면 제 작업과 섞이지 않고, 마음껏 고쳐 보다가 버려도 됩니다.

---

## 9. 문제 해결

### `ModuleNotFoundError: No module named 'wafermap'`

가상환경이 활성화되지 않았거나 VS Code가 다른 인터프리터를 보고 있습니다.

```bash
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -c "import sys; print(sys.executable)"
# 경로에 .venv 가 들어 있어야 정상
```

들어 있는데도 안 되면 `Ctrl+Shift+P` → `Python: Select Interpreter` 재선택 →
VS Code 재시작(`Ctrl+Shift+P` → `Developer: Reload Window`).

### `❌ 데이터가 없습니다. python scripts/build_dataset.py …`

[5단계](#5-데이터와-모델-만들기-)를 건너뛰었습니다. 순서대로 다시 실행하세요.

### `FileNotFoundError: recommendations.json 가 없습니다`

M5 아티팩트만 빠진 경우입니다.

```bash
python scripts/build_recommendations.py
```

### `streamlit: command not found`

가상환경이 꺼져 있습니다. 활성화하거나 이렇게 실행하세요:

```bash
python -m streamlit run app.py
```

### 앱은 뜨는데 화면이 비어 있음 / "Connection error"

터미널의 오류 메시지를 보세요. 대개 데이터가 없거나 파이썬 버전이 낮은 경우입니다.

```bash
python --version    # 3.11 이상인지
```

### torch 설치가 너무 느리거나 실패

CNN은 선택 사항입니다. `requirements.txt`에서 `torch>=2.0` 줄을 주석 처리하고
설치한 뒤, 나중에 필요할 때 CPU 휠로 따로 설치하세요.

### 한글이 깨져 보임 (Windows)

VS Code 터미널에서:
```powershell
chcp 65001
```
또는 VS Code 설정에서 터미널 기본 프로필을 **PowerShell**로 지정하세요.

### 테스트 일부 실패 — `shap 미설치 시 건너뜀`

`skipped`로 표시되면 정상입니다. `shap`이 없으면 M4·M5 테스트가 자동으로 건너뜁니다.
전부 돌리려면 `pip install shap`.

---

## 10. 폴더 구조 빠르게 훑기

```
wafermap/
├─ app.py                      # Streamlit 진입점 — 여기서 시작
├─ views/                      # 화면 (페이지) — 현재 improvement.py 1개
├─ src/wafermap/
│  ├─ config.py                # ★ 모든 설정의 단일 출처 (공정·파라미터·규격)
│  ├─ data/                    # 데이터 생성·로딩 (M1)
│  ├─ features/                # 웨이퍼 맵 피처 추출 (M2)
│  ├─ models/                  # 패턴 분류·원인 규명 모델 (M2, M4)
│  ├─ analysis/                # SPC·커미널리티·기여도·개선안 (M3~M5)
│  ├─ learning/                # 용어사전
│  └─ ui/                      # 화면 공통 (테마·레이아웃·아티팩트 로더)
├─ scripts/                    # 실행 스크립트 (전부 --help 지원)
├─ tests/                      # 297개 테스트
└─ docs/
   ├─ 00_design.md             # 설계서
   ├─ 04_results.md            # ★ 분석 결과와 발견한 문제들
   ├─ 09_roadmap.md            # ★ 남은 작업
   └─ learning_notes/          # ★ 모듈별 코드 해설
```

### 코드 공부는 어디서 시작하나

1. **`docs/05_learning_guide.md`** — 읽는 순서가 정리돼 있습니다
2. **`docs/learning_notes/`** — 모듈마다 "무엇을 / 어떻게 / 왜" + 확인 문제
3. 스크립트를 `--help`와 함께 돌려 보고, 출력의 해석 문구를 읽어 보세요

```bash
python scripts/run_root_cause.py --pattern Donut      # M4 원인 규명 상세
python scripts/run_recommend.py --pattern Donut       # M5 개선안 상세
python scripts/glossary.py --search 커미널리티          # 용어사전
```

> 학습노트의 코드 인용은 **테스트로 강제되어 실제 소스와 항상 일치**합니다
> (`tests/test_learning_notes.py`). 노트가 옛 코드를 설명하는 일은 생기지 않습니다.
