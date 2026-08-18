---
module: src/wafermap/data/synth_wafer.py
milestone: M1
difficulty: 초급
concepts: [확률장, 극좌표, 베르누이 시행, 넘파이 브로드캐스팅, 마스킹, 재현성(시드)]
prerequisites: []
estimated_minutes: 15
---

# 합성 wafer map 생성기

## 이 코드는 무엇을 하나

**입력**: 불량 패턴 이름(`"Edge-Ring"` 등), 난수 생성기, 심각도(severity)
**출력**: `(34, 67)` 크기의 2차원 정수 배열 — `0`=die 없음, `1`=pass, `2`=fail

즉 "Edge-Ring 웨이퍼 한 장 그려 줘"라고 하면 실제 EDS 테스트 결과처럼 보이는
웨이퍼 맵을 만들어 준다. WM-811K 실측 데이터와 **완전히 같은 형식**이라, 나중에
실데이터로 바꿔도 뒤쪽 코드는 아무것도 고칠 필요가 없다.

---

## 핵심 로직

### 1단계: 웨이퍼 기하 만들기 (`build_geometry`)

동그란 웨이퍼 위에 네모난 die를 격자로 깐다. 여기서 **가장 중요한 결정**이 하나 있다.

```python
corner_r = np.sqrt(
    (np.abs(x_mm) + die_width_mm / 2.0) ** 2
    + (np.abs(y_mm) + die_height_mm / 2.0) ** 2
)
mask = corner_r <= r_usable
```

die 중심이 아니라 **die의 네 모서리 중 가장 먼 지점**이 반경 안에 있어야 유효로 친다.
중심만 보면 웨이퍼 경계에 반쯤 걸친 die까지 살아남는데, 실제 팹은 완전한 die만
노광하고 측정하기 때문이다.

> 💡 `np.abs(x_mm) + die_width_mm/2` 는 "원점에서 가장 먼 모서리"를 구하는 요령이다.
> 부호를 절댓값으로 없앤 뒤 반쪽 크기를 더하면, 어느 사분면에 있든 항상 바깥쪽 모서리가 나온다.

### 2단계: 물리 좌표로 극좌표 계산

```python
r_norm = np.where(mask, r_center / r_usable, np.nan)
theta = np.where(mask, np.arctan2(y_mm, x_mm), np.nan)
```

`r_norm`은 0(중심)~1(최외곽)로 정규화된 반경, `theta`는 각도다.
Center/Donut/Edge-Ring 같은 패턴은 전부 이 `r_norm` 하나로 그려진다.

### 3단계: 패턴별 확률장 그리기

각 패턴은 "이 위치의 die가 불량일 확률"을 계산하는 함수 하나로 표현된다.

| 패턴 | 수식 | 의미 |
|---|---|---|
| Center | `A·exp(-(r/w)²)` | 중심에서 최대, 멀어지면 감소 |
| Donut | `A·exp(-((r-r₀)/w)²)` | 반경 `r₀`에서 최대 = 링 모양 |
| Edge-Ring | `A / (1 + exp(-k(r-r_th)))` | 시그모이드 — 임계 반경 밖에서 급상승 |
| Random | `상수` | 위치와 무관 |

Donut이 Center의 수식에서 `r` 대신 `(r - r₀)`만 바뀐 것에 주목하자.
**"최댓값의 위치를 중심에서 특정 반경으로 옮긴다"** 는 한 줄 차이가 완전히 다른 패턴을 만든다.

### 4단계: 확률 → 실제 pass/fail

```python
baseline = rng.uniform(*BASELINE_FAIL_RATE)
prob = np.clip(np.nan_to_num(prob, nan=0.0) + baseline, 0.0, 0.995)

draw = rng.random(geom.shape)
wafer[geom.mask] = np.where(draw[geom.mask] < prob[geom.mask], DIE_FAIL, DIE_PASS)
```

`draw < prob` 가 **베르누이 시행**이다. 0~1 균등난수를 뽑아 확률보다 작으면 불량.
확률이 0.7이면 70% 확률로 불량이 된다.

`baseline`을 더하는 이유도 중요하다. 정상 웨이퍼도 die 몇 개는 떨어진다.
이게 없으면 `none` 클래스가 전부 완벽한 웨이퍼가 되어 분류 문제가 비현실적으로 쉬워진다.

---

## 왜 이렇게 만들었나 — 설계 선택과 대안

### 선택 1: 왜 이미지를 직접 그리지 않고 "확률장"을 거치나?

| 방식 | 결과 |
|---|---|
| 결정론적으로 직접 그리기 | 같은 패턴은 항상 똑같은 모양 → 모델이 통째로 외워 버림 |
| **확률장 + 베르누이 (채택)** | 같은 Edge-Ring이어도 매번 미묘하게 다름 → 실제 데이터와 닮음 |

실제 웨이퍼는 같은 원인이어도 완전히 똑같이 나오지 않는다. 확률을 거치면 그 산포가 공짜로 생긴다.

### 선택 2: 왜 인덱스가 아니라 물리 좌표(mm)를 쓰나? ★ 가장 중요

가상 DRAM die는 4.5mm × 9.0mm로 **가로세로 비율이 2:1**이다.
행/열 인덱스 거리로 원을 그리면 이렇게 된다.

```
인덱스 기준 (틀림)              물리 좌표 기준 (맞음)
    ┌─────────┐                    ┌─────────┐
    │  ▁▃▅▇▅▃▁ │  ← 세로로 눌린      │  ▁▃▅▇▅▃▁ │  ← 진짜 원
    │ ▃█████▃  │     타원            │ ▃█████▃  │
    └─────────┘                    └─────────┘
```

Edge-Ring을 만들었는데 위아래만 두껍고 좌우는 얇은, 물리적으로 존재할 수 없는
모양이 나온다. 반경 기반 패턴이 전부 틀어지므로 `x_mm`, `y_mm`를 계산해서 쓴다.

### 선택 3: 왜 `rng`를 인자로 받나?

```python
def generate_wafer_map(
    pattern: str,
    rng: np.random.Generator,
    geom: WaferGeometry | None = None,
    severity: float = 1.0,
) -> np.ndarray:
```

`np.random.seed()`를 함수 안에서 부르지 않고 호출자가 `rng`를 넘긴다.
전역 시드를 쓰면 다른 코드가 난수를 뽑는 순간 결과가 달라져서, "어제와 같은
데이터셋"을 다시 만들 수 없게 된다. 실험 재현성의 기본이다.

### 선택 4: 왜 Edge-Loc이 Edge-Ring을 재사용하나?

```python
def _prob_edge_loc(geom: WaferGeometry, rng: np.random.Generator, sev: float) -> np.ndarray:
    ring = _prob_edge_ring(geom, rng, sev)
    ...
    return ring * angular
```

먼저 Edge-Ring 링을 만들고(`ring`), 각도 가중치를 곱해(`* angular`) 원주의 일부만 남긴다.

두 패턴은 물리적으로 같은 계열(엣지 발생)이고, 차이는 **원주 전체냐 일부냐**뿐이다.
그리고 실제로 이 둘은 분류기가 가장 자주 혼동하는 쌍이다. 이 구조로 생성해야
그 혼동이 데이터에도 자연스럽게 재현되어, 모델 성능이 현실적으로 나온다.

---

## 확인 문제

**1.** `_prob_donut`에서 `r0`를 0.0으로 바꾸면 어떤 패턴이 되는가?

<details><summary>정답</summary>

`exp(-((r - 0)/w)²)` = `exp(-(r/w)²)` 가 되어 **Center 패턴과 같아진다.**
Donut과 Center는 "확률이 최대가 되는 반경"만 다른 같은 수식이다.
</details>

**2.** `test_geometry_is_symmetric`은 마스크의 상하·좌우 대칭을 검사한다. 이 테스트가 잡아낼 수 있는 버그는?

<details><summary>정답</summary>

die 격자를 웨이퍼 중심에 정렬하는 계산이 틀린 경우다. 예를 들어
`(np.arange(n_cols) - (n_cols - 1) / 2.0)` 에서 `- 1`을 빼먹으면 격자가 반 칸
치우쳐서 좌우 비대칭이 된다. 웨이퍼는 원형이므로 대칭이 깨지면 반드시 버그다.
</details>

**3.** `severity=2.0`을 넣으면 불량률이 정확히 2배가 되는가?

<details><summary>정답</summary>

**아니다.** `amp = np.clip(rng.uniform(...) * sev, 0.0, 0.97)` 에서 `clip`이 상한을
0.97로 자른다. 원래 amp가 0.8이었다면 2배는 1.6이지만 0.97로 잘린다.
확률은 1을 넘을 수 없으므로 당연한 처리이며, 그래서 severity와 불량률의 관계는
비례가 아니라 **단조 증가(포화 있음)** 이다.
</details>

---

## 더 알아보기

- **베르누이 시행 / 이항분포**: 확률 p로 성공/실패가 갈리는 가장 기본적인 확률 모형
- **`np.where`와 마스킹**: numpy에서 조건부 값 선택의 표준 관용구
- **브로드캐스팅**: `geom.x_mm[..., None] - path_x` (스크래치 거리 계산)에서
  `(34,67,1)`과 `(N,)`이 자동으로 `(34,67,N)`이 되는 규칙. numpy 성능의 핵심
- 관련 모듈: [`fdc_simulator.md`](fdc_simulator.md) — 이 맵에 붙일 공정 데이터를 만든다
