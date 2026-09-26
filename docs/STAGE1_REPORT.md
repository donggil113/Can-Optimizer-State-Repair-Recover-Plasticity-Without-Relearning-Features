# P4 Stage 1 결과 보고 (2026-09-26)

**판정: `STATE_REPAIR_BRANCH_ON_HOLD`** (두 조건 모두). 범위: 합성 조건 2개, oracle timing, test seed 3개.
탐색적 진단이며 확증 실험이 아니다. seed 3000–3002는 스모크에서 이미 열람한 개발 자료다.

- 실행: `OMP_NUM_THREADS=1 python3 -m osrepair.stage1 --config configs/p4_stage1.json --out runs/p4_stage1_0404c14`
- 결과 위치: `runs/p4_stage1_0404c14/` (`manifest.json`, `summary.md`, `log.jsonl.gz`, `config.json`)
- 코드 commit `0404c14`. code_sha256 `b45b507e…`는 HEAD 코드와 일치한다. config_sha256 `d9c2b7ac…`
- 설정은 실행 **전에** 커밋했다. `dirty=True`는 실행 중 생긴 run 산출물과 test log뿐이다.
- 자원: CPU 2,078 s / 상한 3,600 s. wall 2,097 s. peak RSS 29 MiB.
  - OS 한도: RLIMIT_AS 3 GiB, RLIMIT_CPU 3,660 s, thread 1.
- cell 72개 모두 PASS: 튜닝 48, 현상 18, 개입 6. FAIL/NOT_RUN 없음.

## 1. 구현 의미 확인 (torch reference)

`.venv`에 CPU torch 2.14.0을 설치했다 (29.4 s, `runs/install/`). 비교 조건은 float64, 고정 β, weight_decay=0,
amsgrad=False, foreach/fused/capturable=False이며, ε는 √v̂ 뒤에 더한다.

vanilla Adam 400 step의 최대 상대 차이:

| 대상 | weight | m | v | update | step counter |
|---|---|---|---|---|---|
| 최대 상대 차이 | 1.7e-14 | 1.4e-13 | 3.9e-16 | 3.6e-11 | 일치 |

MLP forward, gradient, BN buffer도 torch와 일치했다 (`tests/test_torch_parity.py`,
`runs/p4_stage1/torch_parity_measured.txt`).

stock torch 조작과 같은 것 (trace로 확인):

| 우리 개입 | stock torch 조작 |
|---|---|
| `reset_both/decoupled` | 새 torch Adam |
| `reset_t` | `step`만 0 |
| `*_shared_keep` | 해당 `exp_avg*`만 0 |
| `reset_m/shared_reset` | `exp_avg`와 `step`을 0 |

stock 조작과 **같지 않은** 것: `reset_m/decoupled`, `reset_v/decoupled`, `mix`, `beta_switch`. 이들은 moment별 age를
쓰는 custom 규칙이며 명시 수식 테스트로만 검증했다.

## 2. 현상: 비교 가능한 새 과제에서 학습 능력 저하

protocol:
- 50개 과제를 학습한 스트림에서, 한 번도 학습하지 않은 고정 probe 과제를 쓴다.
- 세 시작점 fresh / early / late가 **같은 batch, 같은 dropout mask, 같은 200 step**으로 probe 과제를 학습한다.
  - fresh: 같은 초기 가중치 + 새 optimizer
  - early: 과제 0 직후 (optimizer 상태 유지)
  - late: 과제 49 직후 (optimizer 상태 유지)
- 과제 난이도는 late − early 비교에서 상쇄된다.

튜닝:
- dev seed 1000–1001, 목적은 스트림 전체의 online 정확도.
- 여섯 family 모두 grid 내부에서 선택됐다 (RESOLVED, 확장 없음).
- Adam lr 3e-3 (β 0.9/0.999), β1=β2=0.9 Adam lr 3e-3, SGD(momentum 0.9) lr 0.01.

base Adam, probe learning-curve AUC (seed 3개 평균):

| 조건 | fresh | early | late | late − early (seed별) | 판정 |
|---|---|---|---|---|---|
| random_teacher, C=4 | 0.788 | 0.727 | 0.483 | −0.245 (−0.198, −0.221, −0.315) | ESTABLISHED |
| label_permutation, C=10 | 0.568 | 0.536 | 0.332 | −0.204 (−0.178, −0.282, −0.151) | ESTABLISHED |

- 초기 50 step 정확도와 최종 정확도도 late가 낮다 (최종 기준 −0.18 / −0.20).
- 튜닝한 β1=β2 Adam과 SGD에서도 late − early가 −0.28에서 −0.41로, 더 크게 나타난다 (기술적 관찰).
- late checkpoint의 fc1 가중치 norm은 약 3배다 (9.9 vs 3.5). label_permutation에서는 dead unit이 34–59%다.
  이 진단값들은 기술적 관찰일 뿐 원인으로 단정하지 않는다.

## 3. 같은 late checkpoint에서의 state-only 개입

paired Δ AUC, 평균 [seed별]:

| 개입 | vs keep_all (RT) | vs norm 일치 통제 (RT) | vs keep_all (LP) | vs norm 일치 통제 (LP) | 판정 |
|---|---|---|---|---|---|
| reset_m | +0.021 [+.015 +.032 +.015] | +0.012 | +0.020 [+.027 +.020 +.012] | +0.008 | RT: 크기로 설명, LP: 효과 < MIE |
| reset_v | −0.152 | −0.020 | −0.244 | −0.048 | 양의 효과 없음 |
| reset_both (= 새 torch Adam) | −0.008 | −0.027 | −0.043 | −0.030 | 양의 효과 없음 |
| reset_t (Adam-Rel) | −0.207 | 0.000 | −0.203 | 0.000 | 양의 효과 없음 |
| *오보정* reset_both/shared_keep | **+0.333** | – | **+0.252** | – | artifact 대조군 |
| *오보정* reset_v/shared_keep | +0.063 (−0.16~+0.31) | – | −0.241 | – | artifact 대조군 |

RT = random_teacher, LP = label_permutation.

해석:

1. **새 optimizer 상태로 late 가중치를 학습시켜도 저하가 회복되지 않는다** (reset_both ≈ keep_all). 이 설정의
   저하는 stale optimizer 상태만으로 설명되지 않는다.
2. **reset_m**만 RT에서 사전 기준(평균 ≥ 0.02이고 모든 seed > 0)을 경계선에서 넘는다.
   - norm 일치 통제와의 차이는 +0.012로 MIE보다 작다. 따라서 update 크기로 설명된다.
   - 잔차는 세 seed 모두 양수(+0.010~+0.015)다. seed 3개로는 0이라고도, 동등하다고도 말할 수 없다.
3. **reset_t와 norm 일치 통제의 차이가 정확히 0인 것은 구조적 항등식이다.**
   - m, v를 유지하므로 update가 tensor마다 스칼라 `√(1−β2^k)/(1−β1^k)`배가 될 뿐이다. 즉 per-tensor LR schedule이다.
   - 첫 step의 비율이 실측 0.3162 = √(1−β2)/(1−β1)였다. `TestResetTIsPerTensorRescaling`으로 고정했다.
4. **올바르게 보정한 reset_v는 수치적으로 파국적이다.**
   - m은 유지하고 v̂만 표본 1개의 g²로 바뀌므로, 현재 gradient가 0에 가까운 좌표에서 m̂/√v̂가 폭증한다.
   - 첫 10 step update norm이 RT에서 16, LP에서 330이다 (keep_all은 0.03).
   - "보정이 정확하다"와 "추정량이 쓸 만하다"는 다른 문제다.
5. **가장 큰 이득은 오보정 artifact에서 나왔다.**
   - reset_both/shared_keep는 초기 update가 약 6–8배로 커지고, 가중치를 더 많이 움직이며
     (θ 변화 0.86 vs 0.38), CKA도 더 낮다 (0.47 vs 0.70).
   - 결과는 late의 저하를 넘어 fresh 수준에 이른다 (RT 0.816).
   - 이 설정의 저하가 **유효 step 크기에 매우 민감하다**는 기술적 관찰이다. 이 arm에 대한 LR 일치 통제는 이번 설정에
     없었으므로 원인은 검정하지 않았다.
6. 개입 이후에는 모든 arm에서 가중치가 갱신되고 CKA도 바뀐다. 따라서 **"feature를 재학습하지 않는다"는 주장은 하지 않는다.**
   CKA와 가중치 이동은 기술 통계일 뿐이다.

## 4. 판정 근거와 한계

- 사전 규칙상 올바르게 보정한 개입 가운데 `DIFFERENCE_REMAINS`인 것이 없다. 따라서
  `STATE_REPAIR_BRANCH_ON_HOLD`이며, 현재 state-repair 분기의 추가 학습을 보류한다.
- 한계:
  - seed 3개, 합성 조건 2개, 작은 MLP(H=32), oracle timing이다.
  - probe 과제는 조건마다 1개다. 과제·checkpoint를 독립 seed로 세지 않았다.
  - 튜닝 목적(online 정확도)과 probe 지표(AUC)가 다르다.
  - 비유의나 차이 없음은 동등성이 아니다.
- 새 trigger, 예측기, reset 규칙, bootstrap 확대, 실제 데이터는 하지 않았다.

---

## 정정 이력 (2026-09-26 추가, 위 본문은 원문 그대로 보존)

- 위 3절의 "reset_t와 norm 일치 통제의 차이가 정확히 0인 것은 구조적 항등식"과 "실측 0.3162 = √(1−β2)/(1−β1)"은
  **과도한 표현**이었다.
  - 같은 (m, v)에서 counter만 바꾼 update 비율은 κ(r,t)·(√v+ε√b_t)/(√v+ε√b_r)이다. 모든 좌표에 같은 scalar가
    되는 것은 **ε=0이고 v>0일 때뿐**이다.
  - 0.3162는 r=1, t→∞의 근사다. 유한 age에서는 κ(1,10001)=0.316235이다.
  - ε=1e-8에서는 좌표별 배율이 달라진다. 결정적 반례: 0.316 vs 0.613.
- 보고된 "0"은 probe **정확도의 정확한 동률**이다. 연속량은 최대 6.5e-8 차이가 나서 bitwise 동일이 아니다.
- 근거: `runs/p4_analysis_counter/result.json`, 원고 `paper/` App. B, C, G.
