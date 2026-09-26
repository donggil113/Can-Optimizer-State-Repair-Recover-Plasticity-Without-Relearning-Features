# P4 사전등록 초안 (파일럿용 · 미실행)

- 작성: 2026-09-26, 스모크 실행 **이전**에 커밋됨 (git 이력으로 확인 가능).
- 상태: `DRAFT_FROZEN_FOR_PILOT` — 파일럿은 아직 승인·실행되지 않았다.
  변경이 필요하면 이 파일을 덮어쓰지 않고 하단 "개정 이력"에 날짜와 사유를 추가한다.
- 이번 단계의 스모크 실행(`configs/p4_smoke.json`)은 러너 검증용이며 아래 가설의 검정이 아니다.

## 1. 연구 질문

Adam으로 학습 중인 신경망에서 과제(task) 전환 이후, **가중치·정규화 버퍼·RNG·데이터 위치는 그대로 두고
optimizer 상태(m, v, bias-correction age)만 수정하는 개입**이

1. keep-all 대비 새 과제 학습을 개선하는가,
2. 그 개선이 update 크기(유효 LR), LR/β 튜닝, 또는 bias-correction 오류로 설명되지 않는가,
3. 이전 과제 망각을 늘리지 않는가,
4. keep-all보다 더 큰 표현(feature) 변화 없이 일어나는가,
5. 개입 시점의 관측 가능한 optimizer 통계로 개입 효과를 예측할 수 있는가.

범위 구분 (중요): 짧은 스트림에서 측정되는 것은 **전환 직후 적응 속도(transient adaptation)** 다.
Dohare et al. (Nature 2024)가 말하는 **장기 plasticity loss**(수백~수천 과제에 걸친 학습능력 저하)와는 다른 현상이다.
"plasticity 회복"이라는 표현은 파일럿에서 keep-all의 새 과제 성능이 과제 순번에 따라 실제로 하락함(plasticity loss 관측)을
먼저 보인 뒤에만 사용한다. 관측되지 않으면 결과는 "전환 적응 속도에 대한 효과"로만 보고한다.

## 2. 기여로 주장하지 않는 것

- optimizer(특히 Adam 상태)가 비정상성 학습에 중요하다는 사실 자체 (선행연구 존재, `PRIOR_ART.md`).
- feature-rank/dead-unit 진단이 plasticity를 완전히 설명하지 못한다는 사실 자체.
- optimizer reset, Adam step-counter reset, β1=β2 설정 자체 (선행연구 존재).
- 기존 알고리즘의 조합(예: reset + LR schedule)을 새로운 방법으로 단정하지 않는다.

## 3. 비교군과 정보 접근

| 비교군 | 가중치 변경 | label | task boundary | 추가 정보 | 파라미터 추가 | 비고 |
|---|---|---|---|---|---|---|
| keep_all | 없음 | 현재 batch | – | – | 0 | paired 기준 |
| reset_m / reset_v / reset_both / mix (decoupled age) | 없음 | 현재 batch | **oracle** (분기 시점) | – | 0 | 이번 단계의 모든 branch arm은 oracle timing |
| *_shared_keep / *_shared_reset | 없음 | 현재 batch | oracle | – | 0 | bias-correction 오류 측정용, 방법 아님 |
| lr_bump | 없음 | 현재 batch | oracle | – | 0 | LR 통제 |
| beta_switch | 없음 | 현재 batch | oracle | – | 0 | memory-length 통제, state-only 아님 |
| keepdir_Xmag / Xdir_keepmag | 없음 | 현재 batch | oracle | **다른 arm의 step별 update norm** | 0 | 크기·방향 분해용 진단, 배치 불가 |
| tuned Adam (LR, LR×β), tuned SGD | 없음 | 현재 batch | 없음 | dev seed 튜닝 | 0 | 전체 실행 통제 |
| periodic reset (주기 P) | 없음 | 현재 batch | 없음 | – | 0 | P는 사전 고정, 과제 길이와 다르게 |
| **제안 방법 (파일럿)**: 관측 기반 trigger + state 개입 | 없음 | 현재 batch | **없음** | calibration seed로 정한 임계값 | 0 | 미래 boundary 미제공 |

주의: periodic의 P를 dev에서 튜닝하면 과제 길이(=boundary 주기)가 누설될 수 있다. 파일럿에서 P는 사전 고정 값
`{64, 128, 256}` 전체를 보고하고, dev 스트림의 과제 길이와 test 스트림의 과제 길이를 다르게 둔 설정을 추가한다.

## 4. 목적함수의 가정

- 학습 목적: 현재 과제 batch의 평균 cross-entropy만 최소화한다. replay, 정규화 항, task ID 입력 없음. 출력 head 공유.
- Adam의 bias correction은 "0으로 초기화된 EMA가 정상(stationary) gradient 분포를 평균한다"는 가정에서 유도된다
  (Kingma & Ba 2015 Sec. 3). 전환 직후 이 가정이 깨진다. 이번 구현은 moment별 age를 분리하여, 개입 후에도
  `1 − β^age`가 해당 moment가 실제로 가진 gradient 가중치와 일치하도록 한다 (decoupled). 이것이 "올바른"
  보정이라는 주장은 정상성 가정 아래에서만 성립한다.
- 평가: held-out 새 과제 정확도(학습 속도), 이전 과제 정확도(망각). 두 목표는 상충할 수 있으며 하나로 합치지 않는다.

## 5. 지표

- **Primary**: `new_task_auc_acc` — 분기 후 창 [b, b+W]의 평가 시점(b 제외)에서 새 과제 held-out 정확도 평균.
- Secondary: `new_task_final_acc`, `forgetting_prev_task`(분기 시점 대비 창 끝의 직전 과제 정확도 하락),
  `forgetting_all_prev_mean`, `cka_to_branch_end`(분기 시점 대비 hidden feature의 linear CKA),
  `w1_rel_change_end`, `online_train_acc_window`, `mean_update_norm_first10`.
- 모든 비교는 같은 seed 안의 paired 차이 `arm − keep_all`.

## 6. 최소관심효과 (MIE)

- 새 과제: `new_task_auc_acc` +0.02 (2%p).
- 망각: `forgetting_prev_task` +0.02 이상 악화를 의미 있는 손해로 본다.
- feature 변화: `cka_to_branch_end` −0.02 이상 감소를 "feature를 더 많이 재학습함"으로 본다.
- 파일럿에서 keep_all paired 차이의 seed 간 표준편차가 MIE보다 크면 MIE를 바꾸지 않고 seed 수를 늘린다.

## 7. 가설과 반증 조건

decoupled state 개입 X ∈ {reset_m, reset_v, reset_both, mix} 각각에 대해:

- **H1 효과**: paired Δ(X − keep_all) of primary ≥ MIE, 95% CI 하한 > 0.
  반증: CI 상한 < MIE.
- **H2 크기로 설명되지 않음**: X와 `keepdir_Xmag`(keep_all 방향 + X의 step별 update norm)의 차이.
  |Δ| < MIE (CI가 ±MIE 안) 이면 효과는 유효 step 크기로 설명된다 → "state repair" 주장 기각, 그렇게 보고.
- **H3 튜닝으로 설명되지 않음**: 동일 튜닝 예산의 최강 통제(tuned Adam LR×β, tuned SGD, lr_bump, beta_switch) 대비
  Δ ≥ MIE가 아니면 기각.
- **H4 bias-correction artifact 아님**: 효과가 shared_keep/shared_reset에서만 나타나고 decoupled에서 사라지면 artifact로 보고.
- **H5 망각**: Δ forgetting ≤ +0.02가 아니면 "회복"이 아니라 "trade-off"로 보고.
- **H6 feature 재학습 없음**: Δ cka_to_branch_end ≥ −0.02가 아니면 제목의 "without relearning features" 주장 기각.
- **H7 비-oracle**: 제안 trigger 버전의 효과가 oracle timing 효과의 절반 미만이면 "oracle에서만 유효"로 보고.
- **H8 예측**: 분기 시점 진단(`g2_over_vhat_median` → reset_v, `cos_mhat_g` → reset_m)과 실제 paired Δ의
  Spearman ρ를 calibration seed에서 적합·test seed에서 평가. ρ의 95% CI가 0을 포함하면 기각.
  분기 시점에는 과제 경계뿐 아니라 과제 중간(전환 없음) 지점을 포함하여 음성 사례를 확보한다.

다중비교: H1은 4개 X에 대해 Holm 보정. 나머지는 기술 통계와 CI로 보고.

## 8. Seed · 분할

- dev (튜닝): 1000–1004. calibration (trigger 임계값, 예측기 적합): 2000–2009. test (보고): 3000–3019 (20 paired seeds).
- test seed는 전부 보고한다. 실패·발산·NOT_RUN 포함. 좋은 seed 선택 금지.
- 분할은 `osrepair/stream.py:SPLITS`로 강제되고, `validate_config`가 test seed 튜닝을 거부한다.

## 9. 튜닝 범위 (실행 전 고정)

- base Adam: LR ∈ {3e-4, 1e-3, 3e-3, 1e-2}, β = (0.9, 0.999).
- Adam LR×β: LR 위 4개 × (β1, β2) ∈ {(0.9, 0.999), (0.9, 0.99), (0.9, 0.9), (0.0, 0.99)} = 16.
- SGD: LR ∈ {0.003, 0.01, 0.03, 0.1, 0.3} × momentum ∈ {0, 0.9} = 10.
- mix ρ ∈ {0.25, 0.5, 0.75}; lr_bump factor ∈ {2, 3, 10} × steps ∈ {10, 50}; beta_switch β2 ∈ {0.99, 0.9}.
- 선택 기준: dev seed 평균 primary, 동률은 grid 앞쪽. 모든 grid 결과를 `tuning.jsonl`에 보존.
- state 개입 계열의 dev 실행 수는 가장 강한 통제 계열의 dev 실행 수를 넘지 않게 한다. 실제 실행 수와 wall-clock을
  manifest에 기록하며, 같은 step 상한을 같은 실지출로 부르지 않는다.

## 10. 자원 상한

- CPU 전용, 단일 프로세스, config당 wall-clock 4시간, 메모리 2 GiB. GPU·다운로드·유료 API 없음.
- torch 백엔드로 옮기는 경우 별도 개정으로 등록한다 (현재 torch 미설치 — STATUS.md blocker).

## 11. 분석

- paired 차이의 평균, 95% paired bootstrap CI(10,000 resample, 고정 seed)와 t-interval을 함께 보고.
- 발산(loss 비유한)은 FAIL로 기록하고 해당 arm의 지표에서 제외하지 않는다 (worst-case 값으로 대체한 분석을 추가 보고).

## 개정 이력

- (없음)
