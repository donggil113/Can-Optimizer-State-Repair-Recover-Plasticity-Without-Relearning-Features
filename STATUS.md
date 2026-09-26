# STATUS — P4 (최종 갱신 2026-09-26, Stage 1 이후)

## 판정

| 구분 | 상태 |
|---|---|
| 소프트웨어 | **TECHNICAL_TEST_PASS**: unittest 89개. `.venv`에서는 89/89 통과·skip 0. 시스템 Python에서는 torch parity 5개가 SKIP(PASS 아님) |
| torch parity | **PASS**: MLP forward/grad/BN, vanilla Adam weight/m/v/update/step trace. 최대 상대 차이 ≤ 3.6e-11 |
| 과학 (Stage 1) | **STATE_REPAIR_BRANCH_ON_HOLD**: 두 합성 조건에서 학습 능력 저하는 관측(ESTABLISHED)됐다. 올바르게 보정한 state 개입은 keep_all 대비 양의 효과가 없거나, update 크기(norm 일치 통제)로 설명됐다. 범위: 합성 2조건, oracle timing, seed 3 (이미 열람한 개발 seed) |
| 파일럿 준비 | **NOT_READY_FOR_PILOT** (유지) |

Stage 1 상세는 [`docs/STAGE1_REPORT.md`](docs/STAGE1_REPORT.md), 원자료는 `runs/p4_stage1_0404c14/`.

## Stage 1 요약 (2026-09-26)

- 질문: 비교 가능한 새 과제에서 학습 능력 저하가 실제로 있는가? 있다면 올바른 optimizer-state 개입의 효과가
  update 크기 효과와 구분되는가?
- 설정 `configs/p4_stage1.json`은 실행 전에 커밋했다 (commit `0404c14`).
  - 조건 2개: random_teacher C=4, label_permutation C=10. C=10은 4-class 순열 반복 문제 때문이다.
  - 과제 50개 × 200 step, 고정 probe 과제.
  - 시작점 fresh / early / late가 같은 batch, mask, 예산을 쓴다.
  - dev 튜닝, 경계 확장은 1회만 허용.
- 결과 (probe AUC late − early):
  - RT −0.245, LP −0.204. 모든 seed 음수 → ESTABLISHED.
  - late 가중치에 새 Adam 상태를 줘도 회복되지 않는다 (reset_both ≈ keep_all).
  - reset_m은 RT에서 +0.021이지만, norm 일치 통제와의 차이는 +0.012 < MIE → update 크기로 설명.
  - reset_t는 per-tensor LR 재조정과 같다 (norm 일치 통제와 정확히 같음, 구조적).
  - 올바르게 보정한 reset_v는 m̂/√v̂가 폭증해 성능이 크게 나빠졌다.
  - 가장 큰 이득은 오보정 artifact(reset_both/shared_keep, +0.33/+0.25)였다. update가 6–8배로 커지며,
    저하가 유효 step 크기에 민감하다는 기술적 관찰이다 (원인은 검정하지 않음).
- 개입 이후 가중치가 갱신되므로 "without relearning features"를 주장하지 않는다.
- 이 저장소의 제목 질문은 현재 설정에서 **지지되지 않았다**. 다만 이는 "state repair 불가능"이 아니다
  (seed 3, 합성, oracle).

## (이전) 1단계 기록 — 2026-09-26 초판

## 이번 단계에서 한 일 (지시 1–6 대응)

1. **Paired checkpoint branching runner.** `trainer.snapshot/restore/branch`가 다음을 모두 복제한다: 파라미터,
   BatchNorm `running_mean/var/num_batches_tracked`, optimizer 상태와 hyperparameter, 데이터 RNG, dropout RNG,
   스트림 위치. 복원한 상태끼리 서로 독립인지(aliasing 없음) 테스트한다.
2. **개입 분리.** `keep_all`, `reset_m`, `reset_v`, `reset_both`, `mix`(ρ_m, ρ_v, 기준 상태 지정 가능)를 따로 구현했다.
   선행연구 대조 뒤 `reset_t`(Adam-Rel)를 추가했다.
3. **개입 직후 동일성 검사.** `verify_state_only`가 가중치, 버퍼, RNG, step, hyperparameter, eval/train forward
   출력(같은 dropout mask 포함)을 bitwise로 비교한다. 음성 대조 6종에서 검증기가 변경을 실제로 잡아내는지 확인했다:
   가중치 축소, 1e-15 수준의 가중치 변경, 버퍼 변경, dropout/데이터 RNG 진행, step 변경.
4. **Adam step counter와 bias correction.** moment별 age(`t_m`, `t_v`)를 두었다. 테스트한 항목:
   - Algorithm 1 독립 전사와의 일치, 손으로 계산한 2-step 값, 상수 gradient의 closed form.
   - `decoupled` reset/mix 후 m̂=g, v̂=g²가 정확히 유지됨.
   - `shared_keep`/`shared_reset`의 오보정 closed form. 예: v만 reset하고 counter를 유지하면 첫 update가
     500-step 이력에서 19.9배, 40,000-step 이력에서 1/√(1−β2)=31.6배.
   - Ellis et al. Thm 3.1 / Eq. 4의 t=0 극한.
   - mix의 age 공식과 수치 포화 경계.
5. **짧은 supervised stream.** batch와 RNG를 맞춘 상태에서 learning curve를 기록한다: 매 step 학습 loss/acc와
   update norm, 10 step마다 모든 이전·현재 과제의 held-out 정확도와 CKA/dead-unit/가중치 변화.
6. **통제 연결.** 튜닝 가능한 통제는 dev seed에서 튜닝한다: Adam LR, Adam LR×β, SGD(momentum). 이 밖에 LR bump,
   β1=β2 전환, update-norm 일치(방향과 크기 교차), 비-oracle periodic reset, base Adam 처음부터 재실행
   (keep_all과 bitwise 동일해야 함)을 연결했다.

## 정확한 실행 명령과 실제 결과

```bash
python3 -W error::ResourceWarning -m unittest discover -s tests -t . -v
python3 -m osrepair.cli run --config configs/p4_smoke.json --out runs --run-id p4_smoke_6e4cdaf
```

- 테스트: `runs/test_logs/unittest_<commit>.log`. 73 tests OK, 약 3초, peak RSS 약 25 MB.
- 스모크: `runs/p4_smoke_6e4cdaf/`.
  - 코드는 commit `6e4cdaf`이며 code_sha256 `913bdb85…`가 HEAD 코드와 일치한다. `dirty=True`는 실행 시작 시점에
    추적되지 않던 `runs/`만 가리킨다 (manifest의 `status_porcelain`).
  - config sha256 `d1457796…`
  - 실측: wall 284.7 s, CPU 282.1 s, peak RSS 26.8 MiB. 이 중 튜닝 32 run이 104 s.
  - 환경: Python 3.11.15, Linux, 4 CPU (단일 프로세스 사용).
  - 데이터/모델 hash는 `summary.json`에 있다: seed별 stream fingerprint, 분기 checkpoint의 부분별 hash, arm별
    최종 hash. 전역 `random` 상태는 변하지 않았다.

### 스모크 수치 (기술적 관찰만, 증거 아님: seed 3개, 과제 4개, 합성 데이터)

paired Δ(arm − keep_all), `new_task_auc_acc`, 평균 [최소, 최대]:

| arm | Δ AUC | 초기 10 step update norm (keep_all ≈ 0.14) | Δ CKA (feature 변화) |
|---|---|---|---|
| reset_m (decoupled) | +0.034 [+0.031, +0.040] | +0.16 (약 2.1배) | −0.007 |
| reset_v (decoupled) | −0.043 [−0.068, −0.017] | +0.003 | +0.044 |
| reset_both (decoupled) | −0.001 [−0.024, +0.010] | +0.027 | +0.030 |
| mix 0.5 | +0.006 | +0.032 | −0.001 |
| reset_v, shared_keep (오보정) | +0.078 [+0.052, +0.098] | +4.06 (약 29배) | −0.126 |
| reset_both, shared_keep (오보정) | +0.110 [+0.078, +0.133] | +0.41 (약 3.9배) | −0.135 |
| reset_m, shared_reset (오보정) | −0.219 | −0.10 | +0.208 |
| LR ×3 for 20 steps | +0.048 [+0.043, +0.050] | +0.22 | −0.060 |
| keep_all 방향 + reset_both 크기 | +0.000 | (일치) | −0.006 |
| reset_both 방향 + keep_all 크기 | +0.001 | (일치) | +0.018 |
| tuned Adam LR×β / tuned SGD | +0.005 / +0.000 | – | – |
| periodic reset P=128 (비-oracle) | +0.018 [−0.020, +0.044] | – | −0.038 |

전체 표: `runs/p4_smoke_6e4cdaf/summary.md`.

기술적 해석 (가설 검정 아님):

- **튜닝이 불충분했다.** base Adam LR은 grid 최댓값 0.01에서 선택됐다 (dev 점수 0.474 → 0.681 → 0.764로 단조 증가).
  SGD는 최솟값 0.01에서 선택됐다. 따라서 이 설정에서는 유효 step을 키우는 개입이라면 무엇이든 이득처럼 보일 수
  있다. 실제로 LR bump와 오보정 arm(update 3.9–29배)이 가장 큰 "이득"을 보였다.
- **이득이 가장 큰 arm은 bias-correction 오보정 arm이다.** 이 arm들은 feature를 더 많이 바꿨다 (CKA −0.13).
  올바르게 보정한 reset_both는 효과가 약 0이다. 스모크 수준에서 보이는 양상은 "state repair"가 아니라
  **유효 LR 증가 / bias-correction artifact**와 일치한다. 이 판단은 seed 3개의 기술 관찰이며, 파일럿의 H2–H4로
  검정해야 한다.
- reset_m의 +0.034도 초기 update norm을 약 2배로 키웠다. 스모크 config에는 reset_m용 norm 일치 통제가 없어서
  크기 효과와 분리할 수 없다 (사전등록 개정 1-3).
- **plasticity loss가 관측되지 않았다.** keep_all 경로에서 과제별 최종 held-out 정확도는 0.90 → 0.93으로
  하락하지 않았다. 과제별 AUC는 0.85에서 0.75–0.80으로 낮아졌지만, 음의 전이와 구분되지 않는다. 따라서 이
  스모크로는 "회복"을 말할 수 없다.
- 모든 branch arm은 oracle timing이다 (분기점 = 과제 경계).

### 실제 계산 비용 (같은 step 상한 ≠ 같은 실지출)

- branch arm: 분기 뒤 200 step, seed당 약 2.0 s. 여기에 공유 trunk 600 step(seed당 약 4.8 s)이 더해진다.
- 처음부터 실행하는 arm: 800 step, seed당 약 6.7–7.0 s.
- 튜닝(dev) 실행 수: adam_lr 6, adam_lr_beta 18 (그중 6은 adam_lr과 중복), sgd 8.
  모든 branch arm은 adam_lr 튜닝 비용을 공유한다.
- norm 일치 arm은 다른 arm의 궤적(step별 norm)을 추가로 쓴다. 이는 특권 정보이며 배치 가능한 방법이 아니다.

## Blocker (Stage 1 이후 갱신)

- 해소: torch를 저장소 전용 `.venv`에 설치했다. 공식 CPU index, 29.4 s, 목록은 `requirements-venv.lock.txt`.
  시스템 Python과 전역 설정은 바꾸지 않았다. torch parity는 PASS.
- 남음: 이 환경에서 pure-Python 모델은 step당 약 2.8 ms로 작은 MLP에 한정된다. torch 백엔드 러너는 없다.
- 남음: 실제 데이터 없음 (승인 범위 밖).

### (이전 기록) 초판 blocker

1. **numpy / torch / pytest 없음.** `python3 -c "import torch"` 결과는 `ModuleNotFoundError`. 설치는 승인 범위
   밖이라 하지 않았다. 그래서 전체를 표준 라이브러리로 구현했다. 순수 Python이라 모델 규모는 작은 MLP로
   제한된다 (step당 약 2.7 ms, H=32).
2. **torch.optim.Adam과의 수치 일치: NOT_RUN.** 수식과 closed form으로만 검증했다.
3. 선행연구 중 EWRL 2023 원문(OpenReview 차단)과 Hu et al. 2026은 FULL_TEXT_UNVERIFIED. Asadi 2023의 단일
   moment ablation에서 counter를 어떻게 처리했는지, Lyle 2023 App. B.1의 reset 프로토콜은 확인하지 못했다.

## 미검증 주장

- "state repair가 plasticity를 회복한다": 미평가.
- decoupled age가 비정상 전환 뒤에도 적절한 보정이라는 것: 정상성 가정에서만 유도된다.
- 스모크의 모든 수치: seed 3개의 기술 관찰이며 일반화할 수 없다.
- "선행연구에 없음"이라는 항목(`PRIOR_ART.md` 마지막 절): 읽은 범위 안의 부재일 뿐 신규성 인증이 아니다.

## 파일럿 전 필요 작업 (승인 필요 항목 표시)

1. 파일럿 config: 확장한 튜닝 grid와 가장자리 재튜닝 규칙, 모든 state arm의 norm 일치 통제, `reset_t` 포함
   (사전등록 개정 1).
2. 긴 스트림(≥ 50 과제)에서 keep_all plasticity loss를 먼저 확인. 순수 Python이면 과제 50개 × 200 step ×
   약 20 arm 규모는 CPU 수 시간이 걸린다고 추정한다 (미측정).
3. 비-oracle trigger 구현: 관측 loss 통계 기반이며 임계값은 calibration seed로 정한다. 과제 중간 분기점(음성 사례)
   추가.
4. 분석 스크립트: paired bootstrap CI, Holm 보정, H8 예측자 평가 (Wang et al. 2026 OR baseline 포함).
5. **[승인 필요]** torch 설치·사용 여부. 승인되면 torch 러너가 추가로 복제해야 하는 것:
   - `optimizer.state_dict()`의 `step` tensor와 `exp_avg`, `exp_avg_sq`
   - `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`, DataLoader generator와 worker seed, numpy/python RNG
   - BatchNorm `num_batches_tracked`
   - `torch.use_deterministic_algorithms`

   또한 torch Adam과의 parity 테스트를 추가한다.
6. **[승인 필요]** 실제 데이터셋(예: Permuted MNIST)을 쓰면 다운로드와 사용권 확인이 필요하다. 현재는 합성
   데이터뿐이며 외부 데이터·가중치·encoder를 쓰지 않는다.
