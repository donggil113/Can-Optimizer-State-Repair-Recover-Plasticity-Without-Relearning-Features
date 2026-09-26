# P4 — Can Optimizer-State Repair Recover Plasticity Without Relearning Features?

현재 상태는 [`STATUS.md`](STATUS.md)를 먼저 볼 것. 요약: **TECHNICAL_TEST_PASS + SCIENCE_NOT_EVALUATED**.

이 저장소는 "가중치를 건드리지 않고 optimizer 상태(Adam의 m, v, bias-correction age)만 수정하는 개입"의 효과를
**같은 checkpoint에서 갈라진 paired branch**로 비교하기 위한 최소 러너와 테스트를 담고 있다.
과학적 질문은 아직 평가하지 않았다. 파일럿 설계는 [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md),
선행연구 대조는 [`PRIOR_ART.md`](PRIOR_ART.md).

## 실행 환경

- Python 3.11 **표준 라이브러리만** 사용. numpy/torch/pytest는 이 환경에 없고, 설치는 승인되지 않아 하지 않았다.
- CPU 단일 프로세스.

## 명령

```bash
python3 -m osrepair.cli env                                  # 환경 정보
python3 -W error::ResourceWarning -m unittest discover -s tests -t . -v   # 테스트 (73개, ~3초)
python3 -m osrepair.cli run --config configs/p4_smoke.json --out runs    # 스모크 (~수 분)
```

## 구조

| 파일 | 내용 |
|---|---|
| `osrepair/model.py` | MLP: Linear → BatchNorm1d(running buffers) → ReLU → Dropout → Linear, 수동 backward |
| `osrepair/optim.py` | Adam (moment별 age `t_m`, `t_v`), SGD(momentum, torch 규약) |
| `osrepair/interventions.py` | keep_all / reset_m / reset_v / reset_both / reset_t / mix, counter policy, beta_switch |
| `osrepair/trainer.py` | 전체 상태 snapshot·restore·branch, state-only 검증기, 진단 지표 |
| `osrepair/stream.py` | 합성 과제 스트림, dev/calibration/test seed 분할, oracle boundary 접근자 |
| `osrepair/runner.py` | dev 튜닝 → trunk → checkpoint → paired arms → 무결성 검사 → 로그·manifest |
| `configs/p4_smoke.json` | 스모크 설정 (모든 arm의 timing과 정보 접근 명시) |
| `tests/` | 73개 unittest |
| `runs/` | 실행 산출물 (raw log, manifest, summary) |

## 핵심 설계

1. **Checkpoint = 미래를 결정하는 모든 것**: 파라미터, BatchNorm running mean/var/`num_batches_tracked`,
   optimizer 상태와 hyperparameter, 데이터 샘플링 RNG, dropout RNG, 스트림 위치. 전역 `random`은 쓰지 않는다
   (테스트로 확인).
2. **State-only 검증**: 개입 직후 가중치·버퍼·RNG·위치·hyperparameter가 bitwise 동일하고, eval/train 모드
   forward 출력(동일 dropout mask 포함)이 bitwise 동일한지 확인한다. 가중치·버퍼·RNG·step을 바꾸는 음성 대조로
   검증기가 공허하지 않음을 테스트한다.
3. **Moment별 age**: Adam의 공유 step counter를 `t_m`, `t_v`로 분리했다. 정상 학습에서는 항상 같고 Algorithm 1과
   동일하다. 개입 후 `decoupled` 정책은 `1 − β^age`가 해당 moment의 실제 gradient 가중치와 같도록 age를 설정한다.
   `shared_keep`(torch에서 `exp_avg`만 0으로 만들고 `step`은 둔 경우)과 `shared_reset`은 bias-correction 오류를
   재현하는 측정용 arm이다. 예: v만 0으로 하고 counter를 유지하면 첫 update가 `sqrt((1−β2^(t+1))/(1−β2))`배
   (t≫1000에서 ≈31.6배) 커진다 — 테스트로 확인.
4. **Paired 무결성**: 모든 arm이 같은 batch hash·dropout mask hash 열을 받는지, keep_all branch가 같은 설정의
   처음부터 실행과 bitwise 동일한지 매 seed 검사한다.
