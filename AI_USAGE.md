# AI 사용 기록

## 사용한 도구

- Claude Code (Anthropic), 원격 cloud 세션. 날짜: 2026-09-26.
- 사람(저장소 소유자)의 지시문에 따라 AI가 코드·테스트·설정·문서를 작성하고 실행했다.
- 선행연구 확인은 AI 하위 에이전트가 웹 검색/페이지 조회로 수행했다. 확인 수준은 `PRIOR_ART.md`에 논문별로
  표시했다 (`FULL_TEXT_VERIFIED` / `FULL_TEXT_UNVERIFIED` / `NOT_FOUND`). 사람이 원문을 직접 확인하지 않았다.

## AI가 한 일

- `osrepair/` 전체, `tests/` 전체, `configs/p4_smoke.json`, 모든 문서 작성.
- 테스트와 스모크 실행, 결과 요약.

## AI가 하지 않은 일 / 한계

- 패키지 설치, 다운로드, GPU 사용, 유료 API 호출 없음.
- torch와의 수치 일치(parity)는 검증하지 못했다 (torch 미설치). Adam 구현은 Kingma & Ba Algorithm 1의 수식과
  독립 전사(transcription) 및 closed-form으로만 검증했다.
- 스모크 결과는 러너 검증용이며 연구 질문에 대한 증거로 해석하지 않는다.
- 수치 테스트(fixture)는 구현 검증이지 수학적 증명이 아니다.

## 사람이 검토해야 할 것

1. `PRIOR_ART.md`의 `FULL_TEXT_UNVERIFIED` 항목 원문 확인, 특히 신규성 판단에 영향을 주는 항목.
2. `docs/PREREGISTRATION.md`의 MIE(2%p), 튜닝 범위, seed 수가 연구 목적에 맞는지.
3. 파일럿 진행 전 torch 설치/사용 승인 여부와 데이터·가중치 사용권 (현재는 합성 데이터만 사용, 외부 가중치 없음).

## 작업 중 AI가 스스로 발견·수정한 오류 (기록 보존)

- 테스트 기대값 오류: shared-counter reset의 update 확대율을 점근값(≈31.6배)으로 단정했으나, 이력 500 step에서는
  정확한 값이 `sqrt((1−β2^501)/(1−β2)) ≈ 19.9배`였다. 임계값을 낮추지 않고 정확한 closed-form 검사로 바꾸고,
  점근값은 40,000 step 이력으로 별도 검사했다.
- 수치 조건수: `β^age`가 매우 작을 때 EMA 가중치에서 age를 역산하면 정밀도가 떨어진다(조건수
  ≈ 1/(β^age·|log β|)). bias correction에 쓰이는 가중치 자체의 왕복 정밀도를 검사하도록 테스트를 바로잡았다.
- 의미 불일치: 검증기의 `state_only`(optimizer 밖 불변)와 개입 기록의 `state_only`(hyperparameter 불변)가
  달랐다. end-to-end 테스트가 잡았고, 두 플래그(`outside_optimizer_unchanged`, `state_only`)로 분리했다.
- 산출물 삭제 실수: 문서를 정리하던 중 커밋되지 않은 raw 테스트 로그(`runs/test_logs/unittest_6e4cdaf.log`)를
  지웠다. 해당 commit의 깨끗한 worktree에서 같은 명령으로 다시 생성했고(69/69 통과, 원본과 동일), 로그 머리말에
  재생성 사실을 적었다.
- 선행연구 수식 확인: 하위 에이전트가 옮긴 Ellis et al. Thm 3.1 수식이 처음에는 극한과 맞지 않아 보였다. arXiv HTML
  원문을 직접 확인한 결과, t가 "전환 이후 step 수"이고 이력 길이는 t′→∞였다. 에이전트의 전사는 정확했고,
  테스트 docstring에 정의를 명시했다.

## Stage 1 (2026-09-26, 같은 도구)

- AI가 작성한 것:
  - `.venv` 설치: 공식 PyTorch CPU index, 설치 로그 `runs/install/`.
  - `tests/test_torch_parity.py`, `osrepair/stage1.py`, `configs/p4_stage1.json`, `tests/test_stage1.py`,
    reset_t 항등식 테스트, `docs/STAGE1_REPORT.md`, STATUS 갱신.
- Stage 1에서는 하위 에이전트와 웹 조회를 쓰지 않았다.
- AI가 스스로 발견·수정한 것:
  - `label_permutation`을 4-class로 두면 순열이 24개뿐이라 50개 과제에서 반복되고, probe 과제가 새 과제가 아니게
    된다. 실행 전에 발견해 C=10으로 고정했고, 런타임 검사를 추가했다.
  - 개입 경로 테스트가 tiny grid에서 SKIP됐다. skip은 PASS가 아니므로, 단계 함수를 직접 호출하는 테스트로 바꿨다.
  - reset_t와 norm 일치 통제의 차이가 정확히 0으로 나왔다. 버그 가능성을 먼저 의심해 원로그를 확인했고,
    첫 step 비율 0.3162 = √(1−β2)/(1−β1)로 구조적 항등식임을 확인한 뒤 회귀 테스트로 고정했다.
- 사람이 검토할 것: `docs/STAGE1_REPORT.md`의 해석, 특히 오보정 artifact 이득을 "유효 step 크기 민감성"으로만
  기술한 부분. 원인은 검정하지 않았다.
