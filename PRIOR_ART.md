# 선행연구 대조 (짧은 claim table)

확인일: 2026-09-26. 확인 방법: AI 하위 에이전트가 arXiv PDF/HTML, nature.com HTML을 조회해 해당 절을 읽었다.
그림 이미지는 보지 않았다. 사람이 원문을 직접 확인하지 않았다 (`AI_USAGE.md`).
표기: `FULL_TEXT_VERIFIED` = 본문 해당 절을 읽음, `FULL_TEXT_UNVERIFIED` = 초록/메타데이터만 확인.
★ = 메인 세션에서 원문 수식을 직접 한 번 더 대조함.

## 핵심 선행연구

| # | 논문 (저자, 연도, venue, ID) | 확인 수준 | 이 프로젝트와 관련된 내용 (절) | 결론: 이미 알려진 것 |
|---|---|---|---|---|
| 1 | *Loss of plasticity in deep continual learning*. Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood, Sutton. Nature 632:768–774 (2024). doi:10.1038/s41586-024-07711-7 / arXiv 2306.13812 | FULL_TEXT_VERIFIED | Methods: Online Permuted MNIST에서 Adam(기본 β)의 plasticity loss가 "catastrophic"하며 dead unit이 초기에 약 60%까지 증가. RL에서는 β1=0.9/β2=0.999가 큰 plasticity loss를 일으키고, v의 느린 갱신이 큰 update를 만든다고 설명한다. 해결책으로 β1=β2=0.99를 쓴다. 진단: dead unit, 가중치 크기, effective rank. CBP 논문 본문은 optimizer 상태를 언급하지 않지만, 공개 코드는 재초기화된 unit의 Adam 상태(exp_avg, exp_avg_sq, step)를 0으로 만든다 (코드에서만 확인). | 기본 Adam이 plasticity loss를 악화시킨다는 점, β1=β2 처방, unit 단위의 Adam 상태 reset (CBP 코드). |
| 2 | *Understanding Plasticity in Neural Networks*. Lyle, Zheng, Nikishin, Avila Pires, Pascanu, Dabney. ICML 2023. arXiv 2303.01486 | FULL_TEXT_VERIFIED | §4.1: label 재무작위화 직후 m̂이 v̂보다 빨리 적응해 update가 커지고, 발산과 ReLU 포화가 생긴다. ε을 키우고 β2=0.9로 하면 완화된다. §5.2: weight norm·feature rank·포화 unit은 설정에 따라 plasticity와의 상관 부호가 바뀐다. **App. B.1: toy RL에서 "optimizer 상태만 reset하는 것은 plasticity를 개선하지 않았다".** 어떤 moment와 counter를 reset했는지는 확인하지 못함. | stale v로 인한 과대 update, ε/β2 처방, rank 진단의 한계, **state-only reset에 대한 음성 결과(toy RL)**. |
| 3 | *Predicting Plasticity in Deep Continual Learning: A Theoretical Perspective*. Wang, Srinivasa, Chen, Liu, Payani, Zhang. arXiv 2605.09044 (2026-05, preprint이며 venue 미확인) | FULL_TEXT_VERIFIED | plasticity를 checkpoint에서 **plain SGD** k-step 이득으로 정의한다. rank 기반 예측자의 반례를 보이고 "optimization readiness" OR = S·R (gradient 통계)을 제안한다. checkpoint는 Adam으로 학습하지만 probe는 Adam 상태를 버리고, **optimizer 상태는 예측자에 없다**. | gradient 통계 기반 plasticity 예측이 rank·dormancy보다 낫다는 점. |
| 4 | *Resetting the Optimizer in Deep RL: An Empirical Study*. Asadi, Fakoor, Sabach. NeurIPS 2023. arXiv 2306.17833 | FULL_TEXT_VERIFIED | 알려진 경계(target network update)마다 **m=0, v=0, step=0**으로 reset하면 Rainbow에 도움이 된다. §4.3: m만 reset하면 효과가 없고, v만 reset하면 상당히 효과적이며, 둘 다가 가장 좋다. 단일 moment ablation에서 counter를 어떻게 처리했는지는 확인하지 못함. reset 전후 cos(m, g)는 0.39에서 약 0으로 떨어진다. | 경계 oracle을 쓰는 전체 reset이 RL에서 이득이 있고, v reset의 기여가 크다는 점. |
| 5 | *Adam on Local Time: Addressing Nonstationarity in RL with Relative Adam Timesteps*. Ellis, Jackson, Lupu, Goldie, Fellows, Whiteson, Foerster. NeurIPS 2024. arXiv 2412.17113 | FULL_TEXT_VERIFIED ★ | Adam-Rel은 m과 v를 유지한 채 **step counter t만 0으로** 되돌린다. Thm 3.1: gradient가 g에서 kg로 바뀌면 update는 k→∞에서 (1−β1)/√(1−β2)=√10에 가까워진다. Eq. 4: t를 reset하면 1로 수렴한다 (★ 수식 직접 대조, `tests/test_adam.py::TestResetT`). DQN에서는 전체 reset(Adam-MR)이 기본 Adam보다 나빴다. 저자들은 "contamination 가설은 이득을 설명하지 못한다"고 쓰고, LR annealing/warmup과 닮았다고 인정한다. | counter만 reset하는 방법, bias correction과 update 크기로 전환 직후 과대 update를 설명하는 것, **reset 효과와 LR schedule의 유사성**. |
| 6 | *Overcoming Policy Collapse in Deep RL*. Dohare, Lan, Mahmood. EWRL 2023 | FULL_TEXT_UNVERIFIED (OpenReview 접근 차단. 같은 내용을 담은 Dohare 박사논문 Ch. 7을 대신 읽음) | 2-state MDP에서 근사 0 gradient 뒤에 갑작스러운 gradient가 오면 첫 update가 약 3.16배가 된다. β1=β2가 collapse를 없앤다. | β1=β2 처방과 과대 update 설명. |
| 7 | *Disentangling the Causes of Plasticity Loss in Neural Networks*. Lyle, Zheng, Khetarpal, van Hasselt, Pascanu, Martens, Dabney. CoLLAs 2024 (PMLR 274). arXiv 2402.18762 | FULL_TEXT_VERIFIED | §2.2: optimizer 상태를 plasticity 정의에 포함하는 것이 유용할 수 있다. App. E.3: 새 label 과제 전에 Adam을 reset하면 dead unit이 약 절반으로 줄어든다 (MLP 하나의 예시). probe는 새 optimizer를 쓴다. | 과제 전환 시 Adam reset이 dead unit을 줄인다는 예시, optimizer 상태를 plasticity 정의에 넣자는 제안. |
| 8 | *Adam: A Method for Stochastic Optimization*. Kingma, Ba. ICLR 2015. arXiv 1412.6980 | FULL_TEXT_VERIFIED | Algorithm 1 (구현과 대조: `tests/test_adam.py::test_matches_textbook_algorithm1`). §2.1: 이전 gradient가 전부 0인 극단적 경우 \|Δ\| ≤ α(1−β1)/√(1−β2). §3: bias correction은 0 초기화 EMA와 정상성 가정에서 유도된다. | √10·α 최악 step, bias correction의 가정. **이 저장소의 shared-counter artifact closed form은 이 수식들의 특수 사례다.** |

## 후속·주변 연구 (요약)

| 논문 | 확인 수준 | 관련 내용 | 이미 알려진 것 |
|---|---|---|---|
| Kang & Lee, *Continual Learning of Numerous Tasks from Long-tail Distributions*, arXiv 2404.02754 (2024) | FULL_TEXT_VERIFIED | "Continual Adam": 과제 간 v를 평균·혼합하고, 과제 시작 시 LR을 감쇠한다. m과 v 각각에 대해 reset/carry/average ablation을 했다. v는 유지하는 편이 낫고, m을 유지하면 plasticity 지표가 나빠진다. 주 관심은 망각. | moment별 carry/reset/mixing (망각 맥락). |
| Kumar, Marklund, Van Roy, *L2 Init*, CoLLAs 2024, arXiv 2308.11958 | FULL_TEXT_VERIFIED | Adam의 stale moment가 "ill-suited"라고 보고, Lyle 2023의 optimizer reset 음성 결과를 인용한다. SGD가 Adam보다 plasticity loss가 적다. | 같은 내용. |
| Galashov et al., *Soft Parameter Reset*, NeurIPS 2024, arXiv 2411.04034 | FULL_TEXT_VERIFIED | 가중치를 초기값 쪽으로 옮기는 것과 LR 증가를 결합한다. optimizer 상태 개입은 아니다. | 가중치+LR 결합 soft reset. |
| Lee et al., *Hare & Tortoise*, ICML 2024, arXiv 2406.02596 | FULL_TEXT_VERIFIED | β/ε 튜닝의 이득은 "marginal"이다. 단계 사이에 optimizer를 재초기화하는 것을 표준 절차로 사용한다. | β 튜닝의 한계. |
| Klein et al., *Plasticity Loss in Deep RL: A Survey*, arXiv 2411.04832 v3 (2026) | FULL_TEXT_VERIFIED (§5.2, §5.10) | optimizer reset(Asadi 방식, Adam-Rel)을 하나의 범주로 정리한다. | optimizer reset은 이미 인정된 범주다. |
| Hernandez-Garcia, Figliolia, Millidge, *Can Scale Save Us From Plasticity Loss in LLMs?*, arXiv 2606.24752 (2026) | FULL_TEXT_VERIFIED (method) | 과제마다 optimizer reset과 warmup 재시작을 **통제 조건**으로 사용해, 가중치 수준의 plasticity loss를 stale state와 분리한다. | reset+warmup을 통제로 쓰는 관행. |
| Sahu et al., *Adapt or Forget: Provable Tradeoffs Between Adam and SGD in Nonstationary Optimization*, arXiv 2605.04269 (2026) | FULL_TEXT_VERIFIED (일부 절) | tracking error를 β1이 좌우하는 1차 moment 추적 오차와 β2가 좌우하는 preconditioner 섭동으로 분해한다. drift가 크면 SGD가 Adam보다 유리할 수 있다. | stale m/v의 비용에 대한 이론. |
| Hu et al., *Hidden Failure Modes of Gradient Modification under Adam in CL…*, arXiv 2604.22407 (2026) | FULL_TEXT_UNVERIFIED (초록만) | 수정된 gradient는 m에만 넣고 v는 크기에 충실하게 유지하는 "moment routing" repair (망각 맥락). | moment별 "repair"라는 표현 자체. |

## 이번 프로젝트가 기여로 주장하지 않는 것

- Adam 상태가 비정상성·plasticity에 중요하다는 것 (1, 2, 4, 5, 6, 7).
- rank·dead-unit 진단의 한계 (2, 3).
- m/v/t reset, β1=β2, v mixing 자체 (4, 5, 6, Kang & Lee).
- shared-counter reset의 과대/과소 보정 closed form. Kingma & Ba §2.1–3과 Ellis Thm 3.1의 특수 사례이며,
  이 저장소의 테스트는 구현 검증일 뿐 새 결과가 아니다.
- moment별 age 분리(`decoupled`)는 EMA 가중치 정의에서 바로 나오는 계산이다. 새 알고리즘이라고 주장하지 않는다.
  ablation을 bias-correction 오류 없이 수행하기 위한 **측정 도구**다.

## 읽은 범위 안에서 발견하지 못한 것 (신규성 인증 아님)

에이전트가 읽은 자료 안에서는 아래 항목을 찾지 못했다. 읽지 않은 문헌에 존재할 수 있다.

1. 가중치를 고정한 채 같은 checkpoint에서 분기하는 **paired** 설계로 m / v / m+v / t / mixing을 한 연속학습
   지도학습 설정에서 함께 비교하고, counter 처리를 명시한 연구.
2. state 개입의 효과를 **step별 update-norm 일치 통제**, LR·β 튜닝 통제와 분리한 연구. Ellis et al.는 LR schedule과의
   유사성을 스스로 지적했을 뿐 분리하지 않았다.
3. Adam 상태(v̂의 staleness, cos(m̂, g), age)로 **개입 효과를 예측**하는 연구. Wang et al. 2026은 raw gradient
   통계와 SGD probe를 쓴다.
4. state 개입이 feature 수준 지표(CKA, dead unit)를 바꾸는지, update 크기만 바꾸는지를 구분한 연속 지도학습 연구.

반대 방향의 선행 증거도 있다. Lyle 2023 App. B.1(state-only reset 음성, toy RL)과 Ellis 2024(전체 reset이 DQN에서
오히려 나쁨)는 H1이 기각될 가능성을 시사한다.
