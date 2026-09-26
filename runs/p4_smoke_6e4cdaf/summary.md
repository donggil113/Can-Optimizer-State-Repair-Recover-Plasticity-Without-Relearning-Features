# Run p4_smoke_6e4cdaf

- purpose: TECHNICAL smoke test of the paired branching runner on CPU. Not the pilot. Numbers are not evidence for or against the research question.
- status: COMPLETED
- git: 6e4cdafef852a0236e4ab262e3a5b947f69e79f5 (dirty=True)
- config_sha256: d14577969816ebf7c1ad3d19f488ae1d710379e8154d57f4647dc74f3078a08d
- wall: 284.7s, cpu: 282.1s, peak RSS: 26.8 MiB
- base optimizer (resolved): {'kind': 'adam', 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-08, 'lr': 0.01}

Paired differences vs keep_all over test seeds (mean [min, max]).
Descriptive only: n is tiny and nothing here is a hypothesis test.

| arm | timing | status | d new_task_auc_acc | d forgetting_prev | d cka_end | d upd_norm_first10 |
|---|---|---|---|---|---|---|
| keep_all | oracle_boundary | PASS | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| reset_m | oracle_boundary | PASS | +0.0337 [+0.0307, +0.0395] | +0.0026 [-0.0039, +0.0117] | -0.0073 [-0.0149, +0.0032] | +0.1593 [+0.1386, +0.1757] |
| reset_v | oracle_boundary | PASS | -0.0434 [-0.0680, -0.0172] | +0.0000 [-0.0117, +0.0117] | +0.0444 [+0.0336, +0.0572] | +0.0027 [-0.0497, +0.1003] |
| reset_both | oracle_boundary | PASS | -0.0014 [-0.0244, +0.0102] | +0.0000 [-0.0078, +0.0078] | +0.0302 [+0.0128, +0.0449] | +0.0268 [+0.0225, +0.0313] |
| mix_0.5 | oracle_boundary | PASS | +0.0063 [+0.0051, +0.0078] | +0.0013 [-0.0039, +0.0078] | -0.0009 [-0.0058, +0.0049] | +0.0317 [+0.0244, +0.0365] |
| reset_m__shared_keep | oracle_boundary | PASS | +0.0004 [-0.0047, +0.0047] | -0.0013 [-0.0078, +0.0039] | -0.0008 [-0.0034, +0.0019] | -0.0054 [-0.0136, +0.0008] |
| reset_v__shared_keep | oracle_boundary | PASS | +0.0779 [+0.0520, +0.0975] | -0.0026 [-0.0078, +0.0000] | -0.1262 [-0.2051, -0.0603] | +4.0555 [+1.1206, +8.7940] |
| reset_both__shared_keep | oracle_boundary | PASS | +0.1104 [+0.0777, +0.1328] | -0.0052 [-0.0156, +0.0117] | -0.1349 [-0.1857, -0.1005] | +0.4069 [+0.4005, +0.4123] |
| reset_m__shared_reset | oracle_boundary | PASS | -0.2191 [-0.2303, -0.2113] | -0.0495 [-0.0781, -0.0195] | +0.2082 [+0.1779, +0.2444] | -0.1011 [-0.1046, -0.0967] |
| reset_v__shared_reset | oracle_boundary | PASS | -0.0158 [-0.0383, +0.0133] | -0.0078 [-0.0156, -0.0039] | +0.0593 [+0.0031, +0.0898] | +0.8354 [+0.2910, +1.7525] |
| lr_bump_x3_20 | oracle_boundary | PASS | +0.0477 [+0.0428, +0.0504] | -0.0026 [-0.0078, +0.0039] | -0.0597 [-0.0757, -0.0414] | +0.2247 [+0.2160, +0.2297] |
| beta_switch_b1b2_0.9 | oracle_boundary | PASS | -0.0143 [-0.0455, +0.0074] | -0.0026 [-0.0078, +0.0000] | -0.0335 [-0.0638, -0.0130] | -0.0637 [-0.0695, -0.0530] |
| keepdir_resetbothmag | oracle_boundary | PASS | +0.0001 [-0.0098, +0.0109] | -0.0039 [-0.0156, +0.0039] | -0.0055 [-0.0160, +0.0133] | +0.0268 [+0.0225, +0.0313] |
| resetbothdir_keepmag | oracle_boundary | PASS | +0.0014 [-0.0068, +0.0086] | +0.0000 [-0.0078, +0.0117] | +0.0175 [-0.0039, +0.0284] | -0.0000 [-0.0000, +0.0000] |
| adam_base_replay | none | PASS | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| adam_tuned_lr_beta | none | PASS | +0.0048 [-0.0141, +0.0207] | +0.0195 [+0.0117, +0.0273] | +0.0002 [-0.0071, +0.0058] | -0.0121 [-0.0164, -0.0041] |
| sgd_tuned | none | PASS | +0.0004 [-0.0367, +0.0254] | +0.0000 [-0.0273, +0.0273] | +0.0796 [+0.0511, +0.0958] | -0.0632 [-0.0678, -0.0599] |
| periodic_reset_both_P128 | periodic_no_oracle | PASS | +0.0184 [-0.0197, +0.0439] | +0.0130 [-0.0039, +0.0391] | -0.0376 [-0.0439, -0.0321] | -0.0045 [-0.0106, +0.0026] |

Integrity checks per seed:

- seed 3000: PASS
- seed 3001: PASS
- seed 3002: PASS
