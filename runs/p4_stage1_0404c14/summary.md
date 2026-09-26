# p4_stage1_0404c14

- status: COMPLETED  overall: STATE_REPAIR_BRANCH_ON_HOLD
- git 0404c147bc962e7d3efd63b8f8b01a81e590eff1 dirty=True; config_sha256 d9c2b7aceeeb084f; code_sha256 b45b507e4aef2974
- CPU used 2078.3s (cap 3600), wall 2096.8s, peak RSS 29.2 MiB

## Tuning (dev seeds, online train accuracy over the stream)

- random_teacher_c4/adam: RESOLVED best={'kind': 'adam', 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-08, 'lr': 0.003} edge=None extended=None rows={'0.001': 0.451096875, '0.003': 0.5440499999999999, '0.01': 0.5188937499999999, '0.03': 0.3995375}
- random_teacher_c4/adam_eqbeta: RESOLVED best={'kind': 'adam', 'beta1': 0.9, 'beta2': 0.9, 'eps': 1e-08, 'lr': 0.003} edge=None extended=None rows={'0.001': 0.413575, '0.003': 0.4775875, '0.01': 0.452221875, '0.03': 0.384309375}
- random_teacher_c4/sgd: RESOLVED best={'kind': 'sgd', 'momentum': 0.9, 'lr': 0.01} edge=None extended=None rows={'0.003': 0.491721875, '0.01': 0.512953125, '0.03': 0.450928125, '0.1': 0.320175}
- label_permutation_c10/adam: RESOLVED best={'kind': 'adam', 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-08, 'lr': 0.003} edge=None extended=None rows={'0.001': 0.273459375, '0.003': 0.361478125, '0.01': 0.34735312500000004, '0.03': 0.23689375000000001}
- label_permutation_c10/adam_eqbeta: RESOLVED best={'kind': 'adam', 'beta1': 0.9, 'beta2': 0.9, 'eps': 1e-08, 'lr': 0.003} edge=None extended=None rows={'0.001': 0.228228125, '0.003': 0.26875625000000003, '0.01': 0.24085, '0.03': 0.191815625}
- label_permutation_c10/sgd: RESOLVED best={'kind': 'sgd', 'momentum': 0.9, 'lr': 0.01} edge=None extended=None rows={'0.003': 0.3030875, '0.01': 0.346728125, '0.03': 0.30909375, '0.1': 0.18885000000000002}

## Phenomenon (base family; probe AUC deltas, mean [per seed])

### random_teacher_c4: ESTABLISHED 
- d_probe_auc_acc_late_minus_early: -0.2446 [-0.198, -0.221, -0.315]
- d_probe_early_acc_late_minus_early: -0.1768 [-0.154, -0.159, -0.218]
- d_probe_final_acc_late_minus_early: -0.1771 [-0.141, -0.133, -0.258]
- d_probe_auc_acc_late_minus_fresh: -0.3047 [-0.261, -0.290, -0.363]
- d_probe_auc_acc_early_minus_fresh: -0.0601 [-0.062, -0.069, -0.049]
- intervention stage: STATE_REPAIR_BRANCH_ON_HOLD
  - reset_m: d_auc_vs_keep_all +0.0207 [+0.015, +0.032, +0.015]; vs norm-matched +0.0123 [+0.012, +0.010, +0.015]; EXPLAINED_BY_UPDATE_NORM
  - reset_v: d_auc_vs_keep_all -0.1523 [-0.181, -0.087, -0.188]; vs norm-matched -0.0195 [-0.054, -0.060, +0.056]; NO_POSITIVE_EFFECT_OVER_KEEP_ALL
  - reset_both: d_auc_vs_keep_all -0.0076 [-0.020, -0.049, +0.045]; vs norm-matched -0.0274 [-0.011, -0.058, -0.013]; NO_POSITIVE_EFFECT_OVER_KEEP_ALL
  - reset_t: d_auc_vs_keep_all -0.2071 [-0.183, -0.255, -0.183]; vs norm-matched +0.0000 [+0.000, +0.000, +0.000]; NO_POSITIVE_EFFECT_OVER_KEEP_ALL
  - reset_v__shared_keep: d_auc_vs_keep_all +0.0628 [+0.038, +0.313, -0.163]; vs norm-matched n/a; miscorrection artifact control
  - reset_both__shared_keep: d_auc_vs_keep_all +0.3330 [+0.268, +0.321, +0.410]; vs norm-matched n/a; miscorrection artifact control

### label_permutation_c10: ESTABLISHED 
- d_probe_auc_acc_late_minus_early: -0.2038 [-0.178, -0.282, -0.151]
- d_probe_early_acc_late_minus_early: -0.1263 [-0.121, -0.269, +0.011]
- d_probe_final_acc_late_minus_early: -0.2018 [-0.137, -0.223, -0.246]
- d_probe_auc_acc_late_minus_fresh: -0.2363 [-0.226, -0.236, -0.248]
- d_probe_auc_acc_early_minus_fresh: -0.0325 [-0.048, +0.047, -0.097]
- intervention stage: STATE_REPAIR_BRANCH_ON_HOLD
  - reset_m: d_auc_vs_keep_all +0.0197 [+0.027, +0.020, +0.012]; vs norm-matched +0.0076 [+0.006, +0.021, -0.005]; NO_POSITIVE_EFFECT_OVER_KEEP_ALL
  - reset_v: d_auc_vs_keep_all -0.2443 [-0.290, -0.176, -0.266]; vs norm-matched -0.0484 [-0.081, +0.013, -0.077]; NO_POSITIVE_EFFECT_OVER_KEEP_ALL
  - reset_both: d_auc_vs_keep_all -0.0428 [-0.063, -0.058, -0.008]; vs norm-matched -0.0299 [-0.050, -0.024, -0.016]; NO_POSITIVE_EFFECT_OVER_KEEP_ALL
  - reset_t: d_auc_vs_keep_all -0.2026 [-0.244, -0.234, -0.130]; vs norm-matched +0.0000 [+0.000, +0.000, +0.000]; NO_POSITIVE_EFFECT_OVER_KEEP_ALL
  - reset_v__shared_keep: d_auc_vs_keep_all -0.2413 [-0.287, -0.171, -0.265]; vs norm-matched n/a; miscorrection artifact control
  - reset_both__shared_keep: d_auc_vs_keep_all +0.2515 [+0.273, +0.235, +0.246]; vs norm-matched n/a; miscorrection artifact control

## Secondary families (descriptive)

- random_teacher_c4/adam_eqbeta: late-early AUC -0.2839 [-0.228, -0.236, -0.388]; late-fresh -0.4095 [-0.343, -0.388, -0.498]
- random_teacher_c4/sgd: late-early AUC -0.3542 [-0.273, -0.409, -0.380]; late-fresh -0.3717 [-0.284, -0.429, -0.402]
- label_permutation_c10/adam_eqbeta: late-early AUC -0.2827 [-0.364, -0.280, -0.204]; late-fresh -0.3730 [-0.469, -0.288, -0.362]
- label_permutation_c10/sgd: late-early AUC -0.4061 [-0.510, -0.400, -0.309]; late-fresh -0.3804 [-0.495, -0.330, -0.316]

## Cells

- interventions: PASS x6
- phenomenon: PASS x18
- tune: PASS x48
