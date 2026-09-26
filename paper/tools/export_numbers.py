"""Export manuscript numbers, tables and curve data from raw result files only.

Inputs (read-only):
    runs/p4_stage1_0404c14/manifest.json   (results.* fields)
    runs/p4_stage1_0404c14/log.jsonl.gz    (probe_eval records -> learning curves)
    runs/p4_analysis_counter/result.json   (optional; counter/epsilon analysis)

Outputs:
    paper/tables/numbers.tex        \\newcommand macros used by abstract and body
    paper/tables/provenance.tsv     macro -> value -> source file -> field path
    paper/tables/tab_*.tex          tables
    paper/figures/curve_*.dat       pgfplots data (mean and per-seed columns)

Every number in the manuscript should come from a macro defined here.
"""

import gzip
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = "runs/p4_stage1_0404c14"
ANALYSIS = "runs/p4_analysis_counter/result.json"
OUT_T = os.path.join(ROOT, "paper", "tables")
OUT_F = os.path.join(ROOT, "paper", "figures")
SEEDS = ["3000", "3001", "3002"]
CONDS = {"random_teacher_c4": "RT", "label_permutation_c10": "LP"}
ARMS = {"keep_all": "KeepAll", "reset_m": "ResetM", "reset_v": "ResetV", "reset_both": "ResetBoth",
        "reset_t": "ResetT", "reset_v__shared_keep": "ResetVSharedKeep",
        "reset_both__shared_keep": "ResetBothSharedKeep"}
ARM_TEX = {"keep_all": r"\texttt{keep\_all}", "reset_m": r"\texttt{reset\_m}", "reset_v": r"\texttt{reset\_v}",
           "reset_both": r"\texttt{reset\_both}", "reset_t": r"\texttt{reset\_t}",
           "reset_v__shared_keep": r"\texttt{reset\_v}, shared-keep$^\dagger$",
           "reset_both__shared_keep": r"\texttt{reset\_both}, shared-keep$^\dagger$"}

macros, prov = [], []


def mac(name, value, fmt, src, path):
    text = fmt.format(value) if not isinstance(value, str) else value
    macros.append(f"\\newcommand{{\\{name}}}{{{text}}}")
    prov.append("\t".join([name, text, repr(value), src, path]))
    return value


def mean(xs):
    return sum(xs) / len(xs)


def main():
    with open(os.path.join(ROOT, RUN, "manifest.json")) as f:
        m = json.load(f)
    R = m["results"]
    src = f"{RUN}/manifest.json"
    # ------------------------------------------------------------ run facts
    mac("RunCpuSeconds", m["cpu_seconds_used"], "{:,.0f}", src, "cpu_seconds_used")
    mac("RunWallSeconds", m["wall_seconds"], "{:,.0f}", src, "wall_seconds")
    mac("RunPeakRssMiB", m["peak_rss_kb"] / 1024, "{:.0f}", src, "peak_rss_kb/1024")
    mac("RunCpuCap", m["limits_applied"]["RLIMIT_CPU_seconds"][0] - 60, "{:,.0f}", src,
        "limits_applied.RLIMIT_CPU_seconds[0]-60 (os grace)")
    cells = m["cells"]
    mac("RunCellsTotal", len(cells), "{}", src, "len(cells)")
    mac("RunCellsPass", sum(c["status"] == "PASS" for c in cells), "{}", src, "count(cells.status==PASS)")
    mac("RunCommit", m["git"]["commit"][:7], "{}", src, "git.commit[:7]")
    mac("RunConfigHash", m["config_sha256"][:12], "{}", src, "config_sha256[:12]")
    mac("RunCodeHash", m["code_sha256"][:12], "{}", src, "code_sha256[:12]")
    # ---------------------------------------------------------------- tuning
    for cond, P in CONDS.items():
        for fam, F in (("adam", "Adam"), ("adam_eqbeta", "AdamEqBeta"), ("sgd", "Sgd")):
            t = R["tuning"][cond][fam]
            mac(f"{P}Tune{F}Lr", t["best"]["lr"], "{:g}", src, f"results.tuning.{cond}.{fam}.best.lr")
    # ------------------------------------------------------------ phenomenon
    for cond, P in CONDS.items():
        seeds = R["phenomenon"][cond]["adam"]["seeds"]
        for start, S in (("fresh", "Fresh"), ("early", "Early"), ("late", "Late")):
            for key, K in (("probe_auc_acc", "Auc"), ("probe_early_acc", "EarlyAcc"), ("probe_final_acc", "Final"),
                           ("update_norm_first10", "UpdTen")):
                vals = [seeds[s]["probes"][start][key] for s in SEEDS]
                mac(f"{P}{S}{K}", mean(vals), "{:.3f}", src,
                    f"mean_s results.phenomenon.{cond}.adam.seeds[s].probes.{start}.{key}")
        for a, b, AB in (("late", "early", "LateMinusEarly"), ("late", "fresh", "LateMinusFresh"),
                         ("early", "fresh", "EarlyMinusFresh")):
            for key, K in (("probe_auc_acc", "Auc"), ("probe_early_acc", "EarlyAcc"), ("probe_final_acc", "Final")):
                d = [seeds[s]["probes"][a][key] - seeds[s]["probes"][b][key] for s in SEEDS]
                path = f"results.phenomenon.{cond}.adam.seeds[s].probes.{{{a}-{b}}}.{key}"
                mac(f"{P}{AB}{K}", mean(d), "{:+.3f}", src, "mean_s " + path)
                mac(f"{P}{AB}{K}Min", min(d), "{:+.3f}", src, "min_s " + path)
                mac(f"{P}{AB}{K}Max", max(d), "{:+.3f}", src, "max_s " + path)
        mac(f"{P}Verdict", R["verdicts"][cond]["phenomenon"].replace("_", r"\_"), "{}", src,
            f"results.verdicts.{cond}.phenomenon")
        for fam, F in (("adam_eqbeta", "AdamEqBeta"), ("sgd", "Sgd")):
            sd = R["phenomenon"][cond][fam]["seeds"]
            d = [sd[s]["probes"]["late"]["probe_auc_acc"] - sd[s]["probes"]["early"]["probe_auc_acc"] for s in SEEDS]
            mac(f"{P}{F}LateMinusEarlyAuc", mean(d), "{:+.3f}", src,
                f"mean_s results.phenomenon.{cond}.{fam}.seeds[s].probes.{{late-early}}.probe_auc_acc")
    # --------------------------------------------------------- interventions
    rows = {}
    for cond, P in CONDS.items():
        iv = R["interventions"][cond]["seeds"]
        for arm, A in ARMS.items():
            base = f"results.interventions.{cond}.seeds[s].probes"
            auc = [iv[s]["probes"][arm]["probe_auc_acc"] for s in SEEDS]
            mac(f"{P}{A}Auc", mean(auc), "{:.3f}", src, f"mean_s {base}.{arm}.probe_auc_acc")
            dk = [iv[s]["probes"][arm]["probe_auc_acc"] - iv[s]["probes"]["keep_all"]["probe_auc_acc"] for s in SEEDS]
            mac(f"{P}{A}VsKeep", mean(dk), "{:+.3f}", src, f"mean_s {base}.{{{arm}-keep_all}}.probe_auc_acc")
            mac(f"{P}{A}VsKeepMin", min(dk), "{:+.3f}", src, f"min_s {base}.{{{arm}-keep_all}}.probe_auc_acc")
            mac(f"{P}{A}VsKeepMax", max(dk), "{:+.3f}", src, f"max_s {base}.{{{arm}-keep_all}}.probe_auc_acc")
            u = [iv[s]["probes"][arm]["update_norm_first10"] for s in SEEDS]
            mac(f"{P}{A}UpdTen", mean(u), "{:.3g}", src, f"mean_s {base}.{arm}.update_norm_first10")
            th = [iv[s]["probes"][arm]["theta_rel_change_end"] for s in SEEDS]
            mac(f"{P}{A}Theta", mean(th), "{:.3g}", src, f"mean_s {base}.{arm}.theta_rel_change_end")
            ck = [iv[s]["probes"][arm]["cka_to_start_end"] for s in SEEDS]
            mac(f"{P}{A}Cka", mean(ck), "{:.3f}", src, f"mean_s {base}.{arm}.cka_to_start_end")
            fg = [iv[s]["probes"][arm]["old_task_forgetting"] for s in SEEDS]
            mac(f"{P}{A}Forget", mean(fg), "{:.3f}", src, f"mean_s {base}.{arm}.old_task_forgetting")
            row = {"auc": mean(auc), "dk": dk, "upd": mean(u), "theta": mean(th), "cka": mean(ck), "forget": mean(fg)}
            ctrl = f"keepdir_{arm}mag"
            if ctrl in iv[SEEDS[0]]["probes"]:
                dc = [iv[s]["probes"][arm]["probe_auc_acc"] - iv[s]["probes"][ctrl]["probe_auc_acc"] for s in SEEDS]
                mac(f"{P}{A}VsCtrl", mean(dc), "{:+.3f}", src, f"mean_s {base}.{{{arm}-{ctrl}}}.probe_auc_acc")
                mac(f"{P}{A}VsCtrlMin", min(dc), "{:+.3f}", src, f"min_s {base}.{{{arm}-{ctrl}}}.probe_auc_acc")
                mac(f"{P}{A}VsCtrlMax", max(dc), "{:+.3f}", src, f"max_s {base}.{{{arm}-{ctrl}}}.probe_auc_acc")
                row["dc"] = dc
                if arm != "keep_all":
                    mac(f"{P}{A}Verdict", R["verdicts"][cond]["per_intervention"][arm]["verdict"].replace("_", r"\_"),
                        "{}", src, f"results.verdicts.{cond}.per_intervention.{arm}.verdict")
            rows[(cond, arm)] = row
        mac(f"{P}InterventionStage", R["verdicts"][cond]["intervention_stage"].replace("_", r"\_"), "{}", src,
            f"results.verdicts.{cond}.intervention_stage")
    mac("OverallVerdict", R["overall"].replace("_", r"\_"), "{}", src, "results.overall")
    parity = "runs/p4_stage1/torch_parity_measured.txt"
    with open(os.path.join(ROOT, parity)) as f:
        line = [x for x in f if x.startswith("vanilla Adam")][0]
    d = json.loads(line.split("max rel diff:")[1].split("|")[0].strip().replace("'", '"'))
    for k, K in (("w", "W"), ("m", "M"), ("v", "V"), ("upd", "Upd")):
        mac(f"Parity{K}", float(d[k]), "{:.1e}", parity, f"vanilla Adam 400 steps max rel diff [{k}]")
    write_tables(R, rows)
    write_curves()
    analysis()
    with open(os.path.join(OUT_T, "numbers.tex"), "w") as f:
        f.write("% AUTO-GENERATED by paper/tools/export_numbers.py from raw result files. Do not edit.\n")
        f.write("\n".join(macros) + "\n")
    with open(os.path.join(OUT_T, "provenance.tsv"), "w") as f:
        f.write("macro\ttext\traw_value\tsource_file\tfield_path\n" + "\n".join(prov) + "\n")
    print(f"wrote {len(macros)} macros")


def fmt_range(xs):
    return f"{mean(xs):+.3f} [{min(xs):+.3f}, {max(xs):+.3f}]"


def write_tables(R, rows):
    L = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"Condition & fresh & early & late & late$-$early (AUC) & late$-$early (final acc.) \\", r"\midrule"]
    for cond, P in CONDS.items():
        seeds = R["phenomenon"][cond]["adam"]["seeds"]
        g = lambda st, k: [seeds[s]["probes"][st][k] for s in SEEDS]  # noqa: E731
        d = [a - b for a, b in zip(g("late", "probe_auc_acc"), g("early", "probe_auc_acc"))]
        df = [a - b for a, b in zip(g("late", "probe_final_acc"), g("early", "probe_final_acc"))]
        name = {"RT": "random-teacher ($C{=}4$)", "LP": "label-permutation ($C{=}10$)"}[P]
        L.append(f"{name} & {mean(g('fresh', 'probe_auc_acc')):.3f} & {mean(g('early', 'probe_auc_acc')):.3f} & "
                 f"{mean(g('late', 'probe_auc_acc')):.3f} & {fmt_range(d)} & {fmt_range(df)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT_T, "tab_phenomenon.tex"), "w") as f:
        f.write("\n".join(L) + "\n")

    L = [r"\begin{tabular}{l cc cc}", r"\toprule",
         r" & \multicolumn{2}{c}{random-teacher ($C{=}4$)} & \multicolumn{2}{c}{label-permutation ($C{=}10$)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Arm & $\Delta$ vs.\ keep & $\Delta$ vs.\ norm ctrl. & $\Delta$ vs.\ keep & $\Delta$ vs.\ norm ctrl. \\",
         r"\midrule"]
    for arm in ARMS:
        cells = []
        for cond in CONDS:
            r = rows[(cond, arm)]
            dk = "---" if arm == "keep_all" else fmt_range(r["dk"])
            dc = fmt_range(r["dc"]) if ("dc" in r and arm != "keep_all") else "n/a"
            cells += [dk, dc]
        L.append(f"{ARM_TEX[arm]} & " + " & ".join(cells) + r" \\")
        if arm == "reset_t":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT_T, "tab_interventions.tex"), "w") as f:
        f.write("\n".join(L) + "\n")

    L = [r"\begin{tabular}{l cccc cccc}", r"\toprule",
         r" & \multicolumn{4}{c}{random-teacher ($C{=}4$)} & \multicolumn{4}{c}{label-permutation ($C{=}10$)} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Arm & $\bar{\|u\|}_{1:10}$ & rel.\ $\Delta\theta$ & CKA & old drop & "
         r"$\bar{\|u\|}_{1:10}$ & rel.\ $\Delta\theta$ & CKA & old drop \\", r"\midrule"]
    for arm in ARMS:
        cells = []
        for cond in CONDS:
            r = rows[(cond, arm)]
            cells += [f"{r['upd']:.3g}", f"{r['theta']:.3g}", f"{r['cka']:.3f}", f"{r['forget']:.3f}"]
        L.append(f"{ARM_TEX[arm]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT_T, "tab_descriptive.tex"), "w") as f:
        f.write("\n".join(L) + "\n")

    L = [r"\begin{tabular}{llcl}", r"\toprule", r"Condition & Family & selected LR & dev online acc.\ by LR \\",
         r"\midrule"]
    for cond, P in CONDS.items():
        for fam in ("adam", "adam_eqbeta", "sgd"):
            t = R["tuning"][cond][fam]
            grid = ", ".join(f"{float(k):g}: {v:.3f}" for k, v in t["rows"].items())
            L.append(f"{P} & \\texttt{{{fam.replace('_', chr(92) + '_')}}} & {t['best']['lr']:g} & {grid} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT_T, "tab_tuning.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def write_curves():
    curves = {}
    with gzip.open(os.path.join(ROOT, RUN, "log.jsonl.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") != "probe_eval":
                continue
            if r.get("phase") == "probe" and r.get("family") == "adam":
                key = (r["cond"], r["start"])
            elif r.get("phase") == "intervention" and r["arm"] in ("reset_m", "reset_t", "reset_both",
                                                                  "reset_both__shared_keep"):
                key = (r["cond"], "iv_" + r["arm"])
            else:
                continue
            curves.setdefault(key, {}).setdefault(r["k"], {})[str(r["seed"])] = r["probe_acc"]
    for (cond, name), by_k in curves.items():
        lines = ["k mean " + " ".join(f"s{s}" for s in SEEDS)]
        for k in sorted(by_k):
            v = [by_k[k][s] for s in SEEDS]
            lines.append(f"{k} {mean(v):.4f} " + " ".join(f"{x:.4f}" for x in v))
        with open(os.path.join(OUT_F, f"curve_{CONDS[cond]}_{name}.dat"), "w") as f:
            f.write("\n".join(lines) + "\n")


def analysis():
    path = os.path.join(ROOT, ANALYSIS)
    if not os.path.exists(path):
        return
    with open(path) as f:
        a = json.load(f)
    for name, spec in a.get("macros", {}).items():
        mac(name, spec["value"], spec["fmt"], ANALYSIS, spec["path"])


if __name__ == "__main__":
    main()
