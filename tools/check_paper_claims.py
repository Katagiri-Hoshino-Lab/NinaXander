#!/usr/bin/env python3
"""Gate: every headline number printed in paper/paper_ja.tex must equal the
value derived from the canonical tables in artifacts/metrics/tables/.

Each check recomputes a value from the tables, renders it with the paper's
rounding convention (half-up at printed precision), and requires the exact
LaTeX fragment to appear in the tex source. If an evaluation is re-run and a
canonical value drifts, this gate fails until the paper is updated, so prose
and data cannot diverge silently. Data-side invariants that back qualitative
sentences ("all 32 layers", "all 16 comparisons significant") are asserted
directly on the tables.
"""

import argparse
import csv
import json
import pathlib
import re
import struct
import sys
from decimal import ROUND_HALF_UP, Decimal


def fmt(value, decimals):
    quantum = Decimal(1).scaleb(-decimals) if decimals > 0 else Decimal(1)
    return str(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def fmt_p(p):
    return fmt(p, 3) if float(p) < 0.0995 else fmt(p, 2)


def signed(text):
    return text if text.startswith("-") else "+" + text


def comma(step):
    return f"{int(step):,}".replace(",", "{,}")


def strip_revision_marks(text):
    """Unwrap the co-authors' colour-coded revision macros (\\rev{...}, \\revo{...}, ...).

    The wrappers only colour the text in the PDF; a claim must match whether or not a
    revision round is still marked up.
    """
    out, i = [], 0
    pattern = re.compile(r"\\rev[a-z]?\{")
    while True:
        m = pattern.search(text, i)
        if not m:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:m.start()])
        depth, j = 1, m.end()
        while depth:
            if j >= len(text):
                line = text.count("\n", 0, m.start()) + 1
                raise SystemExit(f"unbalanced {m.group(0)}...}} starting at line {line}")
            if text[j] == "\\":
                j += 2
                continue
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        out.append(strip_revision_marks(text[m.end():j - 1]))
        i = j


def load(table_dir, name):
    with open(table_dir / name, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safetensors_param_count(path):
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    total = 0
    for key, spec in header.items():
        if key == "__metadata__":
            continue
        count = 1
        for dim in spec["shape"]:
            count *= dim
        total += count
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", default="paper/paper_ja.tex")
    parser.add_argument("--table-dir", default="artifacts/metrics/tables")
    parser.add_argument("--release-dir", default="release/huggingface")
    args = parser.parse_args()

    table_dir = pathlib.Path(args.table_dir)
    release_dir = pathlib.Path(args.release_dir)
    raw_tex = pathlib.Path(args.tex).read_text(encoding="utf-8")
    # Line breaks and spaces are typographically irrelevant in the fragments we
    # pin, so matching is done on a space-free form of both sides.
    uncommented = re.sub(r"(?<!\\)((?:\\\\)*)%.*", r"\1", raw_tex)
    tex = re.sub(r"\s+", "", strip_revision_marks(uncommented))

    failures = []
    passed = 0

    def check(claim, fragment):
        nonlocal passed
        if re.sub(r"\s+", "", fragment) in tex:
            passed += 1
        else:
            failures.append(f"{claim}: fragment not in tex: {fragment!r}")

    def invariant(claim, condition, detail=""):
        nonlocal passed
        if condition:
            passed += 1
        else:
            failures.append(f"{claim}: data invariant violated {detail}")

    rep = {r["metric"]: float(r["value"]) for r in load(table_dir, "paper_representation.csv")}
    aa, ab = rep["self_reconstruction_a_r2"], rep["cross_readout_a_to_b_r2"]
    bb, ba = rep["self_reconstruction_b_r2"], rep["cross_readout_b_to_a_r2"]
    rho, raw_corr, f_var = (
        rep["latent_correlation_centered"],
        rep["latent_correlation_raw"],
        rep["latent_alignment_f"],
    )

    check("cross read-out headline", f"$A\\!\\to\\!B$ で ${fmt(ab, 3)}$，$B\\!\\to\\!A$ で ${fmt(ba, 3)}$ に達し")
    check("self read-out headline", f"$A\\!\\to\\!A\\ {fmt(aa, 3)}$，$B\\!\\to\\!B\\ {fmt(bb, 3)}$ は")
    check(
        "rho vs raw",
        f"$\\rho_{{\\mathrm{{raw}}}}$ は ${fmt(raw_corr, 3)}$ であり，中心化相関 ${fmt(rho, 3)}$ より高い",
    )
    check(
        "common-mode identity",
        f"$(1{{-}}{fmt(f_var, 3)})+{fmt(f_var, 3)}\\times{fmt(rho, 3)}={fmt(raw_corr, 3)}$",
    )
    check("constant share", f"潜在の二乗ノルムの ${fmt((1 - f_var) * 100, 0)}\\%$ は")
    invariant(
        "identity residual",
        abs((1 - f_var) + f_var * rho - raw_corr) < 1e-3,
        f"|(1-f)+f*rho-raw| = {abs((1 - f_var) + f_var * rho - raw_corr):.2e}",
    )

    layers = load(table_dir, "paper_layer_metrics.csv")
    per_layer = {int(r["layer"]): r for r in layers if r["row_type"] == "layer"}
    mean_all = next(r for r in layers if r["row_type"] == "mean_0_31")
    invariant(
        "AB<=BB and BA<=AA in all 32 layers",
        all(
            float(r["A->B"]) <= float(r["B->B"]) and float(r["B->A"]) <= float(r["A->A"])
            for r in per_layer.values()
        )
        and len(per_layer) == 32,
    )
    check(
        "shuffle collapse rho",
        f"$\\rho_{{\\mathrm{{ctr}}}}$ は ${fmt(float(mean_all['rho_ctr']), 3)}\\!\\to\\!{fmt(float(mean_all['rho_ctr_shuf']), 3)}$，",
    )
    check(
        "shuffle collapse readouts",
        f"$A\\!\\to\\!B$ で ${fmt(float(mean_all['A->B']), 3)}\\!\\to\\!{fmt(float(mean_all['A->B_shuf']), 3)}$，"
        f"$B\\!\\to\\!A$ で ${fmt(float(mean_all['B->A']), 3)}\\!\\to\\!{fmt(float(mean_all['B->A_shuf']), 3)}$ へ低下する",
    )
    local_ab = "/".join(fmt(float(per_layer[l]["A->B"]), 3) for l in (4, 8, 16, 24))
    local_ba = "/".join(fmt(float(per_layer[l]["B->A"]), 3) for l in (4, 8, 16, 24))
    check("local per-layer readouts", f"\\RtoP{{}} で ${local_ab}$，\\PtoR{{}} で ${local_ba}$")

    curve = {int(r["step"]): r for r in load(table_dir, "paper_training_curve.csv")}
    last_step = max(curve)
    best_step = max(curve, key=lambda s: float(curve[s]["eval_a_to_b_r2"]))
    check(
        "training run length",
        f"${comma(last_step)}$ 反復を実行し，評価用データの $A\\!\\to\\!B$ が最良となった "
        f"${comma(best_step)}$ 反復目",
    )
    check("reported checkpoint", f"${comma(best_step)}$ 反復目のチェックポイントを以降のすべての評価に用いる")
    gen_rows = load(table_dir, "paper_cross_family_generation_samples.csv")
    used_steps = (
        {int(r["checkpoint_step"]) for r in load(table_dir, "paper_representation.csv")}
        | {int(r["checkpoint_step"]) for r in load(table_dir, "paper_cross_family_linearity.csv")}
        | {int(r["checkpoint_step"]) for r in gen_rows if r["checkpoint_step"]}
    )
    invariant("every evaluation uses the selected checkpoint", used_steps == {best_step}, f"{used_steps} vs {best_step}")
    check("selected checkpoint f", f"${comma(best_step)}$ 反復目のチェックポイントの $f$ は ${fmt(f_var, 3)}$ であり")
    check("generation checkpoint", f"${comma(best_step)}$ 反復目のチェックポイントのアダプタを")
    gen_chimeras = sorted({r["config"] for r in gen_rows if "_to_" in r["config"]})
    invariant("generation examples use the two L=4 chimeras", gen_chimeras == ["A_to_B@4", "B_to_A@4"], f"{gen_chimeras}")
    check("generation scope in the conclusion", f"$L{{=}}4$ の ${len(gen_chimeras)}$ 構成は構文的に妥当な文を生成した")
    lrs = [float(curve[s]["learning_rate"]) for s in sorted(curve)]
    invariant("learning rate decays over the printed range", max(lrs) == 1e-3 and min(lrs) == 1e-5 and lrs == sorted(lrs, reverse=True))
    check("learning-rate range", "学習率の初期値を $10^{-3}$，下限を $10^{-5}$ とした")
    gaps = {b - a for a, b in zip(sorted(curve), sorted(curve)[1:])}
    invariant("the curve is logged at one evaluation interval", len(gaps) == 1, f"{gaps}")
    check("evaluation interval", f"${comma(min(gaps, default=-1))}$ 反復ごとに評価用データ")
    ident_err = {
        s: abs((1 - float(r["latent_f"])) + float(r["latent_f"]) * float(r["latent_rho_centered"]) - float(r["latent_rho_raw"]))
        for s, r in curve.items()
    }
    late_err = max(v for s, v in ident_err.items() if s >= last_step // 2)
    check("identity error after training", f"${fmt(late_err * 1e4, 0)}\\!\\times\\!10^{{-4}}$ 程度の誤差で成り立つ")
    check("identity error at iteration 0", f"$0$ 反復目では誤差が ${fmt(ident_err[0] * 1e2, 1)}\\!\\times\\!10^{{-2}}$ に達する")
    # The paper no longer prints the common-mode training-dynamics analysis, so the fragments
    # that pinned it are gone. The equal-KV early-exit baseline and the serving measurements
    # are printed again and are pinned further below.

    corr = load(table_dir, "paper_layer_correspondence.csv")
    diag = [float(r["linear_cka"]) for r in corr if r["source_layer"] == r["target_layer"]]
    off = [float(r["linear_cka"]) for r in corr if r["source_layer"] != r["target_layer"]]
    best_target = {}
    for r in corr:
        s, t, v = int(r["source_layer"]), int(r["target_layer"]), float(r["linear_cka"])
        if s not in best_target or v > best_target[s][1]:
            best_target[s] = (t, v)
    n_diag = sum(1 for s, (t, _) in best_target.items() if s == t)
    n_zero = sum(1 for t, _ in best_target.values() if t == 0)
    n_last = sum(1 for t, _ in best_target.values() if t == 31)
    check(
        "CKA means",
        f"対角要素の平均は ${fmt(sum(diag) / len(diag), 2)}$，それ以外の層の組の平均は ${fmt(sum(off) / len(off), 2)}$",
    )
    check(
        "CKA argmax counts",
        f"$32$ 行中 ${n_diag}$ 行にすぎない．$A$ の各層に最も類似する $B$ の層は，"
        f"$32$ 層のうち ${n_zero}$ 層で層 $0$，${n_last}$ 層で最終層である",
    )
    lin_diag = [float(r["best_linear_r2"]) for r in corr if r["source_layer"] == r["target_layer"]]
    lin_off = [float(r["best_linear_r2"]) for r in corr if r["source_layer"] != r["target_layer"]]
    lin_best = {}
    for r in corr:
        s, t, v = int(r["source_layer"]), int(r["target_layer"]), float(r["best_linear_r2"])
        if s not in lin_best or v > lin_best[s][1]:
            lin_best[s] = (t, v)
    check(
        "off-diagonal linear regressability",
        f"適合 $R^2$ の平均は対角要素で ${fmt(sum(lin_diag) / len(lin_diag), 2)}$，非対角要素で ${fmt(sum(lin_off) / len(lin_off), 2)}$",
    )
    check(
        "linear argmax same-layer count",
        f"同じ番号の層が最良となる行は $32$ 行中 ${sum(1 for s, (t, _) in lin_best.items() if s == t)}$ 行",
    )
    all_lin = lin_diag + lin_off
    shapes = {(int(r["n_windows"]), int(r["window_tokens"]), int(r["sample_rows"])) for r in corr}
    invariant(
        "all-layer rows = sequences x tokens",
        len(shapes) == 1 and all(n * w == rows for n, w, rows in shapes),
        f"shapes = {shapes}",
    )
    nwin, win, rows = next(iter(shapes))
    check("all-layer row source", f"${win}$ トークンの系列 ${nwin}$ 本，すなわち各層の組に共通する ${comma(rows)}$ 行")
    check(
        "figure caption fit range",
        f"適合 $R^2$ は ${fmt(min(all_lin), 2)}$--${fmt(max(all_lin), 2)}$ であり，ほぼすべての組で高い",
    )

    lin_rows = load(table_dir, "paper_cross_family_linearity.csv")
    lin = {r["execution_path"]: r for r in lin_rows if r["scope"] == "mean"}
    layer_rows = [r for r in lin_rows if r["scope"] == "layer"]
    rep_layers = sorted({int(r["layer"]) for r in layer_rows})
    main_layers = {int(r["layers"]) for r in load(table_dir, "paper_representation.csv")}
    fit_rows = {int(r["fit_rows"]) for r in layer_rows}
    eval_rows = {int(r["evaluation_rows"]) for r in layer_rows}
    invariant("linearity per-layer row counts are uniform", len(fit_rows) == 1 and len(eval_rows) == 1, f"{fit_rows} {eval_rows}")
    invariant("representation means are over one layer count", len(main_layers) == 1, f"{main_layers}")
    check(
        "linearity representative layers",
        f"同じ代表 ${len(rep_layers)}$ 層 $\\{{{','.join(map(str, rep_layers))}\\}}$",
    )
    check(
        "linearity fit/evaluation rows",
        f"前半の ${comma(next(iter(fit_rows)))}$ 行で適合し，後半の ${comma(next(iter(eval_rows)))}$ 行で評価する",
    )
    aff_ab, aff_ba = float(lin["A_to_B"]["best_affine_cross_r2"]), float(lin["B_to_A"]["best_affine_cross_r2"])
    fit_ab, fit_ba = float(lin["A_to_B"]["linearised_fit_r2"]), float(lin["B_to_A"]["linearised_fit_r2"])
    ada_ab, ada_ba = float(lin["A_to_B"]["adapter_cross_r2"]), float(lin["B_to_A"]["adapter_cross_r2"])
    invariant("direct affine predicts about half of the variance", all(0.4 < v < 0.6 for v in (aff_ab, aff_ba)), f"{aff_ab} {aff_ba}")
    check(
        "direct affine share of the adapter",
        f"の残差の分散の約半分を予測でき，アダプタの値の ${fmt(aff_ab / ada_ab * 100, 0)}\\%$／${fmt(aff_ba / ada_ba * 100, 0)}\\%$ に達する",
    )
    steps = {b - a for a, b in zip(rep_layers, rep_layers[1:])}
    invariant("representative layers are evenly spaced", len(steps) == 1, f"{rep_layers}")
    check(
        "representative layer spacing",
        f"代表 ${len(rep_layers)}$ 層は，層 ${rep_layers[0]}$ から層 ${rep_layers[-1]}$ まで ${min(steps, default=-1)}$ 層ごとに取った",
    )
    check(
        "linearity adapter values vs 32-layer means",
        f"これらは代表 ${len(rep_layers)}$ 層で評価した値であり，\\S\\ref{{sec:space}}の ${next(iter(main_layers))}$ 層平均 "
        f"${fmt(ab, 3)}$／${fmt(ba, 3)}$ とは層も行も異なる",
    )
    # The abstract and introduction no longer restate the direct-affine reach; it is pinned where the
    # linearity subsection prints it ("direct affine reach" below).
    check(
        "nonaffine share",
        f"残りの ${fmt((1 - fit_ab) * 100, 1)}\\%$／${fmt((1 - fit_ba) * 100, 1)}\\%$ は",
    )
    check("affine imitation", f"\\RtoP{{}} で ${fmt(fit_ab * 100, 1)}\\%$，\\PtoR{{}} で ${fmt(fit_ba * 100, 1)}\\%$ である")
    check("direct affine reach", f"最良の直接アフィン写像は \\RtoP{{}} で ${fmt(aff_ab, 3)}$，\\PtoR{{}} で ${fmt(aff_ba, 3)}$ に達する")
    check("adapter on same rows", f"同じ行でのアダプタの値は \\RtoP{{}} で ${fmt(ada_ab, 3)}$，\\PtoR{{}} で ${fmt(ada_ba, 3)}$")
    check("nonlinear increment", f"増分は $+{fmt(ada_ab - aff_ab, 3)}$／$+{fmt(ada_ba - aff_ba, 3)}$")

    mlp = load(table_dir, "paper_cross_family_mlp_intervention.csv")
    sweep = {}
    for r in mlp:
        if r["result_type"] == "representation_r2":
            sweep.setdefault((r["execution_path"], float(r["alpha"])), []).append(float(r["value"]))
    alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
    sweep_ab = ",".join(fmt(sum(sweep[("A_to_B", a)]) / len(sweep[("A_to_B", a)]), 2) for a in alphas)
    sweep_ba = ",".join(fmt(sum(sweep[("B_to_A", a)]) / len(sweep[("B_to_A", a)]), 2) for a in alphas)
    check("alpha sweep", f"\\RtoP{{}} で ${sweep_ab}$，\\PtoR{{}} で ${sweep_ba}$ となる")
    qa_cells = {}
    for r in mlp:
        if r["result_type"] == "qa_accuracy_norm":
            qa_cells[(r["execution_path"], r["task"], int(r["layer"]), float(r["alpha"]))] = float(r["value"])
    branch = {
        path: [
            (qa_cells[(path, task, layer, 1.0)] - qa_cells[(path, task, layer, 0.0)]) * 100
            for task in ("arc_easy", "sciq")
            for layer in (4, 8, 16, 24)
        ]
        for path in ("A_to_B", "B_to_A")
    }
    check(
        "branch effect ranges",
        f"\\RtoP{{}} で $+{fmt(min(branch['A_to_B']), 1)}$--$+{fmt(max(branch['A_to_B']), 1)}$ 点，"
        f"\\PtoR{{}} で $+{fmt(min(branch['B_to_A']), 1)}$--$+{fmt(max(branch['B_to_A']), 1)}$ 点",
    )

    pairs = load(table_dir, "paper_cross_family_paired_statistics.csv")
    mlp_rows = [r for r in pairs if r["comparison_family"] == "mlp_intervention"]
    invariant(
        "16 MLP McNemar p < 2e-9",
        len(mlp_rows) == 16 and all(float(r["mcnemar_p"]) < 2e-9 for r in mlp_rows),
    )

    def pair_row(task, config_a, config_b):
        return next(
            r for r in pairs if r["task"] == task and r["config_a"] == config_a and r["config_b"] == config_b
        )

    # Table 4 (tab:paired): accuracy differences in points and McNemar p-values against RWKV and between the
    # two directions. These rows replace the prose that used to carry the same comparisons.
    def table_p(p):
        return "<10^{-4}" if float(p) < 1e-4 else fmt_p(p)

    for layer in (4, 8, 16, 24):
        cells = [
            pair_row(task, config_a, config_b)
            for task in ("arc_easy", "sciq")
            for config_a, config_b in (
                (f"A_to_B@{layer}", "rwkv"),
                (f"B_to_A@{layer}", "rwkv"),
                (f"B_to_A@{layer}", f"A_to_B@{layer}"),
            )
        ]
        check(
            f"paired table L={layer} differences",
            f"{layer} & "
            + " & ".join(f"${signed(fmt(float(r['accuracy_difference']) * 100, 1))}$" for r in cells)
            + "\\\\",
        )
        check(
            f"paired table L={layer} p-values",
            "& " + " & ".join(f"(${table_p(r['mcnemar_p'])}$)" for r in cells) + "\\\\",
        )
    cross_vs_rwkv = [
        pair_row(task, f"{path}@{layer}", "rwkv")
        for path in ("A_to_B", "B_to_A")
        for layer in (4, 8, 16, 24)
        for task in ("arc_easy", "sciq")
    ]
    invariant(
        "only RWKV->Pythia L=4 on SciQ significantly exceeds RWKV",
        [
            (r["config_a"], r["task"])
            for r in cross_vs_rwkv
            if float(r["accuracy_difference"]) > 0 and float(r["mcnemar_p"]) < 0.05
        ]
        == [("A_to_B@4", "sciq")],
    )
    check(
        "introduction: only one chimera beats RWKV, on SciQ",
        "RWKV 単体を正答率で有意に上回ったキメラモデルも，理科の問題集 SciQ における一つのみであった",
    )
    vs_pythia = [
        next(
            r
            for r in pairs
            if r["task"] == task and {r["config_a"], r["config_b"]} == {f"{path}@{layer}", "pythia"}
        )
        for path in ("A_to_B", "B_to_A")
        for layer in (4, 8, 16, 24)
        for task in ("arc_easy", "sciq")
    ]
    invariant(
        "all 16 cross configurations significantly below Pythia with p <= 0.003",
        all(float(r["mcnemar_p"]) <= 0.003 for r in vs_pythia)
        and all((float(r["accuracy_difference"]) < 0) == (r["config_b"] == "pythia") for r in vs_pythia),
    )
    check("Pythia comparison p bound", "$p\\le0.003$")

    four = load(table_dir, "paper_four_path_qa_accuracy.csv")
    acc = {
        (r["execution_path"], r["task"], int(r["switch_layer"])): float(r["accuracy_norm"]) * 100
        for r in four
    }
    gaps = {
        task: [acc[("B_to_B", task, l)] - acc[("A_to_B", task, l)] for l in (4, 8, 16, 24)]
        for task in ("arc_easy", "sciq")
    }
    check(
        "Pythia self-recon vs RWKV->Pythia gap ranges",
        f"\\Pythiaself{{}} と \\RtoP{{}} の差は ARC で ${fmt(min(gaps['arc_easy']), 0)}$--${fmt(max(gaps['arc_easy']), 0)}$ 点，"
        f"SciQ で ${fmt(min(gaps['sciq']), 0)}$--${fmt(max(gaps['sciq']), 0)}$ 点",
    )
    ba_ab_16_24 = {
        (task, l): pair_row(task, f"B_to_A@{l}", f"A_to_B@{l}") for task in ("arc_easy", "sciq") for l in (16, 24)
    }
    # The L=16/24 direction gaps themselves are pinned by the Table 4 rows above.
    invariant(
        "BA vs AB deep p < 3e-15",
        all(float(r["mcnemar_p"]) < 3e-15 for r in ba_ab_16_24.values()),
    )
    # The L=4 direction gap is pinned by the Table 4 L=4 rows above.

    qa = load(table_dir, "paper_cross_family_qa_accuracy.csv")
    qa_acc = {(r["config"], r["task"]): float(r["accuracy_norm"]) * 100 for r in qa}
    check(
        "results: only R->P L=4 on SciQ beats RWKV",
        f"RWKV（ARC ${fmt(qa_acc[('rwkv', 'arc_easy')], 1)}\\%$，SciQ ${fmt(qa_acc[('rwkv', 'sciq')], 1)}\\%$）"
        "を有意に上回るのは SciQ における \\RtoPL{4} のみであり",
    )
    ab_affine_gaps = [
        qa_acc[(f"A_to_B@{l}", task)] - qa_acc[(f"A_to_B_affine@{l}", task)]
        for task in ("arc_easy", "sciq")
        for l in (4, 8, 16, 24)
    ]
    check(
        "AB adapter vs affine range",
        f"正答率は ${fmt(min(ab_affine_gaps), 0)}$--${fmt(max(ab_affine_gaps), 0)}$ 点低下する",
    )
    # Table 3 (tab:direction): four-path accuracies, each task's row maximum in bold.
    paths = ("A_to_A", "A_to_B", "B_to_B", "B_to_A")
    for layer in (4, 8, 16, 24):
        best = {task: max(paths, key=lambda p: acc[(p, task, layer)]) for task in ("arc_easy", "sciq")}
        cells = []
        for path in paths:
            for task in ("arc_easy", "sciq"):
                text = fmt(acc[(path, task, layer)], 1)
                cells.append(f"\\textbf{{{text}}}" if best[task] == path else text)
        check(f"four-path table L={layer}", f"{layer} & " + " & ".join(cells) + "\\\\")
    by_layer = lambda path, task: [acc[(path, task, l)] for l in (4, 8, 16, 24)]
    falls = lambda v: all(a > b for a, b in zip(v, v[1:]))
    rises = lambda v: all(a < b for a, b in zip(v, v[1:]))
    invariant(
        "RWKV->Pythia falls monotonically with switch depth",
        all(falls(by_layer("A_to_B", t)) for t in ("arc_easy", "sciq")),
    )
    invariant(
        "Pythia->RWKV is non-monotonic in switch depth",
        all(not (falls(v) or rises(v)) for v in (by_layer("B_to_A", t) for t in ("arc_easy", "sciq"))),
    )
    mean_ab4 = (acc[("A_to_B", "arc_easy", 4)] + acc[("A_to_B", "sciq", 4)]) / 2
    mean_ba4 = (acc[("B_to_A", "arc_easy", 4)] + acc[("B_to_A", "sciq", 4)]) / 2
    check("two-task means", f"${fmt(mean_ab4, 2)}\\%\\to{fmt(mean_ba4, 2)}\\%$ と同等に保つ")
    sum_acc = {
        (r["config"], r["task"]): float(r["accuracy_sum"]) * 100 for r in qa if r["accuracy_sum"]
    }
    check(
        "sum-scoring robustness note",
        f"\\RtoPL{{4}} が SciQ で ${fmt(sum_acc[('A_to_B@4', 'sciq')] - sum_acc[('rwkv', 'sciq')], 1)}$ 点，"
        f"\\PtoRL{{4}} が ARC で ${fmt(sum_acc[('B_to_A@4', 'arc_easy')] - sum_acc[('rwkv', 'arc_easy')], 1)}$ 点，"
        f"SciQ で ${fmt(sum_acc[('B_to_A@4', 'sciq')] - sum_acc[('rwkv', 'sciq')], 1)}$ 点となる",
    )
    invariant(
        "core claims robust under sum scoring",
        all(
            sum_acc[(f"{p}@{l}", t)] < sum_acc[("pythia", t)]
            for p in ("A_to_B", "B_to_A")
            for l in (4, 8, 16, 24)
            for t in ("arc_easy", "sciq")
        )
        and all(
            sum_acc[(f"B_to_A@{l}", t)] > sum_acc[(f"A_to_B@{l}", t)]
            for l in (16, 24)
            for t in ("arc_easy", "sciq")
        ),
    )

    mem = {r["config"]: r for r in load(table_dir, "paper_cross_family_memory_accuracy.csv")}
    ba4, parent_b, parent_a = mem["B_to_A@4"], mem["pure-Pythia"], mem["pure-RWKV"]
    check("KV reduction headline", f"KV キャッシュを ${fmt(float(ba4['kv_reduction_fraction']) * 100, 1)}\\%$ 削減しつつ")
    check(
        "RWKV state size",
        f"層あたり ${fmt(float(ba4['rwkv_state_kib_per_layer']), 0)}$ KiB の再帰状態を持つ．"
        "この値は，参照実装が保持する $5$ 個の状態ベクトル",
    )
    check(
        "table5 pythia row",
        f"P 単体 & -- & {parent_b['transformer_layers']} & {fmt(float(parent_b['kv_kib_per_token']), 0)} & {fmt(float(parent_b['kv_gib_at_context']), 2)} & 0 & "
        f"{fmt(float(parent_b['arc_easy_accuracy_norm']) * 100, 1)}/{fmt(float(parent_b['sciq_accuracy_norm']) * 100, 1)}\\\\",
    )
    check(
        "table5 rwkv row",
        f"{fmt(float(parent_a['arc_easy_accuracy_norm']) * 100, 1)} / {fmt(float(parent_a['sciq_accuracy_norm']) * 100, 1)}",
    )
    check(
        "table5 BA@4 row",
        f"P$\\to$R & {ba4['switch_layer']} & {ba4['transformer_layers']} & {fmt(float(ba4['kv_kib_per_token']), 0)} & {fmt(float(ba4['kv_gib_at_context']), 2)} & "
        f"{fmt(float(ba4['kv_reduction_fraction']) * 100, 1)} & "
        f"{fmt(float(ba4['arc_easy_accuracy_norm']) * 100, 1)}/{fmt(float(ba4['sciq_accuracy_norm']) * 100, 1)}\\\\",
    )
    frontier = sorted(
        ((float(r["kv_reduction_fraction"]), float(r["sciq_accuracy_norm"]), c) for c, r in mem.items()),
        key=lambda x: (-x[0], -x[1]),
    )
    pareto, best_acc = [], -1.0
    for red, acc_v, config in sorted(frontier, key=lambda x: -x[0]):
        if acc_v > best_acc:
            pareto.append(config)
            best_acc = acc_v
    invariant(
        "SciQ frontier membership",
        set(pareto) == {"pure-Pythia", "A_to_B@4", "B_to_A@24", "B_to_A@4", "pure-RWKV"},
        f"pareto = {sorted(pareto)}",
    )

    shift = load(table_dir, "paper_cross_family_domain_shift.csv")
    ppl = {
        (r["config"], r["domain"], int(r["context_tokens"])): float(r["perplexity"])
        for r in shift
        if "affine" not in r["config"]
    }
    check(
        "domain shift parents ctx512",
        f"RWKV が Alpaca で ${fmt(ppl[('pure-RWKV', 'alpaca', 512)], 2)}$，WikiText で ${fmt(ppl[('pure-RWKV', 'wikitext', 512)], 2)}$，"
        f"Pythia がそれぞれ ${fmt(ppl[('pure-Pythia', 'alpaca', 512)], 2)}$ と ${fmt(ppl[('pure-Pythia', 'wikitext', 512)], 2)}$ である",
    )
    check(
        "domain shift chimeras ctx512",
        f"\\RtoPL{{4}} は Alpaca で ${fmt(ppl[('A_to_B@4', 'alpaca', 512)], 2)}$，WikiText で ${fmt(ppl[('A_to_B@4', 'wikitext', 512)], 2)}$，"
        f"\\PtoRL{{4}} は ${fmt(ppl[('B_to_A@4', 'alpaca', 512)], 2)}$ と ${fmt(ppl[('B_to_A@4', 'wikitext', 512)], 2)}$ であり",
    )
    check(
        "domain shift ctx2048 BA",
        f"\\PtoRL{{4}} は Alpaca で ${fmt(ppl[('B_to_A@4', 'alpaca', 2048)], 2)}$，"
        f"WikiText で ${fmt(ppl[('B_to_A@4', 'wikitext', 2048)], 2)}$ となり",
    )
    ratio = ppl[("B_to_A@4", "wikitext", 2048)] / ppl[("pure-RWKV", "wikitext", 2048)]
    check(
        "domain shift ratio",
        f"WikiText の ${fmt(ppl[('B_to_A@4', 'wikitext', 2048)], 2)}$ は，同じ文脈長で perplexity の低い親モデルである RWKV の "
        f"${fmt(ppl[('pure-RWKV', 'wikitext', 2048)], 2)}$ の ${fmt(ratio, 1)}$ 倍",
    )
    for config, x in (("pure-RWKV", 11), ("pure-Pythia", 33), ("A_to_B@4", 55), ("B_to_A@4", 77)):
        check(
            f"figure6 {config}",
            f"{{{fmt(ppl[(config, 'alpaca', 512)], 2)}}}{{{fmt(ppl[(config, 'wikitext', 512)], 2)}}}",
        )

    # Domain-shift affine control, refit per evaluated text on a disjoint prefix of that text.
    ppl_all = {
        (r["config"], r["domain"], int(r["context_tokens"])): float(r["perplexity"]) for r in shift
    }
    check(
        "domain shift affine BA@4 wikitext",
        f"\\PtoRL{{4}} では文脈長 $512$ で ${fmt(ppl_all[('B_to_A_affine@4', 'wikitext', 512)], 2)}$ 対 "
        f"${fmt(ppl_all[('B_to_A@4', 'wikitext', 512)], 2)}$，文脈長 $2048$ で "
        f"${fmt(ppl_all[('B_to_A_affine@4', 'wikitext', 2048)], 2)}$ 対 ${fmt(ppl_all[('B_to_A@4', 'wikitext', 2048)], 2)}$",
    )
    check(
        "domain shift affine BA@4 alpaca",
        f"文脈長 $512$ で ${fmt(ppl_all[('B_to_A_affine@4', 'alpaca', 512)], 2)}$ 対 "
        f"${fmt(ppl_all[('B_to_A@4', 'alpaca', 512)], 2)}$ である",
    )
    grid = [(layer, context) for layer in (4, 8, 16, 24) for context in (512, 1024, 2048)]
    invariant(
        "Pythia->RWKV affine control beats the adapter on WikiText everywhere",
        all(ppl_all[(f"B_to_A_affine@{l}", "wikitext", c)] < ppl_all[(f"B_to_A@{l}", "wikitext", c)] for l, c in grid),
    )
    invariant(
        "RWKV->Pythia affine control is no better than the adapter on WikiText anywhere",
        all(ppl_all[(f"A_to_B_affine@{l}", "wikitext", c)] >= ppl_all[(f"A_to_B@{l}", "wikitext", c)] for l, c in grid),
    )
    fit_bundle = table_dir.parent / "raw" / "b_to_a_longctx_wikitext.json"
    if fit_bundle.exists():
        check("domain shift affine fit rows", f"先頭の ${comma(json.loads(fit_bundle.read_text(encoding='utf-8'))['fit_rows_actual'])}$ トークン")

    # Equal-KV Pythia early exits (tab:earlyexit): chimera minus early exit, ordered by kept Transformer layers.
    exits = [r for r in pairs if r["comparison_family"] == "equal_memory"]
    exit_rows = {}
    for r in exits:
        exit_rows.setdefault(r["config_b"], {})[r["task"]] = r
    ordered = sorted(exit_rows, key=lambda c: int(exit_rows[c]["arc_easy"]["config_a"].split("@")[1]))
    for config in ordered:
        arc, sciq = exit_rows[config]["arc_easy"], exit_rows[config]["sciq"]
        kept = int(arc["config_a"].split("@")[1])
        label = "P$\\to$R" if config.startswith("B_to_A") else "R$\\to$P"
        layer = int(config.split("@")[1])
        check(
            f"early-exit table {config}",
            f"{label} & {layer} & {kept} & "
            f"${signed(fmt(-float(arc['accuracy_difference']) * 100, 1))}$ & ${table_p(arc['mcnemar_p'])}$ & "
            f"${signed(fmt(-float(sciq['accuracy_difference']) * 100, 1))}$ & ${table_p(sciq['mcnemar_p'])}$\\\\",
        )
    invariant(
        "chimeras keeping <=17 Transformer layers beat the equal-KV early exit on both tasks (p<1e-4)",
        all(
            float(r["accuracy_difference"]) < 0 and float(r["mcnemar_p"]) < 1e-4
            for r in exits
            if int(r["config_a"].split("@")[1]) <= 17
        ),
    )
    le17 = {r["config_b"] for r in exits if int(r["config_a"].split("@")[1]) <= 17}
    ge23 = {r["config_b"] for r in exits if int(r["config_a"].split("@")[1]) >= 23}
    check("early-exit <=17 configuration count", f"Transformer 層を $17$ 層以下に減らす ${len(le17)}$ 構成では，両課題とも")
    check("early-exit >=23 configuration count", f"$23$ 層以上を残す ${len(ge23)}$ 構成では差が小さく")
    check(
        "conclusion: early-exit qualification",
        "早期終了を両課題で大きく上回るのは，Transformer 層を $17$ 層以下に減らす構成である",
    )
    invariant(
        "among chimeras keeping >=23 layers only RWKV->Pythia L=8 on ARC differs significantly",
        [
            (r["config_b"], r["task"])
            for r in exits
            if int(r["config_a"].split("@")[1]) >= 23 and float(r["mcnemar_p"]) < 0.05
        ]
        == [("A_to_B@8", "arc_easy")],
    )
    ba4_exit = exit_rows["B_to_A@4"]
    check(
        "early exit BA@4 accuracies",
        f"ARC で ${fmt(float(ba4_exit['arc_easy']['accuracy_b']) * 100, 1)}\\%$ 対 "
        f"${fmt(float(ba4_exit['arc_easy']['accuracy_a']) * 100, 1)}\\%$，SciQ で "
        f"${fmt(float(ba4_exit['sciq']['accuracy_b']) * 100, 1)}\\%$ 対 ${fmt(float(ba4_exit['sciq']['accuracy_a']) * 100, 1)}\\%$",
    )

    # Serving: unpruned prefill latency of the parents, pruned chimera prefill, and pruned resident weights.
    serving = load(table_dir, "paper_cross_family_serving.csv")
    def prefill(benchmark, producer, config, context):
        return float(next(
            r["ms"] for r in serving
            if r["benchmark"] == benchmark and r["phase"] == "prefill" and r["producer_path"] == producer
            and r["config"] == config and r["context_tokens"] == str(context)
        ))
    r512, p512 = prefill("unpruned", "A_to_B", "pure-RWKV", 512), prefill("unpruned", "A_to_B", "pure-Pythia", 512)
    r2048, p2048 = prefill("unpruned", "A_to_B", "pure-RWKV", 2048), prefill("unpruned", "A_to_B", "pure-Pythia", 2048)
    check(
        "prefill ctx512",
        f"RWKV 単体が ${comma(round(r512))}$ ms，Pythia 単体が ${fmt(p512, 0)}$ ms であり，約 ${fmt(r512 / p512, 0)}$ 倍",
    )
    check(
        "prefill ctx2048",
        f"${comma(round(r2048))}$ ms 対 ${fmt(p2048, 0)}$ ms と約 ${fmt(r2048 / p2048, 0)}$ 倍",
    )
    check(
        "pruned prefill L=4",
        f"\\RtoPL{{4}} は ${comma(round(prefill('pruned', 'A_to_B', 'pruned-A_to_B@4', 512)))}$ ms，"
        f"$27$ ブロック実行する \\PtoRL{{4}} は ${comma(round(prefill('pruned', 'B_to_A', 'pruned-B_to_A@4', 512)))}$ ms",
    )
    pruned = [r for r in serving if r["benchmark"] == "pruned" and r["config"].startswith("pruned-")]
    weights = [float(r["weight_mib_total"]) / 1024 for r in pruned]
    parent_weight = lambda config: float(next(r["weight_mib_total"] for r in serving if r["benchmark"] == "pruned" and r["config"] == config)) / 1024
    check(
        "pruned resident weights",
        f"アダプタを含めて常駐する重みは ${fmt(min(weights), 2)}$--${fmt(max(weights), 2)}$ GiB であり，RWKV 単体の "
        f"${fmt(parent_weight('pure-RWKV'), 2)}$ GiB，Pythia 単体の ${fmt(parent_weight('pure-Pythia'), 2)}$ GiB",
    )
    invariant(
        "pruned chimeras reproduce the unpruned outputs",
        bool(pruned) and all(r["gate_relative_error"] != "" and float(r["gate_relative_error"]) == 0.0 for r in pruned),
    )
    for producer in ("A_to_B", "B_to_A"):
        for context in (512, 2048):
            runs = sorted(
                (int(r["rwkv_blocks"]), float(r["ms"])) for r in pruned
                if r["producer_path"] == producer and r["context_tokens"] == str(context)
            )
            invariant(
                f"pruned {producer} prefill grows with RWKV blocks at ctx{context}",
                len(runs) == 4 and all(a[1] < b[1] for a, b in zip(runs, runs[1:])),
                f"runs = {runs}",
            )

    # Learning-rate schedule. torch's ReduceLROnPlateau (mode="max", default threshold_mode "rel") updates its best value
    # only when an evaluation exceeds it by the relative threshold, and halves the rate once more than `patience`
    # evaluations in a row bring no update, so the paper must print patience + 1.
    repo = pathlib.Path(__file__).resolve().parents[1]
    trainer = (repo / "experiments" / "adapter7b.py").read_text(encoding="utf-8")
    invariant(
        "plateau scheduler settings",
        'ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=a.patience,' in trainer
        and "threshold=1e-3, min_lr=1e-5)" in trainer
        and 'sched.step(ev["A->B"])' in trainer
        and not re.search(r"threshold_mode|cooldown", trainer),
    )
    jobs = [p.read_text(encoding="utf-8") for p in sorted((repo / "experiments" / "slurm").glob("*.sbatch"))]
    jobs = [j for j in jobs if "adapter7b.py" in j]
    per_job = [re.findall(r"--patience (\d+)", j) for j in jobs]
    invariant(
        "every training job uses the plateau schedule with one patience",
        bool(jobs) and all("--sched plateau" in j for j in jobs) and all(len(p) == 1 for p in per_job)
        and len({p[0] for p in per_job if p}) == 1,
        f"{per_job}",
    )
    patience = min((int(p[0]) for p in per_job if p), default=-2)
    check(
        "learning-rate halving rule",
        f"基準値を相対で $10^{{-3}}$ より大きく上回った評価でのみ基準値を更新して，更新のない評価が ${patience + 1}$ 回続くたびに学習率を $0.5$ 倍した",
    )

    # Evaluation-set exposure: produced by tools/check_eval_leakage.py, which needs no
    # GPU (it recomputes from the per-item correctness dumps), so a reviewer can rerun it.
    leak_path = table_dir.parent / "raw" / "eval_leakage.json"
    if leak_path.exists():
        leak = json.loads(leak_path.read_text(encoding="utf-8"))
        counts = leak["matched_counts"]
        check(
            "leakage matched counts",
            f"ARC の ${counts['arc_easy']}$ 問と SciQ の ${counts['sciq']}$ 問",
        )
        check(
            "leakage max delta",
            f"最大 ${fmt(leak['max_abs_delta_points'], 2)}$ 点",
        )
        contrast = {
            (c["task"], c["config"]): c["accuracy_on_matched"] - c["accuracy_on_rest"]
            for c in leak["parent_contrast"]
        }
        check(
            "leakage RWKV matched-subset contrast",
            f"変化は ARC で ${signed(fmt(contrast[('arc_easy', 'rwkv')], 1))}$ 点，"
            f"SciQ で ${signed(fmt(contrast[('sciq', 'rwkv')], 1))}$ 点",
        )
        skipped = leak.get("unsearched_short_questions", {})
        if skipped:
            check(
                "leakage unsearched short questions",
                f"内訳は ARC が ${skipped['arc_easy']}$ 問，SciQ が ${skipped['sciq']}$ 問",
            )
        invariant(
            "leakage changes every accuracy by under 1 point",
            leak["max_abs_delta_points"] < 1.0,
            f"max |delta| = {leak['max_abs_delta_points']}",
        )

    # The generation table no longer prints the cross read-out R^2 in its header, so it is not pinned here;
    # the same read-outs are pinned by the headline checks near the top.

    adapter_pkgs = [p for p in sorted(release_dir.glob("*adapter*")) if (p / "model.safetensors").exists()]
    full_pkgs = [p for p in sorted(release_dir.glob("*-best")) if (p / "source").is_dir()]
    layer_windows = int(next(r["n_windows"] for r in layers if r["row_type"] == "layer"))
    if adapter_pkgs and full_pkgs:
        train_cfg = json.loads((adapter_pkgs[0] / "config.json").read_text(encoding="utf-8"))["training"]
        headline_windows = int(train_cfg["held_out_windows"])
        check("selected checkpoint", f"${comma(train_cfg['step'])}$ 反復目のチェックポイントの $f$ は ${fmt(f_var, 3)}$")
        check(
            "held-out set size (headline)",
            f"系列 ${comma(headline_windows)}$ 本，計 ${comma(headline_windows * 112)}$ トークンで求める",
        )
        check(
            "held-out set size (per-layer and shuffle)",
            f"系列 ${comma(layer_windows)}$ 本，計 ${comma(layer_windows * 112)}$ トークンを用いる",
        )
        adapter_params = safetensors_param_count(adapter_pkgs[0] / "model.safetensors")
        parent_bytes = sum(
            json.loads((full_pkgs[0] / side / "model.safetensors.index.json").read_text())["metadata"]["total_size"]
            for side in ("source", "target")
        )
        parent_params = parent_bytes / 2
        check("parents total params", f"親モデルの合計 ${fmt(parent_params / 1e9, 2)}$B パラメータ")
        check(
            "adapter params and share",
            f"アダプタは ${fmt(adapter_params / 1e6, 1)}$M パラメータであり，"
            f"比率は ${fmt(adapter_params / parent_params * 100, 1)}\\%$",
        )
    else:
        # release/ is gitignored: a fresh public checkout has no staged packages, so the two
        # parameter-count bindings cannot be evaluated until the exporters are run.
        print("NOTE: release packages not staged; skipped parents-total/adapter-share parameter checks.")

    print(f"PAPER_CLAIMS checked={passed + len(failures)} passed={passed} failed={len(failures)}")
    for line in failures:
        print("FAIL", line)
    if failures:
        return 1
    print("PAPER_CLAIMS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
