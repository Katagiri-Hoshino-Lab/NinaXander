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
    tex = re.sub(r"\s+", "", raw_tex)

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

    check("four-path headline", f"{fmt(aa, 3)}/{fmt(ab, 3)}/{fmt(bb, 3)}/{fmt(ba, 3)}")
    check("rho vs raw", f"$\\rho_{{\\mathrm{{ctr}}}}$ が ${fmt(rho, 3)}$ であるのに対し生の相関は ${fmt(raw_corr, 3)}$ を示し")
    check(
        "common-mode identity",
        f"$(1{{-}}{fmt(f_var, 3)})+{fmt(f_var, 3)}\\times{fmt(rho, 3)}={fmt(raw_corr, 3)}$",
    )
    check("deepest f", f"最深点の $f$ は ${fmt(f_var, 3)}$")
    check("constant share", f"潜在の持つエネルギーの ${fmt((1 - f_var) * 100, 0)}\\%$ は")
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
        f"$A\\!\\to\\!B$ は ${fmt(float(mean_all['A->B']), 3)}\\!\\to\\!{fmt(float(mean_all['A->B_shuf']), 3)}$，"
        f"$B\\!\\to\\!A$ は ${fmt(float(mean_all['B->A']), 3)}\\!\\to\\!{fmt(float(mean_all['B->A_shuf']), 3)}$ へ崩壊する",
    )
    local_ab = "/".join(fmt(float(per_layer[l]["A->B"]), 3) for l in (4, 8, 16, 24))
    local_ba = "/".join(fmt(float(per_layer[l]["B->A"]), 3) for l in (4, 8, 16, 24))
    check("local per-layer readouts", f"\\RtoP{{}} ${local_ab}$，\\PtoR{{}} ${local_ba}$")

    curve = {int(r["step"]): r for r in load(table_dir, "paper_training_curve.csv")}
    last_step = max(curve)
    best_step = max(curve, key=lambda s: float(curve[s]["eval_a_to_b_r2"]))
    check(
        "appendix run length",
        f"${comma(last_step)}$ 反復を走らせ，保留 $A\\!\\to\\!B$ が最良となる "
        f"${comma(best_step)}$ 反復目を報告する",
    )
    check("reported checkpoint", f"最深点，すなわち ${comma(best_step)}$ 反復目のチェックポイント")
    # The paper no longer prints the common-mode training-dynamics analysis, the
    # equal-KV early-exit baseline, or the pruned-runtime / serving-timing figures,
    # so the fragments that pinned them are gone. The underlying tables are still
    # produced and validated by tools/validate_results.py; re-add the checks here if
    # those paragraphs come back.

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
        f"対角平均は ${fmt(sum(diag) / len(diag), 2)}$，非対角平均は ${fmt(sum(off) / len(off), 2)}$",
    )
    check(
        "CKA argmax counts",
        f"$32$ 行中 ${n_diag}$ 行にすぎない。最類似の列は層 $0$ が ${n_zero}$ 行，"
        f"最終層が ${n_last}$ 行を占める",
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
        f"適合 $R^2$ は対角平均 ${fmt(sum(lin_diag) / len(lin_diag), 2)}$ 対 非対角平均 ${fmt(sum(lin_off) / len(lin_off), 2)}$",
    )
    check(
        "linear argmax same-layer count",
        f"同番号層が最良となるのは $32$ 行中 ${sum(1 for s, (t, _) in lin_best.items() if s == t)}$ 行",
    )
    all_lin = lin_diag + lin_off
    check(
        "figure caption fit range",
        f"適合 $R^2$ は ${fmt(min(all_lin), 2)}$--${fmt(max(all_lin), 2)}$ とほぼ全ペアで高い",
    )

    lin = {r["execution_path"]: r for r in load(table_dir, "paper_cross_family_linearity.csv") if r["scope"] == "mean"}
    aff_ab, aff_ba = float(lin["A_to_B"]["best_affine_cross_r2"]), float(lin["B_to_A"]["best_affine_cross_r2"])
    fit_ab, fit_ba = float(lin["A_to_B"]["linearised_fit_r2"]), float(lin["B_to_A"]["linearised_fit_r2"])
    ada_ab, ada_ba = float(lin["A_to_B"]["adapter_cross_r2"]), float(lin["B_to_A"]["adapter_cross_r2"])
    check("abstract affine", f"で ${fmt(aff_ab, 3)}/{fmt(aff_ba, 3)}$ に達する一方")
    check(
        "abstract nonaffine share",
        f"学習済みアダプタ出力の ${fmt((1 - fit_ab) * 100, 1)}\\%/{fmt((1 - fit_ba) * 100, 1)}\\%$ は",
    )
    check("affine imitation", f"\\RtoP{{}} で ${fmt(fit_ab * 100, 1)}\\%$，\\PtoR{{}} で ${fmt(fit_ba * 100, 1)}\\%$ である")
    check("direct affine reach", f"最良直接アフィン写像は \\RtoP{{}} ${fmt(aff_ab, 3)}$，\\PtoR{{}} ${fmt(aff_ba, 3)}$ に達する")
    check("intro linear reach", f"最良の\\emph{{線形}}写像だけでも $A\\!\\to\\!B\\ {fmt(aff_ab, 3)}$，$B\\!\\to\\!A\\ {fmt(aff_ba, 3)}$ に届く")
    check("adapter on same rows", f"対応するアダプタは同じ行で \\RtoP{{}} ${fmt(ada_ab, 3)}$，\\PtoR{{}} ${fmt(ada_ba, 3)}$")
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

    d_sciq = pair_row("sciq", "A_to_B@4", "rwkv")
    d_arc = pair_row("arc_easy", "A_to_B@4", "rwkv")
    check(
        "claim (d) AB@4 vs RWKV",
        f"RWKV を ${signed(fmt(float(d_sciq['accuracy_difference']) * 100, 1))}$ 点上回り，"
        f"$p{{=}}{fmt_p(d_sciq['mcnemar_p'])}$ となる一方，ARC-Easy では "
        f"${signed(fmt(float(d_arc['accuracy_difference']) * 100, 1))}$ 点，"
        f"$p{{=}}{fmt_p(d_arc['mcnemar_p'])}$ となる",
    )
    d2_arc = pair_row("arc_easy", "B_to_A@4", "rwkv")
    d2_sciq = pair_row("sciq", "B_to_A@4", "rwkv")
    check(
        "claim (d) BA@4 vs RWKV",
        f"ARC ${fmt(float(d2_arc['accuracy_a']) * 100, 1)}$ 対 ${fmt(float(d2_arc['accuracy_b']) * 100, 1)}$ で "
        f"$p{{=}}{fmt_p(d2_arc['mcnemar_p'])}$，SciQ ${fmt(float(d2_sciq['accuracy_a']) * 100, 1)}$ 対 "
        f"${fmt(float(d2_sciq['accuracy_b']) * 100, 1)}$ で $p{{=}}{fmt_p(d2_sciq['mcnemar_p'])}$",
    )

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
        "claim (f) gap ranges",
        f"SciQ ${fmt(min(gaps['sciq']), 0)}$--${fmt(max(gaps['sciq']), 0)}$ 点，"
        f"ARC ${fmt(min(gaps['arc_easy']), 0)}$--${fmt(max(gaps['arc_easy']), 0)}$ 点",
    )
    ba_ab_16_24 = {
        (task, l): pair_row(task, f"B_to_A@{l}", f"A_to_B@{l}") for task in ("arc_easy", "sciq") for l in (16, 24)
    }
    check(
        "BA vs AB deep boundaries",
        f"ARC で ${signed(fmt(float(ba_ab_16_24[('arc_easy', 16)]['accuracy_difference']) * 100, 1))}"
        f"/{signed(fmt(float(ba_ab_16_24[('arc_easy', 24)]['accuracy_difference']) * 100, 1))}$ 点，"
        f"SciQ で ${signed(fmt(float(ba_ab_16_24[('sciq', 16)]['accuracy_difference']) * 100, 1))}"
        f"/{signed(fmt(float(ba_ab_16_24[('sciq', 24)]['accuracy_difference']) * 100, 1))}$ 点上回る",
    )
    invariant(
        "BA vs AB deep p < 3e-15",
        all(float(r["mcnemar_p"]) < 3e-15 for r in ba_ab_16_24.values()),
    )
    l4_arc = pair_row("arc_easy", "B_to_A@4", "A_to_B@4")
    l4_sciq = pair_row("sciq", "B_to_A@4", "A_to_B@4")
    check(
        "BA vs AB shallow",
        f"ARC で ${signed(fmt(float(l4_arc['accuracy_difference']) * 100, 1))}$ 点，"
        f"$p{{=}}{fmt_p(l4_arc['mcnemar_p'])}$ だが，SciQ では "
        f"${signed(fmt(float(l4_sciq['accuracy_difference']) * 100, 1))}$ 点，"
        f"$p{{=}}{fmt_p(l4_sciq['mcnemar_p'])}$",
    )

    qa = load(table_dir, "paper_cross_family_qa_accuracy.csv")
    qa_acc = {(r["config"], r["task"]): float(r["accuracy_norm"]) * 100 for r in qa}
    ab_affine_gaps = [
        qa_acc[(f"A_to_B@{l}", task)] - qa_acc[(f"A_to_B_affine@{l}", task)]
        for task in ("arc_easy", "sciq")
        for l in (4, 8, 16, 24)
    ]
    check(
        "AB adapter vs affine range",
        f"アダプタが全セルで ${fmt(min(ab_affine_gaps), 0)}$--${fmt(max(ab_affine_gaps), 0)}$ 点高い",
    )
    check(
        "best cross QA",
        f"SciQ では \\RtoP{{}} ${fmt(max(acc[('A_to_B', 'sciq', l)] for l in (4, 8, 16, 24)), 1)}\\%$，"
        f"\\PtoR{{}} ${fmt(max(acc[('B_to_A', 'sciq', l)] for l in (4, 8, 16, 24)), 1)}\\%$ に達する",
    )
    seq = lambda path, task: "\\to".join(fmt(acc[(path, task, l)], 1) for l in (4, 8, 16, 24))
    check("AB depth trend", f"${seq('A_to_B', 'sciq')}$，ARC-Easy ${seq('A_to_B', 'arc_easy')}$ と単調に下がる")
    check("BA depth trend", f"SciQ ${seq('B_to_A', 'sciq')}$，ARC-Easy ${seq('B_to_A', 'arc_easy')}$ と非単調")
    mean_ab4 = (acc[("A_to_B", "arc_easy", 4)] + acc[("A_to_B", "sciq", 4)]) / 2
    mean_ba4 = (acc[("B_to_A", "arc_easy", 4)] + acc[("B_to_A", "sciq", 4)]) / 2
    check("two-task means", f"${fmt(mean_ab4, 2)}\\%\\to{fmt(mean_ba4, 2)}\\%$ とほぼ保つ")
    sum_acc = {
        (r["config"], r["task"]): float(r["accuracy_sum"]) * 100 for r in qa if r["accuracy_sum"]
    }
    check(
        "sum-scoring robustness note",
        f"\\RtoPL{{4}} は SciQ ${fmt(sum_acc[('A_to_B@4', 'sciq')] - sum_acc[('rwkv', 'sciq')], 1)}$ 点であり，"
        f"\\PtoRL{{4}} は ARC ${fmt(sum_acc[('B_to_A@4', 'arc_easy')] - sum_acc[('rwkv', 'arc_easy')], 1)}$ 点，"
        f"SciQ ${fmt(sum_acc[('B_to_A@4', 'sciq')] - sum_acc[('rwkv', 'sciq')], 1)}$ 点",
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
    check(
        "abstract KV",
        f"KV を ${fmt(float(ba4['kv_reduction_fraction']) * 100, 1)}\\%$ 削減しつつ，"
        f"ARC-Easy ${fmt(float(ba4['arc_easy_accuracy_norm']) * 100, 1)}\\%$，"
        f"SciQ ${fmt(float(ba4['sciq_accuracy_norm']) * 100, 1)}\\%$ を保つ",
    )
    check(
        "RWKV state size",
        f"層あたり ${fmt(float(ba4['rwkv_state_kib_per_layer']), 0)}$ KiB である。"
        "この値は，参照実装が持つ $5$ 状態ベクトル",
    )
    check(
        "table5 pythia row",
        f"& ${fmt(float(parent_b['kv_kib_per_token']), 0)}$ KiB & ${fmt(float(parent_b['kv_gib_at_context']), 2)}$ GiB & $0\\%$ & "
        f"{fmt(float(parent_b['arc_easy_accuracy_norm']) * 100, 1)} / {fmt(float(parent_b['sciq_accuracy_norm']) * 100, 1)}",
    )
    check(
        "table5 rwkv row",
        f"{fmt(float(parent_a['arc_easy_accuracy_norm']) * 100, 1)} / {fmt(float(parent_a['sciq_accuracy_norm']) * 100, 1)}",
    )
    check(
        "table5 BA@4 row",
        f"\\textbf{{$\\mathbf{{{fmt(float(ba4['kv_reduction_fraction']) * 100, 1)}}}\\%$}} & "
        f"\\textbf{{{fmt(float(ba4['arc_easy_accuracy_norm']) * 100, 1)} / {fmt(float(ba4['sciq_accuracy_norm']) * 100, 1)}}}",
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
        "domain shift ctx512 AB",
        f"Alpaca で perplexity ${fmt(ppl[('A_to_B@4', 'alpaca', 512)], 2)}$，WikiText-103 で ${fmt(ppl[('A_to_B@4', 'wikitext', 512)], 2)}$",
    )
    check(
        "domain shift ctx512 BA",
        f"${fmt(ppl[('B_to_A@4', 'alpaca', 512)], 2)}/{fmt(ppl[('B_to_A@4', 'wikitext', 512)], 2)}$ と両領域で改善",
    )
    check(
        "domain shift parents",
        f"親の ${fmt(ppl[('pure-RWKV', 'wikitext', 512)], 2)}/{fmt(ppl[('pure-Pythia', 'wikitext', 512)], 2)}$ から",
    )
    check(
        "domain shift ctx2048 BA",
        f"Alpaca ${fmt(ppl[('B_to_A@4', 'alpaca', 2048)], 2)}$，WikiText ${fmt(ppl[('B_to_A@4', 'wikitext', 2048)], 2)}$",
    )
    ratio = ppl[("B_to_A@4", "wikitext", 2048)] / ppl[("pure-RWKV", "wikitext", 2048)]
    check(
        "domain shift ratio",
        f"WikiText ${fmt(ppl[('B_to_A@4', 'wikitext', 2048)], 2)}$ は良い方の親 RWKV ${fmt(ppl[('pure-RWKV', 'wikitext', 2048)], 2)}$ の ${fmt(ratio, 1)}$ 倍",
    )
    for config, x in (("pure-RWKV", 11), ("pure-Pythia", 33), ("A_to_B@4", 55), ("B_to_A@4", 77)):
        check(
            f"figure6 {config}",
            f"{{{fmt(ppl[(config, 'alpaca', 512)], 2)}}}{{{fmt(ppl[(config, 'wikitext', 512)], 2)}}}",
        )

    # Evaluation-set exposure: produced by tools/check_eval_leakage.py, which needs no
    # GPU (it recomputes from the per-item correctness dumps), so a reviewer can rerun it.
    leak_path = table_dir.parent / "raw" / "eval_leakage.json"
    if leak_path.exists():
        leak = json.loads(leak_path.read_text(encoding="utf-8"))
        counts = leak["matched_counts"]
        check(
            "leakage matched counts",
            f"ARC-Easy ${counts['arc_easy']}$ 問と SciQ ${counts['sciq']}$ 問",
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
            "leakage unexposed-parent contrast",
            f"変化は ARC で ${signed(fmt(contrast[('arc_easy', 'rwkv')], 1))}$ 点，"
            f"SciQ で ${signed(fmt(contrast[('sciq', 'rwkv')], 1))}$ 点",
        )
        skipped = leak.get("unsearched_short_questions", {})
        if skipped:
            check(
                "leakage unsearched short questions",
                f"その内訳は ARC-Easy ${skipped['arc_easy']}$ 問と SciQ ${skipped['sciq']}$ 問",
            )
        invariant(
            "leakage does not overturn any comparison",
            leak["max_abs_delta_points"] < 1.0,
            f"max |delta| = {leak['max_abs_delta_points']}",
        )

    gen = load(table_dir, "paper_cross_family_generation_samples.csv")
    gen_r2 = {r["execution_path"]: r["adapter_cross_r2"] for r in gen if r["adapter_cross_r2"]}
    for path in ("A_to_B", "B_to_A"):
        check(f"generation R2 {path}", f"横断読み出し $R^2{{=}}{fmt(float(gen_r2[path]), 3)}$")

    adapter_pkgs = [p for p in sorted(release_dir.glob("*adapter*")) if (p / "model.safetensors").exists()]
    full_pkgs = [p for p in sorted(release_dir.glob("*-best")) if (p / "source").is_dir()]
    layer_windows = int(next(r["n_windows"] for r in layers if r["row_type"] == "layer"))
    if adapter_pkgs and full_pkgs:
        train_cfg = json.loads((adapter_pkgs[0] / "config.json").read_text(encoding="utf-8"))["training"]
        headline_windows = int(train_cfg["held_out_windows"])
        check(
            "held-out set sizes",
            f"学習時評価には ${comma(headline_windows)}$ 窓，すなわち "
            f"${comma(headline_windows * 112)}$ トークンを用いる。"
            f"層別分析とシャッフル対照には ${comma(layer_windows)}$ 窓，すなわち "
            f"${comma(layer_windows * 112)}$ トークンを用いる",
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
            f"アダプタは ${fmt(adapter_params / 1e6, 1)}$M であり，"
            f"その比率は ${fmt(adapter_params / parent_params * 100, 1)}\\%$",
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
