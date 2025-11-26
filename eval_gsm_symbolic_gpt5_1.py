import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
import time
from typing import Dict, List, Tuple
import os
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from openai import OpenAI
from tqdm import tqdm  # optional but nice

# Hardcoded API key placeholder. Replace the string below with your key
# Only do this for quick local experiments; do NOT commit this file.
os.environ["OPENAI_API_KEY"] = "sk-YOUR-API-KEY"

client = OpenAI()

# Decoding / API parameters tuned for deterministic reasoning
# (low temperature, full top_p). Adjust as needed.
API_PARAMS = {
    "model": "gpt-5.1",
    "max_output_tokens": 2048,
}


def load_jsonl(path: Path):
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def group_by_instance(examples):
    by_inst = defaultdict(list)
    for ex in examples:
        by_inst[ex["instance"]].append(ex)
    for inst in by_inst:
        by_inst[inst].sort(key=lambda e: e["id"])
    return by_inst


_num_pattern = re.compile(r"-?\d+\.?\d*")


def extract_number(text: str):
    """
    Return the last numeric value (as float) found in `text`,
    or None if nothing is found.
    """
    cleaned = text.replace(",", "")
    matches = _num_pattern.findall(cleaned)
    if not matches:
        return None
    return float(matches[-1])


def ground_truth_answer(example):
    """
    GSM8K-style answer: last non-empty line is '#### <ans>'.
    """
    answer = example["answer"]
    non_empty = [ln for ln in answer.splitlines() if ln.strip()]
    if not non_empty:
        return None
    last = non_empty[-1]
    return extract_number(last)


def build_prompt(target_question: str, shots):
    """
    shots: list of dicts with keys: question, solution, final_answer
    """
    parts = []
    # Preamble / system-style instruction
    parts.append("As an expert problem solver, solve step by step the following mathematical questions.\n")

    # Few-shot Q/A
    for shot in shots:
        q = shot["question"]
        sol = shot["solution"]
        final = shot["final_answer"]
        parts.append(f"Q: {q}")
        parts.append(f"A: Let's think step by step. {sol}. The final answer is {final}.")
        parts.append("")  # blank line

    # Target
    parts.append(f"Q: {target_question}")
    parts.append("A: Let's think step by step.")

    return "\n".join(parts)


def call_gpt_5_1(prompt: str, reasoning_effort: str = "high") -> str:
    """
    Call GPT-5.1 with reasoning. For GSM-Symbolic, use high effort.
    """
    # Use the configured API params and inject the reasoning effort
    params = dict(API_PARAMS)
    params["reasoning"] = {"effort": reasoning_effort}
    params["input"] = prompt

    resp = client.responses.create(**params)
    return resp.output_text  # simple text accessor in Responses API


def evaluate_instance(examples, shots, reasoning_effort: str = "high", workers: int = 1) -> Tuple[float, list, int]:
    """
    Evaluate a single 'instance' dataset, i.e., ~100 examples sharing the same instance id.
    Returns (accuracy, per_example_results).
    """
    correct = 0
    total = 0
    per_example = []

    def _process(ex):
        gt = ground_truth_answer(ex)
        prompt = build_prompt(ex["question"], shots)
        model_out = call_gpt_5_1(prompt, reasoning_effort=reasoning_effort)
        pred = extract_number(model_out)
        is_correct = (gt is not None) and (pred is not None) and math.isclose(pred, gt)
        return {
            "id": ex["id"],
            "instance": ex["instance"],
            "question": ex["question"],
            "gt": gt,
            "pred": pred,
            "model_out": model_out,
            "correct": is_correct,
        }

    if workers <= 1:
        for ex in tqdm(examples, desc="questions", leave=False):
            res = _process(ex)
            per_example.append(res)
            total += 1
            if res["correct"]:
                correct += 1
    else:
        # Parallel execution: submit all work and collect as they finish
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as exc:
            futures = {exc.submit(_process, ex): ex for ex in examples}
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="questions", leave=False):
                try:
                    res = fut.result()
                except Exception as e:
                    # On exception, record a failed result
                    src = futures[fut]
                    res = {
                        "id": src["id"],
                        "instance": src["instance"],
                        "question": src["question"],
                        "gt": ground_truth_answer(src),
                        "pred": None,
                        "model_out": None,
                        "correct": False,
                        "error": str(e),
                    }
                per_example.append(res)
                total += 1
                if res.get("correct"):
                    correct += 1

    acc = correct / total if total else 0.0
    return acc, per_example, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        required=False,
        help="Primary dataset JSONL (default: generated_data/GSM_symbolic.jsonl)",
    )
    parser.add_argument(
        "--data2",
        type=Path,
        required=False,
        help="Secondary dataset JSONL to compare (default: generated_data/GSM_p2.jsonl)",
    )
    parser.add_argument(
        "--shots",
        type=Path,
        required=False,
        help="Path to 8-shot GSM8K JSON file (omit for 0-shot CoT)",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=15,
        help="How many instance ids to evaluate per dataset (default 15).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker threads for API calls per instance (default 8). Start lower if you hit rate limits.",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="high",
        choices=["none", "low", "medium", "high"],
        help="GPT-5.1 reasoning effort.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results_checkpoint.json"),
        help="Where to save incremental checkpoints.",
    )
    parser.add_argument(
        "--plots",
        type=Path,
        default=Path("plots"),
        help="Directory where plots will be saved.",
    )

    args = parser.parse_args()

    # Determine dataset paths (defaults to built-in generated_data files)
    default_sym = Path("ml-gsm-symbolic/generated_data/GSM_symbolic.jsonl")
    default_p2 = Path("ml-gsm-symbolic/generated_data/GSM_p2.jsonl")

    data_paths: Dict[str, Path] = {}
    data_paths["symbolic"] = args.data if args.data else default_sym
    data_paths["p2"] = args.data2 if args.data2 else default_p2

    # load shots if provided
    if args.shots:
        print(f"Loading shots from {args.shots} ...")
        with args.shots.open("r", encoding="utf-8") as f:
            shots = json.load(f)
    else:
        shots = []

    # load datasets
    datasets = {}
    for name, p in data_paths.items():
        print(f"Loading data for {name} from {p} ...")
        datasets[name] = load_jsonl(p)

    # load existing checkpoint if present
    checkpoint = {"summary": {}, "examples": []}
    if args.checkpoint.exists():
        try:
            checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
            print(f"Loaded checkpoint with {len(checkpoint.get('examples', []))} examples")
        except Exception:
            print("Warning: couldn't read existing checkpoint, starting fresh.")

    # initialize acc/count containers
    all_accs: Dict[str, Dict[int, float]] = {"symbolic": {}, "p2": {}}
    all_counts: Dict[str, Dict[int, int]] = {"symbolic": {}, "p2": {}}

    # If checkpoint already contains per-instance summary, load it so plots
    # can be generated when resuming without re-running instances.
    summary = checkpoint.get("summary", {}) if isinstance(checkpoint, dict) else {}
    for ds_name, ds_summary in summary.items():
        try:
            for inst_str, vals in ds_summary.items():
                try:
                    inst = int(inst_str)
                except Exception:
                    # keys might already be ints
                    inst = inst_str
                acc_val = vals.get("accuracy") if isinstance(vals, dict) else None
                n_val = vals.get("n") if isinstance(vals, dict) else None
                if acc_val is not None:
                    all_accs.setdefault(ds_name, {})[inst] = acc_val
                if n_val is not None:
                    all_counts.setdefault(ds_name, {})[inst] = n_val
        except Exception:
            # ignore malformed checkpoint entries
            continue

    for ds_name, data in datasets.items():
        by_inst = group_by_instance(data)
        max_inst = min(args.instances, len(by_inst))

        # determine which instances already present in checkpoint
        completed_instances = set()
        for ex in checkpoint.get("examples", []):
            if ex.get("dataset") == ds_name:
                completed_instances.add(ex.get("instance"))

        for inst in range(max_inst):
            if inst in completed_instances:
                # compute stored accuracy if available
                continue

            inst_examples = by_inst[inst]
            print(f"Evaluating dataset {ds_name} instance {inst} with {len(inst_examples)} examples ...")
            acc, per_ex, n = evaluate_instance(inst_examples, shots, reasoning_effort=args.reasoning_effort, workers=args.workers)

            for e in per_ex:
                e["dataset"] = ds_name
            checkpoint.setdefault("examples", []).extend(per_ex)

            all_accs[ds_name][inst] = acc
            all_counts[ds_name][inst] = n

            checkpoint.setdefault("summary", {})
            checkpoint["summary"].setdefault(ds_name, {})
            checkpoint["summary"][ds_name][str(inst)] = {"accuracy": acc, "n": n}

            # write checkpoint to disk immediately
            args.checkpoint.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
            print(f"Saved checkpoint after dataset={ds_name} instance={inst}")

    # Prepare plots directory
    args.plots.mkdir(parents=True, exist_ok=True)

    def instance_confidence_interval(p: float, n: int) -> float:
        if n <= 1:
            return 0.0
        return math.sqrt(p * (1 - p) / n)

    # Build plotting data from checkpoint/accs
    plot_data = {}
    for ds in ["symbolic", "p2"]:
        insts = sorted(all_accs[ds].keys())
        acc_list = [all_accs[ds][i] for i in insts]
        ns = [all_counts[ds].get(i, 0) for i in insts]
        errs = [instance_confidence_interval(a, n) for a, n in zip(acc_list, ns)]
        plot_data[ds] = {"instances": insts, "acc": acc_list, "err": errs}

    # Build filtered per-instance accuracies (discard null preds) from checkpoint examples
    # We'll compute per-instance accuracy using only examples where 'pred' is not null.
    filtered_accs: Dict[str, Dict[int, float]] = {"symbolic": {}, "p2": {}}
    filtered_counts: Dict[str, Dict[int, int]] = {"symbolic": {}, "p2": {}}
    for ex in checkpoint.get("examples", []):
        ds = ex.get("dataset")
        inst = ex.get("instance")
        if ds is None or inst is None:
            continue
        pred = ex.get("pred")
        correct = bool(ex.get("correct"))
        if pred is None:
            # skip null predictions
            continue
        filtered_counts.setdefault(ds, {}).setdefault(inst, 0)
        filtered_accs.setdefault(ds, {}).setdefault(inst, 0.0)
        filtered_counts[ds][inst] += 1
        if correct:
            filtered_accs[ds][inst] += 1

    # convert counts to accuracy fraction
    for ds in filtered_accs:
        for inst in list(filtered_accs[ds].keys()):
            cnt = filtered_counts[ds].get(inst, 0)
            if cnt > 0:
                filtered_accs[ds][inst] = filtered_accs[ds][inst] / cnt
            else:
                # safety
                filtered_accs[ds][inst] = None

    # prepare filtered plot_data (same structure as plot_data) for plotting in subfolder
    plot_data_filtered = {}
    for ds in ["symbolic", "p2"]:
        insts = sorted(filtered_accs[ds].keys())
        acc_list = [filtered_accs[ds][i] for i in insts]
        ns = [filtered_counts[ds].get(i, 0) for i in insts]
        errs = [instance_confidence_interval(a, n) if a is not None else 0.0 for a, n in zip(acc_list, ns)]
        plot_data_filtered[ds] = {"instances": insts, "acc": acc_list, "err": errs}

    # Create filtered plots directory
    filtered_dir = args.plots / "filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    # Re-run the same plotting suite but using plot_data_filtered and save into filtered_dir
    # Line plot with error bars comparing datasets (filtered)
    plt.figure(figsize=(10, 5))
    for ds, d in plot_data_filtered.items():
        plt.errorbar(d["instances"], d["acc"], yerr=d["err"], label=ds)
    plt.xlabel("Instance ID")
    plt.ylabel("Accuracy")
    plt.title("Per-instance accuracy (filtered, excluding null preds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filtered_dir / "accuracy_by_instance_filtered.png")
    plt.close()

    # Boxplot (filtered)
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=[plot_data_filtered[ds]["acc"] for ds in plot_data_filtered], orient="v")
    plt.xticks(range(len(plot_data_filtered)), list(plot_data_filtered.keys()))
    plt.ylabel("Accuracy")
    plt.title("Distribution of instance accuracies (filtered)")
    plt.tight_layout()
    plt.savefig(filtered_dir / "accuracy_boxplot_filtered.png")
    plt.close()

    # Histograms + KDE (filtered): re-use adaptive bins and scaled KDE helpers above
    # symbolic filtered
    sym_acc_f = [a * 100 for a in plot_data_filtered.get("symbolic", {}).get("acc", []) if a is not None]
    p2_acc_f = [a * 100 for a in plot_data_filtered.get("p2", {}).get("acc", []) if a is not None]

    def _adaptive_bins(vals, nbins=10, pad_left=1, pad_right=1, minpad=1e-6):
        # Compute bin edges such that the central range (nbins equal-width bins)
        # covers the data [vmin, vmax], and there are `pad_left` extra empty
        # bins on the left and `pad_right` on the right (same bin width).
        # Returns (bins, xmin, xmax) where bins has length nbins+pad_left+pad_right+1.
        total_bins = nbins + pad_left + pad_right
        if not vals:
            # default to full 0-100 range when no data
            return np.linspace(0.0, 100.0, total_bins + 1), 0.0, 100.0
        v = np.array(vals)
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        if math.isclose(vmin, vmax):
            # Expand small epsilon around the point so bins are visible
            vmin = vmin - minpad
            vmax = vmax + minpad

        # central bin width based on nbins covering [vmin, vmax]
        bin_width = (vmax - vmin) / float(max(1, nbins))
        # extend range by padding bins on both sides
        xmin = vmin - pad_left * bin_width
        xmax = vmax + pad_right * bin_width
        # add a tiny epsilon so that no data point equals the outer bin edge
        # (this ensures the outermost bins remain empty even if a value == vmax)
        eps = max(minpad, bin_width * 1e-6)
        xmin -= eps
        xmax += eps
        return np.linspace(xmin, xmax, total_bins + 1), xmin, xmax

    # KDE helper (Gaussian kernel) that returns a scaled KDE matching histogram counts
    def _scaled_kde(vals, bins):
        # vals: array-like of sample values
        # bins: array of bin edges
        if not len(vals):
            return None, None
        x = np.linspace(bins[0], bins[-1], 512)
        arr = np.array(vals)
        n = arr.size
        # Silverman's rule of thumb bandwidth
        std = arr.std(ddof=1) if n > 1 else 0.0
        if std <= 0 or math.isclose(std, 0.0):
            # fallback tiny bandwidth
            bw = 1.0
        else:
            bw = 1.06 * std * n ** (-1 / 5)

        # avoid zero bandwidth
        bw = max(bw, 1e-3)

        # gaussian kernel density estimate
        coef = 1.0 / (n * bw * math.sqrt(2 * math.pi))
        xx = x[:, None]
        diffs = (xx - arr[None, :]) / bw
        kern = np.exp(-0.5 * diffs ** 2)
        dens = coef * kern.sum(axis=1)

        # scale density so that area under KDE matches histogram counts
        # area_hist = sum(counts) * bin_width = n (since counts sum to n)
        # so to make heights comparable to histogram counts we multiply by n * bin_width
        bin_width = bins[1] - bins[0]
        scaled = dens * n * bin_width
        return x, scaled

    if sym_acc_f:
        bins, xmin, xmax = _adaptive_bins(sym_acc_f, nbins=10)
        plt.figure(figsize=(8, 5))
        plt.hist(sym_acc_f, bins=bins, color='tab:blue', alpha=0.4, edgecolor='black')
        xk, yk = _scaled_kde(sym_acc_f, bins)
        if xk is not None:
            plt.plot(xk, yk, color='tab:blue', lw=2)
        plt.xlabel("Accuracy (%)")
        plt.ylabel("Frequency (instances)")
        plt.title("Histogram of per-instance accuracy (symbolic) - filtered")
        plt.xlim(xmin, xmax)
        plt.xticks(np.linspace(xmin, xmax, len(bins)))
        plt.tight_layout()
        plt.savefig(filtered_dir / "hist_kde_symbolic_filtered.png")
        plt.close()

    if p2_acc_f:
        bins, xmin, xmax = _adaptive_bins(p2_acc_f, nbins=10)
        plt.figure(figsize=(8, 5))
        plt.hist(p2_acc_f, bins=bins, color='pink', alpha=0.4, edgecolor='black')
        xk, yk = _scaled_kde(p2_acc_f, bins)
        if xk is not None:
            plt.plot(xk, yk, color='pink', lw=2)
        plt.xlabel("Accuracy (%)")
        plt.ylabel("Frequency (instances)")
        plt.title("Histogram of per-instance accuracy (p2) - filtered")
        plt.xlim(xmin, xmax)
        plt.xticks(np.linspace(xmin, xmax, len(bins)))
        plt.tight_layout()
        plt.savefig(filtered_dir / "hist_kde_p2_filtered.png")
        plt.close()

    if sym_acc_f or p2_acc_f:
        combined_f = (sym_acc_f if sym_acc_f else []) + (p2_acc_f if p2_acc_f else [])
        bins, xmin, xmax = _adaptive_bins(combined_f, nbins=10)
        plt.figure(figsize=(8, 5))
        if sym_acc_f:
            plt.hist(sym_acc_f, bins=bins, color='tab:blue', alpha=0.4, edgecolor='black', label='symbolic')
            xk_s, yk_s = _scaled_kde(sym_acc_f, bins)
            if xk_s is not None:
                plt.plot(xk_s, yk_s, color='tab:blue', lw=2)
        if p2_acc_f:
            plt.hist(p2_acc_f, bins=bins, color='pink', alpha=0.4, edgecolor='black', label='p2')
            xk_p, yk_p = _scaled_kde(p2_acc_f, bins)
            if xk_p is not None:
                plt.plot(xk_p, yk_p, color='pink', lw=2)
        plt.xlabel("Accuracy (%)")
        plt.ylabel("Frequency (instances)")
        plt.title("Overlaid histogram of per-instance accuracy (symbolic vs p2) - filtered")
        plt.xlim(xmin, xmax)
        plt.xticks(np.linspace(xmin, xmax, len(bins)))
        plt.legend()
        plt.tight_layout()
        plt.savefig(filtered_dir / "hist_kde_both_filtered.png")
        plt.close()

    # Null-value diagnostics: compute null/pred counts per dataset and plot
    total_examples_by_ds = {"symbolic": 0, "p2": 0}
    null_examples_by_ds = {"symbolic": 0, "p2": 0}
    for ex in checkpoint.get("examples", []):
        ds = ex.get("dataset")
        if ds not in total_examples_by_ds:
            continue
        total_examples_by_ds[ds] += 1
        if ex.get("pred") is None:
            null_examples_by_ds[ds] += 1

    # Bar plot: null counts per dataset
    labels = ["symbolic", "p2"]
    null_counts = [null_examples_by_ds.get(l, 0) for l in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, null_counts, color=["tab:blue", "pink"], alpha=0.7)
    plt.xlabel("Dataset")
    plt.ylabel("Number of null predictions")
    plt.title("Null predictions per dataset (filtered diagnostics)")
    plt.tight_layout()
    plt.savefig(filtered_dir / "nulls_by_dataset.png")
    plt.close()

    # Bar plot: null fraction per dataset
    null_frac = []
    for l in labels:
        tot = total_examples_by_ds.get(l, 0)
        nnull = null_examples_by_ds.get(l, 0)
        frac = (nnull / tot) if tot > 0 else 0.0
        null_frac.append(frac * 100.0)  # percent

    plt.figure(figsize=(6, 4))
    plt.bar(labels, null_frac, color=["tab:blue", "pink"], alpha=0.7)
    plt.xlabel("Dataset")
    plt.ylabel("Null predictions (%)")
    plt.title("Null prediction fraction per dataset (percent)")
    plt.tight_layout()
    plt.savefig(filtered_dir / "null_fraction_by_dataset.png")
    plt.close()

    # Line plot with error bars comparing datasets
    plt.figure(figsize=(10, 5))
    for ds, d in plot_data.items():
        plt.errorbar(d["instances"], d["acc"], yerr=d["err"], label=ds)
    plt.xlabel("Instance ID")
    plt.ylabel("Accuracy")
    plt.title("Per-instance accuracy with approximate CI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plots / "accuracy_by_instance.png")
    plt.close()

    # Boxplot of accuracies across instances
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=[plot_data[ds]["acc"] for ds in plot_data], orient="v")
    plt.xticks(range(len(plot_data)), list(plot_data.keys()))
    plt.ylabel("Accuracy")
    plt.title("Distribution of instance accuracies")
    plt.tight_layout()
    plt.savefig(args.plots / "accuracy_boxplot.png")
    plt.close()

    # Histogram + KDE for the first dataset (figure 2 style)
    # Convert accuracies to percentages for plotting
    sym_acc = [a * 100 for a in plot_data.get("symbolic", {}).get("acc", [])]
    p2_acc = [a * 100 for a in plot_data.get("p2", {}).get("acc", [])]

    # symbolic-only histogram
    if sym_acc:
        bins, xmin, xmax = _adaptive_bins(sym_acc, nbins=10)
        plt.figure(figsize=(8, 5))
        counts, _, _ = plt.hist(sym_acc, bins=bins, color='tab:blue', alpha=0.4, edgecolor='black')
        # scaled KDE
        xk, yk = _scaled_kde(sym_acc, bins)
        if xk is not None:
            plt.plot(xk, yk, color='tab:blue', lw=2)
        plt.xlabel("Accuracy (%)")
        plt.ylabel("Frequency (instances)")
        plt.title("Histogram of per-instance accuracy (symbolic)")
        plt.xlim(xmin, xmax)
        plt.xticks(np.linspace(xmin, xmax, len(bins)))
        plt.tight_layout()
        plt.savefig(args.plots / "hist_kde_symbolic.png")
        plt.close()

    # p2-only histogram
    if p2_acc:
        bins, xmin, xmax = _adaptive_bins(p2_acc, nbins=10)
        plt.figure(figsize=(8, 5))
        counts, _, _ = plt.hist(p2_acc, bins=bins, color='pink', alpha=0.4, edgecolor='black')
        xk, yk = _scaled_kde(p2_acc, bins)
        if xk is not None:
            plt.plot(xk, yk, color='pink', lw=2)
        plt.xlabel("Accuracy (%)")
        plt.ylabel("Frequency (instances)")
        plt.title("Histogram of per-instance accuracy (p2)")
        plt.xlim(xmin, xmax)
        plt.xticks(np.linspace(xmin, xmax, len(bins)))
        plt.tight_layout()
        plt.savefig(args.plots / "hist_kde_p2.png")
        plt.close()

    # Overlaid histograms + KDE for both datasets (figure 6 style)
    if sym_acc or p2_acc:
        # Use combined extent so both datasets share the same bin edges
        combined = (sym_acc if sym_acc else []) + (p2_acc if p2_acc else [])
        bins, xmin, xmax = _adaptive_bins(combined, nbins=10)
        plt.figure(figsize=(8, 5))
        if sym_acc:
            counts_s, _, _ = plt.hist(sym_acc, bins=bins, color='tab:blue', alpha=0.4, edgecolor='black', label='symbolic')
            xk_s, yk_s = _scaled_kde(sym_acc, bins)
            if xk_s is not None:
                plt.plot(xk_s, yk_s, color='tab:blue', lw=2)
        if p2_acc:
            counts_p, _, _ = plt.hist(p2_acc, bins=bins, color='pink', alpha=0.4, edgecolor='black', label='p2')
            xk_p, yk_p = _scaled_kde(p2_acc, bins)
            if xk_p is not None:
                plt.plot(xk_p, yk_p, color='pink', lw=2)

        plt.xlabel("Accuracy (%)")
        plt.ylabel("Frequency (instances)")
        plt.title("Overlaid histogram of per-instance accuracy (symbolic vs p2)")
        plt.xlim(xmin, xmax)
        plt.xticks(np.linspace(xmin, xmax, len(bins)))
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plots / "hist_kde_both.png")
        plt.close()

    # final checkpoint write
    args.checkpoint.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    print(f"Saved final checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()

