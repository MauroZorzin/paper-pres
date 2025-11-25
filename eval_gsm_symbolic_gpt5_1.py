import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm  # optional but nice

client = OpenAI()


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
    resp = client.responses.create(
        model="gpt-5.1",
        reasoning={"effort": reasoning_effort},
        input=prompt,
    )
    return resp.output_text  # simple text accessor in Responses API


def evaluate_instance(examples, shots, reasoning_effort: str = "high"):
    """
    Evaluate a single 'instance' dataset, i.e., ~100 examples sharing the same instance id.
    Returns (accuracy, per_example_results).
    """
    correct = 0
    total = 0
    per_example = []

    for ex in tqdm(examples, desc="questions", leave=False):
        gt = ground_truth_answer(ex)
        prompt = build_prompt(ex["question"], shots)
        model_out = call_gpt_5_1(prompt, reasoning_effort=reasoning_effort)
        pred = extract_number(model_out)

        is_correct = (gt is not None) and (pred is not None) and math.isclose(pred, gt)
        total += 1
        if is_correct:
            correct += 1

        per_example.append(
            {
                "id": ex["id"],
                "instance": ex["instance"],
                "question": ex["question"],
                "gt": gt,
                "pred": pred,
                "correct": is_correct,
            }
        )

    acc = correct / total if total else 0.0
    return acc, per_example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to GSM-Symbolic JSONL (e.g. generated_data/GSM_symbolic.jsonl)",
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
        default=50,
        help="How many instance ids to evaluate (starting from 0). Use e.g. 5 for a quick run.",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="high",
        choices=["none", "low", "medium", "high"],
        help="GPT-5.1 reasoning effort.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results.json"),
        help="Where to save detailed results.",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data} ...")
    data = load_jsonl(args.data)
    by_inst = group_by_instance(data)

    if args.shots:
        print(f"Loading shots from {args.shots} ...")
        with args.shots.open("r", encoding="utf-8") as f:
            shots = json.load(f)
    else:
        shots = []

    accs = {}
    detailed_results = []

    max_inst = min(args.instances, len(by_inst))

    for inst in range(max_inst):
        inst_examples = by_inst[inst]
        print(f"Evaluating instance {inst} with {len(inst_examples)} examples ...")
        acc, per_ex = evaluate_instance(inst_examples, shots, reasoning_effort=args.reasoning_effort)
        accs[inst] = acc
        detailed_results.extend(per_ex)
        print(f"Instance {inst} accuracy: {acc:.3%}")

    mean_acc = statistics.mean(accs.values())
    std_acc = statistics.pstdev(accs.values()) if len(accs) > 1 else 0.0

    summary = {
        "data_path": str(args.data),
        "reasoning_effort": args.reasoning_effort,
        "instances_evaluated": sorted(accs.keys()),
        "instance_acc": accs,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
    }

    out = {
        "summary": summary,
        "examples": detailed_results,
    }

    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("==== Summary ====")
    print(f"Mean accuracy over {len(accs)} instances: {mean_acc:.3%}")
    print(f"Std dev: {std_acc:.3%}")
    print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()

