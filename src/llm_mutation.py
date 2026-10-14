import os
import sys
import re
import json
import time
import glob
import numpy as np
import transformers
import argparse
from pathlib import Path

# Ensure repo root is on sys.path so `src` imports work even when launched from nested dirs
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cfg.constants import *
from utils.print_utils import box_print

from llm_utils import (split_file, submit_mixtral, submit_mixtral_hf,
                       llm_code_qc, str2bool, generate_augmented_code,
                       extract_note, clean_code_from_llm, retrieve_base_code,
                       get_llm_code_generator)


def validate_generated_chunk(code_from_llm):
    """Reject common invalid patterns as defined by config"""
    try:
        for pattern, reason in FORBIDDEN_PATTERNS:
            if pattern in code_from_llm:
                return False, reason
        return True, ""
    except NameError:
        return True, ""


def gene_id_from_path(filename):
    """network_<gene>.py -> <gene>; the seed file (network.py) -> 'seed'."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    prefix = f"{MODEL}_"
    return stem[len(prefix):] if stem.startswith(prefix) else "seed"


def load_parent_metrics(parent_id):
    path = os.path.join(SOTA_ROOT, "results", f"{parent_id}_metrics.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def summarize_metrics(m):
    """Train and validation numbers only. Test numbers are deliberately left out."""
    starts = m.get("starts", [])
    per_start = ", ".join(f"{s['val_accuracy']:.0%}" for s in starts)
    hist = ", ".join(f"{k} x{v}" for k, v in sorted(m.get("gate_histogram", {}).items()))
    lines = [
        f"- Validation accuracy: mean {m['val_accuracy']:.1%}, sd {m.get('val_accuracy_sd', 0.0):.1%} "
        f"over {len(starts)} training starts ({per_start})",
        f"- Train accuracy: {m['train_accuracy']:.1%} (train minus validation: "
        f"{m['train_accuracy'] - m['val_accuracy']:+.1%})",
        f"- Validation cross-entropy: {m['val_cross_entropy']:.3f}",
        f"- Gates: {m['gates']:.0f} total ({m.get('two_qubit_gates', 0)} two-qubit), "
        f"depth {m.get('depth')}, {m.get('n_params')} trainable parameters",
        f"- Gate counts: {hist}",
        f"- Rotations with a trained angle near zero (dead): {m.get('dead_rotations', 0)} of {m.get('rotations', 0)}",
        f"- Training budget fully used: {m.get('budget_spent')}",
    ]
    return "\n".join(lines)


def build_critic_prompt(summary, chunk, max_words):
    return (
        "You are reviewing a variational quantum classifier written in Qiskit. "
        "It classifies the 3-class Wine dataset with 8 qubits (PCA features, angle encoding) "
        "and reads the class from qubits 0 and 1. Higher validation accuracy and fewer gates are both better.\n\n"
        "Measured results for the current circuit (training and validation only):\n"
        f"{summary}\n\n"
        "Another model is about to edit this block of the circuit code:\n"
        f"```python\n{chunk}\n```\n\n"
        f"In at most {max_words} words, say what these numbers suggest is limiting this circuit "
        "and name one specific kind of change to THIS block that could help. "
        "Write plain sentences. Do not write code and do not use code fences."
    )


def get_critique(prompt, llm_model, top_p, max_words):
    llm_code_generator, _ = get_llm_code_generator(llm_model)
    text = llm_code_generator(prompt, max_new_tokens=300, top_p=top_p, temperature=0.2)
    if not text or not isinstance(text, str) or text.startswith("No output received"):
        return None
    if prompt in text:                      # defensive: some servers echo the prompt
        text = text.split(prompt)[-1]
    text = text.replace("```", "").strip()
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text or None


def write_record(output_filename, record):
    results_dir = os.path.join(SOTA_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{gene_id_from_path(output_filename)}_mutation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)


def augment_network(input_filename='network.py', output_filename='network_x.py', template_txt=None,
                    top_p=0.15, llm_model=LLM_DEEPSEEK, temperature=0.1, apply_quality_control=False):

    print(f'Loading {input_filename} code')
    parts = split_file(input_filename)
    augment_idx = np.random.randint(1, len(parts))
    # select code to be augmented randomly
    code2llm = parts[augment_idx]

    template_path = os.path.join(ROOT_DIR, template_txt)
    with open(template_path, 'r') as file:
        template_body = file.read()

    # Prompt templates use a single literal "{}" marker for the code chunk.
    if "{}" not in template_body:
        raise ValueError(f"Prompt template {template_path} must contain a literal {{}} code placeholder")
    txt2llm = template_body.replace("{}", code2llm.strip(), 1)

    record = {"child": gene_id_from_path(output_filename), "parent": gene_id_from_path(input_filename),
              "block_index": int(augment_idx), "template": os.path.basename(template_path),
              "critic_on": bool(QISKIT_CRITIC), "critic_status": "off", "critique": None}

    if QISKIT_CRITIC:
        record["critic_status"] = "pending"
        metrics = load_parent_metrics(record["parent"])
        critique = None
        if metrics is None:
            record["critic_status"] = "skipped: no parent metrics"
        else:
            try:
                summary = summarize_metrics(metrics)
                record["metrics_summary"] = summary
                critique = get_critique(build_critic_prompt(summary, code2llm.strip(), CRITIC_MAX_WORDS),
                                        llm_model, top_p, CRITIC_MAX_WORDS)
            except (KeyError, TypeError, ValueError) as error:
                record["critic_status"] = f"skipped: bad metrics ({error})"
        if critique:
            record["critic_status"] = "ok"
            record["critique"] = critique
            txt2llm += ("\n\nReviewer feedback on the current circuit, based on its measured results:\n"
                        + critique
                        + "\nUse this feedback if it helps, but keep following every constraint above.")
            box_print("CRITIQUE", print_bbox_len=60, new_line_end=False)
            print(critique, flush=True)
        elif record["critic_status"] == "pending":
            record["critic_status"] = "failed: no critique returned"

    code_from_llm = generate_augmented_code(txt2llm, augment_idx-1, apply_quality_control,
                                            top_p, llm_model, temperature)

    outcome = "ok"
    if not code_from_llm or code_from_llm == "ERROR":
        outcome = "no_code"
    else:
        valid_code, invalid_reason = validate_generated_chunk(code_from_llm)
        if not valid_code:
            outcome = f"rejected: {invalid_reason}"
    record["outcome"] = outcome
    write_record(output_filename, record)
    if outcome != "ok":
        # A traceback makes the orchestrator treat this mutation as failed and keep the parent.
        raise RuntimeError(f"Mutation abandoned ({outcome}); parent is kept")

    note_txt = extract_note(code2llm)
    parts[augment_idx] = f"\n{note_txt}{code_from_llm}\n"
    python_network_txt = '# --OPTION--'.join(parts)
    output_file = Path(output_filename)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    output_file.write_text(python_network_txt)

    box_print(f"Python code saved to {os.path.basename(output_filename)}", print_bbox_len=120, new_line_end=False)
    print('Job Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment Python Network Script.')
    parser.add_argument('input_filename', type=str, help='Input file name')
    parser.add_argument('output_filename', type=str, help='Output file name')
    parser.add_argument('template_txt', type=str, help='Template txt')
    parser.add_argument('--llm_model', type=str, default=False, help='LLM Model Name')
    parser.add_argument('--top_p', type=float, default=0.15, help='Top P value for text generation')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature value for text generation')
    parser.add_argument('--apply_quality_control', type=str2bool, default=False, help='Use LLM QC')
    args = parser.parse_args()

    augment_network(input_filename=args.input_filename,
                    output_filename=args.output_filename,
                    template_txt=args.template_txt,
                    llm_model=args.llm_model,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    apply_quality_control=args.apply_quality_control,
                   )