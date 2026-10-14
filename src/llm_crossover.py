import os
import sys
import argparse
import random

# Ensure repo root is on sys.path so `src` imports work from generated bash scripts
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cfg.constants import *
from utils.print_utils import box_print
from llm_utils import (split_file, submit_mixtral, submit_mixtral_hf, 
                       llm_code_qc, str2bool, extract_note, generate_augmented_code, 
                       clean_code_from_llm, retrieve_base_code)
from qiskit_rag import build_rag_section, recent_failures
import json
from pathlib import Path


def augment_network(input_filename_x, input_filename_y, output_filename,
                    top_p=0.15, llm_model=LLM_QWEN, temperature=0.1, apply_quality_control=False,
                    qiskit_rag=False, failure_log=None):
    """Augment Python Network Script.
    
    Parameters
    ----------
    input_filename_x : os.PathLike
        _description_
    input_filename_y : os.PathLike
        _description_
    output_filename : os.PathLike
        _description_
    top_p : float, optional
        _description_, by default 0.15
    llm_model : str, optional
        _description_, by default LLM_QWEN
    temperature : float, optional
        _description_, by default 0.1
    apply_quality_control : bool, optional
        _description_, by default False
    """
    # Split the input files
    parts_x = split_file(input_filename_x)
    parts_y = split_file(input_filename_y)
    # Create tuples of parts to be augmented
    parts = [(x, y, idx) for idx, (x, y) in enumerate(zip(parts_x[1:], parts_y[1:]))]
    random.shuffle(parts)
    # Find differing parts
    for x, y, augment_idx in parts:
        if x.strip() != y.strip():
            break

    # Select a template file
    template_fname = random.choice(['crossover.txt', 'crossover_s.txt'])
    template_path = f'{ROOT_DIR}/templates/CrossOver/{template_fname}'
    with open(template_path, 'r') as file:
        template_txt = file.read()

    # Add code to be augmented
    txt2llm = template_txt.format(x.strip(), y.strip())
    # Crossover got no notes in the Oct 8 RAG run; it now gets the same per-block
    # retrieval as a mutation, queried with both parents' versions of the block.
    rag_sources = []
    if qiskit_rag:
        failures = recent_failures(failure_log, augment_idx + 1)
        rag_txt, rag_sources = build_rag_section(
            "Combine two Qiskit circuit blocks", f"{x}\n{y}", failures)
        txt2llm += rag_txt
        print(f"RAG block {augment_idx + 1}: sources {', '.join(rag_sources)}; "
              f"{len(failures)} recent error(s) fed back", flush=True)
    Path(f"{output_filename}.rag.json").write_text(json.dumps(
        {"op": "crossover", "block": augment_idx + 1, "rag": bool(qiskit_rag), "sources": rag_sources}))
    # Generate augmented code
    code_from_llm = generate_augmented_code(txt2llm, augment_idx, apply_quality_control,
                                            top_p, llm_model, temperature)
    
    if not code_from_llm or code_from_llm.strip() == "ERROR":
        # Same as llm_mutation.py: keep parent X's block rather than splicing the
        # prompt or the "ERROR" marker into the child.
        print("LLM CODE EXTRACTION FAILED. Falling back to parent chunk.", flush=True)
        code_from_llm = parts_x[augment_idx + 1].strip()
    
    # Insert note if present
    # `parts` was built from parts_x[1:], so augment_idx 0 is block 1. Indexing
    # parts_x with augment_idx wrote the merge one slot early: block 1 overwrote the
    # protected header and every other merge replaced its neighbour.
    temp_txt = parts_x[augment_idx + 1]
    note_txt = extract_note(temp_txt)
    # Update the part with augmented code
    parts_x[augment_idx + 1] = f"\n{note_txt}{code_from_llm}\n"
    # Prepare and write the augmented code to output file
    write_augmented_code(output_filename, parts_x, parts_y)
    box_print(f"Python code saved to {os.path.basename(output_filename)}", print_bbox_len=120, new_line_end=False)
    print('Job done')


def write_augmented_code(output_filename, parts_x, parts_y):
    """
    Writes the augmented code to the output file.

    Parameters
    ----------
    output_filename : os.PathLike
        _description_
    parts_x : _type_
        _description_
    parts_y : _type_
        _description_
    """    

    try:
        prompt_log_cross = parts_y[0].split("# --PROMPT LOG--\n")[0]
        prompt_log_cross = f"\n# {'='*10} Start: GeneCrossed\n{prompt_log_cross.strip()}\n# {'='*10} End:\n"
    except IndexError:
        prompt_log_cross = ""

    python_network_txt = prompt_log_cross + '# --OPTION--'.join(parts_x)

    with open(output_filename, 'w') as file:
        file.write(python_network_txt)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Augment Python Network Script.')

    # Add arguments
    parser.add_argument('input_filename_x', type=str, help='Input file name')
    parser.add_argument('input_filename_y', type=str, help='Input file name')
    parser.add_argument('output_filename', type=str, help='Output file name')
    parser.add_argument('--llm_model', type=str, default=False, help='LLM Model Name')
    parser.add_argument('--top_p', type=float, default=0.15, help='Top P value for text generation')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature value for text generation')
    parser.add_argument('--apply_quality_control', type=str2bool, default=False, help='Use LLM QC')
    parser.add_argument('--qiskit_rag', type=str2bool, default=False, help='Append per-block Qiskit RAG notes')
    parser.add_argument('--failure_log', type=str, default=None, help='JSONL of failed children (RAG error feedback)')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    augment_network(input_filename_x=args.input_filename_x,
                    input_filename_y=args.input_filename_y,
                    output_filename=args.output_filename,
                    top_p=args.top_p, 
                    llm_model=args.llm_model,
                    temperature=args.temperature,
                    apply_quality_control=args.apply_quality_control,
                    qiskit_rag=args.qiskit_rag,
                    failure_log=args.failure_log,
                   )
