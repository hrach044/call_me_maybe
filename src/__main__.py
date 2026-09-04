"""Entry point for constrained JSON function-call decoding.

Loads prompts and function schemas, runs a small local LLM under a
character-level JSON state machine to force schema-valid output, and
writes the parsed results to ``data/output/predictions.json``.
"""

import json
from pathlib import Path
from collections import defaultdict

import torch
import time 
from .patch_regex import patch_regex

from .decoding import DecodingContext, State, select_next_token
from .schemas import FunctionSchema
from llm_sdk import Small_LLM_Model


def main() -> None:
    """Run the full generation pipeline.

    Loads prompts and function definitions from ``data/input``, uses a
    local LLM plus a schema-constrained decoding state machine to
    produce a function-call JSON object per prompt, and writes all
    results to ``data/output/predictions.json``.
    """

    print(f"torch threads: {torch.get_num_threads()}")

    # 1. Load input files
    input_dir = Path("data/input")
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_dir / "function_calling_tests.json") as f:
        prompts_data = json.load(f)

    with open(input_dir / "functions_definition.json") as f:
        functions_data = json.load(f)

    # Build FunctionSchema objects for the decoding state machine
    all_functions = {
        fn["name"]: FunctionSchema(**fn) for fn in functions_data
    }

    # Compact version for the prompt text (no "returns", no indentation)
    functions_for_prompt = [
        {"name": fn["name"], "description": fn["description"], "parameters": fn["parameters"]}
        for fn in functions_data
    ]
    system_prompt = f"Available functions:\n{json.dumps(functions_for_prompt)}\n\n"

    # 2. Initialize Model & Pre-decode Vocab Cache
    print("Loading model...")
    model = Small_LLM_Model()

    print("Pre-decoding vocabulary...")
    vocab_size = model._tokenizer.vocab_size
    vocab_strings = [model.decode([i]) for i in range(vocab_size)]

    first_char_to_token_ids: dict[str, list[int]] = defaultdict(list)
    for token_id, token_str in enumerate(vocab_strings):
        if token_str:
            first_char_to_token_ids[token_str[0]].append(token_id)

    results = []

    # 3. Process each prompt
    for item in prompts_data:
        user_prompt = item["prompt"]
        full_prompt = f"{system_prompt}User: {user_prompt}\nJSON Function Call:"

        # Reset State Machine for new prompt
        context = DecodingContext(all_functions=all_functions)
        input_ids = model.encode(full_prompt)[0].tolist()
        generated_ids = []

        print(f"Generating for prompt: '{user_prompt}'...")
         # add at top of file if not already there

        print(f"prompt length in tokens: {len(input_ids)}")
        while context.current_state != State.DONE:
            t0 = time.time()
            logits = model.get_logits_from_input_ids(input_ids)
            t1 = time.time()
            print(f"model call: {t1-t0:.3f}s")
            print(f"          {context.built_text!r}")

            next_token_id, context = select_next_token(logits, vocab_strings, context, first_char_to_token_ids)
            t2 = time.time()
            print(f"select tok: {t2-t1:.3f}s")

            input_ids.append(next_token_id)
            generated_ids.append(next_token_id)

        # 5. Decode generated output
        json_output_str = model.decode(generated_ids)
        parsed_json = json.loads(json_output_str)

        # Build requested output format
        results.append({
            "prompt": user_prompt,
            "name": parsed_json["name"],
            "parameters": parsed_json["parameters"],
        })

    # 6. Save results
    output_path = output_dir / "predictions.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done! Results saved to {output_path}")
    patch_regex()

if __name__ == "__main__":
    main()