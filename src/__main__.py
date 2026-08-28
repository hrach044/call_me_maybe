import json
from pathlib import Path
from llm_sdk import Small_LLM_Model
from .schemas import FunctionSchema
from src.decoding import DecodingContext, State, select_next_token

def main():
    # 1. Load input files
    input_dir = Path("data/input")
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_dir / "function_calling_tests.json") as f:
        prompts_data = json.load(f)

    with open(input_dir / "functions_definition.json") as f:
        functions_data = json.load(f)

    all_functions = {
    fn["name"]: FunctionSchema(**fn) for fn in functions_data
    }
    # System prompt formatting (instructing model to call a function)
    system_prompt = f"Available functions:\n{json.dumps(functions_data, indent=2)}\n\n"

    # 2. Initialize Model & Pre-decode Vocab Cache
    print("Loading model...")
    model = Small_LLM_Model()
    
    print("Pre-decoding vocabulary...")
    vocab_size = model._tokenizer.vocab_size
    vocab_strings = [model.decode([i]) for i in range(vocab_size)]

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

        # 4. Constrained Generation Loop
        while context.current_state != State.DONE:
            logits = model.get_logits_from_input_ids(input_ids)
            print(f"state={context.current_state}, built_text={context.built_text!r}")

            # Intercept logits & enforce state machine rules
            next_token_id, context = select_next_token(logits, vocab_strings, context)
            
            input_ids.append(next_token_id)
            generated_ids.append(next_token_id)

        # 5. Decode generated output
        json_output_str = model.decode(generated_ids)
        parsed_json = json.loads(json_output_str)

        # Build requested output format
        results.append({
            "prompt": user_prompt,
            "output": parsed_json
        })

    # 6. Save results
    output_path = output_dir / "predictions.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done! Results saved to {output_path}")

if __name__ == "__main__":
    main()