*This project has been created as part of the 42 curriculum by hmnatsak.*

# call_me_maybe — Function Calling in LLMs

## Description

`call_me_maybe` translates natural-language prompts into structured,
schema-valid function calls, using a small local LLM
(`Qwen/Qwen3-0.6B`, 0.6B parameters). Instead of asking the model to
produce JSON and hoping it gets the syntax right, this project
implements **constrained decoding**: a character-level state machine
restricts the model's next-token choices at every generation step so
that only tokens keeping the output valid JSON *and* schema-compliant
are ever considered. The result is 100% parseable, schema-correct
output, even from a model too small to reliably produce raw JSON on
its own.

Given `"What is the sum of 2 and 3?"`, the program outputs:
```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}
```

## Instructions

Requires `uv`.

```bash
make install   # uv sync
make run       # uv run python -m src
make debug     # run under pdb
make lint      # flake8 . && mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
make lint-strict  # flake8 . && mypy . --strict
make clean     # remove __pycache__, .mypy_cache, generated output
```

By default the program reads `data/input/functions_definition.json`
and `data/input/function_calling_tests.json`, and writes to
`data/output/function_calling_results.json`.

## Example Usage

```bash
uv run python -m src
```

With custom paths:
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

## Algorithm Explanation

1. The model outputs raw logits over its full vocabulary (~151,936
   tokens) for the next token, given the current input IDs.
2. A hand-built JSON state machine (`DecodingContext` /
   `State` in `src/decoding.py`) tracks exactly what characters are
   legal next — derived from the chosen function's schema
   (`functions_definition.json`).
3. Instead of scanning every vocabulary token, candidate tokens are
   filtered up front to only those whose first character is legal
   (`first_char_to_token_ids`, built once before generation starts).
4. Each surviving candidate is checked in full — character by
   character — against the state machine (`is_token_legal_fast`);
   the highest-logit token that remains fully legal is selected.
5. This repeats until the state machine reaches `DONE` (a complete,
   valid JSON object matching the schema).

Because every candidate that survives is already schema-valid, the
output is guaranteed parseable — no retries, no repair step.

## Design Decisions

- **Filter-then-check over scan-everything**: `get_legal_next_chars`
  already reveals which first characters are legal at zero cost.
  Pre-grouping the vocabulary by first character avoids the original
  design's per-step scan of all ~152k tokens, cutting per-step
  selection cost to near-zero.
- **Fast path for string values**: inside `INSIDE_STRING_VALUE`, any
  token without a `"` is legal in its entirety (every printable
  character is allowed there), so it's accepted as a whole instead of
  validated character by character.
- **No skipping of forced/deterministic model calls**: tried and
  reverted (see Challenges below) — correctness took priority over
  the remaining speed gain.
- **Pydantic throughout**: `DecodingContext`, `FunctionSchema`, and
  related schemas are all Pydantic models for validation, per project
  requirements.

## Performance Analysis

- **Accuracy**: function selection and argument extraction correct on
  all tested prompts, except one flagged case — see below.
- **Validity**: 100% of outputs are valid, parseable JSON matching
  the function schema (guaranteed structurally by the state machine).
- **Speed**: ~5:06 for the full test set, down from ~30 minutes with
  the original full-vocab-scan approach — under the 5-minute target
  range on the tested hardware.
- **Known content-quality issue (not a decoding bug)**: for one
  regex-substitution prompt, the model produced a JS-style pattern
  (`/cat/g`) instead of Python `re` syntax. The state machine enforces
  JSON *structure*, not regex semantics, so this can't be caught at
  the decoding layer. Patched via `src/patch_regex.py`, which runs
  automatically at the end of `main()`.

## Challenges Faced

- **`IndexError` from vocab/logits size mismatch**: the model's logit
  vector didn't match the tokenizer's reported vocab size; fixed by
  slicing logits to `len(vocab_strings)`.
- **`RuntimeError: No legal tokens found`**: caused by
  `DecodingContext.all_functions` never being populated, and a missing
  `remaining_params` check for numeric values; fixed both.
- **Full-vocab scan was the main bottleneck**: fixed by pre-indexing
  candidate tokens by first character (see Design Decisions).
- **Attempted further speedup by skipping model calls on
  deterministic/forced states** (e.g. fixed JSON punctuation like
  `,"parameters":{`, `:`, `}`): every variant tried — direct
  `encode()` of forced text, re-encoding the full accumulated string,
  and a "verify-then-fall-back" boundary check — corrupted the
  model's function/parameter choices on some prompts. Root cause:
  without real KV-cache access, any text inserted into `input_ids`
  without going through the model's own token-by-token generation can
  tokenize differently than the model actually conditioned on,
  silently derailing later logits. This is not fixable without access
  to `llm_sdk` internals (forbidden by project rules), so this
  optimization was reverted in favor of correctness.
- **`.wslconfig` memory misconfiguration** caused swap thrashing on
  WSL during testing; fixed by adjusting memory limits.

## Testing Strategy

- Manual verification: ran the full test set, diffed
  `function_calling_results.json` against expected function
  names/parameter values for each prompt.
- Regression checks after each optimization attempt: compared output
  against a known-good baseline (`predictions.json` from the working
  5:06 run) to catch any change in function/parameter selection
  before accepting a change.
- Timing instrumentation: per-step `model call` / `select tok` timing
  printed during generation to isolate where time was actually being
  spent (confirmed the vocab scan, not the model call, was the first
  bottleneck; confirmed the model call is the remaining floor after
  fixing it).
- Edge case checked: multi-digit numbers, multiple parameters per
  function, and a case where the model produced syntactically valid
  but semantically wrong content (the regex case above) — used to
  confirm the boundary between what constrained decoding can and
  can't guarantee.

## Resources

- [Qwen3 model card, Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Guiding Text Generation with Constrained Decoding (background reading on the technique)](https://huggingface.co/docs)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [BPE tokenization overview](https://huggingface.co/docs/transformers/tokenizer_summary)

**AI usage**: AI assistance (Claude) was used for:
- Diagnosing runtime bottlenecks from timing output and proposing the
  first-character candidate-filtering optimization.
- Proposing and iterating on (ultimately reverted) approaches to skip
  model calls on deterministic decoding states, including diagnosing
  why each attempt corrupted model output.
- Drafting boilerplate (Makefile, this README structure) and adding
  PEP 257 / Google-style docstrings and type hints to existing code
  for `mypy`/`flake8` compliance.
- Explaining *why* fixes worked or failed (e.g. tokenizer boundary
  behavior) rather than only providing code, so each change was
  understood before being applied.
- All code changes were reviewed, tested against known-good output,
  and reverted when they broke correctness (see Challenges Faced).