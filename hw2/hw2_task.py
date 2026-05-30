import torch
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
    PROMPT_LEN,
    MAX_NEW_TOKENS,
    VOCAB_SIZE,
    SEED,
)
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import LlamaConfig, LlamaForCausalLM



def optimized_loop(model, input_ids, n_steps):
    # TODO: fix the performance issues you found — changes may include
    # both `optimized_loop` and `generate_optimized`

    generated_ids = input_ids.clone()
    # generated_tokens = []
    past_key_values = None
    curr_input_ids = input_ids
    for _ in range(n_steps):
        outputs = model(input_ids=curr_input_ids, past_key_values=past_key_values, use_cache=True)
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        # token_value = next_token_id.item()
        # generated_tokens.append(next_token_id.item())
        curr_input_ids = next_token_id.unsqueeze(0)
        generated_ids = torch.cat([generated_ids, curr_input_ids], dim=1)
        past_key_values = outputs.past_key_values

    # print(f"genererated_tokens={generated_tokens}")
    # print(f"generated_ids={generated_ids}")
    # return generated_tokens
    return generated_ids[0, -n_steps:].to(device="cpu").tolist()


def run_profiler(loop_fn, model, input_ids, trace_name: str):
    # TODO: wrap loop_fn(model, input_ids, PROFILE_STEPS) with torch.profiler,
    # print the summary table, and export a Chrome trace to RESULTS_DIR / trace_name
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("model_inference"):
            loop_fn(model, input_ids, PROFILE_STEPS)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace(str(RESULTS_DIR / trace_name))
    return prof


def generate_optimized(optimized_trace_name: str) -> float:
    # TODO: load the model (consider dtype and other loading options),
    # then call profile() and time_generation() on optimized_loop.
    # Return the elapsed time from time_generation so main() can print a speedup.

    torch.manual_seed(SEED)
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=PROMPT_LEN + MAX_NEW_TOKENS + 64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        use_cache=True,
        torch_dtype=torch.float32,
        # device_map="auto",
        # load_in_8bit=True,
        # load_in_4bit=True,
        # trust_remote_code=True,
    )
    model = LlamaForCausalLM(config)
    model.to(device="cuda", dtype=torch.float32)
    model.eval()
    model = torch.compile(model) # , mode="reduce-overhead")

    run_profiler(optimized_loop, model, get_input_ids(), optimized_trace_name)
    return time_generation(optimized_loop, model, get_input_ids(), "Optimized")



def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    run_profiler(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


"""
============================================================================
Writeup
============================================================================

Changes made and speedup per fix:

On a NVIDIA Tesla T4:
`Original`: 1.0x
`model.compile()`: 0.93x
`use_cache=True`: 4.42x
`removing next_token_id.item() from the loop`: 4.64x
`slicing generated_ids before to("cpu")`: 4.76x
`remove mode="reduce-overhead"`: 22.70x

On a NVIDIA H100:
Original: 1.0x
model.compile(): 1.07x
use_cache=True: 5.16x
removing next_token_id.item() from the loop: 5.62x
load_in_8bit=True: 5.68x
mode="reduce-overhead": 0.63x


Biggest impact and why:
The biggest improvement was obtained when using the KV-cache.
Even after using torch.compile(model) the model was still slow since the token generation still had quadratic complexity. With the KV-cache the complexity instead becomes linear (in number of tokens) which makes the model a lot faster. Together with torch.compile (which gives us kernel fusion) I achieved about a 5.16x speedup.
Some additional speedup was achieved by removing the next_token_id.item() from the loop. This is to prevent the CPU from needing to wait for the GPU to finish after every generated token. Somewhat suprisingly, removing `mode="reduce-overhead"` also improved the speedup up to 22.7 times on a T4, but not on a H100. In theory it should have improved the performance, but the fixed cost of the operation used more time than it saved. For longer sequences the situation might have been different.

"""