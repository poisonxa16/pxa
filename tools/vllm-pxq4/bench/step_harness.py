#!/usr/bin/env python3
"""Offline bench + profile harness. Runs inside the kewaii/vllm container.
usage: harness.py --tag T --model PATH [--quant pxq4] [--nrep 1] [--profile 1]
Writes results to /mnt/models/pxa-step/results/<tag>.json and traces to
/mnt/models/pxa-step/traces/<tag>/
"""
import argparse, json, os, statistics, time

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--quant", default=None)
    p.add_argument("--nrep", type=int, default=1)
    p.add_argument("--profile", type=int, default=1)
    p.add_argument("--profile-tokens", type=int, default=48)
    p.add_argument("--profile-prompts", default="4,8")
    p.add_argument("--bench-tokens", type=int, default=200)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--tp", type=int, default=4)
    args = p.parse_args()

    OUT = "/mnt/models/pxa-step/results"
    TRACED = f"/mnt/models/pxa-step/traces/{args.tag}"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TRACED, exist_ok=True)

    from vllm import LLM, SamplingParams
    from vllm.config.profiler import ProfilerConfig

    kw = {}
    if args.quant:
        kw["quantization"] = args.quant
    if args.profile:
        kw["profiler_config"] = ProfilerConfig(
            profiler="torch", torch_profiler_dir=TRACED,
            torch_profiler_with_stack=False)

    t0 = time.time()
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="float16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              enable_prefix_caching=False, trust_remote_code=True,
              max_num_seqs=1, **kw)
    boot_s = time.time() - t0
    print(f"[{args.tag}] boot {boot_s:.1f}s", flush=True)

    PROMPTS = [
     "The history of the steam engine begins in the first century AD, when",
     "def quicksort(arr):\n    \"\"\"Sort the array using quicksort.\"\"\"\n",
     "Explain, step by step, why the sky is blue:\n1.",
     "Translate to French: 'The quick brown fox jumps over the lazy dog.' Then explain each word choice.\n",
     "A train leaves Chicago at 60 mph and another leaves New York at 80 mph. If the cities are 790 miles apart,",
     "Write a short story that begins: The lighthouse keeper had not spoken to anyone in three years, until",
     "List the planets of the solar system with one interesting fact each:\n",
     "SELECT statement to find the top 5 customers by total order value from tables customers(id,name) and orders(id,customer_id,total):\n",
     "The main differences between TCP and UDP are as follows. First,",
     "import numpy as np\n# compute the eigenvalues of a 3x3 matrix\n",
     "Summarize the causes of World War I in a concise paragraph:\n",
     "The recipe for a perfect sourdough loaf starts with the starter.",
    ]

    def sp(n):
        return SamplingParams(temperature=0.0, max_tokens=n, seed=12345,
                              ignore_eos=True)

    # warmup
    llm.generate([PROMPTS[0]], sp(32), use_tqdm=False)

    results = []
    for rep in range(args.nrep):
        for i, pr in enumerate(PROMPTS):
            t0 = time.time()
            o = llm.generate([pr], sp(args.bench_tokens), use_tqdm=False)
            dt = time.time() - t0
            ct = len(o[0].outputs[0].token_ids)
            import hashlib
            txt = o[0].outputs[0].text
            results.append({"rep": rep, "prompt_idx": i, "secs": dt,
                            "completion_tokens": ct, "tokps": ct/dt,
                            "text_sha": hashlib.sha256(txt.encode()).hexdigest()[:16],
                            "token_ids": list(o[0].outputs[0].token_ids)[:64],
                            "text_head": txt[:120]})
            print(f"[{args.tag}] rep{rep} p{i} {ct} tok in {dt:.2f}s = {ct/dt:.2f} tok/s",
                  flush=True)

    tokps = [r["tokps"] for r in results]
    summ = {"tag": args.tag, "model": args.model, "quant": args.quant,
            "n": len(tokps), "median_tokps": statistics.median(tokps),
            "mean_tokps": statistics.mean(tokps),
            "min": min(tokps), "max": max(tokps),
            "stdev": statistics.stdev(tokps) if len(tokps) > 1 else 0.0,
            "boot_s": boot_s}
    print(json.dumps(summ, indent=1), flush=True)

    if args.profile:
        for pi in [int(x) for x in args.profile_prompts.split(",")]:
            print(f"[{args.tag}] profiling prompt {pi} x {args.profile_tokens} steps...", flush=True)
            llm.start_profile(profile_prefix=f"prof_p{pi}")
            llm.generate([PROMPTS[pi]], sp(args.profile_tokens), use_tqdm=False)
            llm.stop_profile()
            time.sleep(5)

    with open(f"{OUT}/{args.tag}.json", "w") as f:
        json.dump({"summary": summ, "results": results}, f, indent=1)
    print(f"[{args.tag}] DONE", flush=True)

if __name__ == "__main__":
    main()
