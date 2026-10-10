# Compact min_new_tokens benchmark evidence

This directory preserves the evidence for patch `fcdd5e12ac1865c94f73807a9dc5370518608e0c`, based on upstream `234ccba7f625e12639e621044496bcc6197c3758`. It is published on a separate evidence branch to keep the upstream PR limited to the implementation and tests.

A100-SXM4-40GB, Qwen/Qwen3-0.6B BF16 (revision `c1899de289a04d12100db370d81485cdf75e47ca`), TP1, Torch 2.14.1+cu130, Triton 3.8.0, triton attention / pytorch sampling, greedy 64-token generation. Independent service blocks use ABBA; 10 timed groups per implementation per case.

| Case | Baseline tokens/s | Patch tokens/s | Change |
|---|---:|---:|---:|
| Batch128, min=64 | 17,094.84 | 17,596.94 | +2.94% |
| Batch128, min=0 control | 18,448.63 | 18,271.69 | -0.96% |
| Batch512, min=64 | 27,626.55 | 29,429.45 | +6.53% |
| Batch512, min=0 control | 31,491.03 | 31,657.26 | +0.53% |
| Single | 366.29 | 362.53 | -1.03% |
| Single SSE | 352.47 | 345.80 | -1.89% |
| Cached batch32 SSE | 7,803.15 | 7,716.33 | -1.11% |

The batch512 comparison disables radix cache on both sides and uses max-running=512/max-total-tokens=98304 rather than 128/32768. Compare implementations within each configuration. Default SSE TTFT changes 43.09→43.64 ms (single), 69.49→70.47 ms (cached batch32); no consistent TTFT gain is claimed. A separate same-process diagnostic gives batch128 +4.07%, single SSE +0.09%.

105 tests and 23 subtests pass. 27,560 timed model responses and 32 natural EOS responses match output IDs, text and finish reason exactly. The timed prompts reach the 64-token length cap without ignore_eos; natural EOS is verified separately. See [summary.json](summary.json) and [REPORT.md](REPORT.md) (detailed report in Chinese).

## Verify raw evidence without a GPU

Download this directory including `results.tar.gz`, then run:

```sh
shasum -a 256 results.tar.gz
# expected: b36c8a021941b75fcf4454b4987649382e2cdc27a320d56b5e319f7023449b0b
tar -xzf results.tar.gz
python3 verify-results.py
```

Archive: 4,395,420 bytes, excludes model weights; expands to approximately 103 MiB of JSON/logs. It contains every raw response and timing sample, server configs, freeze/hardware records, worker logs, failure explorations and previous implementation snapshots. Unpack it for the REPORT links under `results/`. The verifier independently recomputes medians and checks hashes, sample counts, token/text/finish parity and bootstrap contents.

## Reproduce on an isolated Colab GPU

`send-prepare.py` embeds the small scripts and installs fixed dependencies/source/model directly in Colab. Run it in a fresh task-specific runtime; inspect `prepare.log`, worker PID and `READY.json` before starting `start-experiment.py`. That worker runs pytest, production-apply microbenchmarks, CUDA differential checks, default ABBA model measurements, and pytest again. After it finishes, run `run-high.py` for batch512, optionally `control.py` for the separate instrumented diagnostic, then `collect.py` to export results. See REPORT for exact commands and timing definitions. The components were executed in the reported session; the regenerated combined bootstrap was syntax/content checked, without replaying a second fresh VM.

Source and model downloads occur directly in the runtime. Retrieve and verify the archive hash before stopping only your runtime. The original task runtime has been stopped and has zero active assignments.
