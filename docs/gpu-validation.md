# RTX 3060 validation

Measured 2026-11-11 on Windows x64, RTX 3060 12,288 MiB, NVIDIA driver 610.88, CUDA Toolkit 12.8 and Python 3.12.9. The selected device is bound by UUID and remapped to CUDA0. Native startup logs confirm exactly one visible CUDA device, the RTX 3060. The other GPU, an RTX 5060 Ti, remained occupied by unrelated work and was not used by these jobs.

The engine is `ServeurpersoCom/yue2.cpp` at `5a9500fd2979ab99fc82895877aadf8583acbabb`, with GGML `210138933b22221ced02943b7b6e4308610ab530` (0.29.0). It was built locally for CUDA architecture 86 using CMake/Ninja and the installed Visual Studio C++ toolchain. Model revision and all three SHA-256 hashes were checked by `yue-music setup-models --both`. The setup script, CLI launcher and Codex MCP registration were also exercised.

## Observed runs

All runs below use eight synthesis steps, seed 7 for both token generation and acoustic noise, full ABC planning, one song, guidance 1.0, 16,384 context and VAE tile core 256. Short runs use the same style and lyrics. The long run repeats those verse/chorus sections four times; its exact request is in its JSON report. Backend elapsed time includes native startup/loading when needed, inference and artifact saving. It excludes queue time, Python CLI launch and public-service autostart.

| Case | Actual audio seconds | Backend elapsed seconds | Native-process peak dedicated MiB | Peak shared MiB |
| --- | ---: | ---: | ---: | ---: |
| Q8 staged, initial cold run | 30.00 | 109.27 | 4,215.20 | 80 |
| Q8 staged, second process | 30.00 | 80.58 | 4,215.20 | 80 |
| Q8 resident, first request | 30.00 | 67.17 | 6,949.22 | 82 |
| Q8 resident, next request in the same process | 30.00 | 17.99 | 6,949.22 | 82 |
| Q8 resident, long song from a new process | 219.24 | 163.06 | 7,909.25 | 94 |
| Q6 resident, first request | 30.00 | 49.78 | 6,113.21 | 80 |
| Q6 resident, next request in the same process | 30.00 | 14.58 | 6,113.21 | 80 |

The first staged runs overlapped Q6 downloads on the same E: drive. E: is on a SATA hard drive, not an SSD. Those runs establish memory fit and expose loading cost; they are not an isolated speed comparison against resident mode. The resident warm run logged no new model loads and spent 17.0 seconds in the native pipeline. Its WAV SHA-256 matches the first resident request exactly: `f7a0be4a107b89168ce5bf4041e97101c88a931bdbb06bbabff72a1444ca34b1`.

The long song ended on the model's end token after 5,481 semantic frames. It did not reach the requested 300-second ceiling. Its peak was **7.72 GiB dedicated**, leaving about 4.28 GiB below the card's nominal capacity. This headroom is arithmetic from the sampled process peak, not a measurement of every desktop allocation. The result demonstrates a 3-minute-39-second song, not a guarantee for every five-minute request.

Q6 saved 836 MiB in the short resident case, about 0.82 GiB. Its score differed from Q8 despite the same prompt and seed, so the timing difference includes different sampled content and is not an isolated quantized-kernel speed comparison. Neither quantization's musical quality has been scored.

## Interpretation

**The installed default is Q8 resident**, with the bounded context and decode tiles above. It fits the measured workloads and avoids repeated loading from the hard drive. Q6 and staged eviction remain explicit alternatives. The service does not silently downgrade quantization, change song settings after an OOM or select the other GPU.

The 10.5 GiB watchdog samples dedicated memory for the native child PID. Windows PDH sampling runs every 500 ms, so reported peaks are lower bounds on any shorter transient. Shared allocation is reported separately; these counters are not a proof of model offloading. GGML enables a CPU helper backend and host staging. Native logs place the LM, NAR and VAE weight buffers on CUDA0, but this is not FLUX's PyTorch all-tensor residency audit.

Native process termination after each standalone benchmark returned the 3060's reported used memory to zero. Service `stop` also releases the owned child. Models stay resident only while the default worker is alive. Stop FLUX before starting music generation on the same card.

## Verification scope

Queue/service tests cover serialization, idempotent retries with both seeds, cancellation state, interrupted-job recovery, input validation, authentication, foreign-origin rejection and artifact retrieval. Real subprocess tests exercise stdio MCP and Streamable HTTP MCP with missing-model failures. Regression tests verify NVML inventory under the SDK's filtered environment and that the shared worker survives stdio adapter disconnection. The suite now passes 17 tests, including explicit -1 random seeds and persisted seed resolution; one upstream Starlette/AnyIO deprecation warning remains.

Hardware checks verify WAV headers, 48 kHz stereo, nonzero finite samples and RMS amplitude. These checks do not score singing, lyric fidelity, composition or eight-step quality relative to 32 steps. The WAV files remain under `data/outputs` for listening.

The [live MCP validation](live-mcp-validation.json) generated a 20-second song through a real stdio session, retrieved its 1,231-character ABC score, and cancelled a second job after score generation had started. The worker remained reachable after the adapter disconnected. The real audio was then downloaded through the public REST endpoint using `yue-music download`. Final shutdown returned the 3060 to zero reported used VRAM; the other GPU remained occupied.

The first resident trial found a preflight-check error that rejected retained weights as competing usage on a second request. The check was corrected before the successful two-request resident qualification. The unsuccessful trial is retained under `docs/experiments` for traceability.

Raw evidence: [staged cold](benchmark-q8-30.json), [staged repeat](benchmark-q8-30-repeat.json), [Q8 resident pair](benchmark-q8-resident-30-fixed.json), [Q6 resident pair](benchmark-q6-resident-30.json), [long song](benchmark-q8-resident-long.json). Each successful job also retains its ABC score, semantic replay JSON, acoustic latents and request metadata. Newer jobs include a per-job native log with stage timings.
