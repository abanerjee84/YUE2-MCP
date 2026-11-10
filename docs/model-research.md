# YuE2 model and runtime research

Checked 2026-11-11. This note records source research before local implementation. Local installation and measured results are documented separately in the implementation and validation reports.

## Recommendation

Use **YuE2-3B Q8_0 plus the F32 YuE2 VAE through a native GGML/CUDA runtime**. Keep **Q6_K and stage eviction as lower-memory options**. The primary-source evidence below motivated a staged first test. Subsequent local qualification selected a bounded resident profile for speed; see [the hardware validation](gpu-validation.md) for the measured 3060 results. The external measurements alone do not prove every song length fits this Windows machine.

The focused [ServeurpersoCom/yue2.cpp engine](https://github.com/ServeurpersoCom/yue2.cpp) provides a native Windows CUDA build and generation/job APIs. Its published GGUFs include both Q8_0 and Q6_K. [audio.cpp](https://github.com/0xShug0/audio.cpp) is another native engine with YuE2 support; its model packages and memory behavior must be evaluated separately. GGUF is a container, not a guarantee that a file works in an arbitrary llama.cpp executable.

## What YuE2 is

This is the current public YuE2 release, not the older YuE stage-1/stage-2 pipeline. The official [m-a-p/YuE2-3B model card](https://huggingface.co/m-a-p/YuE2-3B) describes one autoregressive/non-autoregressive Mixture-of-Transformers backbone: it creates an editable ABC score and semantic tokens, generates acoustic latents by flow matching, and decodes with a separate VAE. Output is 48 kHz stereo with vocals and accompaniment. The official unquantized local example targets a 24 GB GPU.

The model card licenses the weights **CC BY-NC 4.0**. The [official repository](https://github.com/multimodal-art-projection/YuE) licenses current first-party code Apache 2.0; the native GGML ports have their own code licenses. Commercial usage needs a model-license decision, independently of which MCP wrapper is used.

## Exact model inventory

The following sizes and SHA-256 values are the publisher's Hugging Face LFS metadata, read through the [GGUF repository file API](https://huggingface.co/api/models/Serveurperso/YuE2-GGUF/tree/main). Sizes are disk bytes, not VRAM allocations.

| Component | File | Bytes | Decimal GB | Role |
| --- | --- | ---: | ---: | --- |
| Preferred backbone | `YuE2-3B-Q8_0.gguf` | 3,810,232,064 | 3.810 | AR and NAR model, text tokenizer embedded |
| Memory fallback | `YuE2-3B-Q6_K.gguf` | 2,943,331,072 | 2.943 | Alternative backbone |
| Required decoder | `YuE2-Vae-F32.gguf` | 530,497,344 | 0.530 | Neural audio encoder/decoder |
| Optional encoder | `MERT-v2-FullSong-Q8_0.gguf` | 902,283,712 | 0.902 | Recording analysis |
| Optional transcription head | `SheetSage2-Q8_0.gguf` | 105,629,856 | 0.106 | Recording to ABC score |
| Optional audio tokenizer | `yue2-mothersuperior-realaudio-tokenizer-v4-Q8_0.gguf` | 46,527,328 | 0.047 | Recording to semantic tokens |

Q8 backbone SHA-256: `41121ce97786d7795a325bcf123ca196956c03bb252c9e75384cfc1f2fc19e6b`.

Q6 backbone SHA-256: `42068d5e57713df11b9f6a5a3751072d9bbefe231d1d088f9c100e026ffa4ef9`.

VAE SHA-256: `93e49dfb1970e89ad64cacb17cf13b5d05f6bb30ef7ed3adae3050bcb728638a`.

The Q8 pair totals **4,340,729,408 bytes**, or **4.04 GiB**, on disk. The Q6 pair totals **3,473,828,416 bytes**, or **3.24 GiB**. These totals are arithmetic from the published sizes.

The [Hangry Labs model manifest](https://github.com/hangry-labs/yue2/blob/master/docker/models.json) independently pins the Q8 pair at publisher revision `e630f2b8f6aedf8254c3186438633c1b5eeaf80a`, with the same sizes and checksums. Verify the Q6 artifact against its own pinned revision before deployment.

For basic lyrics-to-song, only backbone and VAE are needed. Optional heads require a matching-quant MERT file alongside them. They are useful for recording transcription, covers and audio tokenization; they should be downloaded only when those features are implemented. [Model provenance guide](https://github.com/hangry-labs/yue2/blob/master/docs/MODELS.md)

The [GGUF publisher](https://huggingface.co/Serveurperso/YuE2-GGUF) describes Q8 as near lossless and publishes Q6_K and Q5_K_M. This is a publisher claim, not a quantified independent listening evaluation. The publisher avoids Q4 for this model and keeps the VAE F32. Prefer Q8 first; select Q6 only after comparing matched songs and peak memory.

## Memory evidence and its limits

The focused port's [architecture reference](https://hangry-labs.github.io/yue2/docs/ARCHITECTURE.html) reports these CUDA weight buffers:

| Module | Q8_0 | Q6_K |
| --- | ---: | ---: |
| AR half | 2,195.1 MB | 1,694.8 MB |
| NAR half | 1,433.5 MB | 1,107.0 MB |
| VAE decoder | 126.7 MB | Same F32 decoder |

Its default strict policy evicts the AR group before loading synthesis modules, while preserving the KV cache. One full-context KV set costs 2,688 MB; classifier-free guidance needs two. A reported 65-second Q8 song peaked **5.8 GB above idle** with the full 24,576 context, or **3.8 GB** with an 8,192 context. Smaller context requires an explicit duration and enough room for both prompt and generation. These are upstream observations, not measurements on the local RTX 3060.

**Inference:** Q6 saves about 500 MB in the AR weight allocation under this strict policy. The 867 MB disk saving does not become an equal peak-VRAM reduction because the AR and NAR halves are staged.

The same project's newer, resident-model benchmark is a materially different workload. Its [November 1 sampled-memory report](https://github.com/hangry-labs/yue2/blob/master/benchmarks/memory/BENCHMARKS.md) used an RTX 5070 Ti and 30-second songs. Peaks above idle were **9,136 to 9,229 MiB**, with **3,755 MiB** resident weight buffers. Device-wide peak reached 12,929 MiB including desktop/driver allocations. Sampling every 100 ms gives a lower bound on the actual peak, not an exact allocator measurement.

The [five-minute benchmark](https://github.com/hangry-labs/yue2/blob/master/benchmarks/long-music/BENCHMARKS.md) kept models resident, used eight flow-matching steps and full context, and observed **10,821 MiB above idle**. Whole-device sampled peak was 13,418 MiB. Five outputs ranged from 286.24 to 300 seconds, with median pipeline time 70.03 seconds on that 5070 Ti. This does not establish 3060 throughput or a safe 12 GB configuration.

The [30-second speed suite](https://github.com/hangry-labs/yue2/blob/master/benchmarks/speed/BENCHMARKS.md) reported 5.33 to 6.97 second median pipeline times across four languages on the same 5070 Ti. It excluded one warmup per language, required zero weight loads, and used eight flow steps. These numbers cannot be used as native Windows RTX 3060 latency estimates.

Practical starting policy, pending measurement:

- Use one request, one rendition and one worker at a time.
- Keep stage eviction enabled; omit `--keep-loaded` initially.
- Keep flash attention enabled. Start with the default VAE tiles; reduce tile core size if decoding determines peak memory.
- Begin with 30-60 second outputs and a context budget that includes the lyrics, score and audio tokens. Increase duration only after measuring peak VRAM.
- Start quality comparisons at the model's normal synthesis settings. Treat eight-step generation as a speed experiment, not an established quality-equivalent replacement.
- Leave 1.5-2 GiB of device headroom as an engineering target. This margin is proposed policy, not a published requirement.
- If Q8 fails, first reduce context, duration and decode tile size; then test Q6. Avoid silently changing a requested song's settings after an OOM.

## Windows dependencies

The [native upstream README](https://github.com/ServeurpersoCom/yue2.cpp/blob/master/README.md) documents Visual C++ Build Tools with the Desktop development with C++ workload, CMake, Ninja and a CUDA Toolkit for NVIDIA builds. Inference consumes the two GGUFs directly; it does not require PyTorch, Transformers, vLLM or Python model-conversion packages.

The actual [CMake project](https://github.com/ServeurpersoCom/yue2.cpp/blob/master/CMakeLists.txt) uses C++17, GGML, Threads, cpp-httplib and yyjson. The WebUI asset is committed, so building the native server does not require npm. It explicitly includes CUDA architecture 86, the RTX 3060's architecture. A dedicated 3060 build can set `CMAKE_CUDA_ARCHITECTURES=86` to avoid compiling unrelated GPU targets. Pin the engine and recursive submodules together.

Do not invoke the upstream [buildcuda.cmd](https://github.com/ServeurpersoCom/yue2.cpp/blob/master/buildcuda.cmd) unchanged: it hardcodes a Visual Studio 18 installation path and recursively removes `build`. Prefer explicit CMake commands in a verified, dedicated build directory after locating the installed compiler toolchain.

The Hangry Labs fork is a different packaging choice. Its [development guide](https://github.com/hangry-labs/yue2/blob/master/docs/DEVELOPMENT.md) makes Docker/NVIDIA CUDA its supported product baseline and describes native Windows builds as advanced. It uses C++20 application code, Task/Python development helpers, Svelte/Node UI builds and FFmpeg for exports. Its [dependency guide](https://github.com/hangry-labs/yue2/blob/master/docs/BUILDING.md) pins GGML 0.28.0 and a GCC13/CUDA12.9.1 Linux build. Avoid adding that whole stack just to create a local Windows MCP wrapper.

The [official Python package manifest](https://github.com/multimodal-art-projection/YuE/blob/main/pyproject.toml) instead pins Python >=3.10, Torch2.10.0 outside macOS, Transformers4.57.6, HuggingFaceHub0.36.2, Safetensors0.7.0, Tiktoken0.12.0, NumPy2.2.6, SoundFile0.13.1 and Accelerate1.13.0. Its fast extras, vLLM0.19.0 and Triton3.6.0, are Linux-only. This is a useful reference/parity environment, but does not satisfy the 12 GB goal merely by installing it.

## Alternative if speed or licensing changes the choice

**ACE-Step 1.5 standard/turbo**, rather than XL, is the strongest documented alternative to investigate for this memory budget. Its [official GPU guide](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GPU_COMPATIBILITY.md) recommends the 0.6B planning LM with CPU/DiT offload and INT8 for 8-12 GB; its 12-16 GB tier permits the 1.7B LM with CPU offload and INT8. On a display-attached 12 GB device, begin with 0.6B and batch one. XL is marked marginal at 12-16 GB and needs more aggressive offload, so it is not the initial efficiency choice.

The [installation guide](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INSTALL.md) supports Windows and Linux, Python3.11-3.12, approximately 10 GB of core models and a native REST server. Its [package manifest](https://github.com/ace-step/ACE-Step-1.5/blob/main/pyproject.toml) pins Windows Torch/TorchAudio2.7.1 CUDA12.8, with Transformers, Diffusers, TorchAO, TorchCodec, FastAPI, Uvicorn and nano-vllm among dependencies. This is a larger Python runtime than yue2.cpp. The [official project](https://ace-step.github.io/ACE-Step-1.5/en/) states MIT licensing and multilingual lyrics support. Do not infer which model sounds better from publisher marketing; compare the same lyrics and style on both models.

## Local acceptance criteria before building around it

The parent investigation reports an RTX 3060 with 12,288 MiB and another occupied GPU. Bind the worker to the 3060 by UUID using process-local `CUDA_VISIBLE_DEVICES`, then select its remapped CUDA0 device. Verify device name/UUID at startup and fail if it does not match. Do not change the occupied second GPU.

First qualify the exact pinned native runtime with Q8 and F32 VAE: short smoke output, a 60-second song and a representative long song, using the same seeds and settings for Q6 comparison. Record full process duration, stage timings, output length, sampled per-process and whole-device peak VRAM, and retained memory after unloading. Confirm that no other process uses the selected GPU during the measurement. Only those local results can establish the promised 12 GB operating envelope.
