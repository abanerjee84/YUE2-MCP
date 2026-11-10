# YUE2-MCP

Local **YuE2 music generation and score editing** through **MCP, a global CLI, or HTTP**. One persistent worker owns the GPU models and serves every connected client through a shared queue.

Built and tested for an **NVIDIA RTX 3060 with 12 GB VRAM on Windows**:

- **YuE2-3B Q8_0 GGUF** backbone with the **F32 YuE2 VAE**.
- Native **GGML/CUDA** inference through [yue2.cpp](https://github.com/ServeurpersoCom/yue2.cpp).
- **48 kHz stereo** WAV or MP3, plus an editable **ABC score**, semantic replay data and acoustic latents.
- Q6_K and staged loading as explicit lower-memory options.
- MCP over stdio and Streamable HTTP, plus REST and a command-line interface.

The host agent supplies the style, lyrics and optional score. YuE2 composes the score and audio. Inference uses native CUDA binaries; it does not install PyTorch, Transformers, Diffusers, ComfyUI or vLLM.

**Model weights are not included.** Setup downloads pinned Hugging Face files and verifies SHA-256. YuE2 weights use **CC BY-NC 4.0**; see [licenses](#attribution-and-model-licenses).

For a first installation, follow the [step-by-step guide](#installation) or give an agent the [installation brief](#installation-instructions-for-an-agent). For daily use, jump to [common commands](#common-commands-start-to-stop).

## Contents

- [Measured results](#measured-results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Installation instructions for an agent](#installation-instructions-for-an-agent)
- [Common commands: start to stop](#common-commands-start-to-stop)
- [Generate music and edit a score](#generate-music-and-edit-a-score)
- [Complete CLI reference](#complete-cli-reference)
- [Agent harness integration](#agent-harness-integration)
- [MCP tools](#mcp-tools)
- [HTTP API](#http-api)
- [Architecture and worker lifecycle](#architecture-and-worker-lifecycle)
- [GPU policy and request limits](#gpu-policy-and-request-limits)
- [Configuration and local files](#configuration-and-local-files)
- [Pinned models and dependencies](#pinned-models-and-dependencies)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Updating and removing the installation](#updating-and-removing-the-installation)
- [Attribution and model licenses](#attribution-and-model-licenses)

## Measured results

Observed on RTX 3060 12 GB, Windows x64, driver 610.88, CUDA Toolkit 12.8 and Python 3.12.9. All rows use full ABC planning, eight synthesis steps, one song, seed 7 for both stages, 16,384-token context and VAE tile core 256.

| Configuration | Actual audio | Backend time | Peak dedicated VRAM | Peak shared memory |
| --- | ---: | ---: | ---: | ---: |
| **Q8 resident, warm request** | **30.00 s** | **17.99 s** | **6,949 MiB / 6.79 GiB** | 82 MiB |
| Q8 resident, first request | 30.00 s | 67.17 s | 6,949 MiB / 6.79 GiB | 82 MiB |
| Q8 resident, longer song including loading | 219.24 s | 163.06 s | 7,909 MiB / 7.72 GiB | 94 MiB |
| Q6 resident, warm request | 30.00 s | 14.58 s | 6,113 MiB / 5.97 GiB | 80 MiB |
| Q6 resident, first request | 30.00 s | 49.78 s | 6,113 MiB / 5.97 GiB | 80 MiB |
| Q8 staged, separate process | 30.00 s | 80.58 s | 4,215 MiB / 4.12 GiB | 80 MiB |

**Q8 resident is the default.** It fits the measured workloads and avoids reloading weights between jobs. The two matched Q8 resident requests produced byte-identical WAV files. Q6 saved about 0.82 GiB in the short resident test, but changed the sampled score; its timing is not a comparison of identical generated content.

Backend time includes native startup/loading when needed, inference and saving artifacts. It excludes public-service autostart, CLI launch and queue time. The E: model drive was a SATA hard drive. Staged tests overlapped model downloads on that drive, so their timing is not an isolated speed comparison. Cold starts depend on storage and system state.

The long song ended naturally at **3 minutes 39 seconds**, before its requested 300-second ceiling. Duration is a ceiling, not an exact composition length. Memory counters were sampled every 500 ms and can miss shorter peaks. Audio integrity was checked, but singing quality and lyric fidelity were not scored.

See the [validation report](docs/gpu-validation.md), [Q8 measurements](docs/benchmark-q8-resident-30-fixed.json), [Q6 measurements](docs/benchmark-q6-resident-30.json) and [long-song measurements](docs/benchmark-q8-resident-long.json).

## Requirements

| Item | Requirement |
| --- | --- |
| OS | Windows x64; installer, launcher and background service lifecycle are Windows-specific |
| GPU | NVIDIA **RTX 3060 12 GB**, with at least 7,000 MiB free at native worker startup |
| Driver | CUDA 12.8-compatible NVIDIA driver; tested with **610.88** |
| Python | **3.12**, selected or downloaded by uv |
| Setup tools | Git and [uv](https://docs.astral.sh/uv/getting-started/installation/) |
| C++ toolchain | Visual Studio **2022** C++ tools, Windows SDK, CMake and Ninja |
| CUDA Toolkit | **12.8** at its default Windows installation path |
| Model storage | **4.34 GB** for Q8 + VAE; **7.28 GB** with both Q8 and Q6 |
| Other storage | Additional space for native sources/build output, Python, packages, temporary files and generated audio |
| System RAM | Used for loading, host staging and normal application work; measured machine had 64 GB |
| Network | Needed for source/package/model installation; generation then reads local files |

The selector requires a GPU name containing RTX 3060, and startup also checks for at least 12,000 MiB total VRAM. Other GPU families, smaller-memory variants and Linux/WSL installations are not validated targets.

The engine is compiled locally. No Docker installation, separate MERT model or recording-transcription model is required for lyrics-to-song generation.

## Installation

Use **PowerShell**. Run one block at a time and let each step finish.

### 1. Check the GPU

~~~powershell
nvidia-smi
~~~

Confirm the RTX 3060 has **12 GB** and sufficient free VRAM. Stop another model worker using that card before generating music. If this command is unavailable, install or repair the NVIDIA driver first.

### 2. Install the setup tools

Check what is already installed:

~~~powershell
git --version
uv --version
~~~

For missing tools, use the official [Git for Windows](https://git-scm.com/install/windows) and [uv installation](https://docs.astral.sh/uv/getting-started/installation/) instructions. Close and reopen PowerShell afterward so PATH changes take effect.

Install **Visual Studio 2022 Build Tools** or use an existing VS 2022 installation. In Visual Studio Installer, select **Desktop development with C++**, including the MSVC v143 x64/x86 toolset, a Windows SDK and **C++ CMake tools for Windows**. The latter supplies CMake and Ninja. See [Microsoft's C++ installation guide](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170).

Install **CUDA Toolkit 12.8** from [NVIDIA's archive](https://developer.nvidia.com/cuda-12-8-0-download-archive) at:

~~~text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
~~~

The build and runtime currently use that location. Installing the toolkit does not require replacing a working newer NVIDIA driver.

### 3. Clone into a permanent folder

~~~powershell
New-Item -ItemType Directory -Path E:\TOOLS -Force | Out-Null
git clone https://github.com/abanerjee84/YUE2-MCP.git E:\TOOLS\YUE2-MCP
Set-Location E:\TOOLS\YUE2-MCP
~~~

If there is no E: drive, choose another permanent folder and use it throughout. Reuse an existing checkout instead of cloning over it. Registrations use absolute paths, so keep the installed folder in place.

### 4. Build and install the application

~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
~~~

This creates .venv, installs locked Python dependencies and the editable application, checks out the pinned yue2.cpp engine and GGML submodule, builds CUDA executables for architecture 86, and installs the user PATH launcher. Initial CUDA compilation takes time.

The build locates Visual Studio through vswhere, with a fallback for the standard VS 2022 Community path. It writes into `vendor/yue2.cpp/build` and does not run the upstream script that deletes its build directory. The execution-policy override applies only to this PowerShell process.

**This step installs the application, not the weights.**

### 5. Download the models

~~~powershell
.\.venv\Scripts\python.exe -m yue_music setup-models
~~~

The default downloads are the Q8 backbone and F32 VAE, about **4.34 GB**. Resumable HTTPS ranges are verified by SHA-256 before installation. If interrupted, run the same command again.

The [GGUF repository](https://huggingface.co/Serveurperso/YuE2-GGUF) was accessible without a separate access form during validation. If Hugging Face requires authentication or acceptance, complete it with your account. Existing local credentials are used. Sign in if needed:

~~~powershell
.\.venv\Scripts\hf.exe auth login
~~~

Keep authentication local. Tokens do not belong in source files, request JSON or issue reports.

### 6. Connect your agent

Run the row for your harness, or several rows if needed:

| Agent harness | Command from the project folder |
| --- | --- |
| Codex | `.\.venv\Scripts\python.exe -m yue_music register codex` |
| Claude Code | `.\.venv\Scripts\python.exe -m yue_music register claude-code` |
| Gemini CLI | `.\.venv\Scripts\python.exe -m yue_music register gemini` |
| Other MCP client | `.\.venv\Scripts\python.exe -m yue_music mcp-config` |

Registration is user-level, preserves unrelated settings and backs up changes. Restart the harness to discover **local-yue**. Registration refuses to replace a conflicting existing entry.

### 7. Verify readiness

~~~powershell
.\.venv\Scripts\python.exe -m yue_music doctor
~~~

Check that `inference.ready` is true, `missing_files` is empty and the selected UUID belongs to the intended RTX 3060. `loaded: false` and `service_running: false` are normal before the first request.

Open a new PowerShell window to use the global command:

~~~powershell
yue-music doctor
~~~

Without refreshing PATH, use `.\bin\yue-music.cmd` or `.\.venv\Scripts\python.exe -m yue_music` instead.

Combined installation shortcut:

~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -DownloadModels -RegisterCodex
~~~

## Installation instructions for an agent

Copy this brief into a local coding agent:

~~~text
Install https://github.com/abanerjee84/YUE2-MCP on this Windows computer.
Use E:\TOOLS\YUE2-MCP unless I choose another permanent location.

1. Inspect an existing checkout first. Preserve local changes and reuse
   existing verified weights instead of downloading duplicate copies.
2. Verify Windows x64, an RTX 3060 with 12 GB, a compatible NVIDIA driver,
   Git, uv, Visual Studio 2022 C++ tools with CMake/Ninja, and CUDA 12.8 at
   its default path. Report a hardware mismatch instead of bypassing GPU
   checks or silently selecting a different card.
3. Clone if needed. Run scripts/setup.ps1 from its folder. Check native
   command exit codes and retain the pinned engine and submodule.
4. Run the project Python with -m yue_music setup-models. Reuse existing
   Hugging Face authentication. Let me complete any required access/login
   locally; keep tokens out of chat and repository files.
5. Register only the harness I use through yue-music register. Preserve
   unrelated settings and explain that a harness restart is needed.
6. Run yue-music doctor. Confirm inference.ready, models and the intended
   GPU UUID. Stop competing work on that card before a short generation;
   do not terminate unrelated processes without scope.
7. Generate examples/song.json, check its result/audio, then stop the
   worker. Report output paths, measured time and memory.

Weights use CC BY-NC 4.0. Do not claim commercial model rights, exact
requested song duration or guaranteed musical quality.
~~~

## Common commands: start to stop

From the project folder in a new PowerShell window:

~~~powershell
# Optional: the first generation can also start the service automatically.
yue-music start
yue-music doctor

# The example request includes its lyrics.
yue-music generate --request .\examples\song.json --wait --out .\data\first-song.wav

# Release the native worker and GPU memory when finished.
yue-music stop
~~~

Start launches the public service. The native process and weights initialize on the first job. Resident mode retains weights for later requests. Connecting another client uses the same queue and does not create another model copy.

Stop requests shutdown and returns stopping. Shutdown cancels active work and releases the owned native process. Before changing profiles or benchmarking, wait until doctor reports `service_running: false`.

## Generate music and edit a score

### Style and lyrics

~~~powershell
yue-music generate --style "English, indie folk pop, warm lead vocal, acoustic guitar, bass, gentle drums, 100 BPM" --lyrics-file .\examples\lyrics.txt --duration 60 --seed 7 --wait --out .\data\my-song.wav
~~~

Use section tags such as `[Verse]` and `[Chorus]`. The [example lyrics](examples/lyrics.txt) are an original verse and chorus used for the tests. Omitting lyrics permits an instrumental request; musical adherence needs listening.

Duration sets a **10–300 second maximum**. The model can finish earlier or hit the cap mid-phrase. Results include `duration_seconds` and `hit_duration_ceiling`. Eight synthesis steps are the speed default; `--steps` 32 performs more synthesis computation. The quality difference has not been scored locally.

Omit `--wait` and `--out` to return immediately with a job ID:

~~~powershell
yue-music generate --request .\examples\song.json
yue-music status JOB_ID
yue-music wait JOB_ID --timeout 3600
yue-music download JOB_ID --out .\data\song.wav
yue-music download JOB_ID --artifact score --out .\data\score.abc
yue-music download JOB_ID --artifact replay --out .\data\replay.json
yue-music cancel JOB_ID
~~~

Replace `JOB_ID` with the returned identifier. Queued cancellation changes that job's state; active cancellation stops this service's native child. The next job starts a new native process.

### Planning and editing

| Mode | Behavior |
| --- | --- |
| `--cot full` | Default: compose a melody-and-chord ABC score before audio |
| `--cot melody` | Compose a melody plan with free accompaniment |
| `--cot off` | Generate directly from style and lyrics |

Download a successful score, revise its ABC notation, then supply it with compatible lyrics:

~~~powershell
yue-music generate --style "English, jazz trio, warm vocal, brushed drums, upright bass" --lyrics-file .\examples\lyrics.txt --abc-file .\data\edited.abc --cot melody --duration 90 --seed 7 --wait
~~~

The `edited.abc` file must contain your revised score. External ABC requires full or melody mode. Editing creates a new recording, without preserving waveform regions. Automatic recording transcription and audio-conditioned covers are not implemented in this wrapper.

### Files and seeds

Successful jobs retain these files under `data/outputs/`:

| Artifact | Contents |
| --- | --- |
| `JOB_ID.wav or .mp3` | 48 kHz stereo audio |
| `JOB_ID.abc` | Generated/supplied score; can be empty with planning off |
| `JOB_ID.replay.json` | Native request, score, semantic tokens and seeds |
| `JOB_ID.vae` | Acoustic latents returned by the engine |
| `JOB_ID.json` | Public request, result and memory measurements |
| `JOB_ID.native.log` | Native stage/load timings and generation log |

Replay JSON and latents are native artifacts. The public CLI does not accept replay-token or latent inputs; use the pinned engine's documented tools for those operations.

Use `--format` mp3 for MP3 instead of WAV16. Match the `--out` extension to the format: copying does not transcode. The token seed `--lm-seed` defaults to the acoustic `--seed`. Omitted seeds are generated and stored. Fixed seeds apply within the tested runtime, not as a cross-hardware guarantee.

For retries, provide the same `--request-id` and generation settings. Reusing an identifier with different settings is rejected.

## Complete CLI reference

Commands emit JSON except the stdio MCP process, native build output and download progress. Use `yue-music COMMAND --help` for flags.

| Command | Purpose |
| --- | --- |
| `doctor` | Inspect GPUs, profile, missing files, readiness and last memory |
| `start` | Start/connect to the shared public service |
| `stop` | Request shutdown and release the native worker |
| `serve` | Run HTTP/MCP in the foreground |
| `generate` | Submit style, lyrics and optional score |
| `status JOB_ID` | Read state, stage, result and error |
| `wait JOB_ID --timeout 3600` | Wait for a terminal state |
| `cancel JOB_ID` | Cancel queued/active work |
| `download JOB_ID --out PATH` | Copy audio; `--artifact` score or replay selects text |
| `setup-models` | Download/verify the configured quantization and VAE |
| `setup-models --both` | Download Q8, Q6 and VAE |
| `configure --quant Q8_0` | Select Q8_0 or Q6_K while stopped |
| `configure --residency resident` | Retain weights; staged evicts between stages |
| `configure --max-seq 16384` | Select 8192, 16384 or 24576 context while stopped |
| `install-cli` | Refresh the user PATH launcher |
| `register HARNESS` | Register codex, claude-code or gemini |
| `mcp-config` | Print a generic stdio entry |
| `mcp-config --http` | Print HTTP MCP endpoint/authentication guidance |
| `mcp` | Run the lightweight stdio adapter |

Generate accepts either `--style` or `--request` FILE. Other flags are `--lyrics`, `--lyrics-file`, `--abc-file`, `--duration`, `--cot`, `--steps`, `--seed`, `--lm-seed`, `--format`, `--request-id`, `--wait`, `--timeout` and `--out`.

A request JSON supplies all generation settings; other generation flags do not override it. The `--wait`, `--timeout` and `--out` flags still control CLI waiting/copy behavior. See [examples/song.json](examples/song.json).

## Agent harness integration

The [installation step](#6-connect-your-agent) registers the adapter at user scope. Codex uses its user `config.toml`, respecting `CODEX_HOME`; Claude Code uses `~/.claude.json`; Gemini uses `~/.gemini/settings.json`. Unrelated settings are preserved and changed files receive backups.

For other clients, run yue-music mcp-config and copy its generated entry. The [example configuration](mcp-config.example.json) shows the shape; replace paths with your installation's absolute paths. Use the project's virtual environment Python.

The MCP server is **local-yue**. Submissions return immediately; poll get_music_job until succeeded, failed or cancelled. Audio paths refer to the service computer. An isolated client should download artifacts over authenticated HTTP.

## MCP tools

| Tool | Behavior |
| --- | --- |
| `generate_music` | Queue a request; return `job_id` immediately |
| `get_music_job` | Read state, stage, paths, duration, seeds, errors and memory |
| `cancel_music_job` | Cancel queued work or the current native job |
| `music_service_status` | Read model availability, GPU and residency profile |
| `get_music_score` | Return a successful job's ABC text |

Generation fields:

| Field | Default / allowed value |
| --- | --- |
| style | Required nonempty text, up to 2,000 characters |
| lyrics | Empty; up to 12,000 characters |
| abc | Empty; up to 20,000 characters; requires full/melody |
| duration | 60; finite value from 10 to 300 seconds |
| cot | full; also melody or off |
| steps | 8; integer from 4 to 32 |
| seed, lm_seed | Optional integers from 0 to 4,294,967,295 |
| `output_format` | wav16; also mp3 |
| `client_request_id` | Optional retry identifier, up to 128 characters |

Character limits do not guarantee a fit in the token context. Long lyrics plus a long score can exceed the chosen context and fail the job.

## HTTP API

Start with yue-music start. Public base URL: **http://127.0.0.1:8766**. Streamable HTTP MCP: **/mcp**.

Every request needs Authorization: Bearer TOKEN. Read the generated token locally from ignored `data/config.json` without printing it:

~~~powershell
$yueConfig = Get-Content -LiteralPath .\data\config.json -Raw | ConvertFrom-Json
$yueHeaders = @{ Authorization = 'Bearer ' + $yueConfig.token }
$yueBase = 'http://127.0.0.1:' + $yueConfig.port
Invoke-RestMethod -Uri "$yueBase/health" -Headers $yueHeaders

$yueRequest = Get-Content -LiteralPath .\examples\song.json -Raw
$yueJob = Invoke-RestMethod -Method Post -Uri "$yueBase/v1/jobs" -Headers $yueHeaders -ContentType 'application/json' -Body $yueRequest
$yueJob.job_id
~~~

Poll GET /v1/jobs/{`job_id`}. After success, download audio:

~~~powershell
Invoke-WebRequest -Uri "$yueBase/v1/jobs/$($yueJob.job_id)/audio" -Headers $yueHeaders -OutFile .\data\http-song.wav
~~~

| Method and path | Result |
| --- | --- |
| `GET /health` | Identity, readiness, GPU, profile and last memory |
| `POST /v1/jobs` | Validate/enqueue; HTTP 202 |
| `GET /v1/jobs/{id}` | State, request, result or error |
| `POST /v1/jobs/{id}/cancel` | Cancel queued/active work |
| `GET /v1/jobs/{id}/audio` | Successful WAV or MP3 |
| `GET /v1/jobs/{id}/score` | Successful ABC text |
| `GET /v1/jobs/{id}/replay` | Successful native replay JSON |
| `POST /admin/shutdown` | Request shutdown of an internally managed service |

Artifacts return 409 until success. Invalid requests return 422; a reused request ID with different settings returns 409; the full 32-job queue returns 429. Inference errors appear in job state, not the original submission's 202 response.

The public service binds to loopback, checks hosts and rejects foreign browser origins. The native worker has a separate unauthenticated loopback endpoint on port 8767. This is a local single-user deployment; neither endpoint is configured for public hosting.

## Architecture and worker lifecycle

~~~text
CLI / stdio MCP adapter / HTTP MCP / REST
                  |
       authenticated public service :8766
                  |
        persistent SQLite FIFO queue
                  |
          one inference thread
                  |
       owned native CUDA worker :8767
                  |
  ABC planning -> semantic tokens -> acoustic flow -> VAE
                  |
        audio + score + replay + latents
~~~

The design adapts [FLUX3-MCP](https://github.com/abanerjee84/FLUX3-MCP)'s singleton client, queue, registration, resumable downloads and telemetry. Image inference is replaced by a native music backend. See [implementation notes](docs/implementation.md) and [model/runtime research](docs/model-research.md).

File locks serialize startup and enforce one worker per data directory. SQLite retains requests, retry identity and terminal results. An interrupted running request is marked failed on restart; queued work remains available.

Windows background startup uses hidden WMI outside the adapter's process job, so adapter disconnection does not kill the shared service. The launch environment restores public Program Files paths needed by this driver's NVML initialization. Both behaviors have regression tests. No Windows service or scheduled task is installed.

## GPU policy and request limits

The RTX 3060 is selected by **UUID**, persisted in configuration and exposed to the native child as CUDA0. The worker does not silently switch GPUs or downgrade quantization after a failure.

Defaults: Q8 resident, batch one, 16,384-token context, flash attention, VAE tile core 256, eight synthesis steps and semantic guidance 1.0. A startup free-memory check and sampled **10.5 GiB dedicated-memory watchdog** catch budget violations. The watchdog stops the owned child; it is not an allocator hard limit and can miss transient peaks.

Windows shared allocations are reported separately. GGML uses CPU helpers and staging; this project does not make FLUX's PyTorch all-tensors-on-CUDA audit claim.

To select Q6:

~~~powershell
yue-music setup-models --both
yue-music stop
# Wait for doctor to report service_running: false.
yue-music configure --quant Q6_K
~~~

To change residency or restore the default, while stopped:

~~~powershell
yue-music configure --residency staged
yue-music configure --quant Q8_0 --max-seq 16384 --residency resident
~~~

Staged mode evicts model groups between stages and requests. It lowers memory use at the cost of loading. Smaller context reduces KV memory but limits prompt/score/audio budgets. The 300-second ceiling is a bound, not an exact duration or universal memory-fit guarantee.

FLUX and YuE2 share the same card. **Stop FLUX before music generation**, and stop YuE2 before returning the card to another model.

## Configuration and local files

Runtime state defaults to the checkout's data/ folder. `YUE_MUSIC_HOME` selects another absolute directory. Keep it consistent across clients so they share one queue and worker.

| Path | Contents |
| --- | --- |
| `data/config.json` | Token, GPU UUID, ports, quantization, context, residency and memory budget |
| `data/models/` | Verified GGUFs, license, manifest and download state |
| `data/jobs.sqlite3` | Queue, requests and results |
| `data/outputs/` | Audio and associated artifacts |
| `data/service.log` | Public service log |
| `data/native.log` | Native worker log |
| `vendor/yue2.cpp/` | Pinned source and GGML submodule |
| `vendor/yue2.cpp/build/` | Executables and build files |
| `.venv/` | Isolated Python environment |
| `bin/` | Command launcher |

Default public/native ports are 8766/8767. Change them in `config.json` only while stopped and keep them distinct. Use configure for quantization, context and residency. Do not publish the token-bearing configuration.

Outputs and job history are retained until removed manually; there is no automatic retention policy. Weights, generated files, credentials and binaries are excluded from Git.

## Pinned models and dependencies

Publisher: [Serveurperso/YuE2-GGUF](https://huggingface.co/Serveurperso/YuE2-GGUF), revision **e630f2b8f6aedf8254c3186438633c1b5eeaf80a**.

| File | Disk bytes | Use |
| --- | ---: | --- |
| `YuE2-3B-Q8_0.gguf` | 3,810,232,064 | Default backbone |
| `YuE2-3B-Q6_K.gguf` | 2,943,331,072 | Optional backbone |
| `YuE2-Vae-F32.gguf` | 530,497,344 | Required decoder |

Hashes are pinned in [models.py](src/yue_music/models.py). Basic generation does not download MERT, SheetSage2 or audio-tokenizer heads.

Native source: [ServeurpersoCom/yue2.cpp](https://github.com/ServeurpersoCom/yue2.cpp) commit 5a9500fd2979ab99fc82895877aadf8583acbabb; GGML submodule 210138933b22221ced02943b7b6e4308610ab530 (0.29.0). The engine carries its own HTTP/JSON/audio dependencies.

Python dependencies are declared in [pyproject.toml](pyproject.toml) and locked in [requirements.lock.txt](requirements.lock.txt), including MCP 1.30.0, FastAPI 0.115.12, Uvicorn 0.34.3, HTTPX 0.28.1, Hugging Face Hub 1.33.0 and pywin32 312. Python handles orchestration; GGML/CUDA handles model inference.

## Testing

From the project folder:

~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\test.ps1
~~~

The validated suite passes **nine tests**: queue serialization, retries/seeds, cancellation state, interrupted-job recovery, bounds, authentication, origins, artifact retrieval, real stdio/HTTP MCP sessions, filtered Windows environments and adapter-disconnect persistence. Model-less tests use temporary directories and do not generate music. The NVML regression probe skips without an available NVIDIA inventory.

One upstream Starlette/AnyIO deprecation warning remains; it does not fail the suite.

For GPU benchmarks, stop the normal service first:

~~~powershell
.\.venv\Scripts\python.exe .\scripts\benchmark.py --quant Q8_0 --duration 30 --keep-loaded --repeats 2 --name local-q8-30
.\.venv\Scripts\python.exe .\scripts\benchmark.py --quant Q6_K --duration 30 --keep-loaded --repeats 2 --name local-q6-30
~~~

Use a new `--name` for each experiment; it selects a report under docs/. The benchmark records input, paths, elapsed time and sampled memory. The first output receives nonempty/finite 48 kHz stereo and RMS checks. The published matched Q8 warm output was also compared with the first by hash.

To exercise generation, score retrieval and active cancellation through stdio:

~~~powershell
.\.venv\Scripts\python.exe .\scripts\validate-live.py
yue-music stop
~~~

This creates music and writes `docs/live-mcp-validation.json`. The [validation report](docs/gpu-validation.md) describes local evidence. Published benchmark JSON has UUIDs redacted and records local artifact paths; generated audio and models are not included.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Command not recognized | Open a new terminal; use `.\bin\yue-music.cmd` meanwhile |
| C++ tools not found | Add VS 2022's C++ workload and CMake tools; verify vswhere or the standard Community path |
| CUDA/CMake build fails | Confirm default-path CUDA 12.8 and the VS 2022 compiler/SDK; inspect the native build error |
| `inference.ready` is false | Inspect `missing_files`, build native executables and run setup-models |
| Interrupted download | Rerun setup-models to reuse completed verified ranges |
| Hugging Face 401/403 | Check access requirements and the local hf auth login account |
| Unavailable GPU UUID | Inspect doctor/nvidia-smi and correct configuration while stopped |
| Insufficient memory | Stop competing work on the 3060; select Q6 or staged mode explicitly |
| Context/prefix error | Shorten lyrics/ABC, lower duration or restore 16,384 context; inspect the native log |
| Profile change rejected | Wait for `service_running: false` after shutdown |
| Port occupied | Stop the relevant local service or select distinct unused ports while stopped |
| MCP tools missing | Register the harness, check absolute paths and restart it |
| Abrupt ending | Check `hit_duration_ceiling`; allow more time or shorten lyrics |
| Native inference fails | Inspect `data/native.log` and `data/service.log`; the next request can restart an unloaded failed worker |

## Updating and removing the installation

Stop before updating. Preserve local changes, pull the desired revision and rerun setup. Retain the pinned engine, submodule and weights unless deliberately validating an upgrade. Verified weights are reused.

~~~powershell
yue-music stop
# After shutdown, from a clean checkout:
git pull --ff-only
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
yue-music doctor
~~~

To move an installation, stop it, preserve wanted files, move the folder, refresh the editable environment/launcher and update MCP absolute paths. Registration rejects a conflicting old entry; remove or update it deliberately before re-registering.

To remove it, stop the worker, remove only local-yue from each registered harness, remove this project's bin directory from user PATH, and remove the folder after preserving wanted outputs. There is no automatic uninstall command, scheduled task or Windows service to unregister.

## Attribution and model licenses

- [YuE2 by Multimodal Art Projection and collaborators](https://github.com/multimodal-art-projection/YuE): original model and reference implementation. Weights use **CC BY-NC 4.0**; current first-party reference code uses Apache 2.0.
- [ServeurpersoCom/yue2.cpp](https://github.com/ServeurpersoCom/yue2.cpp): native GGML engine, MIT-licensed code, with its dependencies' own notices.
- [Serveurperso/YuE2-GGUF](https://huggingface.co/Serveurperso/YuE2-GGUF): converted weights under the model terms.
- [FLUX3-MCP](https://github.com/abanerjee84/FLUX3-MCP): adapted queue, client, installation, downloader and telemetry architecture.

Runtime licensing does not grant commercial rights to YuE2 weights. The downloader retains the publisher's license beside them. See [third-party notices](THIRD_PARTY_NOTICES.md) and [model research](docs/model-research.md).
