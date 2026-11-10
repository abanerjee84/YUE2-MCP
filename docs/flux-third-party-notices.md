# Reference notices from FLUX3-MCP

This is a reference copy of [FLUX3-MCP's notices](https://github.com/abanerjee84/FLUX3-MCP/blob/main/THIRD_PARTY_NOTICES.md). Paths and image-model dependencies below belong to that project. YUE2-MCP does not contain its ConvRot loader, image model weights or image inference dependencies. See the [YUE2-MCP notices](../THIRD_PARTY_NOTICES.md) for this repository's components.

## Original reference text

`src/flux_image/_convrot.py` contains the INT8 transformer loader from
YUZUKI / animede's flux3-klein-studio, commit
`4d2e68b9ed605d9de33e6fb80502c31cc3c719f7`:
https://github.com/animede/flux3-klein-studio/blob/4d2e68b9ed605d9de33e6fb80502c31cc3c719f7/flux3_int8.py

Copyright 2026 YUZUKI. Licensed under Apache License 2.0;
the full license is included at `vendor/LICENSE.animede`.
Only the transformer-loader portion is used. The upstream CPU-offloaded pipeline
and prompt upsampler are not used.
Local changes remove unused imports and require matching lengths in the loader's
parallel tensor-name mapping (`zip(..., strict=True)`).

FLUX 3 Image Klein model weights are separate downloads under their own license:
https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md
The setup command uses existing Hugging Face authentication and does not accept
gated repository terms on the user's behalf.

Diffusers, Transformers, PyTorch, bitsandbytes and comfy-kitchen retain their
respective licenses. comfy-kitchen is a kernel library; the ComfyUI application
is not installed or run by this project.
