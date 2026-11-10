# Third-party notices

This project adapts orchestration modules from the local [FLUX3-MCP](https://github.com/abanerjee84/FLUX3-MCP) project. The original repository's notices are retained at `docs/flux-third-party-notices.md`. No image model weights or image inference code are included.

The native engine is [ServeurpersoCom/yue2.cpp](https://github.com/ServeurpersoCom/yue2.cpp), MIT licensed. Its checkout retains its license and those of GGML, cpp-httplib, yyjson and its audio codec components. Native source and binaries are installation artifacts under the ignored `vendor/` directory.

[YuE2 model weights](https://huggingface.co/m-a-p/YuE2-3B) and [GGUF conversions](https://huggingface.co/Serveurperso/YuE2-GGUF) are separately licensed under CC BY-NC 4.0. The model download keeps the publisher's LICENSE beside the weights. The wrapper's distribution does not grant commercial rights to the model.
