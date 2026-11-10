from __future__ import annotations

import json
import ssl

from .config import Settings

REPO = "Serveurperso/YuE2-GGUF"
REVISION = "e630f2b8f6aedf8254c3186438633c1b5eeaf80a"
FILES = {
    "YuE2-3B-Q8_0.gguf": (
        3810232064,
        "41121ce97786d7795a325bcf123ca196956c03bb252c9e75384cfc1f2fc19e6b",
    ),
    "YuE2-3B-Q6_K.gguf": (
        2943331072,
        "42068d5e57713df11b9f6a5a3751072d9bbefe231d1d088f9c100e026ffa4ef9",
    ),
    "YuE2-Vae-F32.gguf": (
        530497344,
        "93e49dfb1970e89ad64cacb17cf13b5d05f6bb30ef7ed3adae3050bcb728638a",
    ),
}


def setup_models(settings: Settings, both=False):
    import httpx
    import truststore
    from huggingface_hub import hf_hub_download, set_client_factory

    from .download import file_sha256, verified_download

    set_client_factory(
        lambda: httpx.Client(
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            timeout=httpx.Timeout(60, read=300),
        )
    )
    names = list(FILES) if both else [settings.model_file, "YuE2-Vae-F32.gguf"]
    for name in names:
        path = verified_download(REPO, name, REVISION, settings.models / name)
        size, digest = FILES[name]
        if path.stat().st_size != size or file_sha256(path) != digest:
            raise RuntimeError(f"Pinned file verification failed: {name}")
    hf_hub_download(REPO, "LICENSE", revision=REVISION, local_dir=settings.models)
    manifest = {
        "repo": REPO,
        "revision": REVISION,
        "files": {n: FILES[n] for n in names},
    }
    (settings.models / "manifest.json").write_text(
        json.dumps(manifest, indent=2), "utf-8"
    )
    return manifest
