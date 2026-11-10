from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MusicRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    style: str = Field(min_length=1, max_length=2000)
    lyrics: str = Field(default="", max_length=12000)
    abc: str = Field(default="", max_length=20000)
    cot: Literal["full", "melody", "off"] = "full"
    duration: float = Field(default=60, ge=10, le=300, allow_inf_nan=False)
    steps: int = Field(default=8, ge=4, le=32, strict=True)
    seed: int | None = Field(
        default=None,
        ge=-1,
        le=2**32 - 1,
        strict=True,
        description="Use -1, null or omit for a random acoustic seed.",
    )
    lm_seed: int | None = Field(
        default=None,
        ge=-1,
        le=2**32 - 1,
        strict=True,
        description="Null or omitted inherits seed; -1 selects an independent random token seed.",
    )
    output_format: Literal["wav16", "mp3"] = "wav16"
    client_request_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def check_score(self):
        if self.abc and self.cot == "off":
            raise ValueError("An ABC score requires full or melody planning mode")
        return self
