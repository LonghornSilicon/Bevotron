"""Known datasets. Add an entry here; nothing else needs to change."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    hf_name: str  # Hugging Face datasets id
    train_split: str
    val_split: str
    text_field: str


DATASETS: dict[str, DatasetSpec] = {
    "tinystories": DatasetSpec("roneneldan/TinyStories", "train", "validation", "text"),
}
