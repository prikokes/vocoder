import torch
import logging

logger = logging.getLogger(__name__)


import torch
import logging

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: list[dict]):
    return {
        "audio": torch.stack([item["audio"] for item in dataset_items]),  # [B, 1, T]
        "mel": torch.stack([item["mel"] for item in dataset_items]),  # [B, 80, T']
        "audio_path": [item["audio_path"] for item in dataset_items],
    }
