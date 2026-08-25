import hashlib
import logging
import random
from pathlib import Path
from typing import List, Optional, Sequence

import soundfile
import torch
import torchaudio
from torch.utils.data import Dataset

from src.transforms.audio_transforms import AudioToMelSpectrogram

logger = logging.getLogger(__name__)

DEFAULT_AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".opus", ".mp3")

PARTS = ("train", "val", "test", "all")

_SCAN_MEMO: dict = {}


class LibriTTSDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            subsets: Sequence[str] = ("train-clean-100",),
            part: str = "train",
            val_ratio: float = 0.1,
            test_ratio: float = 0.1,
            split_seed: int = 42,
            segment_size: Optional[int] = 16384,
            hop_length: int = 256,
            sample_rate: int = 22050,
            n_mels: int = 80,
            n_fft: int = 1024,
            f_min: float = 0.0,
            f_max: float = 8000.0,
            min_duration_sec: float = 0.5,
            max_duration_sec: Optional[float] = None,
            speakers: Optional[Sequence[str]] = None,
            exclude_speakers: Optional[Sequence[str]] = None,
            max_files_per_speaker: Optional[int] = None,
            audio_extensions: Optional[Sequence[str]] = None,
            limit: Optional[int] = None,
            shuffle_index: bool = False,
            name: str = "libritts",
    ):
        if part not in PARTS:
            raise ValueError(f"Unknown part '{part}', expected one of {PARTS}")
        if not 0.0 <= val_ratio + test_ratio < 1.0:
            raise ValueError(
                f"val_ratio + test_ratio must be in [0, 1), "
                f"got {val_ratio} + {test_ratio}"
            )

        self.root_dir = str(root_dir)
        self.subsets = list(subsets)
        self.part = part
        self.segment_size = segment_size
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.name = name

        self.mel_transform = AudioToMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )

        extensions = tuple(
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in (audio_extensions or DEFAULT_AUDIO_EXTENSIONS)
        )

        index = self._scan(
            root_dir=self.root_dir,
            subsets=self.subsets,
            extensions=extensions,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
        )
        scanned_total = len(index)

        index = self._filter_speakers(index, speakers, exclude_speakers)
        if max_files_per_speaker is not None:
            index = self._cap_per_speaker(index, max_files_per_speaker, split_seed)

        index = self._select_part(index, part, val_ratio, test_ratio, split_seed)

        if shuffle_index:
            random.Random(split_seed).shuffle(index)
        if limit is not None:
            index = index[:limit]

        self._index: List[dict] = index

        if len(self._index) == 0:
            raise RuntimeError(
                f"LibriTTSDataset[{name}] partition '{part}' is empty "
                f"({scanned_total} audio files matched the scan of "
                f"{self.root_dir} / {self.subsets}). Check root_dir, subsets, "
                f"the speaker filters and the duration filters "
                f"(min_duration_sec={min_duration_sec}, "
                f"max_duration_sec={max_duration_sec})."
            )

        n_speakers = len({entry["speaker_id"] for entry in self._index})
        logger.info(
            f"LibriTTSDataset[{name}] part='{part}': {len(self._index)} utterances, "
            f"{n_speakers} speakers, subsets={self.subsets}"
        )

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        data_dict = self._index[idx]
        audio_path = data_dict["audio_path"]

        waveform, sr = self.load_audio(audio_path)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        if self.segment_size is not None:
            waveform = self._segment_audio(waveform)

        mel = self.mel_transform(waveform)
        if mel.dim() == 3 and mel.shape[0] == 1:
            mel = mel.squeeze(0)  # [n_mels, T]

        return {
            "audio": waveform,  # [1, T]
            "mel": mel,  # [n_mels, T']
            "audio_path": audio_path,
        }

    def load_audio(self, audio_path):
        waveform, sample_rate = soundfile.read(audio_path, dtype="float32")
        waveform = torch.from_numpy(waveform)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [T] -> [1, T]
        else:
            waveform = waveform.transpose(0, 1)  # [T, C] -> [C, T]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform, sample_rate

    def _segment_audio(self, audio):
        segment_size = (self.segment_size // self.hop_length) * self.hop_length

        if audio.shape[-1] >= segment_size:
            start = random.randint(0, audio.shape[-1] - segment_size)
            return audio[..., start:start + segment_size]

        pad_size = segment_size - audio.shape[-1]
        return torch.nn.functional.pad(audio, (0, pad_size), mode="constant", value=0)

    @classmethod
    def _scan(cls, root_dir, subsets, extensions, min_duration_sec, max_duration_sec):
        key = (root_dir, tuple(subsets), extensions, min_duration_sec, max_duration_sec)
        if key in _SCAN_MEMO:
            return list(_SCAN_MEMO[key])

        if not Path(root_dir).is_dir():
            raise FileNotFoundError(f"LibriTTS root_dir does not exist: {root_dir}")

        index = []
        for subset in subsets:
            subset_dir = Path(root_dir) / subset
            if not subset_dir.is_dir():
                raise FileNotFoundError(
                    f"LibriTTS subset not found: {subset_dir}. Available entries in "
                    f"{root_dir}: {sorted(p.name for p in Path(root_dir).iterdir())}"
                )

            audio_paths = sorted(
                path for path in subset_dir.rglob("*")
                if path.suffix.lower() in extensions and path.is_file()
            )
            if not audio_paths:
                raise FileNotFoundError(
                    f"No audio files with extensions {extensions} found under "
                    f"{subset_dir}"
                )

            kept = 0
            for audio_path in audio_paths:
                try:
                    info = soundfile.info(str(audio_path))
                except Exception as e:  # unreadable / truncated file
                    logger.warning(f"Skipping unreadable file {audio_path}: {e}")
                    continue

                duration = info.frames / info.samplerate
                if duration < min_duration_sec:
                    continue
                if max_duration_sec is not None and duration > max_duration_sec:
                    continue

                index.append({
                    "audio_path": str(audio_path),
                    "speaker_id": audio_path.relative_to(subset_dir).parts[0],
                    "id": audio_path.stem,
                    "num_frames": info.frames,
                    "orig_sample_rate": info.samplerate,
                })
                kept += 1

            logger.info(
                f"Scanned {subset_dir}: {kept}/{len(audio_paths)} files kept "
                f"after duration filtering"
            )

        _SCAN_MEMO[key] = list(index)
        return index

    @staticmethod
    def _filter_speakers(index, speakers, exclude_speakers):
        allow_set = {str(s) for s in speakers} if speakers else None
        exclude_set = {str(s) for s in exclude_speakers} if exclude_speakers else set()

        if allow_set is None and not exclude_set:
            return index

        return [
            entry for entry in index
            if entry["speaker_id"] not in exclude_set
            and (allow_set is None or entry["speaker_id"] in allow_set)
        ]

    @staticmethod
    def _cap_per_speaker(index, max_files_per_speaker, seed):
        by_speaker = {}
        for entry in index:
            by_speaker.setdefault(entry["speaker_id"], []).append(entry)

        rng = random.Random(seed)
        capped = []
        for speaker_id in sorted(by_speaker):
            entries = list(by_speaker[speaker_id])
            rng.shuffle(entries)
            capped.extend(entries[:max_files_per_speaker])
        return capped

    @staticmethod
    def _bucket(utterance_id, seed):
        """Map an utterance id to a stable float in [0, 1)."""
        digest = hashlib.md5(f"{seed}:{utterance_id}".encode()).hexdigest()
        return int(digest[:8], 16) / 2 ** 32

    @classmethod
    def _select_part(cls, index, part, val_ratio, test_ratio, split_seed):
        if part == "all":
            return list(index)

        selected = []
        for entry in index:
            bucket = cls._bucket(entry["id"], split_seed)
            if bucket < test_ratio:
                entry_part = "test"
            elif bucket < test_ratio + val_ratio:
                entry_part = "val"
            else:
                entry_part = "train"

            if entry_part == part:
                selected.append(entry)
        return selected
