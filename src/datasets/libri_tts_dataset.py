import json
import logging
import os
import random
from pathlib import Path
from typing import List, Optional

import soundfile
import torch
import torchaudio
from torch.utils.data import Dataset

from src.transforms.audio_transforms import AudioToMelSpectrogram

logger = logging.getLogger(__name__)


class LibriTTSDataset(Dataset):
    """
    Multi-speaker dataset for LibriTTS / LibriTTS-R, laid out as
    root_dir/{subset}/{speaker_id}/{chapter_id}/{speaker}_{chapter}_{utt}_{seg}.wav

    There is no single metadata.csv like in LJSpeech: the index is built by
    scanning wav files directly (the directory name right under the subset
    folder gives the speaker id). Since a full scan of train-clean-360 /
    train-other-500 touches hundreds of thousands of files, the resulting
    index is cached to disk after the first run.

    The vocoder itself is not speaker-conditioned, so the returned sample
    shape ("audio", "mel", "audio_path") is identical to LJSpeechDataset and
    needs no changes in collate_fn / trainer / model. LibriTTS audio is
    natively 24kHz; it is resampled to `sample_rate` (22050 by default, to
    match the rest of the pipeline's mel/model config) on load.
    """

    def __init__(
            self,
            root_dir: str,
            subsets: List[str] = ("train-clean-100",),
            segment_size: int = 16384,
            hop_length: int = 256,
            sample_rate: int = 22050,
            n_mels: int = 80,
            n_fft: int = 1024,
            f_min: float = 0.0,
            f_max: float = 8000.0,
            min_duration_sec: float = 0.5,
            max_duration_sec: Optional[float] = None,
            speakers: Optional[List[str]] = None,
            exclude_speakers: Optional[List[str]] = None,
            max_files_per_speaker: Optional[int] = None,
            limit: int = None,
            offset: int = 0,
            shuffle_index: bool = False,
            instance_transforms: dict = None,
            use_index_cache: bool = True,
            index_cache_path: str = None,
            name: str = "libritts",
    ):
        self.root_dir = root_dir
        self.subsets = list(subsets)
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

        index = self._load_or_build_index(
            use_index_cache, index_cache_path, min_duration_sec, max_duration_sec,
        )

        allow_set = set(speakers) if speakers else None
        exclude_set = set(exclude_speakers) if exclude_speakers else set()
        index = [
            entry for entry in index
            if self._speaker_allowed(entry["speaker_id"], allow_set, exclude_set)
        ]

        if max_files_per_speaker is not None:
            index = self._cap_per_speaker(index, max_files_per_speaker)

        self._assert_index_is_valid(index)
        index = self._apply_offset_and_limit(index, offset, limit)
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index: List[dict] = index

        n_speakers = len({entry["speaker_id"] for entry in self._index})
        logger.info(
            f"LibriTTSDataset[{name}]: {len(self._index)} utterances, "
            f"{n_speakers} speakers, subsets={self.subsets}"
        )
        print(len(self._index))

        self.instance_transforms = instance_transforms

    def __getitem__(self, idx):
        data_dict = self._index[idx]
        audio_path = data_dict["audio_path"]

        waveform, sr = self.load_audio(audio_path)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        if self.segment_size is not None:
            waveform = self._segment_audio(waveform)

        mel = self.mel_transform(waveform)

        if mel.dim() == 3 and mel.shape[0] == 1:
            mel = mel.squeeze(0)  # [n_mels, T]

        data = {
            "audio": waveform,
            "mel": mel,  # [n_mels, T]
            "audio_path": audio_path,
        }

        return data

    def __len__(self):
        return len(self._index)

    def load_audio(self, audio_path):
        waveform, sample_rate = soundfile.read(audio_path)
        waveform = torch.from_numpy(waveform)
        waveform = waveform.to(torch.float32)
        if waveform.dim() == 2:
            waveform = waveform.transpose(0, 1)  # [T, C] -> [C, T]

        return waveform, sample_rate

    def _segment_audio(self, audio):
        segment_size = (self.segment_size // self.hop_length) * self.hop_length

        if audio.shape[-1] >= segment_size:
            max_start = audio.shape[-1] - segment_size
            start = random.randint(0, max_start)
            return audio[..., start:start + segment_size]
        else:
            pad_size = segment_size - audio.shape[-1]
            return torch.nn.functional.pad(audio, (0, pad_size), mode='constant', value=0)

    def _load_or_build_index(self, use_cache, cache_path, min_duration_sec, max_duration_sec):
        cache_path = cache_path or os.path.join(
            self.root_dir, ".cache", f"libritts_index_{'_'.join(self.subsets)}.json"
        )

        if use_cache and os.path.exists(cache_path):
            logger.info(f"Loading LibriTTS index from cache: {cache_path}")
            with open(cache_path, "r") as f:
                return json.load(f)

        index = self._scan_subsets(min_duration_sec, max_duration_sec)

        if use_cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(index, f)
            logger.info(f"Cached LibriTTS index ({len(index)} files) to {cache_path}")

        return index

    def _scan_subsets(self, min_duration_sec, max_duration_sec):
        index = []
        for subset in self.subsets:
            subset_dir = os.path.join(self.root_dir, subset)
            if not os.path.isdir(subset_dir):
                logger.warning(f"LibriTTS subset not found: {subset_dir}")
                continue

            for wav_path in Path(subset_dir).rglob("*.wav"):
                speaker_id = wav_path.relative_to(subset_dir).parts[0]
                try:
                    info = soundfile.info(str(wav_path))
                except Exception as e:
                    logger.warning(f"Skipping unreadable file {wav_path}: {e}")
                    continue

                duration = info.frames / info.samplerate
                if duration < min_duration_sec:
                    continue
                if max_duration_sec is not None and duration > max_duration_sec:
                    continue

                index.append({
                    "audio_path": str(wav_path),
                    "speaker_id": speaker_id,
                    "id": wav_path.stem,
                    "num_frames": info.frames,
                    "orig_sample_rate": info.samplerate,
                })

        return index

    @staticmethod
    def _speaker_allowed(speaker_id, allow_set, exclude_set):
        if speaker_id in exclude_set:
            return False
        if allow_set is not None and speaker_id not in allow_set:
            return False
        return True

    @staticmethod
    def _cap_per_speaker(index, max_files_per_speaker):
        by_speaker = {}
        for entry in index:
            by_speaker.setdefault(entry["speaker_id"], []).append(entry)

        rng = random.Random(42)
        capped = []
        for entries in by_speaker.values():
            rng.shuffle(entries)
            capped.extend(entries[:max_files_per_speaker])
        return capped

    def _apply_offset_and_limit(self, index, offset, limit):
        if offset > 0:
            index = index[offset:]
        if limit is not None:
            index = index[:limit]
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "audio_path" in entry, "Missing 'audio_path' in dataset entry"
            assert "speaker_id" in entry, "Missing 'speaker_id' in dataset entry"

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
