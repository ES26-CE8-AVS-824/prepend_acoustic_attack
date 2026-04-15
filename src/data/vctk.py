import random

from datasets import Audio, load_dataset
from tqdm import tqdm


DATASET_ID = "CSTR-Edinburgh/vctk"
LEGACY_DATASET_IDS = {"badayvedat/VCTK"}
UPSTREAM_TRAIN_SPLIT = "train"
REPO_TEST_SPLIT = "test"
REPO_VALIDATION_SPLIT = "validation"
VCTK_SPLIT_SEED = 42
VCTK_TEST_SPEAKER_FRACTION = 0.1


def _decode(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace").strip()
    return str(value) if value is not None else ""


def is_vctk_dataset(dataset_name: str) -> bool:
    normalized = dataset_name.strip().lower()
    return normalized == DATASET_ID.lower() or normalized in {item.lower() for item in LEGACY_DATASET_IDS}


def normalize_vctk_split(split_name: str) -> str:
    normalized = (split_name or REPO_TEST_SPLIT).strip().lower()
    if normalized == REPO_VALIDATION_SPLIT:
        return REPO_TEST_SPLIT
    if normalized not in {UPSTREAM_TRAIN_SPLIT, REPO_TEST_SPLIT}:
        raise ValueError(f"Unsupported VCTK split: {split_name}")
    return normalized


def _get_test_speakers(data):
    speaker_ids = sorted({_decode(speaker_id) for speaker_id in data["speaker_id"] if _decode(speaker_id)})
    if len(speaker_ids) < 2:
        raise ValueError("VCTK synthetic split requires at least two speakers")

    rng = random.Random(VCTK_SPLIT_SEED)
    rng.shuffle(speaker_ids)

    test_speaker_count = max(1, round(len(speaker_ids) * VCTK_TEST_SPEAKER_FRACTION))
    test_speaker_count = min(test_speaker_count, len(speaker_ids) - 1)
    return set(speaker_ids[:test_speaker_count])


def load_vctk_hf_split(split_name: str, limit: int | None = None):
    normalized_split = normalize_vctk_split(split_name)
    full_train = load_dataset(DATASET_ID, split=UPSTREAM_TRAIN_SPLIT, trust_remote_code=True)
    test_speakers = _get_test_speakers(full_train)

    if normalized_split == UPSTREAM_TRAIN_SPLIT:
        subset = full_train.filter(
            lambda speaker_id: _decode(speaker_id) not in test_speakers,
            input_columns=["speaker_id"],
            desc="Creating VCTK train split",
        )
    else:
        subset = full_train.filter(
            lambda speaker_id: _decode(speaker_id) in test_speakers,
            input_columns=["speaker_id"],
            desc="Creating VCTK test split",
        )

    if limit is not None:
        subset = subset.select(range(min(limit, len(subset))))

    return subset


def _vctk(split_train=UPSTREAM_TRAIN_SPLIT, split_test=REPO_TEST_SPLIT):
    train_data = load_vctk_hf_split(split_train)
    test_data = load_vctk_hf_split(split_test)

    train_data = train_data.cast_column("audio", Audio(decode=False))
    test_data = test_data.cast_column("audio", Audio(decode=False))

    return _prep_samples(train_data), _prep_samples(test_data)


def _prep_samples(data):
    samples = []
    for sample in tqdm(data, total=len(data), desc="Processing VCTK samples"):
        speaker_id = _decode(sample.get("speaker_id"))
        utt_id = str(sample.get("text_id", ""))
        audio_info = sample["audio"]
        audio = audio_info.get("path") or sample.get("file") or audio_info

        samples.append(
            {
                "id": f"{speaker_id}_{utt_id}" if speaker_id else utt_id,
                "speaker_id": speaker_id,
                "audio": audio,
                "ref": sample.get("text", "").strip(),
            }
        )
    return samples