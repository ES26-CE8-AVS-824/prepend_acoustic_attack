"""
prepare_vctk_transcripts.py

Extracts ground-truth transcripts from CSTR-Edinburgh/vctk and writes them
to a JSON file in the same format produced by apply_attack.py:

    { "<path>.wav": "<transcript>", ... }

The path keys match what BadayvedatVCTKAudioDataset produces (speaker_id + utterance ID),
so they align with the filenames written by apply_attack.py.

Usage:
    python prepare_vctk_transcripts.py --split test       --output-dir ./vctk_val_original
    python prepare_vctk_transcripts.py --split train      --output-dir ./vctk_train_original
    python prepare_vctk_transcripts.py --split test       --output-dir ./vctk_val_original --limit 200
"""

import json
import os
import sys
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from tqdm import tqdm

from load_multi_audio_files import BadayvedatVCTKAudioDataset, collate_audio_pinned
from src.data.vctk import DATASET_ID, load_vctk_hf_split, normalize_vctk_split

def make_save_filename(path: str) -> str:
    """Mirror the make_save_path() logic from apply_attack.py:
    take the basename and ensure it has a .wav extension."""
    filename = os.path.basename(path)
    if not os.path.splitext(filename)[1]:
        filename += ".wav"
    return filename


def main():
    parser = ArgumentParser(description="Extract CSTR-Edinburgh/vctk ground-truth transcripts to JSON.")
    parser.add_argument(
        "--split", "-p",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Which split to use. validation is kept as a compatibility alias for the repository test split.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Directory where the transcript JSON will be written.",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Number of samples to process (useful for debugging).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    split_name = normalize_vctk_split(args.split)

    print(f"Loading {DATASET_ID}, split='{split_name}' ...", file=sys.stderr)
    hf_dataset = load_vctk_hf_split(split_name, limit=args.limit)

    dataset = BadayvedatVCTKAudioDataset(hf_dataset)

    # Reuse the same DataLoader setup as apply_attack.py for consistency.
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        collate_fn=collate_audio_pinned,
    )

    print(f"Extracting transcripts for {len(dataset)} samples ...", file=sys.stderr)

    transcripts: dict[str, str] = {}
    for _audio_batch, paths, texts in tqdm(dataloader):
        for path, text in zip(paths, texts):
            filename = make_save_filename(path)
            transcripts[filename] = text

    out_path = os.path.join(args.output_dir, "transcriptions_groundtruth.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(transcripts)} transcripts to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()