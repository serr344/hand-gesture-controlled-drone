from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = ROOT / "dataset" / "hagrid-export_1000_images"
OUTPUT_DIR = ROOT / "dataset" / "raw" / "data"

CLASSES = [
    "fist",
    "like",
    "dislike",
    "one",
    "peace",
    "palm",
    "no_gesture",
]

SPLIT_RATIOS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}

RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def split_dataset():
    rng = random.Random(RANDOM_SEED)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {SOURCE_DIR}"
        )

    print("Starting dataset splitting...\n")

    totals = {split: 0 for split in SPLIT_RATIOS}

    for gesture in CLASSES:
        source_folder = SOURCE_DIR / gesture

        if not source_folder.exists():
            raise FileNotFoundError(
                f"Missing class directory: {source_folder}"
            )

        images = sorted(
            path for path in source_folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        rng.shuffle(images)

        total = len(images)

        train_end = int(total * SPLIT_RATIOS["train"])
        val_end = train_end + int(
            total * SPLIT_RATIOS["val"]
        )

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        print(f"Class: {gesture}")
        print(f"Total images: {total}")

        for split_name, split_images in splits.items():
            target_folder = OUTPUT_DIR / split_name / gesture
            target_folder.mkdir(parents=True, exist_ok=True)

            for image_path in split_images:
                shutil.copy2(
                    image_path,
                    target_folder / image_path.name,
                )

            count = len(split_images)
            totals[split_name] += count

            print(f"  {split_name}: {count}")

        print()

    print("Dataset Split Summary")
    print("---------------------")

    for split_name, count in totals.items():
        print(f"{split_name}: {count}")

    print(f"Total: {sum(totals.values())}")
    print("\nDataset splitting completed!")


if __name__ == "__main__":
    split_dataset()
