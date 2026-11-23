"""Five-fold ResNet50 baseline; run --smoke before the full experiment."""

import os

# This must be set before torch initializes its MPS backend.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import csv
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import tifffile
import torch
import yaml
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
RUNS = ROOT / "runs"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def read_mapping():
    with (DATA / "test" / "class_id_mapping.txt").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["class_id", "original_class"]:
            raise ValueError("Unexpected class mapping header")
        pairs = [(row["original_class"], int(row["class_id"])) for row in reader]
    mapping = dict(pairs)
    if len(mapping) != len(pairs) or sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("Class IDs must be unique, contiguous, and start at zero")
    return mapping


def source_records(mapping):
    root = DATA / "data" / "12class_tif"
    directories = {p.name for p in root.iterdir() if p.is_dir()}
    if directories != set(mapping):
        raise ValueError(f"Training directories do not match class mapping: {directories ^ set(mapping)}")
    records = []
    for name in sorted(mapping):
        for path in sorted((root / name).glob("*.tif")):
            records.append({"relative_path": path.relative_to(DATA).as_posix(),
                            "class_name": name, "class_id": mapping[name]})
    if not records:
        raise ValueError("Training set is empty")
    return sorted(records, key=lambda row: row["relative_path"])


def read_folds(records, config):
    path = RUNS / "folds.csv"
    fields = ["relative_path", "class_name", "class_id", "fold"]
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != fields:
                raise ValueError("Existing runs/folds.csv has incompatible columns")
            rows = list(reader)
        try:
            matches = all(
                (row["relative_path"], row["class_name"], int(row["class_id"]))
                == (record["relative_path"], record["class_name"], record["class_id"])
                for row, record in zip(rows, records)
            )
            folds = [int(row["fold"]) for row in rows]
        except (KeyError, ValueError) as exc:
            raise ValueError("Invalid existing fold file") from exc
        if len(rows) != len(records) or not matches or set(folds) != set(range(1, config["folds"] + 1)):
            raise ValueError("Existing folds differ from current data or fold count; refusing to overwrite")
        counts = Counter((record["class_id"], fold) for record, fold in zip(records, folds))
        for class_id in {record["class_id"] for record in records}:
            values = [counts[class_id, fold] for fold in range(1, config["folds"] + 1)]
            if min(values) == 0 or max(values) - min(values) > 1:
                raise ValueError("Existing folds are not stratified")
        print(f"Reusing {path}", flush=True)
        return folds

    labels = [record["class_id"] for record in records]
    splitter = StratifiedKFold(n_splits=config["folds"], shuffle=True, random_state=config["seed"])
    folds = [0] * len(records)
    for fold, (_, valid_indices) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        for index in valid_indices:
            folds[index] = fold
    RUNS.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**record, "fold": fold} for record, fold in zip(records, folds))
    print(f"Saved {path}", flush=True)
    return folds


class RightAngleRotation:
    def __call__(self, image):
        return image.rotate(random.choice((0, 90, 180, 270)))


def image_transform(size, training):
    operations = [transforms.Resize((size, size))]
    if training:
        operations += [transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
                       transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                       RightAngleRotation(),
                       transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                              saturation=0.1, hue=0.02)]
    operations += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(operations)


class SceneDataset(Dataset):
    def __init__(self, records, size, training=False):
        self.records = records
        self.transform = image_transform(size, training)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        array = tifffile.imread(DATA / row["relative_path"])
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError(f"Expected three uint8 channels: {row['relative_path']}")
        image = self.transform(Image.fromarray(array, mode="RGB"))
        return image, row.get("class_id", -1)


def make_loader(records, config, training=False):
    return DataLoader(SceneDataset(records, config["image_size"], training),
                      batch_size=config["batch_size"], shuffle=training,
                      num_workers=config["num_workers"], pin_memory=False)


def sync_mps():
    torch.mps.synchronize()


def predict(model, loader, device):
    model.eval()
    chunks = []
    labels = []
    with torch.inference_mode():
        for images, targets in loader:
            probabilities = model(images.to(device)).softmax(dim=1)
            chunks.append(probabilities.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(chunks), np.concatenate(labels)


def submission_rows(mapping):
    sample = DATA / "sample_submission.csv"
    with sample.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    if len(header) != 2 or header[0] != "image_id" or len(rows) != 240:
        raise ValueError("Unexpected sample submission format")
    names = [row[0] for row in rows]
    images = {path.name for path in (DATA / "test" / "12class_tif").glob("*.tif")}
    if any(len(row) != 2 for row in rows) or len(set(names)) != 240 or set(names) != images:
        raise ValueError("Sample submission names do not match test images")
    return header, names, [{"relative_path": f"test/12class_tif/{name}"} for name in names]


def write_submission(header, names, probs, mapping):
    if probs.shape != (240, len(mapping)) or not np.isfinite(probs).all():
        raise ValueError("Incomplete test probabilities")
    labels = probs.argmax(axis=1).astype(int).tolist()
    path = ROOT / "submission.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(zip(names, labels))
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        written_header, written = next(reader), list(reader)
    if (written_header != header or len(written) != 240
            or [row[0] for row in written] != names
            or set(names) != {p.name for p in (DATA / "test" / "12class_tif").glob("*.tif")}
            or any(len(row) != 2 or not row[1] or int(row[1]) not in mapping.values() for row in written)):
        raise ValueError("Submission self-check failed")
    print("Submission self-check OK: 240 images, exact sample header and order, valid integer IDs")
    print("First five rows:")
    for row in [written_header, *written[:5]]:
        print(row)


def append_result(row):
    path = RUNS / "results.csv"
    fields = ["experiment", "fold", "epochs", "final_val_accuracy", "best_epoch",
              "best_val_accuracy", "duration_seconds"]
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def train_fold(fold, records, assignments, config, device, output, smoke, test_records):
    train = [row for row, assigned in zip(records, assignments) if assigned != fold]
    valid = [row for row, assigned in zip(records, assignments) if assigned == fold]
    train_loader = make_loader(train, config, training=True)
    valid_loader = make_loader(valid, config)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    model = timm.create_model(config["model"], pretrained=True, num_classes=len(read_mapping())).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    steps_per_epoch = len(train_loader)
    warmup_steps = config["warmup_epochs"] * steps_per_epoch
    total_steps = config["epochs"] * steps_per_epoch

    def lr_scale(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    criterion = torch.nn.CrossEntropyLoss()
    output.mkdir(parents=True, exist_ok=True)
    epochs = 1 if smoke else config["epochs"]
    best_accuracy, best_epoch = -1.0, 0
    sync_mps()
    fold_start = time.perf_counter()
    with (output / "epochs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_accuracy", "epoch_seconds", "learning_rate"])
        for epoch in range(1, epochs + 1):
            sync_mps()
            started = time.perf_counter()
            model.train()
            loss_sum = 0.0
            seen = 0
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), targets)
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_sum += loss.item() * len(targets)
                seen += len(targets)
            probabilities, truths = predict(model, valid_loader, device)
            accuracy = float((probabilities.argmax(axis=1) == truths).mean())
            sync_mps()
            seconds = time.perf_counter() - started
            if accuracy > best_accuracy:
                best_accuracy, best_epoch = accuracy, epoch
            writer.writerow([epoch, loss_sum / seen, accuracy, seconds, optimizer.param_groups[0]["lr"]])
            handle.flush()
            print(f"fold={fold} epoch={epoch}/{epochs} loss={loss_sum / seen:.4f} "
                  f"val_acc={accuracy:.4f} time={seconds:.1f}s MPS={torch.backends.mps.is_available()}", flush=True)
            if not smoke and fold == 1 and epoch == 3 and accuracy < 0.50:
                raise RuntimeError(
                    f"Stopping after fold 1 epoch 3: validation accuracy {accuracy:.4f} "
                    "is below the required 0.50; no later folds will run"
                )
    sync_mps()
    duration = time.perf_counter() - fold_start
    if smoke:
        print(f"SMOKE OK: fold={fold}, epoch_seconds={seconds:.1f}, val_accuracy={accuracy:.4f}, "
              f"MPS={device.type == 'mps'}", flush=True)
        return None

    torch.save({"state_dict": model.state_dict(), "model": config["model"],
                "class_ids": sorted(read_mapping().values()), "epoch": epochs}, output / "final.pt")
    np.save(output / "oof_probs.npy", probabilities.astype(np.float32))
    with (output / "oof_files.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "class_id"])
        writer.writerows((row["relative_path"], row["class_id"]) for row in valid)
    matrix = confusion_matrix(truths, probabilities.argmax(axis=1),
                              labels=list(range(len(read_mapping()))))
    np.savetxt(output / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    test_probs, _ = predict(model, make_loader(test_records, config), device)
    append_result({"experiment": "baseline", "fold": fold, "epochs": epochs,
                   "final_val_accuracy": accuracy, "best_epoch": best_epoch,
                   "best_val_accuracy": best_accuracy, "duration_seconds": round(duration, 2)})
    return valid, probabilities, test_probs, accuracy, best_epoch, best_accuracy, duration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run fold 1 for one epoch only")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs" / "baseline.yaml").read_text(encoding="utf-8"))
    if config["folds"] != 5 or config["epochs"] != 10 or config["warmup_epochs"] != 1:
        raise ValueError("Baseline requires 5 folds, 10 epochs, and 1 warmup epoch")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; refusing to run the requested MPS baseline")
    device = torch.device("mps")
    print(f"device={device} fallback={os.environ['PYTORCH_ENABLE_MPS_FALLBACK']} "
          f"model={config['model']} epochs={config['epochs']} batch={config['batch_size']}", flush=True)
    mapping = read_mapping()
    records = source_records(mapping)
    assignments = read_folds(records, config)
    print(f"fold sizes={dict(sorted(Counter(assignments).items()))}", flush=True)
    if args.smoke:
        train_fold(1, records, assignments, config, device, RUNS / "smoke" / "fold_1", True, [])
        return

    header, names, test_records = submission_rows(mapping)
    output = RUNS / "baseline"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Full-run output already exists: {output}; refusing to overwrite")
    oof = np.full((len(records), len(mapping)), np.nan, dtype=np.float32)
    test_sum = np.zeros((len(names), len(mapping)), dtype=np.float64)
    finals, bests, times = [], [], []
    for fold in range(1, config["folds"] + 1):
        result = train_fold(fold, records, assignments, config, device,
                            output / f"fold_{fold}", False, test_records)
        valid, valid_probs, test_probs, final_acc, _, best_acc, duration = result
        indices = [index for index, assigned in enumerate(assignments) if assigned == fold]
        if [records[index] for index in indices] != valid:
            raise ValueError("OOF fold order mismatch")
        oof[indices] = valid_probs.astype(np.float32)
        test_sum += test_probs
        finals.append(final_acc)
        bests.append(best_acc)
        times.append(duration)
    if not np.isfinite(oof).all():
        raise ValueError("OOF predictions incomplete")
    np.save(output / "oof_probs.npy", oof)
    with (output / "oof_files.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "class_name", "class_id", "fold"])
        writer.writerows((row["relative_path"], row["class_name"],
                          row["class_id"], assignments[index])
                         for index, row in enumerate(records))
    average = (test_sum / config["folds"]).astype(np.float32)
    np.save(output / "test_probs.npy", average)
    for title, values in (("mean", finals), ("std", finals)):
        append_result({"experiment": "baseline", "fold": title, "epochs": config["epochs"],
                       "final_val_accuracy": float(np.mean(values)) if title == "mean" else float(np.std(values)),
                       "best_epoch": "", "best_val_accuracy": float(np.mean(bests)) if title == "mean" else float(np.std(bests)),
                       "duration_seconds": round(float(np.sum(times)), 2) if title == "mean" else ""})
    write_submission(header, names, average, mapping)
    print(f"Final validation accuracy: {np.mean(finals):.4f} ± {np.std(finals):.4f}; "
          f"best-epoch reference: {np.mean(bests):.4f} ± {np.std(bests):.4f}")


if __name__ == "__main__":
    main()