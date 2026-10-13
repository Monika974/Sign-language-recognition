"""Train and compare sequence models.

Usage:
    python -m signlang.train --model gru
    python -m signlang.train --model all        # trains every model and writes a comparison table
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow import keras

from signlang.augment import augment_sequence
from signlang.config import load_config
from signlang.data import load_dataset, to_array
from signlang.features import N_FEATURES
from signlang.models import BUILDERS, build_model


def make_train_dataset(seqs, y, seq_len, batch_size):
    """Fresh random augmentation every epoch."""
    def gen():
        rng = np.random.default_rng()
        for i in rng.permutation(len(seqs)):
            yield augment_sequence(seqs[i], seq_len, rng), np.int32(y[i])

    spec = (
        tf.TensorSpec((seq_len, N_FEATURES), tf.float32),
        tf.TensorSpec((), tf.int32),
    )
    return tf.data.Dataset.from_generator(gen, output_signature=spec).batch(batch_size).prefetch(2)


def evaluate(model, X, y, classes):
    probs = model.predict(X, verbose=0)
    order = np.argsort(-probs, axis=1)
    top1 = float(np.mean(order[:, 0] == y))
    k = min(5, len(classes))
    top5 = float(np.mean([y[i] in order[i, :k] for i in range(len(y))]))
    cm = confusion_matrix(y, order[:, 0], labels=range(len(classes)))

    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = [
        {"true": classes[i], "predicted": classes[j], "count": int(off[i, j])}
        for i, j in zip(*np.unravel_index(np.argsort(-off, axis=None)[:10], off.shape))
        if off[i, j] > 0
    ]
    return {"top1": top1, "top5": top5, "most_confused": pairs}, cm


def plot_confusion(cm, classes, path):
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = 6 if len(classes) <= 30 else 9
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if len(classes) <= 30:
        ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
        ax.set_yticks(range(len(classes)), classes, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_one(cfg, name, data, classes):
    t = cfg["train"]
    seq_len = cfg["seq_len"]
    keras.utils.set_random_seed(t["seed"])

    X_tr, y_tr = data["train"]
    X_va = to_array(data["val"][0], seq_len) if len(data["val"][1]) else None
    y_va = data["val"][1]
    X_te = to_array(data["test"][0], seq_len)
    y_te = data["test"][1]

    model = build_model(name, seq_len, N_FEATURES, len(classes), t["dropout"])
    model.compile(
        optimizer=keras.optimizers.Adam(t["lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    out = Path(cfg["paths"]["outputs"]) / name
    out.mkdir(parents=True, exist_ok=True)
    monitor = "val_loss" if X_va is not None else "loss"
    callbacks = [
        keras.callbacks.ModelCheckpoint(out / "best.keras", monitor=monitor, save_best_only=True),
        keras.callbacks.EarlyStopping(monitor=monitor, patience=t["patience"], restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=max(3, t["patience"] // 3)),
    ]
    model.fit(
        make_train_dataset(X_tr, data["train"][1], seq_len, t["batch_size"]),
        validation_data=(X_va, y_va) if X_va is not None else None,
        epochs=t["epochs"],
        callbacks=callbacks,
        verbose=2,
    )

    metrics, cm = evaluate(model, X_te, y_te, classes)
    metrics.update(model=name, n_classes=len(classes), n_test=int(len(y_te)), params=int(model.count_params()))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "meta.json").write_text(json.dumps({"seq_len": seq_len, "n_features": N_FEATURES, "classes": classes, "model": name}, indent=2))
    plot_confusion(cm, classes, out / "confusion_matrix.png")
    print(f"[{name}] top-1 {metrics['top1']:.3f} | top-5 {metrics['top5']:.3f} | params {metrics['params']:,}")
    return metrics


def write_comparison(cfg, results):
    lines = ["| Model | Top-1 | Top-5 | Parameters |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['top1']:.3f} | {r['top5']:.3f} | {r['params']:,} |")
    path = Path(cfg["paths"]["outputs"]) / "model_comparison.md"
    path.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="gru", choices=[*BUILDERS, "all"])
    args = parser.parse_args()
    cfg = load_config(args.config)

    data, classes = load_dataset(cfg["paths"]["landmarks_csv"], cfg["train"]["seed"])
    print(f"Classes: {len(classes)} | train {len(data['train'][1])} | val {len(data['val'][1])} | test {len(data['test'][1])}")
    if len(data["test"][1]) == 0:
        raise SystemExit("No test samples. Add more samples per class or provide a split column.")

    names = list(BUILDERS) if args.model == "all" else [args.model]
    results = [train_one(cfg, n, data, classes) for n in names]
    if len(results) > 1:
        write_comparison(cfg, results)


if __name__ == "__main__":
    main()
