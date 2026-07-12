import json
import math
import random
from pathlib import Path

import torch
from temporalio import activity

from model import IrisNetwork


ARTIFACT_DIR = Path("model/artifacts")
MODEL_PATH = ARTIFACT_DIR / "iris-network.pt"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
CLASSES = ["Iris setosa", "Iris versicolor", "Iris virginica"]
MEANS = [
    [5.006, 3.428, 1.462, 0.246],
    [5.936, 2.770, 4.260, 1.326],
    [6.588, 2.974, 5.552, 2.026],
]
DEVIATIONS = [
    [0.352, 0.379, 0.174, 0.105],
    [0.516, 0.314, 0.470, 0.198],
    [0.636, 0.322, 0.552, 0.275],
]


def dataset(seed: int = 21):
    random.seed(seed)
    rows = []
    labels = []
    for label, (means, deviations) in enumerate(zip(MEANS, DEVIATIONS)):
        for _ in range(50):
            rows.append([
                random.gauss(mean, deviation)
                for mean, deviation in zip(means, deviations)
            ])
            labels.append(label)
    order = list(range(len(rows)))
    random.shuffle(order)
    features = torch.tensor([rows[index] for index in order], dtype=torch.float32)
    targets = torch.tensor([labels[index] for index in order], dtype=torch.long)
    return features, targets


@activity.defn(name="train_model_batch")
async def train_model_batch(batch: int) -> dict:
    checkpoint_path = ARTIFACT_DIR / f"{activity.info().workflow_id}-checkpoint.pt"
    torch.manual_seed(21)
    features, targets = dataset()
    train_features, test_features = features[:120], features[120:]
    train_targets, test_targets = targets[:120], targets[120:]
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0)
    network = IrisNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.025)
    if batch > 0:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        network.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loss_function = torch.nn.CrossEntropyLoss()
    final_loss = 0.0
    for epoch in range(13):
        optimizer.zero_grad()
        output = network((train_features - mean) / std)
        loss = loss_function(output, train_targets)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
    activity.heartbeat({"batch": batch + 1, "loss": round(final_loss, 6)})
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)
    with torch.no_grad():
        predictions = network((test_features - mean) / std).argmax(dim=1)
        accuracy = (predictions == test_targets).float().mean().item()
    if batch == 19:
        torch.save({
            "state_dict": network.state_dict(),
            "mean": mean,
            "std": std,
            "classes": CLASSES,
        }, MODEL_PATH)
    metrics = {
        "accuracy": round(accuracy, 4),
        "loss": round(final_loss, 6),
        "epochs": 260,
        "batches": 20,
        "completed_batches": batch + 1,
        "training_samples": 120,
        "test_samples": 30,
        "parameters": sum(math.prod(parameter.shape) for parameter in network.parameters()),
    }
    if batch == 19:
        METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
        checkpoint_path.unlink(missing_ok=True)
    return metrics
