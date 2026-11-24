"""Train only the EF-HRL surrogate using nominal frozen-policy transitions."""
import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import torch
from torch.nn import functional as F

from modules import DEFAULT_MODEL, ExplicitFeedback, SNS_MLP, input_scale


def load_transitions(path, limit, device):
    with h5py.File(path, "r") as data:
        features, labels = data["input"], data["output"]
        # Also accept the backup's 10 x 13 records (the last channel was a variance trace).
        if (features.ndim != 3 or features.shape[1] < 5 or features.shape[2] < 12
                or labels.shape != (len(features), 3)):
            raise ValueError("Expected paired input[N, >=5, >=12] and output[N, 3]")
        size = min(len(features), limit)
        if size < 4:
            raise ValueError("At least four transitions are required")
        x = torch.empty(size, 60, device=device)
        y = torch.empty(size, 3, device=device)
        for start in range(0, size, 65536):
            end = min(start + 65536, size)
            x[start:end] = torch.from_numpy(features[start:end, -5:, :12].reshape(-1, 60))
            y[start:end] = torch.from_numpy(labels[start:end])
    if not (x.isfinite().all() and y.isfinite().all()):
        raise ValueError("Dataset contains non-finite observations or targets")
    return x.mul_(input_scale().to(device)), y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("loco_error_dataset_nav_guided.h5"))
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-samples", type=int, default=8_192_000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1:
        parser.error("epochs and batch size must be positive")
    torch.manual_seed(args.seed)
    if torch.device(args.device).type == "cuda":
        torch.set_float32_matmul_precision("high")
    x, y = load_transitions(args.data, args.max_samples, args.device)
    order = torch.randperm(len(x), device=args.device)
    split = 3 * len(x) // 4
    train, val = order[:split], order[split:]
    model = SNS_MLP().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"Training {len(train)} samples, validating {len(val)} samples", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = train[torch.randperm(len(train), device=args.device)]
        for idx in shuffled.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x[idx]), y[idx]) + 0.2 * model.get_lipschitz_penalty(4.0)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            squared_error = x.new_zeros(3)
            for idx in val.split(args.batch_size):
                squared_error += (model(x[idx]) - y[idx]).square().sum(dim=0)
            rmse = (squared_error / len(val)).sqrt().tolist()
        print(f"Epoch {epoch}/{args.epochs}  C={model.lipschitz_bound().item():.5f}  "
              f"validation RMSE (vx, vy, yaw rate)={rmse}", flush=True)

    controller = torch.jit.script(ExplicitFeedback(model.cpu().eval()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.output.parent, suffix=".pt", delete=False) as temp:
        tmp = Path(temp.name)
    try:
        torch.jit.save(controller, str(tmp))
        os.replace(tmp, args.output)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"Saved frozen EF-HRL controller to {args.output}", flush=True)


if __name__ == "__main__":
    main()
