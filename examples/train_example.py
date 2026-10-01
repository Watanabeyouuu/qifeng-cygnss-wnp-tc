"""A small wind-field training example using the upstream EDM network and loss."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class WindDataset(Dataset):
    def __init__(self, directory):
        self.files = sorted(Path(directory).glob("*.npy"))
        if not self.files:
            raise ValueError("Prepare the normalized CCMP fields first.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        wind = np.load(self.files[index], allow_pickle=False).astype(np.float32)
        if wind.ndim == 2:
            wind = wind[None, :, :]
        return torch.from_numpy(wind)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--config", default=str(Path(__file__).with_name("training_config.json"))
    )
    parser.add_argument("--outdir", default="example_outputs/training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    dataset = WindDataset(args.data_dir)
    sample = dataset[0]
    expected_shape = (1, config["resolution"], config["resolution"])
    if tuple(sample.shape) != expected_shape or not torch.isfinite(sample).all():
        raise ValueError(f"Expected finite normalized wind fields of shape {expected_shape}.")
    if args.dry_run:
        print(f"Ready: {len(dataset)} fields, shape {tuple(sample.shape)}, wind units m/s divided by 100.")
        return

    from training.networks import EDMPrecond
    from training.loss import EDMLoss

    torch.manual_seed(config["seed"])
    device = torch.device(args.device)
    network = EDMPrecond(
        img_resolution=config["resolution"],
        img_channels=1,
        label_dim=0,
        **config["network"],
    ).to(device).train()
    loss_function = EDMLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=config["learning_rate"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    batches = iter(loader)

    for step in range(config["steps"]):
        try:
            wind = next(batches)
        except StopIteration:
            batches = iter(loader)
            wind = next(batches)
        wind = wind.to(device)
        optimizer.zero_grad()
        loss = loss_function(network, wind, labels=None).mean()
        loss.backward()
        optimizer.step()

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "network_example.pkl", "wb") as output:
        pickle.dump({"net": network.eval().cpu()}, output)
    print(f"Finished {config['steps']} example steps; final loss {loss.item():.6f}.")


if __name__ == "__main__":
    main()
