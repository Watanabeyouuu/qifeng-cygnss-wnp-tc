"""Reconstruct one saved CYGNSS case using a compatible EDM checkpoint."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import autocast
from tqdm import tqdm
from generate import StackedRandomGenerator


def sda_sampler(
    net,
    latents,
    class_labels,
    obs_mask,
    obs_values,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    sigma_y: float,
    gamma: float,
    C: int,
    tau: float,
    batch_size: int,
    precision: str,
    use_amp: bool,
    randn_like=torch.randn_like,
):
    """SDA sampler for masked scalar wind observations."""
    dtype = torch.float32 if precision == 'float32' else torch.float64
    device = latents.device
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    # Heun schedule with final noise level zero
    step_indices = torch.arange(num_steps, dtype=dtype, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    x_next = latents.to(dtype) * t_steps[0]

    def get_scores(x, t):
        score_prior_list = []
        score_like_list = []
        denoised_list = []

        for c in range(0, x.shape[0], batch_size):
            cx = x[c : c + batch_size].detach().requires_grad_(True)
            cl = class_labels[c : c + batch_size]
            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                c_denoised = net(cx.to(torch.float32), t, cl)
            c_denoised = c_denoised.to(dtype)
            denoised_list.append(c_denoised.detach())

            c_score_prior = (c_denoised - cx) / (t ** 2)
            score_prior_list.append(c_score_prior.detach())

            V = sigma_y**2 + t**2 * gamma
            c_obs_val = obs_values[c : c + batch_size]
            c_obs_mask = obs_mask[c : c + batch_size]

            c_diff = (c_denoised - c_obs_val) * c_obs_mask
            c_loss = 0.5 * torch.sum(c_diff ** 2) / V
            c_grad_loss = torch.autograd.grad(c_loss, cx)[0]
            score_like_list.append(-c_grad_loss.detach())

            del cx, c_denoised, c_loss, c_grad_loss, c_score_prior

        score_prior = torch.cat(score_prior_list, dim=0)
        score_like = torch.cat(score_like_list, dim=0)
        denoised = torch.cat(denoised_list, dim=0)
        return score_prior, score_like, denoised

    schedule = zip(t_steps[:-1], t_steps[1:])
    progress = tqdm(schedule, total=num_steps, desc="SDA", leave=False)
    for i, (t_cur, t_next) in enumerate(progress):
        x_cur = x_next

        if C > 0 and t_cur > sigma_min:
            delta = (tau * t_cur) ** 2
            for _ in range(C):
                score_prior, score_like, _ = get_scores(x_cur, t_cur)
                score_total = score_prior + score_like
                z = randn_like(x_cur)
                x_cur = x_cur + 0.5 * delta * score_total + torch.sqrt(delta) * z

        score_prior, score_like, denoised = get_scores(x_cur, t_cur)
        d_cur = (x_cur - denoised) / t_cur
        d_guided = d_cur - t_cur * score_like
        x_next = x_cur + (t_next - t_cur) * d_guided

        if i < num_steps - 1:
            score_prior_2, score_like_2, denoised_2 = get_scores(x_next, t_next)
            d_prime = (x_next - denoised_2) / t_next
            d_prime_guided = d_prime - t_next * score_like_2
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_guided + 0.5 * d_prime_guided)

    return x_next


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True)
    parser.add_argument(
        "--input",
        default=str(Path(__file__).parent / "data/soulik_20180818T180000.npz"),
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs/inference_settings.json"),
    )
    parser.add_argument("--output", default="example_outputs/reconstruction.npz")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    device = torch.device(args.device)

    with open(args.network, "rb") as source:
        checkpoint = pickle.load(source)
    if isinstance(checkpoint, dict):
        network = checkpoint.get("ema", checkpoint.get("net"))
    else:
        network = checkpoint
    network = network.to(device).eval().requires_grad_(False)

    with np.load(args.input, allow_pickle=False) as source:
        fields = {name: source[name].copy() for name in source.files}
    observations = fields["cygnss"]
    expected_shape = (network.img_resolution, network.img_resolution)
    if observations.shape != expected_shape:
        raise ValueError(f"The input grid must match the model resolution {expected_shape}.")
    valid = fields["sea_mask"].astype(bool) & np.isfinite(observations) & (observations > 0)
    divisor = settings["normalization_divisor"]
    values = np.where(valid, observations / divisor, 0).astype(np.float32)
    observation_tensor = torch.from_numpy(values)[None, None].to(device)
    mask_tensor = torch.from_numpy(valid.astype(np.float32))[None, None].to(device)
    random = StackedRandomGenerator(device, [settings["bulk_seed"]])
    latents = random.randn([1, 1, *expected_shape], device=device)
    labels = torch.zeros((1, 0), device=device)

    reconstructed = sda_sampler(
        network, latents, labels, mask_tensor, observation_tensor,
        num_steps=settings["num_steps"],
        sigma_min=settings["sigma_min"],
        sigma_max=settings["sigma_max"],
        rho=settings["rho"],
        sigma_y=settings["sigma_y"],
        gamma=settings["gamma"],
        C=settings["C"],
        tau=settings["tau"],
        batch_size=1,
        precision=settings["precision"],
        use_amp=False,
        randn_like=random.randn_like,
    )
    fields["reconstruction"] = reconstructed.detach().cpu().numpy()[0, 0] * divisor
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **fields)
    print(f"Saved reconstructed field to {output_path}.")


if __name__ == "__main__":
    main()
