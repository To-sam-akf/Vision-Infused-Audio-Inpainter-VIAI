import os
import re
import sys

import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import Options_inpainting
from Data_loaders import viai_a_loader
from Models.VIAI_A_inpainting import VIAIAModel
from utils.viai_a_metrics import compute_viai_a_metrics


hparams = Options_inpainting.Inpainting_Config()
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
device = torch.device("cuda" if use_cuda else "cpu")


def _arg_was_passed(name):
    return any(arg == name or arg.startswith(name + "=") for arg in sys.argv[1:])


def configure_viai_a_defaults():
    if not _arg_was_passed("--name"):
        hparams.name = "VIAI-A"
    if not _arg_was_passed("--train_split_name"):
        hparams.train_split_name = "train_viai_a_split.txt"
    if not _arg_was_passed("--val_split_name"):
        hparams.val_split_name = "val_viai_a_split.txt"
    if not _arg_was_passed("--test_split_name"):
        hparams.test_split_name = "test_viai_a_split.txt"


def checkpoint_step(path):
    match = re.search(r"checkpoint_step(\d+)", os.path.basename(str(path)))
    return int(match.group(1)) if match else -1


def resolve_checkpoint_path(resume_path, checkpoint_dir, name):
    search_dirs = []
    if resume_path is not None:
        candidate = os.path.abspath(resume_path)
        if os.path.exists(candidate):
            return candidate
        search_dirs.append(os.path.dirname(candidate))
    search_dirs.append(os.path.abspath(checkpoint_dir))

    candidates = []
    for directory in search_dirs:
        if not directory or not os.path.isdir(directory):
            continue
        for filename in os.listdir(directory):
            if filename.startswith(f"{name}_checkpoint_step") and filename.endswith(".pth.tar"):
                candidates.append(os.path.join(directory, filename))
    if not candidates:
        raise RuntimeError(
            "No VIAI-A checkpoint found. Pass --resume_path or place a "
            f"{name}_checkpoint_step*.pth.tar file under {checkpoint_dir}."
        )
    return sorted(candidates, key=lambda path: (checkpoint_step(path), os.path.getmtime(path)))[-1]


def batch_metrics(model):
    metrics = compute_viai_a_metrics(
        model.mel_pred,
        model.mel_target_4d,
        model.missing_mask,
        compute_ssim=True,
    )
    return {
        "full_psnr": metrics["psnr_full_sum"],
        "missing_psnr": metrics["psnr_missing_sum"],
        "ssim": metrics["ssim_full_sum"],
        "num_samples": metrics["num_samples"],
    }


def evaluate(model, data_loader):
    totals = {
        "loss_total": 0.0,
        "full_l1": 0.0,
        "missing_l1": 0.0,
        "full_psnr": 0.0,
        "missing_psnr": 0.0,
        "ssim": 0.0,
    }
    sample_count = 0
    batch_count = 0

    progress = tqdm(
        data_loader,
        desc="[VIAI-A test] evaluating",
        unit="batch",
        dynamic_ncols=True,
    )
    for data in progress:
        model.get_blank_space_length(0)
        model.set_inputs(data)
        model.test()
        model.get_loss_items()
        metrics = batch_metrics(model)
        batch_size = metrics["num_samples"]

        totals["loss_total"] += model.loss_total_item
        totals["full_l1"] += model.loss_full_l1_item
        totals["missing_l1"] += model.loss_missing_l1_item
        totals["full_psnr"] += metrics["full_psnr"]
        totals["missing_psnr"] += metrics["missing_psnr"]
        totals["ssim"] += metrics["ssim"]
        sample_count += batch_size
        batch_count += 1
        progress.set_postfix(
            loss=f"{model.loss_total_item:.4f}",
            full_l1=f"{model.loss_full_l1_item:.4f}",
            missing_l1=f"{model.loss_missing_l1_item:.4f}",
            psnr=f"{metrics['full_psnr'] / batch_size:.2f}",
            ssim=f"{metrics['ssim'] / batch_size:.4f}",
        )

    if batch_count == 0:
        raise RuntimeError("VIAI-A test dataloader is empty.")

    return {
        "loss_total": totals["loss_total"] / batch_count,
        "mel_l1_full": totals["full_l1"] / batch_count,
        "mel_l1_missing": totals["missing_l1"] / batch_count,
        "psnr_full": totals["full_psnr"] / sample_count,
        "psnr_missing": totals["missing_psnr"] / sample_count,
        "ssim": totals["ssim"] / sample_count,
        "num_samples": sample_count,
    }


def main():
    configure_viai_a_defaults()
    data_loaders = viai_a_loader.get_data_loaders(hparams.data_root, phases=("test",))
    if "test" not in data_loaders:
        raise RuntimeError(
            f"VIAI-A test split is missing or empty: {os.path.join(hparams.data_root, hparams.test_split_name)}"
        )

    model = VIAIAModel(hparams, device=device)
    checkpoint_path = resolve_checkpoint_path(hparams.resume_path, hparams.checkpoint_dir, hparams.name)
    global_step, global_epoch = model.load_checkpoint(checkpoint_path, reset_optimizer=True)
    print(f"[VIAI-A test] loaded checkpoint: {checkpoint_path} (step={global_step}, epoch={global_epoch})")

    results = evaluate(model, data_loaders["test"])
    print(
        "[VIAI-A test] "
        f"samples={results['num_samples']} "
        f"loss={results['loss_total']:.6f} "
        f"mel_l1_full={results['mel_l1_full']:.6f} "
        f"mel_l1_missing={results['mel_l1_missing']:.6f} "
        f"psnr_full={results['psnr_full']:.3f} "
        f"psnr_missing={results['psnr_missing']:.3f} "
        f"ssim={results['ssim']:.4f}"
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
