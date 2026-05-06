import csv
import json
import os
import re
import sys

import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import Options_inpainting
from Data_loaders import audio_loader as av_loader
from Models.VIAI_AV_inpainting import VIAIAVModel
from utils.viai_a_metrics import compute_viai_a_metrics, save_mel_comparison_batch


hparams = Options_inpainting.Inpainting_Config()
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
device = torch.device("cuda" if use_cuda else "cpu")


RESULT_FIELDS = [
    "checkpoint_path",
    "checkpoint_step",
    "global_step",
    "global_epoch",
    "test_split_name",
    "stage",
    "use_gan",
    "num_samples",
    "loss_total",
    "loss_recon",
    "loss_g_gan",
    "loss_d",
    "eta1",
    "beta_gan",
    "mel_l1_full",
    "mel_l1_missing",
    "psnr_full",
    "psnr_missing",
    "ssim",
]


def _arg_was_passed(name):
    return any(arg == name or arg.startswith(name + "=") for arg in sys.argv[1:])


def configure_viai_av_defaults():
    if not _arg_was_passed("--name"):
        hparams.name = "VIAI-AV"
    if not _arg_was_passed("--results_dir"):
        hparams.results_dir = "./checkpoints/viai_av_test_results"


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
            "No VIAI-AV checkpoint found. Pass --resume_path or place a "
            f"{name}_checkpoint_step*.pth.tar file under {checkpoint_dir}."
        )
    return sorted(candidates, key=lambda path: (checkpoint_step(path), os.path.getmtime(path)))[-1]


def format_step(step):
    if step is None or step < 0:
        return "unknown"
    return f"{step:09d}"


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


def mel_image_output_dir(results_dir, checkpoint_step_value):
    return os.path.join(
        results_dir,
        "mel-image",
        f"step{format_step(checkpoint_step_value)}",
    )


def evaluate(model, data_loader, image_dir=None):
    totals = {
        "loss_total": 0.0,
        "loss_recon": 0.0,
        "loss_g_gan": 0.0,
        "loss_d": 0.0,
        "eta1": 0.0,
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
        desc="[VIAI-AV test] evaluating",
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
        totals["loss_recon"] += model.loss_recon_item
        totals["loss_g_gan"] += model.loss_G_GAN_item
        totals["loss_d"] += model.loss_D_item
        totals["eta1"] += model.eta1_item
        totals["full_l1"] += model.loss_full_l1_item
        totals["missing_l1"] += model.loss_missing_l1_item
        totals["full_psnr"] += metrics["full_psnr"]
        totals["missing_psnr"] += metrics["missing_psnr"]
        totals["ssim"] += metrics["ssim"]
        sample_count += batch_size
        batch_count += 1
        if image_dir is not None:
            save_mel_comparison_batch(
                image_dir,
                sample_count - batch_size,
                model.path_batch,
                model.mel_input_4d,
                model.mel_pred,
                model.mel_target_4d,
            )
        progress.set_postfix(
            loss=f"{model.loss_total_item:.4f}",
            recon=f"{model.loss_recon_item:.4f}",
            g_gan=f"{model.loss_G_GAN_item:.4f}",
            d=f"{model.loss_D_item:.4f}",
            psnr=f"{metrics['full_psnr'] / batch_size:.2f}",
            ssim=f"{metrics['ssim'] / batch_size:.4f}",
        )

    if batch_count == 0:
        raise RuntimeError("VIAI-AV test dataloader is empty.")
    return {
        "loss_total": totals["loss_total"] / batch_count,
        "loss_recon": totals["loss_recon"] / batch_count,
        "loss_g_gan": totals["loss_g_gan"] / batch_count,
        "loss_d": totals["loss_d"] / batch_count,
        "eta1": totals["eta1"] / batch_count,
        "mel_l1_full": totals["full_l1"] / batch_count,
        "mel_l1_missing": totals["missing_l1"] / batch_count,
        "psnr_full": totals["full_psnr"] / sample_count,
        "psnr_missing": totals["missing_psnr"] / sample_count,
        "ssim": totals["ssim"] / sample_count,
        "num_samples": sample_count,
    }


def build_result_record(checkpoint_path, checkpoint_step_value, global_step, global_epoch, results):
    return {
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_step": int(checkpoint_step_value),
        "global_step": int(global_step),
        "global_epoch": int(global_epoch),
        "test_split_name": hparams.test_split_name,
        "stage": "VIAI-AV-stage3",
        "use_gan": True,
        "num_samples": int(results["num_samples"]),
        "loss_total": float(results["loss_total"]),
        "loss_recon": float(results["loss_recon"]),
        "loss_g_gan": float(results["loss_g_gan"]),
        "loss_d": float(results["loss_d"]),
        "eta1": float(results["eta1"]),
        "beta_gan": float(getattr(hparams, "beta_gan", 0.1)),
        "mel_l1_full": float(results["mel_l1_full"]),
        "mel_l1_missing": float(results["mel_l1_missing"]),
        "psnr_full": float(results["psnr_full"]),
        "psnr_missing": float(results["psnr_missing"]),
        "ssim": float(results["ssim"]),
    }


def coerce_csv_record(row):
    record = {}
    int_fields = {"checkpoint_step", "global_step", "global_epoch", "num_samples"}
    float_fields = {
        "loss_total",
        "loss_recon",
        "loss_g_gan",
        "loss_d",
        "eta1",
        "beta_gan",
        "mel_l1_full",
        "mel_l1_missing",
        "psnr_full",
        "psnr_missing",
        "ssim",
    }
    for field in RESULT_FIELDS:
        value = row.get(field, "")
        if field in int_fields and value != "":
            record[field] = int(value)
        elif field in float_fields and value != "":
            record[field] = float(value)
        else:
            record[field] = value
    return record


def write_result_files(record, results_dir, name):
    os.makedirs(results_dir, exist_ok=True)
    step_text = format_step(record["checkpoint_step"])
    json_path = os.path.join(results_dir, f"{name}_step{step_text}_test.json")
    csv_path = os.path.join(results_dir, f"{name}_test_summary.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    records_by_step = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row.get("checkpoint_step"):
                    continue
                old_record = coerce_csv_record(row)
                records_by_step[int(old_record["checkpoint_step"])] = old_record
    records_by_step[int(record["checkpoint_step"])] = record
    sorted_records = [records_by_step[step] for step in sorted(records_by_step)]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for item in sorted_records:
            writer.writerow({field: item.get(field, "") for field in RESULT_FIELDS})
    return json_path, csv_path


def main():
    configure_viai_av_defaults()
    data_loaders = av_loader.get_data_loaders(
        hparams.data_root,
        hparams.speaker_id,
        test_shuffle=False,
        phases=("test",),
    )
    if "test" not in data_loaders:
        raise RuntimeError(
            f"VIAI-AV test split is missing or empty: {os.path.join(hparams.data_root, hparams.test_split_name)}"
        )

    model = VIAIAVModel(hparams, device=device)
    checkpoint_path = resolve_checkpoint_path(hparams.resume_path, hparams.checkpoint_dir, hparams.name)
    global_step, global_epoch = model.load_checkpoint(checkpoint_path, reset_optimizer=True)
    print(f"[VIAI-AV test] loaded checkpoint: {checkpoint_path} (step={global_step}, epoch={global_epoch})")

    checkpoint_step_value = checkpoint_step(checkpoint_path)
    if checkpoint_step_value < 0:
        checkpoint_step_value = global_step
    image_dir = mel_image_output_dir(hparams.results_dir, checkpoint_step_value)
    results = evaluate(model, data_loaders["test"], image_dir=image_dir)
    result_record = build_result_record(
        checkpoint_path,
        checkpoint_step_value,
        global_step,
        global_epoch,
        results,
    )
    json_path, csv_path = write_result_files(
        result_record,
        hparams.results_dir,
        hparams.name,
    )
    print(
        "[VIAI-AV test] "
        f"samples={results['num_samples']} "
        f"loss={results['loss_total']:.6f} "
        f"recon={results['loss_recon']:.6f} "
        f"g_gan={results['loss_g_gan']:.6f} "
        f"d={results['loss_d']:.6f} "
        f"eta1={results['eta1']:.6f} "
        f"mel_l1_full={results['mel_l1_full']:.6f} "
        f"mel_l1_missing={results['mel_l1_missing']:.6f} "
        f"psnr_full={results['psnr_full']:.3f} "
        f"psnr_missing={results['psnr_missing']:.3f} "
        f"ssim={results['ssim']:.4f}"
    )
    print(f"[VIAI-AV test] wrote json: {json_path}")
    print(f"[VIAI-AV test] wrote summary csv: {csv_path}")
    print(f"[VIAI-AV test] wrote mel images: {image_dir}")


if __name__ == "__main__":
    main()
    sys.exit(0)
