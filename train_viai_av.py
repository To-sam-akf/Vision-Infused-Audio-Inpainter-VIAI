import os
import re
import sys
import time
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from tqdm import tqdm

import Options_inpainting
from Data_loaders import audio_loader as av_loader
from Models.VIAI_AV_inpainting import VIAIAVModel
from utils.viai_a_metrics import compute_viai_a_metrics, write_mel_images


hparams = Options_inpainting.Inpainting_Config()
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
device = torch.device("cuda" if use_cuda else "cpu")


def _arg_was_passed(name):
    return any(arg == name or arg.startswith(name + "=") for arg in sys.argv[1:])


def configure_viai_av_defaults():
    if not _arg_was_passed("--name"):
        hparams.name = "VIAI-AV"
    if not _arg_was_passed("--log_event_path"):
        hparams.log_event_path = os.path.join(hparams.checkpoint_dir, "events_viai_av")


def checkpoint_step(path):
    match = re.search(r"checkpoint_step(\d+)", os.path.basename(str(path)))
    return int(match.group(1)) if match else -1


def resolve_latest_checkpoint(directory, prefix):
    if not directory or not os.path.isdir(directory):
        return None
    candidates = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(".pth.tar"):
            path = os.path.join(directory, filename)
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (checkpoint_step(path), os.path.getmtime(path)))[-1]


def resolve_init_checkpoint():
    if hparams.init_from_viai_a is not None:
        candidate = os.path.abspath(hparams.init_from_viai_a)
        if not os.path.exists(candidate):
            raise RuntimeError(
                f"VIAI-A initialization checkpoint not found: {candidate}. "
                "Train/test stage 2 first, or pass an existing checkpoint."
            )
        return candidate
    candidate = resolve_latest_checkpoint(
        os.path.abspath(hparams.checkpoint_dir),
        "VIAI-A-PatchGAN_checkpoint_step",
    )
    if candidate is None:
        raise RuntimeError(
            "train-viai-av requires a Stage2 VIAI-A-PatchGAN checkpoint. "
            "Pass --init_from_viai_a explicitly, or place "
            "VIAI-A-PatchGAN_checkpoint_step*.pth.tar under --checkpoint_dir."
        )
    return candidate


def resolve_resume_checkpoint():
    if hparams.resume_path is not None:
        candidate = os.path.abspath(hparams.resume_path)
        if os.path.exists(candidate):
            return candidate
        raise RuntimeError(f"VIAI-AV resume checkpoint not found: {candidate}")
    candidate = resolve_latest_checkpoint(
        os.path.abspath(hparams.checkpoint_dir),
        f"{hparams.name}_checkpoint_step",
    )
    if candidate is None:
        raise RuntimeError(
            "No VIAI-AV checkpoint found to resume. Pass --resume_path or place "
            f"{hparams.name}_checkpoint_step*.pth.tar under --checkpoint_dir."
        )
    return candidate


def _positive_interval(name):
    return max(0, int(getattr(hparams, name, 0)))


def _should_compute_ssim(train, step):
    if not train:
        return True
    metric_freq = _positive_interval("metric_freq")
    return step <= 1 or (metric_freq > 0 and step % metric_freq == 0)


def _should_write_images(train, step, batch_index):
    image_freq = _positive_interval("tb_image_freq")
    if train:
        return step <= 1 or (image_freq > 0 and step % image_freq == 0)
    return batch_index == 0


def _write_monitoring(writer, prefix, step, model, metrics):
    if writer is None:
        return
    writer.add_scalar(f"{prefix}/psnr_full", metrics["psnr_full"], step)
    writer.add_scalar(f"{prefix}/psnr_missing", metrics["psnr_missing"], step)
    if metrics["ssim_full"] is not None:
        writer.add_scalar(f"{prefix}/ssim_full", metrics["ssim_full"], step)
    writer.add_scalar(f"{prefix}/blank_frames", model.blank_length, step)
    writer.add_scalar(f"{prefix}/lr", model.current_lr, step)


def run_phase(model, data_loader, phase, global_step, writer, global_epoch):
    train = phase == "train"
    totals = {
        "loss_total": 0.0,
        "loss_recon": 0.0,
        "loss_full_l1": 0.0,
        "loss_missing_l1": 0.0,
        "loss_g_gan": 0.0,
        "loss_d": 0.0,
        "psnr_full": 0.0,
        "psnr_missing": 0.0,
        "ssim_full": 0.0,
    }
    sample_count = 0
    ssim_sample_count = 0
    batch_count = 0
    stop_training = False

    progress = tqdm(
        data_loader,
        desc=f"[VIAI-AV {phase}] epoch={global_epoch + 1}",
        unit="batch",
        dynamic_ncols=True,
    )
    for data in progress:
        iter_start_time = time.time()
        model.get_blank_space_length(global_step)
        model.set_inputs(data)
        if train:
            model.optimize_parameters(global_step)
            global_step += 1
        else:
            model.test()
        model.get_loss_items()
        metrics = compute_viai_a_metrics(
            model.mel_pred,
            model.mel_target_4d,
            model.missing_mask,
            compute_ssim=_should_compute_ssim(train, global_step),
        )

        totals["loss_total"] += model.loss_total_item
        totals["loss_recon"] += model.loss_recon_item
        totals["loss_full_l1"] += model.loss_full_l1_item
        totals["loss_missing_l1"] += model.loss_missing_l1_item
        totals["loss_g_gan"] += model.loss_G_GAN_item
        totals["loss_d"] += model.loss_D_item
        totals["psnr_full"] += metrics["psnr_full_sum"]
        totals["psnr_missing"] += metrics["psnr_missing_sum"]
        sample_count += metrics["num_samples"]
        if metrics["ssim_full_sum"] is not None:
            totals["ssim_full"] += metrics["ssim_full_sum"]
            ssim_sample_count += metrics["num_samples"]

        postfix = dict(
            step=global_step,
            loss=f"{model.loss_total_item:.4f}",
            recon=f"{model.loss_recon_item:.4f}",
            g_gan=f"{model.loss_G_GAN_item:.4f}",
            d=f"{model.loss_D_item:.4f}",
            psnr=f"{metrics['psnr_full']:.2f}",
            psnr_miss=f"{metrics['psnr_missing']:.2f}",
            blank=model.blank_length,
        )
        if metrics["ssim_full"] is not None:
            postfix["ssim"] = f"{metrics['ssim_full']:.4f}"
        progress.set_postfix(**postfix)

        if train and global_step % hparams.print_freq == 0:
            elapsed = (time.time() - iter_start_time) / max(1, hparams.batch_size)
            ssim_text = (
                f" ssim={metrics['ssim_full']:.6f}"
                if metrics["ssim_full"] is not None
                else ""
            )
            tqdm.write(
                f"[VIAI-AV train] step={global_step} "
                f"loss={model.loss_total_item:.6f} "
                f"recon={model.loss_recon_item:.6f} "
                f"full_l1={model.loss_full_l1_item:.6f} "
                f"missing_l1={model.loss_missing_l1_item:.6f} "
                f"g_gan={model.loss_G_GAN_item:.6f} "
                f"d={model.loss_D_item:.6f} "
                f"eta1={model.eta1_item:.6f} "
                f"psnr={metrics['psnr_full']:.3f} "
                f"psnr_missing={metrics['psnr_missing']:.3f}"
                f"{ssim_text} "
                f"time_per_sample={elapsed:.4f}s"
            )
        if train and global_step > 0 and global_step % hparams.checkpoint_interval == 0:
            model.save_checkpoint(global_step, global_epoch, hparams.checkpoint_dir)
        model.TF_writer(writer, global_step, prefix=phase)
        _write_monitoring(writer, phase, global_step, model, metrics)
        if _should_write_images(train, global_step, batch_count):
            write_mel_images(
                writer,
                phase,
                global_step,
                model.mel_input_4d,
                model.mel_pred,
                model.mel_target_4d,
                max_items=getattr(hparams, "tb_image_count", 4),
            )

        batch_count += 1
        if train and hparams.max_train_steps is not None and global_step >= hparams.max_train_steps:
            tqdm.write(f"Reached VIAI-AV smoke-test max_train_steps={hparams.max_train_steps}")
            stop_training = True
            break

    if batch_count == 0:
        return global_step, stop_training, None
    averages = {
        "loss_total": totals["loss_total"] / batch_count,
        "loss_recon": totals["loss_recon"] / batch_count,
        "loss_full_l1": totals["loss_full_l1"] / batch_count,
        "loss_missing_l1": totals["loss_missing_l1"] / batch_count,
        "loss_g_gan": totals["loss_g_gan"] / batch_count,
        "loss_d": totals["loss_d"] / batch_count,
        "psnr_full": totals["psnr_full"] / max(1, sample_count),
        "psnr_missing": totals["psnr_missing"] / max(1, sample_count),
        "ssim_full": None
        if ssim_sample_count == 0
        else totals["ssim_full"] / ssim_sample_count,
    }
    ssim_summary = (
        f" ssim={averages['ssim_full']:.6f}"
        if averages["ssim_full"] is not None
        else ""
    )
    tqdm.write(
        f"[VIAI-AV {phase}] "
        f"loss={averages['loss_total']:.6f} "
        f"recon={averages['loss_recon']:.6f} "
        f"full_l1={averages['loss_full_l1']:.6f} "
        f"missing_l1={averages['loss_missing_l1']:.6f} "
        f"g_gan={averages['loss_g_gan']:.6f} "
        f"d={averages['loss_d']:.6f} "
        f"psnr={averages['psnr_full']:.3f} "
        f"psnr_missing={averages['psnr_missing']:.3f}"
        f"{ssim_summary}"
    )
    return global_step, stop_training, averages


def train_loop(model, data_loaders, writer, start_step=0, start_epoch=0):
    global_step = start_step
    global_epoch = start_epoch
    while global_epoch < hparams.nepochs:
        if "train" in data_loaders:
            global_step, stop_training, _ = run_phase(
                model,
                data_loaders["train"],
                "train",
                global_step,
                writer,
                global_epoch,
            )
            if stop_training:
                break
        if "val" in data_loaders:
            with torch.no_grad():
                run_phase(model, data_loaders["val"], "val", global_step, writer, global_epoch)
        global_epoch += 1
        print(f"[VIAI-AV] epoch={global_epoch} current_lr={model.current_lr}")

    model.save_checkpoint(global_step, global_epoch, hparams.checkpoint_dir)


def main():
    configure_viai_av_defaults()
    os.makedirs(hparams.checkpoint_dir, exist_ok=True)
    data_loaders = av_loader.get_data_loaders(hparams.data_root, hparams.speaker_id, phases=("train", "val"))
    model = VIAIAVModel(hparams, device=device)

    global_step = 0
    global_epoch = 0
    if hparams.resume:
        checkpoint_path = resolve_resume_checkpoint()
        global_step, global_epoch = model.load_checkpoint(
            checkpoint_path,
            reset_optimizer=hparams.reset_optimizer,
        )
        print(f"[VIAI-AV] resumed checkpoint: {checkpoint_path} step={global_step} epoch={global_epoch}")
    else:
        init_checkpoint = resolve_init_checkpoint()
        source_step, source_epoch = model.load_viai_a_checkpoint(init_checkpoint)
        print(
            f"[VIAI-AV] initialized from VIAI-A checkpoint "
            f"source_step={source_step} source_epoch={source_epoch}"
        )

    log_event_path = hparams.log_event_path
    if log_event_path is None:
        log_event_path = "log/run-test-" + hparams.name + str(datetime.now()).replace(" ", "_")
    print("TensorBoard event log path:", log_event_path)
    writer = SummaryWriter(log_dir=log_event_path)

    try:
        train_loop(model, data_loaders, writer, start_step=global_step, start_epoch=global_epoch)
    except KeyboardInterrupt:
        print("Interrupted!")
    finally:
        writer.close()
    print("Finished VIAI-AV training")


if __name__ == "__main__":
    main()
    sys.exit(0)
