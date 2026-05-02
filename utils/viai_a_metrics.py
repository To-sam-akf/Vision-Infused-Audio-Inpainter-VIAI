import numpy as np
import torch


_IMAGE_WRITE_WARNING_SHOWN = False


def _as_bchw(tensor):
    if tensor.dim() == 3:
        return tensor.unsqueeze(1)
    if tensor.dim() == 4:
        return tensor
    raise ValueError("Expected a 3D (B, C, T) or 4D (B, 1, C, T) tensor.")


def structural_similarity_2d(pred, target):
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(target, pred, data_range=1.0))
    except ModuleNotFoundError:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        pred_mean = float(pred.mean())
        target_mean = float(target.mean())
        pred_var = float(pred.var())
        target_var = float(target.var())
        covariance = float(((pred - pred_mean) * (target - target_mean)).mean())
        numerator = (2 * pred_mean * target_mean + c1) * (2 * covariance + c2)
        denominator = (pred_mean ** 2 + target_mean ** 2 + c1) * (
            pred_var + target_var + c2
        )
        return numerator / max(denominator, 1e-12)


def compute_viai_a_metrics(mel_pred, mel_target, missing_mask, compute_ssim=True):
    pred = torch.clamp(_as_bchw(mel_pred).detach(), 0.0, 1.0)
    target = torch.clamp(_as_bchw(mel_target).detach(), 0.0, 1.0)
    mask = _as_bchw(missing_mask.detach()).to(device=pred.device, dtype=pred.dtype)
    batch_size = pred.size(0)

    full_mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    masked_sse = torch.sum(((pred - target) ** 2) * mask, dim=(1, 2, 3))
    masked_count = torch.clamp(torch.sum(mask, dim=(1, 2, 3)), min=1.0)
    missing_mse = masked_sse / masked_count

    full_psnr = -10.0 * torch.log10(torch.clamp(full_mse, min=1e-12))
    missing_psnr = -10.0 * torch.log10(torch.clamp(missing_mse, min=1e-12))
    full_psnr_sum = float(full_psnr.sum().cpu().item())
    missing_psnr_sum = float(missing_psnr.sum().cpu().item())

    ssim_full_sum = None
    if compute_ssim:
        pred_np = pred.squeeze(1).cpu().numpy()
        target_np = target.squeeze(1).cpu().numpy()
        ssim_values = [
            structural_similarity_2d(pred_np[index], target_np[index])
            for index in range(batch_size)
        ]
        ssim_full_sum = float(np.sum(ssim_values))

    metrics = {
        "psnr_full_sum": full_psnr_sum,
        "psnr_missing_sum": missing_psnr_sum,
        "ssim_full_sum": ssim_full_sum,
        "num_samples": batch_size,
        "psnr_full": full_psnr_sum / batch_size,
        "psnr_missing": missing_psnr_sum / batch_size,
        "ssim_full": None if ssim_full_sum is None else ssim_full_sum / batch_size,
    }
    return metrics


def mel_image_batches(mel_input, mel_pred, mel_target, max_items=4):
    mel_input = torch.clamp(_as_bchw(mel_input).detach().cpu(), 0.0, 1.0)
    mel_pred = torch.clamp(_as_bchw(mel_pred).detach().cpu(), 0.0, 1.0)
    mel_target = torch.clamp(_as_bchw(mel_target).detach().cpu(), 0.0, 1.0)
    abs_error = torch.clamp(torch.abs(mel_pred - mel_target), 0.0, 1.0)

    count = min(max(1, int(max_items)), mel_target.size(0))
    return {
        "input_masked": mel_input[:count],
        "prediction": mel_pred[:count],
        "target": mel_target[:count],
        "abs_error": abs_error[:count],
    }


def write_mel_images(writer, prefix, step, mel_input, mel_pred, mel_target, max_items=4):
    global _IMAGE_WRITE_WARNING_SHOWN
    if writer is None:
        return
    for name, images in mel_image_batches(
        mel_input,
        mel_pred,
        mel_target,
        max_items=max_items,
    ).items():
        try:
            writer.add_images(f"{prefix}/mel_{name}", images, step)
        except ModuleNotFoundError as exc:
            if exc.name != "PIL":
                raise
            if not _IMAGE_WRITE_WARNING_SHOWN:
                print(
                    "[VIAI-A] TensorBoard Mel image logging requires Pillow. "
                    "Install `pillow` to enable image panels."
                )
                _IMAGE_WRITE_WARNING_SHOWN = True
            return
