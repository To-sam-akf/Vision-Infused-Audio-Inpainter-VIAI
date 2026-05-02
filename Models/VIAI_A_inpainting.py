import os
import random

import torch
import torch.nn as nn

from Data_loaders import mel_loader
from networks import Inpainting_Networks
from networks import New_Inpainting_Networks


class VIAIAModel(object):
    def __init__(self, hparams, device=None):
        self.hparams = hparams
        self.device = device if device is not None else torch.device("cpu")

        self.Mel_Encoder = Inpainting_Networks.MelEncoder(hparams=hparams).to(self.device)
        self.Mel_Decoder = New_Inpainting_Networks.MelDecoder(hparams=hparams).to(self.device)
        self.criterion_l1 = nn.L1Loss()
        self.optimizer_G = torch.optim.Adam(
            list(self.Mel_Encoder.parameters()) + list(self.Mel_Decoder.parameters()),
            lr=hparams.lr,
            betas=(hparams.beta1, hparams.beta2),
        )
        self.current_lr = hparams.lr
        self.blank_length = getattr(hparams, "min_blank_frames", 20)

        self.loss_total_item = 0.0
        self.loss_full_l1_item = 0.0
        self.loss_missing_l1_item = 0.0

    def _eta(self, step, base, interval, floor):
        if interval <= 0:
            return floor
        return max(floor, base ** (float(step) / float(interval)))

    def get_blank_space_length(self, global_step):
        min_blank = max(1, int(getattr(self.hparams, "min_blank_frames", 20)))
        max_blank = max(min_blank, int(getattr(self.hparams, "max_blank_frames", 50)))
        self.blank_length = random.randint(min_blank, max_blank)
        return self.blank_length

    def set_inputs(self, data):
        self.mel_target = data["mel"].float().to(self.device)
        self.path_batch = data["path"]
        self.mel_input, self.missing_mask, self.missing_span = mel_loader.corrupt_mel_spectrogram(
            self.mel_target,
            self.blank_length,
        )
        self.mel_target_4d = self.mel_target.unsqueeze(1)
        self.mel_input_4d = self.mel_input.unsqueeze(1)
        self.missing_mask = self.missing_mask.to(self.device)

    def _forward_inpainter(self):
        mel_features = self.Mel_Encoder(self.mel_input)
        self.mel_pred = self.Mel_Decoder(mel_features, self.mel_input_4d.size())
        return self.mel_pred

    def _compute_losses(self, global_step):
        self.loss_full_l1 = self.criterion_l1(self.mel_pred, self.mel_target_4d)
        masked_abs = torch.abs(self.mel_pred - self.mel_target_4d) * self.missing_mask
        self.loss_missing_l1 = masked_abs.sum() / torch.clamp(self.missing_mask.sum(), min=1.0)
        eta1 = self._eta(
            global_step,
            getattr(self.hparams, "recon_decay_base", 0.9),
            getattr(self.hparams, "recon_decay_interval", 1000.0),
            getattr(self.hparams, "recon_decay_floor", 0.1),
        )
        self.loss_total = eta1 * self.loss_full_l1 + self.loss_missing_l1

    def optimize_parameters(self, global_step):
        self.Mel_Encoder.train()
        self.Mel_Decoder.train()
        self._forward_inpainter()
        self._compute_losses(global_step)
        self.optimizer_G.zero_grad()
        self.loss_total.backward()
        self.optimizer_G.step()
        self.current_lr = self.optimizer_G.param_groups[0]["lr"]

    def test(self):
        self.Mel_Encoder.eval()
        self.Mel_Decoder.eval()
        with torch.no_grad():
            self._forward_inpainter()
            self._compute_losses(global_step=0)

    def get_loss_items(self):
        self.loss_total_item = float(self.loss_total.detach().cpu().item())
        self.loss_full_l1_item = float(self.loss_full_l1.detach().cpu().item())
        self.loss_missing_l1_item = float(self.loss_missing_l1.detach().cpu().item())

    def get_current_errors(self):
        return {
            "loss_total": self.loss_total_item,
            "loss_full_l1": self.loss_full_l1_item,
            "loss_missing_l1": self.loss_missing_l1_item,
        }

    def TF_writer(self, writer, step, prefix="train"):
        if writer is None:
            return
        writer.add_scalar(f"{prefix}/loss_total", self.loss_total_item, step)
        writer.add_scalar(f"{prefix}/loss_full_l1", self.loss_full_l1_item, step)
        writer.add_scalar(f"{prefix}/loss_missing_l1", self.loss_missing_l1_item, step)

    def save_checkpoint(self, global_step, global_epoch, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{self.hparams.name}_checkpoint_step{global_step:09d}.pth.tar",
        )
        torch.save(
            {
                "Mel_Encoder": self.Mel_Encoder.state_dict(),
                "Mel_Decoder": self.Mel_Decoder.state_dict(),
                "optimizer_G": self.optimizer_G.state_dict()
                if self.hparams.save_optimizer_state
                else None,
                "global_step": global_step,
                "global_epoch": global_epoch,
            },
            checkpoint_path,
        )
        print("Saved VIAI-A checkpoint:", checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path, reset_optimizer=False):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.Mel_Encoder.load_state_dict(checkpoint["Mel_Encoder"])
        self.Mel_Decoder.load_state_dict(checkpoint["Mel_Decoder"])
        if not reset_optimizer and checkpoint.get("optimizer_G") is not None:
            self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        return int(checkpoint.get("global_step", 0)), int(checkpoint.get("global_epoch", 0))
