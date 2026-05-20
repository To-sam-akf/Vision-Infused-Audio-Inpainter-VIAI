import os
import random

import torch
import torch.nn as nn

from Data_loaders import mel_loader
from loss_functions import GANLoss
from networks import Discriminator_Networks
from networks import Inpainting_Networks
from networks import New_Inpainting_Networks


class VIAIAModel(object):
    def __init__(self, hparams, device=None):
        self.hparams = hparams
        self.device = device if device is not None else torch.device("cpu")

        self.Mel_Encoder = Inpainting_Networks.MelEncoder(hparams=hparams).to(self.device)
        self.Mel_Decoder = New_Inpainting_Networks.MelDecoder(hparams=hparams).to(self.device)
        self.use_gan = bool(getattr(hparams, "use_gan", False))
        self.netD = None
        self.criterion_gan = None
        self.optimizer_D = None
        if self.use_gan:
            self.netD = Discriminator_Networks.MelDiscriminator().to(self.device)
            self.criterion_gan = GANLoss(use_lsgan=False, device=self.device)
        self.criterion_l1 = nn.L1Loss()
        self.optimizer_G = torch.optim.Adam(
            list(self.Mel_Encoder.parameters()) + list(self.Mel_Decoder.parameters()),
            lr=hparams.lr,
            betas=(hparams.beta1, hparams.beta2),
        )
        if self.use_gan:
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=hparams.lr,
                betas=(hparams.beta1, hparams.beta2),
            )
        self.current_lr = hparams.lr
        self.blank_length = getattr(hparams, "min_blank_frames", 20)

        self.loss_total_item = 0.0
        self.loss_recon_item = 0.0
        self.loss_full_l1_item = 0.0
        self.loss_missing_l1_item = 0.0
        self.loss_G_GAN_item = 0.0
        self.weighted_loss_recon_item = 0.0
        self.weighted_loss_gan_item = 0.0
        self.loss_D_item = 0.0
        self.loss_D_real_item = 0.0
        self.loss_D_fake_item = 0.0
        self.eta1_item = 0.0

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
        # η1(t) = max(floor, base^(global_step / interval))
        self.eta1 = self._eta(
            global_step,
            getattr(self.hparams, "recon_decay_base", 0.9),
            getattr(self.hparams, "recon_decay_interval", 1000.0),
            getattr(self.hparams, "recon_decay_floor", 0.1),
        )
        # 缺失区域单独算 L1，完整谱图也算 L1，然后用一个随训练步数衰减的 eta1 给完整谱图 L1 加权。
        self.loss_recon = self.eta1 * self.loss_full_l1 + self.loss_missing_l1
        if self.use_gan:
            pred_fake = self.netD(self.mel_pred)
            self.loss_G_GAN = self.criterion_gan(pred_fake, True)
            lambda_recon = getattr(self.hparams, "lambda_recon", 1.0)
            self.weighted_loss_recon = lambda_recon * self.loss_recon
            self.weighted_loss_gan = self.loss_G_GAN
            self.loss_total = self.weighted_loss_gan + self.weighted_loss_recon
        else:
            self.weighted_loss_recon = self.loss_recon
            self.weighted_loss_gan = torch.zeros(
                (), device=self.device, dtype=self.loss_recon.dtype
            )
            self.loss_total = self.loss_recon

    def _compute_discriminator_loss(self):
        if not self.use_gan:
            return
        pred_real = self.netD(self.mel_target_4d)
        pred_fake = self.netD(self.mel_pred.detach())
        self.loss_D_real = self.criterion_gan(pred_real, True, softlabel=True)
        self.loss_D_fake = self.criterion_gan(pred_fake, False, softlabel=True)
        self.loss_D = 0.5 * (self.loss_D_real + self.loss_D_fake)

    def optimize_parameters(self, global_step):
        self.Mel_Encoder.train()
        self.Mel_Decoder.train()
        if self.use_gan:
            self.netD.train()
        self._forward_inpainter()
        self._compute_losses(global_step)
        self.optimizer_G.zero_grad()
        self.loss_total.backward()
        self.optimizer_G.step()
        if self.use_gan:
            self.optimizer_D.zero_grad()
            self._compute_discriminator_loss()
            self.loss_D.backward()
            self.optimizer_D.step()
        self.current_lr = self.optimizer_G.param_groups[0]["lr"]

    def test(self, global_step=0):
        self.Mel_Encoder.eval()
        self.Mel_Decoder.eval()
        if self.use_gan:
            self.netD.eval()
        with torch.no_grad():
            self._forward_inpainter()
            self._compute_losses(global_step=global_step)
            self._compute_discriminator_loss()

    def get_loss_items(self):
        # loss_total_item       总损失
        # loss_full_l1_item     全谱图/已知区域相关的 L1 loss
        # loss_missing_l1_item  缺失区域的 L1 loss
        self.loss_total_item = float(self.loss_total.detach().cpu().item())
        self.loss_recon_item = float(self.loss_recon.detach().cpu().item())
        self.loss_full_l1_item = float(self.loss_full_l1.detach().cpu().item())
        self.loss_missing_l1_item = float(self.loss_missing_l1.detach().cpu().item())
        self.weighted_loss_recon_item = float(self.weighted_loss_recon.detach().cpu().item())
        self.weighted_loss_gan_item = float(self.weighted_loss_gan.detach().cpu().item())
        self.eta1_item = float(self.eta1)
        if self.use_gan:
            self.loss_G_GAN_item = float(self.loss_G_GAN.detach().cpu().item())
            self.loss_D_item = float(self.loss_D.detach().cpu().item())
            self.loss_D_real_item = float(self.loss_D_real.detach().cpu().item())
            self.loss_D_fake_item = float(self.loss_D_fake.detach().cpu().item())

    def get_current_errors(self):
        errors = {
            "loss_total": self.loss_total_item,
            "loss_full_l1": self.loss_full_l1_item,
            "loss_missing_l1": self.loss_missing_l1_item,
        }
        if self.use_gan:
            errors.update(
                {
                    "loss_recon": self.loss_recon_item,
                    "loss_g_gan": self.loss_G_GAN_item,
                    "loss_d": self.loss_D_item,
                }
            )
        return errors

    def TF_writer(self, writer, step, prefix="train"):
        if writer is None:
            return
        writer.add_scalar(f"{prefix}/loss_total", self.loss_total_item, step)
        writer.add_scalar(f"{prefix}/loss_full_l1", self.loss_full_l1_item, step)
        writer.add_scalar(f"{prefix}/loss_missing_l1", self.loss_missing_l1_item, step)
        writer.add_scalar(f"{prefix}/eta1", self.eta1_item, step)
        if self.use_gan:
            writer.add_scalar(f"{prefix}/loss_recon", self.loss_recon_item, step)
            writer.add_scalar(f"{prefix}/loss_g_gan", self.loss_G_GAN_item, step)
            writer.add_scalar(f"{prefix}/weighted_loss_recon", self.weighted_loss_recon_item, step)
            writer.add_scalar(f"{prefix}/weighted_loss_gan", self.weighted_loss_gan_item, step)
            writer.add_scalar(f"{prefix}/loss_d", self.loss_D_item, step)
            writer.add_scalar(f"{prefix}/loss_d_real", self.loss_D_real_item, step)
            writer.add_scalar(f"{prefix}/loss_d_fake", self.loss_D_fake_item, step)

    def save_checkpoint(self, global_step, global_epoch, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{self.hparams.name}_checkpoint_step{global_step:09d}.pth.tar",
        )
        checkpoint = {
            "Mel_Encoder": self.Mel_Encoder.state_dict(),
            "Mel_Decoder": self.Mel_Decoder.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict()
            if self.hparams.save_optimizer_state
            else None,
            "global_step": global_step,
            "global_epoch": global_epoch,
            "use_gan": self.use_gan,
        }
        if self.use_gan:
            checkpoint["netD"] = self.netD.state_dict()
            checkpoint["optimizer_D"] = (
                self.optimizer_D.state_dict()
                if self.hparams.save_optimizer_state
                else None
            )
        torch.save(checkpoint, checkpoint_path)
        print("Saved VIAI-A checkpoint:", checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path, reset_optimizer=False):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.Mel_Encoder.load_state_dict(checkpoint["Mel_Encoder"])
        self.Mel_Decoder.load_state_dict(checkpoint["Mel_Decoder"])
        if not reset_optimizer and checkpoint.get("optimizer_G") is not None:
            self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        if self.use_gan and checkpoint.get("netD") is not None:
            self.netD.load_state_dict(checkpoint["netD"])
        if (
            self.use_gan
            and not reset_optimizer
            and checkpoint.get("optimizer_D") is not None
        ):
            self.optimizer_D.load_state_dict(checkpoint["optimizer_D"])
        return int(checkpoint.get("global_step", 0)), int(checkpoint.get("global_epoch", 0))
