import argparse
import os
import torch.nn as nn


class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.initialized = False

    def initialize(self):
        # Basic
        self.parser.add_argument("--name", type=str, default="VIAI-AV")
        self.parser.add_argument("--isTrain", type=bool, default=True)
        self.parser.add_argument("--data_root", type=str, default="./data")
        self.parser.add_argument("--image_path", type=str, default="./data")
        self.parser.add_argument("--speaker_id", type=int, default=None)
        self.parser.add_argument("--test_size", type=float, default=0.05)
        self.parser.add_argument("--metadata_name", type=str, default="metadata.csv")
        self.parser.add_argument("--new_split_name", type=str, default="_new_split.txt")

        # Resume / checkpoint
        self.parser.add_argument("--resume", action="store_true")
        self.parser.add_argument("--resume_path", type=str, default=None)
        self.parser.add_argument("--load_pretrain", action="store_true")
        self.parser.add_argument("--wavenet_pretrain", type=str, default=None)
        self.parser.add_argument("--reset_optimizer", action="store_true")
        self.parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
        self.parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
        self.parser.add_argument("--log_event_path", type=str, default=None)
        self.parser.add_argument("--save_optimizer_state", type=bool, default=True)

        # Runtime
        self.parser.add_argument("--num_workers", type=int, default=4)
        self.parser.add_argument("--batch_size", type=int, default=16)
        self.parser.add_argument("--pin_memory", type=bool, default=True)
        self.parser.add_argument("--mul_gpu", type=bool, default=True)
        self.parser.add_argument("--cuda_on", type=bool, default=True)

        # Visualizer
        self.parser.add_argument("--display_id", type=int, default=0)
        self.parser.add_argument("--display_freq", type=int, default=200)
        self.parser.add_argument("--print_freq", type=int, default=100)
        self.parser.add_argument("--display_winsize", type=int, default=256)
        self.parser.add_argument("--display_port", type=int, default=8097)
        self.parser.add_argument("--display_single_pane_ncols", type=int, default=0)

        # Optimization
        self.parser.add_argument("--nepochs", type=int, default=100)
        self.parser.add_argument("--lr", type=float, default=1e-4)
        self.parser.add_argument("--beta1", type=float, default=0.5)
        self.parser.add_argument("--beta2", type=float, default=0.999)
        self.parser.add_argument("--checkpoint_interval", type=int, default=1000)
        self.parser.add_argument("--train_eval_interval", type=int, default=1000)
        self.parser.add_argument("--test_eval_epoch_interval", type=int, default=1)
        self.parser.add_argument(
            "--max_train_steps",
            type=int,
            default=None,
            help="Stop after this many optimizer steps. Intended for local smoke tests only.",
        )

        # Loss weights (paper-style objective)
        self.parser.add_argument("--lambda_recon", type=float, default=1.0)
        self.parser.add_argument("--beta_gan", type=float, default=0.1)
        self.parser.add_argument("--lambda_sync", type=float, default=1.0)
        self.parser.add_argument("--sync_margin", type=float, default=1.0)
        self.parser.add_argument("--recon_decay_base", type=float, default=0.9)
        self.parser.add_argument("--recon_decay_interval", type=float, default=1000.0)
        self.parser.add_argument("--recon_decay_floor", type=float, default=0.1)
        self.parser.add_argument("--sync_decay_base", type=float, default=0.9)
        self.parser.add_argument("--sync_decay_interval", type=float, default=1000.0)
        self.parser.add_argument("--sync_decay_floor", type=float, default=0.1)

        # Missing-region setup (4s inputs, 0.4s~1.0s gaps)
        self.parser.add_argument("--max_time_sec", type=float, default=None)
        self.parser.add_argument("--max_time_steps", type=int, default=64000)
        self.parser.add_argument("--min_blank_frames", type=int, default=20)
        self.parser.add_argument("--max_blank_frames", type=int, default=50)

        # Mel / audio features
        self.parser.add_argument("--sample_rate", type=int, default=16000)
        self.parser.add_argument("--input_type", type=str, default="raw")
        self.parser.add_argument("--quantize_channels", type=int, default=65536)
        self.parser.add_argument("--rescaling", type=bool, default=True)
        self.parser.add_argument("--rescaling_max", type=float, default=0.999)
        self.parser.add_argument("--silence_threshold", type=float, default=2)
        self.parser.add_argument("--cin_channels", type=int, default=80)
        self.parser.add_argument("--num_mels", type=int, default=80)
        self.parser.add_argument("--max_mel_lengths", type=int, default=200)
        self.parser.add_argument("--fft_size", type=int, default=1280)
        self.parser.add_argument("--hop_size", type=int, default=320)
        self.parser.add_argument("--frame_shift_ms", type=float, default=None)
        self.parser.add_argument("--fmin", type=float, default=125.0)
        self.parser.add_argument("--fmax", type=float, default=7600.0)
        self.parser.add_argument("--min_level_db", type=float, default=-100.0)
        self.parser.add_argument("--ref_level_db", type=float, default=20.0)
        self.parser.add_argument("--allow_clipping_in_normalization", type=bool, default=True)

        # Visual stream and fusion encoder
        self.parser.add_argument("--feature_length", type=int, default=256)
        self.parser.add_argument("--length_feature", type=int, default=256)
        self.parser.add_argument("--image_size", type=int, default=256)
        self.parser.add_argument("--image_rescal_size", type=int, default=256)
        self.parser.add_argument("--image_channel_size", type=int, default=3)
        self.parser.add_argument("--image_hope_size", type=int, default=1)
        self.parser.add_argument("--image", type=bool, default=True)
        self.parser.add_argument("--flow", type=bool, default=True)
        self.parser.add_argument("--load_num", type=int, default=1)
        self.parser.add_argument("--resnet_pretrain", type=bool, default=False)
        self.parser.add_argument("--resnet_pretrain_path", type=str, default=None)

        # Decoder / GAN / WaveNet
        self.parser.add_argument("--norm_type", type=str, default="batch")
        self.parser.add_argument("--out_channels", type=int, default=30)
        self.parser.add_argument("--decode_layers", type=int, default=24)
        self.parser.add_argument("--decode_stacks", type=int, default=3)
        self.parser.add_argument("--residual_channels", type=int, default=256)
        self.parser.add_argument("--gate_channels", type=int, default=256)
        self.parser.add_argument("--skip_out_channels", type=int, default=256)
        self.parser.add_argument("--kernel_size", type=int, default=3)
        self.parser.add_argument("--dropout", type=float, default=0.05)
        self.parser.add_argument("--weight_normalization", type=bool, default=True)
        self.parser.add_argument("--upsample_conditional_features", type=bool, default=True)
        self.parser.add_argument("--upsample_scales", type=int, nargs="+", default=[5, 8, 8])
        self.parser.add_argument("--freq_axis_kernel_size", type=int, default=3)
        self.parser.add_argument("--n_speakers", type=int, default=None)
        self.parser.add_argument("--gin_channels", type=int, default=-1)
        self.parser.add_argument("--file_channel", type=int, default=-1)
        self.parser.add_argument("--log_scale_min", type=float, default=-7.0)

        self.initialized = True
        return self.parser

    def gather_options(self, args=None):
        if not self.initialized:
            self.initialize()
        # Ignore unknown args to make config import-safe in notebooks/tools.
        opt, _ = self.parser.parse_known_args(args=args)
        return opt

    def parse(self, args=None):
        opt = self.gather_options(args=args)

        if opt.length_feature != opt.feature_length:
            opt.length_feature = opt.feature_length
        opt.feature_length = opt.length_feature

        # Keep compatibility with two historical field names.
        if opt.file_channel != -1 and opt.gin_channels == -1:
            opt.gin_channels = opt.file_channel
        if opt.gin_channels != -1 and opt.file_channel == -1:
            opt.file_channel = opt.gin_channels

        if opt.max_time_steps is not None and opt.hop_size > 0:
            opt.max_mel_lengths = opt.max_time_steps // opt.hop_size
        opt.num_mels = opt.cin_channels

        if opt.norm_type.lower() == "instance":
            opt.normlayer = nn.InstanceNorm2d
        else:
            opt.normlayer = nn.BatchNorm2d

        if not opt.checkpoint_dir:
            opt.checkpoint_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not opt.log_event_path:
            opt.log_event_path = os.path.join(opt.checkpoint_dir, "events")

        return opt
