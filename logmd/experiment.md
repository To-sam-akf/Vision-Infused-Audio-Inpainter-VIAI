# 实验排查记录

## 2026-05-17 17:37:01 CST - VIAI-AV loss 方向异常排查

### 背景

训练命令：

```bash
python main.py train-viai-av -- \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --init_from_viai_a checkpoints/VIAI-A-PatchGAN_checkpoint_step000006800.pth.tar \
  --batch_size 16 \
  --num_workers 4 \
  --beta_gan 0.1 \
  --checkpoint_interval 1000 \
  --print_freq 100 \
  --display_id 0
```

现象：TensorBoard 中 `train/loss_full_l1`、`train/loss_missing_l1` 上升，`train/psnr_full`、`train/psnr_missing` 下降；验证集同样变差，表现为“越训越差”。

### 已确认事实

1. 本次导出的 `output.txt` 中没有 `train/loss_sync`、`train/loss_probe_gen`、`train/eta2` 等标量。
2. 因此这次 7126-step 结果不是当前 Stage4 sync/probe 默认开启后的日志，而是旧 Stage3 风格的 VIAI-AV run，或 TensorBoard 指向了旧 event 文件。
3. 论文公式摘录中 VIAI-AV generator loss 为：

```text
L_av_gen = L_av_GAN + beta * L_av_re
L_av_re = eta1(t) * full_l1 + missing_l1
```

4. 当前代码 `Models/VIAI_AV_inpainting.py` 中的实现与论文公式一致：

```python
loss_recon = eta1 * loss_full_l1 + loss_missing_l1
loss_av_gen = loss_G_GAN + beta_gan * loss_recon
```

注意：这里的 `beta_gan` 参数名很容易误导。它在当前代码里实际对应论文中的 `beta`，也就是 reconstruction loss 的权重，不是 GAN loss 的权重。

### 关键标量证据

7126-step run 的末尾标量：

```text
train/loss_total       last = 0.767486
train/loss_g_gan       last = 0.745425
train/loss_recon       last = 0.220610
train/loss_full_l1     last = 0.119271
train/loss_missing_l1  last = 0.164309
train/eta1             last = 0.472039
train/psnr_full        last = 16.311
train/psnr_missing     last = 13.952

val/loss_total         last = 0.947441
val/loss_g_gan         last = 0.918362
val/loss_recon         last = 0.290781
val/loss_full_l1       last = 0.115067
val/loss_missing_l1    last = 0.175714
val/eta1               last = 1.000000
val/psnr_full          last = 16.765
val/psnr_missing       last = 13.485
```

在训练命令 `--beta_gan 0.1` 下：

```text
beta_gan * train/loss_recon ~= 0.1 * 0.220610 = 0.022061
train/loss_g_gan ~= 0.745425
```

因此总损失几乎由 GAN loss 主导，重建 L1 对优化方向的约束很弱。这可以解释 L1/PSNR 随训练变差。

### 初步结论

当前 loss 公式没有相对论文写反；更可能的问题是训练配置中 `--beta_gan 0.1` 让 reconstruction loss 权重过小。若使用者把 `beta_gan` 理解成“GAN loss 权重”，就会和当前代码语义相反。

另外，`Models/VIAI_AV_inpainting.py` 的 `test()` 中调用 `_compute_losses(global_step=0)`，导致验证阶段 `val/eta1` 永远为 1.0。这个问题不会直接影响 PSNR，但会让 `val/loss_recon`、`val/loss_total` 的权重语义与训练当前 step 不一致。

### 建议验证实验

优先做短跑对照，确认是否是重建项权重过小导致反向优化。

实验 A：提高 reconstruction 权重到 1.0。

```bash
python main.py train-viai-av -- \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --init_from_viai_a checkpoints/VIAI-A-PatchGAN_checkpoint_step000006800.pth.tar \
  --batch_size 16 \
  --num_workers 4 \
  --beta_gan 1.0 \
  --checkpoint_interval 1000 \
  --print_freq 100 \
  --display_id 0 \
  --log_event_path checkpoints/events_viai_av_beta1
```

实验 B：进一步提高 reconstruction 权重到 10.0。

```bash
python main.py train-viai-av -- \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --init_from_viai_a checkpoints/VIAI-A-PatchGAN_checkpoint_step000006800.pth.tar \
  --batch_size 16 \
  --num_workers 4 \
  --beta_gan 10.0 \
  --checkpoint_interval 1000 \
  --print_freq 100 \
  --display_id 0 \
  --log_event_path checkpoints/events_viai_av_beta10
```

判断标准：

1. 若 `train/loss_full_l1`、`train/loss_missing_l1` 不再持续上升，且 `train/psnr_full`、`train/psnr_missing` 不再持续下降，则基本坐实主因是 reconstruction 权重过小。
2. 若 `val/psnr_missing` 也随之改善，说明问题不是单纯过拟合，而是目标函数权重方向导致训练目标与评估指标不一致。
3. 若提高 `beta_gan` 后仍恶化，再继续排查视频融合初始化、判别器过强、学习率、数据对齐和 mask 合成逻辑。

### 后续代码修正建议

为避免再次混淆，建议后续将参数语义改清楚：

```text
--lambda_recon  # reconstruction loss 权重，对应论文 beta
--lambda_gan    # GAN loss 权重，默认 1.0
```

并在 TensorBoard 中额外写入：

```text
train/weighted_loss_recon = beta_gan * loss_recon
train/weighted_loss_gan = loss_g_gan
```

这样之后可以直接看到各 loss 项对总损失的实际贡献。
