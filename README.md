# Vision-Infused Deep Audio Inpainting

We present a vision-infused method that can deal with both audio-only and audio-visual associated inpainting Inspired by image inpainting, called `Vision-Infused Audio Inpainter (VIAI)`.

[[Project]](https://hangz-nju-cuhk.github.io/projects/AudioInpainting) [[Paper]](https://arxiv.org/abs/1910.10997) [[Demo]](https://www.youtube.com/watch?v=2C8s_YuRRxk)

<img src='./misc/pipeline2.png' width=880>

## Requirements
* [python 3](https://www.python.org/download/releases/3.6/)
* [PyTorch](https://pytorch.org/)（cloud training should install the CUDA build matching the target GPU)
* [opencv-contrib-python](https://pypi.org/project/opencv-contrib-python/) for TV-L1 optical flow

### 第一阶段：VIAI-A Audio-Only
**Train 调用链**
1. 命令入口  
   `python main.py train-viai-a ...`

2. [main.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/main.py:217)  
   `MODULE_MAP["train-viai-a"] -> train_viai_a`

3. [train_viai_a.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/train_viai_a.py:223)  
   `main()` 创建 dataloader 和 `VIAIAModel`

4. [viai_a_loader.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Data_loaders/viai_a_loader.py:113)  
   `get_data_loaders(..., phases=("train", "val"))`

5. [viai_a_loader.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Data_loaders/viai_a_loader.py:89)  
   `VIAIASplitDataset.__getitem__()`：
   - 读取 `mel.npy`
   - train 阶段随机裁 200 帧 Mel
   - val 阶段取中间 200 帧 Mel
   - 返回：
     ```python
     {
       "mel": Tensor[80, 200],
       "audio": Tensor[64000],
       "path": sample_dir
     }
     ```

6. [train_viai_a.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/train_viai_a.py:70)  
   `run_phase()`

7. [train_viai_a.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/train_viai_a.py:93)  
   每个 batch 先随机缺失长度：
   ```python
   model.get_blank_space_length(global_step)
   ```

8. [train_viai_a.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/train_viai_a.py:94)  
   然后设置输入：
   ```python
   model.set_inputs(data)
   ```

9. [Models/VIAI_A_inpainting.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Models/VIAI_A_inpainting.py:120)  
   `set_inputs()` 中真正执行 mask：
   ```python
   self.mel_input, self.missing_mask, self.missing_span = mel_loader.corrupt_mel_spectrogram(
       self.mel_target,
       self.blank_length,
   )
   ```

10. [mel_loader.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Data_loaders/mel_loader.py:35)  
    `corrupt_mel_spectrogram()`：
    - 随机选择 `start`
    - 生成 `missing_mask`
    - 对缺失区域做插值替换

11. [mel_loader.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Data_loaders/mel_loader.py:5)  
    `build_missing_mask()` 生成 mask：
    ```python
    mask[:, :, :, start:end] = 1.0
    ```

12. [mel_loader.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Data_loaders/mel_loader.py:12)  
    `interpolate_missing_region()` 用左右边界插值填入缺失区域：
    ```python
    mel_4d[:, :, :, start:end] = left * (1.0 - alpha) + right * alpha
    ```

13. [Models/VIAI_A_inpainting.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Models/VIAI_A_inpainting.py:148)  
    train 阶段：
    ```python
    model.optimize_parameters(global_step)
    ```

14. [Models/VIAI_A_inpainting.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Models/VIAI_A_inpainting.py:131)  
    forward：
    ```python
    mel_features = self.Mel_Encoder(self.mel_input)
    self.mel_pred = self.Mel_Decoder(...)
    ```

15. [Models/VIAI_A_inpainting.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Models/VIAI_A_inpainting.py:136)  
    loss：
    ```python
    loss_full_l1 = L1(pred, target)
    loss_missing_l1 = abs(pred - target) * missing_mask
    ```

**Test 调用链**
1. 命令入口  
   `python main.py test-viai-a ...`

2. [main.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/main.py:225)  
   `MODULE_MAP["test-viai-a"] -> test_viai_a`

3. [test_viai_a.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/test_viai_a.py:251)  
   `main()` 创建 test dataloader，加载 checkpoint。

4. [viai_a_loader.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Data_loaders/viai_a_loader.py:113)  
   `get_data_loaders(..., phases=("test",))`

5. [viai_a_loader.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Data_loaders/viai_a_loader.py:81)  
   test 阶段 `train=False`，所以 Mel window 是中间裁剪：
   ```python
   return max_start // 2
   ```

6. [test_viai_a.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/test_viai_a.py:186)  
   `evaluate(model, data_loader, image_dir=...)`

7. [test_viai_a.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/test_viai_a.py:205)  
   每个 test batch 也会随机缺失长度：
   ```python
   model.get_blank_space_length(0)
   ```

8. [test_viai_a.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/test_viai_a.py:206)  
   测试也调用：
   ```python
   model.set_inputs(data)
   ```

9. 所以 test 同样进入 [Models/VIAI_A_inpainting.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Models/VIAI_A_inpainting.py:120)，执行同一个：
   ```python
   mel_loader.corrupt_mel_spectrogram(...)
   ```

10. [test_viai_a.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/test_viai_a.py:207)  
    测试 forward：
    ```python
    model.test()
    ```

11. [Models/VIAI_A_inpainting.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Models/VIAI_A_inpainting.py:158)  
    `test()` 里 no grad forward + loss：
    ```python
    self._forward_inpainter()
    self._compute_losses(global_step=0)
    ```

12. [test_viai_a.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/test_viai_a.py:171)  
    `batch_metrics()` 计算：
    ```python
    compute_viai_a_metrics(model.mel_pred, model.mel_target_4d, model.missing_mask)
    ```

13. [utils/viai_a_metrics.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/utils/viai_a_metrics.py:36)  
    计算：
    - full PSNR
    - missing-region PSNR
    - SSIM

14. [test_viai_a.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/test_viai_a.py:220)  
    保存测试图片：
    ```python
    save_mel_comparison_batch(
        image_dir,
        ...,
        model.mel_input_4d,
        model.mel_pred,
        model.mel_target_4d,
    )
    ```

VIAI-A 第一阶段已经补齐并通过本地 smoke test。该阶段只训练音频谱图修复模型，严格不使用视频帧、光流、visual encoder、sync loss、GAN loss 或 WaveNet。当前链路为：

```text
MUSICES raw video -> 16kHz mono audio -> 80-bin Mel-spectrogram -> 随机 mask 0.4s-1.0s -> MelEncoder + MelDecoder -> L1 / PSNR / SSIM
```

本地已验证：

```text
prepare-viai-a -> split-data --audio-only -> train-viai-a 1 step -> test-viai-a
```

#### 云端环境准备

云端如果已经有 CUDA 版 PyTorch，不需要 `uv`，也不要让本仓库覆盖云端 PyTorch。只补齐 VIAI-A 需要的其他依赖：

```bash
cd /root/Vision-Infused-Audio-Inpainter-VIAI
python -m pip install --upgrade pip
python -m pip install imageio-ffmpeg librosa nnmnkwii "numpy==1.22.4" opencv-contrib-python-headless pillow scikit-image tensorboard tensorboardX tqdm "yt-dlp[default]"
python -c "import torch, librosa, cv2, nnmnkwii, tensorboardX, tqdm; from skimage.metrics import structural_similarity; print(torch.__version__, torch.cuda.is_available(), cv2.__version__)"
```

如果云端已经通过 `conda` 或平台镜像安装了部分依赖，可以只安装缺失项。Python 3.8、`scipy==1.6.3`、`numba==0.56.4` 这类旧环境不要升级到 `numpy>=1.24`，否则会出现依赖冲突。确认 `torch.cuda.is_available()` 输出为 `True` 后再开始训练。

如果 `cv2.VideoCapture(...).isOpened()` 对已有 mp4 返回 `False`，并且 `cv2.getBuildInformation()` 里显示 `FFMPEG: NO`，说明当前 Python 加载的 OpenCV 不支持 mp4 解码。先清理旧 OpenCV 包和残留目录，再安装 headless contrib 版本：

```bash
python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless

CV2_DIR=$(python - <<'PY'
import glob
paths = glob.glob("/usr/local/lib/python*/dist-packages/cv2") + glob.glob("/usr/local/lib/python*/site-packages/cv2")
print(paths[0] if paths else "")
PY
)
if [ -n "$CV2_DIR" ]; then
  rm -rf "$CV2_DIR"
fi

python -m pip install --no-cache-dir "numpy==1.22.4" "opencv-contrib-python-headless==4.10.0.84"
```

重新验证 OpenCV 视频解码和 TV-L1：

```bash
python - <<'PY'
import cv2

print("cv2:", cv2.__version__, cv2.__file__)
print("has tvl1:", hasattr(cv2, "optflow") or hasattr(cv2, "DualTVL1OpticalFlow_create"))
for line in cv2.getBuildInformation().splitlines():
    if "FFMPEG" in line or "GStreamer" in line:
        print(line)

p = "/root/shared-nvme/data/raw_videos/accordion/-DlGdZNAsxA.mp4"
cap = cv2.VideoCapture(p)
print("opened:", cap.isOpened())
print("frames:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("fps:", cap.get(cv2.CAP_PROP_FPS))
cap.release()
PY
```

后续命令统一使用云端数据根目录：

```bash
export DATA_ROOT=/root/shared-nvme/data
```

#### 1. 下载 MUSICES 视频

如果云端可以访问 YouTube，直接下载：

```bash
python main.py prepare-data -- download \
  --json "$DATA_ROOT/MUSICES.json" \
  --data-root "$DATA_ROOT" \
  --skip-existing
```

如果 YouTube 需要 cookies，把 Netscape 格式 cookies 文件上传到云端后运行：

```bash
python main.py prepare-data -- download \
  --json "$DATA_ROOT/MUSICES.json" \
  --data-root "$DATA_ROOT" \
  --skip-existing \
  --yt-dlp-extra-arg=--cookies \
  --yt-dlp-extra-arg=/absolute/path/to/youtube_cookies.txt
```

如果云端不能访问 YouTube，可以先把本地或其他机器下载好的视频目录上传到：

```text
/root/shared-nvme/data/raw_videos/<instrument>/<youtube_id>.mp4
```

然后从下一步开始。

#### 2. 生成 VIAI-A audio-only 样本

该命令只抽取音频并生成 Mel，不提取视频帧和光流：

```bash
python main.py prepare-viai-a -- \
  --json "$DATA_ROOT/MUSICES.json" \
  --data-root "$DATA_ROOT" \
  --skip-existing
```

默认参数已经对齐第一阶段论文设置：16kHz mono、80 Mel bins、STFT length 1280、hop size 320、125Hz-7.6kHz。输出文件位于：

```text
/root/shared-nvme/data/processed/<instrument>/<youtube_id>/source.wav
/root/shared-nvme/data/processed/<instrument>/<youtube_id>/raw_audio.npy
/root/shared-nvme/data/processed/<instrument>/<youtube_id>/mel.npy
```

#### 3. 生成 VIAI-A 训练/验证/测试划分

```bash
python main.py split-data -- --data-root "$DATA_ROOT" --audio-only
wc -l "$DATA_ROOT/train_viai_a_split.txt" "$DATA_ROOT/val_viai_a_split.txt" "$DATA_ROOT/test_viai_a_split.txt"
```

`--audio-only` 模式只要求样本中存在 `raw_audio.npy` 和 `mel.npy`。默认输出：

```text
/root/shared-nvme/data/train_viai_a_split.txt
/root/shared-nvme/data/val_viai_a_split.txt
/root/shared-nvme/data/test_viai_a_split.txt
```

#### 4. 云端训练 VIAI-A

先跑 1 step sanity check：

```bash
python main.py train-viai-a -- --data_root "$DATA_ROOT" --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
```

确认能保存 `checkpoints/VIAI-A_checkpoint_step000000001.pth.tar` 后，开始正式训练：

```bash
python main.py train-viai-a -- --data_root "$DATA_ROOT" --batch_size 16 --num_workers 4 --display_id 0 --checkpoint_interval 1000 --print_freq 100
```

训练时终端 `tqdm` 会实时显示 loss、full/missing PSNR、mask 长度，并按 `--metric_freq` 计算 SSIM。TensorBoard 默认写到 `checkpoints/events_viai_a`，包含 loss、PSNR、SSIM、mask 长度、learning rate 和 Mel 谱图对比图：

```bash
tensorboard --logdir checkpoints/events_viai_a --port 6006
# 端口转发
ssh -p 2233 -L 6006:localhost:6006 -l 'root@ackcs-00gjgrzt' ssh.bj8.bz1.paratera.com
```


常用监督频率参数：

```bash
python main.py train-viai-a -- --data_root "$DATA_ROOT" --batch_size 16 --num_workers 4 --metric_freq 100 --tb_image_freq 500 --tb_image_count 4
```

如果显存不足，优先降低 `--batch_size`，例如：

```bash
python main.py train-viai-a -- --data_root "$DATA_ROOT" --batch_size 8 --num_workers 4 --display_id 0 --checkpoint_interval 1000 --print_freq 100
```

VIAI-A checkpoint 命名格式为：

```text
checkpoints/VIAI-A_checkpoint_step*.pth.tar
```

### 5. 第二阶段：加入 PatchGAN

第二阶段在 8.1 的 VIAI-A audio-only 生成器基础上显式加入 PatchGAN。默认 `train-viai-a` 仍然是第一阶段 L1-only baseline；只有传入 `--use_gan` 时才会启用 `MelDiscriminator + GANLoss(use_lsgan=False)`，也只有这种模式保存的 checkpoint 会包含 PatchGAN 的 `netD/optimizer_D` 权重。开启后训练目标为：

```text
loss_recon = eta1(t) * full_l1 + missing_l1
loss_total = loss_g_gan + beta_gan * loss_recon
loss_d = 0.5 * (loss_d_real + loss_d_fake)
```

这里的 `--beta_gan` 是历史参数名；在 VIAI-A PatchGAN 中它对应论文第 4 页式 (3) 的 β，实际权重加在 reconstruction loss 上。

如果从第一阶段 checkpoint 热启动，建议重置 optimizer，让新的 GAN 目标从干净的优化器状态开始：

```bash
python main.py train-viai-a -- \
  --use_gan \
  --name VIAI-A-PatchGAN \
  --data_root "$DATA_ROOT" \
  --resume \
  --resume_path checkpoints/VIAI-A_checkpoint_step000000001.pth.tar \
  --reset_optimizer \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 2 \
  --display_id 0
```

正式训练示例：

```bash
python main.py train-viai-a -- \
  --use_gan \
  --name VIAI-A-PatchGAN \
  --data_root "$DATA_ROOT" \
  --resume \
  --resume_path checkpoints/VIAI-A_checkpoint_step000001000.pth.tar \
  --reset_optimizer \
  --batch_size 16 \
  --num_workers 4 \
  --beta_gan 0.1 \
  --checkpoint_interval 1000 \
  --print_freq 100 \
  --display_id 0
```

如果传 `--use_gan` 但不传 `--name`，脚本默认使用 `VIAI-A-PatchGAN`，避免覆盖第一阶段 `VIAI-A` checkpoint。TensorBoard 默认写到：

```text
checkpoints/events_viai_a_patchgan
```

第二阶段 checkpoint 命名格式为：

```text
checkpoints/VIAI-A-PatchGAN_checkpoint_step*.pth.tar
```

测试第二阶段 checkpoint 时也要传 `--use_gan`，这样 `loss_total` 会包含 GAN 项；如果不传，则只按生成器的 reconstruction/PSNR/SSIM 路径评估：

```bash
python main.py test-viai-a -- \
  --use_gan \
  --name VIAI-A-PatchGAN \
  --data_root "$DATA_ROOT" \
  --resume_path checkpoints/VIAI-A-PatchGAN_checkpoint_step000002000.pth.tar \
  --batch_size 16 \
  --num_workers 4 \
  --display_id 0 \
  --results_dir checkpoints/viai_a_patchgan_test_results
```

PatchGAN 测试 JSON/CSV 会额外记录：

```text
use_gan, loss_recon, loss_g_gan, loss_d, eta1, beta_gan, lambda_recon
```

其中 `lambda_recon` 仅作为历史配置字段保留在结果表中，VIAI-A PatchGAN 的 loss 计算不读取它。

#### 6. 云端测试 VIAI-A

测试指定 checkpoint：

```bash
python main.py test-viai-a -- --data_root "$DATA_ROOT" --resume_path checkpoints/VIAI-A_checkpoint_step000001000.pth.tar --batch_size 16 --num_workers 4 --display_id 0 --results_dir checkpoints/viai_a_test_results
```

如果不传 `--resume_path`，测试脚本会在 `checkpoints/` 下自动寻找最新的 `VIAI-A_checkpoint_step*.pth.tar`：

```bash
python main.py test-viai-a -- --data_root "$DATA_ROOT" --batch_size 16 --num_workers 4 --display_id 0
```

每次测试会把当前 checkpoint 的指标写入 JSON，并更新一个按 checkpoint step 去重排序的 CSV 总表：

```text
checkpoints/viai_a_test_results/VIAI-A_step000001000_test.json
checkpoints/viai_a_test_results/VIAI-A_test_summary.csv
checkpoints/viai_a_test_results/mel-image/step000001000/*.png
```

对 `1000/2000/.../6800` 等多个 checkpoint 逐个运行 `test-viai-a` 后，直接查看 `VIAI-A_test_summary.csv` 即可横向比较。
`mel-image/stepXXXXXXXXX/` 下会为每个测试样本保存一张 RGB 热力图四联图：masked、interpolated、prediction、groundtruth。

`test-viai-a` 会报告 normalized Mel `[0, 1]` 上的：

```text
mel_l1_full
mel_l1_missing
psnr_full
psnr_missing
ssim
```

注意：第一阶段只评估 Mel-spectrogram 修复质量，不生成 waveform，也不计算 SDR / OPS / MOS。

### 7. 第三阶段：加入视频分支 VIAI-AV

第三阶段对应 `information.md` 8.3，只加入视频动作条件分支：

```text
视频抽帧 -> TV-L1 光流 -> Image ResNet18 + Flow ResNet18 -> Efuse 时间融合 -> MelDecoderImage 融合解码
```

本阶段不加入 `contrastive sync loss`、`VIAI-AA' probe loss`、`η2(t)` 或 WaveNet，这些留到 8.4 及后续阶段。VIAI-AV 输入仍是 4 秒窗口：`80x200` Mel、50 帧 RGB、50 帧 2-channel flow。默认优先从第二阶段 `VIAI-A-PatchGAN` checkpoint 初始化音频侧权重；如果没有 PatchGAN checkpoint，会自动回退到最新的 `VIAI-A` audio-only checkpoint，并让 VIAI-AV 自己的 PatchGAN 判别器随机初始化后继续训练。视频分支始终随机初始化。

先生成包含 image/flow 的 AV 样本。该流程默认使用论文风格设置：裁掉视频前 6 秒、OpenCV shot detection 近似、在有效 shot 内切非重叠 4 秒 clip、每个 clip 保存 50 帧视觉输入、TV-L1 optical flow、`flow_clip=20`、motion crop、square padding：

```bash
python main.py prepare-data -- process \
  --json "$DATA_ROOT/MUSICES.json" \
  --data-root "$DATA_ROOT" \
  --skip-existing \
  --max-clips-per-video 5
```

这个步骤会比较慢，但现在只有 shot detection 仍需要扫描长视频；TV-L1、image/flow 写入和 Mel 生成都以 4 秒 clip 为单位执行，每个 clip 默认只计算 50 帧视觉输入。`--max-clips-per-video 5` 会先从每个源视频的所有有效 4 秒窗口中按固定 seed 抽样最多 5 个 clip，再计算 TV-L1，避免把长视频的所有窗口都展开。脚本默认显示单个视频/clip 内部进度条；如果在日志系统里不想显示进度条，可以加 `--no-progress`。正式复现建议保留默认 `--flow-method tvl1`、`--clip-duration-sec 4.0`、`--clip-hop-sec 4.0`、`--visual-frame-count 50`；只做数据链路 smoke test 时，可临时用 `--flow-method farneback` 提速，但这不再是论文默认设置。

`prepare-data -- process` 会跳过缺失视频，以及路径存在但 OpenCV/ffmpeg 无法打开的坏视频，并把这类 process 阶段问题写入：

```text
$DATA_ROOT/musices_process_failures.csv
```

如果遇到类似 `Unable to open video`，通常说明 mp4 文件存在但不可解码，常见原因是 YouTube 下载中断、下载失败后留下半成品、0 字节文件、权限/挂载异常或视频源失效。可先检查：

```bash
ls -lh "$DATA_ROOT/raw_videos/accordion/yy2vL2RUiPI.mp4"
file "$DATA_ROOT/raw_videos/accordion/yy2vL2RUiPI.mp4"
ffprobe -v error "$DATA_ROOT/raw_videos/accordion/yy2vL2RUiPI.mp4"
```

如果确认文件损坏，可以删除该 mp4 后重新运行下载；如果该 YouTube 视频已经不可下载，保留跳过即可，后续 `split-data` 只会收集成功处理出的样本。

确认 TV-L1 可用：

```bash
python -c "import cv2; print(hasattr(cv2, 'optflow') or hasattr(cv2, 'DualTVL1OpticalFlow_create'))"
```

生成 AV split。该模式会要求每个样本同时存在 `mel.npy`、`raw_audio.npy`、`image_crop/`、`flow_x_crop/`、`flow_y_crop/`：

```bash
python main.py split-data -- --data-root "$DATA_ROOT"
wc -l "$DATA_ROOT/train_new_split.txt" "$DATA_ROOT/val_new_split.txt" "$DATA_ROOT/test_new_split.txt"
```

`split-data` 会检查每个 AV 样本是否满足默认 50 帧视觉输入、200 帧 Mel 和 64000 audio steps；不满足的样本会被排除，并追加记录到：

```text
$DATA_ROOT/viai_av_bad_samples.csv
```

训练时也会对旧 split 中残留的坏样本做同样的记录和跳过。默认继续训练；如果需要恢复 fail-fast 行为，可以给 `train-viai-av` 加 `--strict_av_samples`。

本地单步 smoke test：

```bash
python main.py train-viai-av -- \
  --data_root "$DATA_ROOT" \
  --init_from_viai_a checkpoints/VIAI-A-PatchGAN_checkpoint_step000002000.pth.tar \
  --checkpoint_dir /tmp/viai_av_smoke \
  --log_event_path /tmp/viai_av_smoke/events \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 1 \
  --display_id 0 \
  --print_freq 1
```

正式训练示例：

```bash
python main.py train-viai-av -- \
  --data_root "$DATA_ROOT" \
  --init_from_viai_a checkpoints/VIAI-A-PatchGAN_checkpoint_step000002000.pth.tar \
  --batch_size 16 \
  --num_workers 4 \
  --beta_gan 0.1 \
  --checkpoint_interval 1000 \
  --print_freq 100 \
  --display_id 0
```

如果不传 `--init_from_viai_a`，`train-viai-av` 会在 `--checkpoint_dir` 下先寻找最新的 `VIAI-A-PatchGAN_checkpoint_step*.pth.tar`，找不到再寻找最新的 `VIAI-A_checkpoint_step*.pth.tar`。如果回退到 audio-only checkpoint，日志会提示源 checkpoint 没有 `netD`，VIAI-AV 的 `MelDiscriminator` 会随机初始化并在 AV 阶段重新训练；如果两类 checkpoint 都没有，脚本才会报错。VIAI-AV checkpoint 命名格式为：

```text
checkpoints/VIAI-AV_checkpoint_step*.pth.tar
```

继续训练指定 VIAI-AV checkpoint：

```bash
python main.py train-viai-av -- \
  --resume \
  --resume_path checkpoints/VIAI-AV_checkpoint_step000001000.pth.tar \
  --data_root "$DATA_ROOT" \
  --batch_size 16 \
  --num_workers 4 \
  --display_id 0
```

测试 VIAI-AV checkpoint：

```bash
python main.py test-viai-av -- \
  --resume_path checkpoints/VIAI-AV_checkpoint_step000001000.pth.tar \
  --data_root "$DATA_ROOT" \
  --batch_size 16 \
  --num_workers 4 \
  --display_id 0 \
  --results_dir checkpoints/viai_av_test_results
```

如果本地 smoke split 的 `test_new_split.txt` 为空，可以临时用训练 split 验证测试入口：

```bash
python main.py test-viai-av -- \
  --resume_path /tmp/viai_av_smoke/VIAI-AV_checkpoint_step000000001.pth.tar \
  --data_root "$DATA_ROOT" \
  --test_split_name train_new_split.txt \
  --batch_size 1 \
  --num_workers 0 \
  --display_id 0 \
  --results_dir /tmp/viai_av_smoke_results
```

测试会输出 normalized Mel `[0, 1]` 上的 L1、PSNR、SSIM、GAN/reconstruction loss，并写入：

```text
checkpoints/viai_av_test_results/VIAI-AV_stepXXXXXXXXX_test.json
checkpoints/viai_av_test_results/VIAI-AV_test_summary.csv
checkpoints/viai_av_test_results/mel-image/stepXXXXXXXXX/*.png
```

TensorBoard：

```bash
tensorboard --logdir checkpoints/events_viai_av --port 6006
```

### 8. 第四阶段：加入 sync loss 和 probe loss

第四阶段对应 `information.md` 8.4，在现有 `train-viai-av` / `test-viai-av` 链路中默认启用论文的 audio-video synchronization 和 VIAI-AA' probe branch，不新增入口。目标是避免模型只走 audio-only shortcut，强制视觉特征 `fv` 学到与 clean target audio bottleneck `fa_t = Ea(s_t)` 对齐的节奏信息：

```text
LSync: same-index audio/video feature 拉近，batch 内不同 index 作为负样本并推到 margin γ=1 外
VIAI-AA': s_aa' = Gav(Ea(s_i), Ea(s_t))
Ltotal = L_av_gen + λsync * LSync + λprobe * η2(t) * L_aa'_gen
η2(t) = max(floor, base ** (step / interval))
```

默认参数沿用论文风格：`--sync_margin 1.0`、`--lambda_sync 1.0`、`--lambda_probe 1.0`、`--probe_decay_base 0.9`、`--probe_decay_interval 1000`、`--probe_decay_floor 0.1`。如果没有显式传 `probe_decay_*`，会沿用已有 `sync_decay_*` 作为兼容默认值。sync loss 中的 clean audio target feature 会 `detach()`，所以该项只更新 `VideoEncoder`；probe loss 使用同一个 `MelDecoderImage` 和 `MelDiscriminator`，不新增网络参数。

第四阶段训练仍使用同一个命令：

```bash
python main.py train-viai-av -- \
  --data_root "$DATA_ROOT" \
  --init_from_viai_a checkpoints/VIAI-A-PatchGAN_checkpoint_step000002000.pth.tar \
  --batch_size 16 \
  --num_workers 4 \
  --beta_gan 0.1 \
  --checkpoint_interval 1000 \
  --print_freq 100 \
  --display_id 0
```

如需做 ablation 或临时退回第三阶段损失，可以关闭这两项：

```bash
python main.py train-viai-av -- \
  --data_root "$DATA_ROOT" \
  --init_from_viai_a checkpoints/VIAI-A-PatchGAN_checkpoint_step000002000.pth.tar \
  --disable_sync_loss \
  --disable_probe_loss
```

训练日志和 TensorBoard 会额外记录 `sync`、`probe`、`eta2`、`probe_full_l1`、`probe_missing_l1`、`probe_g_gan`。checkpoint 仍命名为：

```text
checkpoints/VIAI-AV_checkpoint_step*.pth.tar
```

但内部 `stage` 字段会写为 `VIAI-AV-stage4-sync-probe`，并保存 `enable_sync_loss` / `enable_probe_loss`。

测试入口仍是：

```bash
python main.py test-viai-av -- \
  --resume_path checkpoints/VIAI-AV_checkpoint_step000001000.pth.tar \
  --data_root "$DATA_ROOT" \
  --batch_size 16 \
  --num_workers 4 \
  --display_id 0 \
  --results_dir checkpoints/viai_av_test_results
```

测试 JSON/CSV 除原有 L1、PSNR、SSIM、GAN/reconstruction loss 外，还会写入 sync/probe loss，以及 audio→video、video→audio retrieval 的 `R@1/R@5/R@10/R@50/MedR/MeanR`。正式评估建议 `batch_size >= 16`，否则 batch 内负样本和 retrieval 指标统计意义有限。

下面按“运行时序”给你梳理第三阶段 VIAI-AV 的调用链。可以把它理解成：

`prepare-data -> split-data -> train-viai-av -> test-viai-av`

核心变化是：在原 VIAI-A 的 Mel 修复链路上，多接入 `RGB frames + optical flow -> VideoEncoder -> MelDecoderImage`。

**1. 入口分发**
命令从 [main.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/main.py:15) 进入：

```text
python main.py train-viai-av -> train_viai_av.py
python main.py test-viai-av  -> test_viai_av.py
python main.py prepare-data  -> tools.prepare_musices
```

README 第三阶段命令集中写在 [README.md](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/README.md:472)，修改记录对应 [MODIFICATION_LOG.md](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/logmd/MODIFICATION_LOG.md:995)。

**2. 数据准备链路**
数据准备主要在 [tools/prepare_musices.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/tools/prepare_musices.py:1476)：

```text
prepare_musices.main()
  -> process_record()
    -> detect_video_segments()
    -> iter_clip_windows()
    -> process_clip()
      -> extract_audio_from_video()
      -> black_frame_ratio()
      -> extract_frames_and_flow()
      -> crop_motion_region()
      -> export_audio_and_mel()
```

每个 AV clip 最终目录里会有：

```text
raw_audio.npy
mel.npy
image_crop/*.jpg
flow_x_crop/*.jpg
flow_y_crop/*.jpg
```

关键处理点：

- [extract_frames_and_flow()](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/tools/prepare_musices.py:1184)：从 4 秒 clip 中抽视觉帧，写 `image/flow_x/flow_y`。
- [crop_motion_region()](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/tools/prepare_musices.py:1364)：根据光流找运动区域，写 `image_crop/flow_x_crop/flow_y_crop`。
- `split-data` 后生成 `train_new_split.txt / val_new_split.txt / test_new_split.txt`，每行大致是：

```text
sample_dir|mel.npy|raw_audio.npy|mel_frames
```

**3. 训练入口链路**
训练从 [train_viai_av.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/train_viai_av.py:312) 的 `main()` 开始：

```text
train_viai_av.main()
  -> configure_viai_av_defaults()
  -> av_loader.get_data_loaders(..., phases=("train", "val"))
  -> VIAIAVModel(...)
  -> load_checkpoint() 或 load_viai_a_checkpoint()
  -> train_loop()
    -> run_phase("train")
      -> model.get_blank_space_length()
      -> model.set_inputs()
      -> model.optimize_parameters()
      -> compute_viai_a_metrics()
      -> model.TF_writer()
      -> write_mel_images()
      -> model.save_checkpoint()
```

如果不是 resume，会先从 VIAI-A 初始化音频侧权重，位置在 [VIAI_AV_inpainting.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Models/VIAI_AV_inpainting.py:253)：

```text
load_viai_a_checkpoint()
  -> copy Mel_Encoder
  -> copy Mel_Decoder 到 MelDecoderImage 可匹配层
  -> init_deconv_1_1_1()
  -> copy netD，如果源 checkpoint 有 PatchGAN
```

视频分支 `VideoEncoder` 始终随机初始化。

**4. Dataloader 调用链**
训练和测试都复用 [Data_loaders/audio_loader.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Data_loaders/audio_loader.py:739)：

```text
get_data_loaders()
  -> RawAudioDataSource
  -> MelSpecDataSource
  -> ImageSpecDataSource
  -> PyTorchImageDataset
  -> DataLoader(..., collate_fn=collate_fn)
```

单样本读取在 [PyTorchImageDataset.__getitem__](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Data_loaders/audio_loader.py:519)：

```text
raw_audio = raw_audio.npy
mel = mel.npy
video_block, flow_block, start, path = Image[idx]
```

视觉读取在 [sample_data_new()](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Data_loaders/audio_loader.py:222)：

```text
validate_av_sample()
  -> 选一个对齐的 image_start
  -> 读取 image_crop / flow_x_crop / flow_y_crop
  -> resize / crop / flip
  -> normalize 到约 [-1, 1]
  -> transpose 成 channel-first
```

batch 对齐在 [collate_fn()](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Data_loaders/audio_loader.py:558)：

```text
video start
  -> mel_start = start * mel_frames_per_visual_frame
  -> mel_end = mel_start + 200
  -> audio_start = mel_start * hop_size
  -> audio_end = audio_start + 64000
```

最终交给模型的是 8 元组：

```text
video_batch, flow_batch, c_batch, x_batch, y_batch, g_batch, input_lengths, path_batch
```

典型形状：

```text
video_batch: [B, 50, 3, 256, 256]
flow_batch : [B, 50, 2, 256, 256]
c_batch    : [B, 80, 200]
x_batch    : [B, 1, 64000]
```

坏样本检查在 [utils/av_sample_validation.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/utils/av_sample_validation.py:137)，不合格样本会被跳过并记录到 `viai_av_bad_samples.csv`。

**5. 模型前向链路**
模型定义在 [Models/VIAI_AV_inpainting.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Models/VIAI_AV_inpainting.py:33)。

初始化时组装四块：

```text
Mel_Encoder  = Inpainting_Networks.MelEncoder
VideoEncoder = Image_Embedding.ImageEmbedding
Mel_Decoder  = New_Inpainting_Networks.MelDecoderImage
netD         = Discriminator_Networks.MelDiscriminator
```

每个 batch 进入 [set_inputs()](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Models/VIAI_AV_inpainting.py:91)：

```text
set_inputs(data)
  -> 保存 video / flow / mel / audio
  -> corrupt_mel_spectrogram()
    -> 随机遮挡 20~50 个 Mel frames
    -> 缺失区域用左右边界插值初始化
    -> 生成 missing_mask
```

真正前向在 [_forward_inpainter()](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Models/VIAI_AV_inpainting.py:123)：

```text
mel_input
  -> MelEncoder
  -> mel_features

video_batch + flow_batch
  -> VideoEncoder
  -> video_feature

mel_features + video_feature
  -> MelDecoderImage
  -> mel_pred
```

展开后是：

```text
MelEncoder.forward()
  input:  [B, 80, 200]
  output: 多尺度 Mel feature list，最后一层约 [B, 256, 1, 13]

ImageEmbedding.forward()
  RGB  -> ResNet18
  Flow -> ResNet18
  concat
  Conv1d stride=2
  Conv1d stride=2
  output: [B, length_feature, 1, 13]

MelDecoderImage.forward()
  concat Mel 最深层 + video_feature
  deconv / upsample / skip connection
  sigmoid
  output mel_pred: [B, 1, 80, 200]
```

对应文件：

- Mel encoder: [networks/Inpainting_Networks.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/networks/Inpainting_Networks.py:49)
- Video encoder: [networks/Image_Embedding.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/networks/Image_Embedding.py:99)
- AV decoder: [networks/New_Inpainting_Networks.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/networks/New_Inpainting_Networks.py:92)
- PatchGAN discriminator: [networks/Discriminator_Networks.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/networks/Discriminator_Networks.py:5)

**6. 损失与优化链路**
训练优化在 [optimize_parameters()](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Models/VIAI_AV_inpainting.py:157)：

```text
optimize_parameters()
  -> _forward_inpainter()
  -> _compute_losses()
    -> full L1
    -> missing-region L1
    -> eta1 * full_l1 + missing_l1
    -> netD(mel_pred)
    -> GAN generator loss
    -> loss_total = loss_G_GAN + beta_gan * loss_recon
  -> update generator
  -> _compute_discriminator_loss()
    -> netD(real mel)
    -> netD(fake mel.detach())
  -> update discriminator
```

注意这里第三阶段默认 `use_gan=True`，也就是 VIAI-AV 直接带 PatchGAN。

**7. 测试链路**
测试入口在 [test_viai_av.py](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/test_viai_av.py:272)：

```text
test_viai_av.main()
  -> get_data_loaders(..., phases=("test",))
  -> VIAIAVModel()
  -> load_checkpoint()
  -> evaluate()
    -> model.get_blank_space_length(0)
    -> model.set_inputs()
    -> model.test()
    -> compute_viai_a_metrics()
    -> save_mel_comparison_batch()
  -> write_result_files()
```

测试不会反传，只在 [VIAIAVModel.test()](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/Models/VIAI_AV_inpainting.py:175) 中 `no_grad()` 前向并计算 loss。结果写到：

```text
VIAI-AV_stepXXXXXXXXX_test.json
VIAI-AV_test_summary.csv
mel-image/stepXXXXXXXXX/*.png
```

一句话总结：第三阶段的主链路是 `AV clip 数据 -> audio_loader 对齐 50 帧视觉和 200 帧 Mel -> VIAIAVModel 遮挡 Mel -> MelEncoder + VideoEncoder -> MelDecoderImage 融合修复 -> L1 + PatchGAN 训练/测试`。

## License and Citation
The use of this software is RESTRICTED to **non-commercial research and educational purposes**.

```
@InProceedings{Zhou_2019_ICCV,
  author = {Zhou, Hang and Liu, Ziwei and Xu, Xudong and Luo, Ping and Wang, Xiaogang},
  title = {Vision-Infused Deep Audio Inpainting},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2019}
} 
```

## Acknowledgement
The structure of this codebase is borrowed from [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [wavent_vocoder](https://github.com/r9y9/wavenet_vocoder).





## operation
代码上穿到云端
```
rsync -avz --progress \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='data/' \
  --exclude='checkpoints/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.pth.tar' \
  --exclude='*.Zone.Identifier' \
  --exclude='.agents/' \
  --exclude='.codex' \
  --exclude='.python-version' \
  --exclude='NUL' \
  --exclude='uv.lock' \
  --exclude='VIAI.pdf' \
  -e "ssh -p 2233 -l 'root@ackcs-00gjgrzt'" \
  /home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/ \
  ssh.bj8.bz1.paratera.com:/root/Vision-Infused-Audio-Inpainter-VIAI/
```
