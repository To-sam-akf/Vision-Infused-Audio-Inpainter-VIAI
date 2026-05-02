# Vision-Infused Deep Audio Inpainting

We present a vision-infused method that can deal with both audio-only and audio-visual associated inpainting Inspired by image inpainting, called `Vision-Infused Audio Inpainter (VIAI)`.

[[Project]](https://hangz-nju-cuhk.github.io/projects/AudioInpainting) [[Paper]](https://arxiv.org/abs/1910.10997) [[Demo]](https://www.youtube.com/watch?v=2C8s_YuRRxk)

<img src='./misc/pipeline2.png' width=880>

## Requirements
* [python 3](https://www.python.org/download/releases/3.6/)
* [PyTorch](https://pytorch.org/)（cloud training should install the CUDA build matching the target GPU)
* [opencv-contrib-python](https://pypi.org/project/opencv-contrib-python/) for TV-L1 optical flow

## Dataset

The MUSICES dataset can be accessed [here](https://hangz-nju-cuhk.github.io/projects/audio-inpainting/MUSICES.json).

### Data Preparation And Dataset Split

The paper preprocesses MUSICES videos into 16kHz audio, 80-bin Mel-spectrograms, TV-L1 optical flow clipped to 20 pixels, and motion-salient visual crops. It first keeps a fixed video-level test split, then derives train/validation/test samples from the corresponding videos. Public `MUSICES.json` contains video IDs but no shot boundary annotations, so this repository uses an OpenCV frame-difference heuristic as a reproducible approximation of the paper's shot detection step.

Recommended environment setup:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra local-cuda
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python -c "import torch, cv2, librosa, nnmnkwii, tensorboardX, tqdm; from skimage.metrics import structural_similarity; print(torch.__version__, torch.cuda.is_available(), hasattr(cv2, 'optflow'))"
```

1. Prepare manifests and inspect download size:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- manifest --json data/MUSICES.json --data-root data
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- stats --json data/MUSICES.json --data-root data
```

2. Download raw videos:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- download --json data/MUSICES.json --data-root data --skip-existing
```

If YouTube requires login, export a fresh Netscape-format cookie file and pass it through `yt-dlp`:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- download \
  --json data/MUSICES.json \
  --data-root data \
  --skip-existing \
  --yt-dlp-extra-arg=--cookies \
  --yt-dlp-extra-arg=/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/data/youtube_cookies.txt
```

3. Process videos into paper-style shot samples:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- process --json data/MUSICES.json --data-root data --skip-existing
```

Default processing trims the first 6 seconds, detects shot-like segments, skips segments shorter than 4 seconds, skips mostly black or near-silent segments, extracts TV-L1 optical flow, crops motion-salient regions, and writes samples under:

```text
data/processed/<instrument>/<youtube_id>/shot_000000/
data/processed/<instrument>/<youtube_id>/shot_000001/
```

For a local smoke test over a subset, use a larger `--max-videos` than 1 because early `MUSICES.json` entries may not exist locally:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- process --json data/MUSICES.json --data-root data --max-videos 100 --skip-existing
```

To reproduce the old whole-video processing mode, disable shot detection explicitly:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- process --json data/MUSICES.json --data-root data --skip-existing --no-shot-detection --trim-start-sec 0
```

4. Split processed samples without video leakage:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py split-data -- --data-root data
```

`split-data` groups samples by source video key `<instrument>/<youtube_id>` before splitting, so shots from the same original video cannot land in different phases. Defaults follow the paper protocol: 85% train, 5% validation, 10% test.

For a one-sample smoke split:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py split-data -- --data-root data --max-samples 1 --test-size 0 --val-size 0 --allow-empty-eval
wc -l data/train_new_split.txt data/val_new_split.txt data/test_new_split.txt
sed -n '1,5p' data/train_new_split.txt
```

The full preparation flow can also be run as:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- all --json data/MUSICES.json --data-root data --skip-existing
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py split-data -- --data-root data
```

The pipeline will create `data/raw_videos/<instrument>/<youtube_id>.mp4`, shot-level folders under `data/processed/`, and `data/train_new_split.txt`, `data/val_new_split.txt`, `data/test_new_split.txt`. Each processed sample contains `raw_audio.npy`, `mel.npy`, `image/`, `flow_x/`, `flow_y/`, `image_crop/`, `flow_x_crop/`, and `flow_y_crop/`.

<img src='./misc/datastatistic.png' width=880>

### 第一阶段：VIAI-A Audio-Only

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
python -m pip install imageio-ffmpeg librosa nnmnkwii numpy opencv-contrib-python pillow scikit-image tensorboard tensorboardX tqdm "yt-dlp[default]"
python -c "import torch, librosa, cv2, nnmnkwii, tensorboardX, tqdm; from skimage.metrics import structural_similarity; print(torch.__version__, torch.cuda.is_available())"
```

如果云端已经通过 `conda` 或平台镜像安装了部分依赖，可以只安装缺失项。确认 `torch.cuda.is_available()` 输出为 `True` 后再开始训练。

#### 1. 下载 MUSICES 视频

如果云端可以访问 YouTube，直接下载：

```bash
python main.py prepare-data -- download --skip-existing
```

如果 YouTube 需要 cookies，把 Netscape 格式 cookies 文件上传到云端后运行：

```bash
python main.py prepare-data -- download \
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
python main.py prepare-viai-a -- --skip-existing
```

默认参数已经对齐第一阶段论文设置：16kHz mono、80 Mel bins、STFT length 1280、hop size 320、125Hz-7.6kHz。输出文件位于：

```text
/root/shared-nvme/data/processed/<instrument>/<youtube_id>/source.wav
/root/shared-nvme/data/processed/<instrument>/<youtube_id>/raw_audio.npy
/root/shared-nvme/data/processed/<instrument>/<youtube_id>/mel.npy
```

#### 3. 生成 VIAI-A 训练/验证/测试划分

```bash
python main.py split-data -- --audio-only
wc -l /root/shared-nvme/data/train_viai_a_split.txt /root/shared-nvme/data/val_viai_a_split.txt /root/shared-nvme/data/test_viai_a_split.txt
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
python main.py train-viai-a -- --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
```

确认能保存 `checkpoints/VIAI-A_checkpoint_step000000001.pth.tar` 后，开始正式训练：

```bash
python main.py train-viai-a -- --batch_size 16 --num_workers 4 --display_id 0 --checkpoint_interval 1000 --print_freq 100
```

训练时终端 `tqdm` 会实时显示 loss、full/missing PSNR、mask 长度，并按 `--metric_freq` 计算 SSIM。TensorBoard 默认写到 `checkpoints/events_viai_a`，包含 loss、PSNR、SSIM、mask 长度、learning rate 和 Mel 谱图对比图：

```bash
tensorboard --logdir checkpoints/events_viai_a
```

常用监督频率参数：

```bash
python main.py train-viai-a -- --batch_size 16 --num_workers 4 --metric_freq 100 --tb_image_freq 500 --tb_image_count 4
```

如果显存不足，优先降低 `--batch_size`，例如：

```bash
python main.py train-viai-a -- --batch_size 8 --num_workers 4 --display_id 0 --checkpoint_interval 1000 --print_freq 100
```

VIAI-A checkpoint 命名格式为：

```text
checkpoints/VIAI-A_checkpoint_step*.pth.tar
```

#### 5. 云端测试 VIAI-A

测试指定 checkpoint：

```bash
python main.py test-viai-a -- --resume_path checkpoints/VIAI-A_checkpoint_step000001000.pth.tar --batch_size 16 --num_workers 4 --display_id 0
```

如果不传 `--resume_path`，测试脚本会在 `checkpoints/` 下自动寻找最新的 `VIAI-A_checkpoint_step*.pth.tar`：

```bash
python main.py test-viai-a -- --batch_size 16 --num_workers 4 --display_id 0
```

`test-viai-a` 会报告 normalized Mel `[0, 1]` 上的：

```text
mel_l1_full
mel_l1_missing
psnr_full
psnr_missing
ssim
```

注意：第一阶段只评估 Mel-spectrogram 修复质量，不生成 waveform，也不计算 SDR / OPS / MOS。

## Training And Testing

Local smoke tests only verify the pipeline, dataloader, forward pass, backward pass, and one optimizer update. They are not paper metrics.

Train smoke test:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py train -- --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
```

Full VIAI-AV training should use the paper-style 4-second input, 80x200 Mel-spectrogram, 50-frame visual window, TV-L1 optical flow, contrastive margin `gamma=1`, Adam learning rate `1e-4`, and batch size 16 when GPU memory allows:

```bash
nvidia-smi
UV_CACHE_DIR=/tmp/uv-cache uv sync
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -c "import torch, cv2; print(torch.cuda.is_available(), torch.version.cuda, hasattr(cv2, 'optflow'))"
.venv/bin/python main.py prepare-data -- all --json data/MUSICES.json --data-root data --skip-existing
.venv/bin/python main.py split-data -- --data-root data
.venv/bin/python main.py train -- --batch_size 16 --num_workers 4 --display_id 0
```

The `cu121` command is only an example; install the PyTorch wheel that matches the target server driver/CUDA runtime. After manually installing cloud-specific PyTorch, prefer `.venv/bin/python` or `uv run --no-sync` so `uv` does not replace it.

Test-set evaluation:

```bash
.venv/bin/python main.py test -- --resume_path checkpoints/VIAI-AV_checkpoint_step000001000.pth.tar --batch_size 16 --num_workers 4 --display_id 0
```

If you pass a placeholder path such as `checkpoints/VIAI-AV/latest.pth` and it does not exist, the test runner will search the same directory and `--checkpoint_dir` for the newest `VIAI-AV_checkpoint_step*.pth.tar` checkpoint.

Current `main.py test` reports reconstruction loss, Mel L1 loss, sync loss, and audio-video retrieval metrics on `test_new_split.txt`. Full paper audio metrics such as SDR and OPS still require the WaveNet/audio-generation evaluation path.


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
