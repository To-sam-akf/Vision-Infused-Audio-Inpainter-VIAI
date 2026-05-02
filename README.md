# Vision-Infused Deep Audio Inpainting

We present a vision-infused method that can deal with both audio-only and audio-visual associated inpainting Inspired by image inpainting, called `Vision-Infused Audio Inpainter (VIAI)`.

[[Project]](https://hangz-nju-cuhk.github.io/projects/AudioInpainting) [[Paper]](https://arxiv.org/abs/1910.10997) [[Demo]](https://www.youtube.com/watch?v=2C8s_YuRRxk)

<img src='./misc/pipeline2.png' width=880>

## Requirements
* [python 3](https://www.python.org/download/releases/3.6/)
* [PyTorch](https://pytorch.org/)（cloud training should install the CUDA build matching the target GPU)
* [opencv-contrib-python](https://pypi.org/project/opencv-contrib-python/) for TV-L1 optical flow

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
tensorboard --logdir checkpoints/events_viai_a --port 6006
# 端口转发
ssh -p 2233 -L 6006:localhost:6006 -l 'root@ackcs-00gjgrzt' ssh.bj8.bz1.paratera.com
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
python main.py test-viai-a -- --resume_path checkpoints/VIAI-A_checkpoint_step000001000.pth.tar --batch_size 16 --num_workers 4 --display_id 0 --results_dir checkpoints/viai_a_test_results
```

如果不传 `--resume_path`，测试脚本会在 `checkpoints/` 下自动寻找最新的 `VIAI-A_checkpoint_step*.pth.tar`：

```bash
python main.py test-viai-a -- --batch_size 16 --num_workers 4 --display_id 0
```

每次测试会把当前 checkpoint 的指标写入 JSON，并更新一个按 checkpoint step 去重排序的 CSV 总表：

```text
checkpoints/viai_a_test_results/VIAI-A_step000001000_test.json
checkpoints/viai_a_test_results/VIAI-A_test_summary.csv
checkpoints/viai_a_test_results/mel-image/step000001000/*.png
```

对 `1000/2000/.../6800` 等多个 checkpoint 逐个运行 `test-viai-a` 后，直接查看 `VIAI-A_test_summary.csv` 即可横向比较。
`mel-image/stepXXXXXXXXX/` 下会为每个测试样本保存一张四联图：masked input、prediction、target、abs error。

`test-viai-a` 会报告 normalized Mel `[0, 1]` 上的：

```text
mel_l1_full
mel_l1_missing
psnr_full
psnr_missing
ssim
```

注意：第一阶段只评估 Mel-spectrogram 修复质量，不生成 waveform，也不计算 SDR / OPS / MOS。



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
