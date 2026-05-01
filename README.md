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

### Data Preparation

This repository now includes a local data-preparation pipeline that starts from
`MUSICES.json` and converts the dataset into the directory structure expected by
the training loader.

Recommended setup uses `uv`:

```bash
uv sync
uv run python -c "import numpy, librosa, cv2, yt_dlp, imageio_ffmpeg"
```

The preparation pipeline now supports:
* `stats`: estimate raw download size before fetching videos
* live `yt-dlp` progress output during download
* resumable downloads through `.part` files and a download archive
* `imageio-ffmpeg` fallback when system `ffmpeg` is not installed

Suggested workflow:

```bash
uv run python main.py prepare-data -- manifest --json data/MUSICES.json --data-root data
uv run python main.py prepare-data -- stats --json data/MUSICES.json --data-root data
uv run python main.py prepare-data -- download --json data/MUSICES.json --data-root data --skip-existing
uv run python main.py prepare-data -- process --json data/MUSICES.json --data-root data --skip-existing
uv run python main.py prepare-data -- splits --json data/MUSICES.json --data-root data
```

Or run the full pipeline:

```bash
uv run python main.py prepare-data -- all --json data/MUSICES.json --data-root data --skip-existing
```

Notes:
* `--skip-existing` skips already downloaded `.mp4` files and already processed samples.
* If you already have local videos and want to bypass `yt-dlp` entirely, run `process` directly or use `--skip-download` with `all`.
* Download stats are estimates from `yt-dlp` metadata (`filesize` / `filesize_approx`), so totals may change over time.
* If YouTube asks you to sign in, the recommended path is to export a fresh Netscape-format `youtube_cookies.txt` from a private/incognito browser session and pass it with `--cookies`.
* For Windows/WSL, manual export is more reliable than `--cookies-from-browser` with Edge. `tools/export_windows_edge_cookies.sh` is kept only as a best-effort backup helper.
* YouTube downloads also require a JavaScript runtime for challenge solving. The script defaults to `--yt-dlp-js-runtime auto` and will use `deno`, `node`, `bun`, or `qjs` if found in `PATH`.
* If your workspace disk is tight, you can keep manifests and processed data under `data/` while sending raw videos to another drive with `--video-root`, for example `--video-root /mnt/e/raw_videos`.

Example:

```bash
uv run python main.py prepare-data -- download \
  --json data/MUSICES.json \
  --data-root data \
  --skip-existing \
  --max-videos 1 \
  --abort-on-download-error \
  --yt-dlp-extra-arg=--cookies \
  --yt-dlp-extra-arg=/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/data/youtube_cookies.txt
```

Example with raw videos on `E:` while everything else stays in the repo:

```bash
uv run python main.py prepare-data -- download \
  --json data/MUSICES.json \
  --data-root data \
  --video-root /mnt/e/raw_videos \
  --skip-existing \
  --yt-dlp-extra-arg=--cookies \
  --yt-dlp-extra-arg=/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/data/youtube_cookies.txt
```

The pipeline will create:
* `data/musices_manifest.csv`
* `data/musices_download_stats.json`
* `data/musices_download_stats.csv`
* `data/musices_downloaded.txt`
* `data/musices_download_failures.csv`
* `data/raw_videos/<instrument>/<youtube_id>.mp4`
* `data/processed/<instrument>/<youtube_id>/`
* `data/train_new_split.txt`
* `data/test_new_split.txt`

Each processed sample contains:
* `raw_audio.npy`
* `mel.npy`
* `image/`
* `flow_x/`
* `flow_y/`
* `image_crop/`
* `flow_x_crop/`
* `flow_y_crop/`

<img src='./misc/datastatistic.png' width=880>

## Training and Testing

This repository now separates local smoke tests from full training. A local RTX 3060-class GPU is intended only to verify that preprocessing, dataloading, model forward, and one backward/update step run. Do not treat local smoke-test settings as paper-reproduction settings.

Local smoke test example:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra local-cuda
uv run --extra local-cuda python main.py --help
uv run --extra local-cuda python main.py prepare-data -- --help
uv run --extra local-cuda python -c "import torch, cv2, librosa, nnmnkwii, tensorboardX; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
uv run --extra local-cuda python main.py prepare-data -- process --json data/MUSICES.json --data-root data --max-videos 1 --skip-existing
uv run --extra local-cuda python main.py prepare-data -- splits --json data/MUSICES.json --data-root data --max-videos 1
uv run --extra local-cuda python main.py train -- --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
```

If local memory is still tight, temporarily lower `--image_size` or `--load_num` for smoke tests only, and keep that setting out of paper metrics.

Cloud training checklist:

```bash
nvidia-smi
UV_CACHE_DIR=/tmp/uv-cache uv sync
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -c "import torch, cv2; print(torch.cuda.is_available(), torch.version.cuda, hasattr(cv2, 'optflow'))"
.venv/bin/python main.py prepare-data -- all --json data/MUSICES.json --data-root data --skip-existing
.venv/bin/python main.py train -- --batch_size 16 --num_workers 4 --display_id 0
```

For cloud runs, install the PyTorch wheel that matches the server driver/CUDA runtime; the `cu121` command above is only an example. Use `.venv/bin/python` or `uv run --no-sync` after manual PyTorch installation so `uv` does not remove the cloud-specific torch package. Use the default 4-second audio window, 80x200 Mel-spectrogram, 50 video-frame mapping, TV-L1 optical flow, contrastive margin `gamma=1`, and TensorBoard/checkpoint monitoring. Adjust batch size only when GPU memory requires it.


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
