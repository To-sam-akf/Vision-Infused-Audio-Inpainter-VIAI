# Vision-Infused Deep Audio Inpainting

We present a vision-infused method that can deal with both audio-only and audio-visual associated inpainting Inspired by image inpainting, called `Vision-Infused Audio Inpainter (VIAI)`.

[[Project]](https://hangz-nju-cuhk.github.io/projects/AudioInpainting) [[Paper]](https://arxiv.org/abs/1910.10997) [[Demo]](https://www.youtube.com/watch?v=2C8s_YuRRxk)

<img src='./misc/pipeline2.png' width=880>

## Requirements
* [python 3](https://www.python.org/download/releases/3.6/)
* [PyTorch](https://pytorch.org/)（version >= 0.4.1)
* [opencv3](https://opencv.org/releases.html)

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
* If YouTube asks you to sign in, forward cookies or runtime flags to `yt-dlp`, for example:

```bash
uv run python main.py prepare-data -- download \
  --json data/MUSICES.json \
  --data-root data \
  --yt-dlp-extra-arg=--cookies-from-browser \
  --yt-dlp-extra-arg=firefox
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

We are still sorting out the code. For now it is not complete thus not runable, but the architecture is revealed.
Please wait for more details.


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
