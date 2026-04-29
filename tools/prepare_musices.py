import argparse
import csv
import importlib.util
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path


YT_DLP_FORMAT = "mp4/bestvideo+bestaudio/best"
DOWNLOAD_PROGRESS_TEMPLATE = (
    "[download] %(progress._percent_str)s of %(progress._total_bytes_str)s "
    "at %(progress._speed_str)s ETA %(progress._eta_str)s"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare MUSICES data from MUSICES.json into VIAI training format."
    )
    parser.add_argument(
        "action",
        choices=["manifest", "stats", "download", "process", "splits", "all"],
        help="Preparation stage to run.",
    )
    parser.add_argument("--json", dest="json_path", default="data/MUSICES.json")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--video-dir", default="raw_videos")
    parser.add_argument("--processed-dir", default="processed")
    parser.add_argument("--manifest-name", default="musices_manifest.csv")
    parser.add_argument("--train-split-name", default="train_new_split.txt")
    parser.add_argument("--test-split-name", default="test_new_split.txt")
    parser.add_argument("--download-archive-name", default="musices_downloaded.txt")
    parser.add_argument("--stats-json-name", default="musices_download_stats.json")
    parser.add_argument("--stats-csv-name", default="musices_download_stats.csv")
    parser.add_argument("--test-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--fft-size", type=int, default=1280)
    parser.add_argument("--hop-size", type=int, default=320)
    parser.add_argument("--num-mels", type=int, default=80)
    parser.add_argument("--fmin", type=float, default=125.0)
    parser.add_argument("--fmax", type=float, default=7600.0)
    parser.add_argument("--min-level-db", type=float, default=-100.0)
    parser.add_argument("--ref-level-db", type=float, default=20.0)
    parser.add_argument("--frame-size", type=int, default=256)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--flow-clip", type=float, default=20.0)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--abort-on-download-error", action="store_true")
    parser.add_argument("--yt-dlp-bin", default="yt-dlp")
    parser.add_argument(
        "--yt-dlp-extra-arg",
        action="append",
        default=[],
        help="Repeat to forward extra arguments to yt-dlp, for example "
        "`--yt-dlp-extra-arg=--cookies-from-browser --yt-dlp-extra-arg=firefox`.",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    return parser.parse_args()


def require_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is required for frame and optical-flow extraction. "
            "Install it with `uv add opencv-python && uv sync`."
        ) from exc
    return cv2


def require_librosa():
    try:
        import librosa  # type: ignore
        import librosa.filters  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "librosa is required for audio and mel-spectrogram extraction. "
            "Install it with `uv add librosa && uv sync`."
        ) from exc
    return librosa


def require_numpy():
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "numpy is required for dataset preparation. "
            "Install it with `uv add numpy && uv sync`."
        ) from exc
    return np


def require_imageio_ffmpeg():
    try:
        import imageio_ffmpeg  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "imageio-ffmpeg is required when system ffmpeg is unavailable. "
            "Install it with `uv add imageio-ffmpeg && uv sync`."
        ) from exc
    return imageio_ffmpeg


def load_records(json_path, max_videos=None):
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    videos = payload["videos"]
    records = []
    for instrument in sorted(videos.keys()):
        for youtube_id in videos[instrument]:
            records.append(
                {
                    "instrument": instrument,
                    "youtube_id": youtube_id,
                    "sample_key": f"{instrument}/{youtube_id}",
                }
            )
    if max_videos is not None:
        records = records[:max_videos]
    return records


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def manifest_path(data_root, manifest_name):
    return Path(data_root) / manifest_name


def stats_json_path(data_root, stats_json_name):
    return Path(data_root) / stats_json_name


def stats_csv_path(data_root, stats_csv_name):
    return Path(data_root) / stats_csv_name


def download_archive_path(data_root, download_archive_name):
    return Path(data_root) / download_archive_name


def download_failure_path(data_root):
    return Path(data_root) / "musices_download_failures.csv"


def video_output_path(data_root, video_dir, record):
    return Path(data_root) / video_dir / record["instrument"] / f'{record["youtube_id"]}.mp4'


def sample_output_dir(data_root, processed_dir, record):
    return Path(data_root) / processed_dir / record["instrument"] / record["youtube_id"]


def video_url(record):
    return f'https://www.youtube.com/watch?v={record["youtube_id"]}'


def format_bytes(value):
    if value is None:
        return "unknown"
    if value < 1024:
        return f"{value} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        size /= 1024.0
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
    return f"{size:.2f} TiB"


def write_manifest(records, data_root, video_dir, processed_dir, manifest_name):
    output = manifest_path(data_root, manifest_name)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "instrument",
                "youtube_id",
                "sample_key",
                "video_path",
                "sample_dir",
            ],
        )
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["video_path"] = str(video_output_path(data_root, video_dir, record).relative_to(data_root))
            row["sample_dir"] = str(sample_output_dir(data_root, processed_dir, record).relative_to(data_root))
            writer.writerow(row)
    return output


def read_manifest(data_root, manifest_name):
    output = manifest_path(data_root, manifest_name)
    with output.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_command_path(name):
    path = shutil.which(name)
    if path:
        return path
    candidate = Path(name).expanduser()
    if candidate.exists():
        return str(candidate)
    return None


def resolve_yt_dlp_command(preferred):
    preferred_path = resolve_command_path(preferred)
    if preferred_path:
        return [preferred_path]
    if preferred != "yt-dlp":
        raise RuntimeError(
            f"Unable to resolve yt-dlp from `--yt-dlp-bin={preferred}`. "
            "Install it with `uv add \"yt-dlp[default]\" && uv sync`, "
            "or pass the correct executable path."
        )
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    raise RuntimeError(
        "yt-dlp is required for MUSICES downloads and size estimation. "
        "Install it with `uv add \"yt-dlp[default]\" && uv sync`."
    )


def resolve_ffmpeg_binary(preferred):
    preferred_path = resolve_command_path(preferred)
    if preferred_path:
        return preferred_path
    if preferred != "ffmpeg":
        raise RuntimeError(
            f"Unable to resolve ffmpeg from `--ffmpeg-bin={preferred}`. "
            "Pass a valid executable path, install system ffmpeg, or install "
            "`imageio-ffmpeg` with `uv add imageio-ffmpeg && uv sync`."
        )
    imageio_ffmpeg = require_imageio_ffmpeg()
    return imageio_ffmpeg.get_ffmpeg_exe()


def build_yt_dlp_command(
    yt_dlp_command,
    ffmpeg_binary=None,
    extra_args=None,
):
    command = list(yt_dlp_command)
    if ffmpeg_binary:
        command.extend(["--ffmpeg-location", ffmpeg_binary])
    if extra_args:
        command.extend(extra_args)
    return command


def extract_size_candidate(info):
    for key in ("filesize", "filesize_approx"):
        value = info.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def estimate_total_bytes(info):
    direct_size = extract_size_candidate(info)
    if direct_size is not None:
        return direct_size

    nested_total = 0
    nested_found = False
    for key in ("requested_downloads", "requested_formats"):
        entries = info.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_size = extract_size_candidate(entry)
            if entry_size is None:
                continue
            nested_total += entry_size
            nested_found = True
    if nested_found:
        return nested_total
    return None


def inspect_record_stats(record, yt_dlp_command, yt_dlp_extra_args):
    command = build_yt_dlp_command(
        yt_dlp_command,
        extra_args=[
            "--dump-single-json",
            "--quiet",
            "--no-warnings",
            "--skip-download",
            "-f",
            YT_DLP_FORMAT,
        ]
        + list(yt_dlp_extra_args)
        + [video_url(record)],
    )
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        error_text = (result.stderr or result.stdout or "").strip()
        return {
            "instrument": record["instrument"],
            "youtube_id": record["youtube_id"],
            "sample_key": record["sample_key"],
            "status": "error",
            "estimated_total_bytes": None,
            "url": video_url(record),
            "error_message": error_text or f"yt-dlp exited with code {result.returncode}",
        }

    payload = result.stdout.strip()
    if not payload:
        return {
            "instrument": record["instrument"],
            "youtube_id": record["youtube_id"],
            "sample_key": record["sample_key"],
            "status": "error",
            "estimated_total_bytes": None,
            "url": video_url(record),
            "error_message": "yt-dlp returned no metadata output",
        }

    try:
        info = json.loads(payload)
    except json.JSONDecodeError as exc:
        return {
            "instrument": record["instrument"],
            "youtube_id": record["youtube_id"],
            "sample_key": record["sample_key"],
            "status": "error",
            "estimated_total_bytes": None,
            "url": video_url(record),
            "error_message": f"failed to parse yt-dlp metadata: {exc}",
        }

    estimated_bytes = estimate_total_bytes(info)
    return {
        "instrument": record["instrument"],
        "youtube_id": record["youtube_id"],
        "sample_key": record["sample_key"],
        "status": "ok" if estimated_bytes is not None else "unknown_size",
        "estimated_total_bytes": estimated_bytes,
        "url": video_url(record),
        "error_message": "",
    }


def write_stats_files(records, args, yt_dlp_command):
    data_root = Path(args.data_root)
    rows = []
    by_instrument = {}
    estimated_total_bytes = 0
    estimated_record_count = 0
    unknown_record_count = 0

    for index, record in enumerate(records, start=1):
        print(f"[prepare_musices] stats {index}/{len(records)}: {record['sample_key']}")
        row = inspect_record_stats(record, yt_dlp_command, args.yt_dlp_extra_arg)
        rows.append(row)

        instrument_summary = by_instrument.setdefault(
            record["instrument"],
            {
                "record_count": 0,
                "estimated_record_count": 0,
                "unknown_record_count": 0,
                "estimated_total_bytes": 0,
            },
        )
        instrument_summary["record_count"] += 1

        if row["estimated_total_bytes"] is None:
            unknown_record_count += 1
            instrument_summary["unknown_record_count"] += 1
        else:
            estimated_total_bytes += row["estimated_total_bytes"]
            estimated_record_count += 1
            instrument_summary["estimated_record_count"] += 1
            instrument_summary["estimated_total_bytes"] += row["estimated_total_bytes"]

    summary = {
        "record_count": len(records),
        "estimated_record_count": estimated_record_count,
        "unknown_record_count": unknown_record_count,
        "estimated_total_bytes": estimated_total_bytes,
        "estimated_total_human": format_bytes(estimated_total_bytes),
        "by_instrument": by_instrument,
        "records": rows,
    }

    json_output = stats_json_path(data_root, args.stats_json_name)
    csv_output = stats_csv_path(data_root, args.stats_csv_name)
    ensure_dir(json_output.parent)

    with json_output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)

    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "instrument",
                "youtube_id",
                "sample_key",
                "status",
                "estimated_total_bytes",
                "url",
                "error_message",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return summary, json_output, csv_output


def load_stats_index(data_root, stats_json_name):
    json_path = stats_json_path(data_root, stats_json_name)
    if not json_path.exists():
        return None, {}
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    record_map = {}
    for row in payload.get("records", []):
        record_map[row["sample_key"]] = row.get("estimated_total_bytes")
    return payload.get("estimated_total_bytes"), record_map


def summarize_existing_downloads(records, data_root, video_dir):
    sizes = {}
    total_bytes = 0
    completed_count = 0
    for record in records:
        target = video_output_path(data_root, video_dir, record)
        size = target.stat().st_size if target.exists() else 0
        sizes[record["sample_key"]] = size
        if size > 0:
            completed_count += 1
            total_bytes += size
    return sizes, completed_count, total_bytes


def download_video(
    record,
    data_root,
    video_dir,
    yt_dlp_command,
    ffmpeg_binary,
    skip_existing,
    archive_path,
    yt_dlp_extra_args,
    estimated_total_bytes=None,
):
    target = video_output_path(data_root, video_dir, record)
    ensure_dir(target.parent)
    before_bytes = target.stat().st_size if target.exists() else 0
    if skip_existing and target.exists():
        return {
            "sample_key": record["sample_key"],
            "status": "skipped_existing",
            "output_path": str(target),
            "downloaded_bytes": before_bytes,
            "estimated_total_bytes": estimated_total_bytes,
            "error_message": "",
        }

    command = build_yt_dlp_command(
        yt_dlp_command,
        ffmpeg_binary=ffmpeg_binary,
        extra_args=[
            "-f",
            YT_DLP_FORMAT,
            "--merge-output-format",
            "mp4",
            "--continue",
            "--part",
            "--newline",
            "--progress",
            "--progress-template",
            DOWNLOAD_PROGRESS_TEMPLATE,
            "--download-archive",
            str(archive_path),
            "-o",
            str(target.with_suffix(".%(ext)s")),
        ]
        + list(yt_dlp_extra_args)
        + [video_url(record)],
    )

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        after_bytes = target.stat().st_size if target.exists() else before_bytes
        return {
            "sample_key": record["sample_key"],
            "status": "error",
            "output_path": str(target),
            "downloaded_bytes": after_bytes,
            "estimated_total_bytes": estimated_total_bytes,
            "error_message": f"yt-dlp exited with code {exc.returncode}",
        }

    after_bytes = target.stat().st_size if target.exists() else before_bytes
    if after_bytes > before_bytes and before_bytes > 0:
        status = "resumed"
    elif after_bytes > 0 and before_bytes == 0:
        status = "downloaded"
    else:
        status = "up_to_date"

    return {
        "sample_key": record["sample_key"],
        "status": status,
        "output_path": str(target),
        "downloaded_bytes": after_bytes,
        "estimated_total_bytes": estimated_total_bytes,
        "error_message": "",
    }


def write_download_failures(data_root, failures):
    output = download_failure_path(data_root)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_key",
                "status",
                "output_path",
                "downloaded_bytes",
                "estimated_total_bytes",
                "error_message",
            ],
        )
        writer.writeheader()
        for failure in failures:
            writer.writerow(failure)
    return output


def normalize_spectrogram(spectrogram, min_level_db):
    np = require_numpy()
    return np.clip((spectrogram - min_level_db) / -min_level_db, 0.0, 1.0)


def compute_mel_spectrogram(
    wav,
    sample_rate,
    fft_size,
    hop_size,
    num_mels,
    fmin,
    fmax,
    min_level_db,
    ref_level_db,
):
    np = require_numpy()
    librosa = require_librosa()
    stft = librosa.stft(y=wav, n_fft=fft_size, hop_length=hop_size, win_length=fft_size)
    magnitude = np.abs(stft)
    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=fft_size,
        n_mels=num_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel = np.dot(mel_basis, magnitude)
    min_level = np.exp(min_level_db / 20.0 * np.log(10))
    mel_db = 20.0 * np.log10(np.maximum(min_level, mel)) - ref_level_db
    return normalize_spectrogram(mel_db, min_level_db).T.astype(np.float32)


def align_waveform_length(wav, mel_frames, hop_size):
    np = require_numpy()
    target = mel_frames * hop_size
    if len(wav) < target:
        wav = np.pad(wav, (0, target - len(wav)), mode="constant", constant_values=0.0)
    else:
        wav = wav[:target]
    return wav.astype(np.float32)


def export_audio_and_mel(sample_dir, wav_path, args):
    np = require_numpy()
    librosa = require_librosa()
    wav, _ = librosa.load(str(wav_path), sr=args.sample_rate, mono=True)
    if np.max(np.abs(wav)) > 0:
        wav = wav / np.max(np.abs(wav)) * 0.999

    mel = compute_mel_spectrogram(
        wav=wav,
        sample_rate=args.sample_rate,
        fft_size=args.fft_size,
        hop_size=args.hop_size,
        num_mels=args.num_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        min_level_db=args.min_level_db,
        ref_level_db=args.ref_level_db,
    )
    wav = align_waveform_length(wav, mel.shape[0], args.hop_size)

    np.save(sample_dir / "raw_audio.npy", wav.astype(np.float32), allow_pickle=False)
    np.save(sample_dir / "mel.npy", mel.astype(np.float32), allow_pickle=False)
    return mel.shape[0]


def normalize_flow_component(component, flow_clip):
    np = require_numpy()
    scaled = np.clip(component, -flow_clip, flow_clip)
    scaled = 127.0 + scaled * (127.0 / flow_clip)
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def extract_frames_and_flow(video_path, sample_dir, frame_size, frame_stride, flow_clip):
    cv2 = require_cv2()
    np = require_numpy()
    image_dir = sample_dir / "image"
    flow_x_dir = sample_dir / "flow_x"
    flow_y_dir = sample_dir / "flow_y"
    for directory in [image_dir, flow_x_dir, flow_y_dir]:
        ensure_dir(directory)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frames = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % max(frame_stride, 1) == 0:
            frames.append(cv2.resize(frame, (frame_size, frame_size)))
        index += 1
    cap.release()

    if len(frames) < 2:
        raise RuntimeError(f"Video too short to extract optical flow: {video_path}")

    previous_gray = None
    zero_flow = np.full((frame_size, frame_size), 127, dtype=np.uint8)
    for frame_id, frame in enumerate(frames, start=1):
        cv2.imwrite(str(image_dir / f"{frame_id}.jpg"), frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is None:
            flow_x = zero_flow
            flow_y = zero_flow
        else:
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            flow_x = normalize_flow_component(flow[..., 0], flow_clip)
            flow_y = normalize_flow_component(flow[..., 1], flow_clip)
        cv2.imwrite(str(flow_x_dir / f"{frame_id}.jpg"), flow_x)
        cv2.imwrite(str(flow_y_dir / f"{frame_id}.jpg"), flow_y)
        previous_gray = gray


def find_cluster(indices):
    min_idx = 0
    max_idx = -1
    min_num = indices[min_idx]
    max_num = indices[max_idx]
    for _ in range(len(indices) - 1):
        if (min_num + 5) not in indices:
            min_idx += 1
            min_num = indices[min_idx]
    for _ in range(len(indices) - 1):
        if (max_num - 5) not in indices:
            max_idx -= 1
            max_num = indices[max_idx]
    if min_num > max_num:
        max_num = min_num
    return min_num, max_num


def padding_square(image):
    cv2 = require_cv2()
    height, width = image.shape[:2]
    if height == width:
        return image
    larger_side = max(height, width)
    delta = abs(height - width)
    if larger_side == height:
        left, right = delta // 2, delta - delta // 2
        top = bottom = 0
    else:
        top, bottom = delta // 2, delta - delta // 2
        left = right = 0
    value = [127, 127, 127] if image.ndim == 3 else [127]
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)


def compute_crop_bounds(sample_dir):
    cv2 = require_cv2()
    np = require_numpy()
    flow_x_paths = sorted((sample_dir / "flow_x").glob("*.jpg"), key=lambda path: int(path.stem))
    if not flow_x_paths:
        raise RuntimeError(f"No optical-flow frames found in {sample_dir}")

    sum_flow_x = None
    sum_flow_y = None
    for flow_x_path in flow_x_paths:
        frame_id = flow_x_path.stem
        flow_y_path = sample_dir / "flow_y" / f"{frame_id}.jpg"
        flow_x = cv2.imread(str(flow_x_path), 0)
        flow_y = cv2.imread(str(flow_y_path), 0)
        if flow_x is None or flow_y is None:
            continue
        diff_x = np.abs(flow_x.astype(np.int32) - 127)
        diff_y = np.abs(flow_y.astype(np.int32) - 127)
        sum_flow_x = diff_x if sum_flow_x is None else sum_flow_x + diff_x
        sum_flow_y = diff_y if sum_flow_y is None else sum_flow_y + diff_y

    if sum_flow_x is None or sum_flow_y is None:
        return None

    total = sum_flow_x + sum_flow_y
    mask = (total > (2 * len(flow_x_paths))).astype(int)
    sum_w = np.where(np.sum(mask, axis=0) > 0)[0]
    sum_h = np.where(np.sum(mask, axis=1) > 0)[0]
    if len(sum_w) == 0 or len(sum_h) == 0:
        return None

    w_min, w_max = find_cluster(sum_w)
    h_min, h_max = find_cluster(sum_h)
    if (w_max - w_min) < 50 or (h_max - h_min) < 50:
        return None
    return int(w_min), int(w_max), int(h_min), int(h_max)


def crop_motion_region(sample_dir):
    cv2 = require_cv2()
    image_crop_dir = sample_dir / "image_crop"
    flow_x_crop_dir = sample_dir / "flow_x_crop"
    flow_y_crop_dir = sample_dir / "flow_y_crop"
    for directory in [image_crop_dir, flow_x_crop_dir, flow_y_crop_dir]:
        ensure_dir(directory)

    bounds = compute_crop_bounds(sample_dir)
    image_paths = sorted((sample_dir / "image").glob("*.jpg"), key=lambda path: int(path.stem))
    if not image_paths:
        raise RuntimeError(f"No extracted image frames found in {sample_dir}")

    for image_path in image_paths:
        frame_id = image_path.stem
        image = cv2.imread(str(image_path))
        flow_x = cv2.imread(str(sample_dir / "flow_x" / f"{frame_id}.jpg"), 0)
        flow_y = cv2.imread(str(sample_dir / "flow_y" / f"{frame_id}.jpg"), 0)

        if bounds is not None:
            w_min, w_max, h_min, h_max = bounds
            image = image[h_min:h_max, w_min:w_max]
            flow_x = flow_x[h_min:h_max, w_min:w_max]
            flow_y = flow_y[h_min:h_max, w_min:w_max]

        image = padding_square(image)
        flow_x = padding_square(flow_x)
        flow_y = padding_square(flow_y)

        cv2.imwrite(str(image_crop_dir / f"{frame_id}.jpg"), image)
        cv2.imwrite(str(flow_x_crop_dir / f"{frame_id}.jpg"), flow_x)
        cv2.imwrite(str(flow_y_crop_dir / f"{frame_id}.jpg"), flow_y)


def extract_audio_from_video(video_path, wav_path, ffmpeg_binary, skip_existing):
    ensure_dir(wav_path.parent)
    if skip_existing and wav_path.exists():
        return wav_path
    command = [
        ffmpeg_binary,
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path


def processed_sample_ready(sample_dir):
    required_files = [
        sample_dir / "raw_audio.npy",
        sample_dir / "mel.npy",
    ]
    required_dirs = [
        sample_dir / "image_crop",
        sample_dir / "flow_x_crop",
        sample_dir / "flow_y_crop",
    ]
    return all(path.exists() for path in required_files) and all(path.exists() for path in required_dirs)


def process_record(record, args, ffmpeg_binary):
    data_root = Path(args.data_root)
    video_path = video_output_path(data_root, args.video_dir, record)
    sample_dir = sample_output_dir(data_root, args.processed_dir, record)

    if args.skip_existing and processed_sample_ready(sample_dir):
        return {
            "sample_dir": sample_dir,
            "mel_frames": None,
            "status": "skipped_existing",
        }

    ensure_dir(sample_dir)
    wav_path = sample_dir / "source.wav"
    extract_audio_from_video(video_path, wav_path, ffmpeg_binary, args.skip_existing)
    extract_frames_and_flow(
        video_path=video_path,
        sample_dir=sample_dir,
        frame_size=args.frame_size,
        frame_stride=args.frame_stride,
        flow_clip=args.flow_clip,
    )
    crop_motion_region(sample_dir)
    mel_frames = export_audio_and_mel(sample_dir, wav_path, args)
    return {
        "sample_dir": sample_dir,
        "mel_frames": mel_frames,
        "status": "processed",
    }


def write_split_files(records, args):
    np = require_numpy()
    data_root = Path(args.data_root)
    processed_dir = Path(args.processed_dir)

    existing = []
    for record in records:
        sample_dir = sample_output_dir(data_root, processed_dir, record)
        mel_path = sample_dir / "mel.npy"
        audio_path = sample_dir / "raw_audio.npy"
        if not processed_sample_ready(sample_dir):
            continue
        mel = np.load(mel_path, mmap_mode="r")
        existing.append(
            {
                "sample_dir": sample_dir.relative_to(data_root).as_posix(),
                "mel_path": mel_path.relative_to(data_root).as_posix(),
                "audio_path": audio_path.relative_to(data_root).as_posix(),
                "mel_frames": int(mel.shape[0]),
            }
        )

    if not existing:
        raise RuntimeError("No processed samples found. Run the process stage first.")

    rng = random.Random(args.seed)
    existing.sort(key=lambda item: item["sample_dir"])
    rng.shuffle(existing)

    test_count = max(1, int(round(len(existing) * args.test_size)))
    test_items = existing[:test_count]
    train_items = existing[test_count:]
    if not train_items:
        raise RuntimeError("Test split consumed all samples. Reduce --test-size.")

    def write_lines(target_path, items):
        with target_path.open("w", encoding="utf-8") as handle:
            for item in sorted(items, key=lambda row: row["sample_dir"]):
                handle.write(
                    "|".join(
                        [
                            item["sample_dir"],
                            item["mel_path"],
                            item["audio_path"],
                            str(item["mel_frames"]),
                        ]
                    )
                    + "\n"
                )

    write_lines(Path(args.data_root) / args.train_split_name, train_items)
    write_lines(Path(args.data_root) / args.test_split_name, test_items)
    return len(train_items), len(test_items)


def run_download_stage(records, args, yt_dlp_command, ffmpeg_binary):
    data_root = Path(args.data_root)
    archive_path = download_archive_path(data_root, args.download_archive_name)
    ensure_dir(archive_path.parent)

    overall_estimated_bytes, estimated_bytes_by_key = load_stats_index(data_root, args.stats_json_name)
    current_sizes, completed_count, downloaded_total_bytes = summarize_existing_downloads(
        records,
        data_root,
        args.video_dir,
    )

    failures = []
    aborted = False
    for index, record in enumerate(records, start=1):
        sample_key = record["sample_key"]
        estimated_total_bytes = estimated_bytes_by_key.get(sample_key)
        print(
            f"[prepare_musices] downloading {index}/{len(records)}: {sample_key} | "
            f"downloaded={format_bytes(downloaded_total_bytes)}"
            + (
                f" / estimated_total={format_bytes(overall_estimated_bytes)}"
                if overall_estimated_bytes is not None
                else ""
            )
        )
        before_bytes = current_sizes.get(sample_key, 0)
        result = download_video(
            record=record,
            data_root=data_root,
            video_dir=args.video_dir,
            yt_dlp_command=yt_dlp_command,
            ffmpeg_binary=ffmpeg_binary,
            skip_existing=args.skip_existing,
            archive_path=archive_path,
            yt_dlp_extra_args=args.yt_dlp_extra_arg,
            estimated_total_bytes=estimated_total_bytes,
        )

        after_bytes = result["downloaded_bytes"] or 0
        downloaded_total_bytes += max(after_bytes - before_bytes, 0)
        current_sizes[sample_key] = after_bytes
        if before_bytes == 0 and after_bytes > 0:
            completed_count += 1

        if result["status"] == "error":
            failures.append(result)
            print(
                f"[prepare_musices] download failed: {sample_key} | "
                f"error={result['error_message']}"
            )
            if args.abort_on_download_error:
                aborted = True
                break
            continue

        print(
            f"[prepare_musices] download {result['status']}: {sample_key} | "
            f"file={format_bytes(after_bytes)} | cumulative={format_bytes(downloaded_total_bytes)}"
        )

    failure_output = write_download_failures(data_root, failures)
    print(
        f"[prepare_musices] download summary: completed={completed_count}/{len(records)}, "
        f"failures={len(failures)}, failure_log={failure_output}"
    )
    if aborted:
        raise RuntimeError("Aborted because --abort-on-download-error was enabled.")


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    ensure_dir(data_root)
    records = load_records(args.json_path, max_videos=args.max_videos)

    if args.action in {"manifest", "all"}:
        output = write_manifest(
            records=records,
            data_root=data_root,
            video_dir=args.video_dir,
            processed_dir=args.processed_dir,
            manifest_name=args.manifest_name,
        )
        print(f"[prepare_musices] wrote manifest: {output}")
        if args.action == "manifest":
            return

    if args.action in {"stats", "all"}:
        yt_dlp_command = resolve_yt_dlp_command(args.yt_dlp_bin)
        summary, json_output, csv_output = write_stats_files(records, args, yt_dlp_command)
        print(
            f"[prepare_musices] wrote stats: {json_output}, {csv_output} | "
            f"estimated={format_bytes(summary['estimated_total_bytes'])}, "
            f"unknown={summary['unknown_record_count']}"
        )
        if args.action == "stats":
            return

    if args.action in {"download", "all"}:
        if args.skip_download:
            print("[prepare_musices] skipping download stage")
        else:
            yt_dlp_command = resolve_yt_dlp_command(args.yt_dlp_bin)
            ffmpeg_binary = resolve_ffmpeg_binary(args.ffmpeg_bin)
            run_download_stage(records, args, yt_dlp_command, ffmpeg_binary)
        if args.action == "download":
            return

    if args.action in {"process", "all"}:
        ffmpeg_binary = resolve_ffmpeg_binary(args.ffmpeg_bin)
        for index, record in enumerate(records, start=1):
            video_path = video_output_path(data_root, args.video_dir, record)
            if not video_path.exists():
                print(f"[prepare_musices] skip missing video {record['sample_key']}: {video_path}")
                continue
            print(f"[prepare_musices] processing {index}/{len(records)}: {record['sample_key']}")
            result = process_record(record, args, ffmpeg_binary)
            if result["status"] == "skipped_existing":
                print(f"[prepare_musices] process skipped existing: {record['sample_key']}")
        if args.action == "process":
            return

    if args.action in {"splits", "all"}:
        train_count, test_count = write_split_files(records, args)
        print(
            f"[prepare_musices] wrote split files: "
            f"{args.train_split_name} ({train_count}), {args.test_split_name} ({test_count})"
        )


if __name__ == "__main__":
    main()
