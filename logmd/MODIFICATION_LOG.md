# 修改日志

## 2026-04-20 main.py 补全

### 修改内容
1. 将 `main.py` 从占位脚本改为统一入口脚本。
2. 新增 `MODULE_MAP`，支持两类动作：
   - `train` -> `train_whole_sync`
   - `preprocess` -> `Data_loaders.Image_preprocess`
3. 新增参数解析：
   - 位置参数 `action`（默认 `train`）
   - 透传参数 `extra_args`（支持 `--` 后参数原样转发）
4. 新增 `_run_module`，通过 `runpy.run_module` 运行目标模块，并在运行后恢复 `sys.argv`。
5. 新增启动日志输出（目标模块与透传参数）。

### 本次使用的命令行
```bash
sed -n '1,200p' main.py
python -m py_compile main.py
python main.py --help
git diff -- main.py
git status --short main.py && sed -n '1,220p' main.py
```

## 2026-05-01 VIAI 本地跑通测试与云端训练适配

### 背景与目标
1. 本地机器为 RTX 3060 级别显卡，不作为完整论文训练环境，只用于 smoke test。
2. 本地 smoke test 只验证依赖、数据格式、预处理、dataloader、模型 forward/backward 和一次 optimizer update。
3. 正式训练迁移到云端 GPU，云端恢复论文/默认配置后再做长训练和指标核验。
4. 本节同时修正 2026-04-20 日志中的入口描述：当前 `main.py` 实际支持 `prepare-data -> tools.prepare_musices`，不是 `Data_loaders.Image_preprocess`。

### 论文要求核对
1. 数据处理：16kHz audio、STFT frame length 1280、hop size 320、80 mel bins、125Hz-7.6kHz、Mel 归一化到 0-1。
2. 输入形状：4 秒音频、80x200 Mel-spectrogram、对应 50 个视频帧。
3. 缺失区域：训练时随机裁剪 0.4s-1.0s，对应 20-50 个 Mel frame，并用相邻 clean spectrum bins 插值初始化。
4. 视频处理：论文使用 TV-L1 optical flow，flow clipping 到 20 pixels，motion salient crop 后 padding 成正方形，图像和 flow 归一化到 -1 到 1。
5. 训练目标：Adam lr=1e-4；VIAI-AV batch size 16；contrastive synchronization margin `gamma=1`；sync loss 只更新 video encoder。

### 已执行/拟执行代码调整
1. `pyproject.toml`
   - 基础依赖不包含 `torch`，避免云端被本仓库固定到不匹配的 CUDA wheel。
   - 新增 `local-cuda` extra，本地 3060 smoke test 使用 CUDA 版 torch。
   - 云端服务器按实际驱动/CUDA 版本自行安装 PyTorch。
   - 增加训练辅助依赖：`tensorboardX`、`nnmnkwii`。
   - 将 `opencv-python` 替换为 `opencv-contrib-python`，用于 TV-L1 optical flow。
2. `tools/prepare_musices.py`
   - 新增 `--flow-method {tvl1,farneback}`，默认 `tvl1`。
   - 默认使用 OpenCV contrib 的 TV-L1 接口；如环境缺失，提示安装/sync contrib OpenCV。
   - `farneback` 仅保留为非论文 smoke-test fallback。
3. `Data_loaders/audio_loader.py`
   - 移除对 `keras.utils.np_utils` 的依赖，改为本地 `to_categorical()`，减少本地和云端环境负担。
4. `base_options.py` / `train_whole_sync.py`
   - 新增 `--max_train_steps`，用于本地 3060 smoke test 只跑 1 个或少量 optimizer step。
   - 新增 `--sync_margin`，默认 1.0，对齐论文 `gamma=1`。
5. `Models/Whole_Sync_inpainting_modify.py` / `loss_functions.py`
   - `L2ContrastiveLoss` 默认 margin 改为 1.0。
   - sync loss 计算时对 target audio feature 使用 `detach()`，使 sync loss 只推动 video encoder，与论文训练稳定化描述一致。
6. `README.md`
   - 增加本地 3060 smoke-test 命令。
   - 增加云端训练前检查和训练命令。
   - 明确本地降配参数不用于论文指标。

### 本地 3060 smoke-test 命令
```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra local-cuda
uv run --extra local-cuda python main.py --help
uv run --extra local-cuda python main.py prepare-data -- --help
uv run --extra local-cuda python -c "import torch, cv2, librosa, nnmnkwii, tensorboardX; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
uv run --extra local-cuda python main.py prepare-data -- process --json data/MUSICES.json --data-root data --max-videos 1 --skip-existing
uv run --extra local-cuda python main.py split-data -- --data-root data --max-samples 1 --test-size 0 --val-size 0 --allow-empty-eval
uv run --extra local-cuda python main.py train -- --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
```

如 3060 显存仍不足，可临时追加：
```bash
uv run --extra local-cuda python main.py train -- --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0 --image_size 128
```

注意：任何降低 `image_size`、`load_num` 或其他输入规模的设置，只用于本地功能测试，不用于论文指标。

### 云端训练建议命令
```bash
nvidia-smi
UV_CACHE_DIR=/tmp/uv-cache uv sync
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -c "import torch, cv2; print(torch.cuda.is_available(), torch.version.cuda, hasattr(cv2, 'optflow'))"
.venv/bin/python main.py prepare-data -- all --json data/MUSICES.json --data-root data --skip-existing
.venv/bin/python main.py split-data -- --data-root data
.venv/bin/python main.py train -- --batch_size 16 --num_workers 4 --display_id 0
```

云端训练前需要重新确认：
1. CUDA 版 PyTorch 与云端驱动匹配。
2. `opencv-contrib-python` 的 TV-L1 接口可用。
3. 完整数据已经完成 `process` 和 `split-data`，默认生成 85% train / 5% val / 10% test。
4. checkpoint、TensorBoard 日志、retrieval 指标可以正常写入。
5. 先跑 100-500 step sanity training，再启动长训练。

注意：上面的 `cu121` 只是示例，云端需按服务器实际 CUDA/驱动选择 PyTorch wheel。手动安装云端 PyTorch 后，建议用 `.venv/bin/python` 或 `uv run --no-sync` 运行，避免 `uv run` 自动同步时移除手动安装的云端 torch。

### 已知仍未完整复现的部分
1. 论文中的 shot detection、去除非演奏/黑场片段、裁掉每个视频前 6 秒，当前 pipeline 尚未完整自动化。
2. 论文的 10% fixed test + 5% held-out validation 协议尚未完整接入训练 loader。
3. VIAI-AA' probe loss 和 WaveNet spectrogram-to-audio 端到端评估仍需后续补齐。
4. 本地 smoke test 只证明链路可运行，不代表模型收敛或论文指标。

## 2026-05-01 独立 split 工具与 50 帧对齐

### 修改内容
1. 新增 `tools/split_musices.py`，从 `data/processed/<instrument>/<youtube_id>/` 扫描已处理样本并生成：
   - `train_new_split.txt`
   - `val_new_split.txt`
   - `test_new_split.txt`
2. 新 split 工具默认 `test_size=0.10`、`val_size=0.05`，剩余为 train；支持 `--max-samples` 与 `--allow-empty-eval` 方便本地 smoke test。
3. `main.py` 新增 `split-data -> tools.split_musices`。
4. `prepare_musices.py` 原 `splits` action 仅保留 legacy 兼容，并打印提示推荐使用 `split-data`。
5. `base_options.py` 新增 `train_split_name`、`val_split_name`、`test_split_name`，并将 `image_hope_size` 默认改为 2。
6. `Data_loaders/audio_loader.py` 默认读取 `train+val`；空的 val/test split 会跳过，test 保留给最终评估。
7. 4 秒视频窗口改为默认 50 帧：从 25fps 原始帧中按 `image_hope_size=2` 抽帧，并用时间换算对齐 200 个 mel frames。

### 推荐命令
```bash
uv run --extra local-cuda python main.py split-data -- --data-root data
uv run --extra local-cuda python main.py split-data -- --data-root data --max-samples 1 --test-size 0 --val-size 0 --allow-empty-eval
```

### split 数据处理测试运行命令
前置条件：`split-data` 只扫描已经完成预处理的样本，必须先存在至少一个完整目录：
`data/processed/<instrument>/<youtube_id>/`，且其中包含 `raw_audio.npy`、`mel.npy`、`image_crop/`、`flow_x_crop/`、`flow_y_crop/`。

1. 检查 split 命令参数：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py split-data -- --help
```

2. 检查当前是否已有完整 processed 样本目录：
```bash
find data/processed -mindepth 2 -maxdepth 2 -type d | head
find data/processed -mindepth 3 -maxdepth 3 \( -name raw_audio.npy -o -name mel.npy -o -name image_crop -o -name flow_x_crop -o -name flow_y_crop \) | head -30
```

3. 如果还没有完整 processed 样本，先处理一批本地已有视频。不要把 `--max-videos` 设得太小，否则可能只扫到缺失 mp4 并直接跳过：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- process --json data/MUSICES.json --data-root data --max-videos 100 --skip-existing
```

4. 本地 smoke test：只取 1 个可用样本，允许 val/test 为空，用来验证 split 工具、进度条和输出格式：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py split-data -- --data-root data --max-samples 1 --test-size 0 --val-size 0 --allow-empty-eval
```

5. 检查 smoke split 输出：
```bash
wc -l data/train_new_split.txt data/val_new_split.txt data/test_new_split.txt
sed -n '1,5p' data/train_new_split.txt
```

6. 完整数据划分：默认生成 85% train、5% val、10% test：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py split-data -- --data-root data
```

7. 完整 split 后再次检查行数：
```bash
wc -l data/train_new_split.txt data/val_new_split.txt data/test_new_split.txt
```

常见报错：
1. `Processed data directory not found: data/processed`：还没有成功运行 `prepare-data -- process`。
2. `No processed samples found. Run the process stage first.`：`data/processed` 存在，但样本不完整，通常缺少 `mel.npy`、`raw_audio.npy` 或 crop 后的 `image_crop/flow_x_crop/flow_y_crop`。
3. `--skip-existing: command not found`：多行命令中反斜杠 `\` 后面有空格，需保证 `\` 是该行最后一个字符。

## 2026-05-01 论文对齐的数据管线、split 和测试入口

### 背景
前一版 `split-data` 只做样本级 train/val/test 划分，尚未处理论文协议中的几个关键点：
1. 论文先做 video-level train/test split，再从对应视频生成样本，避免同一视频片段跨集合泄漏。
2. 论文会裁掉每个视频前 6 秒，并按 shot 处理视频，去除黑场/静音或非演奏片段。
3. 公开 `MUSICES.json` 只有 instrument 和 YouTube ID，没有原论文内部 shot 边界标注，因此本仓库只能用可复现的 OpenCV frame-difference heuristic 近似 shot detection。
4. 原 `main.py` 没有真实测试入口，README 也缺少完整的“数据准备/划分、训练、测试”命令。

### 本次修改内容
1. `tools/prepare_musices.py`
   - 新增论文风格预处理参数：
     - `--trim-start-sec`，默认 6.0，对齐论文裁掉视频开头 6 秒。
     - `--min-segment-sec`，默认 4.0，对齐 4 秒输入窗口。
     - `--shot-detection` / `--no-shot-detection`，默认开启 shot detection。
     - `--shot-diff-threshold`，默认 35.0，用于 OpenCV 灰度帧差 shot boundary 近似。
     - `--black-frame-threshold`、`--max-black-ratio`，用于过滤黑场片段。
     - `--min-audio-rms`，用于过滤近似静音片段。
   - 默认将每个有效 shot 写到：
     - `data/processed/<instrument>/<youtube_id>/shot_000000/`
     - `data/processed/<instrument>/<youtube_id>/shot_000001/`
   - 每个 shot 样本仍生成 `source.wav`、`raw_audio.npy`、`mel.npy`、`image/`、`flow_x/`、`flow_y/`、`image_crop/`、`flow_x_crop/`、`flow_y_crop/`。
   - 保留旧式整视频处理兼容模式：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- process --json data/MUSICES.json --data-root data --skip-existing --no-shot-detection --trim-start-sec 0
```
   - `prepare-data -- all` 不再自动写 legacy two-way split，完成处理后提示继续运行 `main.py split-data`。

2. `tools/split_musices.py`
   - 递归发现 processed 样本，兼容旧 flat 样本和新 `shot_*` 样本。
   - 对样本按源视频 key `<instrument>/<youtube_id>` 分组，再做 train/val/test split，避免同一视频的不同 shot 泄漏到不同 split。
   - 默认仍为论文比例：85% train、5% val、10% test。
   - split 输出格式保持不变：`sample_dir|mel_path|audio_path|mel_frames`，兼容当前 dataloader。
   - split summary 新增样本数和源视频数，例如 `train=1 samples/1 videos`。
   - 若同一视频目录里同时存在旧 flat 样本和新 `shot_*` 样本，优先使用 shot 样本，跳过父目录旧样本。

3. 测试入口
   - `main.py` 新增 `test -> test_whole_sync`。
   - 新增 `test_whole_sync.py`，加载 `test_new_split.txt`，恢复 checkpoint，仅执行 evaluation，不做 optimizer update。
   - 输出 test reconstruction loss、Mel L1 loss、sync loss，以及 audio-video retrieval 指标。
   - 如果传入的 `--resume_path` 不存在，会在同目录或 `--checkpoint_dir` 下查找最新的 `VIAI-AV_checkpoint_step*.pth.tar`。

4. README
   - 重写数据准备和 dataset split 流程命令。
   - 写入训练 smoke test、云端完整训练、测试集评估命令。
   - 明确说明 shot detection 是公开数据条件下的 OpenCV 近似，不是论文内部人工/原始 shot 标注。
   - 明确当前 `main.py test` 输出当前模型路径下的 loss 和 retrieval 指标；完整 SDR/OPS 仍需要后续补 WaveNet/audio-generation evaluation。

5. 为通过 smoke test 顺手修正的兼容问题
   - `train_whole_sync.py`：`matplotlib` 改为可选导入，缺包时不阻塞训练。
   - `utils/util.py`：`PIL.Image` 改为 `save_image()` 内部懒加载，避免非图片保存路径强依赖 Pillow。
   - `visdom_utils/visualizer.py`：移除顶层 `html` 导入，避免 `display_id=0` 时被 `dominate` 缺包阻塞。
   - `networks/Image_Embedding.py`：将固定 `AvgPool2d(7)` 改为 `AdaptiveAvgPool2d((1, 1))`，修复默认 `image_size=256` 下 `mat1 and mat2 shapes cannot be multiplied` 的视觉 encoder 维度错误。

### README 中记录的完整流程命令
1. 环境检查：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra local-cuda
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python -c "import torch, cv2, librosa, nnmnkwii, tensorboardX, tqdm; print(torch.__version__, torch.cuda.is_available(), hasattr(cv2, 'optflow'))"
```

2. 数据准备：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- manifest --json data/MUSICES.json --data-root data
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- stats --json data/MUSICES.json --data-root data
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- download --json data/MUSICES.json --data-root data --skip-existing
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- process --json data/MUSICES.json --data-root data --skip-existing
```

3. 数据集划分：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py split-data -- --data-root data
```

4. 本地 smoke split：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- process --json data/MUSICES.json --data-root data --max-videos 100 --skip-existing
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py split-data -- --data-root data --max-samples 1 --test-size 0 --val-size 0 --allow-empty-eval
wc -l data/train_new_split.txt data/val_new_split.txt data/test_new_split.txt
sed -n '1,5p' data/train_new_split.txt
```

5. 本地训练 smoke test：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py train -- --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
```

6. 云端完整训练：
```bash
nvidia-smi
UV_CACHE_DIR=/tmp/uv-cache uv sync
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -c "import torch, cv2; print(torch.cuda.is_available(), torch.version.cuda, hasattr(cv2, 'optflow'))"
.venv/bin/python main.py prepare-data -- all --json data/MUSICES.json --data-root data --skip-existing
.venv/bin/python main.py split-data -- --data-root data
.venv/bin/python main.py train -- --batch_size 16 --num_workers 4 --display_id 0
```

7. 测试集评估：
```bash
.venv/bin/python main.py test -- --resume_path checkpoints/VIAI-AV_checkpoint_step000001000.pth.tar --batch_size 16 --num_workers 4 --display_id 0
```

### 本次实际验证结果
已通过：
```bash
.venv/bin/python -m py_compile tools/prepare_musices.py tools/split_musices.py main.py test_whole_sync.py
.venv/bin/python main.py prepare-data -- process --help
.venv/bin/python main.py split-data -- --help
.venv/bin/python main.py test -- --help
.venv/bin/python main.py split-data -- --data-root data --max-samples 1 --test-size 0 --val-size 0 --allow-empty-eval
```

本地训练 smoke test 已通过：
```bash
.venv/bin/python main.py train -- --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
```

关键输出：
```text
Reached local smoke-test max_train_steps=1
Step 1 L1_loss [train] Loss: 0.5894665718078613
VIAI-AV Step 1 [train] EmbeddingL2_loss: 1.0002461671829224
Saved checkpoint: ./checkpoints/VIAI-AV_checkpoint_step000000001.pth.tar
Finished
```

测试入口已通过 smoke 验证。由于本地 smoke split 的 `test_new_split.txt` 为空，使用 `--test_split_name train_new_split.txt` 临时验证测试代码路径：
```bash
.venv/bin/python main.py test -- --resume_path ./checkpoints/VIAI-AV_checkpoint_step000000001.pth.tar --test_split_name train_new_split.txt --batch_size 1 --num_workers 0 --display_id 0
```

关键输出：
```text
[test] losses: reconstruction=1.380574, mel_l1=0.250176, sync=1.061548
[test] Video Retrieval (1 samples): R@1: 100.00, R@5: 100.00, R@10: 100.00, R@50: 100.00, MedR: 1.0, MeanR: 1.0
[test] Audio Retrieval (1 samples): R@1: 100.00, R@5: 100.00, R@10: 100.00, R@50: 100.00, MedR: 1.0, MeanR: 1.0
```

### 仍需后续补齐
1. 当前 shot detection 是 OpenCV frame-difference heuristic，无法等价论文内部原始 shot 边界标注。
2. 黑场/静音过滤是可复现近似规则，尚未完全覆盖论文中的“非演奏片段”人工/规则清洗。
3. `main.py test` 当前评估当前 VIAI 模型路径下的 loss 和 retrieval 指标；论文完整 SDR/OPS 仍需要 WaveNet/audio-generation evaluation 链路。

## 2026-05-01 11.00 pm VIAI-A 第一阶段 audio-only 复现入口

### 背景与目标
根据 `information.md` 中“8.1 第一阶段：复现 VIAI-A”的要求，本次新增一个独立的 audio-only 最小复现链路。该阶段只做 Mel-spectrogram inpainting，不使用视频帧、光流、visual encoder、sync loss、GAN loss 或 WaveNet，目标是先跑通：

```text
MUSICES raw video -> 16kHz mono audio -> 80-bin Mel -> random 0.4s-1.0s mask -> MelEncoder + MelDecoder -> L1 / PSNR / SSIM
```

### 本次修改内容
1. `main.py`
   - 新增 `prepare-viai-a -> tools.prepare_viai_a`。
   - 新增 `train-viai-a -> train_viai_a`。
   - 新增 `test-viai-a -> test_viai_a`。

2. `tools/prepare_viai_a.py`
   - 新增 VIAI-A audio-only 处理脚本。
   - 复用 `tools.prepare_musices` 中的 MUSICES 记录读取、raw video 路径解析、ffmpeg 解析、音频抽取和 Mel 生成函数。
   - 从 `data/raw_videos/<instrument>/<youtube_id>.mp4` 抽取 `source.wav`，并生成 `raw_audio.npy` 与 `mel.npy`。
   - 默认使用论文参数：16kHz mono、STFT length 1280、hop size 320、80 Mel bins、125Hz-7.6kHz。
   - 输出目录复用 `data/processed/<instrument>/<youtube_id>/`，不要求存在 `image_crop/flow_x_crop/flow_y_crop`。

3. `tools/split_musices.py`
   - 新增 `--audio-only`。
   - audio-only 模式只检查 `raw_audio.npy` 和 `mel.npy`。
   - 若未显式指定 split 文件名，默认输出：
     - `train_viai_a_split.txt`
     - `val_viai_a_split.txt`
     - `test_viai_a_split.txt`
   - 仍按 source video key 分组，避免同一原视频跨 train/val/test 泄漏。

4. `Data_loaders/viai_a_loader.py`
   - 新增 VIAI-A 专用 dataloader。
   - 从 split 文件读取 `mel.npy` 与 `raw_audio.npy`。
   - 训练阶段随机裁剪 4 秒窗口，即 200 个 Mel frames。
   - 验证/测试阶段使用居中窗口。
   - batch 只返回 `mel`、`audio`、`path`，不读取图片或光流。

5. `Models/VIAI_A_inpainting.py`
   - 新增 `VIAIAModel`。
   - 只包含 `MelEncoder`、`MelDecoder` 和 Adam optimizer。
   - 使用 `mel_loader.corrupt_mel_spectrogram()` 随机 mask 20-50 个 Mel frames。
   - 优化目标为 `eta1 * full_l1 + missing_l1`。
   - checkpoint 命名为 `VIAI-A_checkpoint_step*.pth.tar`。

6. `train_viai_a.py`
   - 新增 VIAI-A 训练入口。
   - 支持 `--max_train_steps`，用于本地 smoke test。
   - 写入 TensorBoard 标量：total loss、full Mel L1、missing-region Mel L1。

7. `test_viai_a.py`
   - 新增 VIAI-A 测试入口。
   - 自动查找或加载 `VIAI-A_checkpoint_step*.pth.tar`。
   - 在 `test_viai_a_split.txt` 上输出：
     - total loss
     - full Mel L1
     - missing-region Mel L1
     - full PSNR
     - missing-region PSNR
     - SSIM

8. `pyproject.toml`
   - 新增 `scikit-image`，用于 `skimage.metrics.structural_similarity` 计算 SSIM。

9. `README.md`
   - 新增 “Stage 1: VIAI-A Audio-Only” 小节。
   - 写入下载、audio-only 处理、audio-only split、训练 smoke test、完整训练和测试命令。

### README 中记录的 VIAI-A 命令
1. 下载 MUSICES raw videos：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- download --json data/MUSICES.json --data-root data --skip-existing
```

2. 生成 audio-only 样本：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-viai-a -- --json data/MUSICES.json --data-root data --skip-existing
```

3. 本地 smoke test：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-viai-a -- --json data/MUSICES.json --data-root data --max-videos 100 --skip-existing
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py split-data -- --data-root data --audio-only --max-samples 1 --test-size 0 --val-size 0 --allow-empty-eval
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py train-viai-a -- --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
```

4. 完整 audio-only split / train / test：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py split-data -- --data-root data --audio-only
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py train-viai-a -- --batch_size 16 --num_workers 4 --display_id 0
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py test-viai-a -- --resume_path checkpoints/VIAI-A_checkpoint_step000001000.pth.tar --batch_size 16 --num_workers 4 --display_id 0
```

### 后续验证计划
1. 静态检查：
```bash
.venv/bin/python -m py_compile main.py tools/prepare_viai_a.py tools/split_musices.py train_viai_a.py test_viai_a.py Models/VIAI_A_inpainting.py Data_loaders/viai_a_loader.py
```

2. 数据准备 smoke test：
```bash
.venv/bin/python main.py prepare-viai-a -- --json data/MUSICES.json --data-root data --max-videos 100 --skip-existing
.venv/bin/python main.py split-data -- --data-root data --audio-only --max-samples 1 --test-size 0 --val-size 0 --allow-empty-eval
```

3. 训练 smoke test：
```bash
.venv/bin/python main.py train-viai-a -- --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
```

4. 测试 smoke test：
```bash
.venv/bin/python main.py test-viai-a -- --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0
```

### 本次实际验证结果
已通过静态检查：
```bash
.venv/bin/python -m py_compile main.py tools/prepare_viai_a.py tools/split_musices.py train_viai_a.py test_viai_a.py Models/VIAI_A_inpainting.py Data_loaders/viai_a_loader.py
```

新入口 help 已通过：
```bash
.venv/bin/python main.py --help
.venv/bin/python main.py prepare-viai-a -- --help
.venv/bin/python main.py split-data -- --help
```

audio-only 数据准备 smoke test 已通过：
```bash
.venv/bin/python main.py prepare-viai-a -- --json data/MUSICES.json --data-root data --max-videos 4 --skip-existing
```

关键输出：
```text
[prepare_viai_a] skipped existing: accordion/yy2vL2RUiPI -> data/processed/accordion/yy2vL2RUiPI
[prepare_viai_a] processed: accordion/A2p8VW61RGc mel_frames=4517 -> data/processed/accordion/A2p8VW61RGc
[prepare_viai_a] summary: missing=2, processed=1, skipped_existing=1
```

audio-only split smoke test 已通过：
```bash
.venv/bin/python main.py split-data -- --data-root data --audio-only --max-samples 1 --test-size 0 --val-size 0 --allow-empty-eval
```

关键输出：
```text
[split_musices] wrote splits: train=1 samples/1 videos (data/train_viai_a_split.txt), val=0 samples/0 videos (data/val_viai_a_split.txt), test=0 samples/0 videos (data/test_viai_a_split.txt)
```

VIAI-A 训练 smoke test 已通过：
```bash
.venv/bin/python main.py train-viai-a -- --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
```

关键输出：
```text
Reached VIAI-A smoke-test max_train_steps=1
[VIAI-A train] loss=0.591076 full_l1=0.288049 missing_l1=0.303027
Saved VIAI-A checkpoint: ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar
Finished VIAI-A training
```

VIAI-A 测试 smoke test 已通过。由于 smoke split 的 `test_viai_a_split.txt` 为空，本次临时使用 `train_viai_a_split.txt` 验证测试路径：
```bash
.venv/bin/python main.py test-viai-a -- --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0
```

关键输出：
```text
[VIAI-A test] samples=1 loss=0.320611 mel_l1_full=0.152091 mel_l1_missing=0.168520 psnr_full=14.181 psnr_missing=13.249 ssim=0.0129
```

依赖锁更新已通过：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv lock
```

关键输出：
```text
Added scikit-image v0.25.2, v0.26.0
```

## 2026-05-02 VIAI-A 训练实时监督增强

### 修改内容
1. 新增 `utils/viai_a_metrics.py`：
   - 抽出 VIAI-A normalized Mel `[0, 1]` 指标计算。
   - 统一计算 full PSNR、missing-region PSNR、SSIM。
   - 保留 `skimage.metrics.structural_similarity` 缺失时的 SSIM fallback。
   - 提供 TensorBoard Mel 图像写入工具：masked input、prediction、target、abs error。
   - 如果当前环境缺少 `Pillow`，Mel 图像写入会给出一次 warning，但不打断训练。
2. `base_options.py`
   - 新增 `--metric_freq`，默认 100，用于训练阶段间隔计算 SSIM。
   - 新增 `--tb_image_freq`，默认 500，用于间隔写入 Mel 对比图。
   - 新增 `--tb_image_count`，默认 4，限制每次写入 TensorBoard 的样本数量。
3. `train_viai_a.py`
   - `tqdm` 实时显示 loss、full/missing PSNR、mask blank length，并在间隔 step 显示 SSIM。
   - TensorBoard 继续写入 loss，同时新增 PSNR、SSIM、blank frames、learning rate。
   - 训练按 `--tb_image_freq` 写入 Mel 对比图；验证每轮第一批写入一组 Mel 对比图。
4. `test_viai_a.py`
   - 删除本地重复 PSNR/SSIM 实现，改用 `utils.viai_a_metrics`。
   - 保持原输出字段兼容：`mel_l1_full`、`mel_l1_missing`、`psnr_full`、`psnr_missing`、`ssim`。
5. `README.md`
   - VIAI-A 训练章节新增 TensorBoard 启动命令。
   - 记录 `--metric_freq`、`--tb_image_freq`、`--tb_image_count` 常用监督参数。
6. `pyproject.toml`
   - 新增 `pillow` 依赖；`tensorboardX` 写 image panel 时需要 `PIL`。
   - 新增 `tensorboard` 依赖；提供 `tensorboard --logdir` CLI 和 event tag 检查工具。

### 验证命令
```bash
python -m py_compile train_viai_a.py test_viai_a.py utils/viai_a_metrics.py
python main.py train-viai-a -- --data_root data --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
tensorboard --logdir checkpoints/events_viai_a
python main.py test-viai-a -- --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0
```

### 待记录 smoke test 关键输出
```text
[VIAI-A train] step=... loss=... full_l1=... missing_l1=... psnr=... psnr_missing=... ssim=...
TensorBoard event log path: ./checkpoints/events_viai_a
[VIAI-A test] samples=... loss=... mel_l1_full=... mel_l1_missing=... psnr_full=... psnr_missing=... ssim=...
```

### 本次实际验证结果
静态检查已通过：
```bash
.venv/bin/python -m py_compile train_viai_a.py test_viai_a.py utils/viai_a_metrics.py
```

依赖锁和本地环境同步已通过：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv lock
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra local-cuda
```

VIAI-A 训练 smoke test 已通过：
```bash
.venv/bin/python main.py train-viai-a -- --data_root data --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
```

关键输出：
```text
[VIAI-A train] epoch=1 ... loss=0.5728 ... psnr=9.11 psnr_miss=9.31 ssim=-0.0016 step=1
Reached VIAI-A smoke-test max_train_steps=1
[VIAI-A train] loss=0.572797 full_l1=0.292290 missing_l1=0.280507 psnr=9.112 psnr_missing=9.310 ssim=-0.001626
Saved VIAI-A checkpoint: ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar
Finished VIAI-A training
```

TensorBoard event tag 检查已通过：
```text
scalars ['train/blank_frames', 'train/loss_full_l1', 'train/loss_missing_l1', 'train/loss_total', 'train/lr', 'train/psnr_full', 'train/psnr_missing', 'train/ssim_full']
images ['train/mel_abs_error', 'train/mel_input_masked', 'train/mel_prediction', 'train/mel_target']
```

TensorBoard CLI 可用：
```bash
.venv/bin/tensorboard --version
```

关键输出：
```text
2.20.0
```

VIAI-A 测试入口已通过：
```bash
.venv/bin/python main.py test-viai-a -- --data_root data --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0
```

关键输出：
```text
[VIAI-A test] samples=1 loss=0.356585 mel_l1_full=0.163478 mel_l1_missing=0.193108 psnr_full=13.548 psnr_missing=12.252 ssim=0.1417
```

## 2026-05-02 VIAI-A 测试结果 JSON/CSV 持久化

### 修改内容
1. `base_options.py`
   - 新增 `--results_dir`，默认 `./checkpoints/viai_a_test_results`。
2. `test_viai_a.py`
   - 每次测试后保存 checkpoint 专属 JSON：
     - `VIAI-A_step000001000_test.json`
   - 同步维护 CSV 总表：
     - `VIAI-A_test_summary.csv`
   - CSV 按 `checkpoint_step` 升序排列。
   - 同一个 checkpoint 重复测试会覆盖旧行，不会追加重复行。
   - 记录字段包括 checkpoint path/step、global step/epoch、test split、样本数、loss、L1、PSNR、SSIM。
3. `README.md`
   - VIAI-A 测试命令增加 `--results_dir` 示例。
   - 说明逐个测试多个 checkpoint 后可用 `VIAI-A_test_summary.csv` 横向比较。

### 验证命令
```bash
.venv/bin/python -m py_compile test_viai_a.py base_options.py
.venv/bin/python main.py test-viai-a -- --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir checkpoints/viai_a_test_results
.venv/bin/python main.py test-viai-a -- --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir checkpoints/viai_a_test_results
```

### 待记录 smoke test 关键输出
```text
[VIAI-A test] wrote json: checkpoints/viai_a_test_results/VIAI-A_step000000001_test.json
[VIAI-A test] wrote summary csv: checkpoints/viai_a_test_results/VIAI-A_test_summary.csv
```

### 本次实际验证结果
静态检查已通过：
```bash
.venv/bin/python -m py_compile test_viai_a.py base_options.py
```

重复测试同一个 checkpoint 已通过，CSV 未产生重复行：
```bash
.venv/bin/python main.py test-viai-a -- --data_root data --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir checkpoints/viai_a_test_results
.venv/bin/python main.py test-viai-a -- --data_root data --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir checkpoints/viai_a_test_results
wc -l checkpoints/viai_a_test_results/VIAI-A_test_summary.csv
```

关键输出：
```text
[VIAI-A test] wrote json: checkpoints/viai_a_test_results/VIAI-A_step000000001_test.json
[VIAI-A test] wrote summary csv: checkpoints/viai_a_test_results/VIAI-A_test_summary.csv
2 checkpoints/viai_a_test_results/VIAI-A_test_summary.csv
```

本地 smoke test 使用随机 mask，同一 checkpoint 重测的 loss/PSNR/SSIM 可能略有变化；CSV 会保留最近一次该 checkpoint 的结果。

## 2026-05-02 VIAI-A 测试 Mel 图片保存

### 修改内容
1. `utils/viai_a_metrics.py`
   - 新增 `save_mel_comparison_png()`，保存单个样本四联图：
     - masked input
     - prediction
     - target
     - abs error
   - 新增 `save_mel_comparison_batch()`，按 batch 批量保存测试样本图片。
   - PNG 使用 normalized Mel `[0, 1]` 映射到灰度 `[0, 255]`。
2. `test_viai_a.py`
   - 测试时为每个样本保存一张 Mel 对比 PNG。
   - 输出目录为：
     - `<results_dir>/mel-image/stepXXXXXXXXX/`
   - 文件名包含全局样本序号和安全化后的 split sample path。
   - 保留原有 JSON/CSV 指标输出不变。
3. `README.md`
   - VIAI-A 测试结果说明中新增 Mel 图片输出路径和四联图内容。

### 验证命令
```bash
.venv/bin/python -m py_compile test_viai_a.py utils/viai_a_metrics.py
.venv/bin/python main.py test-viai-a -- --data_root data --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir checkpoints/viai_a_test_results
.venv/bin/python -c "from PIL import Image; import glob; path=glob.glob('checkpoints/viai_a_test_results/mel-image/step000000001/*.png')[0]; Image.open(path).verify(); print(path)"
```

### 待记录 smoke test 关键输出
```text
[VIAI-A test] wrote mel images: checkpoints/viai_a_test_results/mel-image/step000000001
checkpoints/viai_a_test_results/mel-image/step000000001/000000_*.png
```

### 本次实际验证结果
静态检查已通过：
```bash
.venv/bin/python -m py_compile test_viai_a.py utils/viai_a_metrics.py
```

VIAI-A 测试 smoke test 已通过：
```bash
.venv/bin/python main.py test-viai-a -- --data_root data --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir checkpoints/viai_a_test_results
```

关键输出：
```text
[VIAI-A test] samples=1 loss=0.328249 mel_l1_full=0.163457 mel_l1_missing=0.164792 psnr_full=13.550 psnr_missing=13.643 ssim=0.1415
[VIAI-A test] wrote json: checkpoints/viai_a_test_results/VIAI-A_step000000001_test.json
[VIAI-A test] wrote summary csv: checkpoints/viai_a_test_results/VIAI-A_test_summary.csv
[VIAI-A test] wrote mel images: checkpoints/viai_a_test_results/mel-image/step000000001
```

PNG 输出和 PIL 校验已通过：
```bash
find checkpoints/viai_a_test_results/mel-image/step000000001 -maxdepth 1 -type f -name '*.png'
.venv/bin/python -c "from PIL import Image; import glob; path=glob.glob('checkpoints/viai_a_test_results/mel-image/step000000001/*.png')[0]; Image.open(path).verify(); print(path)"
```

关键输出：
```text
000000_processed_accordion_A2p8VW61RGc.png
checkpoints/viai_a_test_results/mel-image/step000000001/000000_processed_accordion_A2p8VW61RGc.png
```

## 2026-05-03 VIAI-A 测试 Mel PNG 改为 RGB 热力图

### 修改内容
1. `utils/viai_a_metrics.py`
   - `_mel_to_uint8_image()` 从单通道灰度输出改为 `H x W x 3` RGB 输出。
   - 使用内置 magma-like colormap，不新增 `matplotlib` 等依赖。
   - `save_mel_comparison_png()` 改为 `Image.new("RGB", ...)` 和 `Image.fromarray(..., mode="RGB")`。
2. `README.md`
   - 明确 `mel-image/stepXXXXXXXXX/` 下保存的是 RGB 热力图四联图。

### 验证命令
```bash
.venv/bin/python -m py_compile utils/viai_a_metrics.py test_viai_a.py
.venv/bin/python main.py test-viai-a -- --data_root data --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir checkpoints/viai_a_test_results
.venv/bin/python -c "from PIL import Image; import glob; p=glob.glob('checkpoints/viai_a_test_results/mel-image/step000000001/*.png')[0]; img=Image.open(p); print(img.mode, img.size)"
```

### 待记录 smoke test 关键输出
```text
RGB (...)
```

### 本次实际验证结果
静态检查已通过：
```bash
.venv/bin/python -m py_compile utils/viai_a_metrics.py test_viai_a.py
```

VIAI-A 测试 smoke test 已通过：
```bash
.venv/bin/python main.py test-viai-a -- --data_root data --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir checkpoints/viai_a_test_results
```

PNG mode 检查已通过：
```bash
.venv/bin/python -c "from PIL import Image; import glob; p=glob.glob('checkpoints/viai_a_test_results/mel-image/step000000001/*.png')[0]; img=Image.open(p); print(img.mode, img.size, p)"
```

关键输出：
```text
RGB (812, 98) checkpoints/viai_a_test_results/mel-image/step000000001/000000_processed_accordion_A2p8VW61RGc.png
```

## 2026-05-03 VIAI-A 第二阶段 PatchGAN

### 背景与目标
根据 `information.md` 中“8.2 第二阶段：加入 PatchGAN”的要求，本次在 VIAI-A audio-only 基础上加入可选 PatchGAN 训练。默认 `train-viai-a` 仍保持 8.1 的 L1-only baseline；显式传入 `--use_gan` 后启用判别器和 GAN loss，用于提升生成 Mel-spectrogram 的局部纹理真实感。

### 本次修改内容
1. `base_options.py`
   - 新增 `--use_gan`，默认关闭。
   - 继续复用 `--beta_gan`、`--lambda_recon` 和 `--recon_decay_*`。
2. `Models/VIAI_A_inpainting.py`
   - `--use_gan` 开启时实例化 `Discriminator_Networks.MelDiscriminator()`、`GANLoss(use_lsgan=False)` 和 `optimizer_D`。
   - 将 reconstruction loss 拆为 `loss_recon = eta1 * loss_full_l1 + loss_missing_l1`。
   - PatchGAN 训练目标初版误写为 `loss_total = lambda_recon * loss_recon + beta_gan * loss_G_GAN`，后续已按论文第 4 页式 (3) 修正为 `loss_total = loss_G_GAN + beta_gan * loss_recon`。
   - 判别器目标为 `loss_D = 0.5 * (loss_D_real + loss_D_fake)`，fake 分支使用 `mel_pred.detach()`。
   - checkpoint 新增可选 `netD`、`optimizer_D`、`use_gan` 字段。
   - 从 8.1 checkpoint 热启动时允许缺少 `netD/optimizer_D`，判别器随机初始化。
3. `train_viai_a.py`
   - `--use_gan` 且未传 `--name` 时默认使用 `VIAI-A-PatchGAN`。
   - `--use_gan` 且未传 `--log_event_path` 时默认写到 `checkpoints/events_viai_a_patchgan`。
   - 修复 resume 后 `global_step/global_epoch` 被重置的问题。
   - 训练日志和 TensorBoard 增加 `loss_recon`、`loss_g_gan`、`loss_d`、`loss_d_real`、`loss_d_fake`、`eta1`。
4. `test_viai_a.py`
   - `--use_gan` 时测试也默认使用 `VIAI-A-PatchGAN`。
   - JSON/CSV 新增 `use_gan`、`loss_recon`、`loss_g_gan`、`loss_d`、`eta1`、`beta_gan`、`lambda_recon`。
5. `README.md`
   - 新增“第二阶段：加入 PatchGAN”操作流程。
   - 保留 8.1 baseline 命令，并补充 stage2 smoke train、正式 train、test 命令。

### README 中记录的第二阶段命令
1. 8.1 baseline 训练不变：
```bash
python main.py train-viai-a -- --batch_size 16 --num_workers 4 --display_id 0
```

2. 从 8.1 checkpoint 热启动 stage2 smoke test：
```bash
python main.py train-viai-a -- \
  --use_gan \
  --name VIAI-A-PatchGAN \
  --resume \
  --resume_path checkpoints/VIAI-A_checkpoint_step000000001.pth.tar \
  --reset_optimizer \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 2 \
  --display_id 0
```

3. Stage2 正式训练：
```bash
python main.py train-viai-a -- \
  --use_gan \
  --name VIAI-A-PatchGAN \
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

4. Stage2 测试：
```bash
python main.py test-viai-a -- \
  --use_gan \
  --name VIAI-A-PatchGAN \
  --resume_path checkpoints/VIAI-A-PatchGAN_checkpoint_step000002000.pth.tar \
  --batch_size 16 \
  --num_workers 4 \
  --display_id 0 \
  --results_dir checkpoints/viai_a_patchgan_test_results
```

### 本次实际验证结果
静态检查已通过：
```bash
.venv/bin/python -m py_compile base_options.py Models/VIAI_A_inpainting.py train_viai_a.py test_viai_a.py
```

8.1 baseline 回归 smoke test 已通过。为避免覆盖仓库内 checkpoint，本次验证写入 `/tmp/viai_patchgan_smoke_baseline`：
```bash
.venv/bin/python main.py train-viai-a -- --data_root data --checkpoint_dir /tmp/viai_patchgan_smoke_baseline --log_event_path /tmp/viai_patchgan_smoke_baseline/events --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0 --print_freq 1
```

关键输出：
```text
[VIAI-A train] step=1 loss=0.590856 full_l1=0.293740 missing_l1=0.297116 eta1=1.000000 psnr=9.163 psnr_missing=9.029 ssim=0.033071
Saved VIAI-A checkpoint: /tmp/viai_patchgan_smoke_baseline/VIAI-A_checkpoint_step000000001.pth.tar
```

PatchGAN 从旧 8.1 checkpoint 热启动 smoke test 已通过，验证了旧 checkpoint 缺少 `netD/optimizer_D` 时不会报错：
```bash
.venv/bin/python main.py train-viai-a -- --use_gan --data_root data --checkpoint_dir /tmp/viai_patchgan_smoke_gan --log_event_path /tmp/viai_patchgan_smoke_gan/events --resume --resume_path checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --reset_optimizer --batch_size 1 --num_workers 0 --max_train_steps 2 --display_id 0 --print_freq 1
```

关键输出：
```text
[VIAI-A] resumed checkpoint step=1 epoch=0
[VIAI-A train] step=2 loss=0.786060 full_l1=0.358661 missing_l1=0.349788 eta1=0.999895 recon=0.708411 g_gan=0.776496 d=0.714494 psnr=7.454 psnr_missing=7.864
Saved VIAI-A checkpoint: /tmp/viai_patchgan_smoke_gan/VIAI-A-PatchGAN_checkpoint_step000000002.pth.tar
```

PatchGAN 测试入口已通过：
```bash
.venv/bin/python main.py test-viai-a -- --use_gan --data_root data --resume_path /tmp/viai_patchgan_smoke_gan/VIAI-A-PatchGAN_checkpoint_step000000002.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir /tmp/viai_patchgan_smoke_results
```

关键输出：
```text
[VIAI-A test] samples=1 loss=0.400465 recon=0.332187 g_gan=0.682783 d=0.692535 eta1=1.000000 mel_l1_full=0.168928 mel_l1_missing=0.163259 psnr_full=13.279 psnr_missing=13.602 ssim=0.1412
[VIAI-A test] wrote json: /tmp/viai_patchgan_smoke_results/VIAI-A-PatchGAN_step000000002_test.json
[VIAI-A test] wrote summary csv: /tmp/viai_patchgan_smoke_results/VIAI-A-PatchGAN_test_summary.csv
```

测试 JSON 字段检查已通过，包含：
```text
use_gan=true, loss_recon, loss_g_gan, loss_d, eta1, beta_gan, lambda_recon
```

TensorBoard event tag 检查已通过，PatchGAN 标量包含：
```text
train/loss_recon, train/loss_g_gan, train/loss_d, train/loss_d_real, train/loss_d_fake, train/eta1
```

PatchGAN checkpoint 字段检查已通过：
```text
netD, optimizer_D, use_gan, global_step, global_epoch
```

## 2026-05-03 VIAI-A PatchGAN loss 权重方向修正

### 背景
论文第 4 页式 (3) 写作：
```text
L_total^a = L_Gen^a = L_GAN^a + beta * L_re^a
```

因此 VIAI-A PatchGAN 中 β 应当乘在 reconstruction loss 上，而不是乘在 GAN loss 上。当前代码保留 `--beta_gan` 这个历史参数名，但在 VIAI-A PatchGAN 中它对应论文公式里的 β。

### 修改内容
1. `Models/VIAI_A_inpainting.py`
   - 将 `--use_gan` 分支的生成器总损失从：
```python
self.loss_total = lambda_recon * self.loss_recon + beta_gan * self.loss_G_GAN
```
   - 修正为：
```python
self.loss_total = self.loss_G_GAN + beta_gan * self.loss_recon
```
   - L1-only baseline 分支仍保持 `self.loss_total = self.loss_recon`。
2. `README.md`
   - 第二阶段公式改为 `loss_total = loss_g_gan + beta_gan * loss_recon`。
   - 移除 VIAI-A PatchGAN 训练命令中的 `--lambda_recon 1.0`。
   - 增加 `--beta_gan` 在 VIAI-A PatchGAN 中对应论文 β 的说明。

### 验证命令
```bash
.venv/bin/python -m py_compile Models/VIAI_A_inpainting.py train_viai_a.py test_viai_a.py
.venv/bin/python main.py train-viai-a -- --use_gan --data_root data --checkpoint_dir /tmp/viai_patchgan_formula_smoke --log_event_path /tmp/viai_patchgan_formula_smoke/events --resume --resume_path checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --reset_optimizer --batch_size 1 --num_workers 0 --max_train_steps 2 --display_id 0 --print_freq 1
```

### 待记录 smoke test 关键输出
已通过 PatchGAN smoke test：
```bash
.venv/bin/python main.py train-viai-a -- --use_gan --data_root data --checkpoint_dir /tmp/viai_patchgan_formula_smoke --log_event_path /tmp/viai_patchgan_formula_smoke/events --resume --resume_path checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --reset_optimizer --batch_size 1 --num_workers 0 --max_train_steps 2 --display_id 0 --print_freq 1
```

关键输出：
```text
[VIAI-A] resumed checkpoint step=1 epoch=0
[VIAI-A train] step=2 loss=0.683178 full_l1=0.264623 missing_l1=0.249545 eta1=0.999895 recon=0.514140 g_gan=0.631764 d=0.701329 psnr=9.798 psnr_missing=10.176
Saved VIAI-A checkpoint: /tmp/viai_patchgan_formula_smoke/VIAI-A-PatchGAN_checkpoint_step000000002.pth.tar
```

公式核对：
```text
0.631764 + 0.1 * 0.514140 = 0.683178
```

8.1 baseline 回归 smoke test 已通过，不传 `--use_gan` 时日志不包含 `g_gan/d`，仍然只使用 reconstruction loss：
```bash
.venv/bin/python main.py train-viai-a -- --data_root data --checkpoint_dir /tmp/viai_patchgan_formula_baseline --log_event_path /tmp/viai_patchgan_formula_baseline/events --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0 --print_freq 1
```

关键输出：
```text
[VIAI-A train] step=1 loss=0.560637 full_l1=0.289882 missing_l1=0.270755 eta1=1.000000 psnr=9.138 psnr_missing=9.589 ssim=0.009980
Saved VIAI-A checkpoint: /tmp/viai_patchgan_formula_baseline/VIAI-A_checkpoint_step000000001.pth.tar
```
