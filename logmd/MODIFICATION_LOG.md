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
uv run --extra local-cuda python main.py prepare-data -- splits --json data/MUSICES.json --data-root data --max-videos 1
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
.venv/bin/python main.py train -- --batch_size 16 --num_workers 4 --display_id 0
```

云端训练前需要重新确认：
1. CUDA 版 PyTorch 与云端驱动匹配。
2. `opencv-contrib-python` 的 TV-L1 接口可用。
3. 完整数据已经完成 `process` 和 `splits`。
4. checkpoint、TensorBoard 日志、retrieval 指标可以正常写入。
5. 先跑 100-500 step sanity training，再启动长训练。

注意：上面的 `cu121` 只是示例，云端需按服务器实际 CUDA/驱动选择 PyTorch wheel。手动安装云端 PyTorch 后，建议用 `.venv/bin/python` 或 `uv run --no-sync` 运行，避免 `uv run` 自动同步时移除手动安装的云端 torch。

### 已知仍未完整复现的部分
1. 论文中的 shot detection、去除非演奏/黑场片段、裁掉每个视频前 6 秒，当前 pipeline 尚未完整自动化。
2. 论文的 10% fixed test + 5% held-out validation 协议尚未完整接入训练 loader。
3. VIAI-AA' probe loss 和 WaveNet spectrogram-to-audio 端到端评估仍需后续补齐。
4. 本地 smoke test 只证明链路可运行，不代表模型收敛或论文指标。
