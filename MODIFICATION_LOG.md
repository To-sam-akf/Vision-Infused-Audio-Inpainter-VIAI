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

