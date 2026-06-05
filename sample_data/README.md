# Sample task data

把上游任务目录**完整复制**到本目录下，例如：

```text
sample_data/
  20260603_174958_5cc76120/    # 人物样例（你稍后放入）
    input/source.mp4
    frames/source/
    foreground/matanyone2/combined_alpha/   # 或 foreground/person/combined_alpha/
    matting/person/source_person/pha/
    masks/combined/...
    prompts/...
    outputs/manifest.json
```

## 扫描（不跑模型）

```bash
python scripts/scan_task.py sample_data/20260603_174958_5cc76120
```

## 跑质检

```bash
python scripts/run_qc.py sample_data/20260603_174958_5cc76120 --mode sensitive
```

## Streamlit UI

```bash
streamlit run app.py
```

Task Root 填：`/Users/.../Projects/video_qc_fast/sample_data/<task_id>`
