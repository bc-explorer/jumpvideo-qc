#!/usr/bin/env python3
"""Streamlit UI for live-commerce matting QC."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline import run_qc
from src.resolver import TaskResolver
from src.scanner import blocking_messages, scan_inputs

st.set_page_config(page_title="Video QC", layout="wide")
st.title("直播带货抠图质检识别器")
st.caption("识别 + 报警 only · 不修复 · 不重跑 SAM2 / MatAnyone")

# session state
if "scan_result" not in st.session_state:
    st.session_state.scan_result = None
if "qc_result" not in st.session_state:
    st.session_state.qc_result = None

# --- Task selection ---
st.header("1. 任务选择")
col1, col2 = st.columns([4, 1])
with col1:
    default_root = st.session_state.get("task_root", "")
    task_root = st.text_input(
        "Task Root",
        value=default_root,
        placeholder="G:/code/jumpvideo/ui_runs/person/20260603_xxxx 或本地路径",
    )
with col2:
    st.write("")
    st.write("")
    browse = st.button("Browse (paste path)")

profile = st.selectbox(
    "Profile",
    ["Auto Detect", "live_commerce", "person"],
    index=0,
)
output_video = st.selectbox("Output Video", ["Auto from outputs/manifest.json"], index=0)

if st.button("Scan Task", type="primary"):
    if not task_root.strip():
        st.error("请填写 Task Root")
    else:
        p = Path(task_root.strip()).expanduser()
        if not p.is_dir():
            st.error(f"目录不存在: {p}")
        else:
            st.session_state.task_root = str(p.resolve())
            resolver = TaskResolver(p)
            resolved = resolver.resolve_and_dump()
            scan = scan_inputs(resolved)
            st.session_state.scan_result = {
                "resolved": resolved.to_dict(),
                "input_status": scan.input_status,
                "warnings": scan.warnings,
                "blocking_messages": blocking_messages(scan),
                "profile": resolved.profile,
            }

if st.session_state.scan_result:
    sr = st.session_state.scan_result
    st.success(f"Profile: {sr['profile']}")
    rows = []
    for key, val in sr["input_status"].items():
        st_text = val.get("status", "?")
        detail = ""
        if val.get("frame_count"):
            detail = f", {val['frame_count']} frames"
        rows.append(f"{key:28} {st_text.upper()}{detail}")
    st.code("\n".join(rows))
    for msg in sr.get("blocking_messages", []):
        st.warning(msg)

# --- Detection mode ---
st.header("2. 检测强度")
mode = st.radio(
    "Detection Mode",
    ["Conservative", "Balanced", "Sensitive"],
    index=2,
    horizontal=True,
    help="Sensitive: 误报多、漏报少（默认）",
)
mode_key = mode.lower()

# --- Subtitle ---
st.header("3. 字幕处理")
c1, c2 = st.columns(2)
with c1:
    use_sub_mask = st.checkbox("Use upstream subtitles mask if exists", value=True)
    ignore_lower = st.checkbox("Ignore lower third region", value=True)
    lower_y0 = st.slider("Lower third start (%)", 50, 95, 72) / 100.0
with c2:
    ignore_top = st.checkbox("Ignore top banner region", value=True)
    top_y1 = st.slider("Top banner end (%)", 5, 30, 12) / 100.0

# --- Run ---
st.header("4. 运行和结果")
btn_run, btn_open, btn_json, btn_csv = st.columns(4)

def _build_config(task_path: Path) -> dict:
    cfg = load_config(task_path, mode=mode_key)
    cfg.data["mode"] = mode_key
    prof = profile.replace(" ", "_").lower()
    if prof == "auto_detect":
        prof = "auto"
    cfg.task["profile"] = prof
    cfg.task["output_video"] = "auto"
    cfg.subtitle.update(
        {
            "use_upstream_subtitle_mask": use_sub_mask,
            "ignore_lower_third": ignore_lower,
            "lower_third_region": [0.0, lower_y0, 1.0, 1.0],
            "ignore_top_banner": ignore_top,
            "top_banner_region": [0.0, 0.0, 1.0, top_y1],
        }
    )
    return cfg


with btn_run:
    run_clicked = st.button("Run QC", type="primary")
with btn_open:
    open_clicked = st.button("Open Report")
with btn_json:
    export_json = st.button("Export JSON")
with btn_csv:
    export_csv = st.button("Export Failed Segments CSV")

if run_clicked:
    if not task_root.strip():
        st.error("请先填写 Task Root 并 Scan")
    else:
        task_path = Path(task_root.strip()).expanduser().resolve()
        cfg = _build_config(task_path)
        cfg.dump_yaml(task_path / "qc_config.yaml")
        with st.spinner("Running QC..."):
            try:
                result = run_qc(task_path, config=cfg, write_config=False)
                st.session_state.qc_result = result
            except Exception as e:
                st.exception(e)

if st.session_state.qc_result:
    r = st.session_state.qc_result
    st.subheader(
        f"Decision: **{r['decision']}** · Risk Level: **{r['risk_level']}**"
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("High risks", r["summary"]["high"])
    m2.metric("Medium risks", r["summary"]["medium"])
    m3.metric("Warnings", r["summary"]["warning"])

    segs = r.get("failed_segments", [])
    if segs:
        st.markdown("#### 失败片段")
        for s in segs[:50]:
            line = (
                f"{s.get('start_tc')} - {s.get('end_tc')}  "
                f"{s.get('type')}  {s.get('severity', '').upper()}"
            )
            st.text(line)
            if s.get("preview") and Path(s["preview"]).is_file():
                st.image(s["preview"], width=480)
    else:
        st.info("无失败片段")

    qc_dir = Path(r["task_root"]) / "qc"
    if open_clicked and (qc_dir / "report.html").is_file():
        if sys.platform == "darwin":
            subprocess.run(["open", str(qc_dir / "report.html")], check=False)
        elif os.name == "nt":
            os.startfile(str(qc_dir / "report.html"))  # type: ignore
        else:
            st.markdown(f"[Open report](file://{qc_dir / 'report.html'})")

    if export_json and (qc_dir / "report.json").is_file():
        st.download_button(
            "Download report.json",
            data=(qc_dir / "report.json").read_text(encoding="utf-8"),
            file_name="report.json",
            mime="application/json",
        )
    if export_csv and (qc_dir / "failed_segments.csv").is_file():
        st.download_button(
            "Download failed_segments.csv",
            data=(qc_dir / "failed_segments.csv").read_text(encoding="utf-8"),
            file_name="failed_segments.csv",
            mime="text/csv",
        )

with st.expander("生成的 qc_config.yaml 预览"):
    if task_root.strip() and Path(task_root.strip()).expanduser().is_dir():
        cfg = _build_config(Path(task_root.strip()).expanduser().resolve())
        st.code(yaml.safe_dump(cfg.to_dict(), allow_unicode=True), language="yaml")
