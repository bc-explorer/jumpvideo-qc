"""Report writers: report.json, failed_segments.csv, report.html."""
from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path
from typing import Dict, List

from . import descriptions
from .findings import HIGH, MEDIUM, SEVERITY_RANK, WARNING
from .segment_merge import Segment

SEV_BADGE = {
    HIGH: ("HIGH", "#c0392b"),
    MEDIUM: ("MEDIUM", "#e67e22"),
    WARNING: ("WARNING", "#f1c40f"),
}


def summarize(segments: List[Segment]) -> Dict:
    counts = {HIGH: 0, MEDIUM: 0, WARNING: 0}
    by_type: Dict[str, int] = {}
    for s in segments:
        counts[s.severity] = counts.get(s.severity, 0) + 1
        by_type[s.type] = by_type.get(s.type, 0) + 1
    return {
        "high": counts.get(HIGH, 0),
        "medium": counts.get(MEDIUM, 0),
        "warning": counts.get(WARNING, 0),
        "segments": len(segments),
        "by_type": by_type,
    }


def decide(summary: Dict) -> Dict:
    if summary["high"] > 0:
        return {"decision": "REVIEW", "risk_level": "HIGH"}
    if summary["medium"] > 0:
        return {"decision": "REVIEW", "risk_level": "MEDIUM"}
    if summary["warning"] > 0:
        return {"decision": "PASS", "risk_level": "LOW"}
    return {"decision": "PASS", "risk_level": "NONE"}


def write_json(result: Dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def write_csv(segments: List[Segment], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "start_tc",
                "end_tc",
                "start_time",
                "end_time",
                "type",
                "severity",
                "source",
                "finding_count",
                "preview",
            ]
        )
        for s in segments:
            d = s.to_dict()
            w.writerow(
                [
                    d["id"],
                    d["start_tc"],
                    d["end_tc"],
                    d["start_time"],
                    d["end_time"],
                    d["type"],
                    d["severity"],
                    d["source"],
                    d["finding_count"],
                    os.path.basename(d["preview"]) if d["preview"] else "",
                ]
            )


def _rel_preview(preview: str, report_dir: Path) -> str:
    try:
        return os.path.relpath(preview, report_dir)
    except ValueError:
        return preview


DECISION_ZH = {"REVIEW": "需人工复核", "PASS": "通过"}
RISK_ZH = {"HIGH": "高", "MEDIUM": "中", "LOW": "低", "NONE": "无"}
STATUS_ZH = {"ok": "正常", "missing": "缺失"}


EVIDENCE_LABELS = {
    "person_rank": "人物序号",
    "box": "证据框",
    "roi": "关注区域",
    "yolo_conf": "YOLO 置信度",
    "coverage_basis": "覆盖率依据",
    "person_combined_coverage": "最终 alpha 覆盖率",
    "person_human_coverage": "人像 alpha 覆盖率",
    "assistant_mask_coverage": "助播 mask 覆盖率",
    "face_coverage": "人脸覆盖率",
    "hand_coverage": "手部覆盖率",
    "combined_coverage": "最终 alpha 覆盖率",
    "area": "当前面积",
    "reference_median": "前后参考面积",
    "drop_ratio": "下降比例",
    "object": "对象",
    "center_jump_px": "中心跳变距离",
    "threshold_px": "阈值",
    "prev_frame": "上一帧",
    "blob_count": "连通块数量",
    "iou_prev": "相邻帧 IoU",
    "threshold": "阈值",
    "person_count": "人物数量",
    "raw_detections": "原始检测数量",
    "source": "证据来源",
}


def _evidence_items(evidence: Dict) -> str:
    if not evidence:
        return "<span class='muted'>无额外证据</span>"
    items = []
    for key, value in evidence.items():
        label = EVIDENCE_LABELS.get(key, key)
        if isinstance(value, float):
            text = f"{value:.3f}"
        elif isinstance(value, (list, tuple)):
            text = ", ".join(str(v) for v in value)
        else:
            text = str(value)
        items.append(f"<span class='evitem'><b>{html.escape(label)}</b>{html.escape(text)}</span>")
    return "".join(items)


def _segment_anchor(seg_id: int) -> str:
    return f"seg-{seg_id:03d}"


def write_html(result: Dict, segments: List[Segment], path: Path) -> None:
    path = Path(path)
    report_dir = path.parent
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = result["summary"]
    decision = result["decision"]
    risk = result["risk_level"]
    perf = result.get("performance", {})
    vmeta = result.get("video_meta", {})

    def esc(x):
        return html.escape(str(x))

    # --- input status table (Chinese labels) ---
    rows_status = "".join(
        f"<tr><td>{esc(descriptions.artifact_zh(k))}</td>"
        f"<td class='st-{v.get('status')}'>{esc(STATUS_ZH.get(v.get('status'), v.get('status')))}</td>"
        f"<td>{esc(v.get('frame_count','')) if v.get('frame_count') else ''}</td></tr>"
        for k, v in result.get("input_status", {}).items()
    )

    # --- warnings (map known codes to Chinese names) ---
    warn_items = []
    for w in result.get("warnings", []):
        code = w.get("warning", "")
        zh, _, _ = descriptions.alarm_zh(code)
        extra = (
            f"（帧 {esc(w.get('frame_index'))}）"
            if w.get("frame_index") is not None
            else ""
        )
        name = zh if zh != code else code
        warn_items.append(f"<li>{esc(name)} <code>{esc(code)}</code>{extra}</li>")
    warn_html = "".join(warn_items) or "<li>无</li>"

    # --- blocking messages ---
    block_html = "".join(
        f"<li>{esc(m)}</li>" for m in result.get("blocking_messages", [])
    )
    block_section = (
        f"<h2>受限提示</h2><ul class='blocking'>{block_html}</ul>" if block_html else ""
    )

    # --- by-type overview (Chinese) ---
    by_type = summary.get("by_type", {})
    type_rows = ""
    for t, c in sorted(by_type.items(), key=lambda kv: -kv[1]):
        zh, desc, _ = descriptions.alarm_zh(t)
        type_rows += (
            f"<tr><td>{esc(zh)}</td><td><code>{esc(t)}</code></td>"
            f"<td>{c}</td><td>{esc(desc)}</td></tr>"
        )
    type_table = (
        "<table class='wide'><tr><th>告警</th><th>类型代码</th><th>片段数</th>"
        f"<th>说明</th></tr>{type_rows}</table>"
        if type_rows
        else "<p>无</p>"
    )

    # --- failed segment cards (severity desc, then time) ---
    ordered = sorted(
        segments,
        key=lambda s: (-SEVERITY_RANK.get(s.severity, 0), s.start_frame),
    )
    fps = float(vmeta.get("fps") or 0.0)
    frame_count = int(vmeta.get("frame_count") or perf.get("total_frames") or 0)
    timeline_max = max(frame_count - 1, 1)

    focus_rows = []
    for s in ordered[:12]:
        d = s.to_dict()
        sev_zh, _ = descriptions.severity_zh(d["severity"])
        name_zh, desc_zh, _ = descriptions.alarm_zh(d["type"])
        focus_rows.append(
            f"<tr class='row-{esc(d['severity'])}'>"
            f"<td><span class='dot dot-{esc(d['severity'])}'></span>{esc(sev_zh)}</td>"
            f"<td><a href='#{_segment_anchor(d['id'])}'>{esc(d['start_tc'])} - {esc(d['end_tc'])}</a></td>"
            f"<td>{esc(d['start_frame'])} - {esc(d['end_frame'])}<span class='muted'> / 峰值 {esc(d['peak_frame'])}</span></td>"
            f"<td><b>{esc(name_zh)}</b><br/><span class='muted'>{esc(desc_zh)}</span></td>"
            f"</tr>"
        )
    focus_table = (
        "<table class='review-table'><tr><th>等级</th><th>时间点</th><th>帧号</th><th>质量问题</th></tr>"
        + "".join(focus_rows)
        + "</table>"
        if focus_rows
        else "<p class='empty'>暂无需要人工复核的片段。</p>"
    )

    timeline_items = []
    for s in ordered:
        d = s.to_dict()
        left = max(0.0, min(99.2, (s.peak_frame / timeline_max) * 100.0))
        name_zh, _, _ = descriptions.alarm_zh(d["type"])
        timeline_items.append(
            f"<a class='tick tick-{esc(d['severity'])}' href='#{_segment_anchor(d['id'])}' "
            f"style='left:{left:.3f}%' title='{esc(d['start_tc'])} {esc(name_zh)}'>"
            f"<span>{esc(d['peak_frame'])}</span></a>"
        )
    timeline_html = (
        "<div class='timeline'><div class='track'></div>"
        + "".join(timeline_items)
        + "</div>"
        if timeline_items
        else "<p class='empty'>未发现异常时间点。</p>"
    )

    seg_cards = []
    for s in ordered:
        d = s.to_dict()
        sev = d["severity"]
        sev_zh, color = descriptions.severity_zh(sev)
        name_zh, desc_zh, hint_zh = descriptions.alarm_zh(d["type"])
        img_html = ""
        if d["preview"]:
            rel = _rel_preview(d["preview"], report_dir)
            img_html = f"<img src='{esc(rel)}' loading='lazy'/>"
        ev = esc(json.dumps(d["evidence"], ensure_ascii=False, indent=2))
        anchor = _segment_anchor(d["id"])
        seg_cards.append(
            f"""
            <article class='card' id='{anchor}' data-sev='{esc(sev)}'>
              <div class='card-head'>
                <span class='badge' style='background:{color}'>{esc(sev_zh)}</span>
                <div>
                  <div class='cname'>{esc(name_zh)} <code>{esc(d['type'])}</code></div>
                  <div class='tc'>{esc(d['start_tc'])} - {esc(d['end_tc'])} · 帧 {esc(d['start_frame'])}-{esc(d['end_frame'])} · 峰值帧 {esc(d['peak_frame'])}</div>
                </div>
              </div>
              {img_html}
              <div class='cbody'>
                <div class='desc'>{esc(desc_zh)}</div>
                <div class='hint'><b>复核建议：</b>{esc(hint_zh)}</div>
                <div class='facts'>
                  <span><b>命中帧数</b>{esc(d['finding_count'])}</span>
                  <span><b>问题来源</b><code>{esc(d['source'])}</code></span>
                  <span><b>阶段</b><code>{esc(s.source)}</code></span>
                </div>
                <div class='evidence'>{_evidence_items(d.get('evidence') or {})}</div>
                <details><summary>证据 evidence</summary><pre class='ev'>{ev}</pre></details>
              </div>
            </article>
            """
        )
    seg_html = "\n".join(seg_cards) or "<p>无失败片段。</p>"

    perf_html = ""
    if perf:
        stage_rows = ""
        stage_labels = {
            "setup": "准备 / 扫描",
            "mask_io": "mask 读取与统计",
            "yolo": "YOLO 读帧与推理",
            "sam2": "SAM2 时序分析",
            "preview": "预览图生成",
            "report": "报告写出",
        }
        for key, label in stage_labels.items():
            if key in perf.get("stage_seconds", {}):
                stage_rows += f"<tr><td>{esc(label)}</td><td>{esc(perf['stage_seconds'].get(key))} s</td></tr>"
        stage_table = (
            "<h3>阶段耗时</h3><table class='wide'><tr><th>阶段</th><th>耗时</th></tr>"
            f"{stage_rows}</table>"
            if stage_rows
            else ""
        )
        perf_html = f"""
  <h2>性能报告</h2>
  <table class='wide'>
    <tr><th>指标</th><th>值</th></tr>
    <tr><td>视频时长</td><td>{esc(perf.get('video_duration_seconds'))} s（{esc(perf.get('total_frames'))} 帧 @ {esc(perf.get('fps'))} fps）</td></tr>
    <tr><td>实际处理帧</td><td>{esc(perf.get('sampled_frames'))} 帧（采样步长 {esc(result.get('frame_stride'))}）</td></tr>
    <tr><td>读取数据量</td><td>{esc(perf.get('processed_mb'))} MB</td></tr>
    <tr><td>总耗时</td><td>{esc(perf.get('elapsed_seconds'))} s</td></tr>
    <tr><td>数据吞吐</td><td><b>{esc(perf.get('throughput_mb_s'))} MB/s</b></td></tr>
    <tr><td>处理速度</td><td>{esc(perf.get('frames_per_second'))} 帧/s</td></tr>
    <tr><td>实时倍率</td><td>{esc(perf.get('realtime_factor'))}x（视频时长 / 处理耗时）</td></tr>
    <tr><td>YOLO 策略</td><td>{esc(perf.get('yolo_policy', 'all'))}，实际 YOLO 帧 {esc(perf.get('yolo_frames', ''))}/{esc(perf.get('sampled_frames', ''))}（{esc(perf.get('yolo_frame_ratio', ''))}）</td></tr>
  </table>
  {stage_table}"""

    legend = (
        "<div class='legend'>预览图例："
        "<span class='lg' style='color:#00c800'>● 人体实例掩膜</span>"
        "<span class='lg' style='color:#00ffff'>● combined alpha 边界</span>"
        "<span class='lg' style='color:#ff69b4'>● human alpha 边界</span>"
        "<span class='lg' style='color:#ffd700'>● 人脸 ROI</span>"
        "<span class='lg' style='color:#0064ff'>● 手部点</span>"
        "<span class='lg' style='color:#ff0000'>● 疑似商品/证据框</span>"
        "</div>"
    )

    doc = f"""<!doctype html>
<html lang='zh-CN'><head><meta charset='utf-8'/>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>抠图质检报告</title>
<style>
  * {{ box-sizing:border-box; }}
  html {{ scroll-behavior:smooth; }}
  body {{ font-family:-apple-system, "PingFang SC", "Microsoft YaHei", Segoe UI, Roboto, sans-serif; margin:0; background:#111318; color:#edf0f4; }}
  header {{ padding:22px 28px; background:#191d25; border-bottom:1px solid #2b313d; position:sticky; top:0; z-index:5; }}
  h1 {{ margin:0 0 8px; font-size:22px; letter-spacing:0; }}
  h2 {{ font-size:17px; margin:0 0 12px; }}
  h3 {{ font-size:14px; margin:0 0 8px; color:#cfd6e2; }}
  a {{ color:#8bb2ff; text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
  .meta {{ color:#a7b0bf; font-size:13px; line-height:1.6; word-break:break-all; }}
  .summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(136px,1fr)); gap:12px; padding:18px 28px 8px; }}
  .pill {{ background:#1b202b; border:1px solid #2c3442; padding:14px 16px; border-radius:8px; min-height:78px; }}
  .pill .n {{ font-size:28px; font-weight:750; line-height:1.1; }}
  .pill .l {{ font-size:12px; color:#a7b0bf; margin-bottom:8px; }}
  .decision {{ font-size:18px; font-weight:750; }}
  .risk-HIGH {{ color:#ff6961; }} .risk-MEDIUM {{ color:#ffb15c; }}
  .risk-LOW {{ color:#ffd45a; }} .risk-NONE {{ color:#5fd185; }}
  .verdict {{ padding:4px 28px 18px; color:#cfd6e2; font-size:14px; }}
  section {{ padding:18px 28px; }}
  .panel {{ background:#171b23; border:1px solid #2b313d; border-radius:8px; padding:16px; margin-bottom:16px; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  table.wide, .review-table {{ max-width:none; }}
  td, th {{ border-bottom:1px solid #2b313d; padding:9px 10px; text-align:left; vertical-align:top; }}
  th {{ color:#aeb7c6; font-weight:650; background:#202633; }}
  tr:last-child td {{ border-bottom:0; }}
  .st-ok {{ color:#5fd185; }} .st-missing {{ color:#ff6961; }}
  code {{ background:#0d1016; padding:2px 5px; border-radius:4px; font-size:11px; color:#aeb7c6; word-break:break-all; }}
  .muted {{ color:#a7b0bf; font-size:12px; line-height:1.45; }}
  .empty {{ color:#a7b0bf; margin:0; }}
  .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:8px; }}
  .dot-high, .tick-high {{ background:#ff5148; }}
  .dot-medium, .tick-medium {{ background:#ff9f40; }}
  .dot-warning, .tick-warning {{ background:#f5d45f; }}
  .timeline {{ position:relative; height:70px; margin:10px 4px 2px; }}
  .track {{ position:absolute; left:0; right:0; top:32px; height:8px; border-radius:8px; background:#2b313d; }}
  .tick {{ position:absolute; top:18px; width:18px; height:36px; border-radius:7px; border:2px solid #111318; box-shadow:0 0 0 1px rgba(255,255,255,.18); }}
  .tick span {{ position:absolute; top:39px; left:50%; transform:translateX(-50%); color:#a7b0bf; font-size:10px; white-space:nowrap; }}
  .filters {{ display:flex; flex-wrap:wrap; gap:8px; margin:0 0 12px; }}
  .filters button {{ background:#202633; color:#edf0f4; border:1px solid #354052; border-radius:7px; padding:7px 12px; cursor:pointer; font-size:13px; }}
  .filters button.active {{ background:#34425a; border-color:#6e8ccc; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(390px,1fr)); gap:16px; align-items:start; }}
  .card {{ background:#171b23; border:1px solid #2b313d; border-radius:8px; overflow:hidden; scroll-margin-top:120px; }}
  .card:target {{ border-color:#8bb2ff; box-shadow:0 0 0 2px rgba(139,178,255,.25); }}
  .card img {{ width:100%; display:block; background:#0d1016; }}
  .card-head {{ display:grid; grid-template-columns:auto 1fr; gap:10px; align-items:start; padding:12px 12px 8px; }}
  .badge {{ color:#fff; padding:4px 8px; border-radius:6px; font-size:12px; font-weight:750; }}
  .tc {{ color:#a7b0bf; font-size:12px; line-height:1.45; }}
  .cname {{ font-weight:750; font-size:16px; margin-bottom:5px; }}
  .cbody {{ padding:12px; }}
  .desc {{ font-size:14px; color:#e3e7ee; margin-bottom:8px; line-height:1.55; }}
  .hint {{ font-size:13px; color:#ffd49a; margin-bottom:10px; line-height:1.55; }}
  .facts {{ display:flex; flex-wrap:wrap; gap:8px; margin:8px 0; }}
  .facts span, .evitem {{ display:inline-flex; align-items:center; gap:6px; background:#202633; border:1px solid #2f3746; border-radius:6px; padding:6px 8px; color:#cfd6e2; font-size:12px; }}
  .facts b, .evitem b {{ color:#f0f3f8; font-weight:650; }}
  .evidence {{ display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 10px; }}
  details summary {{ cursor:pointer; color:#8bb2ff; font-size:12px; }}
  pre.ev {{ margin:8px 0 0; padding:10px; background:#0d1016; font-size:11px; color:#aeb7c6; white-space:pre-wrap; word-break:break-word; border-radius:6px; }}
  ul {{ margin:6px 0; }} ul.blocking li {{ color:#ffb3b3; }}
  .legend {{ color:#a7b0bf; font-size:12px; margin-bottom:10px; line-height:1.7; }}
  .legend .lg {{ margin-right:14px; white-space:nowrap; }}
  .support {{ display:grid; grid-template-columns:minmax(280px,1fr) minmax(280px,1fr); gap:16px; }}
  @media (max-width:760px) {{
    header {{ position:static; padding:18px; }}
    .summary, section {{ padding-left:18px; padding-right:18px; }}
    .grid, .support {{ grid-template-columns:1fr; }}
  }}
</style></head>
<body>
<header>
  <h1>直播带货抠图质检报告</h1>
  <div class='meta'>任务：{esc(result.get('task_root',''))}<br/>
  类型：{esc(result.get('profile',''))} &middot; 检测模式：{esc(result.get('mode',''))} &middot;
  分辨率：{esc(vmeta.get('width',''))}×{esc(vmeta.get('height',''))} &middot;
  帧率：{esc(vmeta.get('fps',''))} fps &middot; 总帧数：{esc(vmeta.get('frame_count',''))}</div>
</header>
<div class='summary'>
  <div class='pill'><div class='l'>结论</div><div class='decision'>{esc(DECISION_ZH.get(decision, decision))}</div></div>
  <div class='pill'><div class='l'>风险等级</div><div class='decision risk-{esc(risk)}'>{esc(RISK_ZH.get(risk, risk))}</div></div>
  <div class='pill'><div class='n'>{summary['high']}</div><div class='l'>高危</div></div>
  <div class='pill'><div class='n'>{summary['medium']}</div><div class='l'>中等</div></div>
  <div class='pill'><div class='n'>{summary['warning']}</div><div class='l'>警告</div></div>
  <div class='pill'><div class='n'>{summary.get('segments',0)}</div><div class='l'>失败片段</div></div>
</div>
<div class='verdict'>判定规则：出现高危即「需人工复核（高）」；仅中等即「需人工复核（中）」；仅警告为「通过（低）」；无异常为「通过」。</div>
<section>
  <div class='panel'>
    <h2>异常时间轴</h2>
    {timeline_html}
  </div>
  <div class='panel'>
    <h2>复核重点</h2>
    {focus_table}
  </div>
</section>
<section>
  <h2>失败片段（{len(segments)}）</h2>
  <div class='filters'>
    <button data-f='all' class='active' onclick='flt(this,"all")'>全部</button>
    <button data-f='high' onclick='flt(this,"high")'>仅高危</button>
    <button data-f='medium' onclick='flt(this,"medium")'>仅中等</button>
    <button data-f='warning' onclick='flt(this,"warning")'>仅警告</button>
  </div>
  {legend}
  <div class='grid' id='grid'>{seg_html}</div>
</section>
<section class='support'>
  <div class='panel'>
    <h2>告警类型概览</h2>
    {type_table}
  </div>
  <div class='panel'>
    <h2>提示 / 警告</h2>
    <ul>{warn_html}</ul>
    {block_section}
  </div>
  <div class='panel'>
    <h2>输入状态</h2>
    <table><tr><th>产物</th><th>状态</th><th>帧数</th></tr>{rows_status}</table>
  </div>
  <div class='panel'>
    {perf_html}
  </div>
</section>
<script>
function flt(btn, sev) {{
  document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('#grid .card').forEach(c => {{
    c.style.display = (sev === 'all' || c.dataset.sev === sev) ? '' : 'none';
  }});
}}
</script>
</body></html>"""

    with path.open("w", encoding="utf-8") as f:
        f.write(doc)
