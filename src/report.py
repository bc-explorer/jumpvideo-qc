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
        ev = esc(json.dumps(d["evidence"], ensure_ascii=False))
        seg_cards.append(
            f"""
            <div class='card' data-sev='{esc(sev)}'>
              <div class='card-head'>
                <span class='badge' style='background:{color}'>{esc(sev_zh)}</span>
                <span class='tc'>{esc(d['start_tc'])} - {esc(d['end_tc'])}</span>
              </div>
              <div class='cname'>{esc(name_zh)} <code>{esc(d['type'])}</code></div>
              {img_html}
              <div class='cbody'>
                <div class='desc'>{esc(desc_zh)}</div>
                <div class='hint'><b>复核建议：</b>{esc(hint_zh)}</div>
                <div class='src'>问题来源：<code>{esc(d['source'])}</code> &middot; 命中 {d['finding_count']} 帧</div>
                <details><summary>证据 evidence</summary><pre class='ev'>{ev}</pre></details>
              </div>
            </div>
            """
        )
    seg_html = "\n".join(seg_cards) or "<p>无失败片段。</p>"

    perf_html = ""
    if perf:
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
  </table>"""

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
  body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", Segoe UI, Roboto, sans-serif; margin:0; background:#0f1115; color:#e6e6e6; }}
  header {{ padding:20px 24px; background:#161a22; border-bottom:1px solid #262b36; }}
  h1 {{ margin:0 0 6px; font-size:20px; }}
  h2 {{ font-size:16px; margin:18px 0 8px; }}
  .meta {{ color:#9aa4b2; font-size:13px; line-height:1.6; }}
  .summary {{ display:flex; gap:14px; padding:18px 24px; flex-wrap:wrap; align-items:stretch; }}
  .pill {{ background:#1c2230; padding:12px 16px; border-radius:10px; min-width:120px; }}
  .pill .n {{ font-size:24px; font-weight:700; }}
  .pill .l {{ font-size:12px; color:#9aa4b2; }}
  .decision {{ font-size:18px; font-weight:700; }}
  .risk-HIGH {{ color:#ff6b6b; }} .risk-MEDIUM {{ color:#ffa94d; }}
  .risk-LOW {{ color:#ffd43b; }} .risk-NONE {{ color:#51cf66; }}
  .verdict {{ padding:0 24px; color:#cbd3df; font-size:14px; }}
  section {{ padding:8px 24px 24px; }}
  table {{ border-collapse:collapse; width:100%; max-width:560px; font-size:13px; }}
  table.wide {{ max-width:920px; }}
  td, th {{ border:1px solid #262b36; padding:6px 10px; text-align:left; vertical-align:top; }}
  th {{ background:#1c2230; }}
  .st-ok {{ color:#51cf66; }} .st-missing {{ color:#ff6b6b; }}
  code {{ background:#0c0f14; padding:1px 5px; border-radius:4px; font-size:11px; color:#9aa4b2; }}
  .filters {{ padding:0 24px; }}
  .filters button {{ background:#1c2230; color:#e6e6e6; border:1px solid #2c3340; border-radius:8px; padding:6px 12px; margin-right:8px; cursor:pointer; font-size:13px; }}
  .filters button.active {{ background:#2d3a52; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:16px; }}
  .card {{ background:#161a22; border:1px solid #262b36; border-radius:10px; overflow:hidden; }}
  .card img {{ width:100%; display:block; }}
  .card-head {{ display:flex; gap:8px; align-items:center; padding:10px 10px 0; }}
  .badge {{ color:#fff; padding:2px 8px; border-radius:6px; font-size:12px; font-weight:700; }}
  .tc {{ color:#9aa4b2; font-size:12px; }}
  .cname {{ padding:6px 10px; font-weight:700; font-size:15px; }}
  .cbody {{ padding:8px 10px 12px; }}
  .desc {{ font-size:13px; color:#d6dbe3; margin-bottom:6px; }}
  .hint {{ font-size:12.5px; color:#ffd9a8; margin-bottom:6px; }}
  .src {{ color:#9aa4b2; font-size:12px; margin-bottom:6px; }}
  details summary {{ cursor:pointer; color:#7aa2ff; font-size:12px; }}
  pre.ev {{ margin:6px 0 0; padding:8px; background:#0c0f14; font-size:11px; color:#9aa4b2; white-space:pre-wrap; word-break:break-word; border-radius:6px; }}
  ul {{ margin:6px 0; }} ul.blocking li {{ color:#ffb3b3; }}
  .legend {{ padding:0 24px 8px; color:#9aa4b2; font-size:12px; }}
  .legend .lg {{ margin-left:12px; }}
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
  {perf_html}
</section>
<section>
  <h2>输入状态</h2>
  <table><tr><th>产物</th><th>状态</th><th>帧数</th></tr>{rows_status}</table>
  {block_section}
  <h2>告警类型概览</h2>
  {type_table}
  <h2>提示 / 警告</h2>
  <ul>{warn_html}</ul>
</section>
<section>
  <h2>失败片段（{len(segments)}）</h2>
  {legend}
  <div class='filters'>
    <button data-f='all' class='active' onclick='flt(this,"all")'>全部</button>
    <button data-f='high' onclick='flt(this,"high")'>仅高危</button>
    <button data-f='medium' onclick='flt(this,"medium")'>仅中等</button>
    <button data-f='warning' onclick='flt(this,"warning")'>仅警告</button>
  </div>
  <div class='grid' id='grid'>{seg_html}</div>
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
