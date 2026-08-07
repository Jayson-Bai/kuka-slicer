"""Stable JSON and self-contained HTML rendering for validation reports."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

from .validator import SurfaceValidationReport


def render_html_report(report: SurfaceValidationReport) -> str:
    """Render the report without external assets so it can be archived directly."""

    payload = report.payload()
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(check['name'])}</td>"
        f"<td class=\"{escape(check['status'])}\">{escape(check['status'])}</td>"
        f"<td>{escape(check['summary'])}</td>"
        f"<td><pre>{escape(json.dumps(check['details'], ensure_ascii=False, indent=2))}</pre></td>"
        "</tr>"
        for check in payload["checks"]
    )
    return f"""<!doctype html>
<html lang=\"zh-CN\">
<meta charset=\"utf-8\">
<title>曲面路径可打印性验证报告</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 32px; color: #1f2937; }}
h1 {{ margin-bottom: 4px; }} .status {{ font-weight: 700; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
th, td {{ border: 1px solid #d1d5db; padding: 10px; vertical-align: top; text-align: left; }}
th {{ background: #f3f4f6; }} pre {{ margin: 0; white-space: pre-wrap; }}
.pass {{ color: #166534; }} .warning {{ color: #a16207; }} .fail {{ color: #b91c1c; }}
</style>
<body>
<h1>曲面路径可打印性验证报告</h1>
<p>总体结论：<span class=\"status {escape(payload['overall_status'])}\">{escape(payload['overall_status'])}</span>；{escape(payload['decision'])}</p>
<p>实际 Z 范围：{payload['geometry']['curved_z_bounds_mm'][0]:.6f} 至 {payload['geometry']['curved_z_bounds_mm'][1]:.6f} mm；实际高度：{payload['geometry']['actual_max_height_mm']:.6f} mm。</p>
<table><thead><tr><th>检查项</th><th>状态</th><th>结论</th><th>明细</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""


def write_validation_reports(
    report: SurfaceValidationReport, json_path: Path, html_path: Path | None = None
) -> None:
    """Write requested reports; this function never changes either NPZ input."""

    json_path.write_text(json.dumps(report.payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if html_path is not None:
        html_path.write_text(render_html_report(report), encoding="utf-8")
