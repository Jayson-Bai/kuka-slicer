"""A small independent local UI for configuring and exporting surface mapping."""

from __future__ import annotations

import json
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from .contracts import SourceNPZ, SurfaceTarget, load_surface_target, read_source_npz
from .mapper import SurfaceMappingPlan, map_source_job
from .progression import LayerProgression


MAX_SOURCE_NPZ_BYTES = 256 * 1024 * 1024
MAX_SURFACE_CONFIG_BYTES = 2 * 1024 * 1024


def mapping_preview_payload(source: SourceNPZ, target: SurfaceTarget, plan: SurfaceMappingPlan) -> dict[str, object]:
    """Return lightweight mapping diagnostics for the independent UI."""

    result = map_source_job(source, target, plan)
    return {
        "source": _source_summary(source),
        "target": {
            "file_name": target.source_file_name,
            "sha256": target.source_sha256,
            "width_mm": target.width_mm,
            "height_mm": target.height_mm,
        },
        "plan": {
            "start_logical_layer": plan.progression.start_logical_layer,
            "end_logical_layer": plan.progression.end_logical_layer,
            "curve": plan.progression.curve,
            "offset_mode": plan.offset_mode,
            "applied_z_offset_mm": result.applied_z_offset_mm,
            "alpha_by_layer": result.alpha_by_layer,
        },
        "result": {
            "source_z_bounds_mm": result.source_z_bounds_mm,
            "mapped_z_bounds_mm": result.mapped_z_bounds_mm,
            "xy_bounds_mm": result.xy_bounds_mm,
            "extrusion": "preserved_unrecalculated",
            "orientation": "preserved_unrecalculated",
        },
    }


def run_surface_mapper_server(host: str, port: int) -> None:
    """Start the standalone local surface-mapper UI."""

    server = ThreadingHTTPServer((host, port), SurfaceMapperHandler)
    server.source_jobs = {}
    server.surface_targets = {}
    print(f"KUKA surface mapper running at http://{host}:{port}")
    server.serve_forever()


class SurfaceMapperHandler(BaseHTTPRequestHandler):
    """Serve only mapper state; no slicer or Core implementation is imported here."""

    server: ThreadingHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(surface_mapper_html())
            return
        if parsed.path == "/api/preview":
            try:
                params = parse_qs(parsed.query)
                source, target = self._session_values(params)
                plan = _plan_from_params(params, source)
                self._send_json({"ok": True, **mapping_preview_payload(source, target, plan)})
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/source-npz":
                source = read_source_npz(
                    self._read_body(MAX_SOURCE_NPZ_BYTES),
                    source_name=unquote(self.headers.get("X-Source-File-Name", "flat.npz")),
                )
                source_id = secrets.token_urlsafe(16)
                self.server.source_jobs[source_id] = source
                self._send_json({"ok": True, "source_id": source_id, "source": _source_summary(source)})
                return
            if parsed.path == "/api/surface-config":
                target = load_surface_target(self._read_body(MAX_SURFACE_CONFIG_BYTES))
                target_id = secrets.token_urlsafe(16)
                self.server.surface_targets[target_id] = target
                self._send_json(
                    {
                        "ok": True,
                        "target_id": target_id,
                        "target": {
                            "file_name": target.source_file_name,
                            "sha256": target.source_sha256,
                            "width_mm": target.width_mm,
                            "height_mm": target.height_mm,
                        },
                    }
                )
                return
            if parsed.path == "/api/map":
                params = parse_qs(parsed.query)
                source, target = self._session_values(params)
                plan = _plan_from_params(params, source)
                content = map_source_job(source, target, plan).source.to_bytes()
                self._send_bytes(content, "application/octet-stream", "curved.npz")
                return
            self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _session_values(self, params: dict[str, list[str]]) -> tuple[SourceNPZ, SurfaceTarget]:
        source_id = params.get("source_id", [""])[0]
        target_id = params.get("target_id", [""])[0]
        source = self.server.source_jobs.get(source_id)
        target = self.server.surface_targets.get(target_id)
        if source is None:
            raise ValueError("please import a planar source NPZ first")
        if target is None:
            raise ValueError("please import a target surface JSON first")
        return source, target

    def _read_body(self, maximum: int) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length <= 0:
            raise ValueError("uploaded file is empty")
        if length > maximum:
            raise ValueError(f"uploaded file exceeds the {maximum // (1024 * 1024)} MB limit")
        return self.rfile.read(length)

    def _send_html(self, content: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def _send_json(self, payload: dict[str, object], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_bytes(self, content: bytes, content_type: str, filename: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


def _source_summary(source: SourceNPZ) -> dict[str, object]:
    material_keys = [key for key in source.path_keys if key.endswith(("_R", "_F", "_T"))]
    path_count = sum(int(source.arrays[key].shape[0]) for key in material_keys)
    point_count = sum(int(np.isfinite(source.arrays[key][..., 0]).sum()) for key in material_keys)
    return {
        "file_name": source.source_name,
        "layer_indices": list(source.layer_indices),
        "layer_count": len(source.layer_indices),
        "path_count": path_count,
        "point_count": point_count,
        "xy_bounds_mm": source.xy_bounds_mm,
        "z_bounds_mm": source.z_bounds_mm,
        "has_extrusion": any(key.endswith("_E") for key in source.arrays),
    }


def _plan_from_params(params: dict[str, list[str]], source: SourceNPZ) -> SurfaceMappingPlan:
    layers = source.layer_indices
    start = _integer(params, "start_logical_layer", layers[0])
    end = _integer(params, "end_logical_layer", layers[-1])
    curve = params.get("curve", ["smoothstep"])[0]
    offset_mode = params.get("offset_mode", ["auto"])[0]
    z_offset = _number(params, "z_offset_mm", 0.0)
    return SurfaceMappingPlan(
        progression=LayerProgression(start, end, curve=curve),  # type: ignore[arg-type]
        offset_mode=offset_mode,  # type: ignore[arg-type]
        z_offset_mm=z_offset,
    )


def _integer(params: dict[str, list[str]], name: str, default: int) -> int:
    try:
        return int(params.get(name, [str(default)])[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _number(params: dict[str, list[str]], name: str, default: float) -> float:
    try:
        value = float(params.get(name, [str(default)])[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def surface_mapper_html() -> str:
    """Return a small accessible browser shell around the mapper's public API."""

    return r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>曲面映射器</title>
  <style>
    :root { color: #162235; background: #f3f7fb; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; }
    main { max-width: 1180px; margin: auto; padding: 26px; }
    header { margin: 0 0 18px; } h1 { margin: 0; font-size: 24px; } header p { color: #58677d; margin: 7px 0 0; line-height: 1.55; }
    .workspace { display: grid; grid-template-columns: minmax(310px, .9fr) minmax(360px, 1.1fr); gap: 18px; }
    .panel { background: #fff; border: 1px solid #dbe5f0; border-radius: 12px; box-shadow: 0 8px 24px rgba(35, 60, 90, .06); }
    .controls { padding: 18px; } h2 { font-size: 16px; margin: 0 0 12px; } h3 { font-size: 14px; margin: 18px 0 8px; }
    .field { display: grid; gap: 5px; margin: 10px 0; } label { font-size: 13px; color: #32445d; }
    input, select, button { font: inherit; } input, select { min-height: 36px; border: 1px solid #bdcadb; border-radius: 7px; padding: 6px 8px; background: #fff; }
    input[type="file"] { padding: 6px 0; border: 0; } input[type="range"] { padding: 0; }
    button { width: 100%; border: 0; border-radius: 7px; padding: 9px 11px; background: #0b6fc5; color: #fff; cursor: pointer; }
    button:hover { background: #075eaf; } button:disabled { background: #9ba9ba; cursor: not-allowed; }
    .secondary { margin-top: 8px; background: #eaf2fb; color: #075eaf; } .secondary:hover { background: #dcebf9; }
    .divider { height: 1px; background: #e7edf5; margin: 18px 0; } .meta { color: #596a81; min-height: 18px; margin: 7px 0 0; font-size: 12px; line-height: 1.55; word-break: break-word; }
    .results { padding: 18px; } .cards { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .card { border: 1px solid #dce6f0; border-radius: 8px; padding: 11px; } .card span { display: block; color: #66778e; font-size: 12px; } .card strong { display: block; margin-top: 4px; color: #1e3858; font-size: 15px; }
    .layerReadout { margin: 10px 0 0; padding: 11px; background: #f4f8fc; border-radius: 8px; color: #304661; line-height: 1.6; font-size: 13px; }
    .bar { height: 8px; border-radius: 8px; overflow: hidden; background: #dce6f1; margin-top: 8px; } .bar > i { display: block; height: 100%; width: 0; background: #0b8f6f; transition: width .15s ease; }
    .notice { border-left: 3px solid #0b8f6f; padding: 9px 11px; background: #f0faf6; color: #215b4a; font-size: 13px; line-height: 1.55; }
    .status { min-height: 22px; margin: 15px 0 0; color: #56677d; font-size: 13px; } .status.error { color: #b42318; }
    @media (max-width: 820px) { main { padding: 15px; } .workspace { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>曲面映射器</h1>
      <p>读取平面 NPZ 与目标曲面 JSON；仅按逻辑层逐点修改 Z，不改变 XY、路径顺序、材料分组或层号。</p>
    </header>
    <section class="workspace">
      <form class="panel controls" id="mapperForm">
        <h2>输入文件</h2>
        <div class="field"><label for="sourceFile">平面切片路径（flat.npz）</label><input id="sourceFile" type="file" accept=".npz,application/octet-stream"></div>
        <button type="button" id="importSource">导入平面路径</button>
        <p class="meta" id="sourceMeta">尚未导入 NPZ。</p>
        <div class="field"><label for="targetFile">目标曲面（graded_surface_v1.json）</label><input id="targetFile" type="file" accept=".json,application/json"></div>
        <button type="button" id="importTarget">导入目标曲面</button>
        <p class="meta" id="targetMeta">尚未导入曲面 JSON。</p>
        <div class="divider"></div>
        <fieldset id="planFields" disabled>
          <h2>映射策略</h2>
          <div class="field"><label for="startLayer">起始逻辑层</label><input id="startLayer" type="number" step="1"></div>
          <div class="field"><label for="endLayer">完成逻辑层</label><input id="endLayer" type="number" step="1"></div>
          <div class="field"><label for="curve">渐变曲线</label><select id="curve"><option value="smoothstep">平滑渐变（推荐）</option><option value="linear">线性渐变</option></select></div>
          <div class="field"><label for="offsetMode">Z 安全抬升</label><select id="offsetMode"><option value="auto">自动：不低于平面路径最低 Z</option><option value="manual">手动指定</option></select></div>
          <div class="field"><label for="zOffset">手动抬升量（mm）</label><input id="zOffset" type="number" step="0.01" value="0" disabled></div>
          <div class="field"><label for="layerFocus">检查逻辑层</label><input id="layerFocus" type="range" step="1"></div>
        </fieldset>
        <button type="button" id="exportMapped" disabled>映射并导出 curved.npz</button>
        <p class="status" id="status" role="status" aria-live="polite">请先导入两个输入文件。</p>
      </form>
      <section class="panel results" aria-labelledby="previewTitle">
        <h2 id="previewTitle">映射预览</h2>
        <div class="notice">该预览计算完整的 Z 映射，但不会写入任何文件；点击导出后才会下载新 NPZ。</div>
        <div class="cards" style="margin-top: 12px">
          <div class="card"><span>平面路径 Z 范围</span><strong id="flatZ">—</strong></div>
          <div class="card"><span>映射后 Z 范围</span><strong id="mappedZ">—</strong></div>
          <div class="card"><span>自动／实际 Z 抬升</span><strong id="zOffsetReadout">—</strong></div>
          <div class="card"><span>路径点数量</span><strong id="pointCount">—</strong></div>
        </div>
        <h3>当前层的曲面完成度</h3>
        <div class="layerReadout"><span id="layerText">导入路径后可检查每一逻辑层。</span><div class="bar" aria-hidden="true"><i id="alphaBar"></i></div></div>
        <h3>保持不变的内容</h3>
        <p class="meta">X/Y、R/F/T 分组、逻辑层号、路径与点的顺序、已有 E 数值均被保留。本版本不重算 E，也不生成 ABC 姿态。</p>
      </section>
    </section>
  </main>
  <script>
    const sourceFile = document.getElementById('sourceFile');
    const targetFile = document.getElementById('targetFile');
    const sourceMeta = document.getElementById('sourceMeta');
    const targetMeta = document.getElementById('targetMeta');
    const statusEl = document.getElementById('status');
    const planFields = document.getElementById('planFields');
    const exportButton = document.getElementById('exportMapped');
    const sourceId = { value: null }; const targetId = { value: null }; let preview = null;
    const ids = { start: document.getElementById('startLayer'), end: document.getElementById('endLayer'), curve: document.getElementById('curve'), offsetMode: document.getElementById('offsetMode'), zOffset: document.getElementById('zOffset'), layer: document.getElementById('layerFocus') };
    function setStatus(message, error = false) { statusEl.className = error ? 'status error' : 'status'; statusEl.textContent = message; }
    function fmtRange(values) { return `${values[0].toFixed(3)} ～ ${values[1].toFixed(3)} mm`; }
    function params() { const query = new URLSearchParams({ source_id: sourceId.value || '', target_id: targetId.value || '', start_logical_layer: ids.start.value, end_logical_layer: ids.end.value, curve: ids.curve.value, offset_mode: ids.offsetMode.value, z_offset_mm: ids.zOffset.value }); return query; }
    function ready() { const enabled = Boolean(sourceId.value && targetId.value); planFields.disabled = !enabled; exportButton.disabled = !enabled; return enabled; }
    function updateLayerReadout() { if (!preview) return; const layer = Number(ids.layer.value); const alpha = preview.plan.alpha_by_layer[String(layer)]; document.getElementById('layerText').textContent = `逻辑层 ${layer}：sₖ = ${(alpha * 100).toFixed(1)}%，仅该比例的目标曲面 H(X,Y) 会加到本层路径 Z。`; document.getElementById('alphaBar').style.width = `${alpha * 100}%`; }
    async function refreshPreview() { if (!ready()) return; setStatus('正在计算映射预览…'); try { const response = await fetch(`/api/preview?${params().toString()}`); const result = await response.json(); if (!response.ok || !result.ok) throw new Error(result.error || '无法预览映射'); preview = result; document.getElementById('flatZ').textContent = fmtRange(result.result.source_z_bounds_mm); document.getElementById('mappedZ').textContent = fmtRange(result.result.mapped_z_bounds_mm); document.getElementById('zOffsetReadout').textContent = `${result.plan.applied_z_offset_mm.toFixed(3)} mm`; document.getElementById('pointCount').textContent = result.source.point_count.toLocaleString(); updateLayerReadout(); setStatus('预览已更新：逻辑层号和路径顺序将保持不变。'); } catch (error) { setStatus(error.message, true); } }
    async function upload(file, endpoint, name) { const response = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream', 'X-Source-File-Name': encodeURIComponent(file.name) }, body: file }); const result = await response.json(); if (!response.ok || !result.ok) throw new Error(result.error || '导入失败'); return result; }
    document.getElementById('importSource').addEventListener('click', async () => { const file = sourceFile.files[0]; if (!file) return setStatus('请选择 flat.npz。', true); setStatus('正在读取平面路径…'); try { const result = await upload(file, '/api/source-npz'); sourceId.value = result.source_id; const layers = result.source.layer_indices; ids.start.value = layers[0]; ids.end.value = layers[layers.length - 1]; ids.layer.min = layers[0]; ids.layer.max = layers[layers.length - 1]; ids.layer.value = layers[0]; sourceMeta.textContent = `${result.source.file_name}：${result.source.layer_count} 个逻辑层，${result.source.path_count.toLocaleString()} 条路径，${result.source.point_count.toLocaleString()} 个点。`; ready(); await refreshPreview(); } catch (error) { setStatus(error.message, true); } });
    document.getElementById('importTarget').addEventListener('click', async () => { const file = targetFile.files[0]; if (!file) return setStatus('请选择 graded_surface_v1.json。', true); setStatus('正在读取目标曲面…'); try { const result = await upload(file, '/api/surface-config'); targetId.value = result.target_id; targetMeta.textContent = `${result.target.file_name}：XY ${result.target.width_mm.toFixed(3)} × ${result.target.height_mm.toFixed(3)} mm。`; ready(); await refreshPreview(); } catch (error) { setStatus(error.message, true); } });
    ids.offsetMode.addEventListener('change', () => { ids.zOffset.disabled = ids.offsetMode.value !== 'manual'; refreshPreview(); });
    [ids.start, ids.end, ids.curve, ids.zOffset].forEach((input) => input.addEventListener('input', refreshPreview)); ids.layer.addEventListener('input', updateLayerReadout);
    exportButton.addEventListener('click', async () => { if (!ready()) return; setStatus('正在生成 curved.npz…'); try { const response = await fetch(`/api/map?${params().toString()}`, { method: 'POST' }); if (!response.ok) { const result = await response.json(); throw new Error(result.error || '导出失败'); } const blob = await response.blob(); const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = 'curved.npz'; link.click(); URL.revokeObjectURL(link.href); setStatus('已导出 curved.npz，可作为 Core 前的外部源路径输入。'); } catch (error) { setStatus(error.message, true); } });
  </script>
</body>
</html>'''
