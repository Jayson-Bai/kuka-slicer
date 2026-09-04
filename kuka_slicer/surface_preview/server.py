"""A small local web server for interactively inspecting surface equations."""

from __future__ import annotations

import json
import math
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from .model import DoubleSineSurface
from .stl_domain import STLProjectionDomain, stl_projection_domain_from_bytes


DEFAULT_PREVIEW_WIDTH_MM = 120.0
DEFAULT_PREVIEW_HEIGHT_MM = 100.0
DEFAULT_PREVIEW_SAMPLES = 48
MAX_PREVIEW_SAMPLES = 120
MAX_STL_BYTES = 64 * 1024 * 1024


def _query_float(
    params: dict[str, list[str]],
    name: str,
    default: float,
    *,
    positive: bool = False,
) -> float:
    raw = params.get(name, [str(default)])[0]
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _query_samples(params: dict[str, list[str]]) -> int:
    raw = params.get("samples", [str(DEFAULT_PREVIEW_SAMPLES)])[0]
    try:
        samples = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("samples must be an integer") from exc
    if not 8 <= samples <= MAX_PREVIEW_SAMPLES:
        raise ValueError(f"samples must be in the range [8, {MAX_PREVIEW_SAMPLES}]")
    return samples


def surface_payload(
    params: dict[str, list[str]], domain: STLProjectionDomain | None = None, *, include_projection_geometry: bool = True
) -> dict[str, object]:
    """Create browser-safe sampled geometry from URL query parameters."""

    surface = DoubleSineSurface(
        amplitude_mm=_query_float(params, "amplitude_mm", 0.8),
        wavelength_x_mm=_query_float(params, "wavelength_x_mm", 40.0, positive=True),
        wavelength_y_mm=_query_float(params, "wavelength_y_mm", 50.0, positive=True),
        phase_x_rad=_query_float(params, "phase_x_rad", 0.0),
        phase_y_rad=_query_float(params, "phase_y_rad", 0.0),
        z_reference_mm=_query_float(params, "z_reference_mm", 0.0),
    )
    width_mm = domain.width_mm if domain else _query_float(
        params, "width_mm", DEFAULT_PREVIEW_WIDTH_MM, positive=True
    )
    height_mm = domain.height_mm if domain else _query_float(
        params, "height_mm", DEFAULT_PREVIEW_HEIGHT_MM, positive=True
    )
    samples = _query_samples(params)
    grid = surface.sample_grid(
        width_mm=width_mm,
        height_mm=height_mm,
        samples=samples,
        x_min_mm=0.0 if domain else None,
        y_min_mm=0.0 if domain else None,
    )
    grid_payload: dict[str, object] = {"x": grid.x.tolist(), "y": grid.y.tolist(), "z": grid.z.tolist()}
    if domain:
        center_x = (grid.x[:-1, :-1] + grid.x[:-1, 1:] + grid.x[1:, 1:] + grid.x[1:, :-1]) / 4.0
        center_y = (grid.y[:-1, :-1] + grid.y[:-1, 1:] + grid.y[1:, 1:] + grid.y[1:, :-1]) / 4.0
        grid_payload["material_mask"] = domain.material_mask(center_x, center_y).tolist()
    return {
        "surface": {
            "type": "double_sine_product",
            "amplitude_mm": surface.amplitude_mm,
            "wavelength_x_mm": surface.wavelength_x_mm,
            "wavelength_y_mm": surface.wavelength_y_mm,
            "phase_x_rad": surface.phase_x_rad,
            "phase_y_rad": surface.phase_y_rad,
            "z_reference_mm": surface.z_reference_mm,
        },
        "domain": {
            "width_mm": width_mm,
            "height_mm": height_mm,
            "samples": samples,
            "mode": "stl_projection" if domain else "rectangle",
            "projection": domain.preview_payload(include_polygons=include_projection_geometry) if domain else None,
        },
        "statistics": grid.summary(),
        "grid": grid_payload,
    }


def graded_surface_config_payload(
    params: dict[str, list[str]], domain: STLProjectionDomain
) -> dict[str, object]:
    """Build the portable geometry-only sidecar consumed by the mapper."""

    surface = surface_payload(params, domain)["surface"]
    projection = domain.preview_payload()
    return {
        "format": "graded_surface_v1",
        "units": "mm",
        "coordinate_system": {
            "plane": "XY",
            "build_axis": "Z",
            "origin": "stl_xy_min",
            "source_build_axis": domain.build_axis,
        },
        "domain": {
            "mode": "stl_projection",
            "boundary": "stl_outer_contour",
            "edge_policy": "closed_perimeter",
            "source": {
                "file_name": projection["file_name"],
                "sha256": projection["sha256"],
                "triangle_count": projection["triangle_count"],
                "xy_bounds_mm": projection["source_xy_bounds_mm"],
                "projection_layer_height_mm": projection["projection_layer_height_mm"],
            },
        },
        "surface": surface,
    }


def run_surface_preview_server(host: str, port: int) -> None:
    """Start the independent local surface-preview server."""

    server = ThreadingHTTPServer((host, port), SurfacePreviewHandler)
    server.preview_domains = {}
    print(f"KUKA surface preview running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        server.server_close()


class SurfacePreviewHandler(BaseHTTPRequestHandler):
    """Serve the future-embeddable web shell and sampled surface data."""

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(surface_preview_html())
            return
        if parsed.path == "/api/surface":
            try:
                params = parse_qs(parsed.query)
                payload = surface_payload(
                    params,
                    self._domain_from_params(params),
                    include_projection_geometry=params.get("compact", [""])[0] != "1",
                )
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"ok": True, **payload})
            return
        if parsed.path == "/api/export-surface-config":
            try:
                params = parse_qs(parsed.query)
                config = graded_surface_config_payload(
                    params, self._domain_from_params(params, required=True)
                )
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(
                config,
                attachment_name="graded_surface_v1.json",
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/stl-domain":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            params = parse_qs(parsed.query)
            build_axis = params.get("build_axis", ["z"])[0]
            if build_axis not in ("x", "y", "z"):
                raise ValueError("build_axis must be x, y, or z")
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise ValueError("STL upload requires Content-Length")
            content_length = int(raw_length)
            if not 0 < content_length <= MAX_STL_BYTES:
                raise ValueError(f"STL must be between 1 byte and {MAX_STL_BYTES // 1024 // 1024} MB")
            data = self.rfile.read(content_length)
            file_name = unquote(self.headers.get("X-STL-File-Name", "model.stl"))
            domain = stl_projection_domain_from_bytes(
                data,
                file_name=file_name,
                build_axis=build_axis,
            )
        except (TypeError, ValueError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        domain_id = secrets.token_urlsafe(18)
        self.server.preview_domains[domain_id] = domain
        self._send_json({"ok": True, "domain_id": domain_id, "projection": domain.preview_payload()})

    def _domain_from_params(
        self, params: dict[str, list[str]], *, required: bool = False
    ) -> STLProjectionDomain | None:
        domain_id = params.get("domain_id", [""])[0]
        if not domain_id:
            if required:
                raise ValueError("import an STL before exporting a surface configuration")
            return None
        domain = self.server.preview_domains.get(domain_id)
        if domain is None:
            raise ValueError("the imported STL is no longer available; import it again")
        return domain

    def _send_html(self, content: str) -> None:
        encoded = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(
        self,
        payload: dict[str, object],
        status: HTTPStatus = HTTPStatus.OK,
        *,
        attachment_name: str | None = None,
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        if attachment_name:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment_name}"')
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


def surface_preview_html() -> str:
    """Return the self-contained browser shell without third-party dependencies."""

    return r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KUKA 曲面预览器</title>
  <style>
    :root { color-scheme: light; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; color: #152033; background: #f4f7fb; }
    * { box-sizing: border-box; }
    body { margin: 0; min-width: 320px; }
    main { max-width: 1260px; margin: 0 auto; padding: 24px; }
    header { margin-bottom: 18px; }
    h1 { margin: 0; font-size: clamp(22px, 3vw, 32px); letter-spacing: -.02em; }
    header p { color: #526074; margin: 8px 0 0; line-height: 1.55; }
    .workspace { display: grid; grid-template-columns: minmax(260px, 340px) minmax(0, 1fr); gap: 18px; align-items: start; }
    .panel { background: #fff; border: 1px solid #dbe3ef; border-radius: 14px; box-shadow: 0 10px 30px rgba(32, 52, 82, .07); }
    .controls { padding: 18px; }
    .controls h2, .preview h2 { margin: 0 0 14px; font-size: 16px; }
    .field { display: grid; grid-template-columns: 1fr 112px; align-items: center; gap: 10px; margin: 10px 0; }
    label { font-size: 13px; color: #38475d; }
    input { width: 100%; border: 1px solid #bdcadb; border-radius: 7px; padding: 8px; color: #142238; font: inherit; }
    input:focus { outline: 3px solid rgba(36, 122, 207, .18); border-color: #247acf; }
    .divider { height: 1px; background: #e6ecf4; margin: 17px 0; }
    button { width: 100%; border: 0; border-radius: 8px; padding: 10px 12px; background: #126fd1; color: white; font: 600 14px inherit; cursor: pointer; }
    button:hover { background: #075eaf; }
    button:disabled { background: #9ba9ba; cursor: not-allowed; }
    button.secondary { background: #eaf2fb; color: #0b5da9; margin-top: 8px; }
    button.secondary:hover { background: #dcebf9; }
    select { width: 100%; border: 1px solid #bdcadb; border-radius: 7px; padding: 8px; color: #142238; font: inherit; background: #fff; }
    .fileInput { margin: 8px 0 0; font-size: 12px; }
    .modelMeta { min-height: 18px; margin: 9px 0 0; color: #526074; font-size: 12px; line-height: 1.5; word-break: break-word; }
    .hint { margin: 12px 0 0; color: #66758b; font-size: 12px; line-height: 1.55; }
    .designSummary { margin: 10px 0 0; padding: 9px 10px; border: 1px solid #d9e5f4; border-radius: 8px; background: #f5f9fd; color: #40516a; font-size: 12px; line-height: 1.55; }
    .designSummary.error { border-color: #f0c5c2; background: #fff7f6; color: #a52a21; }
    details.advanced { margin-top: 12px; color: #40516a; font-size: 13px; }
    details.advanced summary { cursor: pointer; color: #2e405a; font-weight: 600; }
    .advancedBody { padding-top: 4px; }
    .preview { overflow: hidden; }
    .previewHead { padding: 18px 18px 0; display: flex; justify-content: space-between; gap: 12px; align-items: start; }
    .stats { display: flex; flex-wrap: wrap; gap: 7px; justify-content: end; }
    .stat { border: 1px solid #dae4f1; border-radius: 999px; padding: 4px 8px; color: #40516a; font-size: 12px; white-space: nowrap; }
    canvas { display: block; width: 100%; height: min(65vh, 620px); min-height: 400px; background: linear-gradient(180deg, #fbfdff 0%, #eef4fa 100%); touch-action: none; cursor: grab; }
    canvas.isDragging { cursor: grabbing; }
    .navigationHint { margin: 0; padding: 10px 18px 14px; color: #66758b; font-size: 12px; line-height: 1.5; border-top: 1px solid #edf1f6; }
    .status { min-height: 18px; padding: 0 18px 16px; color: #68778c; font-size: 12px; }
    .status.error { color: #b42318; }
    @media (max-width: 820px) { main { padding: 14px; } .workspace { grid-template-columns: 1fr; } canvas { min-height: 330px; } }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>渐变双正弦曲面预览器</h1>
      <p>首版只负责曲面方程和几何诊断；不修改切片路径，也不写入 NPZ。</p>
    </header>
    <section class="workspace">
      <form class="panel controls" id="surfaceForm">
        <h2>输入模型</h2>
        <label for="stlFile">导入用于最终切片的平面蜂窝 STL</label>
        <input class="fileInput" id="stlFile" type="file" accept=".stl,model/stl,application/sla">
        <div class="field"><label for="build_axis">源构建轴</label><select id="build_axis"><option value="z" selected>Z</option><option value="y">Y</option><option value="x">X</option></select></div>
        <button type="button" id="importStl">导入并提取 XY 投影</button>
        <p class="modelMeta" id="modelMeta">尚未导入 STL：当前仅显示矩形参考域。</p>
        <div class="divider"></div>
        <h2>曲面参数</h2>
        <div class="field"><label for="amplitude_mm">幅值 A（mm）</label><input id="amplitude_mm" type="number" step="0.01" value="0.8"></div>
        <div class="field"><label for="wavelength_x_mm">X 波长 λx（mm）</label><input id="wavelength_x_mm" type="number" min="0.001" step="0.1" value="40"></div>
        <div class="field"><label for="wavelength_y_mm">Y 波长 λy（mm）</label><input id="wavelength_y_mm" type="number" min="0.001" step="0.1" value="50"></div>
        <div class="field"><label for="phase_x_rad">X 相位 φx（rad）</label><input id="phase_x_rad" type="number" step="0.01" value="0"></div>
        <div class="field"><label for="phase_y_rad">Y 相位 φy（rad）</label><input id="phase_y_rad" type="number" step="0.01" value="0"></div>
        <div class="field"><label for="z_reference_mm">Z 基准（mm）</label><input id="z_reference_mm" type="number" step="0.01" value="0"></div>
        <div class="divider"></div>
        <h2>固定六边形格栅</h2>
        <div class="field"><label for="wall_width_mm">设计墙宽（mm）</label><input id="wall_width_mm" type="number" min="0.001" step="0.01" value="2"></div>
        <div class="field"><label for="base_cell_size_mm">目标六边形边长（mm）</label><input id="base_cell_size_mm" type="number" min="0.001" step="0.01" value="5"></div>
        <p class="hint">设计墙宽用于格栅几何，不等同于实际挤出道宽；目标边长沿承载曲面测量。</p>
        <div class="designSummary" id="latticeDesignSummary" aria-live="polite"></div>
        <div class="divider"></div>
        <h2>对称层间渐变</h2>
        <div class="field"><label for="surface_start_layer">曲面起始层</label><input id="surface_start_layer" type="number" min="0" step="1" value="3"></div>
        <p class="hint">沿用旧版语义：起始层本身保持平面，下一层才开始增大曲率。</p>
        <div class="designSummary" id="layerProgressionSummary" aria-live="polite"></div>
        <details class="advanced">
          <summary>高级参数（共形计算）</summary>
          <div class="advancedBody">
            <div class="field"><label for="samples_x">曲面采样 X</label><input id="samples_x" type="number" min="2" step="1" value="48"></div>
            <div class="field"><label for="samples_y">曲面采样 Y</label><input id="samples_y" type="number" min="2" step="1" value="48"></div>
            <p class="hint">这两个值将在导出共形配置时用于 LSCM 和格栅计算；当前旧版曲面预览仍使用下方独立的显示网格密度。</p>
          </div>
        </details>
        <div class="divider"></div>
        <h2>预览域</h2>
        <div class="field"><label for="width_mm">X 宽度（mm）</label><input id="width_mm" type="number" min="0.001" step="1" value="120"></div>
        <div class="field"><label for="height_mm">Y 高度（mm）</label><input id="height_mm" type="number" min="0.001" step="1" value="100"></div>
        <div class="field"><label for="samples">显示网格密度</label><input id="samples" type="number" min="8" max="120" step="1" value="48"></div>
        <button type="button" id="exportConfig" disabled>导出旧版曲面 JSON</button>
        <button type="button" class="secondary" id="reset">恢复示例参数</button>
        <p class="hint">方程：H(x,y)=A·sin(2πx/λx+φx)·sin(2πy/λy+φy)+Zref。当前按钮仍只导出旧版 <code>graded_surface_v1</code>，不会写入共形格栅参数。</p>
      </form>
      <section class="panel preview">
        <div class="previewHead"><h2>三维曲面</h2><div class="stats" id="stats"></div></div>
        <canvas id="canvas" aria-label="双正弦曲面预览"></canvas>
        <p class="navigationHint">左键拖拽旋转；中键拖拽平移；右键上下拖拽缩放；滚轮缩放；双击恢复视角。</p>
        <div class="status" id="status">正在生成曲面…</div>
      </section>
    </section>
  </main>
  <script>
    const surfaceIds = ['amplitude_mm', 'wavelength_x_mm', 'wavelength_y_mm', 'phase_x_rad', 'phase_y_rad', 'z_reference_mm', 'width_mm', 'height_mm', 'samples'];
    const conformalDesignIds = ['wall_width_mm', 'base_cell_size_mm', 'surface_start_layer', 'samples_x', 'samples_y'];
    const canvas = document.getElementById('canvas');
    const statusEl = document.getElementById('status');
    const statsEl = document.getElementById('stats');
    const modelMetaEl = document.getElementById('modelMeta');
    const exportConfigButton = document.getElementById('exportConfig');
    let payload = null;
    let queued = 0;
    let domainId = null;
    let domainProjection = null;
    const initialView = { yaw: -42 * Math.PI / 180, pitch: 54 * Math.PI / 180, zoom: 1, panX: 0, panY: 0 };
    const view = { ...initialView };
    let drag = null;

    function positiveNumber(id) {
      const value = Number(document.getElementById(id).value);
      return Number.isFinite(value) && value > 0 ? value : null;
    }

    function nonNegativeInteger(id) {
      const value = Number(document.getElementById(id).value);
      return Number.isInteger(value) && value >= 0 ? value : null;
    }

    function updateConformalDesignSummary() {
      const wallWidth = positiveNumber('wall_width_mm');
      const cellSize = positiveNumber('base_cell_size_mm');
      const latticeSummary = document.getElementById('latticeDesignSummary');
      if (wallWidth === null || cellSize === null) {
        latticeSummary.className = 'designSummary error';
        latticeSummary.textContent = '设计墙宽和目标六边形边长都必须是正数。';
      } else {
        const nominalFill = (2 * wallWidth) / (Math.sqrt(3) * cellSize);
        if (nominalFill >= 1) {
          latticeSummary.className = 'designSummary error';
          latticeSummary.textContent = `名义填充率为 ${(nominalFill * 100).toFixed(1)}%，必须小于 100%。请减小墙宽或增大单元边长。`;
        } else {
          latticeSummary.className = 'designSummary';
          latticeSummary.textContent = `名义填充率：${(nominalFill * 100).toFixed(1)}%。实际填充率将在 Gate 6 按生成几何测量。`;
        }
      }

      const startLayer = nonNegativeInteger('surface_start_layer');
      const samplesX = nonNegativeInteger('samples_x');
      const samplesY = nonNegativeInteger('samples_y');
      const progressionSummary = document.getElementById('layerProgressionSummary');
      if (startLayer === null) {
        progressionSummary.className = 'designSummary error';
        progressionSummary.textContent = '曲面起始层必须是非负整数。';
      } else if (samplesX === null || samplesX < 2 || samplesY === null || samplesY < 2) {
        progressionSummary.className = 'designSummary error';
        progressionSummary.textContent = '曲面采样 X 和 Y 都必须是不小于 2 的整数。';
      } else {
        progressionSummary.className = 'designSummary';
        progressionSummary.textContent = `曲面起始层：${startLayer}；共形采样：${samplesX} × ${samplesY}。生成共形配置时将结合主切片器的逻辑层数校验镜像回落层。`;
      }
    }

    function parameters() {
      const query = new URLSearchParams();
      surfaceIds.forEach((id) => query.set(id, document.getElementById(id).value));
      if (domainId) {
        query.set('domain_id', domainId);
        query.set('compact', '1');
      }
      return query;
    }

    function colour(fraction) {
      const value = Math.max(0, Math.min(1, fraction));
      return `hsl(204, 68%, ${86 - value * 42}%)`;
    }

    function project(x, y, z, yaw, pitch, scale, cx, cy) {
      const xr = x * Math.cos(yaw) - y * Math.sin(yaw);
      const yr = x * Math.sin(yaw) + y * Math.cos(yaw);
      const yp = yr * Math.cos(pitch) - z * Math.sin(pitch);
      const depth = yr * Math.sin(pitch) + z * Math.cos(pitch);
      return { x: cx + xr * scale, y: cy - yp * scale, depth };
    }

    function heightAt(x, y) {
      const surface = payload.surface;
      return surface.z_reference_mm + surface.amplitude_mm
        * Math.sin((2 * Math.PI * x) / surface.wavelength_x_mm + surface.phase_x_rad)
        * Math.sin((2 * Math.PI * y) / surface.wavelength_y_mm + surface.phase_y_rad);
    }

    function appendProjectedRing(ctx, ring, zMid, yaw, pitch, scale, cx, cy) {
      if (ring.length < 2) return;
      const first = project(ring[0][0], ring[0][1], heightAt(ring[0][0], ring[0][1]) - zMid, yaw, pitch, scale, cx, cy);
      ctx.moveTo(first.x, first.y);
      ring.slice(1).forEach(([x, y]) => {
        const point = project(x, y, heightAt(x, y) - zMid, yaw, pitch, scale, cx, cy);
        ctx.lineTo(point.x, point.y);
      });
      ctx.closePath();
    }

    function clipToProjection(ctx, projection, zMid, yaw, pitch, scale, cx, cy) {
      ctx.beginPath();
      projection.polygons.forEach((polygon) => {
        appendProjectedRing(ctx, polygon.outer, zMid, yaw, pitch, scale, cx, cy);
        polygon.holes.forEach((ring) => appendProjectedRing(ctx, ring, zMid, yaw, pitch, scale, cx, cy));
      });
      ctx.clip('evenodd');
    }

    function drawProjectionBoundaries(ctx, projection, zMid, yaw, pitch, scale, cx, cy) {
      if (!projection) return;
      ctx.strokeStyle = 'rgba(12, 44, 82, .94)';
      ctx.lineWidth = 1.2;
      projection.polygons.forEach((polygon) => {
        const rings = drag ? [polygon.outer] : [polygon.outer, ...polygon.holes];
        rings.forEach((ring) => {
          ctx.beginPath();
          appendProjectedRing(ctx, ring, zMid, yaw, pitch, scale, cx, cy);
          ctx.stroke();
        });
      });
    }

    function render() {
      if (!payload) return;
      const rect = canvas.getBoundingClientRect();
      const pixelRatio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * pixelRatio));
      canvas.height = Math.max(1, Math.round(rect.height * pixelRatio));
      const ctx = canvas.getContext('2d');
      ctx.scale(pixelRatio, pixelRatio);
      const width = rect.width;
      const height = rect.height;
      ctx.clearRect(0, 0, width, height);
      const { x, y, z } = payload.grid;
      const stats = payload.statistics;
      const projection = payload.domain.projection;
      const materialMask = payload.grid.material_mask;
      const { yaw, pitch } = view;
      const span = Math.max(payload.domain.width_mm, payload.domain.height_mm, stats.z_range_mm * 2, 1);
      const scale = Math.min(width, height) * 0.72 * view.zoom / span;
      const cx = width / 2 + view.panX;
      const cy = height / 2 + 8 + view.panY;
      const zMid = (stats.z_min_mm + stats.z_max_mm) / 2;
      const exactProjectionClip = Boolean(projection && !drag);
      const cells = [];
      for (let row = 0; row < z.length - 1; row += 1) {
        for (let col = 0; col < z[row].length - 1; col += 1) {
          if (!exactProjectionClip && materialMask && !materialMask[row][col]) continue;
          const p = [
            project(x[row][col], y[row][col], z[row][col] - zMid, yaw, pitch, scale, cx, cy),
            project(x[row][col + 1], y[row][col + 1], z[row][col + 1] - zMid, yaw, pitch, scale, cx, cy),
            project(x[row + 1][col + 1], y[row + 1][col + 1], z[row + 1][col + 1] - zMid, yaw, pitch, scale, cx, cy),
            project(x[row + 1][col], y[row + 1][col], z[row + 1][col] - zMid, yaw, pitch, scale, cx, cy),
          ];
          const averageZ = (z[row][col] + z[row][col + 1] + z[row + 1][col + 1] + z[row + 1][col]) / 4;
          cells.push({ p, depth: p.reduce((sum, point) => sum + point.depth, 0) / 4, averageZ });
        }
      }
      cells.sort((a, b) => a.depth - b.depth);
      if (exactProjectionClip) {
        ctx.save();
        clipToProjection(ctx, projection, zMid, yaw, pitch, scale, cx, cy);
      }
      cells.forEach((cell) => {
        const fraction = stats.z_range_mm === 0 ? 0.5 : (cell.averageZ - stats.z_min_mm) / stats.z_range_mm;
        ctx.beginPath();
        ctx.moveTo(cell.p[0].x, cell.p[0].y);
        cell.p.slice(1).forEach((point) => ctx.lineTo(point.x, point.y));
        ctx.closePath();
        ctx.fillStyle = colour(fraction);
        ctx.fill();
      });
      if (exactProjectionClip) ctx.restore();
      drawProjectionBoundaries(ctx, projection, zMid, yaw, pitch, scale, cx, cy);
      ctx.fillStyle = 'rgba(21,32,51,.68)';
      ctx.font = '12px Segoe UI, Microsoft YaHei, sans-serif';
      ctx.fillText('X / Y：mm   Z：mm', 14, height - 16);
    }

    function showStats(statistics) {
      const values = [
        `Z：${statistics.z_min_mm.toFixed(3)} ～ ${statistics.z_max_mm.toFixed(3)} mm`,
        `起伏：${statistics.z_range_mm.toFixed(3)} mm`,
        `最大坡度：${statistics.max_slope.toFixed(3)}`,
      ];
      statsEl.replaceChildren(...values.map((value) => {
        const item = document.createElement('span');
        item.className = 'stat';
        item.textContent = value;
        return item;
      }));
    }

    async function refresh() {
      const sequence = ++queued;
      statusEl.className = 'status';
      statusEl.textContent = '正在更新曲面…';
      try {
        const response = await fetch(`/api/surface?${parameters().toString()}`);
        const result = await response.json();
        if (!response.ok || !result.ok) throw new Error(result.error || '无法生成曲面');
        if (sequence !== queued) return;
        if (result.domain.projection && result.domain.projection.polygons) {
          domainProjection = result.domain.projection;
        }
        if (domainProjection) result.domain.projection = domainProjection;
        payload = result;
        showStats(result.statistics);
        render();
        statusEl.textContent = result.domain.mode === 'stl_projection'
          ? '已更新：曲面已裁剪到导入 STL 的 XY 材料投影。'
          : '已更新。导入 STL 后将按真实 XY 投影裁剪曲面。';
      } catch (error) {
        if (sequence !== queued) return;
        statusEl.className = 'status error';
        statusEl.textContent = error.message;
      }
    }

    let timer = null;
    function scheduleRefresh() { clearTimeout(timer); timer = setTimeout(refresh, 120); }
    surfaceIds.forEach((id) => document.getElementById(id).addEventListener('input', scheduleRefresh));
    conformalDesignIds.forEach((id) => document.getElementById(id).addEventListener('input', updateConformalDesignSummary));
    document.getElementById('importStl').addEventListener('click', async () => {
      const input = document.getElementById('stlFile');
      const file = input.files[0];
      if (!file) {
        statusEl.className = 'status error';
        statusEl.textContent = '请先选择一个 STL 文件。';
        return;
      }
      const button = document.getElementById('importStl');
      button.disabled = true;
      statusEl.className = 'status';
      statusEl.textContent = '正在提取 STL 的 XY 投影…';
      try {
        const buildAxis = document.getElementById('build_axis').value;
        const response = await fetch(`/api/stl-domain?build_axis=${encodeURIComponent(buildAxis)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/sla', 'X-STL-File-Name': encodeURIComponent(file.name) },
          body: file,
        });
        const result = await response.json();
        if (!response.ok || !result.ok) throw new Error(result.error || '无法读取 STL');
        domainId = result.domain_id;
        const projection = result.projection;
        domainProjection = projection;
        document.getElementById('width_mm').value = projection.width_mm.toFixed(3);
        document.getElementById('height_mm').value = projection.height_mm.toFixed(3);
        document.getElementById('width_mm').readOnly = true;
        document.getElementById('height_mm').readOnly = true;
        modelMetaEl.textContent = `${projection.file_name}：${projection.triangle_count.toLocaleString()} 个三角面，XY ${projection.width_mm.toFixed(3)} × ${projection.height_mm.toFixed(3)} mm，材料投影 ${projection.material_area_mm2.toFixed(3)} mm²。`;
        exportConfigButton.disabled = false;
        refresh();
      } catch (error) {
        statusEl.className = 'status error';
        statusEl.textContent = error.message;
      } finally {
        button.disabled = false;
      }
    });
    exportConfigButton.addEventListener('click', async () => {
      if (!domainId) return;
      try {
        const response = await fetch(`/api/export-surface-config?${parameters().toString()}`);
        if (!response.ok) {
          const result = await response.json();
          throw new Error(result.error || '无法导出曲面配置');
        }
        const blob = await response.blob();
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'graded_surface_v1.json';
        link.click();
        URL.revokeObjectURL(link.href);
        statusEl.className = 'status';
        statusEl.textContent = '已导出旧版曲面 JSON；后续曲面映射器将与同一 STL 一起使用它。';
      } catch (error) {
        statusEl.className = 'status error';
        statusEl.textContent = error.message;
      }
    });
    canvas.addEventListener('contextmenu', (event) => event.preventDefault());
    canvas.addEventListener('pointerdown', (event) => {
      const mode = event.button === 0 ? 'rotate' : event.button === 1 ? 'pan' : 'zoom';
      drag = { pointerId: event.pointerId, mode, x: event.clientX, y: event.clientY, zoom: view.zoom };
      canvas.setPointerCapture(event.pointerId);
      canvas.classList.add('isDragging');
      event.preventDefault();
    });
    canvas.addEventListener('pointermove', (event) => {
      if (!drag || event.pointerId !== drag.pointerId) return;
      const dx = event.clientX - drag.x;
      const dy = event.clientY - drag.y;
      if (drag.mode === 'rotate') {
        view.yaw += dx * 0.012;
        view.pitch = Math.max(0.08, Math.min(Math.PI / 2 - 0.08, view.pitch - dy * 0.012));
      } else if (drag.mode === 'pan') {
        view.panX += dx;
        view.panY += dy;
      } else {
        view.zoom = Math.max(0.2, Math.min(5, drag.zoom * Math.exp(-dy * 0.012)));
      }
      drag.x = event.clientX;
      drag.y = event.clientY;
      if (drag.mode === 'zoom') drag.zoom = view.zoom;
      render();
    });
    function endDrag(event) {
      if (!drag || event.pointerId !== drag.pointerId) return;
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      drag = null;
      canvas.classList.remove('isDragging');
      render();
    }
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);
    canvas.addEventListener('lostpointercapture', endDrag);
    window.addEventListener('pointerup', endDrag);
    canvas.addEventListener('wheel', (event) => {
      view.zoom = Math.max(0.2, Math.min(5, view.zoom * Math.exp(-event.deltaY * 0.0012)));
      render();
      event.preventDefault();
    }, { passive: false });
    canvas.addEventListener('dblclick', () => {
      Object.assign(view, initialView);
      render();
    });
    document.getElementById('reset').addEventListener('click', () => {
      const defaults = { amplitude_mm: 0.8, wavelength_x_mm: 40, wavelength_y_mm: 50, phase_x_rad: 0, phase_y_rad: 0, z_reference_mm: 0, wall_width_mm: 2, base_cell_size_mm: 5, surface_start_layer: 3, samples_x: 48, samples_y: 48, width_mm: 120, height_mm: 100, samples: 48 };
      Object.entries(defaults).forEach(([id, value]) => {
        if (domainId && (id === 'width_mm' || id === 'height_mm')) return;
        document.getElementById(id).value = value;
      });
      updateConformalDesignSummary();
      refresh();
    });
    window.addEventListener('resize', render);
    updateConformalDesignSummary();
    refresh();
  </script>
</body>
</html>'''
