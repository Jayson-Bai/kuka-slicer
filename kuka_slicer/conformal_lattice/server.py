"""Independent, read-only local preview for conformal-lattice payloads."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Mapping


def run_conformal_lattice_preview_server(payload: Mapping[str, object], host: str = "127.0.0.1", port: int = 8766) -> None:
    """Serve an already-built preview payload; no endpoint modifies geometry."""

    if payload.get("read_only") is not True:
        raise ValueError("preview server accepts only a read-only conformal lattice payload")
    server = ThreadingHTTPServer((host, port), _PreviewHandler)
    server.preview_payload = dict(payload)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


class _PreviewHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._send_html(conformal_lattice_preview_html())
        elif self.path == "/api/preview":
            self._send_json({"ok": True, **self.server.preview_payload})
        else:
            self._send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)

    def _send_html(self, content: str) -> None:
        encoded = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: Mapping[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def conformal_lattice_preview_html() -> str:
    """Return the zero-dependency, canvas-backed inspector shell."""

    return r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>共形格栅检查器</title><style>
:root{--ink:#13252d;--paper:#e9edf0;--panel:#f9fbfc;--grid:#cad5d9;--accent:#c55a2d;--teal:#086c71;--muted:#60727a}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Microsoft YaHei UI",system-ui,sans-serif}.shell{max-width:1280px;margin:auto;padding:24px}.mast{display:flex;justify-content:space-between;align-items:end;border-bottom:3px solid var(--ink);padding-bottom:16px}.mast h1{margin:0;font-size:30px;letter-spacing:-.04em}.mast p{margin:0;color:var(--muted);font:12px ui-monospace,monospace}.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:18px 0}.tabs button{border:1px solid var(--grid);background:var(--panel);color:var(--ink);padding:8px 11px;cursor:pointer}.tabs button[aria-selected="true"]{background:var(--ink);color:white;border-color:var(--ink)}.stage{background:var(--panel);border:1px solid var(--grid);min-height:600px;display:grid;grid-template-columns:minmax(0,1fr) 260px}.canvas-wrap{min-height:520px;padding:18px}canvas{width:100%;height:100%;min-height:520px;background:#fff;border:1px solid var(--grid)}aside{border-left:1px solid var(--grid);padding:18px}.metric{border-bottom:1px solid var(--grid);padding:10px 0;font-size:13px}.metric b{display:block;font:11px ui-monospace,monospace;color:var(--muted);margin-bottom:4px}.key{display:flex;gap:8px;margin-top:12px;font-size:12px;color:var(--muted)}.swatch{width:12px;height:12px;display:inline-block;background:var(--accent)}@media(max-width:760px){.stage{grid-template-columns:1fr}aside{border-left:0;border-top:1px solid var(--grid)}}
</style></head><body><main class="shell"><header class="mast"><h1>共形格栅检查器</h1><p>只读 · 不修改结构几何</p></header><nav class="tabs" aria-label="视图切换" id="tabs"></nav><section class="stage"><div class="canvas-wrap"><canvas id="canvas" aria-label="共形格栅视图"></canvas></div><aside id="metrics"></aside></section></main>
<script>
const views=[['surface_3d','三维格栅'],['uv','UV 参数域'],['conformal_distortion','共形畸变'],['fill_ratio','填充率'],['orientation','方向场'],['defects','拓扑缺陷'],['layers','层间结构']];let payload,active='surface_3d';const canvas=document.querySelector('#canvas'),ctx=canvas.getContext('2d');
function fit(){const r=canvas.getBoundingClientRect(),d=devicePixelRatio;canvas.width=r.width*d;canvas.height=r.height*d;ctx.setTransform(d,0,0,d,0,0);return r}
function project(points,three){const xs=points.map(p=>three?p[0]-.55*p[1]:p[0]),ys=points.map(p=>three?-.25*p[0]-.45*p[1]+p[2]:p[1]);const minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys);const r=fit(),s=.86*Math.min(r.width/(maxx-minx||1),r.height/(maxy-miny||1));return p=>[(r.width-(maxx-minx)*s)/2+(three?p[0]-.55*p[1]:p[0]-minx)*s,(r.height-(maxy-miny)*s)/2+(maxy-(three?-.25*p[0]-.45*p[1]+p[2]:p[1]))*s]}
function lattice(view,three){const points=view.lattice_nodes_xyz||view.lattice_nodes_uv,edges=view.lattice_edges||payload.surface_3d.lattice_edges,xy=project(points,three);ctx.clearRect(0,0,canvas.width,canvas.height);ctx.strokeStyle='#086c71';ctx.lineWidth=1.25;for(const [a,b] of edges){const p=xy(points[a]),q=xy(points[b]);ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke()}return {nodes:points.length,edges:edges.length}}
function heat(values,label){const surface=payload.uv,points=surface.surface_uv,faces=surface.faces,xy=project(points,false),lo=Math.min(...values),hi=Math.max(...values);ctx.clearRect(0,0,canvas.width,canvas.height);faces.forEach((f,i)=>{const t=(values[i]-lo)/(hi-lo||1),p=f.map(j=>xy(points[j]));ctx.fillStyle=`hsl(${190-170*t} 66% ${78-35*t}%)`;ctx.beginPath();ctx.moveTo(...p[0]);p.slice(1).forEach(v=>ctx.lineTo(...v));ctx.closePath();ctx.fill()});return {[label]:`${lo.toPrecision(4)} — ${hi.toPrecision(4)}`}}
function draw(){let m={};if(active==='surface_3d')m=lattice(payload.surface_3d,true);else if(active==='uv')m=lattice(payload.uv,false);else if(active==='conformal_distortion')m=heat(payload.conformal_distortion.ratio_per_face,'conformal ratio');else if(active==='fill_ratio'){const a=payload.fill_ratio.actual;m=a?{MAE:a.report.mae,P95:a.report.p95_absolute_error,'evaluated cells':a.report.evaluated_cell_count}:{status:'尚未执行 Gate 6 验证'};lattice(payload.uv,false)}else if(active==='orientation'){m=payload.orientation.report;lattice(payload.surface_3d,true)}else if(active==='defects'){m=payload.defects.report;lattice(payload.uv,false)}else if(active==='layers'){m=payload.layers?payload.layers.report:{status:'未生成层间结构'};payload.layers?lattice({lattice_nodes_xyz:payload.layers.node_positions_xyz[0],lattice_edges:payload.layers.lattice_edges},true):lattice(payload.surface_3d,true)}document.querySelector('#metrics').innerHTML=Object.entries(m).slice(0,9).map(([k,v])=>`<div class="metric"><b>${k}</b>${typeof v==='object'?JSON.stringify(v):v}</div>`).join('')}
function tabs(){const n=document.querySelector('#tabs');views.forEach(([id,label])=>{const b=document.createElement('button');b.textContent=label;b.setAttribute('aria-selected',id===active);b.onclick=()=>{active=id;[...n.children].forEach(x=>x.setAttribute('aria-selected',x===b));draw()};n.append(b)})}fetch('/api/preview').then(r=>r.json()).then(r=>{payload=r;tabs();draw()}).catch(e=>document.querySelector('#metrics').textContent=`无法载入预览：${e}`);addEventListener('resize',draw);
</script></body></html>'''
