"""Command line interface for external NPZ preprocessing."""

from __future__ import annotations

import argparse
import sys

from .export_runner import convert_external_npz, resolve_output_path
from .process_params import FiberProcessParams, ProcessParams, ResinProcessParams


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="External path NPZ -> my_project system NPZ")
    parser.add_argument("--source", required=True, help="External source NPZ location")
    parser.add_argument("--out", default="", help="Output system NPZ location; defaults next to source NPZ")
    parser.add_argument("--resin-layer-height-mm", type=float, default=0.5)
    parser.add_argument("--resin-extrusion-scale", type=float, default=1.0)
    parser.add_argument("--resin-feed-mm-s", type=float, default=10.0)
    parser.add_argument("--first-layer-resin-feed-mm-s", type=float, default=10.0)
    parser.add_argument("--resin-temperature-c", type=float, default=250.0)
    parser.add_argument("--resin-prime-length-mm", type=float, default=18.0)
    parser.add_argument("--resin-prime-speed-mm-s", type=float, default=15.0)
    parser.add_argument("--resin-retract-length-mm", type=float, default=15.0)
    parser.add_argument("--resin-retract-speed-mm-s", type=float, default=30.0)
    parser.add_argument("--no-resin-fan", action="store_false", dest="resin_fan")
    parser.add_argument("--fiber-layer-height-mm", type=float, default=0.1)
    parser.add_argument("--fiber-extrusion-scale", type=float, default=1.0)
    parser.add_argument("--fiber-feed-mm-s", type=float, default=10.0)
    parser.add_argument("--first-layer-fiber-feed-mm-s", type=float, default=10.0)
    parser.add_argument("--fiber-start-accel-s", type=float, default=2.0)
    parser.add_argument("--fiber-temperature-c", type=float, default=250.0)
    parser.add_argument("--fiber-prime-length-mm", type=float, default=12.0)
    parser.add_argument("--fiber-prime-speed-mm-s", type=float, default=5.0)
    parser.add_argument("--fiber-retract-length-mm", type=float, default=10.0)
    parser.add_argument("--fiber-retract-speed-mm-s", type=float, default=5.0)
    parser.add_argument("--no-fiber-fan", action="store_false", dest="fiber_fan")
    parser.add_argument("--travel-feed-mm-s", type=float, default=10.0)
    parser.add_argument("--first-layer-travel-feed-mm-s", type=float, default=10.0)
    parser.add_argument("--default-a", type=float, default=0.0)
    parser.add_argument("--default-b", type=float, default=0.0)
    parser.add_argument("--default-c", type=float, default=0.0)
    parser.add_argument("--prime-settle-s", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--cut-lift-mm", type=float, default=20.0)
    parser.add_argument("--cut-wait-s", type=float, default=15.0)
    parser.set_defaults(resin_fan=True, fiber_fan=True)
    return parser


def params_from_args(args) -> ProcessParams:
    return ProcessParams(
        resin=ResinProcessParams(
            layer_height_mm=args.resin_layer_height_mm,
            extrusion_scale=args.resin_extrusion_scale,
            feed_mm_s=args.resin_feed_mm_s,
            first_layer_feed_mm_s=args.first_layer_resin_feed_mm_s,
            temperature_c=args.resin_temperature_c,
            fan_enabled=args.resin_fan,
            prime_length_mm=args.resin_prime_length_mm,
            prime_speed_mm_s=args.resin_prime_speed_mm_s,
            retract_length_mm=args.resin_retract_length_mm,
            retract_speed_mm_s=args.resin_retract_speed_mm_s,
        ),
        fiber=FiberProcessParams(
            layer_height_mm=args.fiber_layer_height_mm,
            extrusion_scale=args.fiber_extrusion_scale,
            feed_mm_s=args.fiber_feed_mm_s,
            first_layer_feed_mm_s=args.first_layer_fiber_feed_mm_s,
            start_accel_s=args.fiber_start_accel_s,
            temperature_c=args.fiber_temperature_c,
            fan_enabled=args.fiber_fan,
            prime_length_mm=args.fiber_prime_length_mm,
            prime_speed_mm_s=args.fiber_prime_speed_mm_s,
            retract_length_mm=args.fiber_retract_length_mm,
            retract_speed_mm_s=args.fiber_retract_speed_mm_s,
        ),
        travel_feed_mm_s=args.travel_feed_mm_s,
        first_layer_travel_feed_mm_s=args.first_layer_travel_feed_mm_s,
        default_a=args.default_a,
        default_b=args.default_b,
        default_c=args.default_c,
        prime_settle_s=args.prime_settle_s,
        dt=args.dt,
    )


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = resolve_output_path(args.source, args.out)
        stats = convert_external_npz(
            args.source,
            args.out,
            params_from_args(args),
            cut_lift_mm=args.cut_lift_mm,
            cut_wait_s=args.cut_wait_s,
        )
    except Exception as exc:
        print(f"[错误] 外部 NPZ 处理失败: {exc}", file=sys.stderr)
        return 1
    print(f"[信息] 输出 NPZ 位置: {output}")
    print(
        "[信息] 外部 NPZ 处理完成: rows=%d parts=%d total=%.3fs"
        % (stats.get("rows", 0), stats.get("parts", 0), stats.get("total_s", 0.0))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
