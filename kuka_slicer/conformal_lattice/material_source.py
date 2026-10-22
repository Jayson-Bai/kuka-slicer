"""Shared designer/export source preparation; no Core resampling or file writes."""
from ..fiber_interlayers import plan_flat_resin_interlayers


def prepare_material_source(spec, core_params, params=None):
    # Local import avoids a module cycle with the HTTP export handler.
    from ..ui_server import _bool_param, _float_param
    params = params or {}
    is_planar = spec.source_provider == "planar"
    resin = core_params.resin
    layer_height = float(resin.layer_height_mm)
    conformal_fiber_enabled = _bool_param(params, "conformal_fiber_enabled", True)
    e_per_mm = float(resin.e_per_mm())
    if e_per_mm <= 0.0:
        raise ValueError("当前 Core 树脂 E/mm 必须为正数")
    bead_area = 2.0 * layer_height * float(resin.extrusion_scale)
    if bead_area <= 0.0:
        raise ValueError("当前 Core 树脂挤出倍率必须为正数")
    from .path_bridge import ExtrusionVolumeModel
    from .pipeline import run_conformal_lattice_pipeline

    run = run_conformal_lattice_pipeline(
        spec,
        physical_layer_height_mm=layer_height,
        extrusion=ExtrusionVolumeModel(
            bead_cross_section_area_mm2=bead_area,
            # This effective volume/E value makes the authoritative
            # source profile exactly match the selected Core calibration,
            # including an explicit E/mm override when one is supplied.
            e_volume_per_unit_mm3=bead_area / e_per_mm,
            preview_line_width_mm=2.0,
        ),
        # Gate 6 samples every cell against the surface triangles.  It is
        # a quality-analysis tool, not part of manufacturing generation;
        # keeping it out of this request lets the prepared one-stroke
        # conformal paths proceed directly to Core.
        validate_fill_ratio=False,
        fiber_layer_height_mm=float(core_params.fiber.layer_height_mm),
        # Plan resin layers independently of fiber. Fiber interfaces are
        # selected on that grid and get a local Z lift afterward.
        plan_for_continuous_fiber=conformal_fiber_enabled,
    )
    if run.path_graph is None:
        raise RuntimeError("共形蜂窝路径桥接未生成一笔画路径")
    # The legacy structural macro partitions are replaced unconditionally
    # by continuous courses below.  Start from the retained perimeter and
    # grip material instead of rendering/deleting that large temporary
    # graph, without changing the eventual Core SourceJob geometry.
    conformal_source_job = run.path_graph.to_external_base_source_job()
    from .brim import ConformalBrimSettings, apply_conformal_brim

    brim_report = apply_conformal_brim(
        conformal_source_job,
        ConformalBrimSettings(
            enabled=_bool_param(params, "prusa_brim_enabled", False),
            width_mm=_float_param(params, "prusa_brim_width", 5.0),
            brim_type=params.get("prusa_brim_type", ["outer_only"])[0],  # type: ignore[arg-type]
            separation_mm=_float_param(params, "prusa_brim_separation", 0.0),
            one_stroke=_bool_param(params, "prusa_brim_one_stroke", False),
            # The conformal bridge and its preview use this calibrated
            # two-millimetre bead centreline.
            line_width_mm=2.0,
        ),
    )
    from .fiber_reinforcement import (
        ContinuousCourseFiberSettings,
        apply_continuous_course_fiber_strategy,
        derive_symmetric_curvature_fiber_interfaces,
    )

    if not run.continuous_course_paths_by_layer:
        raise RuntimeError("当前共形工作区不能生成连续填充路径")
    if is_planar:
        planar_schedule = plan_flat_resin_interlayers(
            group.layer_index
            for group in conformal_source_job.material_paths
            if group.material == "R"
        )
        planar_window = planar_schedule.physical_interface_window
        first_fiber_interface, last_fiber_interface = planar_window or (1, 1)
        fiber_interface_source = planar_schedule.source
    else:
        first_fiber_interface, last_fiber_interface = derive_symmetric_curvature_fiber_interfaces(
            run.layer_embedding
        )
        fiber_interface_source = "design_json_symmetric_nonzero_curvature"
    stack_plan = run.layer_embedding.report.get("physical_stack_plan")
    if not isinstance(stack_plan, dict):
        stack_plan = {}
    # The SourceJob stores bead centre-lines. Preserve the height-guided
    # physical stack contract alongside it so the final UI report does not
    # mistake the last centre-line Z for the printed part extent.
    conformal_source_job.meta["physical_stack_plan"] = dict(stack_plan)
    planned_fiber_layers = tuple(int(value) for value in stack_plan.get("fiber_layer_indices", []))
    resin_z_preplanned_for_fiber = bool(stack_plan.get("resin_z_preplanned_for_fiber", False))
    fiber_reinforcement = apply_continuous_course_fiber_strategy(
        source_job=conformal_source_job,
        graph=run.path_graph,
        course_paths_by_layer=run.continuous_course_paths_by_layer,
        settings=ContinuousCourseFiberSettings(
            enabled=conformal_fiber_enabled,
            first_after_resin_layer_physical=first_fiber_interface,
            last_after_resin_layer_physical=last_fiber_interface,
            layer_interface_source=fiber_interface_source,
            resin_z_preplanned_for_fiber=resin_z_preplanned_for_fiber,
            planned_resin_layer_indices=planned_fiber_layers,
            # Preserve the existing extrusion policies while unifying
            # height planning; planar resin E is not changed here.
            compensate_fiber_gap_extrusion=not is_planar,
            print_outer_contour_fiber=spec.lattice.get("print_outer_contour_fiber", False),
        ),
        fiber_layer_height_mm=float(core_params.fiber.layer_height_mm),
        fiber_e_per_mm=float(core_params.fiber.e_per_mm()),
    )
    return conformal_source_job, run, brim_report, fiber_reinforcement
