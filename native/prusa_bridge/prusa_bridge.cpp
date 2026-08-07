#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <algorithm>
#include <cmath>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "libslic3r/ExPolygon.hpp"
#include "libslic3r/Model.hpp"
#include "libslic3r/Point.hpp"
#include "libslic3r/Print.hpp"
#include "libslic3r/PrintConfig.hpp"
#include "libslic3r/TriangleMesh.hpp"
#include "libslic3r/TriangleMeshSlicer.hpp"

namespace py = pybind11;

namespace {

using Slic3r::ExPolygons;
using Slic3r::MeshSlicingParams;
using Slic3r::MeshSlicingParamsEx;

MeshSlicingParams::SlicingMode slicing_mode_from_name(const std::string &name)
{
    if (name == "regular")
        return MeshSlicingParams::SlicingMode::Regular;
    if (name == "even_odd")
        return MeshSlicingParams::SlicingMode::EvenOdd;
    if (name == "positive")
        return MeshSlicingParams::SlicingMode::Positive;
    if (name == "positive_largest_contour")
        return MeshSlicingParams::SlicingMode::PositiveLargestContour;
    throw std::invalid_argument("mode must be regular, even_odd, positive, or positive_largest_contour");
}

indexed_triangle_set mesh_from_arrays(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &vertices,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &faces)
{
    const py::buffer_info vertex_info = vertices.request();
    const py::buffer_info face_info = faces.request();
    if (vertex_info.ndim != 2 || vertex_info.shape[1] != 3)
        throw py::value_error("vertices must have shape [vertex_count, 3]");
    if (face_info.ndim != 2 || face_info.shape[1] != 3)
        throw py::value_error("faces must have shape [face_count, 3]");

    const auto vertex_count = static_cast<size_t>(vertex_info.shape[0]);
    const auto face_count = static_cast<size_t>(face_info.shape[0]);
    const auto *vertex_data = static_cast<const float *>(vertex_info.ptr);
    const auto *face_data = static_cast<const int32_t *>(face_info.ptr);

    indexed_triangle_set mesh;
    mesh.vertices.reserve(vertex_count);
    mesh.indices.reserve(face_count);
    for (size_t index = 0; index < vertex_count; ++index) {
        const float x = vertex_data[index * 3];
        const float y = vertex_data[index * 3 + 1];
        const float z = vertex_data[index * 3 + 2];
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
            throw py::value_error("vertices must contain only finite values");
        mesh.vertices.emplace_back(x, y, z);
    }
    for (size_t index = 0; index < face_count; ++index) {
        const int32_t a = face_data[index * 3];
        const int32_t b = face_data[index * 3 + 1];
        const int32_t c = face_data[index * 3 + 2];
        if (a < 0 || b < 0 || c < 0 ||
            static_cast<size_t>(a) >= vertex_count ||
            static_cast<size_t>(b) >= vertex_count ||
            static_cast<size_t>(c) >= vertex_count)
            throw py::value_error("faces contain an out-of-range vertex index");
        mesh.indices.emplace_back(a, b, c);
    }
    // STL input commonly repeats every triangle's three vertices.  Prusa's
    // manifold-aware slicer needs a shared-vertex topology to distinguish a
    // real through-hole from many unrelated triangle boundaries.  Retaining
    // the raw duplicated vertices made valid holes collapse into tiny
    // fragments during the native slice stage.
    Slic3r::its_merge_vertices(mesh);
    return mesh;
}

py::list polygon_to_python(const Slic3r::Polygon &polygon)
{
    py::list points;
    for (const Slic3r::Point &point : polygon.points) {
        const Slic3r::Vec2d unscaled = Slic3r::unscale(point);
        points.append(py::make_tuple(unscaled.x(), unscaled.y()));
    }
    return points;
}

py::list expolygons_to_python(const std::vector<ExPolygons> &slices)
{
    py::list layers;
    for (const ExPolygons &slice : slices) {
        py::list layer;
        for (const Slic3r::ExPolygon &expolygon : slice) {
            py::dict result;
            result["outer"] = polygon_to_python(expolygon.contour);
            py::list holes;
            for (const Slic3r::Polygon &hole : expolygon.holes)
                holes.append(polygon_to_python(hole));
            result["holes"] = std::move(holes);
            layer.append(std::move(result));
        }
        layers.append(std::move(layer));
    }
    return layers;
}

py::list slice_expolygons(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &vertices,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &faces,
    const std::vector<float> &z_values,
    const std::string &mode,
    float closing_radius,
    float extra_offset,
    double resolution)
{
    if (z_values.empty())
        throw py::value_error("z_values must not be empty");
    if (!std::isfinite(closing_radius) || !std::isfinite(extra_offset) || !std::isfinite(resolution))
        throw py::value_error("slicing parameters must be finite");
    for (const float z : z_values) {
        if (!std::isfinite(z))
            throw py::value_error("z_values must contain only finite values");
    }

    indexed_triangle_set mesh = mesh_from_arrays(vertices, faces);
    MeshSlicingParamsEx parameters;
    parameters.mode = slicing_mode_from_name(mode);
    parameters.closing_radius = closing_radius;
    parameters.extra_offset = extra_offset;
    parameters.resolution = resolution;

    std::vector<ExPolygons> slices;
    {
        py::gil_scoped_release release;
        slices = Slic3r::slice_mesh_ex(mesh, z_values, parameters);
    }
    return expolygons_to_python(slices);
}

struct MotionLayer {
    double z = 0.0;
    std::vector<std::vector<Slic3r::Vec3d>> paths;
    std::vector<std::vector<double>> extrusion;
    std::vector<std::string> roles;
    std::vector<std::vector<Slic3r::Vec3d>> travel;
    struct Event {
        std::string kind;
        size_t index = 0;
    };
    std::vector<Event> motions;
};

bool same_point(const Slic3r::Vec3d &left, const Slic3r::Vec3d &right)
{
    return (left - right).squaredNorm() <= 1e-14;
}

std::string upper_copy(std::string value)
{
    for (char &character : value)
        character = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
    return value;
}

std::string path_role_from_gcode_type(const std::string &type)
{
    const std::string upper = upper_copy(type);
    if (upper.find("BRIM") != std::string::npos)
        return "brim";
    if (upper.find("SUPPORT MATERIAL") != std::string::npos)
        return "raft";
    if (upper.find("EXTERNAL PERIMETER") != std::string::npos)
        return "outer_contour";
    if (upper.find("PERIMETER") != std::string::npos)
        return "inner_contour";
    if (upper.find("INFILL") != std::string::npos)
        return "infill";
    return "other";
}

bool gcode_value(const std::string &line, char wanted, double &value)
{
    const char target = static_cast<char>(std::toupper(static_cast<unsigned char>(wanted)));
    const char *cursor = line.c_str();
    while (*cursor != '\0') {
        if (std::toupper(static_cast<unsigned char>(*cursor)) == target &&
            (cursor == line.c_str() || std::isspace(static_cast<unsigned char>(cursor[-1])))) {
            char *end = nullptr;
            const double parsed = std::strtod(cursor + 1, &end);
            if (end != cursor + 1) {
                value = parsed;
                return true;
            }
        }
        ++cursor;
    }
    return false;
}

bool gcode_command(const std::string &line, char &letter, int &code)
{
    const char *cursor = line.c_str();
    while (std::isspace(static_cast<unsigned char>(*cursor)))
        ++cursor;
    if (*cursor == ';' || *cursor == '\0')
        return false;
    const char command_letter = static_cast<char>(std::toupper(static_cast<unsigned char>(*cursor)));
    if (command_letter != 'G' && command_letter != 'M')
        return false;
    char *end = nullptr;
    const long parsed = std::strtol(cursor + 1, &end, 10);
    if (end == cursor + 1)
        return false;
    letter = command_letter;
    code = static_cast<int>(parsed);
    return true;
}

void append_deposit(
    MotionLayer &layer,
    const Slic3r::Vec3d &from,
    const Slic3r::Vec3d &to,
    double e_from,
    double e_to,
    const std::string &role)
{
    if (!layer.paths.empty() && layer.roles.back() == role &&
        same_point(layer.paths.back().back(), from)) {
        layer.paths.back().push_back(to);
        layer.extrusion.back().push_back(e_to);
        return;
    }
    layer.paths.push_back({from, to});
    layer.extrusion.push_back({e_from, e_to});
    layer.roles.push_back(role);
    layer.motions.push_back({"deposit", layer.paths.size() - 1});
}

void append_travel(
    MotionLayer &layer,
    const Slic3r::Vec3d &from,
    const Slic3r::Vec3d &to)
{
    if (!layer.travel.empty() && same_point(layer.travel.back().back(), from)) {
        layer.travel.back().push_back(to);
        return;
    }
    layer.travel.push_back({from, to});
    layer.motions.push_back({"travel", layer.travel.size() - 1});
}

py::list vec_paths_to_python(const std::vector<std::vector<Slic3r::Vec3d>> &paths)
{
    py::list result;
    for (const auto &path : paths) {
        py::list points;
        for (const Slic3r::Vec3d &point : path)
            points.append(py::make_tuple(point.x(), point.y(), point.z()));
        result.append(std::move(points));
    }
    return result;
}

py::list extrusion_to_python(const std::vector<std::vector<double>> &values)
{
    py::list result;
    for (const auto &path_values : values) {
        py::list path;
        for (double value : path_values)
            path.append(value);
        result.append(std::move(path));
    }
    return result;
}

py::list motions_to_python(const std::vector<MotionLayer::Event> &motions)
{
    py::list result;
    for (const MotionLayer::Event &motion : motions) {
        py::dict event;
        event["kind"] = motion.kind;
        event["index"] = motion.index;
        result.append(std::move(event));
    }
    return result;
}

py::dict parse_print_gcode(const std::filesystem::path &path)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("PrusaSlicer did not produce readable G-code");

    std::map<long long, MotionLayer> layers;
    Slic3r::Vec3d position(0.0, 0.0, 0.0);
    bool absolute_xyz = true;
    bool absolute_e = true;
    double raw_e = 0.0;
    double cumulative_e = 0.0;
    std::string current_role = "other";
    std::string line;
    while (std::getline(input, line)) {
        const size_t comment = line.find(';');
        if (comment != std::string::npos) {
            const std::string text = line.substr(comment + 1);
            const std::string upper = upper_copy(text);
            constexpr const char *prefix = "TYPE:";
            if (upper.rfind(prefix, 0) == 0)
                current_role = path_role_from_gcode_type(text.substr(std::char_traits<char>::length(prefix)));
            line.erase(comment);
        }

        char letter = '\0';
        int code = 0;
        if (!gcode_command(line, letter, code))
            continue;
        if (letter == 'G' && code == 90) { absolute_xyz = true; continue; }
        if (letter == 'G' && code == 91) { absolute_xyz = false; continue; }
        if (letter == 'M' && code == 82) { absolute_e = true; continue; }
        if (letter == 'M' && code == 83) { absolute_e = false; continue; }

        double x = 0.0, y = 0.0, z = 0.0, e = 0.0;
        const bool has_x = gcode_value(line, 'X', x);
        const bool has_y = gcode_value(line, 'Y', y);
        const bool has_z = gcode_value(line, 'Z', z);
        const bool has_e = gcode_value(line, 'E', e);
        if (letter == 'G' && code == 92) {
            if (has_x) position.x() = x;
            if (has_y) position.y() = y;
            if (has_z) position.z() = z;
            if (has_e) raw_e = e;
            continue;
        }
        if (letter != 'G' || (code != 0 && code != 1))
            continue;

        const Slic3r::Vec3d before = position;
        if (has_x) position.x() = absolute_xyz ? x : position.x() + x;
        if (has_y) position.y() = absolute_xyz ? y : position.y() + y;
        if (has_z) position.z() = absolute_xyz ? z : position.z() + z;
        const double next_raw_e = has_e ? (absolute_e ? e : raw_e + e) : raw_e;
        const double deposited = std::max(0.0, next_raw_e - raw_e);
        const bool xy_move = (position.head<2>() - before.head<2>()).squaredNorm() > 1e-14;
        const bool planar = std::abs(position.z() - before.z()) <= 1e-7;
        if (xy_move && planar && position.z() > 0.0) {
            const long long layer_key = static_cast<long long>(std::llround(position.z() * 1000000.0));
            MotionLayer &layer = layers[layer_key];
            layer.z = position.z();
            if (deposited > 1e-12) {
                append_deposit(layer, before, position, cumulative_e, cumulative_e + deposited, current_role);
                cumulative_e += deposited;
            } else {
                append_travel(layer, before, position);
            }
        }
        raw_e = next_raw_e;
    }

    py::dict result;
    py::list output_layers;
    for (const auto &[_, layer] : layers) {
        py::dict output_layer;
        output_layer["z"] = layer.z;
        output_layer["paths"] = vec_paths_to_python(layer.paths);
        output_layer["extrusion"] = extrusion_to_python(layer.extrusion);
        output_layer["roles"] = layer.roles;
        output_layer["travel"] = vec_paths_to_python(layer.travel);
        output_layer["motions"] = motions_to_python(layer.motions);
        output_layers.append(std::move(output_layer));
    }
    result["layers"] = std::move(output_layers);
    return result;
}

py::dict slice_print_paths(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &vertices,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &faces,
    double layer_height,
    double first_layer_height,
    double line_width,
    int perimeter_count,
    double infill_density,
    const std::string &infill_pattern,
    const std::vector<double> &fill_angle_schedule,
    double perimeter_infill_overlap,
    int raft_layers,
    double raft_expansion,
    double raft_first_layer_density,
    double raft_first_layer_expansion,
    double raft_contact_distance,
    double raft_contact_layer_height,
    double raft_contact_density,
    double raft_contact_extrusion_width,
    const std::string &perimeter_generator,
    bool gap_fill_enabled,
    const std::optional<double> &infill_anchor,
    const std::optional<double> &infill_anchor_max,
    const std::optional<double> &external_perimeter_width,
    const std::optional<double> &perimeter_width,
    const std::optional<double> &infill_width,
    double xy_size_compensation,
    double elephant_foot_compensation,
    double avoid_crossing_max_detour,
    const std::string &seam_position,
    bool brim_enabled,
    double brim_width,
    const std::string &brim_type,
    double brim_separation)
{
    if (!std::isfinite(layer_height) || !std::isfinite(first_layer_height) || !std::isfinite(line_width) || !std::isfinite(infill_density) || !std::isfinite(perimeter_infill_overlap) ||
        !std::isfinite(raft_expansion) || !std::isfinite(raft_first_layer_density) || !std::isfinite(raft_first_layer_expansion) || !std::isfinite(raft_contact_distance) ||
        !std::isfinite(raft_contact_layer_height) || !std::isfinite(raft_contact_density) || !std::isfinite(raft_contact_extrusion_width) ||
        !std::isfinite(xy_size_compensation) || !std::isfinite(elephant_foot_compensation) || !std::isfinite(avoid_crossing_max_detour) ||
        !std::isfinite(brim_width) || !std::isfinite(brim_separation) ||
        layer_height <= 0.0 || line_width <= 0.0 || perimeter_count < 0 ||
        infill_density < 0.0 || infill_density > 100.0 ||
        perimeter_infill_overlap < 0.0 || perimeter_infill_overlap >= 100.0 ||
        raft_layers < 0 || raft_expansion < 0.0 ||
        raft_first_layer_density < 10.0 || raft_first_layer_density > 100.0 ||
        raft_first_layer_expansion < 0.0 || raft_contact_distance < 0.0 ||
        raft_contact_layer_height < 0.0 || raft_contact_density < 0.0 || raft_contact_density > 100.0 || raft_contact_extrusion_width < 0.0 ||
        elephant_foot_compensation < 0.0 || avoid_crossing_max_detour < 0.0 ||
        brim_width < 0.0 || brim_separation < 0.0)
        throw py::value_error("invalid Prusa print-path configuration");
    if (perimeter_generator != "arachne" && perimeter_generator != "classic")
        throw py::value_error("perimeter_generator must be arachne or classic");
    if (seam_position != "aligned" && seam_position != "nearest" &&
        seam_position != "rear" && seam_position != "random")
        throw py::value_error("unsupported seam_position");
    if (brim_type != "outer_only" && brim_type != "outer_and_inner" && brim_type != "no_brim")
        throw py::value_error("unsupported brim_type");
    for (const std::optional<double> *value : {
            &infill_anchor,
            &infill_anchor_max,
            &external_perimeter_width,
            &perimeter_width,
            &infill_width,
        }) {
        if (value->has_value() && (!std::isfinite(**value) || **value < 0.0))
            throw py::value_error("advanced Prusa dimensions must be non-negative finite values");
    }
    for (const std::optional<double> *value : {
            &external_perimeter_width,
            &perimeter_width,
            &infill_width,
        }) {
        if (value->has_value() && **value <= 0.0)
            throw py::value_error("Prusa extrusion widths must be positive");
    }
    if (first_layer_height <= 0.0)
        first_layer_height = layer_height;
    for (const double angle : fill_angle_schedule)
        if (!std::isfinite(angle))
            throw py::value_error("fill_angle_schedule must contain only finite angles");

    indexed_triangle_set its = mesh_from_arrays(vertices, faces);
    double model_height = 0.0;
    for (const Slic3r::Vec3f &vertex : its.vertices)
        model_height = std::max(model_height, static_cast<double>(vertex.z()));
    Slic3r::Model model;
    Slic3r::ModelObject *object = model.add_object();
    object->name = "kuka_slicer_part";
    object->add_volume(Slic3r::TriangleMesh(std::move(its)));
    object->add_instance();

    Slic3r::DynamicPrintConfig config;
    config.apply(Slic3r::FullPrintConfig::defaults());
    config.set("layer_height", layer_height);
    config.set_deserialize_strict("first_layer_height", std::to_string(first_layer_height));
    config.set("perimeters", perimeter_count);
    config.set_deserialize_strict("fill_density", std::to_string(infill_density) + "%");
    config.set_deserialize_strict("fill_pattern", infill_pattern);
    config.set_deserialize_strict("infill_overlap", std::to_string(perimeter_infill_overlap) + "%");
    config.set_deserialize_strict("extrusion_width", std::to_string(line_width));
    // The bridge treats line_width as the process nozzle/track width.  Prusa's
    // default 0.4 mm nozzle would otherwise reject the application's 0.5 mm+
    // resin layers during Print::validate().
    config.set_deserialize_strict("nozzle_diameter", std::to_string(line_width));
    config.set("skirts", 0);
    config.set("brim_width", brim_enabled ? brim_width : 0.0);
    config.set_deserialize_strict("brim_type", brim_type);
    config.set("brim_separation", brim_separation);
    config.set("support_material", false);
    config.set("support_material_auto", false);
    config.set("raft_layers", raft_layers);
    config.set("raft_expansion", raft_expansion);
    config.set_deserialize_strict("raft_first_layer_density", std::to_string(raft_first_layer_density) + "%");
    config.set("raft_first_layer_expansion", raft_first_layer_expansion);
    config.set("raft_contact_distance", raft_contact_distance);
    config.set("raft_contact_layer_height", raft_contact_layer_height);
    config.set_deserialize_strict("raft_contact_density", std::to_string(raft_contact_density) + "%");
    config.set("raft_contact_extrusion_width", raft_contact_extrusion_width);
    config.set("wipe_tower", false);
    config.set("avoid_crossing_perimeters", true);
    config.set_deserialize_strict("perimeter_generator", perimeter_generator);
    config.set("gap_fill_enabled", gap_fill_enabled);
    if (infill_anchor.has_value())
        config.set_deserialize_strict("infill_anchor", std::to_string(*infill_anchor));
    if (infill_anchor_max.has_value())
        config.set_deserialize_strict("infill_anchor_max", std::to_string(*infill_anchor_max));
    if (external_perimeter_width.has_value())
        config.set_deserialize_strict("external_perimeter_extrusion_width", std::to_string(*external_perimeter_width));
    if (perimeter_width.has_value())
        config.set_deserialize_strict("perimeter_extrusion_width", std::to_string(*perimeter_width));
    if (infill_width.has_value())
        config.set_deserialize_strict("infill_extrusion_width", std::to_string(*infill_width));
    config.set("xy_size_compensation", xy_size_compensation);
    config.set("elefant_foot_compensation", elephant_foot_compensation);
    config.set_deserialize_strict(
        "avoid_crossing_perimeters_max_detour",
        std::to_string(avoid_crossing_max_detour));
    config.set_deserialize_strict("seam_position", seam_position);

    if (!fill_angle_schedule.empty()) {
        double bottom_z = 0.0;
        size_t layer_index = 0;
        while (bottom_z < model_height + 1e-7) {
            const double top_z = bottom_z + (layer_index == 0 ? first_layer_height : layer_height);
            // FillRectilinear alternates 90 degrees on odd layers. The layer
            // override compensates for that native behavior so the caller's
            // schedule describes the actual printed direction.
            double configured_angle = fill_angle_schedule[layer_index % fill_angle_schedule.size()] -
                ((layer_index & 1U) ? 90.0 : 0.0);
            configured_angle = std::fmod(configured_angle, 360.0);
            if (configured_angle < 0.0)
                configured_angle += 360.0;
            // A layer-range config is not a bare option dictionary.  Prusa's
            // own GUI seeds each range with these two fields before applying
            // an override (see ObjectList::get_default_layer_config()).
            // Print::apply() uses them while it creates PrintRegions, so a
            // range containing only fill_angle can dereference a missing
            // layer-height / extruder option in native code.
            Slic3r::ModelConfig layer_config;
            layer_config.set("layer_height", top_z - bottom_z);
            layer_config.set("extruder", 0);
            layer_config.set("fill_angle", configured_angle);
            object->layer_config_ranges.emplace(
                Slic3r::t_layer_height_range(bottom_z, top_z),
                std::move(layer_config));
            bottom_z = top_z;
            ++layer_index;
        }
    }

    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path gcode_path = std::filesystem::temp_directory_path() /
        ("kuka-prusa-paths-" + std::to_string(unique) + ".gcode");
    try {
        {
            py::gil_scoped_release release;
            Slic3r::Print print;
            print.apply(model, config);
            const std::string validation_error = print.validate();
            if (!validation_error.empty())
                throw std::runtime_error("PrusaSlicer print validation failed: " + validation_error);
            print.process();
            print.export_gcode(gcode_path.string(), nullptr);
        }
        py::dict result = parse_print_gcode(gcode_path);
        std::ifstream gcode_input(gcode_path, std::ios::binary);
        if (!gcode_input)
            throw std::runtime_error("PrusaSlicer did not produce readable G-code");
        result["gcode"] = py::bytes(std::string(
            std::istreambuf_iterator<char>(gcode_input),
            std::istreambuf_iterator<char>()));
        gcode_input.close();
        std::filesystem::remove(gcode_path);
        return result;
    } catch (...) {
        std::error_code error;
        std::filesystem::remove(gcode_path, error);
        throw;
    }
}

} // namespace

PYBIND11_MODULE(prusa_bridge, module)
{
    module.doc() = "PrusaSlicer geometry and full FFF path bridge for KUKA Surface Slicer.";
    module.attr("__version__") = "PrusaSlicer-2.9.6";
    module.def(
        "slice_expolygons",
        &slice_expolygons,
        py::arg("vertices"),
        py::arg("faces"),
        py::arg("z_values"),
        py::arg("mode") = "regular",
        py::arg("closing_radius") = 0.0F,
        py::arg("extra_offset") = 0.0F,
        py::arg("resolution") = 0.0,
        "Slice a triangle mesh at one or more Z values into Prusa ExPolygons.");
    module.def(
        "slice_print_paths",
        &slice_print_paths,
        py::arg("vertices"),
        py::arg("faces"),
        py::kw_only(),
        py::arg("layer_height"),
        py::arg("first_layer_height") = 0.0,
        py::arg("line_width"),
        py::arg("perimeter_count"),
        py::arg("infill_density"),
        py::arg("infill_pattern"),
        py::arg("fill_angle_schedule") = std::vector<double>{},
        py::arg("perimeter_infill_overlap") = 0.0,
        py::arg("raft_layers") = 0,
        py::arg("raft_expansion") = 3.0,
        py::arg("raft_first_layer_density") = 80.0,
        py::arg("raft_first_layer_expansion") = 3.0,
        py::arg("raft_contact_distance") = 0.25,
        py::arg("raft_contact_layer_height") = 0.0,
        py::arg("raft_contact_density") = 0.0,
        py::arg("raft_contact_extrusion_width") = 0.0,
        py::arg("perimeter_generator") = "arachne",
        py::arg("gap_fill_enabled") = true,
        py::arg("infill_anchor") = std::nullopt,
        py::arg("infill_anchor_max") = std::nullopt,
        py::arg("external_perimeter_width") = std::nullopt,
        py::arg("perimeter_width") = std::nullopt,
        py::arg("infill_width") = std::nullopt,
        py::arg("xy_size_compensation") = 0.0,
        py::arg("elephant_foot_compensation") = 0.0,
        py::arg("avoid_crossing_max_detour") = 0.0,
        py::arg("seam_position") = "random",
        py::arg("brim_enabled") = false,
        py::arg("brim_width") = 5.0,
        py::arg("brim_type") = "outer_only",
        py::arg("brim_separation") = 0.0,
        "Run the Prusa FFF path planner and return deposited XYZ/E paths plus travel XYZ paths.");
}
