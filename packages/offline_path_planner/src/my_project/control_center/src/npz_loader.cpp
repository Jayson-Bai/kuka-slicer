#include "control_center/npz_loader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

#include "cnpy.h"

namespace fs = std::filesystem;

namespace control_center
{

namespace
{

std::string trim_cstr(const char * data, size_t max_len)
{
  size_t len = 0;
  while (len < max_len && data[len] != '\0') {
    ++len;
  }
  return std::string(data, len);
}

std::vector<std::string> decode_fixed_strings(const cnpy::NpyArray & arr)
{
  std::vector<std::string> out;
  const size_t n = arr.num_vals;
  const size_t stride = arr.word_size;
  const char * raw = arr.data<char>();
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    out.emplace_back(trim_cstr(raw + i * stride, stride));
  }
  return out;
}

template<typename T>
std::vector<T> to_vec(const cnpy::NpyArray & arr)
{
  const T * data = arr.data<T>();
  return std::vector<T>(data, data + arr.num_vals);
}

void load_vocab(
  const cnpy::npz_t & npz,
  const std::string & keys_name,
  const std::string & vals_name,
  std::unordered_map<uint8_t, std::string> & out)
{
  auto keys = decode_fixed_strings(npz.at(keys_name));
  auto vals = to_vec<uint8_t>(npz.at(vals_name));
  out.clear();
  const size_t n = std::min(keys.size(), vals.size());
  for (size_t i = 0; i < n; ++i) {
    out[vals[i]] = keys[i];
  }
}

fs::path timing_sidecar_for(const std::string & path)
{
  fs::path p(path);
  std::string stem = p.stem().string();
  const std::string filename = p.filename().string();
  if (p.extension() == ".json" || filename.find("_manifest.json") != std::string::npos) {
    const std::string suffix = "_manifest";
    if (stem.size() >= suffix.size() &&
      stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) == 0)
    {
      stem.erase(stem.size() - suffix.size());
    }
    return p.parent_path() / (stem + "_timing.json");
  }

  const std::string part_marker = "_part";
  const auto part_pos = stem.rfind(part_marker);
  if (part_pos != std::string::npos && part_pos + part_marker.size() < stem.size()) {
    bool numeric_suffix = true;
    for (size_t i = part_pos + part_marker.size(); i < stem.size(); ++i) {
      numeric_suffix = numeric_suffix && std::isdigit(
        static_cast<unsigned char>(stem[i]));
    }
    if (numeric_suffix) {
      stem.erase(part_pos);
    }
  }
  return p.parent_path() / (stem + ".timing.json");
}

bool read_json_number(
  const std::string & content, const std::string & key, double & value)
{
  const std::string token = "\"" + key + "\"";
  const auto key_pos = content.find(token);
  if (key_pos == std::string::npos) {
    return false;
  }
  const auto colon_pos = content.find(':', key_pos + token.size());
  if (colon_pos == std::string::npos) {
    return false;
  }
  const auto number_pos = content.find_first_not_of(" \t\r\n", colon_pos + 1);
  if (number_pos == std::string::npos) {
    return false;
  }
  const char * begin = content.c_str() + number_pos;
  char * end = nullptr;
  value = std::strtod(begin, &end);
  return end != begin && std::isfinite(value);
}

}  // namespace

NpzLoader::NpzLoader(const std::string & path, size_t preload_chunks)
: preload_chunks_(preload_chunks)
{
  files_ = resolve_files(path);
  if (files_.empty()) {
    error_ = "no npz files found for: " + path;
    ok_ = false;
    return;
  }
  try {
    load_timing_metadata(path);
    load_initial();
    ok_ = true;
  } catch (const std::exception & e) {
    error_ = e.what();
    ok_ = false;
  }
}

void NpzLoader::load_timing_metadata(const std::string & path)
{
  timing_metadata_valid_ = false;
  timing_valid_ = false;
  total_planned_time_s_ = 0.0;

  const fs::path sidecar = timing_sidecar_for(path);
  std::ifstream in(sidecar);
  if (!in) {
    return;
  }
  const std::string content(
    (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  double total = 0.0;
  if (!read_json_number(content, "total_planned_time_s", total) || total < 0.0) {
    return;
  }
  total_planned_time_s_ = total;
  timing_metadata_valid_ = true;
  timing_valid_ = timing_rows_valid_ && timing_metadata_valid_;
}

bool NpzLoader::has_next() const
{
  if (!cache_.empty()) {
    const auto & chunk = cache_.front();
    if (chunk_row_idx_ < chunk.size) {
      return true;
    }
  }
  return next_file_idx_ < files_.size();
}

bool NpzLoader::next_row(NpzRow & out)
{
  if (cache_.empty() || chunk_row_idx_ >= cache_.front().size) {
    if (!load_next_chunk()) {
      return false;
    }
  }
  const auto & c = cache_.front();
  const size_t i = chunk_row_idx_++;
  out.seq = c.seq[i];
  out.x = c.x[i];
  out.y = c.y[i];
  out.z = c.z[i];
  out.a = c.a[i];
  out.b = c.b[i];
  out.c = c.c[i];
  out.e = c.e[i];
  out.tool_id = c.tool_id[i];
  out.move_type = c.move_type[i];
  out.src_line = c.src_line[i];
  out.event_flag = c.event_flag[i];
  out.event_type = c.event_type[i];
  out.payload = c.payload[i];
  out.trigger_seq = c.trigger_seq[i];
  out.layer_index = c.layer_index[i];
  out.total_layers = c.total_layers[i];
  out.path_id = c.path_id[i];
  out.path_end_flag = c.path_end_flag[i] != 0;
  out.planned_time_s = 0.0F;
  out.planned_total_time_s = static_cast<float>(total_planned_time_s_);
  out.planned_time_valid = false;
  if (timing_valid_ && i < c.planned_time_s.size() &&
    std::isfinite(c.planned_time_s[i]))
  {
    out.planned_time_s = c.planned_time_s[i];
    out.planned_time_valid = true;
  }

  if (chunk_row_idx_ >= c.size) {
    cache_.pop_front();
    chunk_row_idx_ = 0;
    ensure_preload();
  }
  return true;
}

void NpzLoader::seek(uint32_t target_seq)
{
  cache_.clear();
  chunk_row_idx_ = 0;
  next_file_idx_ = 0;
  ok_ = true;
  error_.clear();

  while (next_file_idx_ < files_.size()) {
    if (!load_next_chunk()) {
      break;
    }
    if (cache_.back().seq.empty()) {
      cache_.pop_back();
      continue;
    }
    if (cache_.back().seq.back() >= target_seq) {
      const auto & c = cache_.back();
      for (size_t i = 0; i < c.size; ++i) {
        if (c.seq[i] >= target_seq) {
          chunk_row_idx_ = i;
          break;
        }
      }
      ensure_preload();
      return;
    }
    cache_.pop_back();
  }
}

void NpzLoader::load_initial()
{
  for (size_t i = 0; i < preload_chunks_; ++i) {
    if (!load_next_chunk()) {
      break;
    }
  }
}

void NpzLoader::ensure_preload()
{
  while (cache_.size() < preload_chunks_) {
    if (!load_next_chunk()) {
      break;
    }
  }
}

bool NpzLoader::load_next_chunk()
{
  if (next_file_idx_ >= files_.size()) {
    return false;
  }
  const std::string & file = files_[next_file_idx_++];
  cache_.push_back(load_chunk(file));
  return true;
}

std::vector<std::string> NpzLoader::resolve_files(const std::string & path) const
{
  fs::path p(path);
  std::vector<std::string> out;

  if (p.extension() == ".json" ||
    p.filename().string().find("_manifest.json") != std::string::npos)
  {
    return resolve_from_manifest(path);
  }

  return resolve_from_base(path);
}

std::vector<std::string> NpzLoader::resolve_from_base(const std::string & path) const
{
  fs::path p(path);
  std::vector<std::string> out;

  if (p.extension() != ".npz") {
    p.replace_extension(".npz");
  }

  if (fs::exists(p)) {
    out.push_back(p.string());
    return out;
  }

  const fs::path dir = p.parent_path().empty() ? fs::current_path() : p.parent_path();
  const std::string stem = p.stem().string();
  const std::string prefix = stem + "_part";

  if (!fs::exists(dir)) {
    return out;
  }

  for (const auto & entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto & f = entry.path();
    if (f.extension() != ".npz") {
      continue;
    }
    const std::string name = f.stem().string();
    if (name.rfind(prefix, 0) == 0) {
      out.push_back(f.string());
    }
  }

  std::sort(out.begin(), out.end());
  return out;
}

std::vector<std::string> NpzLoader::resolve_from_manifest(const std::string & path) const
{
  fs::path manifest_path(path);
  if (manifest_path.extension() != ".json") {
    manifest_path.replace_extension(".json");
  }
  if (!fs::exists(manifest_path)) {
    return {};
  }

  std::ifstream in(manifest_path);
  if (!in) {
    return {};
  }
  std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::vector<std::string> out;

  const std::string key = "\"base_path\"";
  std::size_t pos = 0;
  while (true) {
    pos = content.find(key, pos);
    if (pos == std::string::npos) {
      break;
    }
    pos = content.find(":", pos);
    if (pos == std::string::npos) {
      break;
    }
    pos = content.find("\"", pos);
    if (pos == std::string::npos) {
      break;
    }
    auto start = pos + 1;
    auto end = content.find("\"", start);
    if (end == std::string::npos) {
      break;
    }
    std::string base_path = content.substr(start, end - start);
    fs::path bp(base_path);
    if (bp.is_relative()) {
      bp = manifest_path.parent_path() / bp;
    }
    if (bp.is_absolute() && !fs::exists(bp)) {
      // Fallback: manifest moved to another machine, rebuild relative path.
      std::vector<fs::path> tail;
      for (const auto & p : bp.relative_path()) {
        tail.push_back(p);
      }
      if (tail.size() >= 2) {
        fs::path alt = manifest_path.parent_path() / tail[tail.size() - 2] / tail[tail.size() - 1];
        if (fs::exists(alt) || fs::exists(alt.parent_path())) {
          bp = alt;
        }
      }
    }
    auto files = resolve_from_base(bp.string());
    out.insert(out.end(), files.begin(), files.end());
    pos = end + 1;
  }

  return out;
}

NpzChunk NpzLoader::load_chunk(const std::string & file)
{
  cnpy::npz_t npz = cnpy::npz_load(file);

  if (move_type_vocab_.empty() && npz.count("move_type_vocab_keys")) {
    load_vocab(npz, "move_type_vocab_keys", "move_type_vocab_vals", move_type_vocab_);
  }
  if (event_type_vocab_.empty() && npz.count("event_type_vocab_keys")) {
    load_vocab(npz, "event_type_vocab_keys", "event_type_vocab_vals", event_type_vocab_);
  }

  NpzChunk c;
  c.seq = to_vec<uint32_t>(npz.at("seq"));
  c.x = to_vec<float>(npz.at("x"));
  c.y = to_vec<float>(npz.at("y"));
  c.z = to_vec<float>(npz.at("z"));
  c.a = to_vec<float>(npz.at("a"));
  c.b = to_vec<float>(npz.at("b"));
  c.c = to_vec<float>(npz.at("c"));
  c.e = to_vec<float>(npz.at("e"));
  {
    auto raw_tool = to_vec<uint8_t>(npz.at("tool_id"));
    c.tool_id.reserve(raw_tool.size());
    for (auto v : raw_tool) {
      c.tool_id.push_back(static_cast<int32_t>(v));
    }
  }
  c.move_type = to_vec<uint8_t>(npz.at("move_type"));
  c.src_line = decode_fixed_strings(npz.at("src_line"));
  c.event_flag = to_vec<uint8_t>(npz.at("event_flag"));
  c.event_type = to_vec<uint8_t>(npz.at("event_type"));
  c.payload = decode_fixed_strings(npz.at("payload"));
  c.trigger_seq = to_vec<int32_t>(npz.at("trigger_seq"));
  c.size = c.seq.size();
  bool chunk_timing_valid = false;
  if (npz.count("planned_time_s")) {
    auto planned_time = to_vec<float>(npz.at("planned_time_s"));
    chunk_timing_valid = planned_time.size() == c.size;
    if (chunk_timing_valid) {
      for (const auto value : planned_time) {
        if (!std::isfinite(value)) {
          chunk_timing_valid = false;
          break;
        }
      }
    }
    if (chunk_timing_valid) {
      c.planned_time_s = std::move(planned_time);
    }
  }
  if (!chunk_timing_valid) {
    c.planned_time_s.assign(c.size, 0.0F);
    timing_rows_valid_ = false;
  }
  timing_valid_ = timing_rows_valid_ && timing_metadata_valid_;
  if (npz.count("layer_index")) {
    c.layer_index = to_vec<uint32_t>(npz.at("layer_index"));
  } else {
    c.layer_index.assign(c.size, 0);
  }
  if (npz.count("total_layers")) {
    c.total_layers = to_vec<uint32_t>(npz.at("total_layers"));
  } else {
    c.total_layers.assign(c.size, 0);
  }
  if (npz.count("path_id")) {
    c.path_id = to_vec<uint32_t>(npz.at("path_id"));
  } else {
    c.path_id.assign(c.size, 0);
  }
  if (npz.count("path_end_flag")) {
    c.path_end_flag = to_vec<uint8_t>(npz.at("path_end_flag"));
  } else {
    c.path_end_flag.assign(c.size, 0);
  }

  return c;
}

}  // namespace control_center
