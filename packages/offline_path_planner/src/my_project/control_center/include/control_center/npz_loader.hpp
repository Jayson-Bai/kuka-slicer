#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace control_center
{

struct NpzRow
{
  uint32_t seq{};
  float x{};
  float y{};
  float z{};
  float a{};
  float b{};
  float c{};
  float e{};
  int32_t tool_id{};
  uint8_t move_type{};
  std::string src_line;
  uint8_t event_flag{};
  uint8_t event_type{};
  std::string payload;
  int32_t trigger_seq{};
  uint32_t layer_index{};
  uint32_t total_layers{};
  uint32_t path_id{};
  bool path_end_flag{};
  float planned_time_s{};
  float planned_total_time_s{};
  bool planned_time_valid{};
};

struct NpzChunk
{
  std::vector<uint32_t> seq;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<float> a;
  std::vector<float> b;
  std::vector<float> c;
  std::vector<float> e;
  std::vector<int32_t> tool_id;
  std::vector<uint8_t> move_type;
  std::vector<std::string> src_line;
  std::vector<uint8_t> event_flag;
  std::vector<uint8_t> event_type;
  std::vector<std::string> payload;
  std::vector<int32_t> trigger_seq;
  std::vector<uint32_t> layer_index;
  std::vector<uint32_t> total_layers;
  std::vector<uint32_t> path_id;
  std::vector<uint8_t> path_end_flag;
  std::vector<float> planned_time_s;
  size_t size{0};
};

class NpzLoader
{
public:
  explicit NpzLoader(const std::string & path, size_t preload_chunks = 2);

  bool ok() const {return ok_;}
  const std::string & error() const {return error_;}

  bool timing_valid() const {return timing_valid_;}
  double total_planned_time_s() const {return total_planned_time_s_;}

  bool has_next() const;
  bool next_row(NpzRow & out);
  void seek(uint32_t target_seq);

  const std::unordered_map<uint8_t, std::string> & move_type_vocab() const
  {
    return move_type_vocab_;
  }
  const std::unordered_map<uint8_t, std::string> & event_type_vocab() const
  {
    return event_type_vocab_;
  }

private:
  void load_initial();
  void ensure_preload();
  bool load_next_chunk();

  std::vector<std::string> resolve_files(const std::string & path) const;
  std::vector<std::string> resolve_from_manifest(const std::string & path) const;
  std::vector<std::string> resolve_from_base(const std::string & path) const;
  NpzChunk load_chunk(const std::string & file);
  void load_timing_metadata(const std::string & path);

  std::unordered_map<uint8_t, std::string> move_type_vocab_;
  std::unordered_map<uint8_t, std::string> event_type_vocab_;

  std::vector<std::string> files_;
  size_t next_file_idx_{0};
  size_t preload_chunks_{2};
  bool ok_{false};
  std::string error_;
  bool timing_valid_{false};
  bool timing_rows_valid_{true};
  bool timing_metadata_valid_{false};
  double total_planned_time_s_{0.0};

  std::deque<NpzChunk> cache_;
  size_t chunk_row_idx_{0};
};

}  // namespace control_center
