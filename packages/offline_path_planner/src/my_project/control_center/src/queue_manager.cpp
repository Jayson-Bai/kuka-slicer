#include "control_center/queue_manager.hpp"

#include <algorithm>
#include <cstdlib>

namespace control_center
{

QueueManager::QueueManager(size_t low_water, size_t high_water)
: low_water_(low_water), high_water_(high_water) {}

void QueueManager::set_watermarks(size_t low_water, size_t high_water)
{
  low_water_ = low_water;
  high_water_ = high_water;
}

void QueueManager::clear()
{
  traj_queue_.clear();
  event_queue_.clear();
}

void QueueManager::fill(NpzLoader & loader)
{
  if (!loader.ok()) {
    return;
  }
  if (traj_queue_.size() >= low_water_) {
    return;
  }

  NpzRow row;
  while (traj_queue_.size() < high_water_ && loader.has_next()) {
    if (!loader.next_row(row)) {
      break;
    }

    if (row.event_flag == 1) {
      my_project_interfaces::msg::PlannedEvent ev;
      ev.event_type = loader.event_type_vocab().count(row.event_type) ?
        loader.event_type_vocab().at(row.event_type) :
        "unknown";
      ev.payload = row.payload;
      ev.event_src_line = parse_src_line(row.src_line);
      ev.trigger_seq = row.trigger_seq >= 0 ? static_cast<uint32_t>(row.trigger_seq) :
        row.seq;
      event_queue_.push_back(ev);
      continue;
    }

    my_project_interfaces::msg::TrajectoryPoint tp;
    tp.x = row.x;
    tp.y = row.y;
    tp.z = row.z;
    tp.a = row.a;
    tp.b = row.b;
    tp.c = row.c;
    tp.e = row.e;
    tp.tool_id = row.tool_id;
    tp.seq = row.seq;
    tp.layer_index = row.layer_index;
    tp.total_layers = row.total_layers;
    tp.path_id = row.path_id;
    tp.path_end_flag = row.path_end_flag;
    tp.planned_time_s = row.planned_time_s;
    tp.planned_total_time_s = row.planned_total_time_s;
    tp.planned_time_valid = row.planned_time_valid && loader.timing_valid();
    traj_queue_.push_back(tp);
  }
}

bool QueueManager::pop_next_traj(my_project_interfaces::msg::TrajectoryPoint & out)
{
  if (traj_queue_.empty()) {
    return false;
  }
  out = traj_queue_.front();
  traj_queue_.pop_front();
  return true;
}

bool QueueManager::pop_next_event(my_project_interfaces::msg::PlannedEvent & out)
{
  if (event_queue_.empty()) {
    return false;
  }
  out = event_queue_.front();
  event_queue_.pop_front();
  return true;
}

bool QueueManager::peek_next_traj_seq(uint32_t & out) const
{
  if (traj_queue_.empty()) {
    return false;
  }
  out = traj_queue_.front().seq;
  return true;
}

bool QueueManager::peek_next_event_trigger(uint32_t & out) const
{
  if (event_queue_.empty()) {
    return false;
  }
  out = event_queue_.front().trigger_seq;
  return true;
}

bool QueueManager::should_enter_wait(uint32_t next_seq) const
{
  if (event_queue_.empty()) {
    return false;
  }
  return event_queue_.front().trigger_seq <= next_seq;
}

int32_t QueueManager::parse_src_line(const std::string & s)
{
  if (s.empty()) {
    return -1;
  }
  const char * start = s.c_str();
  char * end = nullptr;
  long val = std::strtol(start, &end, 10);
  if (end == start) {
    return -1;
  }
  return static_cast<int32_t>(val);
}

}  // namespace control_center
