#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>

#include "control_center/npz_loader.hpp"
#include "my_project_interfaces/msg/planned_event.hpp"
#include "my_project_interfaces/msg/trajectory_point.hpp"

namespace control_center
{

class QueueManager
{
public:
  QueueManager(size_t low_water, size_t high_water);

  void set_watermarks(size_t low_water, size_t high_water);

  size_t traj_size() const {return traj_queue_.size();}
  size_t event_size() const {return event_queue_.size();}

  void clear();
  void fill(NpzLoader & loader);

  bool pop_next_traj(my_project_interfaces::msg::TrajectoryPoint & out);
  bool pop_next_event(my_project_interfaces::msg::PlannedEvent & out);
  bool peek_next_traj_seq(uint32_t & out) const;
  bool peek_next_event_trigger(uint32_t & out) const;
  bool should_enter_wait(uint32_t next_seq) const;

private:
  static int32_t parse_src_line(const std::string & s);

  size_t low_water_{0};
  size_t high_water_{0};
  std::deque<my_project_interfaces::msg::TrajectoryPoint> traj_queue_;
  std::deque<my_project_interfaces::msg::PlannedEvent> event_queue_;
};

}  // namespace control_center
