/*
 * computing cost-to-go matrices
 * Author: Keisuke Okumura
 * Affiliation: TokyoTech & OSX
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <queue>
#include <cmath>
#include <cassert>

namespace py = pybind11;

constexpr uint16_t MOTIONS_2D_NUM = 4;
constexpr int MOTIONS_2D[MOTIONS_2D_NUM][2] = {
  {-1,  0},  // left
  { 1,  0},  // right
  { 0, -1},  // up
  { 0,  1}   // down
};


py::array_t<int> getCostToGo2D
(
 const py::array_t<double>& goal,
 const py::array_t<int>& occupancy_map,
 const py::array_t<int>& occupancy_agent
)
{
  const auto occupancy_map_info = occupancy_map.request();
  const uint16_t map_size = occupancy_map_info.shape[0];
  const uint16_t agent_size = occupancy_agent.request().shape[0];
  assert(agent_size % 2 == 1);
  const uint16_t rad_int = (agent_size - 1) / 2;
  const uint16_t max_cost = 65535;
  std::vector<std::vector<uint16_t>>
    cost_to_go(map_size, std::vector<uint16_t>(map_size, max_cost));

  const uint16_t goal_x = *goal.data(0) < 1
    ? std::floor(*goal.data(0) * map_size)
    : map_size - 1;
  const uint16_t goal_y = *goal.data(1) < 1
    ? std::floor(*goal.data(1) * map_size)
    : map_size - 1;
  const uint16_t goal_idx = goal_y * map_size + goal_x;

  std::queue<uint16_t> OPEN;
  cost_to_go[goal_x][goal_y] = 0;
  OPEN.push(goal_idx);

  while (!OPEN.empty()) {
    auto idx = OPEN.front();
    OPEN.pop();
    int x = idx % map_size;
    int y = idx / map_size;

    // expand neighbors
    for (uint16_t i = 0; i < MOTIONS_2D_NUM; ++i) {
      int _x = x + MOTIONS_2D[i][0];
      int _y = y + MOTIONS_2D[i][1];
      if (_x < 0 || map_size <= _x || _y < 0 || map_size <= _y) continue;

      // check whether already expanded
      if (cost_to_go[_x][_y] != max_cost) continue;
      // check occupancy
      if (*occupancy_map.data(_x, _y) == 1) continue;
      // check collision
      {
        bool flg_collide = false;
        for (int k = -rad_int; k <= rad_int; ++k) {
          for (int l = -rad_int; l <= rad_int; ++l) {
            int x_k = _x + k;
            int y_l = _y + l;

            // case: outside
            if (x_k < 0 || map_size <= x_k || y_l < 0 || map_size <= y_l) {
              if (*occupancy_agent.data(k + rad_int, l + rad_int) == 1) {
                flg_collide = true;
                break;
              } else {
                continue;
              }
            }
            // case: free space
            if (*occupancy_map.data(x_k, y_l) == 0) continue;
            // case: not agent loc
            if (*occupancy_agent.data(k + rad_int, l + rad_int) == 0) continue;
            // collision
            flg_collide = true;
            break;
          }
          if (flg_collide) break;
        }
        if (flg_collide) continue;
      }

      // insert
      cost_to_go[_x][_y] = cost_to_go[x][y] + 1;
      OPEN.push(_y * map_size + _x);
    }
  }

  py::array_t<int> numpy_cost_to_go = py::cast(cost_to_go);
  return numpy_cost_to_go;
}

py::array_t<int> getFov2dOccupancyCost
(
 const py::array_t<double>& current_pos,
 const py::array_t<int>& occupancy_map,
 const py::array_t<int>& cost_to_go,
 const uint16_t fov_size,
 const bool flatten
)
{
  assert(fov_size % 2 == 1);

  const auto occupancy_map_info = occupancy_map.request();
  const uint16_t map_size = occupancy_map_info.shape[0];

  const uint16_t current_x = *current_pos.data(0) < 1
    ? std::floor(*current_pos.data(0) * map_size)
    : map_size - 1;
  const uint16_t current_y = *current_pos.data(1) < 1
    ? std::floor(*current_pos.data(1) * map_size)
    : map_size - 1;

  // determine coordination
  const uint16_t buf = (fov_size - 1) / 2;
  const uint16_t x_lb = std::max(0, current_x - buf);
  const uint16_t x_ub = std::min(map_size - 1, current_x + buf);
  const uint16_t y_lb = std::max(0, current_y - buf);
  const uint16_t y_ub = std::min(map_size - 1, current_y + buf);

  const auto c = *cost_to_go.data(current_x, current_y);

  if (!flatten) {

    std::vector<std::vector<std::vector<uint16_t>>>
      fov(2, std::vector<std::vector<uint16_t>>
          (fov_size, std::vector<uint16_t>(fov_size, 0)));

    // create fov
    for (auto x = x_lb; x <= x_ub; ++x) {
      for (auto y = y_lb; y <= y_ub; ++y) {
        auto x_fov = x - current_x + buf;
        auto y_fov = y - current_y + buf;
        auto idx = x_fov * fov_size + y_fov;

        // occupancy
        fov[0][x_fov][y_fov] = 1 - *occupancy_map.data(x, y);
        // cost_to_go
        auto _c = *cost_to_go.data(x, y);
        fov[1][x_fov][y_fov] = (_c >= 0 && (c < 0 || _c < c));
      }
    }

    py::array_t<int> numpy_fov = py::cast(fov);
    return numpy_fov;


  } else {

    const auto fov_size_pow2 = fov_size*fov_size;
    std::vector<uint16_t> fov(fov_size_pow2 * 2, 0);

    for (auto x = x_lb; x <= x_ub; ++x) {
      for (auto y = y_lb; y <= y_ub; ++y) {
        auto x_fov = x - current_x + buf;
        auto y_fov = y - current_y + buf;
        auto idx = x_fov * fov_size + y_fov;

        // occupancy
        fov[idx] = 1 - *occupancy_map.data(x, y);
        // cost_to_go
        auto _c = *cost_to_go.data(x, y);
        fov[idx + fov_size_pow2] = (_c >= 0 && (c < 0 || _c < c));
      }
    }

    py::array_t<int> numpy_fov = py::cast(fov);
    return numpy_fov;
  }
}

PYBIND11_MODULE(cost_to_go_wrapper, m)
{
  m.doc() = "cost_to_go_wrapper";
  m.def("getCostToGo2D",
        &getCostToGo2D,
        "creating cost_to_go",
        py::return_value_policy::move);
  m.def("getFov2dOccupancyCost",
        &getFov2dOccupancyCost,
        "get fov",
        py::return_value_policy::move);
}
