/*
 * fast collision checker between two spheres moving in constant velocities
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

bool continuousCollideSpheres
(
 const py::array_t<double>& from1,
 const py::array_t<double>& to1,
 const double& rad1,
 const py::array_t<double>& from2,
 const py::array_t<double>& to2,
 const double& rad2
)
{
  // the following can be derived from the simple calculus

  const auto dim = from1.request().shape[0];

  double a = 0;
  double b = 0;
  for (int i = 0; i < dim; ++i) {
    auto val = - (*from1.data(i)) + (*to1.data(i)) + (*from2.data(i)) - (*to2.data(i));
    a += val*val;
    b += val*((*from1.data(i)) - (*from2.data(i)));
  }

  auto distPow2 = [&] (const double t) {
    double total = 0;
    for (int i = 0; i < dim; ++i) {
      auto val = ((1-t) * (*from1.data(i)) + t * (*to1.data(i))) - ((1-t) * (*from2.data(i)) + t * (*to2.data(i)));
      total += val*val;
    }
    return total;
  };

  double min_dist_pow2 = 0;
  if (b >= 0) {
    min_dist_pow2 = distPow2(0);
  } else if (a + b <= 0) {
    min_dist_pow2 = distPow2(1);
  } else {
    min_dist_pow2 = distPow2(-b/a);
  }

  const auto rad = rad1 + rad2;
  return min_dist_pow2 <= rad*rad;
}

PYBIND11_MODULE(sphere_collision_check_wrapper, m)
{
  m.doc() = "sphere_collision_check_wrapper";
  m.def("continuousCollideSpheres",
        &continuousCollideSpheres,
        "check collision of two moving spheres",
        py::return_value_policy::move);
}
