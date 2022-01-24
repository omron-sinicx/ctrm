/*
 * python wrapper for SPARS in OMPL
 * Author: Keisuke Okumura
 * Affiliation: TokyoTech & OSX
 *
 * ref
 * - Dobson, A., Krontiris, A., & Bekris, K. E. (2013).
 *   Sparse roadmap spanners.
 *   In Algorithmic Foundations of Robotics X (pp. 279-296). Springer, Berlin, Heidelberg.
 *
 * - Sucan, I. A., Moll, M., & Kavraki, L. E. (2012).
 *   The open motion planning library.
 *   IEEE Robotics & Automation Magazine, 19(4), 72-82.
 *
 * - https://ompl.kavrakilab.org/classompl_1_1geometric_1_1SPARS.html
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/geometric/planners/prm/SPARS.h>
#include <fcl/shape/geometric_shapes.h>
#include <fcl/continuous_collision.h>
#include <chrono>
#include <cmath>

namespace ob = ompl::base;
namespace og = ompl::geometric;
namespace py = pybind11;

using Time = std::chrono::system_clock;

static int cnt_continuous_collide = 0;
static double elapsed_continuous_collide = 0;
static int cnt_static_collide = 0;
static double elapsed_static_collide = 0;


class MyMotionValidator : public ob::MotionValidator
{
private:
  double rad;
  double max_speed;
  std::shared_ptr<fcl::Sphere> agent_body;
  std::vector<fcl::CollisionObject*> objs;

public:
  MyMotionValidator
  (ob::SpaceInformationPtr si, double _rad, double _speed,
   const std::vector<py::dict> &obs)
    : MotionValidator(si),
      rad(_rad),
      max_speed(_speed),
      agent_body(std::make_shared<fcl::Sphere>(rad))
  {
    fcl::Matrix3f rot;
    rot.setIdentity();

    for (auto o : obs) {
      // o is dictionary
      auto type = std::string(py::str(o["type"]));
      if (type == "sphere") {
        auto _r = (double)py::float_(o["rad"]);
        auto _x = (double)py::float_(o["x"]);
        auto _y = (double)py::float_(o["y"]);
        auto shape = std::make_shared<fcl::Sphere>(_r);
        fcl::Transform3f tf(rot, fcl::Vec3f(_x, _y, 0));
        objs.push_back(new fcl::CollisionObject(shape, tf));
      } else if (type == "box") {
        auto size_x = (double)py::float_(o["size_x"]);
        auto size_y = (double)py::float_(o["size_y"]);
        auto _x = (double)py::float_(o["x"]);
        auto _y = (double)py::float_(o["y"]);
        auto shape = std::make_shared<fcl::Box>(size_x, size_y, 0);
        fcl::Transform3f tf(rot, fcl::Vec3f(_x, _y, 0));
        objs.push_back(new fcl::CollisionObject(shape, tf));
      }
    }
  }

  ~MyMotionValidator () { for (auto o : objs) delete o; }

  bool checkMotion(const ob::State *s1, const ob::State *s2) const
  {
    auto s1_2d = s1->as<ob::SE2StateSpace::StateType>();
    auto x1 = s1_2d->getX();
    auto y1 = s1_2d->getY();

    auto s2_2d = s2->as<ob::SE2StateSpace::StateType>();
    auto x2 = s2_2d->getX();
    auto y2 = s2_2d->getY();

    // check distance
    if (std::pow(x1-x2, 2) + std::pow(y1-y2, 2) > std::pow(max_speed, 2))
        return false;

    ++cnt_continuous_collide;
    auto start = Time::now();
    auto res = !collide(x1, y1, x2, y2);
    elapsed_continuous_collide += std::chrono::duration_cast<std::chrono::milliseconds>
      (Time::now()-start).count();

    return res;
  }

  bool collide(double x1, double y1, double x2, double y2) const
  {
    fcl::Transform3f agent_tf_s(fcl::Vec3f(x1, y1, 0));
    fcl::Transform3f agent_tf_f(fcl::Vec3f(x2, y2, 0));

    fcl::CollisionObject agent_obj(agent_body, agent_tf_s);

    for (auto o : objs) {
      fcl::ContinuousCollisionRequest req;
      fcl::ContinuousCollisionResult res;
      fcl::continuousCollide(o, o->getTransform(),
                             &agent_obj, agent_tf_f,
                             req, res);
      if (res.is_collide) return true;
    }
    return false;
  }

  bool checkMotion(const ob::State* s1, const ob::State* s2,
                   std::pair<ob::State*, double>& lastValid) const
  {
    return checkMotion(s1, s2);
  }
};


// <x, y, neighbor indexes>
using Roadmap2d = std::vector<std::tuple<double, double, std::vector<unsigned int>>>;
std::tuple<Roadmap2d, int, double, int, double> getSparsRoadmap2d
(const double rad,
 const double speed,
 const double sparse_delta_fraction,
 const double dense_delta_fraction,
 const double stretch_factor,
 const int max_sample_num,
 const double time_limit,
 const std::vector<py::dict> &obs,
 const double lower_bound,
 const double upper_bound)
{
  // define space
  auto space(std::make_shared<ob::SE2StateSpace>());
  ob::RealVectorBounds bounds(2);
  bounds.setLow(lower_bound);
  bounds.setHigh(upper_bound);
  space->setBounds(bounds);

  // create state valid function
  auto rad2 = std::pow(rad, 2);
  auto isStateValid = [&](const ob::State *state) -> bool
  {
    ++cnt_static_collide;
    auto start = Time::now();

    auto state_2d = state->as<ob::SE2StateSpace::StateType>();
    auto x = state_2d->getX();
    auto y = state_2d->getY();

    // check with static object, without fcl
    for (auto o : obs) {
      // o is dictionary
      auto type = std::string(py::str(o["type"]));
      if (type == "sphere") {
        auto _r = (double)py::float_(o["rad"]);
        auto _x = (double)py::float_(o["x"]);
        auto _y = (double)py::float_(o["y"]);
        if (std::pow((x - _x), 2) + std::pow((y - _y), 2) < std::pow(rad+_r, 2)) {
          elapsed_static_collide += std::chrono::duration_cast<std::chrono::milliseconds>
            (Time::now()-start).count();
          return false;
        }
      } else if (type == "box") {
        auto size_x = (double)py::float_(o["size_x"]);
        auto size_y = (double)py::float_(o["size_y"]);
        auto _x = (double)py::float_(o["x"]);
        auto _y = (double)py::float_(o["y"]);

        auto x1 = _x - size_x / 2 - rad;
        auto x2 = _x + size_x / 2 + rad;
        auto y1 = _y - size_y / 2 - rad;
        auto y2 = _y + size_y / 2 + rad;

        if (x < x1 || x2 < x || y < y1 || y2 < y) continue;

        auto x3 = _x - size_x / 2;
        auto x4 = _x + size_x / 2;
        auto y3 = _y - size_y / 2;
        auto y4 = _y + size_y / 2;

        if (_x <= x3 && _y <= y3) {  // left upper
          if (std::pow(_x - x3, 2) + std::pow(_y - y3, 2) > rad2) continue;
        } else if (_x <= x3 && _y >= y4) {  // left lower
          if (std::pow(_x - x3, 2) + std::pow(_y - y4, 2) > rad2) continue;
        } else if (_x >= x4 && _y <= y3) {  // right upper
          if (std::pow(_x - x4, 2) + std::pow(_y - y3, 2) > rad2) continue;
        } else if (_x >= x4 && _y >= y4) {  // right lower
          if (std::pow(_x - x4, 2) + std::pow(_y - y4, 2) > rad2) continue;
        }

        // collision
        elapsed_static_collide += std::chrono::duration_cast<std::chrono::milliseconds>
          (Time::now()-start).count();
        return false;
      }
    }

    elapsed_static_collide += std::chrono::duration_cast<std::chrono::milliseconds>
      (Time::now()-start).count();
    return true;
  };

  // define space info
  auto si(std::make_shared<ob::SpaceInformation>(space));
  si->setStateValidityChecker(isStateValid);
  si->setMotionValidator(std::make_shared<MyMotionValidator>(si, rad, speed, obs));

  // define problem
  auto pdef(std::make_shared<ob::ProblemDefinition>(si));

  // define planner
  auto planner = std::make_shared<og::SPARS>(si);
  planner->setProblemDefinition(pdef);
  planner->setup();

  // set speed
  planner->setSparseDeltaFraction(sparse_delta_fraction);
  planner->setDenseDeltaFraction(dense_delta_fraction);
  planner->setStretchFactor(stretch_factor);

  // create roadmap
  auto ptc_tl = ob::timedPlannerTerminationCondition(time_limit);
  auto ptc = ob::PlannerTerminationCondition
    ([&] { return ptc_tl || planner->milestoneCount() > max_sample_num;  });
  planner->constructRoadmap(ptc);

  // get result
  ob::PlannerData pdat(si);
  planner->getPlannerData(pdat);

  // convert result
  Roadmap2d roadmap;
  auto motion_validator = si->getMotionValidator();
  for (int i = 0; i < pdat.numVertices(); ++i) {
    auto v = pdat.getVertex(i).getState()->as<ob::SE2StateSpace::StateType>();
    auto x = v->getX();
    auto y = v->getY();
    std::vector<unsigned int> _edges;
    pdat.getEdges(i, _edges);
    std::vector<unsigned int> edges;
    for (auto j : _edges) {
      if (motion_validator->checkMotion(pdat.getVertex(i).getState(),
                                        pdat.getVertex(j).getState())) {
        edges.push_back(j);
      }
    }
    roadmap.push_back(std::make_tuple(x, y, edges));
  }

  return std::make_tuple
    (roadmap,
     cnt_continuous_collide,
     elapsed_continuous_collide,
     cnt_static_collide,
     elapsed_static_collide);
}

PYBIND11_MODULE(spars_wrapper, m)
{
  m.doc() = "spars_wrapper";
  m.def("getSparsRoadmap2d",
        &getSparsRoadmap2d,
        "create 2d roadmap",
        py::return_value_policy::move);
}
