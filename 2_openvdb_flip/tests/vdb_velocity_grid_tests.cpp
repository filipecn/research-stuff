#include <catch2/catch.hpp>

#include <openvdb/points/PointSample.h>
#include <vdb/velocity_grid.h>

using namespace vdb;

TEST_CASE("OpenVDB", "[openvdb]") {
  openvdb::initialize();
  SECTION("check staggered grid") {
    openvdb::Vec3s velocity_background(1.0, 2.0, 3.0);
    openvdb::VectorGrid::Ptr grid =
        openvdb::VectorGrid::create(velocity_background);
    grid->setGridClass(openvdb::GRID_STAGGERED);
    auto acc = grid->getAccessor();
    for (int i = 0; i < 10; ++i)
      for (int j = 0; j < 10; ++j)
        for (int k = 0; k < 10; ++k)
          acc.setValue(openvdb::Coord(i, j, k), openvdb::Vec3f(i, j, k));
    for (int i = 1; i < 9; ++i)
      for (int j = 1; j < 9; ++j)
        for (int k = 1; k < 9; ++k) {
          openvdb::Vec3f expected(i + 0.5, j + 0.5, k + 0.5);
          auto sample = openvdb::tools::StaggeredBoxSampler::sample(
              acc, openvdb::Vec3f(i, j, k));
          REQUIRE(openvdb::math::isApproxEqual(expected, sample));
        }
  }
}