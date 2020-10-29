#include <catch2/catch.hpp>

#include <openvdb/points/PointSample.h>
#include <openvdb/points/PointScatter.h>
#include <tbb/atomic.h>
#include <vdb/velocity_grid.h>

using namespace openvdb;

TEST_CASE("VelocityGrid", "[velocity_grid]") {
  initialize();
  SECTION("Default Constructor") { vdb::VelocityGrid v; }
  SECTION("resize") {
    vdb::VelocityGrid v;
    v.resize(math::Coord(10, 10, 10));
    auto box = v.cbbox();
    REQUIRE(math::isExactlyEqual(
        box, CoordBBox(math::Coord(0, 0, 0), math::Coord(10, 10, 10))));
    auto bbox = v.grid()->evalActiveVoxelBoundingBox();
    REQUIRE(math::isExactlyEqual(box, bbox));
    REQUIRE(11 * 11 * 11 == v.grid()->activeVoxelCount());
  }
  SECTION("foreach_s") {
    vdb::VelocityGrid v;
    v.setSpacing(0.2);
    v.resize({3, 3, 3});
    int count = 0;
    v.foreach_s([&](math::Coord c, Vec3f &value) {
      value = Vec3f(c.x(), c.y(), c.z());
      count++;
    });
    REQUIRE(count == 4 * 4 * 4);
    v.foreach_s([](math::Coord c, Vec3f &value) {
      REQUIRE(math::isExactlyEqual(value, Vec3f(c.x(), c.y(), c.z())));
    });
  }
  SECTION("foreach") {
    vdb::VelocityGrid v;
    v.setSpacing(0.2);
    v.resize({3, 3, 3});
    v.foreach ([&](math::Coord c, Vec3f &value) {
      value = Vec3f(c.x(), c.y(), c.z());
    });
    v.foreach ([](math::Coord c, Vec3f &value) {
      REQUIRE(math::isExactlyEqual(value, Vec3f(c.x(), c.y(), c.z())));
    });
  }

  SECTION("Sample enright velocity field") {
    return;
    math::Coord resolution(128, 128, 128);
    float spacing = 1. / resolution.x();
    float half_spacing = spacing * 0.5;
    vdb::VelocityGrid v;
    v.resize(resolution);
    REQUIRE(v.grid()->activeVoxelCount() == 129 * 129 * 129);
    v.setSpacing(spacing);
    auto enright_component = [](int i, const Vec3d &p) -> float {
      float pix = 3.14159265358979323 * p.x();
      float piy = 3.14159265358979323 * p.y();
      float piz = 3.14159265358979323 * p.z();
      if (i == 0)
        return 2. * sinf(pix) * sinf(pix) * sinf(2 * piy) * sinf(2 * piz);
      if (i == 1)
        return -sinf(2 * pix) * sinf(piy) * sinf(piy) * sinf(2 * piz);
      return -sinf(2 * pix) * sinf(2 * piy) * sinf(piz) * sinf(piz);
    };
    // apply enright field
    tbb::atomic<int> count = 0;
    v.foreach ([&](math::Coord c, Vec3f &value) {
      auto wp = v.grid()->transform().indexToWorld(c);
      value.x() = enright_component(0, wp + Vec3d(half_spacing, 0, 0));
      value.y() = enright_component(1, wp + Vec3d(0, half_spacing, 0));
      value.z() = enright_component(2, wp + Vec3d(0, 0, half_spacing));
      count.fetch_and_add(1);
    });
    REQUIRE(count == 129 * 129 * 129);
    // create points (scatter points over grid)
    auto points = points::uniformPointScatter(*(v.grid()), 1000000);
    points->setTransform(math::Transform::createLinearTransform(spacing));
    // add point attribute
    points::appendAttribute<Vec3f>(points->tree(), "V");
    std::cerr << "interpolating ... ";
    // interpolate field
    points::boxSample(*points, *(v.grid()), "V");
    std::cerr << " done!\n";
    // check interpolation values
    for (auto leaf = points->tree().beginLeaf(); leaf; ++leaf) {
      points::AttributeHandle<Vec3f> p_handle(leaf->attributeArray("P"));
      points::AttributeHandle<Vec3f> v_handle(leaf->attributeArray("V"));
      for (auto index = leaf->beginIndexOn(); index; ++index) {
        Coord xyz;
        index.getCoord(xyz);
        auto wp = points->transform().indexToWorld(xyz + p_handle.get(*index));
        Vec3f ans(enright_component(0, wp), enright_component(1, wp),
                  enright_component(2, wp));
        auto expected = tools::StaggeredBoxSampler::sample(
            v.grid()->getAccessor(), xyz + p_handle.get(*index));
        if (!math::isApproxEqual(expected, v_handle.get(*index)))
          std::cerr << xyz << ": " << points->transform().indexToWorld(xyz)
                    << " + " << p_handle.get(*index) << " = " << wp << " I"
                    << v_handle.get(*index) << " " << ans << "ex " << expected
                    << std::endl;
        REQUIRE(math::isApproxEqual(expected, v_handle.get(*index)));
        // points->transform().indexToWorld(index.getCoord());
        // std::cerr << handle.get(*index) << " ";
      }
    }
  }
}