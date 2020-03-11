#include <catch2/catch.hpp>

#include <openvdb/tools/LevelSetSphere.h>
#include <vdb/injector.h>
#include <vdb/io.h>

using namespace vdb;
using namespace openvdb;

TEST_CASE("Particle System IO", "[particle_system][io]") {
  initialize();
  SECTION("simple write") {
    ParticleSystem ps = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                         {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    IO::write(ps, "simple");
  }
  SECTION("Animation") {
    // Generate particles inside a sphere
    FloatGrid::Ptr sphere_grid =
        tools::createLevelSetSphere<FloatGrid>(.15f, Vec3f(.35, .35, .35), .01);
    ParticleSystem ps;
    FluidInjector::injectLevelSet(sphere_grid, ps);
    // Generate advect particles under a Enright velocity field and save frames
  }
}
