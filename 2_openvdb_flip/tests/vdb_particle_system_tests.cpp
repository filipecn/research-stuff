#include <catch2/catch.hpp>

#include <vdb/particle_system.h>

using namespace vdb;

TEST_CASE("ParticleSystem", "[particle_system]") {
  openvdb::initialize();
  SECTION("Default Constructor") {
    ParticleSystem ps;
    REQUIRE(ps.size() == 0);
  }
  SECTION("Init list P Constructor") {
    ParticleSystem ps = {{0.f, 0.f, 0.f}};
    REQUIRE(ps.size() == 1);
    std::cerr << ps << std::endl;
  }
  SECTION("Init list P V Constructor") {
    ParticleSystem ps = {{0.f, 0.f, 0.f, 1.f, 0.f, 0.f},
                         {0.f, 1.f, 0.f, 1.f, 1.f, 0.f},
                         {0.f, 0.f, 1.f, 1.f, 0.f, 0.f}};
    REQUIRE(ps.size() == 3);
    std::cerr << ps << std::endl;
  }
}
