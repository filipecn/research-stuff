#include <catch2/catch.hpp>

#include <2d/smoke_solver2_steps.h>
#include <hermes/hermes.h>

using namespace hermes::cuda;

TEST_CASE("applyForce", "[simulation]") {
  SECTION("2d") {
    size2 size(1000);
    VectorGrid2<f32> velocity(ponos::VectorGridType::STAGGERED);
    velocity.u() = 0;
    velocity.v() = 0;
    VectorGrid2<f32> force;
    force.u() = -1;
    force.v() = 1;
    force.setResolution(size);
    ponos::Array2<u8> h_solid(size.ponos());
    for (auto e : h_solid)
      if (e.index.i == 0 || e.index.j == 0)
        e.value = 1;
      else
        e.value = 0;
    Array2<u8> solid = h_solid;
    applyForceField(velocity, solid, force, 0.1);
  }
}

TEST_CASE("buoyancyForce", "[simulation]") {
  SECTION("2d") {
    size2 size(10);
    VectorGrid2<f32> force_field;
    force_field.setResolution(size);
    force_field.u() = 0;
    force_field.v() = 1;
    Array2<u8> solid(size);
    solid = 0;
    Grid2<f32> density(size);
    density = 1;
    Grid2<f32> temperature(size);
    temperature = 400;
    f32 ambient_temperature = 273;
    f32 alpha = 1;
    f32 beta = 0;
    addBuoyancyForce(force_field, solid, density, temperature,
                     ambient_temperature, alpha, beta);
    ponos::VectorGrid2<f32> h_force = force_field.hostData();
    for (auto e : h_force.u().data())
      REQUIRE(e.value == Approx(0).margin(1e-8));
    for (auto e : h_force.v().data())
      REQUIRE(e.value ==
              Approx(9.81 * (-alpha * 1 + beta * (400 - 273))).margin(1e-8));
  }
}

TEST_CASE("divergence", "[simulation]") {
  SECTION("2d") {
    // y
    // |
    //  ---x
    ponos::Array2<u8> h_solid = {
        {1, 1, 0, 1, 1}, //  S S F S S  -  -  0  -  -
        {1, 0, 0, 0, 1}, //  S F F F S  -  1  2  3  -
        {0, 0, 0, 0, 0}, //  F F F F F  4  5  6  7  8
        {0, 0, 0, 0, 0}, //  F F F F F  9 10 11 12 13
        {1, 0, 0, 0, 1}  //  S F F F S  - 14 15 16  -
    };
    // velocity field
    //              (j+1)*10 + i
    //  i*10 + j       i,j         (i+1)*10 + j
    //              j*10 + i
    ponos::size2 size(5);
    ponos::VectorGrid2<f32> velocity(ponos::VectorGridType::STAGGERED);
    velocity.setResolution(size);
    for (auto e : velocity.u().accessor())
      e.value = e.index.i * 10 + e.index.j;
    for (auto e : velocity.v().accessor())
      e.value = e.index.j * 10 + e.index.i;
    Array2<u8> d_solid = h_solid;
    VectorGrid2<f32> d_velocity = velocity;
    VectorGrid2<f32> d_solid_vel(ponos::VectorGridType::STAGGERED);
    d_solid_vel.setResolution(size2(size));
    d_solid_vel = 0;
    Grid2<f32> d_div(size2(size), vec2(0.01f));
    d_div.setSpacing(vec2(0.01));
    computeDivergence(d_velocity, d_solid, d_solid_vel, d_div);
    ponos::Grid2<f32> h_div = d_div.hostData();
    for (auto e : h_div.data())
      if (h_solid[e.index])
        REQUIRE(e.value == Approx(0).margin(1e-8));
      else {
        f32 value = 20;
        if (h_solid.stores(e.index.up()) && h_solid[e.index.up()])
          value -= (e.index.j + 1) * 10 + e.index.i;
        if (h_solid.stores(e.index.down()) && h_solid[e.index.down()])
          value += e.index.j * 10 + e.index.i;
        if (h_solid.stores(e.index.right()) && h_solid[e.index.right()])
          value -= (e.index.i + 1) * 10 + e.index.j;
        if (h_solid.stores(e.index.left()) && h_solid[e.index.left()])
          value += e.index.i * 10 + e.index.j;
        REQUIRE(e.value == Approx(-100 * value).margin(1e-8));
      }
  }
}