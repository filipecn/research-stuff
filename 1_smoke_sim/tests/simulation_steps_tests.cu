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
    // d_div.setSpacing(vec2(0.01));
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

TEST_CASE("pressure", "[simulation]") {
  SECTION("2d") {
    SECTION("setupPressureSystem") {
      ponos::size2 size(5);
      // y
      // |
      //  ---x
      ponos::Array2<u8> h_solid = {
          {1, 1, 0, 1, 1}, //  S S F S S  -  -  0  -  -
          {1, 0, 0, 0, 1}, //  S F F F S  -  1  2  3  -
          {0, 0, 0, 0, 0}, //  F F F F F  4  5  6  7  8
          {1, 0, 0, 0, 0}, //  S F F F F  -  9 10 11 12
          {1, 1, 1, 0, 1}  //  S S S F S  -  -  - 13  -
      };
      ponos::Array2<i32> h_indices_ans = {{-1, -1, 0, -1, -1},
                                          {-1, 1, 2, 3, -1},
                                          {4, 5, 6, 7, 8},
                                          {-1, 9, 10, 11, 12},
                                          {-1, -1, -1, 13, -1}};
      // make divergent value to be same as indices
      ponos::Array2<f32> div_data = {{-1, -1, 0, -1, -1},
                                     {-1, 1, 2, 3, -1},
                                     {4, 5, 6, 7, 8},
                                     {-1, 9, 10, 11, 12},
                                     {-1, -1, -1, 13, -1}};
      ponos::Grid2<f32> h_div(size, ponos::vec2(0.01));
      h_div = div_data;
      std::vector<ponos::index2> index_map(14, ponos::index2(-1));
      for (auto ij : ponos::Index2Range<i32>(size))
        if (h_indices_ans[ij] >= 0)
          index_map[h_indices_ans[ij]] = ij;
      Array2<u8> d_solid = h_solid;
      Grid2<f32> d_div(h_div);
      Vector<f32> d_rhs;
      size2 s(size);
      FDMatrix2<f32> d_A(s);
      setupPressureSystem(d_div, d_solid, d_A, 0.001, d_rhs);
      auto h_A = d_A.hostData();
      // check index data
      for (auto e : h_A.indexData())
        REQUIRE(e.value == h_indices_ans[e.index]);
      // check index map
      for (size_t i = 0; i < 14; ++i)
        REQUIRE(i == h_A.indexData()[index_map[i]]);
      // check system
      for (size_t r = 0; r < 14; ++r) {
        auto ij_r = index_map[r];
        for (size_t c = 0; c < 14; ++c) {
          auto ij_c = index_map[c];
          std::cerr << h_A(ij_r, ij_c) << " ";
        }
        std::cerr << std::endl;
      }
      // check rhs
      auto h_rhs = d_rhs.hostData();
      std::cerr << h_rhs << std::endl;
    }
    SECTION("setupPressureSystem - closed domain") {
      ponos::size2 size(5);
      // y
      // |
      //  ---x
      ponos::Array2<u8> h_solid = {
          {1, 1, 1, 1, 1}, //  S S S S S  -  -  -  -  -
          {1, 0, 0, 0, 1}, //  S F F F S  -  0  1  2  -
          {1, 0, 0, 0, 1}, //  S F F F S  -  3  4  5  -
          {1, 0, 0, 0, 1}, //  S F F F S  -  6  7  8  -
          {1, 1, 1, 1, 1}  //  S S S S S  -  -  -  -  -
      };
      ponos::Array2<i32> h_indices_ans = {{-1, -1, -1, -1, -1},
                                          {-1, 0, 1, 2, -1},
                                          {-1, 3, 4, 5, -1},
                                          {-1, 6, 7, 8, -1},
                                          {-1, -1, -1, -1, -1}};
      std::vector<ponos::index2> index_map(9, ponos::index2(-1));
      for (auto ij : ponos::Index2Range<i32>(size))
        if (h_indices_ans[ij] >= 0)
          index_map[h_indices_ans[ij]] = ij;
      Array2<u8> d_solid = h_solid;
      Grid2<f32> d_div(size2(size), vec2(0.01f));
      Vector<f32> d_rhs;
      size2 s(size);
      FDMatrix2<f32> d_A(s);
      setupPressureSystem(d_div, d_solid, d_A, 0.001, d_rhs);
      auto h_A = d_A.hostData();
      // check index data
      for (auto e : h_A.indexData())
        REQUIRE(e.value == h_indices_ans[e.index]);
      // check index map
      for (size_t i = 0; i < 9; ++i)
        REQUIRE(i == h_A.indexData()[index_map[i]]);
      // check system
      for (size_t r = 0; r < 9; ++r) {
        auto ij_r = index_map[r];
        for (size_t c = 0; c < 9; ++c) {
          auto ij_c = index_map[c];
          std::cerr << h_A(ij_r, ij_c) << " ";
        }
        std::cerr << std::endl;
      }
    }
    SECTION("solvePressureSystem") {
      ponos::size2 size(5);
      // y
      // |
      //  ---x
      ponos::Array2<u8> h_solid = {
          {1, 1, 1, 1, 1}, //  S S S S S  -  -  -  -  -
          {1, 0, 0, 0, 1}, //  S F F F S  -  0  1  2  -
          {1, 0, 0, 0, 1}, //  S F F F S  -  3  4  5  -
          {1, 0, 0, 0, 1}, //  S F F F S  -  6  7  8  -
          {1, 0, 0, 0, 1}  //  S F F F S  -  9 10 11  -
      };
      ponos::Array2<f32> div_data = {{0, 0, 0, 0, 0},
                                     {0, -10, -10, -10, 0},
                                     {0, 0, 0, 0, 0},
                                     {0, 0, 0, 0, 0},
                                     {0, 0, 0, 0, 0}};
      ponos::Grid2<f32> h_div(size, ponos::vec2(0.01f));
      h_div = div_data;
      Array2<u8> d_solid = h_solid;
      Grid2<f32> d_div;
      d_div = h_div;
      size2 s(size);
      FDMatrix2<f32> d_A(s);
      Grid2<f32> d_pressure(s, vec2(0.01));
      solvePressureSystem(d_A, d_div, d_pressure, d_solid, 0.001);
      std::cerr << d_pressure.data().hostData() << std::endl;
    }
  }
}