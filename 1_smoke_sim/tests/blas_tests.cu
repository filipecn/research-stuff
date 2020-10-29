#include <catch2/catch.hpp>

#include <2d/blas/pcg2.h>
#include <hermes/hermes.h>

using namespace hermes::cuda;

TEST_CASE("PCG2", "[smoke_sim][blas]") {
  // 3 4 5 0  x     -1
  // 4 3 0 5  x  =  -1
  // 5 0 3 4  x     -1
  // 0 5 4 3  x     -1
  ponos::FDMatrix2<f32> A(ponos::size2(2));
  int index = 0;
  for (ponos::index2 ij : ponos::Index2Range<i32>(A.gridSize()))
    A.indexData()[ij] = index++;
  for (ponos::index2 ij : ponos::Index2Range<i32>(A.gridSize())) {
    A(ij, ij) = 3;
    A(ij, ij.right()) = 4;
    A(ij, ij.up()) = 5;
  }
  FDMatrix2<f32> d_A = A;
  Vector<f32> d_x(4, 0);
  Vector<f32> d_rhs(4, -1);
  pcg(d_x, d_A, d_rhs, 100, 1e-6);
  auto r = d_A * d_x - d_rhs;
  REQUIRE(BLAS::infNorm(r) < 1e-6);
}