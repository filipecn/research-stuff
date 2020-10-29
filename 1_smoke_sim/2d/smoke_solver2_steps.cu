/// Copyright (c) 2020, FilipeCN.
///
/// The MIT License (MIT)
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.
///
///\file kernels2.cu
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-27
///
///\brief

#include "blas/pcg2.h"
#include "smoke_solver2_steps.h"
#include <defs.h>

using namespace hermes::cuda;
/*
texture<float, cudaTextureType2D> pressureTex2;
texture<unsigned char, cudaTextureType2D> solidTex2;
*/

__global__ void __applyForceField(Grid2Accessor<f32> velocity,
                                  Array2Accessor<u8> solid,
                                  Grid2Accessor<f32> force, f32 dt, index2 d) {
  index2 index(blockIdx.x * blockDim.x + threadIdx.x,
               blockIdx.y * blockDim.y + threadIdx.y);
  if (velocity.stores(index) && !solid[index] && !solid[index.plus(-d.i, -d.j)])
    velocity[index] +=
        dt * (force[index.plus(-d.i, -d.j)] + force[index]) * 0.5f;
}

void applyForceField(VectorGrid2<f32> &velocity, Array2<u8> &solid,
                     VectorGrid2<f32> &force_field, f32 dt) {
  {
    ThreadArrayDistributionInfo td(velocity.u().resolution());
    __applyForceField<<<td.gridSize, td.blockSize>>>(
        velocity.u().accessor(), solid.accessor(), force_field.u().accessor(),
        dt, index2(1, 0));
  }
  {
    ThreadArrayDistributionInfo td(velocity.v().resolution());
    __applyForceField<<<td.gridSize, td.blockSize>>>(
        velocity.v().accessor(), solid.accessor(), force_field.v().accessor(),
        dt, index2(0, 1));
  }
}

__global__ void __addBuoyancyForce(VectorGrid2Accessor<f32> f,
                                   Array2Accessor<u8> solid,
                                   Grid2Accessor<f32> density,
                                   Grid2Accessor<f32> temperature, f32 tamb,
                                   f32 alpha, f32 beta) {
  index2 index(blockIdx.x * blockDim.x + threadIdx.x,
               blockIdx.y * blockDim.y + threadIdx.y);
  if (f.v().stores(index) && !solid[index])
    f.v()[index] =
        9.81 * (-alpha * density[index] + beta * (temperature[index] - tamb));
}

void addBuoyancyForce(VectorGrid2<f32> &force_field, Array2<u8> &solid,
                      Grid2<f32> &density, Grid2<f32> &temperature,
                      f32 ambient_temperature, f32 alpha, f32 beta) {
  ThreadArrayDistributionInfo td(force_field.resolution());
  __addBuoyancyForce<<<td.gridSize, td.blockSize>>>(
      force_field.accessor(), solid.accessor(), density.accessor(),
      temperature.accessor(), ambient_temperature, alpha, beta);
}

__global__ void __computeDivergence(VectorGrid2Accessor<f32> vel,
                                    Array2Accessor<u8> solid,
                                    VectorGrid2Accessor<f32> svel,
                                    Grid2Accessor<f32> divergence, vec2 invdx) {
  index2 index(blockIdx.x * blockDim.x + threadIdx.x,
               blockIdx.y * blockDim.y + threadIdx.y);
  if (divergence.stores(index)) {
    if (solid[index]) {
      divergence[index] = 0;
      return;
    }
    f32 left = vel.u()[index];
    f32 right = vel.u()[index.right()];
    f32 bottom = vel.v()[index];
    f32 top = vel.v()[index.up()];
    if (solid.contains(index.left()) && solid[index.left()])
      left = svel.u()[index];
    if (solid.contains(index.right()) && solid[index.right()])
      right = svel.u()[index.right()];
    if (solid.contains(index.down()) && solid[index.down()])
      bottom = svel.v()[index];
    if (solid.contains(index.up()) && solid[index.up()])
      top = svel.v()[index.up()];
    divergence[index] = dot(invdx, vec2(right - left, top - bottom));
  }
}

void computeDivergence(VectorGrid2<f32> &velocity, Array2<u8> &solid,
                       VectorGrid2<f32> &solidVelocity,
                       Grid2<f32> &divergence) {
  auto info = divergence.info();
  vec2 inv(-1.f / divergence.spacing().x);
  hermes::cuda::ThreadArrayDistributionInfo td(divergence.resolution());
  __computeDivergence<<<td.gridSize, td.blockSize>>>(
      velocity.accessor(), solid.accessor(), solidVelocity.accessor(),
      divergence.accessor(), inv);
}

__global__ void __fillPressureMatrix(FDMatrix2Accessor<f32> A,
                                     Array2Accessor<u8> solid, f32 scale) {
  index2 ij(blockIdx.x * blockDim.x + threadIdx.x,
            blockIdx.y * blockDim.y + threadIdx.y);
  if (A.stores(ij, ij)) {
    if (solid[ij])
      return;
    A(ij, ij) = 0.f;
    A(ij, ij.right()) = 0.f;
    A(ij, ij.up()) = 0.f;
    // left - right
    if (solid.contains(ij.left())) {
      if (!solid[ij.left()])
        A(ij, ij) += scale;
    } else // consider outside domain fluid
      A(ij, ij) += scale;
    if (solid.contains(ij.right())) {
      if (!solid[ij.right()]) {
        A(ij, ij) += scale;
        A(ij, ij.right()) = -scale;
      } // else // EMPTY
      //   A(i, j).diag += scale;
    } else
      A(ij, ij) += scale;
    // bottom - top
    if (solid.contains(ij.down())) {
      if (!solid[ij.down()])
        A(ij, ij) += scale;
    } else // consider outside domain fluid
      A(ij, ij) += scale;
    if (solid.contains(ij.up())) {
      if (!solid[ij.up()]) {
        A(ij, ij) += scale;
        A(ij, ij.up()) = -scale;
      } // else // EMPTY
      //   A(i, j).diag += scale;
    } else
      A(ij, ij) += scale;
  }
}

__global__ void __buildRHS(Array2Accessor<i32> indices,
                           Grid2Accessor<f32> divergence,
                           Array1Accessor<f32> rhs) {
  index2 ij(blockIdx.x * blockDim.x + threadIdx.x,
            blockIdx.y * blockDim.y + threadIdx.y);
  if (indices.contains(ij) && indices[ij] >= 0) {
    // TODO: fix for solid neighbors
    rhs[indices[ij]] = divergence[ij];
  }
}

__global__ void __1To2(Array2Accessor<i32> indices, Array1Accessor<f32> v,
                       Array2Accessor<f32> m) {
  index2 ij(blockIdx.x * blockDim.x + threadIdx.x,
            blockIdx.y * blockDim.y + threadIdx.y);
  if (indices.contains(ij) && indices[ij] >= 0)
    m[ij] = v[indices[ij]];
}

size_t setupPressureSystem(Grid2<f32> &divergence, Array2<u8> &solid,
                           FDMatrix2<f32> &pressure_matrix, f32 dt,
                           Vector<f32> &rhs) {
  // fill matrix
  f32 scale = dt / (divergence.spacing().x * divergence.spacing().x);
  ThreadArrayDistributionInfo td(divergence.resolution());
  __fillPressureMatrix<<<td.gridSize, td.blockSize>>>(
      pressure_matrix.accessor(), solid.accessor(), scale);
  // compute indices
  ponos::Array2<i32> h_indices(divergence.resolution().ponos());
  auto h_solid = solid.hostData();
  int curIndex = 0;
  for (auto e : h_indices)
    if (!h_solid[e.index]) {
      e.value = curIndex++;
    } else
      e.value = -1;
  pressure_matrix.indexData() = h_indices;
  // rhs
  rhs.resize(curIndex);
  __buildRHS<<<td.gridSize, td.blockSize>>>(
      pressure_matrix.indexData().accessor(), divergence.accessor(),
      rhs.data().accessor());
  return curIndex;
}

void solvePressureSystem(FDMatrix2<f32> &A, Grid2<f32> &divergence,
                         Grid2<f32> &pressure, Array2<u8> &solid, f32 dt) {
  pressure = 0;
  // setup system
  Vector<f32> rhs;
  PROFILE(setupPressureSystem(divergence, solid, A, dt, rhs));
  // apply incomplete Cholesky preconditioner
  // solve system
  Vector<f32> x(rhs.size(), 0.f);
  std::cerr << "solve " << rhs.size() << std::endl;
  int it = 0;
  PROFILE(it = pcg(x, A, rhs, rhs.size(), 1e-6));
  std::cerr << it << " iterations\n";
  auto residual = BLAS::infNorm(rhs - A * x);
  std::cerr << residual << "\n";
  hermes::cuda::ThreadArrayDistributionInfo td(pressure.resolution());
  __1To2<<<td.gridSize, td.blockSize>>>(A.indexData().accessor(),
                                        x.data().accessor(),
                                        pressure.data().accessor());
}

__global__ void __projectionStep(Grid2Accessor<f32> vel,
                                 Grid2Accessor<f32> pressure,
                                 Array2Accessor<u8> solid, index2 d,
                                 float scale) {
  index2 ij(blockIdx.x * blockDim.x + threadIdx.x,
            blockIdx.y * blockDim.y + threadIdx.y);
  if (vel.stores(ij)) {
    if (solid[ij - d])
      vel[ij] = 0; // TODO: must receive solid velocity
    else if (solid[ij])
      vel[ij] = 0;
    else {
      f32 b = pressure[ij - d];
      f32 f = pressure[ij];
      vel[ij] -= scale * (f - b);
    }
  }
}

void projectionStep(Grid2<f32> &pressure, Array2<u8> &solid,
                    VectorGrid2<f32> &velocity, f32 dt) {
  {
    float invdx = 1.0 / velocity.u().spacing().x;
    float scale = dt * invdx;
    ThreadArrayDistributionInfo td(velocity.u().resolution());
    __projectionStep<<<td.gridSize, td.blockSize>>>(
        velocity.u().accessor(), pressure.accessor(), solid.accessor(),
        index2(1, 0), scale);
  }
  {
    float invdx = 1.0 / velocity.v().spacing().y;
    float scale = dt * invdx;
    ThreadArrayDistributionInfo td(velocity.v().resolution());
    __projectionStep<<<td.gridSize, td.blockSize>>>(
        velocity.v().accessor(), pressure.accessor(), solid.accessor(),
        index2(0, 1), scale);
  }
}