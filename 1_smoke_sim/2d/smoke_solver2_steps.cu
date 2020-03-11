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

using namespace hermes::cuda;
/*
texture<float, cudaTextureType2D> pressureTex2;
texture<unsigned char, cudaTextureType2D> solidTex2;
*/

__global__ void __applyForceField(Grid2Accessor<f32> velocity,
                                  Array2Accessor<u8> solid,
                                  Grid2Accessor<f32> force, f32 dt, vec2u d) {
  index2 index(blockIdx.x * blockDim.x + threadIdx.x,
               blockIdx.y * blockDim.y + threadIdx.y);
  if (velocity.stores(index) && !solid[index] && !solid[index.plus(-d.x, -d.y)])
    velocity[index] +=
        dt * (force[index.plus(-d.x, -d.y)] + force[index]) * 0.5f;
}

void applyForceField(VectorGrid2<f32> &velocity, Array2<u8> &solid,
                     VectorGrid2<f32> &force_field, f32 dt) {
  {
    ThreadArrayDistributionInfo td(velocity.u().resolution());
    __applyForceField<<<td.gridSize, td.blockSize>>>(
        velocity.u().accessor(), solid.accessor(), force_field.u().accessor(),
        dt, vec2u(1, 0));
  }
  {
    ThreadArrayDistributionInfo td(velocity.v().resolution());
    __applyForceField<<<td.gridSize, td.blockSize>>>(
        velocity.v().accessor(), solid.accessor(), force_field.v().accessor(),
        dt, vec2u(0, 1));
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
/*
__global__ void __fillPressureMatrix(MemoryBlock2Accessor<FDMatrix2Entry> A,
                                     RegularGrid2Accessor<unsigned char> solid,
                                     float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (A.isIndexValid(i, j)) {
    if (solid(i, j))
      return;
    A(i, j).diag = 0;
    A(i, j).x = 0;
    A(i, j).y = 0;
    // left - right
    if (solid.isIndexStored(i - 1, j) && !solid(i - 1, j))
      A(i, j).diag += scale;
    if (solid.isIndexStored(i + 1, j)) {
      if (!solid(i + 1, j)) {
        A(i, j).diag += scale;
        A(i, j).x = -scale;
      } // else // EMPTY
      //   A(i, j).diag += scale;
    } else
      A(i, j).diag += scale;
    // bottom - top
    if (solid.isIndexStored(i, j - 1) && !solid(i, j - 1))
      A(i, j).diag += scale;
    if (solid.isIndexStored(i, j + 1)) {
      if (!solid(i, j + 1)) {
        A(i, j).diag += scale;
        A(i, j).y = -scale;
      } // else // EMPTY
      //   A(i, j).diag += scale;
    } else
      A(i, j).diag += scale;
  }
}

__global__ void __buildRHS(MemoryBlock2Accessor<int> indices,
                           RegularGrid2Accessor<float> divergence,
                           CuMemoryBlock1Accessor<double> rhs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (indices.isIndexValid(i, j)) {
    if (indices(i, j) >= 0)
      rhs[indices(i, j)] = divergence(i, j);
  }
}

__global__ void __1To2(MemoryBlock2Accessor<int> indices,
                       CuMemoryBlock1Accessor<double> v,
                       MemoryBlock2Accessor<float> m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (indices.isIndexValid(i, j)) {
    if (indices(i, j) >= 0)
      m(i, j) = v[indices(i, j)];
  }
}

template <>
size_t setupPressureSystem(RegularGrid2Df &divergence, RegularGrid2Duc &solid,
                           FDMatrix2D &pressureMatrix, float dt,
                           CuMemoryBlock1d &rhs) {
  // fill matrix
  float scale = dt / (divergence.spacing().x * divergence.spacing().x);
  hermes::cuda::ThreadArrayDistributionInfo td(divergence.resolution());
  __fillPressureMatrix<<<td.gridSize, td.blockSize>>>(
      pressureMatrix.dataAccessor(), solid.accessor(), scale);
  // compute indices
  auto res = divergence.resolution();
  MemoryBlock2<MemoryLocation::HOST, int> h_indices(res);
  h_indices.allocate();
  MemoryBlock2<MemoryLocation::HOST, unsigned char> h_solid(res);
  h_solid.allocate();
  memcpy(h_solid, solid.data());
  auto solidAcc = h_solid.accessor();
  auto indicesAcc = h_indices.accessor();
  int curIndex = 0;
  for (size_t j = 0; j < res.y; j++)
    for (size_t i = 0; i < res.x; i++)
      if (!solidAcc(i, j)) {
        indicesAcc(i, j) = curIndex++;
      } else
        indicesAcc(i, j) = -1;
  memcpy(pressureMatrix.indexData(), h_indices);
  // rhs
  rhs.resize(curIndex);
  __buildRHS<<<td.gridSize, td.blockSize>>>(pressureMatrix.indexDataAccessor(),
                                            divergence.accessor(),
                                            rhs.accessor());
  return curIndex;
}

template <>
void solvePressureSystem(FDMatrix2D &A, RegularGrid2Df &divergence,
                         RegularGrid2Df &pressure, RegularGrid2Duc &solid,
                         float dt) {
  // setup system
  CuMemoryBlock1d rhs;
  setupPressureSystem(divergence, solid, A, dt, rhs);
  // apply incomplete Cholesky preconditioner
  // solve system
  CuMemoryBlock1d x = std::vector<double>(rhs.size(), 0.f);
  // FDMatrix3H H(A.gridSize());
  // H.copy(A);
  // auto acc = H.accessor();
  // std::cerr << acc << "rhs\n" << rhs << std::endl;
  std::cerr << "solve\n";
  pcg(x, A, rhs, rhs.size(), 1e-12);
  // std::cerr << residual << "\n" << x << std::endl;
  // store pressure values
  CuMemoryBlock1d sol = std::vector<double>(rhs.size(), 0);
  mul(A, x, sol);
  sub(sol, rhs, sol);
  if (infnorm(sol, sol) > 1e-6)
    std::cerr << "WRONG PCG!\n";
  // std::cerr << sol << std::endl;
  hermes::cuda::ThreadArrayDistributionInfo td(pressure.resolution());
  __1To2<<<td.gridSize, td.blockSize>>>(A.indexDataAccessor(), x.accessor(),
                                        pressure.data().accessor());
}

__global__ void __projectionStepU(RegularGrid2Accessor<float> u, float scale) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (u.isIndexStored(x, y)) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    if (tex2D(solidTex2, xc - 1, yc))
      u(x, y) = 0; // tex2D(uSolidTex2, xc - 1, yc);
    else if (tex2D(solidTex2, xc, yc))
      u(x, y) = 0; // tex2D(uSolidTex2, xc, yc);
    else {
      float l = tex2D(pressureTex2, xc - 1, yc);
      float r = tex2D(pressureTex2, xc, yc);
      u(x, y) -= scale * (r - l);
    }
  }
}

__global__ void __projectionStepV(RegularGrid2Accessor<float> v, float scale) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (v.isIndexStored(x, y)) {
    float xc = x + 0.5;
    float yc = y + 0.5;
    if (tex2D(solidTex2, xc, yc - 1))
      v(x, y) = 0; // tex2D(vSolidTex2, xc, yc - 1);
    else if (tex2D(solidTex2, xc, yc))
      v(x, y) = 0; // tex2D(vSolidTex2, xc, yc);
    else {
      float b = tex2D(pressureTex2, xc, yc - 1);
      float t = tex2D(pressureTex2, xc, yc);
      v(x, y) -= scale * (t - b);
    }
  }
}

template <>
void projectionStep_t(RegularGrid2Df &pressure, RegularGrid2Duc &solid,
                      StaggeredGrid2D &velocity, float dt) {
  pressureTex2.filterMode = cudaFilterModePoint;
  pressureTex2.normalized = 0;
  solidTex2.filterMode = cudaFilterModePoint;
  solidTex2.normalized = 0;
  Array2<f32> pArray(pressure.resolution());
  Array2<u8r> sArray(solid.resolution());
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  CHECK_CUDA(cudaBindTextureToArray(pressureTex2, pArray.data(), channelDesc));
  channelDesc = cudaCreateChannelDesc<unsigned char>();
  CHECK_CUDA(cudaBindTextureToArray(solidTex2, sArray.data(), channelDesc));
  memcpy(pArray, pressure.data());
  memcpy(sArray, solid.data());
  {
    auto info = velocity.u().info();
    float invdx = 1.0 / info.spacing.x;
    float scale = dt * invdx;
    hermes::cuda::ThreadArrayDistributionInfo td(info.resolution);
    __projectionStepU<<<td.gridSize, td.blockSize>>>(velocity.u().accessor(),
                                                     scale);
  }
  {
    auto info = velocity.v().info();
    float invdx = 1.0 / info.spacing.y;
    float scale = dt * invdx;
    hermes::cuda::ThreadArrayDistributionInfo td(info.resolution);
    __projectionStepV<<<td.gridSize, td.blockSize>>>(velocity.v().accessor(),
                                                     scale);
  }
  cudaUnbindTexture(pressureTex2);
  cudaUnbindTexture(solidTex2);
}*/