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
///\file semi_lagrangian_integrator2.cu
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-27
///
///\brief

#include "semi_lagrangian_integrator2.h"

using namespace hermes::cuda;

SemiLagrangianIntegrator2::SemiLagrangianIntegrator2() = default;

__global__ void __advect(VectorGrid2Accessor<f32> vel, Array2Accessor<u8> solid,
                         Grid2Accessor<f32> solid_phi, Grid2Accessor<f32> in,
                         Grid2Accessor<f32> out, f32 dt) {
  index2 index(blockIdx.x * blockDim.x + threadIdx.x,
               blockIdx.y * blockDim.y + threadIdx.y);
  if (in.stores(index)) {
    if (solid[index]) {
      out[index] = solid_phi[index];
      return;
    }
    point2f p = in.worldPosition(index);
    vec2f v = vel[index];
    point2f pos = p - v * dt;
    // TODO: clip on solid walls
    out[index] = in(pos);
  }
}

void SemiLagrangianIntegrator2::advect(VectorGrid2<f32> &velocity,
                                       Array2<u8> &solid, Grid2<f32> &solid_phi,
                                       Grid2<f32> &phi, Grid2<f32> &phi_out,
                                       f32 dt) {
  hermes::cuda::ThreadArrayDistributionInfo td(phi.resolution());
  __advect<<<td.gridSize, td.blockSize>>>(velocity.accessor(), solid.accessor(),
                                          solid_phi.accessor(), phi.accessor(),
                                          phi_out.accessor(), dt);
}

__global__ void __advect(VectorGrid2Accessor<f32> vel, Grid2Accessor<f32> in,
                         Grid2Accessor<f32> out, f32 dt) {
  index2 index(blockIdx.x * blockDim.x + threadIdx.x,
               blockIdx.y * blockDim.y + threadIdx.y);
  if (in.stores(index)) {
    point2f p = in.worldPosition(index);
    vec2f v = vel(p);
    point2f pos = p - v * dt;
    // TODO: clip on solid walls
    out[index] = in(pos);
  }
}

void SemiLagrangianIntegrator2::advect(VectorGrid2<f32> &velocity,
                                       Array2<MaterialType> &material,
                                       Grid2<f32> &in, Grid2<f32> &out,
                                       f32 dt) {
  hermes::cuda::ThreadArrayDistributionInfo td(in.resolution());
  __advect<<<td.gridSize, td.blockSize>>>(velocity.accessor(), in.accessor(),
                                          out.accessor(), dt);
}