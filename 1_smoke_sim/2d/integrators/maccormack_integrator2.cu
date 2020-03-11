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
///\file mac_cormack_integrator2.cu
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-27
///
///\brief

#include "maccormack_integrator2.h"

using namespace hermes::cuda;

MacCormackIntegrator2::MacCormackIntegrator2() = default;

void MacCormackIntegrator2::set(Info2 info) {
  if (info.resolution != phi_n_hat.info().resolution)
    phi_n_hat.setResolution(info.resolution);
  phi_n_hat.setOrigin(info.origin());
  phi_n_hat.setSpacing(info.spacing());
  if (info.resolution != phi_n1_hat.info().resolution)
    phi_n1_hat.setResolution(info.resolution);
  phi_n1_hat.setOrigin(info.origin());
  phi_n1_hat.setSpacing(info.spacing());
  integrator.set(info);
}

__global__ void
__computePhiN1(VectorGrid2Accessor<f32> vel, Array2Accessor<u8> solid,
               Grid2Accessor<f32> solid_phi, Grid2Accessor<f32> phi_n_hat,
               Grid2Accessor<f32> phi_n1_hat, Grid2Accessor<f32> in,
               Grid2Accessor<f32> out, f32 dt) {
  index2 index(blockIdx.x * blockDim.x + threadIdx.x,
               blockIdx.y * blockDim.y + threadIdx.y);
  if (in.stores(index)) {
    if (solid[index]) {
      out[index] = solid_phi[index];
      return;
    }
    vec2f v = vel[index];
    point2f p = in.worldPosition(index);
    point2f wp = p - v * dt;
    point2f npos = in.gridPosition(wp);
    index2 pos(npos.x, npos.y);
    f32 nodeValues[4];
    nodeValues[0] = in[pos];
    nodeValues[1] = in[pos.right()];
    nodeValues[2] = in[pos.plus(1, 1)];
    nodeValues[3] = in[pos.up()];
    f32 phiMin = min(nodeValues[3],
                     min(nodeValues[2], min(nodeValues[1], nodeValues[0])));
    f32 phiMax = max(nodeValues[3],
                     max(nodeValues[2], max(nodeValues[1], nodeValues[0])));

    out[index] = phi_n1_hat[index] + 0.5 * (in[index] - phi_n_hat[index]);
    out[index] = max(min(out[index], phiMax), phiMin);
  }
}

void MacCormackIntegrator2::advect(VectorGrid2<f32> &velocity,
                                   Array2<u8> &solid, Grid2<f32> &solid_phi,
                                   Grid2<f32> &phi, Grid2<f32> &phi_out,
                                   f32 dt) {
  // phi^_n+1 = A(phi_n)
  integrator.advect(velocity, solid, solid_phi, phi, phi_n1_hat, dt);
  // phi^_n = Ar(phi^_n+1)
  integrator.advect(velocity, solid, solid_phi, phi_n1_hat, phi_n_hat, -dt);
  // phi_n+1 = phi^_n+1 + 0.5 * (phi_n - phi^_n)
  hermes::cuda::ThreadArrayDistributionInfo td(phi.resolution());
  __computePhiN1<<<td.gridSize, td.blockSize>>>(
      velocity.accessor(), solid.accessor(), solid_phi.accessor(),
      phi_n_hat.accessor(), phi_n1_hat.accessor(), phi.accessor(),
      phi_out.accessor(), dt);
}

__global__ void __computePhiN1(VectorGrid2Accessor<f32> vel,
                               Grid2Accessor<f32> phi_n_hat,
                               Grid2Accessor<f32> phi_n1_hat,
                               Grid2Accessor<f32> in, Grid2Accessor<f32> out,
                               f32 dt) {
  index2 index(blockIdx.x * blockDim.x + threadIdx.x,
               blockIdx.y * blockDim.y + threadIdx.y);
  if (in.stores(index)) {
    // if (solid[index]) {
    //   out[index] = solid_phi[index];
    //   return;
    // }
    vec2f v = vel[index];
    point2f p = in.worldPosition(index);
    point2f wp = p - v * dt;
    point2f npos = in.gridPosition(wp);
    index2 pos(npos.x, npos.y);
    f32 nodeValues[4];
    nodeValues[0] = in[index];
    nodeValues[1] = in[index.right()];
    nodeValues[2] = in[index.plus(1, 1)];
    nodeValues[3] = in[index.up()];
    f32 phiMin = min(nodeValues[3],
                     min(nodeValues[2], min(nodeValues[1], nodeValues[0])));
    f32 phiMax = max(nodeValues[3],
                     max(nodeValues[2], max(nodeValues[1], nodeValues[0])));

    out[index] = phi_n1_hat[index] + 0.5 * (in[index] - phi_n_hat[index]);
    out[index] = max(min(out[index], phiMax), phiMin);
  }
}

void MacCormackIntegrator2::advect(VectorGrid2<f32> &velocity,
                                   Array2<MaterialType> &material,
                                   Grid2<f32> &phi, Grid2<f32> &phi_out,
                                   f32 dt) {
  // phi^_n+1 = A(phi_n)
  integrator.advect(velocity, material, phi, phi_n1_hat, dt);
  // phi^_n = Ar(phi^_n+1)
  integrator.advect(velocity, material, phi_n1_hat, phi_n_hat, -dt);
  // phi_n+1 = phi^_n+1 + 0.5 * (phi_n - phi^_n)
  hermes::cuda::ThreadArrayDistributionInfo td(phi.resolution());
  __computePhiN1<<<td.gridSize, td.blockSize>>>(
      velocity.accessor(), phi_n_hat.accessor(), phi_n1_hat.accessor(),
      phi.accessor(), phi_out.accessor(), dt);
}