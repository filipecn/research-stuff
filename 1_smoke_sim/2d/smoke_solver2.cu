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
///\file smoke_solver2.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-27
///
///\brief

#include "colliders/box_collider2.h"
#include "smoke_solver2.h"
#include "smoke_solver2_steps.h"

using namespace hermes::cuda;

__global__ void __setupScene(Collider2<f32> **solids, Collider2<f32> **scene,
                             int res) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    f32 d = 1.0 / res;
    // floor
    solids[0] = new BoxCollider2<f32>(bbox2(point2(0.f), point2(1.f, d)));
    // ceil
    solids[1] = new BoxCollider2<f32>(bbox2(point2(0.f, 1.f - d), point2(1.f)));
    // left
    solids[2] = new BoxCollider2<f32>(bbox2(point2(0.f), point2(d, 1.f)));
    // right
    solids[3] = new BoxCollider2<f32>(bbox2(point2(1.f - d, 0.f), point2(1.f)));
    *scene = new Collider2Set<f32>(solids, 4);
  }
}

__global__ void __freeScene(Collider2<f32> **solids) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    for (int i = 0; i < 5; ++i)
      delete solids[i];
}

__global__ void __rasterColliders(Collider2<f32> *const *colliders,
                                  Array2Accessor<u8> solid,
                                  Grid2Accessor<f32> u, Grid2Accessor<f32> v,
                                  Info2 info) {
  index2 index(blockIdx.x * blockDim.x + threadIdx.x,
               blockIdx.y * blockDim.y + threadIdx.y);
  if (solid.contains(index)) {
    if ((*colliders)->intersect(info.toWorld(point2(index.i, index.j))))
      solid[index] = 1;
    else
      solid[index] = 0;
    u[index] = u[index.right()] = 0;
    v[index] = v[index.up()] = 0;
  }
}

__global__ void __normalizeIFFT(f32 *g_data, int width, int height, f32 N) {

  // index = x * height + y

  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

  unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

  int index = yIndex * width + xIndex;

  g_data[index] = g_data[index] / N;
}

SmokeSolver2::SmokeSolver2() {
  uIntegrator_.reset(new MacCormackIntegrator2());
  vIntegrator_.reset(new MacCormackIntegrator2());
  integrator_.reset(new MacCormackIntegrator2());
  for (int i = 0; i < 2; ++i)
    velocity_[i].setGridType(ponos::VectorGridType::STAGGERED);
  solid_velocity_.setGridType(ponos::VectorGridType::STAGGERED);
  addScalarField(); // 0 density
  addScalarField(); // 1 temperature
}

SmokeSolver2::~SmokeSolver2() {
  __freeScene<<<1, 1>>>(scene_.list);
  CHECK_CUDA(cudaFree(scene_.list));
  CHECK_CUDA(cudaFree(scene_.colliders));
}

void SmokeSolver2::setUIntegrator(Integrator2 *integrator) {
  uIntegrator_.reset(integrator);
}

void SmokeSolver2::setVIntegrator(Integrator2 *integrator) {
  vIntegrator_.reset(integrator);
}

void SmokeSolver2::setIntegrator(Integrator2 *integrator) {
  integrator_.reset(integrator);
}

void SmokeSolver2::init() {
  CHECK_CUDA(cudaMalloc(&scene_.list, 6 * sizeof(Collider2<f32> *)));
  CHECK_CUDA(cudaMalloc(&scene_.colliders, sizeof(Collider2<f32> *)));
  scene_.target_temperature = 273.f;
  scene_.smoke_source = 0;
  for (size_t i = 0; i < 2; i++) {
    velocity_[i].u() = 0.f;
    velocity_[i].v() = 0.f;
    scalar_fields_[i][0] = 0.f;
    scalar_fields_[i][1] = 273.f;
  }
  solid_scalar_fields_[0] = 0.f;
  solid_scalar_fields_[1] = 273.f;
  vorticity_field_.u() = 0.f;
  vorticity_field_.v() = 0.f;
}

void SmokeSolver2::setResolution(const ponos::size2 &res) {
  info_.resolution = size2(res.width, res.height);
  for (size_t i = 0; i < 2; i++) {
    velocity_[i].setResolution(info_.resolution);
    for (auto &f : scalar_fields_[i])
      f.setResolution(info_.resolution);
  }
  for (auto &f : solid_scalar_fields_)
    f.setResolution(info_.resolution);
  vorticity_field_.setResolution(info_.resolution);
  pressure_.setResolution(info_.resolution);
  divergence_.setResolution(info_.resolution);
  solid_.resize(info_.resolution);
  solid_velocity_.setResolution(info_.resolution);
  force_field_.setResolution(info_.resolution);
  integrator_->set(scalar_fields_[0][0].info());
  uIntegrator_->set(velocity_[0].u().info());
  vIntegrator_->set(velocity_[0].v().info());
  pressure_matrix_.resize(info_.resolution);
  scene_.resize(info_.resolution);
}

void SmokeSolver2::setSpacing(const ponos::vec2f &s) {
  info_.setSpacing(vec2f(s.x, s.y));
  for (size_t i = 0; i < 2; i++) {
    velocity_[i].setSpacing(info_.spacing());
    for (auto &f : scalar_fields_[i])
      f.setSpacing(info_.spacing());
  }
  for (auto &f : solid_scalar_fields_)
    f.setSpacing(info_.spacing());
  vorticity_field_.setSpacing(info_.spacing());
  pressure_.setSpacing(info_.spacing());
  divergence_.setSpacing(info_.spacing());
  solid_velocity_.setSpacing(info_.spacing());
  force_field_.setSpacing(info_.spacing());
  integrator_->set(scalar_fields_[0][0].info());
  uIntegrator_->set(velocity_[0].u().info());
  vIntegrator_->set(velocity_[0].v().info());
}

void SmokeSolver2::setOrigin(const ponos::point2f &o) {
  point2f p(o.x, o.y);
  for (size_t i = 0; i < 2; i++) {
    velocity_[i].setOrigin(p);
    for (auto &f : scalar_fields_[i])
      f.setOrigin(p);
  }
  for (auto &f : solid_scalar_fields_)
    f.setOrigin(p);
  vorticity_field_.setOrigin(p);
  pressure_.setOrigin(p);
  divergence_.setOrigin(p);
  solid_velocity_.setOrigin(p);
  force_field_.setOrigin(p);
  integrator_->set(scalar_fields_[0][0].info());
  uIntegrator_->set(velocity_[0].u().info());
  vIntegrator_->set(velocity_[0].v().info());
}

size_t SmokeSolver2::addScalarField() {
  // TODO: remove size2(10)
  scalar_fields_[src].emplace_back(size2(10));
  scalar_fields_[dst].emplace_back(size2(10));
  solid_scalar_fields_.emplace_back(size2(10));
  solid_scalar_fields_.emplace_back(size2(10));
  return scalar_fields_[src].size() - 1;
}

void SmokeSolver2::step(f32 dt) {
  velocity_[dst] = velocity_[src];
  for (size_t i = 0; i < scalar_fields_[src].size(); i++)
    integrator_->advect(velocity_[dst], solid_, solid_scalar_fields_[i],
                        scalar_fields_[src][i], scalar_fields_[dst][i], dt);
  src = src ? 0 : 1;
  dst = dst ? 0 : 1;
  force_field_.u() = 0.f;
  force_field_.v() = 0.f;
  addBuoyancyForce(force_field_, solid_, scalar_fields_[src][0],
                   scalar_fields_[src][1], 273, 1.0f, 0.0f);
  // addVorticityConfinementForce(force_field_, velocity_[src],
  // solid_,vorticity_field_, 2.f, dt);
  // applyForceField(velocity_[src], solid_, force_field_, dt);
  // injectSmoke(scalar_fields_[src][0], scene_.smoke_source, dt);

  // computeDivergence(velocity_[src], solid_, solid_velocity_, divergence_);
  // solvePressureSystem(pressure_matrix_, divergence_, pressure_, solid_, dt);
  // projectionStep_t(pressure_, solid_, velocity_[src], dt);
}

void SmokeSolver2::rasterColliders() {
  __setupScene<<<1, 1>>>(scene_.list, scene_.colliders, info_.resolution.width);
  CHECK_CUDA(cudaDeviceSynchronize());
  ThreadArrayDistributionInfo td(info_.resolution);
  __rasterColliders<<<td.gridSize, td.blockSize>>>(
      scene_.colliders, solid_.accessor(), solid_velocity_.u().accessor(),
      solid_velocity_.v().accessor(), info_);
}

Scene2<f32> &SmokeSolver2::scene() { return scene_; }

Grid2<f32> &SmokeSolver2::scalarField(size_t i) {
  return scalar_fields_[src][i];
}

VectorGrid2<f32> &SmokeSolver2::velocity() { return velocity_[src]; }

Array2<u8> &SmokeSolver2::solid() { return solid_; }

Grid2<f32> &SmokeSolver2::divergence() { return divergence_; }
