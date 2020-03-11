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
///\file smoke_solver2.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-27
///
///\brief

#ifndef SMOKE_SOLVER_2_H
#define SMOKE_SOLVER_2_H

#include "integrators/maccormack_integrator2.h"
#include "scene2.h"
#include <hermes/hermes.h>
#include <ponos/ponos.h>

/// Eulerian grid based solver for smoke simulations. Stores its data in fast
/// device texture memory.
class SmokeSolver2 {
public:
  SmokeSolver2();
  ~SmokeSolver2();
  void setUIntegrator(Integrator2 *integrator);
  void setVIntegrator(Integrator2 *integrator);
  void setIntegrator(Integrator2 *integrator);
  void init();
  void setResolution(const ponos::size2 &res);
  /// Sets cell size
  /// \param _dx scale
  void setSpacing(const ponos::vec2f &s);
  /// Sets lower left corner position
  /// \param o offset
  void setOrigin(const ponos::point2f &o);
  size_t addScalarField();
  /// Advances one simulation step
  /// \param dt time step
  void step(float dt);
  /// Raster collider bodies and velocities into grid simulations
  void rasterColliders();
  Scene2<float> &scene();
  hermes::cuda::Grid2<f32> &scalarField(size_t i);
  hermes::cuda::VectorGrid2<f32> &velocity();
  hermes::cuda::Array2<u8> &solid();
  hermes::cuda::Grid2<f32> &divergence();

private:
  Scene2<float> scene_;
  std::shared_ptr<Integrator2> vIntegrator_;
  std::shared_ptr<Integrator2> uIntegrator_;
  std::shared_ptr<Integrator2> integrator_;
  hermes::cuda::FDMatrix2<f32> pressure_matrix_;
  hermes::cuda::VectorGrid2<f32> velocity_[2];
  hermes::cuda::VectorGrid2<f32> solid_velocity_;
  hermes::cuda::VectorGrid2<f32> force_field_, vorticity_field_;
  hermes::cuda::Grid2<f32> pressure_;
  hermes::cuda::Grid2<f32> divergence_;
  hermes::cuda::Array2<u8> solid_;
  std::vector<hermes::cuda::Grid2<f32>> scalar_fields_[2];
  std::vector<hermes::cuda::Grid2<f32>> solid_scalar_fields_;
  hermes::cuda::Info2 info_;
  size_t src = 0;
  size_t dst = 1;
};

#endif