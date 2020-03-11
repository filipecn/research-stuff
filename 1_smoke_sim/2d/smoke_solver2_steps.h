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
///\file kernels2.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-27
///
///\brief

#ifndef SMOKE_SOLVER_2_STEPS_H
#define SMOKE_SOLVER_2_STEPS_H

#include <hermes/hermes.h>

void injectTemperature(hermes::cuda::Grid2<f32> &temperature,
                       hermes::cuda::Grid2<f32> &targetTemperature, f32 dt);

void injectSmoke(hermes::cuda::Grid2<f32> &smoke,
                 hermes::cuda::Grid2<u8> &source, f32 dt);
/// \brief
///
/// \param velocity **[in]**
/// \param solid **[in]**
/// \param force_field **[in]**
/// \param dt **[in]**
void applyForceField(hermes::cuda::VectorGrid2<f32> &velocity,
                     hermes::cuda::Array2<u8> &solid,
                     hermes::cuda::VectorGrid2<f32> &force_field, f32 dt);
/// \brief
///
/// \param force_field **[in]**
/// \param solid **[in]**
/// \param density **[in]**
/// \param temperature **[in]**
/// \param ambient_temperature **[in]**
/// \param alpha **[in]**
/// \param beta **[in]**
void addBuoyancyForce(hermes::cuda::VectorGrid2<f32> &force_field,
                      hermes::cuda::Array2<u8> &solid,
                      hermes::cuda::Grid2<f32> &density,
                      hermes::cuda::Grid2<f32> &temperature,
                      f32 ambient_temperature, f32 alpha, f32 beta);

void addVorticityConfinementForce(
    hermes::cuda::VectorGrid2<f32> &forceField,
    hermes::cuda::VectorGrid2<f32> &velocity, hermes::cuda::Grid2<u8> &solid,
    hermes::cuda::VectorGrid2<f32> &vorticityField, f32 eta, f32 dt);
/// \brief
///
/// \param velocity **[in]**
/// \param solid **[in]**
/// \param solidVelocity **[in]**
/// \param divergence **[out]**
void computeDivergence(hermes::cuda::VectorGrid2<f32> &velocity,
                       hermes::cuda::Array2<u8> &solid,
                       hermes::cuda::VectorGrid2<f32> &solidVelocity,
                       hermes::cuda::Grid2<f32> &divergence);

size_t setupPressureSystem(hermes::cuda::Grid2<f32> &divergence,
                           hermes::cuda::Grid2<u8> &solid,
                           hermes::cuda::FDMatrix2<f32> &A, f32 dt,
                           hermes::cuda::Array1<f64> &rhs);

void solvePressureSystem(hermes::cuda::FDMatrix2<f32> &pressureMatrix,
                         hermes::cuda::Grid2<f32> &divergence,
                         hermes::cuda::Grid2<f32> &pressure,
                         hermes::cuda::Grid2<u8> &solid, f32 dt);

void projectionStep(hermes::cuda::Grid2<f32> &pressure,
                    hermes::cuda::Grid2<u8> &solid,
                    hermes::cuda::VectorGrid2<f32> &velocity, f32 dt);

#endif