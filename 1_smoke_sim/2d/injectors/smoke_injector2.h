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
///\file smoke_injector2.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-02-10
///
///\brief

#ifndef SMOKE_INJECTOR2_H
#define SMOKE_INJECTOR2_H

#include <hermes/hermes.h>

class SmokeInjector2 {
public:
  virtual void inject(hermes::cuda::Grid2<f32> &field, float dt) = 0;
};

class SphereSmokeInjector2 : public SmokeInjector2 {
public:
  SphereSmokeInjector2(const hermes::cuda::point2 center, f32 radius);
  void inject(hermes::cuda::Grid2<f32> &field, float dt) override;

  hermes::cuda::point2 center;
  f32 radius;
};

#endif