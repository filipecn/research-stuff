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

#include "smoke_injector2.h"

using namespace hermes::cuda;

SphereSmokeInjector2::SphereSmokeInjector2(const point2 center, f32 radius)
    : center(center), radius(radius) {}

__global__ void __injectCircle(Grid2Accessor<f32> acc, point2 center,
                               f32 radius2) {
  index2 index(blockIdx.x * blockDim.x + threadIdx.x,
               blockIdx.y * blockDim.y + threadIdx.y);
  if (acc.stores(index)) {
    auto cp = acc.worldPosition(index);
    acc[index] = 0;
    if ((cp - center).length2() <= radius2)
      acc[index] = 1;
  }
}

void SphereSmokeInjector2::inject(Grid2<f32> &field, float dt) {
  auto td = ThreadArrayDistributionInfo(field.resolution());
  __injectCircle<<<td.gridSize, td.blockSize>>>(field.accessor(), center,
                                                radius * radius);
}