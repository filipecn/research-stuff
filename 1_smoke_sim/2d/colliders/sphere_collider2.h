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
///\file sphere_collider2.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-27
///
///\brief

#ifndef SPHERE_COLLIDER_2_H
#define SPHERE_COLLIDER_2_h

#include "collider2.h"

template <typename T> class SphereCollider2 : public Collider2<T> {
public:
  __host__ __device__ SphereCollider2(const hermes::cuda::Point2<T> &center,
                                      T radius)
      : c(center), r(radius) {}
  __host__ __device__ bool
  intersect(const hermes::cuda::Point2<T> &p) const override {
    return hermes::cuda::distance2(c, p) <= r * r;
  }
  __host__ __device__ T distance(const hermes::cuda::Point2<T> &p,
                                 hermes::cuda::Point2<T> *s) override {
    return hermes::cuda::distance(c, p) - r;
  }

private:
  hermes::cuda::Point2<T> c;
  T r;
};

#endif