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
///\file box_collider2.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-27
///
///\brief

#ifndef BOX_COLLIDER_2_H
#define BOX_COLLIDER_2_H

#include "collider2.h"

template <typename T> class BoxCollider2 : public Collider2<T> {
public:
  __host__ __device__ BoxCollider2(const hermes::cuda::BBox2<T> &box)
      : box(box) {}
  __host__ __device__ bool
  intersect(const hermes::cuda::Point2<T> &p) const override {
    return box.contains(p);
  }
  __host__ __device__ T distance(const hermes::cuda::Point2<T> &p,
                                 hermes::cuda::Point2<T> *s) override {
    return 0;
  }

private:
  hermes::cuda::BBox2<T> box;
};

#endif