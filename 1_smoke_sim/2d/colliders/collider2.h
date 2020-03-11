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
///\file collider2.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-27
///
///\brief

#ifndef COLLIDER_2_H
#define COLLIDER_2_H

#include <hermes/hermes.h>

template <typename T> class Collider2 {
public:
  /// \param v
  void setVelocity(const hermes::cuda::Vector2<T> &v) { velocity = v; }
  /// \param p
  /// \return true
  /// \return false
  virtual __host__ __device__ bool
  intersect(const hermes::cuda::Point2<T> &p) const = 0;
  /// \param p
  /// \return T
  virtual __host__ __device__ T distance(const hermes::cuda::Point2<T> &p,
                                         hermes::cuda::Point2<T> *s) = 0;

  hermes::cuda::Vector2<T> velocity;
};

template <typename T> class Collider2Set : public Collider2<T> {
public:
  __host__ __device__ Collider2Set(Collider2<T> **l, size_t n)
      : list(l), n(n) {}
  __host__ __device__ bool
  intersect(const hermes::cuda::Point2<T> &p) const override {
    for (size_t i = 0; i < n; i++)
      if (list[i]->intersect(p))
        return true;
    return false;
  }
  __host__ __device__ T distance(const hermes::cuda::Point2<T> &p,
                                 hermes::cuda::Point2<T> *s) override {
    T mdist = hermes::cuda::Constants::greatest<T>();
    bool negative = false;
    hermes::cuda::Point2<T> cp;
    for (int i = 0; i < n; i++) {
      T dist = list[i]->distance(p, &cp);
      if (abs(dist) < mdist) {
        mdist = abs(dist);
        negative = dist < 0;
        *s = cp;
      }
    }
    return (negative ? -1 : 1) * mdist;
  }
  __host__ __device__ int size() const { return n; }

private:
  Collider2<T> **list;
  size_t n = 0;
};

#endif