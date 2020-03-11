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
///\file scene2.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-27
///
///\brief

#ifndef SCENE_2_H
#define SCENE_2_h

#include "colliders/collider2.h"
#include <hermes/hermes.h>

template <typename T> class Scene2 {
public:
  Collider2<T> **colliders = nullptr;
  Collider2<T> **list = nullptr;
  void resize(hermes::cuda::size2 size) {
    target_temperature.resize(size);
    smoke_source.resize(size);
  }
  hermes::cuda::Array2<f32> target_temperature;
  hermes::cuda::Array2<u8> smoke_source;
};

#endif