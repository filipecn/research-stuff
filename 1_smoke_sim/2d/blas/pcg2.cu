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
///\file pcg2.cu
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-28
///
///\brief

#include "pcg2.h"
#include <hermes/blas/blas.h>

using namespace hermes::cuda;

int pcg(Vector<f32> &x, FDMatrix2<f32> &A, Vector<f32> &b,
        size_t maxNumberOfIterations, float tolerance) {
  // cpu memory
  std::vector<double> h_r(b.size(), 0);
  std::vector<double> h_z(b.size(), 0);
  std::vector<double> precon(b.size(), 0);
  // FDMatrix2H h_A(A.gridSize());
  // h_A.copy(A);
  // mic0(precon, h_A, 0.97, 0.25);
  // device memory
  Vector<f32> m;
  Vector<f32> r(b.size(), 0); // residual
  Vector<f32> z(b.size(), 0); // auxiliar
  Vector<f32> s(b.size(), 0); // search
  std::cerr << "max " << maxNumberOfIterations << std::endl;
  // r = b - A * x
  r = b - A * x;
  if (BLAS::infNorm(r) <= tolerance)
    return 0;
  // z = M * r
  z = r;
  // memcpy(h_r, r);
  // memcpy(h_z, z);
  // applyMIC0(h_A, precon, h_r, h_z);
  // memcpy(z, h_z);
  // s = z
  s = z;
  // sigma = z '* r
  f32 sigma = BLAS::dot(z, r);
  // std::cerr << "sigma = " << sigma << std::endl;
  size_t it = 0;
  // std::cerr << "S " << s << std::endl;
  while (it < maxNumberOfIterations) {
    // z = As
    z = A * s;
    // std::cerr << "Z " << z << std::endl;
    // std::cerr << "S " << s << std::endl;
    // alpha = sigma / (z '* s)
    f32 alpha = sigma / BLAS::dot(z, s);
    // std::cerr << "alpha " << alpha << std::endl;
    // x = alpha * s + x
    BLAS::axpy(alpha, s, x, x);
    // r = r - alpha * z
    BLAS::axpy(-alpha, z, r, r);
    // std::cerr << "r norm test\n";
    // std::cerr << r << std::endl;
    if (BLAS::infNorm(r) <= tolerance)
      return it;
    // z = M * r
    z = r;
    // memcpy(h_r, r);
    // memcpy(h_z, z);
    // applyMIC0(h_A, precon, h_r, h_z);
    // memcpy(z, h_z);
    // sigmaNew = z '* r
    f32 sigmaNew = BLAS::dot(z, r);
    // std::cerr << "sigmaNew " << sigmaNew << std::endl;
    // if (sigmaNew < tolerance * tolerance)
    //   break;
    // s = z + (sigmaNew / sigma) * s
    BLAS::axpy(sigmaNew / sigma, s, z, s);
    sigma = sigmaNew;
    ++it;
  }
  // auto acc = h_A.accessor();
  // std::cerr << "BAD PCG!\n" << acc << std::endl;
  // std::cerr << b << std::endl;
  return -1;
}
