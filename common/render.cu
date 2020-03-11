#include "render.h"

using namespace hermes::cuda;

texture<float, cudaTextureType2D> scalar_tex2;
texture<float, cudaTextureType2D> density_tex2;
texture<unsigned char, cudaTextureType2D> solid_tex2;

__device__ f32 clamp(f32 x, f32 a, f32 b) { return max(a, min(b, x)); }

__device__ int clamp(int x, int a, int b) { return max(a, min(b, x)); }

__device__ int rgbToInt(f32 r, f32 g, f32 b, f32 a) {
  r = clamp(r, 0.0f, 255.0f);
  g = clamp(g, 0.0f, 255.0f);
  b = clamp(b, 0.0f, 255.0f);
  a = clamp(a, 0.0f, 255.0f);
  return (int(a) << 24) | (int(b) << 16) | (int(g) << 8) | int(r);
}

__global__ void __renderDensity(unsigned int *out, size2 size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < size.width && y < size.height) {
    auto value =
        tex2D(density_tex2, x / float(size.width), y / float(size.height)) *
        255;
    uchar4 c4 = make_uchar4(value, value, value, 255);
    out[y * size.width + x] = rgbToInt(c4.x, c4.y, c4.z, c4.w);
  }
}

void renderDensity(Grid2<f32> &in, unsigned int *out) {
  CuArray2<f32> pArray = in.data();
  auto td = ThreadArrayDistributionInfo(in.resolution());
  density_tex2.normalized = 1;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  CHECK_CUDA(cudaBindTextureToArray(density_tex2, pArray.data(), channelDesc));
  __renderDensity<<<td.gridSize, td.blockSize>>>(out, in.resolution());
  cudaUnbindTexture(density_tex2);
}

__global__ void __renderSolids(Array2CAccessor<u8> solid, unsigned int *out) {
  index2 index(blockIdx.x * blockDim.x + threadIdx.x,
               blockIdx.y * blockDim.y + threadIdx.y);
  if (solid.contains(index)) {
    if (solid[index] > 0) {
      uchar4 c4 = make_uchar4(solid[index], 100, 0, 255);
      out[index.j * solid.size().width + index.i] =
          rgbToInt(c4.x, c4.y, c4.z, c4.w);
    }
  }
}

void renderSolids(const Array2<u8> &in, unsigned int *out) {
  auto td = ThreadArrayDistributionInfo(in.size());
  __renderSolids<<<td.gridSize, td.blockSize>>>(in.constAccessor(), out);
}