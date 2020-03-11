#include <circe/circe.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <hermes/hermes.h>

class CudaOpenGLInterop {
public:
  CudaOpenGLInterop(ponos::size2 size)
      : size_(ponos::size3(size.width, size.height, 1)) {
    using namespace hermes::cuda;
    unsigned int size_tex_data = sizeof(GLubyte) * size_.total() * 4;
    CHECK_CUDA(cudaMalloc(&cuda_dev_render_buffer, size_tex_data));
    circe::TextureAttributes ta;
    ta.width = size_.width;
    ta.height = size_.height;
    ta.internalFormat = GL_RGBA8;
    ta.format = GL_RGBA;
    ta.type = GL_UNSIGNED_BYTE;
    ta.target = GL_TEXTURE_2D;
    circe::TextureParameters tp;
    tp[GL_TEXTURE_MIN_FILTER] = GL_NEAREST;
    tp[GL_TEXTURE_MAG_FILTER] = GL_NEAREST;
    // tp[GL_TEXTURE_WRAP_S] = GL_CLAMP_TO_BORDER;
    // tp[GL_TEXTURE_WRAP_T] = GL_CLAMP_TO_BORDER;
    // tp[GL_TEXTURE_WRAP_R] = GL_CLAMP_TO_BORDER;
    texture.set(ta, tp);
    // Register this texture with CUDA
    CHECK_CUDA(cudaGraphicsGLRegisterImage(
        &cuda_tex_resource, texture.textureObjectId(), GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard));
    using namespace circe;
    CHECK_GL_ERRORS;
  }

  CudaOpenGLInterop(ponos::size3 size) : size_(size) {
    using namespace hermes::cuda;
    unsigned int size_tex_data = size_.total() * sizeof(float);
    CHECK_CUDA(cudaMalloc(&cuda_dev_render_buffer, size_tex_data));
    circe::TextureAttributes ta;
    ta.width = size_.width;
    ta.height = size_.height;
    ta.depth = size_.depth;
    ta.internalFormat = GL_RED;
    ta.format = GL_RED;
    ta.type = GL_FLOAT;
    ta.target = GL_TEXTURE_3D;
    circe::TextureParameters tp;
    tp.target = GL_TEXTURE_3D;
    tp[GL_TEXTURE_MIN_FILTER] = GL_LINEAR;
    tp[GL_TEXTURE_MAG_FILTER] = GL_LINEAR;
    tp[GL_TEXTURE_WRAP_S] = GL_CLAMP_TO_BORDER;
    tp[GL_TEXTURE_WRAP_T] = GL_CLAMP_TO_BORDER;
    tp[GL_TEXTURE_WRAP_R] = GL_CLAMP_TO_BORDER;
    texture.set(ta, tp);
    // Register this texture with CUDA
    CHECK_CUDA(cudaGraphicsGLRegisterImage(
        &cuda_tex_resource, texture.textureObjectId(), GL_TEXTURE_3D,
        cudaGraphicsRegisterFlagsWriteDiscard));
    using namespace circe;
    CHECK_GL_ERRORS;
  }

  ~CudaOpenGLInterop() {
    if (cuda_dev_render_buffer)
      cudaFree(cuda_dev_render_buffer);
  }

  void sendToTexture() {
    using namespace hermes::cuda;
    // We want to copy cuda_dev_render_buffer data to the texture
    // Map buffer objects to get CUDA device pointers
    cudaArray *texture_ptr;
    CHECK_CUDA(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
    CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&texture_ptr,
                                                     cuda_tex_resource, 0, 0));

    int num_texels = size_.total();
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    CHECK_CUDA(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dev_render_buffer,
                                 size_tex_data, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));
  }

  void bindTexture(GLenum t) { texture.bind(t); }
  template <typename T> T *bufferPointer() {
    return (T *)cuda_dev_render_buffer;
  }

private:
  circe::Texture texture;
  ponos::size3 size_;
  void *cuda_dev_render_buffer = nullptr;
  struct cudaGraphicsResource *cuda_tex_resource = nullptr;
};

void renderDensity(hermes::cuda::Grid2<f32> &in, unsigned int *out);

void renderSolids(const hermes::cuda::Array2<u8> &in, unsigned int *out);