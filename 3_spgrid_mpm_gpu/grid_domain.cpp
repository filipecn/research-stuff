#include "grid_domain.h"

Grid::~Grid() {
  using namespace hermes::cuda;
  CHECK_CUDA(cudaFree(d_grid));
  CHECK_CUDA(cudaFree(d_channels));
}

void Grid::resize(const hermes::cuda::size3 &resolution) {
  this->resolution = resolution;
  using namespace hermes::cuda;
  using namespace SPGrid;
  // As SPGrid stores data by interleaving bits and combining fields through
  // bit masks created from data layout, it needs to know beforehand the size
  // and types of our data.
  using T_DATA_STRUCT = DATA_STRUCT;
  using SPG_Allocator = SPGrid_Allocator<T_DATA_STRUCT, 3>;
  using T_MASK = typename SPG_Allocator::Array<real_t>::mask;

  // Allocate memory for the sp grid data structure
  CHECK_CUDA(cudaMalloc((void **)&d_grid, sizeof(struct DATA_STRUCT) *
                                              resolution.total() *
                                              memory_scale));
  // Allocate channel pointers
  CHECK_CUDA(cudaMalloc((void **)&d_channels, sizeof(real_t *) * 15));
  // Compute individual channel pointers in host side
  // Here we choose the following attribution for each channel:
  //  ch0 : velocity x
  //  ch1 : velocity y
  //  ch2 : velocity z
  //  ch3 : pressure
  //  ch4 : divergence
  h_channels[0] = reinterpret_cast<real_t *>(
      (uint64_t)d_grid + uint64_t(OffsetOfMember(&DATA_STRUCT::ch0) *
                                  T_MASK::elements_per_block)); // velocity x
  h_channels[1] = reinterpret_cast<real_t *>(
      (uint64_t)d_grid + uint64_t(OffsetOfMember(&DATA_STRUCT::ch1) *
                                  T_MASK::elements_per_block)); // velocity y
  h_channels[2] = reinterpret_cast<real_t *>(
      (uint64_t)d_grid + uint64_t(OffsetOfMember(&DATA_STRUCT::ch2) *
                                  T_MASK::elements_per_block)); // velocity z
  h_channels[3] = reinterpret_cast<real_t *>(
      (uint64_t)d_grid + uint64_t(OffsetOfMember(&DATA_STRUCT::ch3) *
                                  T_MASK::elements_per_block)); // pressure
  h_channels[4] = reinterpret_cast<real_t *>(
      (uint64_t)d_grid + uint64_t(OffsetOfMember(&DATA_STRUCT::ch4) *
                                  T_MASK::elements_per_block)); // divergence
  // Transfer pointers information to device
  CHECK_CUDA(cudaMemcpy(d_channels, h_channels, sizeof(real_t *) * 15,
                        cudaMemcpyHostToDevice));
  // Compute pointer to flags channel
  d_flags = reinterpret_cast<unsigned *>(
      (uint64_t)d_grid + uint64_t(OffsetOfMember(&DATA_STRUCT::flags) *
                                  T_MASK::elements_per_block));
}

void Grid::clear() {
  using namespace hermes::cuda;
  CHECK_CUDA(cudaMemset(d_grid, 0,
                        sizeof(struct DATA_STRUCT) * resolution.total() *
                            memory_scale));
}
