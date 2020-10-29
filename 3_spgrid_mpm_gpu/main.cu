#include "grid_domain.h"
#include <iostream>

using namespace hermes::cuda;
using namespace SPGrid;

constexpr int D = 3;

int main(int argc, char *argv[]) {
  // set some conveninent aliases
  using T_STRUCT = DATA_STRUCT;
  using SPG_Allocator = SPGrid_Allocator<T_STRUCT, D>;
  using T_MASK = typename SPG_Allocator::Array<real_t>::mask;
  // data
  std::vector<uint64_t> h_masks = {T_MASK::xmask, T_MASK::ymask, T_MASK::zmask};
  uint64_t *d_masks;
  CHECK_CUDA(cudaMalloc((void **)&d_masks, sizeof(uint64_t) * D));
  // setup grid
  Grid grid;
  grid.d_masks = d_masks;
  grid.h_masks = h_masks.data();
  grid.resize(hermes::cuda::size3(128, 128, 128));
  // cleanup
  grid.clear();
  CHECK_CUDA(cudaFree(d_masks));
  return 0;
}
