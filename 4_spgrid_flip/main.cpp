/// SPGrid arranges grid data in lexicographically ordered blocks of 4x4x4
/// cells. These blocks are then mapped to 1D memory offsets via morton code.
/// The structure also accommodates storage for multiple data channels (up to
/// 16) interleaved in memory at block granularity, i.e., simulation variables
/// are stored along with grid geometry. The geometric block size is chosen to
/// fit all channels of a block in exactly a 4KB memory page (thus the size of
/// 64 cells per block).
/// Suppose 4 float channels: u, v, w, p (allowing 1KB of 256 entries per
/// channel, suggesting a geometric block size of 8x8x4). Channels are then
/// interleaved in memory as:
///   u_base   w_base                                   4KB page
///     |       |                                          |
///     u | v | w | p | u | v | w | p | ... | u | v | w | p
///         |       |---------------|
///      v_base   p_base           offset{p(i,j,k)}
/// For fast access to an entry (i,j,k) of a channel, a base pointer for each
/// data channel is precomputed and the offset between the base address and the
/// entry point can be calculated through bit interleaving (not trivial).
/// Block access:
///   - A bitmap is used to record all geometric blocks that have been touched.
///   - A flattened array of 64-bit packed offsets, for the first entry of each
///     block.
/// Stencil access:
///   - Stencil neighbor locations are calculated in linearized offset space,
///     without explicitly translating into geometric coordinates.
///   - Stencil offsets are translated to the flattened array packed offsets.
///   - The sum of a stencil packed offset to a packed location (neighbor
///     fetching) is provided by the PackedAdd method.
///   - A shadow grid of 6x6x6 per block can be used to pre-compute in-bulk all
///     linearized offsets reaching one cell outside the geometric block.
/// Adaptive Discretization:
///   Octree emulation is achieved by the use of a pyramid of SPGrids. The flag
/// channel is used to indicate cells that are present in the octree and also
/// faces that carry velocity degrees of freedom. Active cells and faces are
/// defined as follows:
///     - A cell at a given level of the pyramid is active if it is geometrically
///       present and undivided in the octree
///     - A face at a given level is active if that face is geometrically present
///       and undivided in the octree. This implies that an active face has at
///       one cell neighbor that is active at its level.
#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Set.h>
#include "SPGrid_Block_Iterator.h"
/// The flags field in the struct representing each grid position serves to
/// identify ghost cells, active faces, etc. The bit masks for these fields
/// are included here:
#include "FLUIDS_SIMULATION_FLAGS.h"
#include "GEOMETRY_BLOCK.h"
#include <iostream>

using namespace SPGrid;
// Field values type
typedef float real_t;
// Grid dimensions
#define DIM 2
// SPGrid stores user data in-place, i.e., data is stored in memory along the
// grid structure. We can store up to 16 custom fields per position, which are
// represented by a struct. SPGrid calls each field a channel, so there can be
// at most 16 channels.
// We could choose, for example, the following attribution for each channel:
//  flags : One field is used to stores bits that describe the properties of
//          each cell.
//  ch0   : velocity x
//  ch1   : velocity y
//  ch2   : velocity z
//  ch3   : pressure
//  ch4   : divergence
typedef struct DATA_STRUCT {
  unsigned flags;
  real_t ch0;
  real_t ch1;
  real_t ch2;
  real_t ch3;
  real_t ch4;
  real_t ch5;
  real_t ch6;
  real_t ch7;
  real_t ch8;
  real_t ch9;
  real_t ch10;
  real_t ch11;
  real_t ch12;
  real_t ch13;
  real_t ch14;
} DS;
// As SPGrid stores data by interleaving bits and combining fields through
// bit masks created from data layout, it needs to know beforehand the size
// and types of our data.
using DS_Allocator = SPGrid_Allocator<DS, DIM>;
using DS_MASK = typename DS_Allocator::Array<>::mask;
// set some convenient aliases
using Data_array_type = typename DS_Allocator::Array<real_t>::type;
using Const_data_array_type = typename DS_Allocator::Array<const real_t>::type;
using Flags_array_type = typename DS_Allocator::Array<unsigned>::type;
using Flags_set_type = SPGrid_Set<Flags_array_type>;

/// Lets create a simple quadtree of just 4 levels refining children 3 and 0:
/// | 2 | 3 |
/// |---|---|
/// | 0 | 1 |
/// \tparam T_SET flags set
template<class T_SET>
class Quadtree {
public:
  Quadtree(T_SET &set_input, const GEOMETRY_BLOCK &domain) : set{set_input} {
    iterate(domain, 0, 0);
  }
  bool consume(GEOMETRY_BLOCK block, int level, int child) {
    if (level < 3 && child % 3 == 0)
      return true;
    // the first cell in the block represents the block
    set.Mask(std_array<int, 2>(block.imin(0), block.imin(1)), SPGrid_Cell_Type_Interior);
    return false;
  }
  void iterate(const GEOMETRY_BLOCK &block, int level, int child) {
    int imin = block.imin.data[0];
    int jmin = block.imin.data[1];
    int kmin = block.imin.data[2];
    int imax = block.imax.data[0];
    int jmax = block.imax.data[1];
    int kmax = block.imax.data[2];

    float Xmin = block.Xmin.data[0];
    float Ymin = block.Xmin.data[1];
    float Zmin = block.Xmin.data[2];
    float Xmax = block.Xmax.data[0];
    float Ymax = block.Xmax.data[1];
    float Zmax = block.Xmax.data[2];

    if (!consume(block, level, child))
      return;

    int half_size = (imax - imin) / 2;
    float half_dx = (Xmax - Xmin) / 2;

    std_array<int, 3> root_i;
    std_array<float, 3> root_X;
    // CHILD 0
    root_i = std_array<int, 3>(imin, jmin, kmin);
    root_X = std_array<float, 3>(Xmin, Ymin, Zmin);
    iterate(GEOMETRY_BLOCK(root_i, root_i + half_size, root_X, root_X + half_dx), level + 1, 0);
    // CHILD 1
    root_i = std_array<int, 3>(imin + half_size, jmin, kmin);
    root_X = std_array<float, 3>(Xmin + half_dx, Ymin, Zmin);
    iterate(GEOMETRY_BLOCK(root_i, root_i + half_size, root_X, root_X + half_dx), level + 1, 1);
    // CHILD 2
    root_i = std_array<int, 3>(imin, jmin + half_size, kmin);
    root_X = std_array<float, 3>(Xmin, Ymin + half_dx, Zmin);
    iterate(GEOMETRY_BLOCK(root_i, root_i + half_size, root_X, root_X + half_dx), level + 1, 2);
    // CHILD 3
    root_i = std_array<int, 3>(imin + half_size, jmin + half_size, kmin);
    root_X = std_array<float, 3>(Xmin + half_dx, Ymin + half_dx, Zmin);
    iterate(GEOMETRY_BLOCK(root_i, root_i + half_size, root_X, root_X + half_dx), level + 1, 3);
  }
private:
  T_SET &set;
};

int main(int argc, char **argv) {
  // Grid resolution, must be power of 2
  int size = 8;
  // Simulation domain
  GEOMETRY_BLOCK block(0, size, 0.f, size);
  // SPGrid allocator
  DS_Allocator allocator(size, size);
  // We can access fields by getting separate accessors for each field
  Data_array_type vx = allocator.Get_Array(&DS::ch0);
  Data_array_type vy = allocator.Get_Array(&DS::ch1);
  Data_array_type vz = allocator.Get_Array(&DS::ch2);
  Flags_array_type flags = allocator.Get_Array(&DS::flags);
  // TODO: ??
  Flags_set_type flag_set(flags);
  // Init quadtree
  Quadtree<Flags_set_type> quadtree(flag_set, block);
  // update offsets
  flag_set.Refresh_Block_Offsets();
  // Iterate over all blocks
  for (SPGrid_Block_Iterator<Flags_array_type::MASK> iterator(flag_set.Get_Blocks()); iterator.Valid();
       iterator.Next()) {
    // here we can access channel values for each block
    // (int)(iterator.Data(flags) & SPGrid_Cell_Type_Interior)
    // iterator.Data(vx)
    // block index coordinates: iterator.Index()
  }
  // Another approach is to use directly the block offset stream:
  const unsigned long *const b = flag_set.Get_Blocks().first;
  const int block_count = flag_set.Get_Blocks().second;
  // Channels can be accessed directly as well
  auto vx_ptr = (float *) vx.Get_Data_Ptr();
  auto mask_ptr = (unsigned *) flags.Get_Data_Ptr();
  for (int index = 0; index < block_count; ++index) {
    // here we add the block offset to the channel base ptr
    auto x = reinterpret_cast<float *>((unsigned long) vx_ptr + b[index]);
    auto mask = reinterpret_cast<unsigned *>((unsigned long) mask_ptr + b[index]);
    // the packed offset is
    unsigned long packed_offset = (unsigned long) x - (unsigned long) vx_ptr;
    unsigned long base_addr = reinterpret_cast<unsigned long >(vx_ptr);
  }
  return 0;
}