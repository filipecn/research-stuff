#ifndef GRID_DOMAIN_H
#define GRID_DOMAIN_H

#include <SPGrid/SPGrid_Allocator.h>
#include <SPGrid/SPGrid_Set.h>
#include <SPGrid/SPGrid_Utilities.h>
#include <SPGrid/std_array.h>
#include <hermes/common/cuda.h>
#include <hermes/common/size.h>

// SPGrid stores user data in-place, i.e., data is stored in memory along the
// grid structure. We can store up to 16 custom fields per position, which are
// represented by a struct. SPGrid calls each field a channel, so there can be
// at most 16 channels. For example:
struct DATA_STRUCT {
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
};
// This class holds all SPGrid structure
class Grid {
public:
  ~Grid();
  void resize(const hermes::cuda::size3 &resolution);
  void clear();

  // Grid resolution
  hermes::cuda::size3 resolution;
  // Used portion of the grid
  real_t memory_scale{0.6};
  // SPGrid data is organized by channels that represent the fields we want to
  // store.
  real_t *d_grid{nullptr};      // SPGrid raw data
  real_t *h_channels[15];       // list of channel pointers in host
  real_t **d_channels{nullptr}; // list of channel pointers
  unsigned *d_flags{nullptr};   // channel pointer for flags channel

  // These channels are interleaved in memory and are accessed through bit
  // masks:
  const u64 *d_masks{nullptr};
  const u64 *h_masks{nullptr};
};

#endif