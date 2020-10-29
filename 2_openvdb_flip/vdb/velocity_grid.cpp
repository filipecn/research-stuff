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
/// \file velocity_grid.cpp
/// \author FilipeCN (filipedecn@gmail.com)
/// \date 2020-03-08
///
/// \brief

#include "velocity_grid.h"
#include <openvdb/tools/ValueTransformer.h>
#include <tbb/parallel_for.h>

using namespace openvdb;

namespace vdb {

VelocityGrid::VelocityGrid() {
  grid_ = VectorGrid::create();
  grid_->setGridClass(GRID_STAGGERED);
}

VelocityGrid::~VelocityGrid() = default;

void VelocityGrid::setSpacing(real_t spacing) {
  grid_->setTransform(math::Transform::createLinearTransform(spacing));
}

void VelocityGrid::resize(const math::Coord &max__coordinates) {
  grid_box_ = CoordBBox(math::Coord(0, 0, 0), max__coordinates);
  grid_->denseFill(grid_box_, Vec3f(0, 0, 0), true);
  // grid_->clip(grid_box_);
}

CoordBBox VelocityGrid::cbbox() const { return grid_box_; }

VectorGrid *VelocityGrid::grid() { return grid_.get(); }

void VelocityGrid::foreach (
    const std::function<void(math::Coord, Vec3f &)> &callback) {
  struct Op {
    Op(const std::function<void(math::Coord, Vec3f &)> &f) : f(f) {}
    inline void operator()(const VectorGrid::ValueOnIter &iter) const {
      Vec3f value = *iter;
      f(iter.getCoord(), value);
      iter.setValue(value);
    }
    const std::function<void(math::Coord, Vec3f &)> &f;
  };
  tools::foreach (grid_->beginValueOn(), Op(callback));
  return;
  struct LeafProcessor {
    LeafProcessor(const std::function<void(math::Coord, Vec3f &)> &f) : f(f) {}
    // Define an IteratorRange that splits the iteration space of a leaf
    // iterator.
    using IterRange =
        tree::IteratorRange<typename VectorGrid::TreeType::LeafCIter>;
    void operator()(IterRange &range) const {
      // Note: this code must be thread-safe.
      // Iterate over a subrange of the leaf iterator's iteration space.
      for (; range; ++range) {
        // Retrieve the leaf node to which the iterator is pointing.
        const VectorGrid::TreeType::LeafNodeType &leaf = *range.iterator();
        // Update the global counter.
        for (auto index = leaf.beginValueOn(); index; ++index) {
          Vec3f value;
          math::Coord ijk;
          index.getCoord(ijk);
          f(ijk, value);
        }
      }
    }
    const std::function<void(math::Coord, Vec3f &)> &f;
  };
  LeafProcessor proc(callback);
  // Wrap a leaf iterator in an IteratorRange.
  LeafProcessor::IterRange range(grid_->tree().cbeginLeaf());
  // Iterate over leaf nodes in parallel.
  tbb::parallel_for(range, proc);
}

void VelocityGrid::foreach_s(
    const std::function<void(math::Coord, Vec3f &)> &callback) {
  auto acc = grid_->getAccessor();
  for (int i = grid_box_.min().x(); i <= grid_box_.max().x(); ++i)
    for (int j = grid_box_.min().y(); j <= grid_box_.max().y(); ++j)
      for (int k = grid_box_.min().z(); k <= grid_box_.max().z(); ++k) {
        math::Coord ijk(i, j, k);
        Vec3f value = acc.getValue(ijk);
        callback(ijk, value);
        acc.setValue(ijk, value);
      }
}

void VelocityGrid::sample(points::PointDataGrid::Ptr &points,
                          const std::string &attribute_name) const {}

} // namespace vdb