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
/// \file injector.cpp
/// \author FilipeCN (filipedecn@gmail.com)
/// \date 2020-03-10
///
/// \brief

#include "injector.h"

using namespace openvdb;

namespace vdb {

size_t FluidInjector::injectLevelSet(FloatGrid::Ptr level_set,
                                     ParticleSystem &ps) {
  // Use the topology to create a PointDataTree
  points::PointDataTree::Ptr point_tree(
      new points::PointDataTree(level_set->tree(), 0, TopologyCopy()));
  // Ensure all tiles have been voxelized
  point_tree->voxelizeActiveTiles();
  auto position_type = ps.positionType();
  // Create a new Attribute Descriptor with position only
  points::AttributeSet::Descriptor::Ptr descriptor(
      openvdb::points::AttributeSet::Descriptor::create(position_type));
  // Determine the number of points / voxel and points / leaf.
  auto points_per_voxel = ps.particlesPerVoxel();
  auto voxels_per_leaf = ps.voxelsPerLeaf();
  auto points_per_leaf = points_per_voxel * voxels_per_leaf;
  // Iterate over the leaf nodes in the point tree.
  for (auto leafIter = point_tree->beginLeaf(); leafIter; ++leafIter) {
    // Initialize the attributes using the descriptor and point count.
    leafIter->initializeAttributes(descriptor, points_per_leaf);
    // Initialize the voxel offsets
    Index offset(0);
    for (Index index = 0; index < voxels_per_leaf; ++index) {
      offset += points_per_voxel;
      leafIter->setOffsetOn(index, offset);
    }
  }
  std::cerr << points_per_leaf << std::endl;

  std::cerr << level_set->tree().leafCount() << std::endl;
  // Create the points grid.
  points::PointDataGrid::Ptr points =
      openvdb::points::PointDataGrid::create(point_tree);
  // Copy the transform from the sphere grid.
  points->setTransform(level_set->transform().copy());
  // Randomize the point positions.
  std::mt19937 generator(/*seed=*/0);
  std::uniform_real_distribution<> distribution(-0.5, 0.5);
  // Iterate over the leaf nodes in the point tree.
  for (auto leafIter = points->tree().beginLeaf(); leafIter; ++leafIter) {
    // Create an AttributeWriteHandle for position.
    // Note that the handle only requires the value type, not the codec.
    points::AttributeArray &array = leafIter->attributeArray("P");
    points::AttributeWriteHandle<Vec3f> handle(array);
    // Iterate over the point indices in the leaf.
    for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
      // Compute a new random position (in the range -0.5 => 0.5).
      Vec3f position_voxel_space(distribution(generator));
      // Set the position of this point.
      // As point positions are stored relative to the voxel center, it is
      // not necessary to convert these voxel space values into
      // world-space during this process.
      handle.set(*indexIter, position_voxel_space);
    }
  }
  // Verify the point count.
  Index count = points::pointCount(points->tree());
  std::cout << "PointCount=" << count << std::endl;
}

} // namespace vdb