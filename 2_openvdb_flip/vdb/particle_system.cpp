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
/// \file vdb_particle_system.cpp
/// \author FilipeCN (filipedecn@gmail.com)
/// \date 2020-03-07
///
/// \brief

#include "particle_system.h"

using namespace openvdb;

namespace vdb {

ParticleSystem::ParticleSystem() = default;

ParticleSystem::ParticleSystem(
    std::initializer_list<std::initializer_list<real_t>> list) {
  std::vector<Vec3R> positions, velocities;
  for (auto e : list) {
    if (e.size() >= 3)
      positions.push_back(Vec3R(e.begin()[0], e.begin()[1], e.begin()[2]));
    if (e.size() >= 6)
      velocities.push_back(Vec3R(e.begin()[3], e.begin()[4], e.begin()[5]));
    else
      velocities.push_back(Vec3R(0, 0, 0));
  }
  setPositions(positions);
  setVelocities(velocities);
}

ParticleSystem::~ParticleSystem() = default;

size_t ParticleSystem::size() const {
  return (point_index_grid_) ? point_index_grid_->tree().activeVoxelCount() : 0;
}

void ParticleSystem::setPositionType(openvdb::NamePair attribute_type) {
  position_attribute_type_ = attribute_type;
}

NamePair ParticleSystem::positionType() const {
  return position_attribute_type_;
}

Index ParticleSystem::particlesPerVoxel() const { return points_per_voxel_; }

Index ParticleSystem::voxelsPerLeaf() const { return voxels_per_leaf_; }

void ParticleSystem::setPositions(const std::vector<Vec3R> &new_positions) {
  // To init the VDB point grid:
  //  - Compute the grid transform based on point density per voxel
  //  - Create a point index grid to allow fast search operations
  //  - Create a point data grid to store particle attributes
  //  - Populate particle attributes on the grid
  // Before building VDB structures we need to wrap our point atribute vectors
  // into interfaces that communicate to VDB
  points::PointAttributeVector<Vec3R> positions_wrapper(new_positions);
  // The first step is to compute the voxel size in order to get the grid
  // tranform. Defining a point density usually offers a good balance of memory
  // against performance.
  float voxel_size =
      points::computeVoxelSize(positions_wrapper, points_per_voxel_);
  // create the transform from voxel_size
  transform_ = math::Transform::createLinearTransform(voxel_size);
  // Create the point index grid
  // Note: Leaf nodes store a single point-index array and the voxels are only
  // integer offsets into that array. The actual points are never stored in the
  // acceleration structure, only offsets into an external array.
  point_index_grid_ = tools::createPointIndexGrid<tools::PointIndexGrid>(
      positions_wrapper, *transform_);
  /// Create the point data grid
  /// Note: Point attributes are stored in leaf nodes and ordered by voxel for
  /// fast random and sequential access.
  grid_ = points::createPointDataGrid<points::NullCodec, points::PointDataGrid>(
      *point_index_grid_, positions_wrapper, *transform_);
  // Register particle velocities before putting it on the grid
  points::TypedAttributeArray<Vec3R, points::NullCodec>::registerType();
  NamePair velocity_attribute =
      points::TypedAttributeArray<Vec3R, points::NullCodec>::attributeType();
  points::appendAttribute(grid_->tree(), "V", velocity_attribute);
  grid_->setName("Particles");
}

void ParticleSystem::setVelocities(const std::vector<Vec3R> &new_velocities) {
  // Before building VDB structures we need to wrap our point atribute vectors
  // into interfaces that communicate to VDB
  points::PointAttributeVector<Vec3R> velocities_wrapper(new_velocities);
  // Populate the attributes on the points
  points::populateAttribute<points::PointDataTree, tools::PointIndexTree,
                            points::PointAttributeVector<Vec3R>>(
      grid_->tree(), point_index_grid_->tree(), "V", velocities_wrapper);
}

points::PointDataGrid::Ptr ParticleSystem::grid() { return grid_; }

std::ostream &operator<<(std::ostream &os, const ParticleSystem &ps) {
  os << "ParticleSystem Info\n";
  // Iterate over all the leaf nodes in the grid.
  for (auto leaf_iter = ps.grid_->tree().cbeginLeaf(); leaf_iter; ++leaf_iter) {
    // Verify the leaf origin.
    os << "Leaf" << leaf_iter->origin() << std::endl;
    // Extract the position attribute from the leaf by name (P is position).
    const points::AttributeArray &position_array =
        leaf_iter->constAttributeArray("P");
    // Extract the velocity attribute from the leaf by name (V).
    const points::AttributeArray &velocity_array =
        leaf_iter->constAttributeArray("V");
    // Create read-only handles for position and radius.
    points::AttributeHandle<Vec3f> position_handle(position_array);
    points::AttributeHandle<Vec3R> velocity_handle(velocity_array);
    // Iterate over the point indices in the leaf.
    for (auto index_iter = leaf_iter->beginIndexOn(); index_iter;
         ++index_iter) {
      // Extract the voxel-space position of the point.
      Vec3f voxel_position = position_handle.get(*index_iter);
      // Extract the world-space position of the voxel.
      Vec3d xyz = index_iter.getCoord().asVec3d();
      // Compute the world-space position of the point.
      Vec3f world_position =
          ps.grid_->transform().indexToWorld(voxel_position + xyz);
      // Extract the radius of the point.
      Vec3R velocity = velocity_handle.get(*index_iter);
      // Verify the index, world-space position and radius of the point.
      os << "* PointIndex=[" << *index_iter << "] ";
      os << "WorldPosition=" << world_position << " ";
      os << "Velocity=" << velocity << std::endl;
    }
  }
  return os;
}

} // namespace vdb