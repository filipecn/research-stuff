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
/// \file vdb_particle_system.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date 2020-03-07
///
/// \brief

#ifndef VDB_PARTICLE_SYSTEM_H
#define VDB_PARTICLE_SYSTEM_H

// #include <stdint.h>

#include <initializer_list>
#include <iostream>
#include <openvdb/openvdb.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointCount.h>
#include <ponos/common/defs.h>

namespace vdb {

/// \brief Particle system using OpenVDB points
/// Particle positions are sored directly as grid leafs and the properties are
/// represented by grid attributes.
class ParticleSystem {
public:
  /// Default constructor
  ParticleSystem();
  /// Init the particle system with a initial set of particle positions
  /// Ex1:  Initializing all particles with only positions (velocity = 0):
  ///       ParticleSystem ps = {{0,0,0}, {0,1,0}};
  /// Ex2:  Initializing all particles with custom velocity (1,0,0) as well:
  //        ParticleSystem ps = {{0,0,0,1,0,0},{0,1,0,1,0,0}};
  /// \param list **[in]**
  ParticleSystem(std::initializer_list<std::initializer_list<real_t>> list);
  ~ParticleSystem();
  /// \return size_t active particle count
  size_t size() const;
  /// \brief Set the Position Type object
  ///
  /// \param attribute_type **[in]**
  void setPositionType(openvdb::NamePair attribute_type);
  /// \return openvdb::NamePair
  openvdb::NamePair positionType() const;
  /// \return openvdb::Index
  openvdb::Index particlesPerVoxel() const;
  /// \return openvdb::Index
  openvdb::Index voxelsPerLeaf() const;
  /// \brief Repleace particle positions
  /// Note: pre-existent particles are removed
  /// \param new_positions **[in]**
  void setPositions(const std::vector<openvdb::Vec3R> &new_positions);
  /// \brief Overwrites particle velocities
  /// \param new_velocities **[in]**
  void setVelocities(const std::vector<openvdb::Vec3R> &new_velocities);
  /// Adds a new particle to the system
  /// \param position **[in]**
  void add(openvdb::Vec3R position);
  /// \return openvdb::points::PointDataGrid::Ptr
  openvdb::points::PointDataGrid::Ptr grid();

  friend std::ostream &operator<<(std::ostream &os, const ParticleSystem &ps);

private:
  openvdb::NamePair position_attribute_type_ =
      openvdb::points::TypedAttributeArray<
          openvdb::Vec3f,
          openvdb::points::FixedPointCodec<false>>::attributeType();
  openvdb::points::AttributeSet::Descriptor::Ptr position_descriptor_;
  /// Number of particles per voxel
  openvdb::Index points_per_voxel_{8};
  /// Number of voxels per leaf
  openvdb::Index voxels_per_leaf_{
      openvdb::points::PointDataGrid::TreeType::LeafNodeType::SIZE};
  /// VDB point grid transform
  openvdb::math::Transform::Ptr transform_;
  /// Space-partitioning acceleration structure for points. Partitions the
  /// points into voxels to accelerate range and nearest neighbor searches.
  openvdb::tools::PointIndexGrid::Ptr point_index_grid_;
  /// Attribute-owned data structure for points.
  openvdb::points::PointDataGrid::Ptr grid_;
};

std::ostream &operator<<(std::ostream &os, const ParticleSystem &ps);

} // namespace vdb

#endif