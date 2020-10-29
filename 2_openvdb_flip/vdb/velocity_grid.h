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
/// \file velocity_grid.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date 2020-03-08
///
/// \brief

#ifndef VDB_VELOCITY_GRID_H
#define VDB_VELOCITY_GRID_H

#include <openvdb/openvdb.h>
#include <openvdb/points/PointAttribute.h>
#include <ponos/common/defs.h>

namespace vdb {

/// \brief Staggered grid using VDB vector grid
class VelocityGrid {
public:
  /// Default constructor
  VelocityGrid();
  ~VelocityGrid();
  /// Set cell size
  /// \param spacing **[in]**
  void setSpacing(real_t spacing);
  /// \brief
  ///
  /// \param max__coordinates **[in]**
  void resize(const openvdb::math::Coord &max__coordinates);
  /// \brief
  ///
  /// \param f **[in]**
  void foreach (
      const std::function<void(openvdb::math::Coord, openvdb::Vec3f &)> &f);
  void foreach_s(const std::function<void(openvdb::math::Coord,
                                          openvdb::Vec3f &)> &callback);
  /// \brief
  ///
  /// \param points **[in]**
  /// \param attribute_name **[in]**
  void sample(openvdb::points::PointDataGrid::Ptr &points,
              const std::string &attribute_name) const;
  /// \brief
  ///
  /// \return openvdb::CoordBBox
  openvdb::CoordBBox cbbox() const;
  /// \brief
  ///
  /// \return openvdb::VectorGrid*
  openvdb::VectorGrid *grid();

private:
  openvdb::CoordBBox grid_box_;
  /// A separate grid for each velocity component
  openvdb::VectorGrid::Ptr grid_;
};

} // namespace vdb

#endif