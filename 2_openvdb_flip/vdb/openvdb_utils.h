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
/// \file openvdb_utils.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date 2020-03-13
///
/// \brief

#ifndef VDB_OPENVDB_UTILS_H
#define VDB_OPENVDB_UTILS_H

#include <openvdb/openvdb.h>
#include <openvdb/points/PointCount.h>

namespace vdb {

struct AttributesDescriptor {
  /// \brief Construct a new Attributes Descriptor object
  ///
  /// \param d **[in]**
  AttributesDescriptor(
      openvdb::points::AttributeSet::DescriptorPtr d =
          openvdb::points::AttributeSet::Descriptor::create(
              openvdb::points::TypedAttributeArray<
                  openvdb::Vec3f,
                  openvdb::points::NullCodec>::attributeType()));
  /// \brief
  ///
  /// \tparam T
  /// \param names **[in]**
  template <typename T> void add(const std::vector<std::string> &names) {
    if (std::is_same<T, float>::value)
      for (auto name : names)
        float_attributes_.insert(name);
    else if (std::is_same<T, openvdb::Vec3f>::value)
      for (auto name : names)
        vec3f_attributes_.insert(name);
    else if (std::is_same<T, openvdb::Vec3d>::value)
      for (auto name : names)
        vec3d_attributes_.insert(name);
    else
      std::cerr << "attribute type not supported!\n";
  }
  /// \brief
  ///
  /// \tparam T
  /// \return size_t
  template <typename T> size_t size() const {
    if (std::is_same<T, float>::value)
      return float_attributes_.size();
    if (std::is_same<T, openvdb::Vec3f>::value)
      return vec3f_attributes_.size();
    if (std::is_same<T, openvdb::Vec3d>::value)
      return vec3d_attributes_.size();
    return 0;
  }
  /// \param tree **[in/out]**
  void appendTo(openvdb::points::PointDataTree::Ptr &tree) const;

  openvdb::points::AttributeSet::DescriptorPtr descriptor;

private:
  std::set<std::string> float_attributes_, vec3f_attributes_, vec3d_attributes_;
};

/// \brief
///
/// \param leaf **[in]**
void printLeafPoints(const openvdb::points::PointDataTree::LeafNodeType *leaf);

/// \brief
///
/// \param tree_a **[in]**
/// \param tree_b **[in]**
/// \param attributes_descriptor **[in]**
void combine(openvdb::points::PointDataTree::Ptr &tree_a,
             const openvdb::points::PointDataTree::Ptr &tree_b,
             AttributesDescriptor &attributes_descriptor);

} // namespace vdb

#endif
