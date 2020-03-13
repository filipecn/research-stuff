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
/// \file openvdb_utils.cpp
/// \author FilipeCN (filipedecn@gmail.com)
/// \date 2020-03-13
///
/// \brief

#include "openvdb_utils.h"

using namespace openvdb;

namespace vdb {

void printLeafPoints(const openvdb::points::PointDataTree::LeafNodeType *leaf) {
  const auto &array = leaf->attributeArray("P");
  points::AttributeHandle<Vec3f> handle(array);
  std::cerr << "Leaf points:\n";
  for (int x = 0; x < 8; ++x)
    for (int y = 0; y < 8; ++y)
      for (int z = 0; z < 8; ++z) {
        Coord xyz(x, y, z);
        std::cerr << xyz << ":\t";
        // get tree1 data
        for (auto index = leaf->beginIndexVoxel(Coord(x, y, z)); index; ++index)
          std::cerr << handle.get(*index) << " ";
        std::cerr << std::endl;
      }
}

void combine(openvdb::points::PointDataTree *tree_a,
             const openvdb::points::PointDataTree *tree_b) {
  // The main idea is to iterate over tree_b voxels and merge the points into
  // the correspondent voxels of tree_a. The algorithm goes like this:
  // for each active leaf of tree_b
  //    - retrieve voxel data from leaf
  //    - retrieve voxel data from correspondent leaf of tree_a
  //    - create a new vector of offsets and data
  //    - iterate over each voxel of the leaf and append to these vectors data
  //      from tree_a and tree_b
  //    - reset tree_a's leaf data and put the new vectors

  // define common attribute information
  using PositionAttribute =
      points::TypedAttributeArray<Vec3f, points::NullCodec>;
  NamePair position_type = PositionAttribute::attributeType();
  points::AttributeSet::Descriptor::Ptr descriptor(
      points::AttributeSet::Descriptor::create(position_type));
  Index voxels_per_leaf = points::PointDataGrid::TreeType::LeafNodeType::SIZE;
  // for each active leaf of tree_b, merge its data with the correspondent leaf
  // of tree_a
  for (auto tree_b_leaf = tree_b->cbeginLeaf(); tree_b_leaf; ++tree_b_leaf) {
    // retrieve correspondend leaf of tree 1
    points::PointDataTree::LeafNodeType *tree_a_leaf = nullptr;
    for (auto index_iter = tree_b_leaf->beginIndexOn(); index_iter;
         ++index_iter) {
      math::Coord c;
      index_iter.getCoord(c);
      tree_a_leaf = tree_a->touchLeaf(c);
      break;
    }
    if (!tree_a_leaf)
      // in case of no active voxels in tree_b_leaf (is that possible!?)
      continue;
    // retrieve data handles (a copy from tree_a's array,  since it will be
    // overwritten)
    auto array_a_copy = tree_a_leaf->attributeArray("P").copy();
    points::AttributeHandle<Vec3f> handle_a_copy(*array_a_copy);
    auto &array_b = tree_b_leaf->attributeArray("P");
    points::AttributeHandle<Vec3f> handle_b(array_b);
    // reset tree_a leaf data with the sum of the sizes
    tree_a_leaf->initializeAttributes(descriptor,
                                      array_a_copy->size() + array_b.size());
    // get output handle (tree_a)
    auto &array_a = tree_a_leaf->attributeArray("P");
    points::AttributeWriteHandle<Vec3f> handle(array_a);
    // retrieve data from voxels and merge into one single array
    std::vector<PointDataIndex32> offsets;
    std::vector<Vec3f> data;
    for (int x = 0; x < 8; ++x)
      for (int y = 0; y < 8; ++y)
        for (int z = 0; z < 8; ++z) {
          Coord xyz(x, y, z);
          // get tree_a data
          for (auto index = tree_a_leaf->beginIndexVoxel(xyz); index; ++index)
            data.push_back(handle_a_copy.get(*index));
          // get tree_b data
          for (auto index = tree_b_leaf->beginIndexVoxel(xyz); index; ++index)
            data.push_back(handle_b.get(*index));
          offsets.push_back(data.size());
        }
    // set offsets
    for (Index index = 0; index < voxels_per_leaf; ++index)
      tree_a_leaf->setOffsetOn(index, offsets[index]);
    tree_a_leaf->validateOffsets();
    // save data
    for (auto index_iter = tree_a_leaf->beginIndexOn(); index_iter;
         ++index_iter)
      handle.set(*index_iter, data[*index_iter]);
  }
}

} // namespace vdb