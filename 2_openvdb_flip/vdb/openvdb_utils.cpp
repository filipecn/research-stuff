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

void printLeafPoints(const points::PointDataTree::LeafNodeType *leaf) {
  auto m = leaf->attributeSet().descriptorPtr()->map();
  std::cerr << "Leaf points:\n";
  auto leaf_dim = points::PointDataTree::LeafNodeType::DIM;
  for (unsigned int x = 0; x < leaf_dim; ++x)
    for (unsigned int y = 0; y < leaf_dim; ++y)
      for (unsigned int z = 0; z < leaf_dim; ++z) {
        Coord xyz(x, y, z);
        std::cerr << xyz << ":\t";
        // get tree1 data
        for (auto index = leaf->beginIndexVoxel(Coord(x, y, z)); index;
             ++index) {
          std::cerr << "INDEX " << *index << " ";
          // iterate attributes
          for (auto s : m) {
            const auto &array = leaf->attributeArray(s.first);
            std::cerr << s.first << " ";
            if (leaf->attributeSet().get(s.second)->valueTypeIsVector()) {
              if (leaf->attributeSet().get(s.second)->valueTypeSize() == 24) {
                points::AttributeHandle<Vec3d> handle(array);
                std::cerr << handle.get(*index);
              } else {
                points::AttributeHandle<Vec3f> handle(array);
                std::cerr << handle.get(*index);
              }
            }
            if (leaf->attributeSet()
                    .get(s.second)
                    ->valueTypeIsFloatingPoint()) {
              if (leaf->attributeSet().get(s.second)->valueTypeSize() == 4) {
                points::AttributeHandle<float> handle(array);
                std::cerr << handle.get(*index);
              }
            }
            std::cerr << " ";
          }
          std::cerr << "\t";
        }
        std::cerr << std::endl;
      }
}

struct AttributeHandles {
  AttributeHandles(const points::PointDataTree::LeafNodeType *leaf) {
    auto m = leaf->attributeSet().descriptorPtr()->map();
    for (auto s : m) {
      if (leaf->attributeSet().get(s.second)->valueTypeIsVector()) {
        if (leaf->attributeSet().get(s.second)->valueTypeSize() == 24)
          addAttribute<Vec3d>(s.first, leaf->constAttributeArray(s.first));
        else
          addAttribute<Vec3f>(s.first, leaf->constAttributeArray(s.first));
      } else if (leaf->attributeSet()
                     .get(s.second)
                     ->valueTypeIsFloatingPoint()) {
        if (leaf->attributeSet().get(s.second)->valueTypeSize() == 4)
          addAttribute<float>(s.first, leaf->constAttributeArray(s.first));
      }
    }
  }
  template <typename T>
  void addAttribute(const std::string &name,
                    const points::AttributeArray &array) {
    if (std::is_same<T, float>::value &&
        float_handles_map_.find(name) != float_handles_map_.end()) {
      float_handles_map_[name] = float_handles_.size();
      float_handles_.emplace_back(array);
    } else if (std::is_same<T, Vec3f>::value &&
               vec3f_handles_map_.find(name) != vec3f_handles_map_.end()) {
      vec3f_handles_map_[name] = vec3f_handles_.size();
      vec3f_handles_.emplace_back(array);
    } else if (std::is_same<T, Vec3d>::value &&
               vec3d_handles_map_.find(name) != vec3d_handles_map_.end()) {
      vec3d_handles_map_[name] = vec3d_handles_.size();
      vec3d_handles_.emplace_back(array);
    }
  }

private:
  std::map<std::string, size_t> float_handles_map_, vec3f_handles_map_,
      vec3d_handles_map_;
  std::vector<points::AttributeHandle<float>> float_handles_;
  std::vector<points::AttributeHandle<Vec3f>> vec3f_handles_;
  std::vector<points::AttributeHandle<Vec3d>> vec3d_handles_;
};

struct AttributeWriteHandles {
  AttributeWriteHandles(points::PointDataTree::LeafNodeType *leaf) {
    auto m = leaf->attributeSet().descriptorPtr()->map();
    for (auto s : m) {
      if (leaf->attributeSet().get(s.second)->valueTypeIsVector()) {
        if (leaf->attributeSet().get(s.second)->valueTypeSize() == 24)
          addAttribute<Vec3d>(s.first, leaf->attributeArray(s.first));
        else
          addAttribute<Vec3f>(s.first, leaf->attributeArray(s.first));
      } else if (leaf->attributeSet()
                     .get(s.second)
                     ->valueTypeIsFloatingPoint()) {
        if (leaf->attributeSet().get(s.second)->valueTypeSize() == 4)
          addAttribute<float>(s.first, leaf->attributeArray(s.first));
      }
    }
  }

  template <typename T>
  void addAttribute(const std::string &name, points::AttributeArray &array) {
    if (std::is_same<T, float>::value &&
        float_handles_map_.find(name) != float_handles_map_.end()) {
      float_handles_map_[name] = float_handles_.size();
      float_handles_.emplace_back(array);
    } else if (std::is_same<T, Vec3f>::value &&
               vec3f_handles_map_.find(name) != vec3f_handles_map_.end()) {
      vec3f_handles_map_[name] = vec3f_handles_.size();
      vec3f_handles_.emplace_back(array);
    } else if (std::is_same<T, Vec3d>::value &&
               vec3d_handles_map_.find(name) != vec3d_handles_map_.end()) {
      vec3d_handles_map_[name] = vec3d_handles_.size();
      vec3d_handles_.emplace_back(array);
    }
  }

private:
  std::map<std::string, size_t> float_handles_map_, vec3f_handles_map_,
      vec3d_handles_map_;
  std::vector<points::AttributeWriteHandle<float>> float_handles_;
  std::vector<points::AttributeWriteHandle<Vec3f>> vec3f_handles_;
  std::vector<points::AttributeWriteHandle<Vec3d>> vec3d_handles_;
};

struct LeafData {
  template <typename T> void addAttribute(const std::string &name) {
    if (std::is_same<T, float>::value &&
        float_arrays_map_.find(name) != float_arrays_map_.end()) {
      float_arrays_map_[name] = float_arrays_.size();
      float_arrays_.push_back(std::vector<float>());
    } else if (std::is_same<T, Vec3d>::value &&
               vec3d_arrays_map_.find(name) != vec3d_arrays_map_.end()) {
      vec3d_arrays_map_[name] = vec3d_arrays_.size();
      vec3d_arrays_.push_back(std::vector<Vec3d>());
    } else if (std::is_same<T, Vec3f>::value &&
               vec3f_arrays_map_.find(name) != vec3f_arrays_map_.end()) {
      vec3f_arrays_map_[name] = vec3f_arrays_.size();
      vec3f_arrays_.push_back(std::vector<Vec3f>());
    }
  }

private:
  std::map<std::string, size_t> float_arrays_map_, vec3f_arrays_map_,
      vec3d_arrays_map_;
  std::vector<std::vector<float>> float_arrays_;
  std::vector<std::vector<Vec3f>> vec3f_arrays_;
  std::vector<std::vector<Vec3d>> vec3d_arrays_;
};

void combine(points::PointDataTree::Ptr &tree_a,
             const points::PointDataTree::Ptr &tree_b) {
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
  Index voxels_per_leaf = points::PointDataGrid::TreeType::LeafNodeType::SIZE;
  auto leaf_dim = points::PointDataTree::LeafNodeType::DIM;
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
    auto descriptor = tree_a_leaf->attributeSet().descriptorPtr();
    if (!tree_b_leaf->attributeSet().descriptorPtr()->hasSameAttributes(
            *descriptor)) {
      std::cerr << "trees must share same attribute set!";
      return;
    }
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
    for (unsigned int x = 0; x < leaf_dim; ++x)
      for (unsigned int y = 0; y < leaf_dim; ++y)
        for (unsigned int z = 0; z < leaf_dim; ++z) {
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