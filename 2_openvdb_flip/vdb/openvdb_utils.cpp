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
#include <openvdb/points/PointAttribute.h>

using namespace openvdb;

namespace vdb {

AttributesDescriptor::AttributesDescriptor(
    points::AttributeSet::DescriptorPtr d)
    : descriptor(d) {}

void AttributesDescriptor::appendTo(points::PointDataTree::Ptr &tree) const {
  for (auto s : float_attributes_)
    points::appendAttribute<float>(*tree, s);
  for (auto s : vec3f_attributes_)
    points::appendAttribute<Vec3f>(*tree, s);
  for (auto s : vec3d_attributes_)
    points::appendAttribute<Vec3d>(*tree, s);
}

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
  AttributeHandles(const points::PointDataTree::LeafNodeType *leaf,
                   bool copy = false) {
    auto m = leaf->attributeSet().descriptorPtr()->map();
    for (auto s : m) {
      if (leaf->attributeSet().get(s.second)->valueTypeIsVector()) {
        if (leaf->attributeSet().get(s.second)->valueTypeSize() == 24)
          addAttribute<Vec3d>(s.first, leaf->constAttributeArray(s.first));
        else {
          if (copy) {
            copies_.push_back(leaf->constAttributeArray(s.first).copy());
            addAttribute<Vec3f>(s.first, *copies_[copies_.size() - 1]);
          } else
            addAttribute<Vec3f>(s.first, leaf->constAttributeArray(s.first));
        }
      } else if (leaf->attributeSet()
                     .get(s.second)
                     ->valueTypeIsFloatingPoint()) {
        if (leaf->attributeSet().get(s.second)->valueTypeSize() == 4)
          addAttribute<float>(s.first, leaf->constAttributeArray(s.first));
      }
    }
  }
  template <typename T> T get(const std::string &name, Index index) {
    if (std::is_same<T, Vec3f>::value &&
        vec3f_handles_map_.find(name) != vec3f_handles_map_.end())
      return vec3f_handles_[vec3f_handles_map_[name]].get(index);
    else if (std::is_same<T, Vec3d>::value &&
             vec3d_handles_map_.find(name) != vec3d_handles_map_.end())
      return vec3d_handles_[vec3d_handles_map_[name]].get(index);
    std::cerr << "error: inexistent attribute name!\n";
    return T{};
  }

  template <typename T>
  void addAttribute(const std::string &name,
                    const points::AttributeArray &array) {
    if (std::is_same<T, float>::value &&
        float_handles_map_.find(name) == float_handles_map_.end()) {
      float_handles_map_[name] = float_handles_.size();
      float_handles_.emplace_back(array);
    } else if (std::is_same<T, Vec3f>::value &&
               vec3f_handles_map_.find(name) == vec3f_handles_map_.end()) {
      vec3f_handles_map_[name] = vec3f_handles_.size();
      vec3f_handles_.emplace_back(array);
    } else if (std::is_same<T, Vec3d>::value &&
               vec3d_handles_map_.find(name) == vec3d_handles_map_.end()) {
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
  std::vector<points::AttributeArray::Ptr> copies_;
};

template <> float AttributeHandles::get(const std::string &name, Index index) {
  if (float_handles_map_.find(name) != float_handles_map_.end())
    return float_handles_[float_handles_map_[name]].get(index);
  std::cerr << "error: inexistent attribute name!\n";
  return 0;
}

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
  void set(const std::string &name, Index index, T value) {
    if (std::is_same<T, Vec3f>::value &&
        vec3f_handles_map_.find(name) != vec3f_handles_map_.end())
      vec3f_handles_[vec3f_handles_map_[name]].set(index, value);
    else if (std::is_same<T, Vec3d>::value &&
             vec3d_handles_map_.find(name) != vec3d_handles_map_.end())
      vec3d_handles_[vec3d_handles_map_[name]].set(index, value);
  }

  template <typename T> T get(const std::string &name, Index index) {
    if (std::is_same<T, Vec3f>::value &&
        vec3f_handles_map_.find(name) != vec3f_handles_map_.end())
      return vec3f_handles_[vec3f_handles_map_[name]].get(index);
    else if (std::is_same<T, Vec3d>::value &&
             vec3d_handles_map_.find(name) != vec3d_handles_map_.end())
      return vec3d_handles_[vec3d_handles_map_[name]].get(index);
    std::cerr << "error: inexistent vec attribute name!\n";
    return T{};
  }

  template <typename T>
  void addAttribute(const std::string &name, points::AttributeArray &array) {
    if (std::is_same<T, float>::value &&
        float_handles_map_.find(name) == float_handles_map_.end()) {
      float_handles_map_[name] = float_handles_.size();
      float_handles_.emplace_back(array);
    } else if (std::is_same<T, Vec3f>::value &&
               vec3f_handles_map_.find(name) == vec3f_handles_map_.end()) {
      vec3f_handles_map_[name] = vec3f_handles_.size();
      vec3f_handles_.push_back(points::AttributeWriteHandle<Vec3f>(array));
    } else if (std::is_same<T, Vec3d>::value &&
               vec3d_handles_map_.find(name) == vec3d_handles_map_.end()) {
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

template <>
float AttributeWriteHandles::get(const std::string &name, Index index) {
  if (float_handles_map_.find(name) != float_handles_map_.end())
    return float_handles_[float_handles_map_[name]].get(index);
  std::cerr << "error: inexistent attribute name!\n";
  return 0;
}

template <>
void AttributeWriteHandles::set(const std::string &name, Index index,
                                float value) {
  if (float_handles_map_.find(name) != float_handles_map_.end())
    float_handles_[float_handles_map_[name]].set(index, value);
  else
    std::cerr << "error: float attribute does not exist!\n";
}

struct LeafData {
  LeafData(const points::PointDataTree::LeafNodeType *leaf) {
    auto m = leaf->attributeSet().descriptorPtr()->map();
    for (auto s : m) {
      if (leaf->attributeSet().get(s.second)->valueTypeIsVector()) {
        if (leaf->attributeSet().get(s.second)->valueTypeSize() == 24)
          addAttribute<Vec3d>(s.first);
        else
          addAttribute<Vec3f>(s.first);
      } else if (leaf->attributeSet()
                     .get(s.second)
                     ->valueTypeIsFloatingPoint()) {
        if (leaf->attributeSet().get(s.second)->valueTypeSize() == 4)
          addAttribute<float>(s.first);
      }
    }
  }
  template <typename T> void addAttribute(const std::string &name) {
    if (std::is_same<T, float>::value &&
        float_arrays_map_.find(name) == float_arrays_map_.end()) {
      float_arrays_map_[name] = float_arrays_.size();
      float_arrays_.push_back(std::vector<float>());
    } else if (std::is_same<T, Vec3d>::value &&
               vec3d_arrays_map_.find(name) == vec3d_arrays_map_.end()) {
      vec3d_arrays_map_[name] = vec3d_arrays_.size();
      vec3d_arrays_.push_back(std::vector<Vec3d>());
    } else if (std::is_same<T, Vec3f>::value &&
               vec3f_arrays_map_.find(name) == vec3f_arrays_map_.end()) {
      vec3f_arrays_map_[name] = vec3f_arrays_.size();
      vec3f_arrays_.push_back(std::vector<Vec3f>());
    }
  }

  void appendFromIndexAt(Index index, AttributeHandles &handles) {
    for (auto s : float_arrays_map_)
      float_arrays_[s.second].push_back(handles.get<float>(s.first, index));
    for (auto s : vec3f_arrays_map_)
      vec3f_arrays_[s.second].push_back(handles.get<Vec3f>(s.first, index));
    for (auto s : vec3d_arrays_map_)
      vec3d_arrays_[s.second].push_back(handles.get<Vec3d>(s.first, index));
  }
  size_t dataSize() const {
    if (float_arrays_.size())
      return float_arrays_[0].size();
    if (vec3f_arrays_.size())
      return vec3f_arrays_[0].size();
    if (vec3d_arrays_.size())
      return vec3d_arrays_[0].size();
    return 0;
  }

  void populate(Index index, AttributeWriteHandles &handles) const {
    for (auto s : float_arrays_map_)
      handles.set<float>(s.first, index, float_arrays_[s.second][index]);
    for (auto s : vec3f_arrays_map_)
      handles.set<Vec3f>(s.first, index, vec3f_arrays_[s.second][index]);
    for (auto s : vec3d_arrays_map_)
      handles.set<Vec3d>(s.first, index, vec3d_arrays_[s.second][index]);
  }

private:
  std::map<std::string, size_t> float_arrays_map_, vec3f_arrays_map_,
      vec3d_arrays_map_;
  std::vector<std::vector<float>> float_arrays_;
  std::vector<std::vector<Vec3f>> vec3f_arrays_;
  std::vector<std::vector<Vec3d>> vec3d_arrays_;
}; // namespace vdb

void combine(points::PointDataTree::Ptr &tree_a,
             const points::PointDataTree::Ptr &tree_b,
             AttributesDescriptor &attributes_descriptor) {
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
    // init new leaf data
    LeafData leaf_data(tree_a_leaf);
    std::vector<PointDataIndex32> offsets;
    { // gather data from both trees
      // retrieve attribute handles
      AttributeHandles a_handles(tree_a_leaf);
      AttributeHandles b_handles(tree_b_leaf.getLeaf());
      // retrieve data from voxels and merge into one single array
      for (unsigned int x = 0; x < leaf_dim; ++x)
        for (unsigned int y = 0; y < leaf_dim; ++y)
          for (unsigned int z = 0; z < leaf_dim; ++z) {
            Coord xyz(x, y, z);
            // get tree_a data
            for (auto index = tree_a_leaf->beginIndexVoxel(xyz); index; ++index)
              leaf_data.appendFromIndexAt(*index, a_handles);
            // get tree_b data
            for (auto index = tree_b_leaf->beginIndexVoxel(xyz); index; ++index)
              leaf_data.appendFromIndexAt(*index, b_handles);
            offsets.push_back(leaf_data.dataSize());
          }
    }
    // reset tree_a leaf data with the sum of the sizes
    tree_a_leaf->initializeAttributes(attributes_descriptor.descriptor,
                                      leaf_data.dataSize());
    attributes_descriptor.appendTo(tree_a);
    AttributeWriteHandles a_write_handles(tree_a_leaf);
    // set offsets
    for (Index index = 0; index < voxels_per_leaf; ++index)
      tree_a_leaf->setOffsetOn(index, offsets[index]);
    tree_a_leaf->validateOffsets();
    // save data
    for (auto index_iter = tree_a_leaf->beginIndexOn(); index_iter;
         ++index_iter)
      leaf_data.populate(*index_iter, a_write_handles);
  }
}

} // namespace vdb