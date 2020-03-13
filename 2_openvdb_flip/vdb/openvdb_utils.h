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

/// \brief
///
/// \param leaf **[in]**
void printLeafPoints(const openvdb::points::PointDataTree::LeafNodeType *leaf);

/// \brief
///
/// \param tree_a **[in/out]**
/// \param tree_b **[in]**
void combine(openvdb::points::PointDataTree *tree_a,
             const openvdb::points::PointDataTree *tree_b);

} // namespace vdb

#endif
