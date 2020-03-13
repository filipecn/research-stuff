#include <catch2/catch.hpp>

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <vdb/openvdb_utils.h>

using namespace openvdb;

TEST_CASE("OpenVDB Sanity checks", "[openvdb]") {
  initialize();
  SECTION("PointDataTree") {
    points::PointDataTree::Ptr point_tree(new points::PointDataTree());
    point_tree->setActiveState(math::Coord(0, 0, 0), true);
    REQUIRE(point_tree->activeVoxelCount() == 1);
    REQUIRE(point_tree->activeLeafVoxelCount() == 1);
    REQUIRE(point_tree->leafCount() == 1);
    // Define the position type and codec using fixed-point 16-bit compression.
    using PositionAttribute =
        points::TypedAttributeArray<Vec3f, points::FixedPointCodec<false>>;
    NamePair position_type = PositionAttribute::attributeType();
    // Create a new Attribute Descriptor with position only
    points::AttributeSet::Descriptor::Ptr descriptor(
        points::AttributeSet::Descriptor::create(position_type));
    Index voxels_per_leaf = points::PointDataGrid::TreeType::LeafNodeType::SIZE;
    // Iterate over the leaf nodes in the point tree.
    for (auto leaf_iter = point_tree->beginLeaf(); leaf_iter; ++leaf_iter) {
      // Initialize the attributes using the descriptor and point count.
      leaf_iter->initializeAttributes(descriptor, voxels_per_leaf);
      Index offset(0);
      for (Index index = 0; index < voxels_per_leaf; ++index) {
        if (index < 3 || index > 500)
          offset += 1;
        leaf_iter->setOffsetOn(index, offset);
      }
    }
    REQUIRE(point_tree->activeVoxelCount() == 512);
    REQUIRE(14 == points::pointCount(*point_tree));
  }
  SECTION("PointDataGrid") {
    points::PointDataTree::Ptr point_tree(new points::PointDataTree());
    point_tree->setActiveState(math::Coord(0, 0, 0), true);
    // Create a new Attribute Descriptor with position only
    points::AttributeSet::Descriptor::Ptr descriptor(
        points::AttributeSet::Descriptor::create(
            points::TypedAttributeArray<Vec3f,
                                        points::NullCodec>::attributeType()));
    Index voxels_per_leaf = points::PointDataGrid::TreeType::LeafNodeType::SIZE;
    // Iterate over the leaf nodes in the point tree.
    for (auto leaf_iter = point_tree->beginLeaf(); leaf_iter; ++leaf_iter) {
      // Initialize the attributes using the descriptor and point count.
      leaf_iter->initializeAttributes(descriptor, voxels_per_leaf);
      Index offset(0);
      for (Index index = 0; index < voxels_per_leaf; ++index) {
        offset += 1;
        leaf_iter->setOffsetOn(index, offset);
      }

      points::AttributeArray &array = leaf_iter->attributeArray("P");
      points::AttributeWriteHandle<openvdb::Vec3f> handle(array);

      for (auto index_iter = leaf_iter->beginIndexOn(); index_iter;
           ++index_iter)
        handle.set(*index_iter, Vec3f(0, 0, 0));
    }
    points::PointDataGrid::Ptr points =
        points::PointDataGrid::create(point_tree);
    points->setTransform(math::Transform::createLinearTransform(0.1));
    for (auto leaf_iter = point_tree->beginLeaf(); leaf_iter; ++leaf_iter) {
      points::AttributeArray &array = leaf_iter->attributeArray("P");
      points::AttributeWriteHandle<openvdb::Vec3f> handle(array);
      for (auto index_iter = leaf_iter->beginIndexOn(); index_iter;
           ++index_iter) {
        auto ic = handle.get(*index_iter) + index_iter.getCoord().asVec3d();
        auto wp = points->transform().indexToWorld(ic);
        REQUIRE(wp.x() == Approx(ic.x() / 10).margin(1e-8));
        REQUIRE(wp.y() == Approx(ic.y() / 10).margin(1e-8));
        REQUIRE(wp.z() == Approx(ic.z() / 10).margin(1e-8));
      }
    }
  }
}

TEST_CASE("Combine", "[utils]") {
  SECTION("same topology") {
    // define common information
    using PositionAttribute =
        points::TypedAttributeArray<Vec3f, points::NullCodec>;
    NamePair position_type = PositionAttribute::attributeType();
    points::AttributeSet::Descriptor::Ptr descriptor(
        points::AttributeSet::Descriptor::create(position_type));
    Index voxels_per_leaf = points::PointDataGrid::TreeType::LeafNodeType::SIZE;
    // create first tree
    points::PointDataTree::Ptr point_tree1(new points::PointDataTree());
    {
      auto leaf = point_tree1->touchLeaf(Coord(0, 0, 0));
      leaf->initializeAttributes(descriptor, voxels_per_leaf);
      Index offset(0);
      for (Index index = 0; index < voxels_per_leaf; ++index) {
        offset += 1;
        leaf->setOffsetOn(index, offset);
      }
      points::AttributeArray &array = leaf->attributeArray("P");
      points::AttributeWriteHandle<openvdb::Vec3f> handle(array);
      for (auto index_iter = leaf->beginIndexOn(); index_iter; ++index_iter)
        handle.set(*index_iter, Vec3f(0, 0, 0));
      REQUIRE(512 == points::pointCount(*point_tree1));
    }
    // create second tree
    points::PointDataTree::Ptr point_tree2(new points::PointDataTree());
    {
      auto leaf = point_tree2->touchLeaf(Coord(0, 0, 0));
      leaf->initializeAttributes(descriptor, voxels_per_leaf);
      Index offset(0);
      for (Index index = 0; index < voxels_per_leaf; ++index) {
        offset += 1;
        leaf->setOffsetOn(index, offset);
      }
      points::AttributeArray &array = leaf->attributeArray("P");
      points::AttributeWriteHandle<openvdb::Vec3f> handle(array);
      for (auto index_iter = leaf->beginIndexOn(); index_iter; ++index_iter)
        handle.set(*index_iter, Vec3f(0.2, 0.2, 0.2));
      REQUIRE(512 == points::pointCount(*point_tree2));
    }
    vdb::combine(point_tree1.get(), point_tree2.get());
    auto tree1_leaf = point_tree1->touchLeaf(Coord(0, 0, 0));
    vdb::printLeafPoints(tree1_leaf);
    REQUIRE(1024 == points::pointCount(*point_tree1));
  }
  SECTION("alternating voxels") {
    // tree_a: even voxels (3 points per voxel)
    // tree_b: odd voxels (2 points per voxel)
    // at the end, tree_a must have all voxels filled

    // define common information
    using PositionAttribute =
        points::TypedAttributeArray<Vec3f, points::NullCodec>;
    NamePair position_type = PositionAttribute::attributeType();
    points::AttributeSet::Descriptor::Ptr descriptor(
        points::AttributeSet::Descriptor::create(position_type));
    Index voxels_per_leaf = points::PointDataGrid::TreeType::LeafNodeType::SIZE;
    // create first tree
    points::PointDataTree::Ptr point_tree1(new points::PointDataTree());
    {
      auto leaf = point_tree1->touchLeaf(Coord(0, 0, 0));
      leaf->initializeAttributes(descriptor, 3 * (voxels_per_leaf / 2));
      Index offset(0);
      for (Index index = 0; index < voxels_per_leaf; ++index) {
        if (index % 2 == 0)
          offset += 3;
        leaf->setOffsetOn(index, offset);
      }
      leaf->validateOffsets();
      points::AttributeArray &array = leaf->attributeArray("P");
      points::AttributeWriteHandle<Vec3f> handle(array);
      for (auto index_iter = leaf->beginIndexOn(); index_iter; ++index_iter)
        handle.set(*index_iter, Vec3f(0.1 * (*index_iter % 3 + 1), 0, 0));
    }
    // create first tree
    points::PointDataTree::Ptr point_tree2(new points::PointDataTree());
    {
      auto leaf = point_tree2->touchLeaf(Coord(0, 0, 0));
      leaf->initializeAttributes(descriptor, 2 * (voxels_per_leaf / 2));
      Index offset(0);
      for (Index index = 0; index < voxels_per_leaf; ++index) {
        if (index % 2)
          offset += 2;
        leaf->setOffsetOn(index, offset);
      }
      leaf->validateOffsets();
      points::AttributeArray &array = leaf->attributeArray("P");
      points::AttributeWriteHandle<Vec3f> handle(array);
      for (auto index_iter = leaf->beginIndexOn(); index_iter; ++index_iter)
        handle.set(*index_iter, Vec3f(-0.1 * (*index_iter % 2 + 1), 0, 0));
    }
    vdb::combine(point_tree1.get(), point_tree2.get());
    auto tree1_leaf = point_tree1->touchLeaf(Coord(0, 0, 0));
    vdb::printLeafPoints(tree1_leaf);
  }
}
