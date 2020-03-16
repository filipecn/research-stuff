#include <catch2/catch.hpp>

#include <openvdb/openvdb.h>
#include <openvdb/points/PointAttribute.h>
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
  SECTION("Attributes descriptor") {
    // create a point tree with multiple attribute types
    // configure position attribute
    points::AttributeSet::DescriptorPtr descriptor(
        points::AttributeSet::Descriptor::create(
            points::TypedAttributeArray<Vec3f,
                                        points::NullCodec>::attributeType()));
    // create tree
    points::PointDataTree point_tree;
    // get leaf
    points::PointDataTree::LeafNodeType *leaf =
        point_tree.touchLeaf(Coord(0, 0, 0));
    // init leaf
    Index voxels_per_leaf = points::PointDataGrid::TreeType::LeafNodeType::SIZE;
    leaf->initializeAttributes(descriptor, voxels_per_leaf);
    // add Vec3d type attribute representing velocity
    points::appendAttribute<Vec3d>(point_tree, "V");
    // add float type attribute representing radius
    points::appendAttribute<float>(point_tree, "R");
    // check attributes
    REQUIRE(leaf->attributeSet().descriptor().find("P") == 0);
    REQUIRE(leaf->attributeSet().descriptor().find("V") == 1);
    REQUIRE(leaf->attributeSet().descriptor().find("R") == 2);
    REQUIRE(leaf->attributeSet().get(0)->valueTypeIsVector());
    REQUIRE(leaf->attributeSet().get(1)->valueTypeIsVector());
    REQUIRE(leaf->attributeSet().get(2)->valueTypeIsFloatingPoint());
    // set leaf offsets
    for (Index index = 0; index < voxels_per_leaf; ++index)
      leaf->setOffsetOn(index, index + 1);
    leaf->validateOffsets();
    { // set values
      points::AttributeWriteHandle<Vec3f> p_handle(leaf->attributeArray("P"));
      points::AttributeWriteHandle<Vec3d> v_handle(leaf->attributeArray("V"));
      points::AttributeWriteHandle<float> r_handle(leaf->attributeArray("R"));
      for (auto index = leaf->beginIndexOn(); index; ++index) {
        p_handle.set(*index, Vec3f(0, 0.001 * *index, 0.2));
        v_handle.set(*index, Vec3d(0.001 * *index, 0, 0.2));
        r_handle.set(*index, *index);
      }
    }
    // vdb::printLeafPoints(leaf);
  }
}

TEST_CASE("AttributesDescriptor") {
  SECTION("Constructor") {
    vdb::AttributesDescriptor ad(points::AttributeSet::Descriptor::create(
        points::TypedAttributeArray<Vec3f,
                                    points::NullCodec>::attributeType()));
    REQUIRE(ad.descriptor->size() == 1);
    vdb::AttributesDescriptor ad2;
    REQUIRE(ad2.descriptor->size() == 1);
  }
  SECTION("add") {
    vdb::AttributesDescriptor ad;
    ad.add<float>({"A", "B", "C"});
    ad.add<Vec3d>({"D", "E"});
    ad.add<Vec3f>({"F"});
    REQUIRE(ad.size<float>() == 3);
    REQUIRE(ad.size<Vec3d>() == 2);
    REQUIRE(ad.size<Vec3f>() == 1);
  }
  SECTION("populate") {
    vdb::AttributesDescriptor ad;
    ad.add<float>({"R"});
    ad.add<Vec3f>({"V"});
    ad.add<Vec3d>({"X"});
    // create tree
    points::PointDataTree::Ptr point_tree(new points::PointDataTree());
    // get leaf
    auto *leaf = point_tree->touchLeaf(Coord(0, 0, 0));
    // init leaf
    Index voxels_per_leaf = points::PointDataGrid::TreeType::LeafNodeType::SIZE;
    leaf->initializeAttributes(ad.descriptor, voxels_per_leaf);
    for (Index index = 0; index < voxels_per_leaf; ++index)
      leaf->setOffsetOn(index, index + 1);
    leaf->validateOffsets();
    ad.appendTo(point_tree);
    { // set values
      points::AttributeWriteHandle<Vec3f> p_handle(leaf->attributeArray("P"));
      points::AttributeWriteHandle<Vec3d> x_handle(leaf->attributeArray("X"));
      points::AttributeWriteHandle<Vec3f> v_handle(leaf->attributeArray("V"));
      points::AttributeWriteHandle<float> r_handle(leaf->attributeArray("R"));
      for (auto index = leaf->beginIndexOn(); index; ++index) {
        p_handle.set(*index, Vec3f(0, 0.001 * *index, 0.2));
        x_handle.set(*index, Vec3f(0.3, 0.001 * *index, 0.2));
        v_handle.set(*index, Vec3d(0.001 * *index, 0, 0.2));
        r_handle.set(*index, *index);
      }
    }
    // vdb::printLeafPoints(leaf);
  }
}

TEST_CASE("Combine", "[utils]") {
  SECTION("same topology") {
    // define common information
    using PositionAttribute =
        points::TypedAttributeArray<Vec3f, points::NullCodec>;
    NamePair position_type = PositionAttribute::attributeType();
    vdb::AttributesDescriptor attributes_desc(
        points::AttributeSet::Descriptor::create(position_type));
    Index voxels_per_leaf = points::PointDataGrid::TreeType::LeafNodeType::SIZE;
    // create first tree
    points::PointDataTree::Ptr point_tree1(new points::PointDataTree());
    {
      auto leaf = point_tree1->touchLeaf(Coord(0, 0, 0));
      leaf->initializeAttributes(attributes_desc.descriptor, voxels_per_leaf);
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
      leaf->initializeAttributes(attributes_desc.descriptor, voxels_per_leaf);
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
    vdb::combine(point_tree1, point_tree2, attributes_desc);
    // auto tree1_leaf = point_tree1->touchLeaf(Coord(0, 0, 0));
    // vdb::printLeafPoints(tree1_leaf);
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
    vdb::AttributesDescriptor attributes_desc(
        points::AttributeSet::Descriptor::create(position_type));
    Index voxels_per_leaf = points::PointDataGrid::TreeType::LeafNodeType::SIZE;
    // create first tree
    points::PointDataTree::Ptr point_tree1(new points::PointDataTree());
    {
      auto leaf = point_tree1->touchLeaf(Coord(0, 0, 0));
      leaf->initializeAttributes(attributes_desc.descriptor,
                                 3 * (voxels_per_leaf / 2));
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
      leaf->initializeAttributes(attributes_desc.descriptor,
                                 2 * (voxels_per_leaf / 2));
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
    vdb::combine(point_tree1, point_tree2, attributes_desc);
    // auto tree1_leaf = point_tree1->touchLeaf(Coord(0, 0, 0));
    // vdb::printLeafPoints(tree1_leaf);
  }
  SECTION("multiple attributes") {
    // init leaf
    Index voxels_per_leaf = points::PointDataGrid::TreeType::LeafNodeType::SIZE;
    // set common descriptor
    vdb::AttributesDescriptor attributes_descriptor(
        points::AttributeSet::Descriptor::create(
            points::TypedAttributeArray<Vec3f,
                                        points::NullCodec>::attributeType()));
    // add Vec3d type attribute representing velocity
    attributes_descriptor.add<float>({"R"});
    // add float type attribute representing radius
    attributes_descriptor.add<Vec3d>({"V"});
    // create first tree
    points::PointDataTree::Ptr point_tree_a(new points::PointDataTree());
    // get leaf from first tree
    points::PointDataTree::LeafNodeType *leaf_a =
        point_tree_a->touchLeaf(Coord(0, 0, 0));
    leaf_a->initializeAttributes(attributes_descriptor.descriptor,
                                 voxels_per_leaf);
    attributes_descriptor.appendTo(point_tree_a);
    // create first tree
    points::PointDataTree::Ptr point_tree_b(new points::PointDataTree());
    // get leaf from first tree
    points::PointDataTree::LeafNodeType *leaf_b =
        point_tree_b->touchLeaf(Coord(0, 0, 0));
    leaf_b->initializeAttributes(attributes_descriptor.descriptor,
                                 voxels_per_leaf * 2);
    attributes_descriptor.appendTo(point_tree_b);
    { // populate tree a with 1 point per voxel
      // set leaf offsets
      for (Index index = 0; index < voxels_per_leaf; ++index)
        leaf_a->setOffsetOn(index, index + 1);
      leaf_a->validateOffsets();
      { // set values
        points::AttributeWriteHandle<Vec3f> p_handle(
            leaf_a->attributeArray("P"));
        points::AttributeWriteHandle<Vec3d> v_handle(
            leaf_a->attributeArray("V"));
        points::AttributeWriteHandle<float> r_handle(
            leaf_a->attributeArray("R"));
        for (auto index = leaf_a->beginIndexOn(); index; ++index) {
          p_handle.set(*index, Vec3f(0, 0.001 * *index, 0.1));
          v_handle.set(*index, Vec3d(0.001 * *index, 0, 0.1));
          r_handle.set(*index, *index);
        }
      }
    }
    { // populate tree b with 2 points per voxel
      // set leaf offsets
      Index offset(0);
      for (Index index = 0; index < voxels_per_leaf; ++index) {
        offset += 2;
        leaf_b->setOffsetOn(index, offset);
      }
      leaf_b->validateOffsets();
      { // set values
        points::AttributeWriteHandle<Vec3f> p_handle(
            leaf_b->attributeArray("P"));
        points::AttributeWriteHandle<Vec3d> v_handle(
            leaf_b->attributeArray("V"));
        points::AttributeWriteHandle<float> r_handle(
            leaf_b->attributeArray("R"));
        for (auto index = leaf_b->beginIndexOn(); index; ++index) {
          p_handle.set(*index, Vec3f(0, 0.001 * *index, 0.2));
          v_handle.set(*index, Vec3d(0.001 * *index, 0, 0.2));
          r_handle.set(*index, *index);
        }
      }
    }
    vdb::combine(point_tree_a, point_tree_b, attributes_descriptor);
    // vdb::printLeafPoints(leaf_a);
    REQUIRE(voxels_per_leaf * 3 == points::pointCount(*point_tree_a));
  }
}
