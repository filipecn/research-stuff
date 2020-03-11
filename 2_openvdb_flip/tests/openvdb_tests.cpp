#include <catch2/catch.hpp>

#include <openvdb/openvdb.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/tools/LevelSetSphere.h>

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

      points::AttributeArray &array = leaf_iter->attributeArray("P");
      points::AttributeWriteHandle<openvdb::Vec3f> handle(array);

      for (auto index_iter = leaf_iter->beginIndexOn(); index_iter;
           ++index_iter) {
        handle.set(*index_iter, Vec3f(0, 0, 0));
        Coord c;
        index_iter.getCoord(c);
        std::cerr << c << std::endl;
      }
    }
    REQUIRE(point_tree->activeVoxelCount() == 1);
    REQUIRE(voxels_per_leaf == points::pointCount(*point_tree));
  }
}
