set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/OpenVDB" ${CMAKE_MODULE_PATH})

find_package(OpenVDB REQUIRED)

set(VDBLIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(VDBLIB_HEADERS 
    vdb/injector.h
    vdb/io.h
    vdb/particle_system.h
    vdb/openvdb_utils.h
    vdb/velocity_grid.h
)
set(VDBLIB_SOURCES 
    vdb/injector.cpp
    vdb/io.cpp
    vdb/particle_system.cpp
    vdb/openvdb_utils.cpp
    vdb/velocity_grid.cpp
)
add_library(vdblib STATIC ${VDBLIB_SOURCES} ${VDBLIB_HEADERS})
set_target_properties(vdblib PROPERTIES
        OUTPUT_NAME "vdblib"
        FOLDER "VDBLIB")
target_include_directories(vdblib PUBLIC 
    ${OpenVDB_INCLUDE_DIRS} 
    ${PONOS_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(vdblib 
    ${PONOS_LIBRARIES}
    OpenVDB::openvdb 
)
add_dependencies(vdblib ponos)


add_executable(main main.cpp)
target_include_directories(main PUBLIC ${OpenVDB_INCLUDE_DIRS})
target_link_libraries(main OpenVDB::openvdb)

add_subdirectory(tests)
