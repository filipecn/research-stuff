PROJECT(SPGRID_FLIP)

set(CMAKE_CXX_COMPILER /opt/intel/bin/icc)

set(SOURCES
        Blocked_Copy_Helper.cpp
        Laplace_Helper.cpp
        PTHREAD_QUEUE.cpp
        SPGrid/Core/SPGrid_Utilities.cpp
        )

add_executable(spgrid_demo main_original.cpp ${SOURCES})
target_link_libraries(spgrid_demo PUBLIC pthread)
target_include_directories(spgrid_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(spgrid_flip main.cpp ${SOURCES})
target_link_libraries(spgrid_flip PUBLIC pthread)
target_include_directories(spgrid_flip PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
