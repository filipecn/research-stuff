//#####################################################################
// Copyright (c) 2014, the authors of submission papers_0203
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <SPGrid/Core/SPGrid_Allocator.h>
#include <SPGrid/Core/SPGrid_Set.h>
#include "Blocked_Copy_Helper.h"
#include "Laplace_Helper.h"
#include <SPGrid/Data_Structures/std_array.h>

#include "PTHREAD_QUEUE.h"
#include "ADAPTIVE_SPHERE_RASTERIZER.h"
#include "DENSE_CUBE_RASTERIZER.h"
#include "FACE_INITIALIZER.h"
#include "GEOMETRY_BLOCK.h"
#include "HIERARCHICAL_RASTERIZER.h"

//#define LOOP_AT_END
//#define BLOCKED_COPY
//#define DENSE_CUBE

extern PTHREAD_QUEUE* pthread_queue;
using namespace SPGrid;

typedef float T;
typedef struct Foo_struct {
    T x,y,z;
    unsigned flags;
} Foo;
typedef SPGrid_Allocator<Foo,3> Foo_Allocator;
typedef SPGrid_Allocator<Foo,3>::Array<>::mask Foo_Mask;
typedef SPGrid_Allocator<Foo,3>::Array<T>::type Data_array_type;
typedef SPGrid_Allocator<Foo,3>::Array<const T>::type Const_data_array_type;
typedef SPGrid_Allocator<Foo,3>::Array<unsigned>::type Flags_array_type;
typedef SPGrid_Set<Flags_array_type> Flags_set_type;
typedef std_array<int,3> Vec3i;
typedef std_array<float,3> Vec3f;

int main(int argc,char* argv[]) {

    if (argc != 3) {
        printf("Please specify size (power of two), and number of threads\n");
        exit(1);
    }
    int size = atoi(argv[1]);
    if ((size & (size-1)) != 0) {
        printf("For this limited demo, size must be a power of two.\n");
        exit(1);
    }
    int n_threads = atoi(argv[2]);
    pthread_queue = new PTHREAD_QUEUE(n_threads);

    Foo_Allocator allocator(size,size,size);
    Data_array_type d1 = allocator.Get_Array(&Foo::x);
    Const_data_array_type d2 = allocator.Get_Const_Array(&Foo::y);
    Const_data_array_type d3 = allocator.Get_Const_Array(&Foo::z);
    Flags_array_type flags = allocator.Get_Array(&Foo::flags);
    Flags_set_type flag_set(flags);
    
    Vec3i imin(0);
    Vec3i imax(size);
    Vec3f Xmin(0.f);
    Vec3f Xmax(1.f);
    Vec3f center(.5f,.5f,.5f);
    float inner_radius=.3f;
    float outer_radius=.31f;

    int active_cells = 0;
#ifdef DENSE_CUBE
    if (size > 1024) {
        std::cout << "DENSE_CUBE mode has no memory savings, a size of " << size
                  << " will allocate roughly " 
                  << ((size>>10)*(size>>10)*(size>>10)*sizeof(Foo))
                  << "GB\nFeel free to remove this warning if needed.\n";
        exit(1);
    }
    std::cout << "Flagging active cells (in a dense cube)...";
    active_cells = DENSE_CUBE_RASTERIZER<Flags_set_type>::Rasterize(flag_set, size);
#else
    if (size < 256) {
        std::cout << "This is a sparse configuration, crank that size up!\n";
    }
    std::cout << "Flagging active cells (on narrow band)...";
    GEOMETRY_BLOCK block(imin,imax,Xmin,Xmax);
    ADAPTIVE_SPHERE_RASTERIZER<Flags_set_type> adaptive_sphere(flag_set,center,inner_radius,outer_radius);
    HIERARCHICAL_RASTERIZER<ADAPTIVE_SPHERE_RASTERIZER<Flags_set_type> > rasterizer(adaptive_sphere);
    rasterizer.Iterate(block);
    active_cells = adaptive_sphere.total_active;
    // Narrow band init
#endif
    std::cout << "done.\n";
    uint64_t bigsize = size;
    std::cout << "Activated " << active_cells << " cells, out of a possible "
              << bigsize*bigsize*bigsize << "\n\n";
    
    flag_set.Refresh_Block_Offsets();
    // Face flag initialization
    FACE_INITIALIZER<Foo, 3>::Flag_Active_Faces(flag_set);
    printf("Finished flagging active cell faces.\n");

    T c = 0.1f;

#ifdef BLOCKED_COPY
    // Perform parallel x = y + (c * z)
    Blocked_Copy_Helper<T, Data_array_type::MASK::elements_per_block> helper(
        (T*)d1.Get_Data_Ptr(),
        (T*)d2.Get_Data_Ptr(),
        c,
        (T*)d3.Get_Data_Ptr(),
        (unsigned*)flags.Get_Data_Ptr(),
        flag_set.Get_Blocks().first,
        flag_set.Get_Blocks().second);
    helper.Run_Parallel(n_threads);
    printf("Finished running SAXPY kernel.\n");
#else
    Laplace_Helper<T,NextLogTwo<sizeof(Foo)>::value,3> helper(
        (T*)d1.Get_Data_Ptr(),
        (T*)d3.Get_Data_Ptr(),
        (unsigned*)flags.Get_Data_Ptr(),
        flag_set.Get_Blocks().first,
        flag_set.Get_Blocks().second,
        1,
        2./3.);
    helper.Run_Parallel(n_threads);
    printf("Finished running Laplace kernel.\n");
#endif
       
#ifdef LOOP_AT_END
    std::cout << "Looping forever, check out my memory usage!\n";
    while(1) {
        usleep(1000);
    }
#endif
} 

