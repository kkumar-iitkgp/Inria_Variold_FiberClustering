
cmake_minimum_required(VERSION 2.8)

PROJECT(compute_varifold_using_EP_and_MSmeasure)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Find ITK.
FIND_PACKAGE(ITK REQUIRED)
IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ENDIF(ITK_FOUND)

INCLUDE_DIRECTORIES (/usr/include/eigen3)

add_executable(compute_varifold_using_EP_and_MSmeasure compute_varifold_using_EP_and_MSmeasure.cpp)

if(VTK_LIBRARIES)
  target_link_libraries(compute_varifold_using_EP_and_MSmeasure  ${VTK_LIBRARIES} ITKCommon )
else()
  target_link_libraries(compute_varifold_using_EP_and_MSmeasure  vtkHybrid vtkWidgets)
endif()
