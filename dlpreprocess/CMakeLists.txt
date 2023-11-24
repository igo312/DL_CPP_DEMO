


cmake_minimum_required(VERSION 3.2.0)

project(Dlpreprocess)
set(Target dlpreprocess)

set(CMAKE_CXX_COMPILER "dlcc")
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/../install CACHE PATH ${CMAKE_SOURCE_DIR}/../install FORCE)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

# 指定h文件
file(GLOB header_files ${CMAKE_SOURCE_DIR}/include/*.h)

# 指定cu文件
file(GLOB_RECURSE cc_files ${CMAKE_SOURCE_DIR}/src/*.cc)

# 根据需要设置相应的编译选项
if (DEFINED_ENV{DLI_V2})
    set(DLI_V2 $ENV{DLI_V2})
endif()

if (DLI_V2)
    message(STATUS "Using DLI_V2")
    add_definitions(-DDLI_V2)
    set(CXX_FLAGS -std=c++11 -fPIC -fpermissive --cuda-gpu-arch=dlgput64 -x cuda)
else()
message(STATUS "Using DLI_V1")
    set(CXX_FLAGS -std=c++11 -fPIC -fpermissive --cuda-gpu-arch=dlgpuc64 -x cuda)
endif()

add_library(${Target} SHARED ${cc_files})
set_target_properties(${Target} PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${Target} PRIVATE curt)
target_compile_options(${Target} PRIVATE ${CXX_FLAGS} -Wno-attributes -Wno-c++11-narrowing -Wdouble-promotion)
install(FILES ${header_files} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/dlpreprocess)
install(TARGETS ${Target} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
