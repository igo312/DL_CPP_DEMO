cmake_minimum_required(VERSION 3.05)
project(DlUtil)
set(Target dlutil)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/../install CACHE PATH ${CMAKE_SOURCE_DIR}/../install FORCE)
endif()

# 指定h文件
file(GLOB dlutil_headers ${CMAKE_SOURCE_DIR}/*.h)

# 指定cpp文件
file(GLOB_RECURSE dlutil_srcs ${CMAKE_SOURCE_DIR}/*.cc)

include_directories(${CMAKE_SOURCE_DIR}/../)

# 根据需要设置相应的编译选项
set(CXX_FLAGS -std=c++11 -fPIC -fpermissive -Wno-attributes)
# 兼容老版本gcc
add_compile_options(-std=c++11)
add_library(${Target} SHARED ${dlutil_srcs})
set_target_properties(${Target} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(${Target} PRIVATE  -I${CMAKE_BINARY_DIR} ${CXX_FLAGS})
install(FILES ${dlutil_headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/dlutil)
install(TARGETS ${Target} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)