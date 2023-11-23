cmake_minimum_required(VERSION 3.05)
project(DlnneImpl)
set(Target dlnneimpl)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/../install CACHE PATH ${CMAKE_SOURCE_DIR}/../install FORCE)
endif()

# 指定h文件
file(GLOB dlnne_headers ${CMAKE_SOURCE_DIR}/*.h)

# 指定cpp文件
file(GLOB_RECURSE dlnne_srcs ${CMAKE_SOURCE_DIR}/*.cc)

# 添加头文件目录和链接目录
set(SDK_DIR $ENV{DLICC_PATH}/../)
include_directories(${SDK_DIR}/include)
include_directories(${SDK_DIR}/include/dlnne)
include_directories(${CMAKE_SOURCE_DIR}/../)
include_directories(${CMAKE_SOURCE_DIR}/../install/include/dlutil)
link_directories(${SDK_DIR}/lib ${CMAKE_SOURCE_DIR}/../install/lib)

# 根据需要设置相应的编译选项
set(CXX_FLAGS -std=c++11 -fPIC -fpermissive -Wno-attributes)
# 兼容老版本gcc
add_compile_options(-std=c++11)
add_library(${Target} SHARED ${dlnne_srcs})
set_target_properties(${Target} PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${Target} PRIVATE curt dlnne dlutil)
target_compile_options(${Target} PRIVATE -I${SDK_DIR}/include -I${CMAKE_BINARY_DIR} ${CXX_FLAGS})
install(FILES ${dlnne_headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/dlnne_impl)
install(TARGETS ${Target} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)