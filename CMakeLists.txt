cmake_minimum_required(VERSION 3.05)
project(DlnneTest)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install CACHE PATH ${CMAKE_SOURCE_DIR}/install FORCE)
endif()

find_package(OpenCV REQUIRED)

set(DLDEMO_LIB_DIR ${CMAKE_SOURCE_DIR}/install/lib)

# 指定头文件目录
set(SDK_DIR $ENV{DLICC_PATH}/../)
include_directories(${SDK_DIR}/include)
include_directories(${SDK_DIR}/include/dlnne)
include_directories(${CMAKE_SOURCE_DIR}/dlutil)
include_directories(${CMAKE_SOURCE_DIR}/dlnne_impl)
include_directories(${CMAKE_SOURCE_DIR}/install/include/dlpreprocess)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${DLDEMO_LIB_DIR} ${SDK_DIR}/lib)

# 根据需要设置相应的编译选项
set(CXX_FLAGS -std=c++11 -fpermissive -Wno-attributes)
# 兼容老版本gcc
add_compile_options(-std=c++11)

file(GLOB impl_files ${CMAKE_SOURCE_DIR}/dlnne_impl/*.cc)
file(GLOB util_files ${CMAKE_SOURCE_DIR}/dlutil/*.cc)
file(GLOB src_tests ${CMAKE_SOURCE_DIR}/test/*.cc)

file(GLOB dlnne_headers ${CMAKE_SOURCE_DIR}/dlnne_impl/*.h)
file(GLOB dlutil_headers ${CMAKE_SOURCE_DIR}/dlutil/*.h)

foreach(test ${src_tests})
    string(REGEX REPLACE ".+/(.+)\\..*" "\\1" exe_name ${test})
    add_executable(${exe_name} ${test} ${util_files} ${impl_files} )
    target_compile_options(${exe_name} PRIVATE -I${SDK_DIR}/include ${CXX_FLAGS})
    target_link_libraries(${exe_name} PRIVATE curt pthread dlnne dlpreprocess ${OpenCV_LIBS}
                          ${DLDEMO_LIB} "-Wl,--disable-new-dtags")
    install(TARGETS ${exe_name} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endforeach()

install(FILES ${dlnne_headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/dlnne_impl)
install(FILES ${dlutil_headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/dlutil)