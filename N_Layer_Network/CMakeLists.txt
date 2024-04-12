cmake_minimum_required(VERSION 3.27)
project(N_Layer_Network)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -fexperimental-library -stdlib=libc++")

add_executable(N_Layer_Network
        NLayerNetwork.cpp
)

target_include_directories(N_Layer_Network PUBLIC tomlplusplus tomlplusplus/include)

target_compile_options(N_Layer_Network PUBLIC -Og -g3 -glldb -Wall -Wextra -fsanitize=address -fno-omit-frame-pointer)
target_link_options(N_Layer_Network PUBLIC -Og -g3 -glldb -Wall -Wextra -fsanitize=address -fno-omit-frame-pointer)
