cmake_minimum_required(VERSION 3.27)
project(ATCS_Neural_Networks)

set(CMAKE_CXX_STANDARD 17)

add_executable(ATCS_Neural_Networks
        ABCBackpropNetwork/ABCBackPropagation.cpp
)

target_include_directories(ATCS_Neural_Networks PUBLIC ABCBackpropNetwork ABCBackpropNetwork/tomlplusplus/include)

target_compile_options(ATCS_Neural_Networks PUBLIC -Og -g3 -glldb -Wall -Wextra -fsanitize=address -fno-omit-frame-pointer)
target_link_options(ATCS_Neural_Networks PUBLIC -Og -g3 -glldb -Wall -Wextra -fsanitize=address -fno-omit-frame-pointer)