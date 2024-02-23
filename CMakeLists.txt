cmake_minimum_required(VERSION 3.27)
project(ATCS_Neural_Networks)

set(CMAKE_CXX_STANDARD 17)

add_executable(ATCS_Neural_Networks main.cpp
        AB1Network/AB1Network.cpp
        ABCNetwork/ABCNetwork.cpp
)

target_compile_options(ATCS_Neural_Networks PUBLIC -Ofast -flto -march=native -mtune=native -Wall -Wextra -fsanitize=address)
target_link_options(ATCS_Neural_Networks PUBLIC -Ofast -flto -march=native -mtune=native -Wall -Wextra -fsanitize=address)