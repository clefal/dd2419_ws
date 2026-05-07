cmake_minimum_required(VERSION 3.5)

# The default build type must be set before project()
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "robp_phidgets: You did not request a specific build type: Choosing 'Release' for maximum performance")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

project(robp_phidgets)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_ros REQUIRED)
find_package(angles REQUIRED)
find_package(libphidget22 REQUIRED)
find_package(phidgets_api REQUIRED)
find_package(rclcpp REQUIRED)
find_package(robp_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

include_directories(include)

add_executable(robp_phidgets 
	src/phidgets.cpp
	src/encoder.cpp
	src/encoders.cpp
	src/motor.cpp
	src/motors.cpp
	src/spatial.cpp
	src/temperature.cpp
)

ament_target_dependencies(robp_phidgets
	angles
	libphidget22
  phidgets_api
  rclcpp
  robp_interfaces
  sensor_msgs
  std_srvs
  tf2
  tf2_geometry_msgs
)

install(TARGETS
	robp_phidgets
  DESTINATION lib/${PROJECT_NAME}
)

# install(DIRECTORY include/
#   DESTINATION include
# )

# install(DIRECTORY launch
#   DESTINATION share/${PROJECT_NAME}
# )

ament_package()