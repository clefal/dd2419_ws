// robp_phidgets
#include <robp_phidgets/encoders.hpp>
#include <robp_phidgets/motors.hpp>
#include <robp_phidgets/spatial.hpp>
#include <robp_phidgets/temperature.hpp>

// ROS
#include <rclcpp/rclcpp.hpp>

int main(int argc, char* argv[])
{
	rclcpp::init(argc, argv);

	auto spatial     = std::make_shared<robp_phidgets::Spatial>();
	auto temperature = std::make_shared<robp_phidgets::Temperature>();
	auto encoders    = std::make_shared<robp_phidgets::Encoders>();
	auto motors      = std::make_shared<robp_phidgets::Motors>();

	rclcpp::executors::StaticSingleThreadedExecutor executor;
	executor.add_node(spatial);
	executor.add_node(temperature);
	executor.add_node(encoders);
	executor.add_node(motors);

	executor.spin();

	// rclcpp::spin(motors);

	rclcpp::shutdown();

	return 0;
}