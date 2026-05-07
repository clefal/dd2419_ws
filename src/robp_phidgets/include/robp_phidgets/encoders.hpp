#ifndef ROBP_PHIDGETS_ENCODERS_HPP
#define ROBP_PHIDGETS_ENCODERS_HPP

// robp_phidgets
#include <robp_phidgets/encoder.hpp>

// robp_interfaces
#include <robp_interfaces/msg/encoders.hpp>

// ROS
#include <rclcpp/rclcpp.hpp>

// STL
#include <memory>

namespace robp_phidgets
{
class Encoders : public rclcpp::Node
{
 public:
	Encoders();

 private:
	void publish();

 private:
	std::unique_ptr<Encoder> left_;
	std::unique_ptr<Encoder> right_;

	rclcpp::Publisher<robp_interfaces::msg::Encoders>::SharedPtr pub_;
};
}  // namespace robp_phidgets

#endif  // ROBP_PHIDGETS_ENCODERS_HPP