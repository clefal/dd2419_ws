#ifndef ROBP_PHIDGETS_MOTORS_HPP
#define ROBP_PHIDGETS_MOTORS_HPP

// robp_phidgets
#include <robp_phidgets/motor.hpp>

// robp_interfaces
#include <robp_interfaces/msg/duty_cycles.hpp>

// ROS
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/empty.hpp>

// STL
#include <cstdint>
#include <memory>

namespace robp_phidgets
{
class Motors : public rclcpp::Node
{
 public:
	Motors();

 private:
	void dutyCyclesCallback(robp_interfaces::msg::DutyCycles const& msg);

	void publish();

	void failsafe();

 private:
	std::unique_ptr<Motor> left_;
	std::unique_ptr<Motor> right_;

	rclcpp::Publisher<robp_interfaces::msg::DutyCycles>::SharedPtr pub_;

	rclcpp::Subscription<robp_interfaces::msg::DutyCycles>::SharedPtr sub_;

	std::uint32_t                failsafe_time_{};
	rclcpp::TimerBase::SharedPtr failsafe_timer_;
	bool                         failsafe_first_{true};
};
}  // namespace robp_phidgets

#endif  // ROBP_PHIDGETS_MOTORS_HPP