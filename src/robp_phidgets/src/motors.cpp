// robp_phidgets
#include <robp_phidgets/motors.hpp>

namespace robp_phidgets
{
Motors::Motors() : Node("motors")
{
	// default open any device
	int serial_num_left  = this->declare_parameter("motors_left_serial", -1);
	int serial_num_right = this->declare_parameter("motors_right_serial", -1);
	// only used if the device is on a VINT hub_port
	int    hub_port_left    = this->declare_parameter("motors_left_hub_port", 4);
	int    hub_port_right   = this->declare_parameter("motors_right_hub_port", 2);
	double acceleration     = this->declare_parameter("motors_acceleration", 100.0);
	double braking_strength = this->declare_parameter("motors_braking_strength", 1.0);
	double current_limit    = this->declare_parameter("motors_current_limit", 2.0);
	double data_rate        = this->declare_parameter("motors_data_rate", 10.0);
	failsafe_time_          = this->declare_parameter("motors_failsafe_timeout_ms", 500);
	bool native_failsafe    = this->declare_parameter("motors_native_failsafe", false);
	std::uint32_t native_failsafe_extra_time =
	    this->declare_parameter("motors_native_failsafe_extra_timeout_ms", 100);

	if (hub_port_left == hub_port_right) {
		RCLCPP_FATAL(this->get_logger(), "Left and right port cannot be the same");
		exit(1);
	}

	pub_ = this->create_publisher<robp_interfaces::msg::DutyCycles>(
	    "motor/current_duty_cycles", 1);

	left_  = std::make_unique<Motor>(this, serial_num_left, hub_port_left, false, 0,
	                                 std::bind(&Motors::publish, this));
	right_ = std::make_unique<Motor>(this, serial_num_right, hub_port_right, false, 0,
	                                 std::bind(&Motors::publish, this));

	left_->setAcceleration(acceleration);
	right_->setAcceleration(acceleration);

	left_->setTargetBrakingStrength(braking_strength);
	right_->setTargetBrakingStrength(braking_strength);

	left_->setCurrentLimit(current_limit);
	right_->setCurrentLimit(current_limit);

	left_->setDataRate(data_rate);
	right_->setDataRate(data_rate);

	failsafe_timer_ = this->create_wall_timer(std::chrono::milliseconds(failsafe_time_),
	                                          std::bind(&Motors::failsafe, this));
	if (native_failsafe) {
		left_->setFailsafe(failsafe_time_ + native_failsafe_extra_time);
		right_->setFailsafe(failsafe_time_ + native_failsafe_extra_time);
	}

	sub_ = this->create_subscription<robp_interfaces::msg::DutyCycles>(
	    "motor/duty_cycles", 1,
	    std::bind(&Motors::dutyCyclesCallback, this, std::placeholders::_1));
}

void Motors::dutyCyclesCallback(robp_interfaces::msg::DutyCycles const& msg)
{
	if (!rclcpp::ok() || !left_ || !right_) {
		return;
	}

	this->failsafe_timer_->reset();
	failsafe_first_ = true;

	if (1 >= std::abs(msg.duty_cycle_left) && 1 >= std::abs(msg.duty_cycle_right)) {
		left_->setTargetVelocity(msg.duty_cycle_left);
		right_->setTargetVelocity(-msg.duty_cycle_right);
	} else {
		RCLCPP_WARN(this->get_logger(),
		            "Duty cycles (%f, %f) is out out of range ([-1, 1], [-1, 1]). Stopping "
		            "motors!",
		            msg.duty_cycle_left, msg.duty_cycle_right);
		left_->setTargetVelocity(0);
		right_->setTargetVelocity(0);
	}
}

void Motors::publish()
{
	if (!rclcpp::ok() || !left_ || !right_ || !left_->hasUpdate() || !right_->hasUpdate()) {
		return;
	}

	auto msg              = std::make_unique<robp_interfaces::msg::DutyCycles>();
	msg->header.stamp     = this->now();
	msg->header.frame_id  = "";
	msg->duty_cycle_left  = left_->velocityUpdate();
	msg->duty_cycle_right = -right_->velocityUpdate();

	pub_->publish(std::move(msg));
}

void Motors::failsafe()
{
	if (!rclcpp::ok() || !left_ || !right_) {
		return;
	}

	if (failsafe_first_) {
		failsafe_first_ = false;
		RCLCPP_WARN(this->get_logger(), "No motor command in over %d ms. Stopping motors!",
		            failsafe_time_);
	}
	left_->setTargetVelocity(0);
	right_->setTargetVelocity(0);
}
}  // namespace robp_phidgets