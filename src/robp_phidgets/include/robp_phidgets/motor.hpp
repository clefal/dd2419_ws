#ifndef ROBP_PHIDGETS_MOTOR_HPP
#define ROBP_PHIDGETS_MOTOR_HPP

// Phidget
#include <libphidget22/phidget22.h>

// STL
#include <functional>
#include <mutex>

// ROS
#include <rclcpp/node.hpp>

namespace robp_phidgets
{
class Motor
{
 public:
	Motor(rclcpp::Node *node, int32_t serial_number, int hub_port, bool is_hub_port_device,
	      int channel, std::function<void()> callback);

	~Motor();

	double acceleration() const;

	void setAcceleration(double acceleration);

	double targetBrakingStrength() const;

	void setTargetBrakingStrength(double braking);

	double brakingStrength() const;

	double currentLimit() const;

	void setCurrentLimit(double limit);

	double dataRate() const;

	void setDataRate(double rate);

	void setFailsafe(uint32_t time_ms);

	void resetFailsafe();

	double targetVelocity() const;

	void setTargetVelocity(double velocity);

	double velocity() const;

	int port() const;

	double velocityUpdate() const;

	bool hasUpdate() const;

 private:
	void create();

	void close();

	void assignEventHandlers();

	void init();

	static void velocityUpdateCallback(PhidgetDCMotorHandle ch, void *ctx, double velocity);

	static void attachCallback(PhidgetHandle ch, void *ctx);

	static void detachCallback(PhidgetHandle ch, void *ctx);

	static void errorCallback(PhidgetHandle ch, void *ctx, Phidget_ErrorEventCode code,
	                          char const *description);

 private:
	rclcpp::Node *node_;

	PhidgetDCMotorHandle motor_;

	std::function<void()> callback_;

	int hub_port_;

	double   acceleration_{-1};
	double   target_braking_strength_{-1};
	double   current_limit_{-1};
	double   data_rate_{-1};
	uint32_t failsafe_time_{0};

	mutable std::mutex m_;
	mutable double     last_velocity_{};
	mutable bool       update_{};
};
}  // namespace robp_phidgets

#endif  // ROBP_PHIDGETS_MOTOR_HPP