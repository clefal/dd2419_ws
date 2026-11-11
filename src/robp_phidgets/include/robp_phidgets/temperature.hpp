#ifndef ROBP_PHIDGETS_TEMPERATURE_HPP
#define ROBP_PHIDGETS_TEMPERATURE_HPP

// Phidget
#include <libphidget22/phidget22.h>

// ROS
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/temperature.hpp>

// STL
#include <limits>

namespace robp_phidgets
{
class Temperature : public rclcpp::Node
{
 public:
	Temperature();

	~Temperature();

	double dataRate() const;

	void setDataRate(double rate);

	double temperature() const;

	double temperatureChangeTrigger() const;

	void setTemperatureChangeTrigger(double change);

	int port() const;

 private:
	void create();

	void close();

	void assignEventHandlers();

	void init();

	void publish(double temperature);

	static void temperatureChangeCallback(PhidgetTemperatureSensorHandle ch, void *ctx,
	                                      double temperature);

	static void attachCallback(PhidgetHandle ch, void *ctx);

	static void detachCallback(PhidgetHandle ch, void *ctx);

	static void errorCallback(PhidgetHandle ch, void *ctx, Phidget_ErrorEventCode code,
	                          char const *description);

 private:
	PhidgetTemperatureSensorHandle temperature_;

	rclcpp::Publisher<sensor_msgs::msg::Temperature>::SharedPtr pub_;

	int         hub_port_;
	std::string frame_id_;

	double data_rate_{-1};
	double temperature_change_trigger_{std::numeric_limits<double>::quiet_NaN()};
};
}  // namespace robp_phidgets

#endif  // ROBP_PHIDGETS_TEMPERATURE_HPP