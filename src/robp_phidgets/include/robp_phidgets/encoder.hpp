#ifndef ROBP_PHIDGETS_ENCODER_HPP
#define ROBP_PHIDGETS_ENCODER_HPP

// Phidget
#include <libphidget22/phidget22.h>

// STL
#include <atomic>
#include <functional>

// ROS
#include <rclcpp/node.hpp>

namespace robp_phidgets
{
class Encoder
{
 public:
	Encoder(rclcpp::Node *node, int32_t serial_number, int hub_port,
	        bool is_hub_port_device, int channel, std::function<void()> callback);

	~Encoder();

	double dataRate() const;

	void setDataRate(double rate);

	int64_t indexPosition() const;

	int64_t position() const;

	void setPosition(int64_t position);

	uint32_t positionChangeTrigger() const;

	void setPositionChangeTrigger(uint32_t trigger);

	int port() const;

	int change() const;

	bool changed() const;

 private:
	void create();

	void close();

	void assignEventHandlers();

	void init();

	static void positionChangeCallback(PhidgetEncoderHandle ch, void *ctx,
	                                   int position_change, double time_change,
	                                   int index_triggered);

	static void attachCallback(PhidgetHandle ch, void *ctx);

	static void detachCallback(PhidgetHandle ch, void *ctx);

	static void errorCallback(PhidgetHandle ch, void *ctx, Phidget_ErrorEventCode code,
	                          char const *description);

 private:
	rclcpp::Node *node_;

	PhidgetEncoderHandle encoder_;

	std::function<void()> callback_;

	int hub_port_;

	double data_rate_{-1};
	int    position_change_trigger_{-1};

	mutable std::atomic_int  position_change_{};
	mutable std::atomic_bool changed_ = false;
};
}  // namespace robp_phidgets

#endif  // ROBP_PHIDGETS_ENCODER_HPP