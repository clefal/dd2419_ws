// robp_phidgets
#include <robp_phidgets/encoder.hpp>

// phidgets api
#include <phidgets_api/phidget22.hpp>

// ROS
#include <rclcpp/logging.hpp>

namespace robp_phidgets
{
Encoder::Encoder(rclcpp::Node *node, int32_t serial_number, int hub_port,
                 bool is_hub_port_device, int channel, std::function<void()> callback)
    : node_(node), callback_(callback), hub_port_(hub_port)
{
	create();
	assignEventHandlers();
	::phidgets::helpers::openWaitForAttachment(reinterpret_cast<PhidgetHandle>(encoder_),
	                                           serial_number, hub_port, is_hub_port_device,
	                                           channel);
}

Encoder::~Encoder() { close(); }

void Encoder::create()
{
	PhidgetReturnCode ret = PhidgetEncoder_create(&encoder_);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to create encoder on port " + std::to_string(hub_port_), ret);
	}
}

void Encoder::close()
{
	if (nullptr != encoder_) {
		PhidgetHandle handle = reinterpret_cast<PhidgetHandle>(encoder_);
		::phidgets::helpers::closeAndDelete(&handle);
		encoder_ = nullptr;
	}
}

void Encoder::assignEventHandlers()
{
	PhidgetReturnCode ret;

	ret = PhidgetEncoder_setOnPositionChangeHandler(encoder_, positionChangeCallback, this);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to set on position change handler on port " + std::to_string(hub_port_),
		    ret);
	}

	ret = Phidget_setOnAttachHandler(reinterpret_cast<PhidgetHandle>(encoder_),
	                                 attachCallback, this);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to set attach handler on port " + std::to_string(hub_port_), ret);
	}

	ret = Phidget_setOnDetachHandler(reinterpret_cast<PhidgetHandle>(encoder_),
	                                 detachCallback, this);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to set detach handler on port " + std::to_string(hub_port_), ret);
	}

	ret = Phidget_setOnErrorHandler(reinterpret_cast<PhidgetHandle>(encoder_),
	                                errorCallback, this);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to set error handler on port " + std::to_string(hub_port_), ret);
	}
}

double Encoder::dataRate() const
{
	double            rate;
	PhidgetReturnCode ret = PhidgetEncoder_getDataRate(encoder_, &rate);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to get data rate for encoder on port " + std::to_string(hub_port_), ret);
	}
	return rate;
}

void Encoder::setDataRate(double rate)
{
	PhidgetReturnCode ret = PhidgetEncoder_setDataRate(encoder_, rate);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to set data rate for encoder on port " + std::to_string(hub_port_), ret);
	}
	data_rate_ = rate;
}

int64_t Encoder::indexPosition() const
{
	int64_t           idx;
	PhidgetReturnCode ret = PhidgetEncoder_getIndexPosition(encoder_, &idx);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to get index position for encoder on port " + std::to_string(hub_port_),
		    ret);
	}
	return idx;
}

int64_t Encoder::position() const
{
	int64_t           pos;
	PhidgetReturnCode ret = PhidgetEncoder_getPosition(encoder_, &pos);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to get position for encoder on port " + std::to_string(hub_port_), ret);
	}
	return pos;
}

void Encoder::setPosition(int64_t position)
{
	PhidgetReturnCode ret = PhidgetEncoder_setPosition(encoder_, position);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to set position for encoder on port " + std::to_string(hub_port_), ret);
	}
}

uint32_t Encoder::positionChangeTrigger() const
{
	uint32_t          change;
	PhidgetReturnCode ret = PhidgetEncoder_getPositionChangeTrigger(encoder_, &change);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to get position change trigger for encoder on port " +
		        std::to_string(hub_port_),
		    ret);
	}
	return change;
}

void Encoder::setPositionChangeTrigger(uint32_t trigger)
{
	PhidgetReturnCode ret = PhidgetEncoder_setPositionChangeTrigger(encoder_, trigger);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to set position change trigger for encoder on port " +
		        std::to_string(hub_port_),
		    ret);
	}
	position_change_trigger_ = trigger;
}

int Encoder::port() const { return hub_port_; }

int Encoder::change() const
{
	changed_ = false;
	return position_change_.exchange(0);
}

bool Encoder::changed() const { return changed_; }

void Encoder::init()
{
	if (0 <= data_rate_) {
		setDataRate(data_rate_);
	}
	if (0 <= position_change_trigger_) {
		setPositionChangeTrigger(position_change_trigger_);
	}
}

void Encoder::positionChangeCallback(PhidgetEncoderHandle /* ch */, void *ctx,
                                     int position_change, double /* time_change */,
                                     int /* index_triggered */)
{
	static_cast<Encoder *>(ctx)->position_change_ += position_change;
	static_cast<Encoder *>(ctx)->changed_ = true;
	static_cast<Encoder *>(ctx)->callback_();
}

void Encoder::attachCallback(PhidgetHandle /* ch */, void *ctx)
{
	Encoder *e = static_cast<Encoder *>(ctx);
	RCLCPP_INFO(e->node_->get_logger(), "Attach encoder on port %d", e->port());
	e->init();
}

void Encoder::detachCallback(PhidgetHandle /* ch */, void *ctx)
{
	Encoder *e = static_cast<Encoder *>(ctx);
	RCLCPP_INFO(e->node_->get_logger(), "Detach encoder on port %d", e->port());
}

void Encoder::errorCallback(PhidgetHandle /* ch */, void *ctx,
                            Phidget_ErrorEventCode code, char const *description)
{
	Encoder *e = static_cast<Encoder *>(ctx);
	RCLCPP_ERROR(e->node_->get_logger(), "Error encoder on port %d: %s", e->port(),
	             description);
	PhidgetLog_log(PHIDGET_LOG_ERROR, "Error %d: %s", code, description);
}
}  // namespace robp_phidgets