// robp_phidgets
#include <robp_phidgets/spatial.hpp>

// phidgets api
#include <phidgets_api/phidget22.hpp>

// ROS
#include <angles/angles.h>

#include <rclcpp/logging.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// STL
#include <chrono>

namespace robp_phidgets
{
Spatial::Spatial() : Node("spatial")
{
	int32_t serial_number = this->declare_parameter("spatial_serial_num", -1);
	hub_port_             = this->declare_parameter("spatial_hub_port", 0);
	frame_id_             = this->declare_parameter("spatial_frame_id", "imu_link");
	double data_rate      = this->declare_parameter("spatial_data_rate", 500.0);
	int    algorithm      = this->declare_parameter("spatial_algorithm", 1);
	double ahrs_angular_velocity_threshold =
	    this->declare_parameter("spatial_ahrs_angular_velocity_threshold", 0.5);
	double ahrs_angular_velocity_delta_threshold =
	    this->declare_parameter("spatial_ahrs_angular_velocity_delta_threshold", 0.1);
	double ahrs_acceleration_threshold =
	    this->declare_parameter("spatial_ahrs_acceleration_threshold", 0.05);
	double ahrs_mag_time   = this->declare_parameter("spatial_ahrs_mag_time", 120.0);
	double ahrs_accel_time = this->declare_parameter("spatial_ahrs_accel_time", 120.0);
	double ahrs_bias_time  = this->declare_parameter("spatial_ahrs_bias_time", 1.25);
	bool   heating_enabled = this->declare_parameter("spatial_heating_enabled", false);
	// double linear_acceleration_stdev =
	//     this->declare_parameter("spatial_linear_acceleration_stdev", 280.0);
	// double angular_velocity_stdev =
	//     this->declare_parameter("spatial_angular_velocity_stdev", 0.095);
	// double magnetic_field_stdev =
	// this->declare_parameter("spatial_magnetic_field_stdev", 1.1); int
	// time_resynchronization_interval_ms =
	//     this->declare_parameter("spatial_magnetic_field_stdev", 5000);
	double cc_mag_field = this->declare_parameter("spatial_cc_mag_field", 0.52859);
	double cc_offset0   = this->declare_parameter("spatial_cc_offset0", 0.03921);
	double cc_offset1   = this->declare_parameter("spatial_cc_offset1", 0.19441);
	double cc_offset2   = this->declare_parameter("spatial_cc_offset2", -0.03493);
	double cc_gain0     = this->declare_parameter("spatial_cc_gain0", 1.81704);
	double cc_gain1     = this->declare_parameter("spatial_cc_gain1", 1.81028);
	double cc_gain2     = this->declare_parameter("spatial_cc_gain2", 2.04819);
	double cc_t0        = this->declare_parameter("spatial_cc_t0", 0.00142);
	double cc_t1        = this->declare_parameter("spatial_cc_t1", -0.03591);
	double cc_t2        = this->declare_parameter("spatial_cc_t2", 0.00160);
	double cc_t3        = this->declare_parameter("spatial_cc_t3", -0.05038);
	double cc_t4        = this->declare_parameter("spatial_cc_t4", -0.03942);
	double cc_t5        = this->declare_parameter("spatial_cc_t5", -0.05673);

	imu_pub_ = this->create_publisher<sensor_msgs::msg::Imu>("imu/data_raw", 1);
	mag_pub_ = this->create_publisher<sensor_msgs::msg::MagneticField>("imu/mag", 1);

	create();
	assignEventHandlers();
	::phidgets::helpers::openWaitForAttachment(reinterpret_cast<PhidgetHandle>(spatial_),
	                                           serial_number, hub_port_, false, 0);

	calibrate_srv_ = this->create_service<std_srvs::srv::Empty>(
	    "calibrate", std::bind(&Spatial::calibrateCallback, this, std::placeholders::_1,
	                           std::placeholders::_2));

	calibrate();

	setDataRate(data_rate);
	setAlgorithm(0 == algorithm
	                 ? SPATIAL_ALGORITHM_NONE
	                 : (1 == algorithm ? SPATIAL_ALGORITHM_AHRS : SPATIAL_ALGORITHM_IMU));
	setAHRSParameters(ahrs_angular_velocity_threshold,
	                  ahrs_angular_velocity_delta_threshold, ahrs_acceleration_threshold,
	                  ahrs_mag_time, ahrs_accel_time, ahrs_bias_time);
	if (heatingEnabled() != heating_enabled) {
		setHeatingEnabled(heating_enabled);
	}
	setMagnetometerCorrectionParameters(cc_mag_field, cc_offset0, cc_offset1, cc_offset2,
	                                    cc_gain0, cc_gain1, cc_gain2, cc_t0, cc_t1, cc_t2,
	                                    cc_t3, cc_t4, cc_t5);

	initialized_ = true;
}

Spatial::~Spatial() { close(); }

void Spatial::create()
{
	PhidgetReturnCode ret = PhidgetSpatial_create(&spatial_);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to create temperature sensor on port " + std::to_string(hub_port_), ret);
	}
}

void Spatial::close()
{
	if (nullptr != spatial_) {
		PhidgetHandle handle = reinterpret_cast<PhidgetHandle>(spatial_);
		::phidgets::helpers::closeAndDelete(&handle);
		spatial_ = nullptr;
	}
}

void Spatial::assignEventHandlers()
{
	PhidgetReturnCode ret;

	ret = PhidgetSpatial_setOnAlgorithmDataHandler(spatial_, algorithmCallback, this);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to set on algorithm data handler on port " + std::to_string(hub_port_),
		    ret);
	}

	ret = PhidgetSpatial_setOnSpatialDataHandler(spatial_, spatialCallback, this);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to set on spatial data handler on port " + std::to_string(hub_port_),
		    ret);
	}

	ret = Phidget_setOnAttachHandler(reinterpret_cast<PhidgetHandle>(spatial_),
	                                 attachCallback, this);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to set attach handler on port " + std::to_string(hub_port_), ret);
	}

	ret = Phidget_setOnDetachHandler(reinterpret_cast<PhidgetHandle>(spatial_),
	                                 detachCallback, this);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to set detach handler on port " + std::to_string(hub_port_), ret);
	}

	ret = Phidget_setOnErrorHandler(reinterpret_cast<PhidgetHandle>(spatial_),
	                                errorCallback, this);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to set error handler on port " + std::to_string(hub_port_), ret);
	}
}

void Spatial::calibrate()
{
	using namespace std::chrono_literals;

	RCLCPP_WARN(this->get_logger(),
	            "Calibrating IMU, this takes around 2 seconds to finish. Make sure that "
	            "the device "
	            "is not moved during this time.");
	zeroGyro();
	rclcpp::sleep_for(2s);
	RCLCPP_INFO(this->get_logger(), "Calibrating IMU done.");
}

void Spatial::setAHRSParameters(double angular_velocity_threshold,
                                double angular_velocity_delta_threshold,
                                double acceleration_threshold, double mag_time,
                                double accel_time, double bias_time)
{
	PhidgetReturnCode ret = PhidgetSpatial_setAHRSParameters(
	    spatial_, angular_velocity_threshold, angular_velocity_delta_threshold,
	    acceleration_threshold, mag_time, accel_time, bias_time);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to set AHRS parameters for spatial on port " + std::to_string(hub_port_),
		    ret);
	}
	angular_velocity_threshold_       = angular_velocity_threshold;
	angular_velocity_delta_threshold_ = angular_velocity_delta_threshold;
	acceleration_threshold_           = acceleration_threshold;
	mag_time_                         = mag_time;
	accel_time_                       = accel_time;
	bias_time_                        = bias_time;
}

Phidget_SpatialAlgorithm Spatial::algorithm() const
{
	Phidget_SpatialAlgorithm alg;
	PhidgetReturnCode        ret = PhidgetSpatial_getAlgorithm(spatial_, &alg);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to get algorithm for spatial on port " + std::to_string(hub_port_), ret);
	}
	return alg;
}

void Spatial::setAlgorithm(Phidget_SpatialAlgorithm algorithm)
{
	PhidgetReturnCode ret = PhidgetSpatial_setAlgorithm(spatial_, algorithm);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to set algorithm for spatial on port " + std::to_string(hub_port_), ret);
	}
	algorithm_ = algorithm;
}

double Spatial::dataRate() const
{
	double            rate;
	PhidgetReturnCode ret = PhidgetSpatial_getDataRate(spatial_, &rate);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to get data rate for spatial on port " + std::to_string(hub_port_), ret);
	}
	return rate;
}

void Spatial::setDataRate(double rate)
{
	PhidgetReturnCode ret = PhidgetSpatial_setDataRate(spatial_, rate);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to set data rate for spatial on port " + std::to_string(hub_port_), ret);
	}
	data_rate_        = rate;
	data_interval_ns_ = 1000.0 * 1000.0 * 1.0 / rate;
}

bool Spatial::heatingEnabled() const
{
	int               enabled;
	PhidgetReturnCode ret = PhidgetSpatial_getHeatingEnabled(spatial_, &enabled);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to get heating enabled for spatial on port " + std::to_string(hub_port_),
		    ret);
	}
	return enabled;
}

void Spatial::setHeatingEnabled(bool enabled)
{
	PhidgetReturnCode ret = PhidgetSpatial_setDataRate(spatial_, enabled);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to set heating enabled for spatial on port " + std::to_string(hub_port_),
		    ret);
	}
	heating_enabled_ = enabled;
}

void Spatial::setMagnetometerCorrectionParameters(double magnetic_field, double offset0,
                                                  double offset1, double offset2,
                                                  double gain0, double gain1,
                                                  double gain2, double T0, double T1,
                                                  double T2, double T3, double T4,
                                                  double T5)
{
	PhidgetReturnCode ret = PhidgetSpatial_setMagnetometerCorrectionParameters(
	    spatial_, magnetic_field, offset0, offset1, offset2, gain0, gain1, gain2, T0, T1,
	    T2, T3, T4, T5);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to set magnetometer correction parameters for spatial on "
		    "port " +
		        std::to_string(hub_port_),
		    ret);
	}
	cp_magnetic_field_ = magnetic_field;
	cp_offset0_        = offset0;
	cp_offset1_        = offset1;
	cp_offset2_        = offset2;
	cp_gain0_          = gain0;
	cp_gain1_          = gain1;
	cp_gain2_          = gain2;
	cp_T0_             = T0;
	cp_T1_             = T1;
	cp_T2_             = T2;
	cp_T3_             = T3;
	cp_T4_             = T4;
	cp_T5_             = T5;
}

tf2::Quaternion Spatial::quaternion() const
{
	PhidgetSpatial_SpatialQuaternion quat;
	PhidgetReturnCode                ret = PhidgetSpatial_getQuaternion(spatial_, &quat);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to get quaternion for spatial on port " + std::to_string(hub_port_), ret);
	}
	return {quat.x, quat.y, quat.z, quat.w};
}

void Spatial::resetMagnetometerCorrectionParameters()
{
	PhidgetReturnCode ret = PhidgetSpatial_resetMagnetometerCorrectionParameters(spatial_);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to reset magnetometer correction parameters for spatial on "
		    "port " +
		        std::to_string(hub_port_),
		    ret);
	}
}

void Spatial::saveMagnetometerCorrectionParameters()
{
	PhidgetReturnCode ret = PhidgetSpatial_saveMagnetometerCorrectionParameters(spatial_);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to save magnetometer correction parameters for spatial on "
		    "port " +
		        std::to_string(hub_port_),
		    ret);
	}
}

void Spatial::zeroAlgorithm()
{
	PhidgetReturnCode ret = PhidgetSpatial_zeroAlgorithm(spatial_);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to zero algorithm for spatial on port " + std::to_string(hub_port_), ret);
	}
}

void Spatial::zeroGyro()
{
	PhidgetReturnCode ret = PhidgetSpatial_zeroGyro(spatial_);
	if (EPHIDGET_OK != ret) {
		throw ::phidgets::Phidget22Error(
		    "Failed to zero gyro for spatial on port " + std::to_string(hub_port_), ret);
	}
}

int Spatial::port() const { return hub_port_; }

void Spatial::init()
{
	orientation_         = tf2::Quaternion(0, 0, 0, 1);
	angular_velocity_    = tf2::Vector3();
	linear_acceleration_ = tf2::Vector3();
	magnetic_field_      = tf2::Vector3();

	if (!initialized_) {
		return;
	}

	if (0 <= data_rate_) {
		setDataRate(data_rate_);
	}
	if (heatingEnabled() != heating_enabled_) {
		setHeatingEnabled(heating_enabled_);
	}
	setAHRSParameters(angular_velocity_threshold_, angular_velocity_delta_threshold_,
	                  acceleration_threshold_, mag_time_, accel_time_, bias_time_);
	setAlgorithm(algorithm_);
	setMagnetometerCorrectionParameters(cp_magnetic_field_, cp_offset0_, cp_offset1_,
	                                    cp_offset2_, cp_gain0_, cp_gain1_, cp_gain2_,
	                                    cp_T0_, cp_T1_, cp_T2_, cp_T3_, cp_T4_, cp_T5_);
}

void Spatial::publish()
{
	uint64_t imu_diff_in_ns = last_data_timestamp_ns_ - data_time_zero_ns_;
	uint64_t time_in_ns     = ros_time_zero_.nanoseconds() + imu_diff_in_ns;

	if (time_in_ns < last_ros_stamp_ns_) {
		RCLCPP_WARN(this->get_logger(),
		            "Time went backwards (%lu < %lu)! Not publishing message.", time_in_ns,
		            last_ros_stamp_ns_);
		return;
	}
	last_ros_stamp_ns_    = time_in_ns;
	rclcpp::Time ros_time = rclcpp::Time(time_in_ns);

	auto imu_msg             = std::make_unique<sensor_msgs::msg::Imu>();
	imu_msg->header.frame_id = frame_id_;
	imu_msg->header.stamp    = ros_time;

	auto mag_msg    = std::make_unique<sensor_msgs::msg::MagneticField>();
	mag_msg->header = imu_msg->header;

	// build covariance matrices
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			if (i == j) {
				int idx                                      = j * 3 + i;
				imu_msg->angular_velocity_covariance[idx]    = angular_velocity_variance_;
				imu_msg->linear_acceleration_covariance[idx] = linear_acceleration_variance_;
				mag_msg->magnetic_field_covariance[idx]      = magnetic_field_variance_;
			}
		}
	}

	imu_msg->orientation = tf2::toMsg(orientation_);
	// TODO: imu_msg->orientation_covariance = ...;
	imu_msg->angular_velocity    = tf2::toMsg(angular_velocity_);
	imu_msg->linear_acceleration = tf2::toMsg(linear_acceleration_);

	mag_msg->magnetic_field = tf2::toMsg(magnetic_field_);

	imu_pub_->publish(std::move(imu_msg));
	mag_pub_->publish(std::move(mag_msg));
}

void Spatial::algorithmCallback(PhidgetSpatialHandle /* ch */, void *ctx,
                                double const q[4], double /* timestamp */)
{
	Spatial *s      = static_cast<Spatial *>(ctx);
	s->orientation_ = tf2::Quaternion(q[0], q[1], q[2], q[3]);
}

void Spatial::spatialCallback(PhidgetSpatialHandle /* ch */, void *ctx,
                              double const acc[3], double const ar[3],
                              double const mag[3], double timestamp)
{
	Spatial *s = static_cast<Spatial *>(ctx);

	if (!rclcpp::ok()) {
		return;
	}

	rclcpp::Time now = s->now();

	if (0 == s->last_cb_time_.seconds() && 0 == s->last_cb_time_.nanoseconds()) {
		s->last_cb_time_ = now;
		return;
	}

	rclcpp::Duration time_since_last_cb = now - s->last_cb_time_;
	uint64_t         this_ts_ns = static_cast<uint64_t>(timestamp * 1000.0 * 1000.0);

	if (s->synchronize_timestamps_) {
		// if (time_since_last_cb.toNSec() >= (s->data_interval_ns_ -
		// s->cb_delta_epsilon_ns_) &&
		//     time_since_last_cb.toNSec() <= (s->data_interval_ns_ +
		//     s->cb_delta_epsilon_ns_)) {
		s->ros_time_zero_          = now;
		s->data_time_zero_ns_      = this_ts_ns;
		s->synchronize_timestamps_ = false;
		s->can_publish_            = true;
		// } else {
		// 	ROS_DEBUG(
		// 	    "Data not within acceptable window for synchronization: "
		// 	    "expected between %ld and %ld, saw %ld",
		// 	    s->data_interval_ns_ - s->cb_delta_epsilon_ns_,
		// 	    s->data_interval_ns_ + s->cb_delta_epsilon_ns_,
		// time_since_last_cb.toNSec());
		// }
	}

	if (s->can_publish_) {
		constexpr double g = 9.80665;

		s->angular_velocity_ =
		    tf2::Vector3(angles::from_degrees(ar[0]), angles::from_degrees(ar[1]),
		                 angles::from_degrees(ar[2]));
		s->linear_acceleration_ = tf2::Vector3(acc[0], acc[1], acc[2]) * -g;
		if (PUNK_DBL != mag[0]) {
			s->magnetic_field_ = tf2::Vector3(mag[0], mag[1], mag[2]) * 1e-4;
		} else {
			constexpr double nan = std::numeric_limits<double>::quiet_NaN();
			s->magnetic_field_   = tf2::Vector3(nan, nan, nan);
		}

		s->last_data_timestamp_ns_ = this_ts_ns;

		s->publish();
	}

	rclcpp::Duration diff = now - s->ros_time_zero_;
	if (0 < s->time_resync_interval_ns_ &&
	    diff.nanoseconds() >= s->time_resync_interval_ns_) {
		s->synchronize_timestamps_ = true;
	}

	s->last_cb_time_ = now;
}

void Spatial::attachCallback(PhidgetHandle /* ch */, void *ctx)
{
	Spatial *s = static_cast<Spatial *>(ctx);
	RCLCPP_INFO(s->get_logger(), "Attach spatial on port %d", s->port());
	s->init();
	s->synchronize_timestamps_ = true;
	s->can_publish_            = false;
	s->last_cb_time_           = rclcpp::Time();
}

void Spatial::detachCallback(PhidgetHandle /* ch */, void *ctx)
{
	Spatial *s = static_cast<Spatial *>(ctx);
	RCLCPP_INFO(s->get_logger(), "Detach spatial on port %d", s->port());
}

void Spatial::errorCallback(PhidgetHandle /* ch */, void *ctx,
                            Phidget_ErrorEventCode code, char const *description)
{
	Spatial *s = static_cast<Spatial *>(ctx);
	RCLCPP_ERROR(s->get_logger(), "Error spatial on port %d: %s", s->port(), description);
	PhidgetLog_log(PHIDGET_LOG_ERROR, "Error %d: %s", code, description);
}

void Spatial::calibrateCallback(
    std::shared_ptr<std_srvs::srv::Empty::Request> const /* request */,
    std::shared_ptr<std_srvs::srv::Empty::Response> /* response */)
{
	calibrate();
}
}  // namespace robp_phidgets