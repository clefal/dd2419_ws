#ifndef ROBP_PHIDGETS_SPATIAL_HPP
#define ROBP_PHIDGETS_SPATIAL_HPP

// Phidget
#include <libphidget22/phidget22.h>

// ROS
#include <tf2/LinearMath/Quaternion.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/magnetic_field.hpp>
#include <std_srvs/srv/empty.hpp>

// STL
#include <string>

namespace robp_phidgets
{
class Spatial : public rclcpp::Node
{
 public:
	Spatial();

	~Spatial();

	void calibrate();

	void setAHRSParameters(double angular_velocity_threshold,
	                       double angular_velocity_delta_threshold,
	                       double acceleration_threshold, double mag_time,
	                       double accel_time, double bias_time);

	Phidget_SpatialAlgorithm algorithm() const;

	void setAlgorithm(Phidget_SpatialAlgorithm algorithm);

	double dataRate() const;

	void setDataRate(double rate);

	bool heatingEnabled() const;

	void setHeatingEnabled(bool enabled);

	void setMagnetometerCorrectionParameters(double magnetic_field, double offset0,
	                                         double offset1, double offset2, double gain0,
	                                         double gain1, double gain2, double T0,
	                                         double T1, double T2, double T3, double T4,
	                                         double T5);

	tf2::Quaternion quaternion() const;

	void resetMagnetometerCorrectionParameters();

	void saveMagnetometerCorrectionParameters();

	void zeroAlgorithm();

	void zeroGyro();

	int port() const;

 private:
	void create();

	void close();

	void assignEventHandlers();

	void init();

	void publish();

	static void algorithmCallback(PhidgetSpatialHandle ch, void *ctx,
	                              double const quaternion[4], double timestamp);

	static void spatialCallback(PhidgetSpatialHandle ch, void *ctx,
	                            double const acceleration[3], double const angular_rate[3],
	                            double const magnetic_field[3], double timestamp);

	static void attachCallback(PhidgetHandle ch, void *ctx);

	static void detachCallback(PhidgetHandle ch, void *ctx);

	static void errorCallback(PhidgetHandle ch, void *ctx, Phidget_ErrorEventCode code,
	                          char const *description);

	void calibrateCallback(std::shared_ptr<std_srvs::srv::Empty::Request> const request,
	                       std::shared_ptr<std_srvs::srv::Empty::Response>      response);

 private:
	PhidgetSpatialHandle spatial_{};

	rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr           imu_pub_;
	rclcpp::Publisher<sensor_msgs::msg::MagneticField>::SharedPtr mag_pub_;

	rclcpp::Service<std_srvs::srv::Empty>::SharedPtr calibrate_srv_;

	int hub_port_;

	std::string frame_id_{"imu_link"};

	// Time
	rclcpp::Time ros_time_zero_;
	bool         synchronize_timestamps_{true};
	uint64_t     data_time_zero_ns_{0};
	uint64_t     last_data_timestamp_ns_{0};
	uint64_t     last_ros_stamp_ns_{0};
	int64_t      time_resync_interval_ns_{0};
	int64_t      data_interval_ns_{0};
	bool         can_publish_{false};
	rclcpp::Time last_cb_time_;
	int64_t      cb_delta_epsilon_ns_{1000 * 1000};

	tf2::Quaternion orientation_;
	double          angular_velocity_variance_;
	tf2::Vector3    angular_velocity_;
	double          linear_acceleration_variance_;
	tf2::Vector3    linear_acceleration_;

	double       magnetic_field_variance_;
	tf2::Vector3 magnetic_field_;

	bool initialized_{false};

	double data_rate_{-1};

	bool heating_enabled_{false};

	// AHRS parameters
	double angular_velocity_threshold_{};
	double angular_velocity_delta_threshold_{};
	double acceleration_threshold_{};
	double mag_time_{};
	double accel_time_{};
	double bias_time_{};

	// Algorithm
	Phidget_SpatialAlgorithm algorithm_;

	// Magnetometer correction parameters
	double cp_magnetic_field_{};
	double cp_offset0_{};
	double cp_offset1_{};
	double cp_offset2_{};
	double cp_gain0_{};
	double cp_gain1_{};
	double cp_gain2_{};
	double cp_T0_{};
	double cp_T1_{};
	double cp_T2_{};
	double cp_T3_{};
	double cp_T4_{};
	double cp_T5_{};
};

void accelerationChange(PhidgetAccelerometerHandle ch, void *ctx,
                        double const acceleration[3], double timestamp);
}  // namespace robp_phidgets

#endif  // ROBP_PHIDGETS_SPATIAL_HPP