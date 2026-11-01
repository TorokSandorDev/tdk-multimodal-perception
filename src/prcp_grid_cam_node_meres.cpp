#include <future>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/point.hpp"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>

using namespace std::chrono_literals;

struct GridMap {
    float position_x = 0.0f;
    float position_y = 0.0f;
    float cell_size = 0.05f;
    float length_x = 10.0f;
    float length_y = 10.0f;
    int grid_width = 200;
    int cell_num_x = 200;
    int cell_num_y = 200;

    void initGrid(nav_msgs::msg::OccupancyGrid &grid) const {
        if (cell_size <= 0.001f) return; 
        grid.header.frame_id = "ground_link";
        grid.info.resolution = cell_size;
        grid.info.width = static_cast<uint32_t>(cell_num_x);
        grid.info.height = static_cast<uint32_t>(cell_num_y);
        grid.info.origin.position.x = position_x - length_x / 2.0; 
        grid.info.origin.position.y = position_y - length_y / 2.0; 
        grid.info.origin.position.z = 0.0;
        grid.info.origin.orientation.w = 1.0;
    }

    void paramRefresh() {
        if (cell_size <= 0.01f || std::isnan(cell_size)) {
            cell_size = 0.05f;
        }
        float safe_len_x = (length_x > 0.1f) ? length_x : 10.0f;
        float safe_len_y = (length_y > 0.1f) ? length_y : 10.0f;

        cell_num_x = static_cast<int>(std::ceil(safe_len_x / cell_size));
        cell_num_y = static_cast<int>(std::ceil(safe_len_y / cell_size));
        
        if (cell_num_x > 5000) cell_num_x = 5000;
        if (cell_num_y > 5000) cell_num_y = 5000;
        if (cell_num_x < 10) cell_num_x = 10;
        if (cell_num_y < 10) cell_num_y = 10;
        grid_width = cell_num_x;
    }
};

class PrcpGridCamNode : public rclcpp::Node {
public:
    PrcpGridCamNode() : Node("prcp_grid_cam_pub") {
        this->declare_parameter<std::string>("camera_info_topic", "camera_info_topic");
        this->declare_parameter<std::string>("camera_stream_topic", "camera_stream_topic");
        this->declare_parameter<float>("position_x", 0.0);
        this->declare_parameter<float>("position_y", 0.0);
        this->declare_parameter<float>("cell_size", 0.05);
        this->declare_parameter<float>("length_x", 10.0);
        this->declare_parameter<float>("length_y", 10.0);
        this->declare_parameter<std::string>("publish_topic_name", "/cam/grid/drivable_area");
        this->declare_parameter<std::string>("target_frame_id", "zed_camera_front");
        this->declare_parameter<int>("grid_fill_value", 0);       
        this->declare_parameter<int>("grid_background_value", 100); 
        this->declare_parameter<std::string>("target_color", "red");

        this->declare_parameter<float>("max_distance_x", 40.0);
        this->declare_parameter<float>("max_distance_y", 20.0);
        this->declare_parameter<float>("pitch_compensation_deg", 0.0);
        this->declare_parameter<float>("camera_horizon_offset", 0.0); 
        this->declare_parameter<float>("fov_angle_deg", 55.0);

        
        max_distance_x_ = this->get_parameter("max_distance_x").as_double();
        max_distance_y_ = this->get_parameter("max_distance_y").as_double();
        pitch_compensation_deg_ = this->get_parameter("pitch_compensation_deg").as_double();

        target_frame_id_ = this->get_parameter("target_frame_id").as_string();
        camera_info_topic_   = this->get_parameter("camera_info_topic").as_string();
        camera_stream_topic_ = this->get_parameter("camera_stream_topic").as_string();
        publish_topic_name  = this->get_parameter("publish_topic_name").as_string();
        target_color_ = this->get_parameter("target_color").as_string();    
        
        grid_map_.position_x = this->get_parameter("position_x").as_double();
        grid_map_.position_y = this->get_parameter("position_y").as_double();
        grid_map_.cell_size = this->get_parameter("cell_size").as_double();
        grid_map_.length_x = this->get_parameter("length_x").as_double();
        grid_map_.length_y = this->get_parameter("length_y").as_double();
        
        grid_fill_value_ = this->get_parameter("grid_fill_value").as_int();
        grid_background_value_ = this->get_parameter("grid_background_value").as_int();
        
        grid_map_.paramRefresh(); 

        rclcpp::QoS qos_profile = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        param_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&PrcpGridCamNode::onParameterChange, this, std::placeholders::_1));

        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            camera_info_topic_, qos_profile, 
            std::bind(&PrcpGridCamNode::cameraInfoCallback, this, std::placeholders::_1)
        );

        segmentation_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            camera_stream_topic_, qos_profile,
            std::bind(&PrcpGridCamNode::segToGridMapCallback, this, std::placeholders::_1)
        );

        occupancy_grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(publish_topic_name, qos_profile);
        RCLCPP_INFO(this->get_logger(), "Centerline Extracting & Infinite Raycast Node started.");
        if (target_color_ == "green") {
            metrics_csv_.open("lane_metrics.csv", std::ios::out | std::ios::trunc);
            metrics_csv_ << "frame,raw_tracks,pred_tracks,bridged_gaps,bridged_rows,"
                         << "raw_max_span_px,pred_max_span_px,raw_grid_cells,pred_grid_cells,"
                         << "raw_valid_continuity,pred_valid_continuity\n";
            RCLCPP_INFO(this->get_logger(), "Sávkövetési naplózás elindítva: lane_metrics.csv");
        }
    }

private:
    rcl_interfaces::msg::SetParametersResult onParameterChange(const std::vector<rclcpp::Parameter> &parameters) {
        for (const auto & param : parameters) {
            if (param.get_name() == "cell_size") grid_map_.cell_size = param.as_double();
            else if (param.get_name() == "position_x") grid_map_.position_x = param.as_double();
            else if (param.get_name() == "position_y") grid_map_.position_y = param.as_double();
            else if (param.get_name() == "length_x") grid_map_.length_x = param.as_double();
            else if (param.get_name() == "length_y") grid_map_.length_y = param.as_double();
            else if (param.get_name() == "max_distance_x") max_distance_x_ = param.as_double();
            else if (param.get_name() == "max_distance_y") max_distance_y_ = param.as_double();
            else if (param.get_name() == "pitch_compensation_deg") pitch_compensation_deg_ = param.as_double();
        }
        grid_map_.paramRefresh();
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        return result;
    }    

    GridMap grid_map_;
    nav_msgs::msg::OccupancyGrid occupancy_grid_msg_;
    sensor_msgs::msg::CameraInfo::SharedPtr base_camera_info_;
    bool camera_info_received_{false};
    
    int grid_fill_value_{0};
    int grid_background_value_{100};
    double pitch_compensation_deg_{0.0};
    double max_distance_x_{40.0};
    double max_distance_y_{20.0};

    std::string camera_info_topic_, camera_stream_topic_, publish_topic_name, target_frame_id_, target_color_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr segmentation_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_pub_;
    rclcpp::Node::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
    std::ofstream metrics_csv_;
    int frame_idx_{0};

    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        if (!camera_info_received_) {
            base_camera_info_ = msg;
            camera_info_received_ = true;
        }
    }

    void segToGridMapCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (msg->data.empty() || !camera_info_received_ || !base_camera_info_) return;

        cv_bridge::CvImagePtr cv_ptr;
        try { cv_ptr = cv_bridge::toCvCopy(msg, "bgr8"); } 
        catch (const cv_bridge::Exception &e) { return; }
        
        int current_w = cv_ptr->image.cols; 
        int current_h = cv_ptr->image.rows; 
        if (current_w <= 0 || current_h <= 0 || base_camera_info_->width <= 0) return;

        double scale = static_cast<double>(current_w) / static_cast<double>(base_camera_info_->width);
        

        double unpadded_h = static_cast<double>(base_camera_info_->height) * scale;
        double pad_y = (static_cast<double>(current_h) - unpadded_h) / 2.0;

        double camera_horizon_offset_ = this->get_parameter("camera_horizon_offset").as_double();


        double fx = base_camera_info_->k[0] * scale;
        double cx = base_camera_info_->k[2] * scale;
        double fy = base_camera_info_->k[4] * scale;
        double cy = (base_camera_info_->k[5] * scale) + pad_y + camera_horizon_offset_;

        std::string frame_to_use = target_frame_id_.empty() ? msg->header.frame_id : target_frame_id_;
        geometry_msgs::msg::TransformStamped transform_ground_to_cam;
        try {
            transform_ground_to_cam = tf_buffer_->lookupTransform(frame_to_use, "ground_link", tf2::TimePointZero);
        } catch (const tf2::TransformException &ex) { return; }

        cv::Mat mask;
        if (target_color_ == "red") {
            cv::inRange(cv_ptr->image, cv::Scalar(0, 0, 100), cv::Scalar(50, 50, 255), mask);
            

            
        } else if (target_color_ == "green") {
            cv::inRange(cv_ptr->image, cv::Scalar(0, 100, 0), cv::Scalar(50, 255, 50), mask);
        } else {
            cv::cvtColor(cv_ptr->image, mask, cv::COLOR_BGR2GRAY);
            cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);
        }

        cv::rectangle(mask, cv::Point(0, 0), cv::Point(mask.cols-1, mask.rows-1), cv::Scalar(0), 2);

        grid_map_.initGrid(occupancy_grid_msg_);
        occupancy_grid_msg_.header.stamp = this->now();
        occupancy_grid_msg_.header.frame_id = "ground_link";

        int width = grid_map_.cell_num_x;
        int height = grid_map_.cell_num_y;
        double res = grid_map_.cell_size;
        double origin_x = occupancy_grid_msg_.info.origin.position.x;
        double origin_y = occupancy_grid_msg_.info.origin.position.y;

        tf2::Transform tf2_g2c;
        tf2::fromMsg(transform_ground_to_cam.transform, tf2_g2c);

        cv::Mat map_x(height, width, CV_32FC1, cv::Scalar(-1));
        cv::Mat map_y(height, width, CV_32FC1, cv::Scalar(-1));

        double theta = pitch_compensation_deg_ * M_PI / 180.0;
        double cos_t = std::cos(theta);
        double sin_t = std::sin(theta);

        for (int gy = 0; gy < height; ++gy) {
            for (int gx = 0; gx < width; ++gx) {
                double world_x = origin_x + (gx + 0.5) * res;
                double world_y = origin_y + (gy + 0.5) * res;

                if (world_x <= 0.0 || world_x > max_distance_x_) continue; 
                if (std::abs(world_y) > max_distance_y_) continue;

                tf2::Vector3 p_ground(world_x, world_y, 0.0);
                tf2::Vector3 p_cam_ros = tf2_g2c * p_ground;

                double z_opt_base = p_cam_ros.x();
                double x_opt_base = -p_cam_ros.y();
                double y_opt_base = -p_cam_ros.z();

                double z_opt = y_opt_base * sin_t + z_opt_base * cos_t;
                double y_opt = y_opt_base * cos_t - z_opt_base * sin_t;

                if (z_opt_base <= 0.01 || z_opt <= 0.01) continue;

                double x_opt = x_opt_base;

                float u = static_cast<float>(fx * (x_opt / z_opt) + cx);
                float v = static_cast<float>(fy * (y_opt / z_opt) + cy);

                if (u < 0.0f || u >= static_cast<float>(current_w) || 
                    v < 0.0f || v >= static_cast<float>(current_h)) {
                    continue; 
                }

                map_x.at<float>(gy, gx) = u;
                map_y.at<float>(gy, gx) = v;
            }
        }

        cv::Mat warped_mask;

        if (target_color_ == "red") {
            cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
            cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, close_kernel);
            
            cv::GaussianBlur(mask, mask, cv::Size(11, 11), 0);
            
            cv::remap(mask, warped_mask, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
            
            cv::threshold(warped_mask, warped_mask, 127, 255, cv::THRESH_BINARY);
            
        } else if (target_color_ == "green") {
            cv::Mat thin_mask = cv::Mat::zeros(height, width, CV_8UC1);
            cv::Mat thin_mask_raw = cv::Mat::zeros(height, width, CV_8UC1);

            struct Track {
                float u;
                int v;
                float slope; 
                std::vector<cv::Point2f> pts;
            };

            std::vector<Track> active_tracks;
            std::vector<Track> finished_tracks;
            int bridged_gaps_count = 0;
            int bridged_rows_total = 0;

            std::vector<Track> raw_active_tracks;
            std::vector<Track> raw_finished_tracks;
            
            for (int r = mask.rows - 1; r >= 0; --r) { 
                std::vector<float> current_centers;
                int run_start = -1;
                
                for (int c = 0; c < mask.cols; ++c) { 
                    if (mask.at<uchar>(r, c) > 127) {
                        if (run_start == -1) run_start = c;
                    } else {
                        if (run_start != -1) {
                            current_centers.push_back((run_start + c - 1) / 2.0f);
                            run_start = -1;
                        }
                    }
                }
                if (run_start != -1) {
                    current_centers.push_back((run_start + mask.cols - 1) / 2.0f);
                }
                
                std::vector<Track> next_tracks;
                for (float curr_u : current_centers) {
                    float best_prev_u = -1;
                    int best_prev_v = -1;
                    float best_prev_slope = 0.0f;
                    float min_dist_found = 99999.0f; 
                    int best_idx = -1;
                    
                    for (size_t i = 0; i < active_tracks.size(); ++i) {
                        float prev_u = active_tracks[i].u;
                        int prev_v = active_tracks[i].v;
                        float prev_slope = active_tracks[i].slope;
                        
                        int dv = r - prev_v; 
                        float predicted_u = prev_u + prev_slope * dv;
                        
                        float dist = std::abs(curr_u - predicted_u);
                        float max_allowed_dist = 15.0f + std::abs(dv) * 2.0f;
                        
                        if (dist < max_allowed_dist && dist < min_dist_found) {
                            min_dist_found = dist;
                            best_prev_u = prev_u;
                            best_prev_v = prev_v;
                            best_prev_slope = prev_slope;
                            best_idx = static_cast<int>(i);
                        }
                    }
                    
                    float smoothed_u = curr_u;
                    float new_slope = 0.0f;
                    
                    if (best_idx != -1) {
                        int gap_rows = std::abs(r - best_prev_v);
                        if (gap_rows > 1) {
                            bridged_gaps_count++;
                            bridged_rows_total += (gap_rows - 1);
                        }

                        float dv = static_cast<float>(r - best_prev_v);
                        float pred_u = best_prev_u + best_prev_slope * dv;
                        float error = curr_u - pred_u;
                        
                        float beta = 0.7f;  
                        smoothed_u = curr_u;
                        new_slope = best_prev_slope + (beta / dv) * error;
                        new_slope = std::max(-100.0f, std::min(100.0f, new_slope)); 
                        
                        std::vector<cv::Point2f> current_pts = active_tracks[best_idx].pts;
                        current_pts.push_back(cv::Point2f(smoothed_u, static_cast<float>(r)));
                        
                        active_tracks.erase(active_tracks.begin() + best_idx);
                        next_tracks.push_back({smoothed_u, r, new_slope, current_pts});
                    } else {
                        std::vector<cv::Point2f> new_pts;
                        new_pts.push_back(cv::Point2f(smoothed_u, static_cast<float>(r)));
                        next_tracks.push_back({smoothed_u, r, 0.0f, new_pts});
                    }
                }
                
                for (size_t i = 0; i < active_tracks.size(); ++i) {
                    if (std::abs(r - active_tracks[i].v) < 15) {
                        next_tracks.push_back(active_tracks[i]);
                    } else {
                        if (active_tracks[i].pts.size() > 2) {
                            finished_tracks.push_back(active_tracks[i]);
                        }
                    }
                }
                active_tracks = next_tracks;

                std::vector<Track> raw_next_tracks;
                for (float curr_u : current_centers) {
                    float min_dist_found = 99999.0f;
                    int best_idx = -1;

                    for (size_t i = 0; i < raw_active_tracks.size(); ++i) {
                        float dist = std::abs(curr_u - raw_active_tracks[i].u);
                        if (dist < 15.0f && dist < min_dist_found) {
                            min_dist_found = dist;
                            best_idx = static_cast<int>(i);
                        }
                    }

                    if (best_idx != -1) {
                        std::vector<cv::Point2f> current_pts = raw_active_tracks[best_idx].pts;
                        current_pts.push_back(cv::Point2f(curr_u, static_cast<float>(r)));
                        raw_active_tracks.erase(raw_active_tracks.begin() + best_idx);
                        raw_next_tracks.push_back({curr_u, r, 0.0f, current_pts});
                    } else {
                        std::vector<cv::Point2f> new_pts;
                        new_pts.push_back(cv::Point2f(curr_u, static_cast<float>(r)));
                        raw_next_tracks.push_back({curr_u, r, 0.0f, new_pts});
                    }
                }
                for (size_t i = 0; i < raw_active_tracks.size(); ++i) {
                    if (raw_active_tracks[i].pts.size() > 2) {
                        raw_finished_tracks.push_back(raw_active_tracks[i]);
                    }
                }
                raw_active_tracks = raw_next_tracks;
            }
            
            for (size_t i = 0; i < active_tracks.size(); ++i) {
                if (active_tracks[i].pts.size() > 2) {
                    finished_tracks.push_back(active_tracks[i]);
                }
            }
            for (size_t i = 0; i < raw_active_tracks.size(); ++i) {
                if (raw_active_tracks[i].pts.size() > 2) {
                    raw_finished_tracks.push_back(raw_active_tracks[i]);
                }
            }
            
            tf2::Transform tf2_c2g = tf2_g2c.inverse();
            tf2::Vector3 cam_origin = tf2_c2g.getOrigin();
            tf2::Matrix3x3 cam_basis = tf2_c2g.getBasis();

            auto projectTracksToMask = [&](const std::vector<Track>& tracks, cv::Mat& out_mask, int& max_span_px) {
                max_span_px = 0;
                for (size_t i = 0; i < tracks.size(); ++i) {
                    if (!tracks[i].pts.empty()) {
                        int span = static_cast<int>(std::abs(tracks[i].pts.front().y - tracks[i].pts.back().y));
                        if (span > max_span_px) max_span_px = span;
                    }

                    std::vector<cv::Point> grid_pts;
                    for (const auto& pt : tracks[i].pts) {
                        double u = pt.x;
                        double v = pt.y;

                        double ray_opt_x = (u - cx) / fx;
                        double ray_opt_y = (v - cy) / fy;
                        double ray_opt_z = 1.0;

                        double ray_opt_base_y = ray_opt_y * cos_t + ray_opt_z * sin_t;
                        double ray_opt_base_z = -ray_opt_y * sin_t + ray_opt_z * cos_t;
                        double ray_opt_base_x = ray_opt_x;

                        tf2::Vector3 V_cam(ray_opt_base_z, -ray_opt_base_x, -ray_opt_base_y);
                        tf2::Vector3 Dir = cam_basis * V_cam;

                        if (Dir.z() >= -0.01) continue;
                        double t_ray = -cam_origin.z() / Dir.z();
                        if (t_ray <= 0) continue;
                        
                        double world_x = cam_origin.x() + t_ray * Dir.x();
                        double world_y = cam_origin.y() + t_ray * Dir.y();

                        if (world_x > max_distance_x_ || std::abs(world_y) > max_distance_y_) continue;

                        int gx = static_cast<int>(std::round((world_x - origin_x) / res));
                        int gy = static_cast<int>(std::round((world_y - origin_y) / res));
                        grid_pts.push_back(cv::Point(gx, gy));
                    }
                    
                    if (grid_pts.size() > 1) {
                        std::vector<std::vector<cv::Point>> to_draw = {grid_pts};
                        cv::polylines(out_mask, to_draw, false, cv::Scalar(255), 2, cv::LINE_8);
                    }
                }
            };

            int pred_max_span = 0;
            int raw_max_span = 0;
            projectTracksToMask(finished_tracks, thin_mask, pred_max_span);
            projectTracksToMask(raw_finished_tracks, thin_mask_raw, raw_max_span);

            int pred_cells = cv::countNonZero(thin_mask);
            int raw_cells = cv::countNonZero(thin_mask_raw);

            int raw_valid = (raw_max_span >= 50) ? 1 : 0;
            int pred_valid = (pred_max_span >= 50) ? 1 : 0;

            if (metrics_csv_.is_open()) {
                frame_idx_++;
                metrics_csv_ << frame_idx_ << ","
                             << raw_finished_tracks.size() << ","
                             << finished_tracks.size() << ","
                             << bridged_gaps_count << ","
                             << bridged_rows_total << ","
                             << raw_max_span << ","
                             << pred_max_span << ","
                             << raw_cells << ","
                             << pred_cells << ","
                             << raw_valid << ","
                             << pred_valid << "\n";
                metrics_csv_.flush();
            }

            warped_mask = thin_mask;
        }

        cv::Mat grid_mat(height, width, CV_8SC1, cv::Scalar(grid_background_value_));
        for (int gy = 0; gy < height; ++gy) {
            for (int gx = 0; gx < width; ++gx) {
                if (warped_mask.at<uchar>(gy, gx) > 127) {
                    grid_mat.at<int8_t>(gy, gx) = grid_fill_value_;
                }
            }
        }

        occupancy_grid_msg_.data.assign(grid_mat.begin<int8_t>(), grid_mat.end<int8_t>());
        occupancy_grid_pub_->publish(occupancy_grid_msg_);
    }
};  

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PrcpGridCamNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}