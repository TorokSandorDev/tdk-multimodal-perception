#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/u_int32.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl_ros/transforms.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/multi/geometries/multi_linestring.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ximgproc.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

#include <pcl/filters/radius_outlier_removal.h>

using point_type = boost::geometry::model::d2::point_xy<double>;
using polygon_type = boost::geometry::model::polygon<point_type>;
using linestring_type = boost::geometry::model::linestring<point_type>;

class GridMap{
    public: 
      float position_x = 0.0;
      float position_y = 0.0;
      float cell_size = 0.15;
      float length_x = 100.0;
      float length_y = 100.0;
      int grid_width;
      float topleft_x;
      float topleft_y;
      int cell_num_x;
      int cell_num_y;
      
      void initGrid(nav_msgs::msg::OccupancyGrid& grid) {
        grid.header.frame_id = "base_link";
        grid.info.origin.position.z = 0;
        grid.info.origin.orientation.w = 0; 
        grid.info.origin.orientation.x = 0;
        grid.info.origin.orientation.y = 0;
        grid.info.origin.orientation.z = 1; 

        grid.info.origin.position.x = position_x + length_x / 2; 
        grid.info.origin.position.y = position_y + length_y / 2; 

        grid.info.width = length_x / cell_size;
        grid.info.height = length_y /cell_size;
        grid.info.resolution = cell_size;
      }
      
      void paramRefresh(){
        topleft_x = position_x + length_x / 2;
        topleft_y = position_y + length_y / 2;

        cell_num_x = int(length_x / cell_size);
        cell_num_y = int(length_y / cell_size);

        grid_width = int(length_x / cell_size);
      }
  };

struct GlobalPoint
{
  geometry_msgs::msg::Point point; 
  double occupancy; 
  float distance; 
};



class PrcpGridFusionNode : public rclcpp::Node {
public:
    PrcpGridFusionNode() : Node("prcp_grid_fusion"), phase2_active_(false) {

        this->declare_parameter<float>("cell_size", 0.15);
        this->declare_parameter<float>("length_x", 100.0);
        this->declare_parameter<float>("length_y", 100.0);
        this->declare_parameter<float>("distance_covered_confidence", 2.0);
        
        this->declare_parameter<float>("logs_odds_increase_camera", 0.5);
        this->declare_parameter<float>("obstacle_fill_weight", 1.5);
        this->declare_parameter<float>("lanes_fill_weight", 1.0);
        this->declare_parameter<double>("blur_kernel_x", 80.0);
        this->declare_parameter<double>("blur_kernel_y", 2.0);
        this->declare_parameter<double>("obstacle_blur_kernel_x", 8.0);
        this->declare_parameter<double>("obstacle_blur_kernel_y", 8.0);
        this->declare_parameter<double>("lanes_blur_kernel_x", 8.0);
        this->declare_parameter<double>("lanes_blur_kernel_y", 8.0);
        this->declare_parameter<double>("ror_radius", 0.20);
        this->declare_parameter<int>("ror_min_neighbors", 5);

        this->declare_parameter<double>("distance_threshold", 50.0);
        this->declare_parameter<double>("logs_odds_increase_camera_phase2", 2.0);
        this->declare_parameter<double>("obstacle_fill_weight_phase2", 1.5);
        this->declare_parameter<double>("lanes_fill_weight_phase2", 1.0);
        this->declare_parameter<double>("blur_kernel_x_phase2", 80.0);
        this->declare_parameter<double>("blur_kernel_y_phase2", 2.0);
        this->declare_parameter<double>("obstacle_blur_kernel_x_phase2", 8.0);
        this->declare_parameter<double>("obstacle_blur_kernel_y_phase2", 8.0);
        this->declare_parameter<double>("lanes_blur_kernel_x_phase2", 8.0);
        this->declare_parameter<double>("lanes_blur_kernel_y_phase2", 8.0);
        this->declare_parameter<double>("ror_radius_phase2", 0.20);
        this->declare_parameter<int>("ror_min_neighbors_phase2", 5);

        this->declare_parameter<double>("exclusion_box_x_min", 0.0);
        this->declare_parameter<double>("exclusion_box_x_max", 1.0);
        this->declare_parameter<double>("exclusion_box_y_min", -1.0);
        this->declare_parameter<double>("exclusion_box_y_max", 1.0);
        this->declare_parameter<double>("freespace_box_x_min", -1.0);
        this->declare_parameter<double>("freespace_box_x_max", 3.0);
        this->declare_parameter<double>("freespace_box_y_min", -2.5);
        this->declare_parameter<double>("freespace_box_y_max", 2.5);

        this->declare_parameter<int>("drivable_closing_kernel_size", 160);
        this->declare_parameter<bool>("force_closing_no_temporal", false);

        this->declare_parameter<float>("obstacle_points_height_max", 0.85);
        this->declare_parameter<float>("obstacle_points_height_min", 0.0);

        this->declare_parameter<bool>("enable_road_blur", true);
        this->declare_parameter<bool>("enable_obstacle_blur", true);
        this->declare_parameter<bool>("enable_lanes_blur", true);

        this->declare_parameter<bool>("enable_road_blur_phase2", false);
        this->declare_parameter<bool>("enable_obstacle_blur_phase2", false);
        this->declare_parameter<bool>("enable_lanes_blur_phase2", false);

        this->declare_parameter<float>("position_x", 0.0);
        this->declare_parameter<float>("position_y", 0.0);

        obstacle_fill_weight_ = this->get_parameter("obstacle_fill_weight").as_double();
        lanes_fill_weight_ = this->get_parameter("lanes_fill_weight").as_double();
        logs_odds_increase_camera_ = this->get_parameter("logs_odds_increase_camera").as_double();
        blur_kernel_x_ = this->get_parameter("blur_kernel_x").as_double();
        blur_kernel_y_ = this->get_parameter("blur_kernel_y").as_double();
        obstacle_blur_kernel_x_ = this->get_parameter("obstacle_blur_kernel_x").as_double();
        obstacle_blur_kernel_y_ = this->get_parameter("obstacle_blur_kernel_y").as_double();
        lanes_blur_kernel_x_ = this->get_parameter("lanes_blur_kernel_x").as_double();
        lanes_blur_kernel_y_ = this->get_parameter("lanes_blur_kernel_y").as_double();
        ror_radius_ = this->get_parameter("ror_radius").as_double();
        ror_min_neighbors_ = this->get_parameter("ror_min_neighbors").as_int();

        distance_threshold_ = this->get_parameter("distance_threshold").as_double();
        obstacle_fill_weight_phase2_ = this->get_parameter("obstacle_fill_weight_phase2").as_double();
        lanes_fill_weight_phase2_ = this->get_parameter("lanes_fill_weight_phase2").as_double();
        logs_odds_increase_camera_phase2_ = this->get_parameter("logs_odds_increase_camera_phase2").as_double();
        blur_kernel_x_phase2_ = this->get_parameter("blur_kernel_x_phase2").as_double();
        blur_kernel_y_phase2_ = this->get_parameter("blur_kernel_y_phase2").as_double();
        obstacle_blur_kernel_x_phase2_ = this->get_parameter("obstacle_blur_kernel_x_phase2").as_double();
        obstacle_blur_kernel_y_phase2_ = this->get_parameter("obstacle_blur_kernel_y_phase2").as_double();
        lanes_blur_kernel_x_phase2_ = this->get_parameter("lanes_blur_kernel_x_phase2").as_double();
        lanes_blur_kernel_y_phase2_ = this->get_parameter("lanes_blur_kernel_y_phase2").as_double();
        ror_radius_phase2_ = this->get_parameter("ror_radius_phase2").as_double();
        ror_min_neighbors_phase2_ = this->get_parameter("ror_min_neighbors_phase2").as_int();

        this->declare_parameter<int>("drivable_mask_threshold", 75);

        exclusion_box_x_min_ = this->get_parameter("exclusion_box_x_min").as_double();
        exclusion_box_x_max_ = this->get_parameter("exclusion_box_x_max").as_double();
        exclusion_box_y_min_ = this->get_parameter("exclusion_box_y_min").as_double();
        exclusion_box_y_max_ = this->get_parameter("exclusion_box_y_max").as_double();

        freespace_box_x_min_ = this->get_parameter("freespace_box_x_min").as_double();
        freespace_box_x_max_ = this->get_parameter("freespace_box_x_max").as_double();
        freespace_box_y_min_ = this->get_parameter("freespace_box_y_min").as_double();
        freespace_box_y_max_ = this->get_parameter("freespace_box_y_max").as_double();


        enable_road_blur_ = this->get_parameter("enable_road_blur").as_bool();
        enable_obstacle_blur_ = this->get_parameter("enable_obstacle_blur").as_bool();
        enable_lanes_blur_ = this->get_parameter("enable_lanes_blur").as_bool();

        enable_road_blur_phase2_ = this->get_parameter("enable_road_blur_phase2").as_bool();
        enable_obstacle_blur_phase2_ = this->get_parameter("enable_obstacle_blur_phase2").as_bool();
        enable_lanes_blur_phase2_ = this->get_parameter("enable_lanes_blur_phase2").as_bool();

        float pos_x_ = this->get_parameter("position_x").as_double();
        float pos_y_ = this->get_parameter("position_y").as_double();

        grid_map_.position_x = pos_x_;
        grid_map_.position_y = pos_y_;

        cell_size_ = this->get_parameter("cell_size").as_double();
        length_x_ = this->get_parameter("length_x").as_double();
        length_y_ = this->get_parameter("length_y").as_double();

        grid_map_.cell_size = cell_size_;
        grid_map_.length_x = length_x_;
        grid_map_.length_y = length_y_;
        force_closing_no_temporal_ = this->get_parameter("force_closing_no_temporal").as_bool();

        distance_covered_confidence_ = this->get_parameter("distance_covered_confidence").as_double();
        drivable_closing_kernel_size_ = this->get_parameter("drivable_closing_kernel_size").as_int();
        last_transform.transform.translation.x = 0.0;
        last_transform.transform.translation.y = 0.0;

        obstacle_points_height_max_ = this->get_parameter("obstacle_points_height_max").as_double();
        obstacle_points_height_min_ = this->get_parameter("obstacle_points_height_min").as_double();

        obstacle_points_check_in = rclcpp::Time(0, 0, RCL_ROS_TIME);
        camera_map_check_in = rclcpp::Time(0, 0, RCL_ROS_TIME);
        last_distance_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

        auto clock = this->get_clock();
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(clock);
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this);

        bool sim_time_check = false;
        this->get_parameter("use_sim_time", sim_time_check);
        RCLCPP_INFO(this->get_logger(), ">>> KAPOTT SIM_TIME PARAMETER: %s <<<", sim_time_check ? "IGEN" : "NEM");

        rclcpp::QoS qos_profile = rclcpp::QoS(rclcpp::KeepLast(1))
          .best_effort()
          .durability_volatile()
          .liveliness(RMW_QOS_POLICY_LIVELINESS_AUTOMATIC);


        camera_lanes_check_in = rclcpp::Time(0, 0, RCL_ROS_TIME);
        vision_lanes_grid_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>("/cam/grid/bad_lines", qos_profile, std::bind(&PrcpGridFusionNode::visionLanesGridCallback, this, std::placeholders::_1));

        obstacle_points_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/ground_segmentation/obstacle_points", qos_profile, std::bind(&PrcpGridFusionNode::obstaclePointsCallback, this, std::placeholders::_1));
        vision_occupancy_grid_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>("/cam/grid/drivable_area", qos_profile, std::bind(&PrcpGridFusionNode::visionOccupancyGridCallback, this, std::placeholders::_1));
        covered_distance_sub_ = this->create_subscription<std_msgs::msg::Float32>("/distance", qos_profile, std::bind(&PrcpGridFusionNode::distance_callback, this, std::placeholders::_1));

        click_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/move_base_simple/goal", 
        10, 
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
            if (click_count_ % 2 == 0) {
                start_point_.x = msg->pose.position.x;
                start_point_.y = msg->pose.position.y;
                has_line_ = false;
                RCLCPP_INFO(this->get_logger(), "Kezdőpont rögzítve: x=%.2f, y=%.2f", start_point_.x, start_point_.y);
            } else {
                end_point_.x = msg->pose.position.x;
                end_point_.y = msg->pose.position.y;
                has_line_ = true;
                RCLCPP_INFO(this->get_logger(), "Végpont rögzítve: x=%.2f, y=%.2f -> Profil generálása...", end_point_.x, end_point_.y);
            }
            click_count_++;
        });

        timer_ = this->create_wall_timer(std::chrono::milliseconds(50), std::bind(&PrcpGridFusionNode::Mapping, this));
        occupancy_grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/occupancy_map", qos_profile);
        profile_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/line_profile", qos_profile);

  }

private:
    GridMap grid_map_;

    nav_msgs::msg::OccupancyGrid occupancy_grid_msg_;
    nav_msgs::msg::OccupancyGrid camera_grid;
    std::vector<GlobalPoint> global_occ;

    geometry_msgs::msg::TransformStamped last_transform;

    float cell_size_{0.15};
    float length_x_{100.0};
    float length_y_{100.0};
    float distance_covered_confidence_{2.0};
    
    float logs_odds_increase_camera_{0.5};
    float obstacle_fill_weight_{1.5};
    float lanes_fill_weight_{1.0};
    float blur_kernel_x_{80.0};
    float blur_kernel_y_{2.0};
    float obstacle_blur_kernel_x_{8.0};
    float obstacle_blur_kernel_y_{8.0};
    float lanes_blur_kernel_x_{8.0};
    float lanes_blur_kernel_y_{8.0};
    double ror_radius_{0.20};
    int ror_min_neighbors_{5};
    int drivable_mask_threshold_{75};

    double distance_threshold_{50.0};
    float logs_odds_increase_camera_phase2_{2.0};
    float obstacle_fill_weight_phase2_{1.5};
    float lanes_fill_weight_phase2_{1.0};
    float blur_kernel_x_phase2_{80.0};
    float blur_kernel_y_phase2_{2.0};
    float obstacle_blur_kernel_x_phase2_{8.0};
    float obstacle_blur_kernel_y_phase2_{8.0};
    float lanes_blur_kernel_x_phase2_{8.0};
    float lanes_blur_kernel_y_phase2_{8.0};
    double ror_radius_phase2_{0.20};
    int ror_min_neighbors_phase2_{5};

    double exclusion_box_x_min_{0.0};
    double exclusion_box_x_max_{1.0};
    double exclusion_box_y_min_{-1.0};
    double exclusion_box_y_max_{1.0};

    double freespace_box_x_min_{-1.0};
    double freespace_box_x_max_{3.0};
    double freespace_box_y_min_{-2.5};
    double freespace_box_y_max_{2.5};

    bool enable_road_blur_phase2_{false};
    bool enable_obstacle_blur_phase2_{false};
    bool enable_lanes_blur_phase2_{false};

    float scale_back_ratio_{1.0};

    bool enable_road_blur_{true};
    bool enable_obstacle_blur_{true};
    bool enable_lanes_blur_{true};
    bool force_closing_no_temporal_{false};

    int drivable_closing_kernel_size_{160};
    bool is_odom_lost_{false};
    
    float obstacle_points_height_max_{0.85};
    float obstacle_points_height_min_{0.0};

    bool phase2_active_;

    pcl::PointCloud<pcl::PointXYZ> obstacle_points;
    
    double covered_distance_{0.0};

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::Time camera_map_check_in, obstacle_points_check_in;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Time last_distance_time_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr obstacle_points_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr vision_occupancy_grid_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr covered_distance_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr click_sub_;
    
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr profile_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debug_pub_;

    cv::cuda::GpuMat gpu_occupancy_image, gpu_clear_img, gpu_blurred_img;
    cv::Ptr<cv::cuda::Filter> filter2D;
    bool filter_initialized = false;
    
    nav_msgs::msg::OccupancyGrid camera_lanes_grid;
    rclcpp::Time camera_lanes_check_in;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr vision_lanes_grid_sub_;

    geometry_msgs::msg::Point start_point_, end_point_;
    bool has_line_{false};
    int click_count_{0};

    void extractAndPublishLineProfile(const nav_msgs::msg::OccupancyGrid& grid_msg) {
        if (!has_line_) return;

        std_msgs::msg::Float32MultiArray profile_msg;

        double dx = end_point_.x - start_point_.x;
        double dy = end_point_.y - start_point_.y;
        double distance = std::hypot(dx, dy);

        int steps = std::max(1, static_cast<int>(distance / grid_map_.cell_size));

        for (int i = 0; i <= steps; ++i) {
            double t = static_cast<double>(i) / steps;
            double current_x = start_point_.x + t * dx;
            double current_y = start_point_.y + t * dy;

            int idx = getIndex(current_x, current_y); 

            float value = -1.0;
            if (idx >= 0 && idx < static_cast<int>(grid_msg.data.size())) {
                value = static_cast<float>(grid_msg.data[idx]);
            }
            profile_msg.data.push_back(value);
        }

        profile_pub_->publish(profile_msg);

    }

    void obstaclePointsCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud_in);


        if (cloud_in->points.empty()) {
            obstacle_points.clear();
            obstacle_points.header.frame_id = "base_link";
            obstacle_points_check_in = this->get_clock()->now();
            return;
        }

        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud_in, *cloud_in, indices);

        if (cloud_in->points.size() < 5) {
            obstacle_points.clear();
            obstacle_points.header.frame_id = "base_link";
            obstacle_points_check_in = this->get_clock()->now();
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
        
        double active_ror_radius = (covered_distance_ > distance_threshold_) ? ror_radius_phase2_ : ror_radius_;
        int active_ror_min_neighbors = (covered_distance_ > distance_threshold_) ? ror_min_neighbors_phase2_ : ror_min_neighbors_;

        outrem.setInputCloud(cloud_in);
        outrem.setRadiusSearch(active_ror_radius);
        outrem.setMinNeighborsInRadius(active_ror_min_neighbors);
        outrem.setKeepOrganized(false);
        outrem.filter(*cloud_filtered);

        cloud_filtered->header.stamp = 0; 

        try
        {
            pcl_ros::transformPointCloud("base_link", *cloud_filtered, obstacle_points, *tf_buffer_);
        }
        catch(tf2::TransformException& ex){
            RCLCPP_WARN(this->get_logger(), "TF hiba (obstacle): %s", ex.what());
            return;
        }

        for (auto& pt : obstacle_points.points) {
            pt.x -= 1.0; 
        }

        obstacle_points.header.frame_id = "base_link";
        obstacle_points_check_in = this->get_clock()->now();
    }

    void visionOccupancyGridCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr input_grid) {
    camera_map_check_in = this->get_clock()->now();

        if (!is_odom_lost_ && !force_closing_no_temporal_) { 
            camera_grid = *input_grid;
            return;
        }

        int height = input_grid->info.height;
        int width = input_grid->info.width;
        size_t grid_size = input_grid->data.size();

        if (grid_size != (size_t)(height * width)) {
            RCLCPP_ERROR(this->get_logger(), "Incoming drivable grid size mismatch.");
            return;
        }

        cv::Mat binary_grid(height, width, CV_8UC1);
        for(size_t i = 0; i < grid_size; ++i) {
            if (input_grid->data[i] == 0) {
                binary_grid.data[i] = 255;
            } else {
                binary_grid.data[i] = 0;
            }
        }

        cv::Mat closed_grid;
        int k_size = std::max(1, drivable_closing_kernel_size_);
        if (k_size % 2 == 0) k_size += 1; 
        
        cv::Mat closing_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k_size, k_size));
        cv::morphologyEx(binary_grid, closed_grid, cv::MORPH_CLOSE, closing_kernel);

        camera_grid.data.resize(grid_size);
        camera_grid.info = input_grid->info;
        camera_grid.header = input_grid->header;
        
        for(size_t i = 0; i < grid_size; ++i) {
            if (closed_grid.data[i] == 255) {
                camera_grid.data[i] = 0; 
            } else {
                camera_grid.data[i] = -1; 
            }
        }
    }

    void visionLanesGridCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr input_grid) {
        camera_lanes_grid = *input_grid;
        camera_lanes_check_in = this->get_clock()->now();
    }

    void distance_callback(const std_msgs::msg::Float32::SharedPtr input_distance) {
        covered_distance_ = input_distance->data;
    }

    void add_obstacle_points(std::vector<uint8_t> &occ_map)
    {
        std::vector<int> point_counts(occ_map.size(), 0);
        
        for(const auto& point : obstacle_points)
        {
            if (point.z >= obstacle_points_height_min_ && point.z <= obstacle_points_height_max_)
            {
                if (point.x >= exclusion_box_x_min_ && point.x <= exclusion_box_x_max_ &&
                    point.y >= exclusion_box_y_min_ && point.y <= exclusion_box_y_max_) 
                {
                    continue;
                }

                int idx = getIndex(point.x, point.y);
                if (idx >= 0 && idx < static_cast<int>(occ_map.size())) {
                    point_counts[idx]++;
                }
            }
        }

        for (int y = 0; y < grid_map_.cell_num_y; ++y) {
            for (int x = 0; x < grid_map_.cell_num_x; ++x) {
                int idx = y * grid_map_.cell_num_x + x;
                
                if (point_counts[idx] > 0) {
                    geometry_msgs::msg::Point pt = index_to_point(idx);
                    double dist = std::hypot(pt.x, pt.y);
                    
                    int neighborhood_points = 0;
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            int nx = x + dx;
                            int ny = y + dy;
                            if (nx >= 0 && nx < grid_map_.cell_num_x && ny >= 0 && ny < grid_map_.cell_num_y) {
                                neighborhood_points += point_counts[ny * grid_map_.cell_num_x + nx];
                            }
                        }
                    }

                    int required_points = 1;
                    
                    if (dist < 6.0) {
                        required_points = 4;
                    } else if (dist < 12.0) {
                        required_points = 2;
                    } else {
                        required_points = 1;
                    }

                    if (neighborhood_points >= required_points) {
                        occ_map[idx] = 100;
                    }
                }
            }
        }
    }

    cv::Mat post_processing(cv::Mat &raw_image, cv::Mat &obstacle_image, cv::Mat &lanes_image)
    {
        cv::Mat clear_img, blurred_img, blurred_obstacle_img, blurred_lanes_img, map;

        bool is_phase2 = (covered_distance_ > distance_threshold_);

        if (is_phase2 && !phase2_active_) {
            phase2_active_ = true;
            RCLCPP_INFO(this->get_logger(), 
                "TÁVOLSÁGI KÜSZÖB ELÉRVE (%.2f m / %.2f m)! Átváltás Phase 2 paraméterekre!", 
                covered_distance_, distance_threshold_);
        } else if (!is_phase2 && phase2_active_) {
            phase2_active_ = false;
            RCLCPP_INFO(this->get_logger(), 
                "Távolság a küszöb alá csökkent (%.2f m / %.2f m). Visszaállás Phase 1-re!", 
                covered_distance_, distance_threshold_);
        }
        
        float active_obstacle_fill_weight = is_phase2 ? obstacle_fill_weight_phase2_ : obstacle_fill_weight_;
        float active_lanes_fill_weight    = is_phase2 ? lanes_fill_weight_phase2_ : lanes_fill_weight_;
        
        float active_blur_kernel_x          = is_phase2 ? blur_kernel_x_phase2_ : blur_kernel_x_;
        float active_blur_kernel_y          = is_phase2 ? blur_kernel_y_phase2_ : blur_kernel_y_;
        float active_obstacle_blur_kernel_x = is_phase2 ? obstacle_blur_kernel_x_phase2_ : obstacle_blur_kernel_x_;
        float active_obstacle_blur_kernel_y = is_phase2 ? obstacle_blur_kernel_y_phase2_ : obstacle_blur_kernel_y_;
        float active_lanes_blur_kernel_x    = is_phase2 ? lanes_blur_kernel_x_phase2_ : lanes_blur_kernel_x_;
        float active_lanes_blur_kernel_y    = is_phase2 ? lanes_blur_kernel_y_phase2_ : lanes_blur_kernel_y_;

        bool active_enable_road_blur = is_phase2 ? enable_road_blur_phase2_ : enable_road_blur_;
        bool active_enable_obstacle_blur = is_phase2 ? enable_obstacle_blur_phase2_ : enable_obstacle_blur_;
        bool active_enable_lanes_blur = is_phase2 ? enable_lanes_blur_phase2_ : enable_lanes_blur_;

        auto make_odd = [](float val) {
            int int_val = std::max(1, static_cast<int>(val));
            return (int_val % 2 == 0) ? int_val + 1 : int_val;
        };

        // Út blur
        if (active_enable_road_blur) {
            gpu_occupancy_image.upload(raw_image);

            static float last_used_kernel_x = -1.0;
            static float last_used_kernel_y = -1.0;

            int kx_road = make_odd(active_blur_kernel_x * scale_back_ratio_);
            int ky_road = make_odd(active_blur_kernel_y * scale_back_ratio_);

            if (!filter_initialized || active_blur_kernel_x != last_used_kernel_x || active_blur_kernel_y != last_used_kernel_y) {
                filter2D = cv::cuda::createGaussianFilter(raw_image.type(), raw_image.type(), cv::Size(kx_road, ky_road), 0, 0);
                filter_initialized = true;
                last_used_kernel_x = active_blur_kernel_x;
                last_used_kernel_y = active_blur_kernel_y;
            }

            cv::cuda::bilateralFilter(gpu_occupancy_image, gpu_clear_img, 20, 60, 60);
            filter2D->apply(gpu_clear_img, gpu_blurred_img);
            gpu_blurred_img.download(blurred_img);  
        } else {
            blurred_img = raw_image.clone();
        }

        // // Út blur (nincs belógás a dobozba)
        // if (active_enable_road_blur) {
        //     gpu_occupancy_image.upload(raw_image);

        //     static float last_used_kernel_x = -1.0;
        //     static float last_used_kernel_y = -1.0;

        //     int kx_road = make_odd(active_blur_kernel_x * scale_back_ratio_);
        //     int ky_road = make_odd(active_blur_kernel_y * scale_back_ratio_);

        //     if (!filter_initialized || active_blur_kernel_x != last_used_kernel_x || active_blur_kernel_y != last_used_kernel_y) {
        //         filter2D = cv::cuda::createGaussianFilter(raw_image.type(), raw_image.type(), cv::Size(kx_road, ky_road), 0, 0);
        //         filter_initialized = true;
        //         last_used_kernel_x = active_blur_kernel_x;
        //         last_used_kernel_y = active_blur_kernel_y;
        //     }

        //     cv::cuda::bilateralFilter(gpu_occupancy_image, gpu_clear_img, 20, 60, 60);
        //     filter2D->apply(gpu_clear_img, gpu_blurred_img);
        //     gpu_blurred_img.download(blurred_img);  

        //     int min_x_idx = static_cast<int>((grid_map_.topleft_x - freespace_box_x_max_) / grid_map_.cell_size);
        //     int max_x_idx = static_cast<int>((grid_map_.topleft_x - freespace_box_x_min_) / grid_map_.cell_size);
        //     int min_y_idx = static_cast<int>((grid_map_.topleft_y - freespace_box_y_max_) / grid_map_.cell_size);
        //     int max_y_idx = static_cast<int>((grid_map_.topleft_y - freespace_box_y_min_) / grid_map_.cell_size);

        //     min_x_idx = std::max(0, min_x_idx);
        //     max_x_idx = std::min(grid_map_.cell_num_x - 1, max_x_idx);
        //     min_y_idx = std::max(0, min_y_idx);
        //     max_y_idx = std::min(grid_map_.cell_num_y - 1, max_y_idx);

        //     if (min_x_idx <= max_x_idx && min_y_idx <= max_y_idx) {
        //         cv::Rect freespace_rect(min_x_idx, min_y_idx, (max_x_idx - min_x_idx) + 1, (max_y_idx - min_y_idx) + 1);
        //         blurred_img(freespace_rect).setTo(0);
        //     }

        // } else {
        //     blurred_img = raw_image.clone();
        // }

        // // Akadály blur
        // if (active_enable_obstacle_blur) {
        //     int kx = make_odd(active_obstacle_blur_kernel_x);
        //     int ky = make_odd(active_obstacle_blur_kernel_y);
            
        //     cv::Mat drivable_mask;
        //     cv::threshold(raw_image, drivable_mask, drivable_mask_threshold_, 255, cv::THRESH_BINARY_INV);

        //     cv::Mat obstacles_on_road;
        //     cv::bitwise_and(obstacle_image, drivable_mask, obstacles_on_road);
        //     cv::Mat float_obs;
        //     obstacles_on_road.convertTo(float_obs, CV_32F);

        //     cv::GaussianBlur(float_obs, float_obs, cv::Size(kx, ky), 0, 0);

        //     float_obs *= active_obstacle_fill_weight; 
        //     cv::min(float_obs, 100.0f, float_obs);
            
        //     cv::Mat road_blurred_obstacles;
        //     float_obs.convertTo(road_blurred_obstacles, CV_8UC1);
        //     cv::max(road_blurred_obstacles, obstacle_image, blurred_obstacle_img);

        // } else {
        //     cv::Mat float_obs;
        //     obstacle_image.convertTo(float_obs, CV_32F);
        //     float_obs *= active_obstacle_fill_weight;
        //     cv::min(float_obs, 100.0f, float_obs);
        //     float_obs.convertTo(blurred_obstacle_img, CV_8UC1);
        // }

        // Akadály blur
        if (active_enable_obstacle_blur) {
            int kx = make_odd(active_obstacle_blur_kernel_x);
            int ky = make_odd(active_obstacle_blur_kernel_y);
            
            cv::Mat float_obs;
            obstacle_image.convertTo(float_obs, CV_32F);

            cv::GaussianBlur(float_obs, float_obs, cv::Size(kx, ky), 0, 0);

            float_obs *= active_obstacle_fill_weight; 
            cv::min(float_obs, 100.0f, float_obs);
            
            float_obs.convertTo(blurred_obstacle_img, CV_8UC1);

        } else {
            cv::Mat float_obs;
            obstacle_image.convertTo(float_obs, CV_32F);
            float_obs *= active_obstacle_fill_weight;
            cv::min(float_obs, 100.0f, float_obs);
            float_obs.convertTo(blurred_obstacle_img, CV_8UC1);
        }

        // Vonal blur
        if (active_enable_lanes_blur) {
            int kx = make_odd(active_lanes_blur_kernel_x);
            int ky = make_odd(active_lanes_blur_kernel_y);
            
            cv::Mat float_lanes;
            lanes_image.convertTo(float_lanes, CV_32F);

            cv::GaussianBlur(float_lanes, float_lanes, cv::Size(kx, ky), 0, 0);
            
            float_lanes *= active_lanes_fill_weight; 
            cv::min(float_lanes, 100.0f, float_lanes);
            
            float_lanes.convertTo(blurred_lanes_img, CV_8UC1);
            
        } else {
            cv::Mat float_lanes;
            lanes_image.convertTo(float_lanes, CV_32F);
            float_lanes *= active_lanes_fill_weight;
            cv::min(float_lanes, 100.0f, float_lanes);
            float_lanes.convertTo(blurred_lanes_img, CV_8UC1);
        }

        cv::max(raw_image, blurred_img, map);
        cv::max(map, blurred_obstacle_img, map);
        cv::max(map, blurred_lanes_img, map);

        return map;
    }

    geometry_msgs::msg::Point index_to_point(int index)
    {
        geometry_msgs::msg::Point pp;
        
        int cell_x = index % grid_map_.grid_width;
        int cell_y = index / grid_map_.grid_width;

        pp.x = grid_map_.topleft_x - (cell_x * grid_map_.cell_size) - (grid_map_.cell_size / 2.0);
        pp.y = grid_map_.topleft_y - (cell_y * grid_map_.cell_size) - (grid_map_.cell_size / 2.0);

        return pp;
    }
    
    int getIndex(double x, double y) {
        double dx = grid_map_.topleft_x - x;
        double dy = grid_map_.topleft_y - y;

        if (dx < 0.0 || dy < 0.0) {
            return -1; 
        }

        int cell_x = static_cast<int>(dx / grid_map_.cell_size);
        int cell_y = static_cast<int>(dy / grid_map_.cell_size);
        
        if (cell_x >= 0 && cell_x < grid_map_.cell_num_x && 
            cell_y >= 0 && cell_y < grid_map_.cell_num_y) {
            return cell_y * grid_map_.cell_num_x + cell_x;
        } else {
            return -1;
        }
    }
    
    void add_camera_map(std::vector<double> &map, std::vector<float> &occ_distances, bool has_odom)
    {
        if (camera_grid.header.frame_id.empty()) {
             return;
        }
        
        geometry_msgs::msg::TransformStamped transform_msg;
        try
        {
            transform_msg = tf_buffer_->lookupTransform("base_link", camera_grid.header.frame_id, tf2::TimePointZero);
        }catch(tf2::TransformException& ex){
            return;
        }

        tf2::Transform tf2_transform;
        tf2::fromMsg(transform_msg.transform, tf2_transform);

        int height = camera_grid.info.height;
        int width = camera_grid.info.width;
        double res = camera_grid.info.resolution;
        double origin_x = camera_grid.info.origin.position.x;
        double origin_y = camera_grid.info.origin.position.y;

        tf2::Matrix3x3 rot_mat = tf2_transform.getBasis();
        tf2::Vector3 origin = tf2_transform.getOrigin();
        double m00 = rot_mat[0][0], m01 = rot_mat[0][1];
        double m10 = rot_mat[1][0], m11 = rot_mat[1][1];
        double dx = origin.x(), dy = origin.y();

        float active_logs_odds = (covered_distance_ > distance_threshold_) ? logs_odds_increase_camera_phase2_ : logs_odds_increase_camera_;

        for(int y = 0; y < height; y++)
        {
            double py_base = origin_y + (y * res) + (res / 2.0);
            for(int x = 0; x < width; x++)
            {
                int cam_idx = y * width + x;
                if (cam_idx < 0 || cam_idx >= (int)camera_grid.data.size()) continue;

                if(camera_grid.data[cam_idx] == 0)
                {
                    double px_base = origin_x + (x * res) + (res / 2.0);

                    double pt_base_x = m00 * px_base + m01 * py_base + dx - 1.0;
                    double pt_base_y = m10 * px_base + m11 * py_base + dy;

                    int fusion_idx = getIndex(pt_base_x, pt_base_y);

                    if(fusion_idx >= 0 && fusion_idx < (int)map.size())
                    {
                        if(map[fusion_idx] == -3.0) {
                            map[fusion_idx] = has_odom ? 2.0 : 10.0;
                        } else {
                            map[fusion_idx] += active_logs_odds; 
                            if (map[fusion_idx] > 15.0) {
                                map[fusion_idx] = 15.0;
                            }
                        }
                        occ_distances[fusion_idx] = covered_distance_;
                    }
                }
            }
        }
    }

    void add_camera_lanes(std::vector<uint8_t> &occ_map)
    {
        if (camera_lanes_grid.header.frame_id.empty()) {
             return;
        }
        
        geometry_msgs::msg::TransformStamped transform_msg;
        try {
            transform_msg = tf_buffer_->lookupTransform("base_link", camera_lanes_grid.header.frame_id, tf2::TimePointZero);
        } catch(tf2::TransformException& ex) {
            return;
        }

        tf2::Transform tf2_transform;
        tf2::fromMsg(transform_msg.transform, tf2_transform);

        int height = camera_lanes_grid.info.height;
        int width = camera_lanes_grid.info.width;
        double res = camera_lanes_grid.info.resolution;
        double origin_x = camera_lanes_grid.info.origin.position.x;
        double origin_y = camera_lanes_grid.info.origin.position.y;

        tf2::Matrix3x3 rot_mat = tf2_transform.getBasis();
        tf2::Vector3 origin = tf2_transform.getOrigin();
        double m00 = rot_mat[0][0], m01 = rot_mat[0][1];
        double m10 = rot_mat[1][0], m11 = rot_mat[1][1];
        double dx = origin.x(), dy = origin.y();

        for(int y = 0; y < height; y++)
        {
            double py_base = origin_y + (y * res) + (res / 2.0);
            for(int x = 0; x < width; x++)
            {
                int cam_idx = y * width + x;
                if (cam_idx < 0 || cam_idx >= (int)camera_lanes_grid.data.size()) continue;

                if(camera_lanes_grid.data[cam_idx] == 100)
                {
                    double px_base = origin_x + (x * res) + (res / 2.0);

                    double pt_base_x = m00 * px_base + m01 * py_base + dx - 1.0;
                    double pt_base_y = m10 * px_base + m11 * py_base + dy;

                    int fusion_idx = getIndex(pt_base_x, pt_base_y);

                    if(fusion_idx >= 0 && fusion_idx < (int)occ_map.size())
                    {
                        occ_map[fusion_idx] = 100;
                    }
                }
            }
        }
    }
    
    void occ_to_global(const geometry_msgs::msg::TransformStamped &transform_msg, std::vector<double> &occ_map, std::vector<float> &occ_distances)
    {
        tf2::Transform tf2_transform;
        tf2::fromMsg(transform_msg.transform, tf2_transform);

        GlobalPoint g_point;
        global_occ.reserve(occ_map.size()); 

        for(size_t i = 0; i < occ_map.size(); i++)
        {
            if (occ_map[i] == -3.0) continue; 

            geometry_msgs::msg::Point current_point = index_to_point(i);
            
            tf2::Vector3 pt_vec(current_point.x, current_point.y, current_point.z);
            tf2::Vector3 pt_global_vec = tf2_transform * pt_vec;

            g_point.point.x = pt_global_vec.x();
            g_point.point.y = pt_global_vec.y();
            g_point.point.z = pt_global_vec.z();
            g_point.occupancy = occ_map[i];
            g_point.distance = occ_distances[i];
            
            global_occ.push_back(g_point);
        }
    }

    void global_to_occ(const geometry_msgs::msg::TransformStamped &transform_msg, std::vector<double> &occ_map, std::vector<float> &occ_distances)
    {
        tf2::Transform tf2_transform;
        tf2::fromMsg(transform_msg.transform, tf2_transform);

        for(const auto &g_point : global_occ)
        {
            if((covered_distance_ - g_point.distance) < distance_covered_confidence_)
            {
                tf2::Vector3 pt_global_vec(g_point.point.x, g_point.point.y, g_point.point.z);
                tf2::Vector3 pt_local_vec = tf2_transform * pt_global_vec;

                int index = getIndex(pt_local_vec.x(), pt_local_vec.y());

                if(index >= 0 && index < static_cast<int>(occ_map.size()))
                {
                    occ_map[index] = g_point.occupancy;
                    occ_distances[index] = g_point.distance;
                }
            }
        }
        
        global_occ.clear();
    }

    double get_occupancy(double log_odds)
    {
      if(log_odds == -1) return -0.01;
      return 1.0 / (1 + exp(log_odds));
    }

    void apply_box_to_map(std::vector<uint8_t> &occ_map, double x_min_world, double x_max_world, double y_min_world, double y_max_world, uint8_t fill_value)
    {
        int min_x_idx = static_cast<int>((grid_map_.topleft_x - x_max_world) / grid_map_.cell_size);
        int max_x_idx = static_cast<int>((grid_map_.topleft_x - x_min_world) / grid_map_.cell_size);
        int min_y_idx = static_cast<int>((grid_map_.topleft_y - y_max_world) / grid_map_.cell_size);
        int max_y_idx = static_cast<int>((grid_map_.topleft_y - y_min_world) / grid_map_.cell_size);

        min_x_idx = std::max(0, min_x_idx);
        max_x_idx = std::min(grid_map_.cell_num_x - 1, max_x_idx);
        min_y_idx = std::max(0, min_y_idx);
        max_y_idx = std::min(grid_map_.cell_num_y - 1, max_y_idx);

        for (int y = min_y_idx; y <= max_y_idx; ++y) {
            for (int x = min_x_idx; x <= max_x_idx; ++x) {
                int index = y * grid_map_.cell_num_x + x;
                if (index >= 0 && index < static_cast<int>(occ_map.size())) { 
                    occ_map[index] = fill_value; 
                }
            }
        }
    }
    
    void Mapping()
    {
        nav_msgs::msg::OccupancyGrid occupancyGrid_msg;
        
        grid_map_.initGrid(occupancyGrid_msg);
        grid_map_.paramRefresh();
        size_t map_size = grid_map_.cell_num_x * grid_map_.cell_num_y;

        std::vector<double> occ_points(map_size, -3.0);
        std::vector<float> occ_distances(map_size, 0.0);
        
        bool has_odom = true;
        geometry_msgs::msg::TransformStamped transform_odom_base;
        geometry_msgs::msg::TransformStamped transform_base_odom;

        try {
            transform_odom_base = tf_buffer_->lookupTransform("base_link", "odom", tf2::TimePointZero);
            transform_base_odom = tf_buffer_->lookupTransform("odom", "base_link", tf2::TimePointZero);

            rclcpp::Time tf_time(transform_odom_base.header.stamp);

            static int64_t last_tf_ns = 0;
            static int tf_frozen_cycles = 0;
            if (tf_time.nanoseconds() == last_tf_ns) {
                tf_frozen_cycles++;
            } else {
                last_tf_ns = tf_time.nanoseconds();
                tf_frozen_cycles = 0;
            }

            static double last_distance_val = -1.0;
            static int dist_frozen_cycles = 0;
            
            if (std::abs(covered_distance_ - last_distance_val) < 0.0001) {
                dist_frozen_cycles++;
            } else {
                last_distance_val = covered_distance_;
                dist_frozen_cycles = 0; 
            }

            if (tf_frozen_cycles > 20 || dist_frozen_cycles > 20) {
                has_odom = false;
            }

        } catch(tf2::TransformException& ex) {
            has_odom = false;
        }

        is_odom_lost_ = !has_odom;

        static int print_counter = 0;
        print_counter++;
        if (print_counter >= 20) {
            RCLCPP_INFO(this->get_logger(), "Odom allapota: %s | Megtett ut: %.2f m", has_odom ? "OK" : "ELVESZETT", covered_distance_);
            print_counter = 0;
        }

        bool use_temporal = has_odom && !force_closing_no_temporal_;

        if (!use_temporal) {
            global_occ.clear();
        }

        if (use_temporal) {
            global_to_occ(transform_odom_base, occ_points, occ_distances);
        }

        if(this->get_clock()->now() - camera_map_check_in < std::chrono::milliseconds(1000))
        {
            add_camera_map(occ_points, occ_distances, use_temporal);
        }
        if (use_temporal) {
            occ_to_global(transform_base_odom, occ_points, occ_distances);
        }

        std::vector<uint8_t> occ_map_pre;
        occ_map_pre.reserve(occ_points.size());
        for(const auto &val : occ_points)
            occ_map_pre.push_back(get_occupancy(val) * 100);  
        
        apply_box_to_map(occ_map_pre, freespace_box_x_min_, freespace_box_x_max_, freespace_box_y_min_, freespace_box_y_max_, 0);

        cv::Mat occupancyImage = cv::Mat(grid_map_.cell_num_y, grid_map_.cell_num_x, CV_8UC1, occ_map_pre.data());
        
        std::vector<uint8_t> occ_obs(occ_points.size(), 0);
        if(this->get_clock()->now() - obstacle_points_check_in < std::chrono::milliseconds(1000)) {
            add_obstacle_points(occ_obs);
        }

        std::vector<uint8_t> occ_lanes(occ_points.size(), 0);
        if(this->get_clock()->now() - camera_lanes_check_in < std::chrono::milliseconds(1000)) {
            add_camera_lanes(occ_lanes);
        }

        cv::Mat obstacleImage  = cv::Mat(grid_map_.cell_num_y, grid_map_.cell_num_x, CV_8UC1, occ_obs.data());
        cv::Mat lanesImage = cv::Mat(grid_map_.cell_num_y, grid_map_.cell_num_x, CV_8UC1, occ_lanes.data());

        cv::Mat mapImage = post_processing(occupancyImage, obstacleImage, lanesImage);

        cv::Mat flat = mapImage.reshape(1, mapImage.total() * mapImage.channels());
        std::vector<uint8_t> final_data;
        if (mapImage.isContinuous()) {
            final_data.assign(flat.data, flat.data + flat.total());
        } else {
            cv::Mat continuous_flat = flat.clone();
            final_data.assign(continuous_flat.data, continuous_flat.data + continuous_flat.total());
        }

        if(this->get_clock()->now() - obstacle_points_check_in < std::chrono::milliseconds(1000))
        {
            add_obstacle_points(final_data); 
        }

        if(this->get_clock()->now() - camera_lanes_check_in < std::chrono::milliseconds(1000))
        {
            add_camera_lanes(final_data); 
        }

        apply_box_to_map(final_data, exclusion_box_x_min_, exclusion_box_x_max_, exclusion_box_y_min_, exclusion_box_y_max_, 0); 
        
        for (auto& val : final_data) {
            if (val == 99) val = 98;
        }

        occupancyGrid_msg.data.reserve(final_data.size());
        for(const auto& val : final_data) {
            occupancyGrid_msg.data.push_back(static_cast<int8_t>(val));
        }
        
        occupancy_grid_pub_->publish(occupancyGrid_msg);

        extractAndPublishLineProfile(occupancyGrid_msg);

        std_msgs::msg::String msg;
        std::string status = "";

        if (!has_odom) {
            status += "NO ODOM - CURRENT ONLY | ";
        }

        if(this->get_clock()->now() - obstacle_points_check_in < std::chrono::milliseconds(1000)) {
            status += "Obstacle OK | ";
        } else {
            status += "Obstacle ERROR | ";
        }

        if(this->get_clock()->now() - camera_map_check_in < std::chrono::milliseconds(1000)) {
            status += "Camera OK";
        } else {
            status += "Camera ERROR";
        }

        msg.data = status;
    }
    
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PrcpGridFusionNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}