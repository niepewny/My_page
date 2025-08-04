#pragma once

#include <fstream>
#include <future>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/radius_outlier_removal.h>
#include "PointReconstructor.h"
#include "PointFinderr.h"
#include <pcl/io/pcd_io.h>

class CloudMaker
{
public:

    std::vector<PointFinder> pointFinder;
    std::vector<PointReconstructor> pointReconstructor;
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> radiousFilter;

    //@directory is the path to a directory containing all the required data
    CloudMaker(std::string directory);
    ~CloudMaker();

    pcl::PointCloud<pcl::PointXYZ>::Ptr apply();

private:
    int numOfImgs;

    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector <cv::Mat> descriptors;
    std::vector<cv::Mat> points3d;
    std::vector<cv::Mat> P;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;

    // Skips data while reading from file stream (can skip line-by-line or element-by-element depending on 'line' flag)
    void skipData(std::ifstream& file, int numOfData, bool line = 0);

    // Converts a column-wise cv::Mat to a std::vector<T>
    template <typename T>
    void mat2vec(cv::Mat& mat, std::vector<T>& vec);

    void cameraExtrinsics(std::vector<std::string> path, std::vector<cv::Mat>& P);

    // Merges and validates multiple sets of 3D points reconstructed from different views
    // Handles NaNs and empty matrices, returns a single combined cv::Mat
    cv::Mat checkAndJoinResults(std::vector<cv::Mat> points3d);

    void exportPoints(std::vector<cv::Mat> points3d, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    // Filters points using a set of image masks and their corresponding projection matrices
    // Keeps only the 3D points that project inside all given masks
    void filterByMask(cv::Mat& pointsInOut, std::vector<cv::Mat>projectionMatrices, std::vector<cv::Mat> masks);
};