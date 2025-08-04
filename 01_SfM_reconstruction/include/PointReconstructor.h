#pragma once

#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/cudafeatures2d.hpp>

class PointReconstructor
{
public:

    static bool applyNext;

    PointReconstructor(cv::Mat projL, cv::Mat projR);

    // Main entry point — performs matching, optional RANSAC filtering and triangulation between a stereo pair
    // If initAll is true the reconstruction starts from the beginning. If not, only steps that are required based on prior changes are made.
    // Outputs a matrix of 3D points (euclidean) or empty matrix if matching failed
    void apply(std::vector<cv::KeyPoint>& keypointsl, cv::Mat descriptorsl, std::vector<cv::KeyPoint>& keypointsr, cv::Mat& descriptorsr, cv::Mat& points3d, bool initAll);

    //setters:

    static void PointReconstructor::setZmin(float zmin);

    static void PointReconstructor::setZmax(float zmax);

    static void PointReconstructor::setdEuler(float deuler);

    static void PointReconstructor::setdT(float dt);

    static void PointReconstructor::setRansacThreshold(float ransacthreshold);

    static void PointReconstructor::setMaxDistance(float maxdistance);

    static void PointReconstructor::setMinDistRatio(float mindistratio);

    static void PointReconstructor::setUseRansac(bool useransac);

    static void PointReconstructor::setInits(bool value);


private:
    static bool init[2], useRansac;
    static double ransacThreshold;
    static float Zmin, Zmax, maxDistance, minDistRatio, dEuler, dT;
    std::vector<cv::Mat> projMatr;
    std::vector<cv::Mat> K;
    cv::Mat matchingMask, RTl2r;
    std::vector<std::vector<double>> D;
    std::vector < std::vector < cv::Point_<float>>> matchedPoints;

    //splits columns of matrix and stores it in vector of 1-column matrixes
    void mat2vec(cv::Mat& mat, std::vector<cv::Mat>& vec);

    //matches points from left camera images with points from right camera images and pre-filters the matches
    void match(std::vector<cv::KeyPoint>& keypointsL, cv::Mat& descriptorsL, std::vector<cv::KeyPoint>& keypointsR, cv::Mat& descriptorsR,
        std::vector < std::vector < cv::Point_<float>>>& matchedPoints);
   
    // Converts a matrix of homogeneous 3D points to Euclidean 3D coordinates
    void determine3d(cv::Mat hPoints, std::vector<cv::Mat>projMatr, cv::Mat& points3d);
    
    // Triangulates corresponding points from two views using the linear DLT method
    // Input: matched 2D points and their corresponding projection matrices
    // Output: homogeneous 3D points (4xN matrix)
    void triangulateLinear(cv::Mat& mat_P_l, cv::Mat& mat_P_r, std::vector < cv::Point_<float>>& warped_back_l, std::vector < cv::Point_<float>>& warped_back_r, cv::Mat& homopoints);
};