#pragma once

#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <fstream>

class PointFinder
{

public:
    // A flag used to tell other objects that they should run their apply()
    static std::atomic_bool applyNext;
    // Input mask (pixels to ignore in feature extraction)
    cv::Mat mask;

    PointFinder();

    // Loads image and mask from disk
    PointFinder(std::string path, std::string Mname);

    // Detects keypoints and extracts descriptors from image, using internal ORB
    void apply(std::vector<cv::KeyPoint>& keyPoints, cv::Mat& descriptors, cv::Mat P);

    static void setInits(bool value);

    static void setORBscaleFactor(float scaleFactor);

    static void setORBnLevels(int nLevels);

    static void setORBedgeThreshold(int threshold);

    static void setORBpatchSize(int size);

    static void setORBfastThreshold(int threshold);

    static void setORBfirstLevel(int firstLevel);

    // Returns original image size (imgOrg)
    cv::Size getImgSize();

private:
    cv::Mat imgOrg;
    cv::Mat img;

    // Shared ORB detector instance (configuration applies globally)
    static cv::Ptr<cv::ORB> ORB;
    cv::Mat K;
    std::vector<double> D;

    // Internal flags — init[0] = new data extraction, image preprocessing, init[1] = other change
    static std::atomic_bool init[2];

    // Prepares the image for feature extraction
    void preprocess();

    // Clones the internal ORB instance with current parameters
    cv::Ptr<cv::ORB> cloneORB() const;
};