#include "CloudMaker.h"
#include <math.h>
#include <unordered_map>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>


CloudMaker::CloudMaker(std::string directory)
{
    std::vector<std::string> IMnames, Dnames, Mnames;

    cv::glob(directory + "/*.png", IMnames, false);
    cv::glob(directory + "/*txt", Dnames, false);
    cv::glob(directory + "/*jpg", Mnames, false);
    int numOfImgs = IMnames.size();

    keypoints = std::vector<std::vector<cv::KeyPoint>>(numOfImgs);
    descriptors = std::vector <cv::Mat>(numOfImgs);
    points3d = std::vector<cv::Mat>(numOfImgs - 1);
    cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    P = std::vector<cv::Mat>(numOfImgs);

    cameraExtrinsics(Dnames, P);

    std::vector<std::thread> Thr;
    pointFinder = std::vector<PointFinder>(numOfImgs);

    for (int i = 0; i < numOfImgs; i++)
    {
        Thr.emplace_back([&, i]() { pointFinder[i] = PointFinder::PointFinder(IMnames[i], Mnames[i]); });
    }

    for (int i = 0; i < Thr.size(); i++)
    {
        Thr[i].join();
    }
    Thr.clear();

    for (int i = 0; i < numOfImgs - 1; i++)
    {
        pointReconstructor.push_back(PointReconstructor::PointReconstructor(P[i], P[i + 1]));
    }

    radiousFilter = pcl::RadiusOutlierRemoval<pcl::PointXYZ>(true);
    radiousFilter.setRadiusSearch(0.35);
    radiousFilter.setMinNeighborsInRadius(7);
}

CloudMaker::~CloudMaker() {}

pcl::PointCloud<pcl::PointXYZ>::Ptr CloudMaker::apply()
{
    std::vector<std::thread> Thr;

    for (int i = 0; i < pointFinder.size(); i++)
    {
        Thr.emplace_back(&PointFinder::apply, &pointFinder[i], std::ref(keypoints[i]), std::ref(descriptors[i]), std::ref(P[i]));
    }
    for (int i = 0; i < Thr.size(); i++)
    {
        Thr[i].join();
    }
    Thr.clear();

    PointFinder::setInits(0);

    for (int i = 0; i < pointReconstructor.size(); i++)
    {
        pointReconstructor[i].apply(keypoints[i], descriptors[i], keypoints[i + 1], descriptors[i + 1], points3d[i], PointFinder::applyNext);
    }

    PointFinder::applyNext = 0;
    PointReconstructor::setInits(0);

    std::vector<cv::Mat> masks;
    for (int i = 0; i < pointFinder.size(); i++)
    {
        masks.push_back(pointFinder[i].mask.clone());
    }

    cv::Mat concatenatedCloud = checkAndJoinResults(points3d);

    filterByMask(concatenatedCloud, P, masks);
    points3d = { concatenatedCloud };

    if (PointReconstructor::applyNext)
    {
        exportPoints(points3d, cloud);
        PointReconstructor::applyNext = 0;
    }

    radiousFilter.setInputCloud(cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);
    radiousFilter.filter(*temp);
    cloud.swap(temp);

    return cloud;
}

cv::Mat CloudMaker::checkAndJoinResults(std::vector<cv::Mat> points3d)
{
    cv::Mat concatenatedCloud;

    std::vector<cv::Mat> nonEmptyPoints3d;

    for (const auto& mat : points3d) {
        if (!mat.empty()) {
            nonEmptyPoints3d.push_back(mat);
        }
    }

    if (!nonEmptyPoints3d.empty()) {
        cv::vconcat(nonEmptyPoints3d, concatenatedCloud);
    }
    else {
        concatenatedCloud = cv::Mat();
    }

    return concatenatedCloud;
}

void CloudMaker::filterByMask(cv::Mat& pointsInOut, std::vector<cv::Mat>projectionMatrices, std::vector<cv::Mat> masks)
{
    std::vector < cv::Mat> matVec;
    cv::Mat matConcat, matFiltered, point;

    pointsInOut = pointsInOut.t();
    cv::Mat homo = cv::Mat::ones(cv::Size(pointsInOut.cols, 1), CV_64F);
    cv::split(pointsInOut, matVec);
    matVec.push_back(homo);
    cv::vconcat(matVec, matConcat);
    matVec.clear();
    int cols = masks[0].cols;
    int rows = masks[0].rows;
    int x, y;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    int outliers = 0;
    int cl = 0;

    for (int i = 0; i < masks.size(); i++)
    {
        cv::dilate(masks[i], masks[i], element);

        matFiltered = projectionMatrices[i] * matConcat;

        for (int j = 0; j < matConcat.cols; j++)
        {
            point = matFiltered.col(j);
            point /= point.at<double>(2, 0);
            x = point.at<double>(0, 0);
            y = point.at<double>(1, 0);
            if (x > 0 && x < cols && y > 0 && y < rows)
            {
                if (masks[i].at<unsigned char>(y, x) == 255)
                {
                    matVec.push_back(matConcat.col(j));
                    cl++;
                }
                else
                {
                    outliers++;
                }
            }
            else
            {
                matVec.push_back(matConcat.col(j));
                cl++;
            }
        }
        cv::hconcat(matVec, matConcat);
        matVec.clear();
    }
    matConcat = matConcat(cv::Rect(0, 0, matConcat.cols, 3));
    for (int i = 0; i < matConcat.rows; i++)
    {
        matVec.push_back(matConcat.row(i));
    }
    cv::merge(matVec, pointsInOut);
    pointsInOut = pointsInOut.t();
}


void CloudMaker::skipData(std::ifstream& file, int numOfData, bool line)
{
    std::string skipper;

    if (line)
    {
        for (int i = 0; i < numOfData; ++i) {
            getline(file, skipper, '\n');
        }
    }
    else
    {
        for (int i = 0; i < numOfData; i++)
        {
            file >> skipper;
        }
    }
}
template <typename T>
void CloudMaker::mat2vec(cv::Mat& mat, std::vector<T>& vec)
{
    for (int i = 0; i < mat.rows; i++)
    {
        vec.push_back(mat.at<T>(i, 0));
    }
}

void::CloudMaker::cameraExtrinsics(std::vector<std::string> path, std::vector<cv::Mat>& P)
{
    std::ifstream file;

    for (int i = 0; i < path.size(); i++)
    {
        file.open(path[i]);

        if (!file)
        {
            std::cout << "Bad extrinsics .txt file" << std::endl;
            exit(1);
        }
         P[i] = cv::Mat::zeros(3, 4, CV_64F);

        skipData(file, 1, 1);

        for (int y = 0; y < 3; y++)
            for (int x = 0; x < 4; x++)
            {
                {
                    file >> P[i].at<double>(y, x);
                }
            }
        file.close();
    }
}


void CloudMaker::exportPoints(std::vector<cv::Mat> points3d, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    cloud->clear();
    cv::Vec3f data;
    pcl::PointXYZ point;
    int width = 0;
    for (int i = 0; i < points3d.size(); i++)
    {
        points3d[i].convertTo(points3d[i], CV_32F);

        for (int j = 0; j < points3d[i].rows; j++)
        {
            data = points3d[i].at<cv::Vec3f>(j, 0);
            point.x = data[0];
            point.y = data[1];
            point.z = data[2];
            cloud->points.push_back(point);
        }
    }
}

