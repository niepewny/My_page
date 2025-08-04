#include "PointFinderr.h"

cv::Ptr<cv::ORB> PointFinder::ORB;
std::atomic_bool PointFinder::init[2] = {1, 1};
std::atomic_bool PointFinder::applyNext;

PointFinder::PointFinder() {}

PointFinder::PointFinder(std::string path, std::string Mname)
{
    ORB = cv::ORB::create(100000, 1.1f, 60, 5,
        0, 2, cv::ORB::HARRIS_SCORE, 5, 20);

    imgOrg = cv::imread(path);
    mask = cv::imread(Mname, cv::IMREAD_GRAYSCALE);
    cv::bitwise_not(mask, mask);
    if (imgOrg.size == 0)
    {
        std::cout << "Image reading error :(" << std::endl;
        exit(0);
    }
    D = { 0, 0, 0, 0 };

}


void PointFinder::apply(std::vector<cv::KeyPoint>& keyPoints, cv::Mat& descriptors, cv::Mat P)
{
    cv::Mat R, T, euler;
    bool localInit[2] = {init[0], init[1]};
    if (localInit[0])
    {
        cv::decomposeProjectionMatrix(P, K, R, T, cv::noArray(), cv::noArray(), cv::noArray(), euler);
        preprocess();
        localInit[1] = 1;
    }
    if (localInit[1])
    {
        cv::Ptr<cv::ORB> localORB = cloneORB();

        localORB->detect(img, keyPoints, mask);


        for (int i = 0; i < keyPoints.size(); i++)
        {
            if (keyPoints[i].angle > 0)
                keyPoints[i].angle = euler.at<double>(2, 0);
            else
                keyPoints[i].angle = 360.0 + euler.at<double>(2, 0);
        }

        localORB->compute(img, keyPoints, descriptors);

        applyNext = 1;
    }

}


void PointFinder::preprocess()
{
    cv::cvtColor(imgOrg, img, cv::COLOR_BGR2GRAY);
    cv::undistort(img.clone(), img, K, D);

    cv::cuda::GpuMat GPUimg(img), GPUmask(mask);

    cv::cuda::bitwise_and(GPUimg, GPUmask, GPUimg);

    cv::cuda::equalizeHist(GPUimg, GPUimg);


    GPUimg.download(img);
}

cv::Ptr<cv::ORB> PointFinder::cloneORB() const {
    return cv::ORB::create(
        ORB->getMaxFeatures(),
        ORB->getScaleFactor(),
        ORB->getNLevels(),
        ORB->getEdgeThreshold(),
        ORB->getFirstLevel(),
        ORB->getWTA_K(),
        ORB->getScoreType(),
        ORB->getPatchSize(),
        ORB->getFastThreshold()
    );
}

void PointFinder::setInits(bool value) {
    init[0] = value;
    init[1] = value;
}

void PointFinder::setORBscaleFactor(float scaleFactor)
{
    if (ORB->getScaleFactor() != scaleFactor)
    {
        ORB->setScaleFactor(scaleFactor);
        init[1] = 1;
    }
}
void PointFinder::setORBnLevels(int nLevels)
{
    if (ORB->getNLevels() != nLevels)
    {
        ORB->setNLevels(nLevels);
        init[1] = 1;
    }
}
void PointFinder::setORBedgeThreshold(int threshold)
{
    if (ORB->getEdgeThreshold() != threshold)
    {
        ORB->setEdgeThreshold(threshold);
        init[1] = 1;
    }
}
void PointFinder::setORBpatchSize(int size)
{
    if (ORB->getPatchSize() != size)
    {
        ORB->setPatchSize(size);
        init[1] = 1;
    }
}
void PointFinder::setORBfastThreshold(int threshold)
{
    if (ORB->getFastThreshold() != threshold)
    {
        ORB->setFastThreshold(threshold);
        init[1] = 1;
    }
}
void PointFinder::setORBfirstLevel(int firstLevel)
{
    if (ORB->getFirstLevel() != firstLevel)
    {
        ORB->setFirstLevel(firstLevel);
        init[1] = 1;
    }
}

cv::Size PointFinder::getImgSize()
{
    return cv::Size(imgOrg.cols, imgOrg.rows);
}
