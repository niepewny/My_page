#include "PointReconstructor.h"
#include "MatchingMaskBuilder.h"


    float PointReconstructor::Zmin;
    float PointReconstructor::Zmax;
    float PointReconstructor::dEuler;
    float PointReconstructor::dT;
    double PointReconstructor::ransacThreshold;
    float PointReconstructor::maxDistance;
    float PointReconstructor::minDistRatio;
    bool PointReconstructor::useRansac;
    bool PointReconstructor::init[2] = { 1, 1 };
    bool PointReconstructor::applyNext = 1;

    void PointReconstructor::setZmin(float zmin){
        if (Zmin != zmin) { Zmin = zmin; init[0] = 1; }
    }

    void PointReconstructor::setZmax(float zmax){
        if (Zmax != zmax) { Zmax = zmax; init[0] = 1;}
    }

    void PointReconstructor::setdEuler(float deuler){
        if (dEuler != deuler) { dEuler = deuler; init[0] = 1; }
    }

    void PointReconstructor::setdT(float dt) {
        if (dT != dt) { dT = dt; init[0] = 1; }
    }


    void PointReconstructor::setRansacThreshold(float ransacthreshold) {
        if (ransacThreshold != ransacthreshold) { ransacThreshold = ransacthreshold; init[1] = 1; }
    }

    void PointReconstructor::setMaxDistance(float maxdistance) {
        if (maxDistance != maxdistance) { maxDistance = maxdistance; init[1] = 1; }
    }

    void PointReconstructor::setMinDistRatio(float mindistratio) {
        if (minDistRatio != mindistratio) { minDistRatio = mindistratio; init[1] = 1; }
    }

    void PointReconstructor::setUseRansac(bool useransac) {
        if (useRansac != useransac) { useRansac = useransac; init[1] = 1; }
    }

    void PointReconstructor::setInits(bool value) {
        init[0] = value;
        init[1] = value;
    }


    PointReconstructor::PointReconstructor(cv::Mat projL, cv::Mat projR)
    {
        projMatr = { projL, projR };        
        D = { { 0, 0, 0, 0 }, { 0, 0, 0, 0 } };

        cv::Mat k, R, T;
        for (int i = 0; i < 2; i++)
        {
            cv::decomposeProjectionMatrix(projMatr[i], k, R, T);
            K.push_back(k);
        }

        matchedPoints = std::vector < std::vector < cv::Point_<float>>>(2);

    }

    void PointReconstructor::apply(std::vector<cv::KeyPoint>& keypointsL, cv::Mat descriptorsL,
        std::vector<cv::KeyPoint>& keypointsR, cv::Mat& descriptorsR, cv::Mat& points3d, bool initAll)
    {
        if (init[0] || initAll)
        {
            MatchingMaskBuilder::Params P;
            P.Zmin = Zmin; P.Zmax = Zmax;
            P.dEuler = dEuler; P.dT = dT;
            P.gridCell = 8;
            P.useParallel = true;

            MatchingMaskBuilder builder;

            matchingMask = builder.build(keypointsL, keypointsR,
                projMatr[0], projMatr[1],
                K[0], K[1], P);

            init[1] = 1;
        }
        if (init[1])
        {
            matchedPoints[0].clear();
            matchedPoints[1].clear();
            match(keypointsL, descriptorsL, keypointsR, descriptorsR, matchedPoints);


            if (matchedPoints[0].size() > 50)
            {
                cv::Mat ransacMask;
                cv::Mat E;
                if (useRansac == 0)
                {
                    E = cv::findEssentialMat(matchedPoints[0], matchedPoints[1], K[0], D[0], K[1], D[1], cv::LMEDS, 0.999, ransacThreshold, ransacMask);
                }
                else
                {
                    E = cv::findEssentialMat(matchedPoints[0], matchedPoints[1], K[0], D[0], K[1], D[1], cv::RANSAC, 0.999, ransacThreshold, ransacMask);
                }


                std::vector<std::vector<cv::Point_<float>>> goodMatches(2);
                cv::Mat hPoints;

                for (int i = 0; i < ransacMask.rows; i++)
                {
                    if (ransacMask.at<unsigned char>(i, 0) == 1)
                    {
                        goodMatches[0].push_back(matchedPoints[0][i]);
                        goodMatches[1].push_back(matchedPoints[1][i]);
                    }
                }

                if (goodMatches[0].size() > 0)
                {
                    triangulateLinear(projMatr[0], projMatr[1], goodMatches[0], goodMatches[1], hPoints);
                }

                determine3d(hPoints, projMatr, points3d);
            }
            else
            {
                points3d = cv::Mat();
            }

            applyNext = 1;
        }

    }


    void PointReconstructor::mat2vec(cv::Mat& mat, std::vector<cv::Mat>& vec)
    {
        for (int i = 0; i < mat.cols; i++)
        {
            vec.push_back(mat.col(i).clone());
        }
    }


    void PointReconstructor::match(std::vector<cv::KeyPoint>& keypointsL, cv::Mat& descriptorsL, std::vector<cv::KeyPoint>& keypointsR, cv::Mat& descriptorsR,
        std::vector < std::vector < cv::Point_<float>>>& matchedPoints)
    {
        auto matcherPtr = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> match;


        cv::cuda::GpuMat GPUdesL(descriptorsL);
        cv::cuda::GpuMat GPUdesR(descriptorsR);
        cv::cuda::GpuMat GPUmask(matchingMask);


        matcherPtr->radiusMatch(GPUdesL, GPUdesR, match, maxDistance, GPUmask);

        for(int i = 0; i < match.size(); i++)
        {
            if (match[i].size() > 1)
            {
                if (match[i][1].distance * minDistRatio < match[i][0].distance )
                {
                    matchedPoints[0].push_back(keypointsL[match[i][0].queryIdx].pt);
                    matchedPoints[1].push_back(keypointsR[match[i][0].trainIdx].pt);
                }
            }
            else if(match[i].size() == 1)
            {
                matchedPoints[0].push_back(keypointsL[match[i][0].queryIdx].pt);
                matchedPoints[1].push_back(keypointsR[match[i][0].trainIdx].pt);
            }
        }

    }

    void PointReconstructor::determine3d(cv::Mat hPoints, std::vector<cv::Mat>projMatr, cv::Mat& points3d)
    {
        std::vector<cv::Mat>converter;
        hPoints = hPoints.t();
        mat2vec(hPoints, converter);
        cv::merge(converter, hPoints);

        cv::convertPointsFromHomogeneous(hPoints, points3d);
    }


    void  PointReconstructor::triangulateLinear(cv::Mat& mat_P_l, cv::Mat& mat_P_r, std::vector < cv::Point_<float>>& warped_back_l, std::vector < cv::Point_<float>>& warped_back_r, cv::Mat& homopoints)
    {
        for (int i = 0; i < warped_back_l.size(); i++)
        {
            cv::Mat A(4, 3, CV_64FC1), b(4, 1, CV_64FC1), X(3, 1, CV_64FC1), X_homogeneous(4, 1, CV_64FC1), W(1, 1, CV_64FC1);
            W.at<double>(0, 0) = 1.0;
            A.at<double>(0, 0) = (warped_back_l[i].x / 1) * mat_P_l.at<double>(2, 0) - mat_P_l.at<double>(0, 0);
            A.at<double>(0, 1) = (warped_back_l[i].x / 1) * mat_P_l.at<double>(2, 1) - mat_P_l.at<double>(0, 1);
            A.at<double>(0, 2) = (warped_back_l[i].x / 1) * mat_P_l.at<double>(2, 2) - mat_P_l.at<double>(0, 2);
            A.at<double>(1, 0) = (warped_back_l[i].y / 1) * mat_P_l.at<double>(2, 0) - mat_P_l.at<double>(1, 0);
            A.at<double>(1, 1) = (warped_back_l[i].y / 1) * mat_P_l.at<double>(2, 1) - mat_P_l.at<double>(1, 1);
            A.at<double>(1, 2) = (warped_back_l[i].y / 1) * mat_P_l.at<double>(2, 2) - mat_P_l.at<double>(1, 2);
            A.at<double>(2, 0) = (warped_back_r[i].x / 1) * mat_P_r.at<double>(2, 0) - mat_P_r.at<double>(0, 0);
            A.at<double>(2, 1) = (warped_back_r[i].x / 1) * mat_P_r.at<double>(2, 1) - mat_P_r.at<double>(0, 1);
            A.at<double>(2, 2) = (warped_back_r[i].x / 1) * mat_P_r.at<double>(2, 2) - mat_P_r.at<double>(0, 2);
            A.at<double>(3, 0) = (warped_back_r[i].y / 1) * mat_P_r.at<double>(2, 0) - mat_P_r.at<double>(1, 0);
            A.at<double>(3, 1) = (warped_back_r[i].y / 1) * mat_P_r.at<double>(2, 1) - mat_P_r.at<double>(1, 1);
            A.at<double>(3, 2) = (warped_back_r[i].y / 1) * mat_P_r.at<double>(2, 2) - mat_P_r.at<double>(1, 2);
            b.at<double>(0, 0) = -((warped_back_l[i].x / 1) * mat_P_l.at<double>(2, 3) - mat_P_l.at<double>(0, 3));
            b.at<double>(1, 0) = -((warped_back_l[i].y / 1) * mat_P_l.at<double>(2, 3) - mat_P_l.at<double>(1, 3));
            b.at<double>(2, 0) = -((warped_back_r[i].x / 1) * mat_P_r.at<double>(2, 3) - mat_P_r.at<double>(0, 3));
            b.at<double>(3, 0) = -((warped_back_r[i].y / 1) * mat_P_r.at<double>(2, 3) - mat_P_r.at<double>(1, 3));
            solve(A, b, X, cv::DECOMP_SVD);
            vconcat(X, W, X_homogeneous);
            if (i == 0)
            {
                homopoints = X_homogeneous;
            }
            else
            {
                hconcat(homopoints, X_homogeneous, homopoints);
            }
        }
    }