#include <string.h>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <opencv2/highgui.hpp>
#include <random>
#include <opencv2/calib3d.hpp>
#include "nlohmann/json.hpp"
#include <fstream>
#include <unordered_set>
#include "ConfigLoader.h"
#include "DatasetGenerator.h"
#include "makeJson.h"


DatasetGenerator::DatasetGenerator(const Config& config)
{
	cfg = config;
}

void DatasetGenerator::run()
{
	srand(static_cast<unsigned int>(time(0)));

	cv::Mat A1, A2, A3, output, mask, object, background, maskSum;
	std::string PATHoutputCOLLAGE, label;
	std::vector<std::vector<std::string>> datasets;
	int maxTx, minTx, maxTy, minTy;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> emementsNumberDistribution(cfg.minComponentsPerImg, cfg.maxComponentsPerImg);
	cv::Size backgroundSize;


	datasets = splitDatasets();

	std::vector<std::string> backgroundImages = loadBackgroundImages();
	std::uniform_int_distribution<size_t> bgDist(0, backgroundImages.size() - 1);
	backgroundSize = cfg.normalizedBackgroundSize;

	cfg.outputPath += getDateString();
	std::filesystem::create_directories(cfg.outputPath);


	for (int setId = 0; setId < cfg.numOfSets; setId++)
	{
		prepareOutputDirs(PATHoutputCOLLAGE, setId);
		nlohmann::json annotationFile;
		initJson(annotationFile);
		int elementNumber = 0;

		for (int collageId = 0; collageId < cfg.imagesPerSet; collageId++)
		{
			std::vector<std::vector<cv::Point>> linesOrigins;
			std::vector<cv::Mat> masks;

			background = cv::imread(backgroundImages[bgDist(gen)], cv::IMREAD_GRAYSCALE);
			cv::resize(background, background, backgroundSize, 0, 0, cv::INTER_AREA);

			maskSum = cv::Mat(backgroundSize, CV_8U, cv::Scalar(0));
			output = background.clone();

			int elementsPerImg = emementsNumberDistribution(gen);
			std::vector<std::string>paths = getImagesForCollage(elementsPerImg, datasets[setId], gen);
			bool gotAnyAnot = false;

			for (int element = 0; element < elementsPerImg; element++)
			{
				std::vector<cv::Point> pathPoints;
				getObject(paths[element], object, label, pathPoints);
				getMask(object, mask);
				normalizeSize(object, mask, pathPoints, label);
				applyRandomGaussianBlur(object, mask, gen);
				cv::bitwise_and(object, mask, object);
				distortObject(object, mask, gen, pathPoints);
				if (label == std::string(1, cfg.randomImageSymbol))  pathPoints = std::vector<cv::Point>(0);
				else if (label == "T")  pathPoints = pathPoints;
				else pathPoints = findExtremePointsX(mask);
				rotateObject(A1, mask, gen);
				reflectObject(mask, A2, gen);

				findTranslationLimits(mask, background.cols, background.rows, maxTx, minTx, maxTy, minTy);

				if (tryTranslateObject(mask, maskSum, backgroundSize, maxTx, minTx, maxTy, minTy, A3))
				{
					cv::Rect bbox = findBbox(mask);
					maskSum(bbox) = 255;

					transformObjectAffine(mask, A1, A2, A3, backgroundSize, object, pathPoints);
					linesOrigins.push_back(pathPoints);
					masks.push_back(mask);
					addObjectToOutput(mask, object, output, gen);
					int id = std::distance(cfg.symbols.begin(), std::find(cfg.symbols.begin(), cfg.symbols.end(), label[0]));
					if (label[0] != 'S')
					{
						double scaleX = (double)cfg.outputSize.width / cfg.normalizedBackgroundSize.width;
						double scaleY = (double)cfg.outputSize.height / cfg.normalizedBackgroundSize.height;
						bbox.x *= scaleX;
						bbox.y *= scaleY;
						bbox.width *= scaleX;
						bbox.height *= scaleY;
						appendAnnotations(annotationFile, collageId, elementNumber, id, bbox);
						elementNumber++;
						gotAnyAnot = true;
					}
				}
			}
			if (gotAnyAnot) {
				drawRandomLines(maskSum, masks, linesOrigins, output);
				applyRandomLighting(output, gen);
				addNoise(output, gen);
				appendImage(annotationFile, collageId, std::to_string(collageId) + ".png", cfg.outputSize.width, cfg.outputSize.height);
				cv::resize(output, output, cfg.outputSize);
				cv::imwrite(PATHoutputCOLLAGE + std::to_string(collageId) + ".png", output);
			}
			else {
				collageId -= 1; //if nothing was added to json, the generator should try again
			}
		}
		std::ofstream file(cfg.outputPath + "\\annotation_" + std::to_string(setId) + ".json");
		file << std::setw(4) << annotationFile << std::endl;
		file.close();
	}
}

std::vector<cv::Point> DatasetGenerator::findColorBlobs(const cv::Mat& image, cv::Vec3b targetColor, int tolerance = 20) {

	cv::Mat mask;
	inRange(image, targetColor, targetColor, mask);

	std::vector<std::vector<cv::Point>> contours;
	findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	std::vector<cv::Point> centroids;
	for (auto contour : contours) {
		centroids.push_back(contour[0]);
	}

	return centroids;
}

cv::Point DatasetGenerator::randomExtremePoint(const cv::Mat& img) {
	int width = img.cols;
	int height = img.rows;
	int edge = rand() % 4;

	cv::Point pt;
	switch (edge) {
	case 0:
		pt = cv::Point(rand() % width, 0);
		break;
	case 1:
		pt = cv::Point(rand() % width, height - 1);
		break;
	case 2:
		pt = cv::Point(0, rand() % height);
		break;
	case 3:
		pt = cv::Point(width - 1, rand() % height);
		break;
	}
	return pt;
}

void DatasetGenerator::drawRandomLines(cv::Mat joinedMasks, std::vector<cv::Mat> masks, std::vector<std::vector<cv::Point>>& points, cv::Mat& outputImage) {

	cv::Mat pathsImage = cv::Mat::zeros(joinedMasks.rows, joinedMasks.cols, CV_8U);

	for (size_t i = 0; i < points.size(); ++i) {
		const auto& objectPoints = points[i];
		const auto& objectMask = masks[i];


		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(cfg.minDistanceBetweenObjects * 2 + 6, cfg.minDistanceBetweenObjects * 2 + 6));
		cv::dilate(objectMask, objectMask, kernel);
		cv::Rect objectBbox = findBbox(objectMask);
		cv::Mat joinedMasksWithoutOne = joinedMasks.clone();
		joinedMasksWithoutOne(objectBbox) = 0;
		kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(cfg.minDistanceBetweenObjects * 5, cfg.minDistanceBetweenObjects * 5));
		cv::dilate(joinedMasksWithoutOne, joinedMasksWithoutOne, kernel);

		for (const auto& startPt : objectPoints) {
			bool validLine = false;
			cv::Point endPt;

			int tryNr = 0;
			while (!validLine) {
				endPt = randomExtremePoint(joinedMasks);

				cv::LineIterator lineIt(outputImage, startPt, endPt);

				validLine = true;
				for (int k = 0; k < cfg.lineIteratorSkip; ++k, ++lineIt);
				for (int j = cfg.lineIteratorSkip; j < lineIt.count; j++, ++lineIt) {
					cv::Point pt = lineIt.pos();
					if (objectMask.at<uchar>(pt.y, pt.x) != 0) {
						validLine = false;
						break;
					}
					if (joinedMasksWithoutOne.at<uchar>(pt.y, pt.x) != 0) {
						endPt = pt;
						break;
					}
				}
				tryNr++;
				if (tryNr >= cfg.drawingLinesMaxTrials) break;
			}

			if (validLine) {
				int randThickness = cfg.minLineThickness + rand() % (cfg.maxLineThickness - cfg.minLineThickness + 1);
				cv::line(pathsImage, startPt, endPt, 255, randThickness);
			}
		}
	}
	std::uniform_int_distribution<> blurKernelSteps(0, ((cfg.maxLineBlurKernelSize - cfg.minLineBlurKernelSize) / 2));
	int kernelSize = cfg.minLineBlurKernelSize + 2 * blurKernelSteps(gen);  // always odd

	cv::GaussianBlur(pathsImage, pathsImage, cv::Size(kernelSize, kernelSize), 0);

	outputImage = outputImage - pathsImage;
}

void DatasetGenerator::applyRandomLighting(cv::Mat& image, std::mt19937& gen) {

	cv::Mat lightingMask(image.size(), CV_32F, cv::Scalar(1.0));

	std::uniform_real_distribution<> dis_light(cfg.minRandomLightingIntensity, cfg.maxRandomLightingIntensity);
	std::uniform_int_distribution<> dis_radius(cfg.minRandomLightingRadius, cfg.maxRandomLightingRadius);
	std::uniform_int_distribution<> dis_x(0, image.cols);
	std::uniform_int_distribution<> dis_y(0, image.rows);

	for (int i = 0; i < cfg.numberOfLightSpots; ++i) {
		int radius = dis_radius(gen);
		float intensity = dis_light(gen);
		cv::Point center(dis_x(gen), dis_y(gen));
		cv::circle(lightingMask, center, radius, cv::Scalar(intensity), -1);
	}

	cv::GaussianBlur(lightingMask, lightingMask, cv::Size(cfg.lightingBlurKernelSize, cfg.lightingBlurKernelSize), 0);
	cv::Mat imageFloat;
	image.convertTo(imageFloat, CV_32F, 1.0 / 255.0);
	cv::Mat resultFloat = imageFloat.mul(lightingMask);
	resultFloat.convertTo(image, CV_8U, 255.0);
	cv::threshold(image, image, 255, 255, cv::THRESH_TRUNC);
}

void DatasetGenerator::addNoise(cv::Mat& image, std::mt19937& gen) {
	std::uniform_int_distribution<> dist(0, 100);
	int noiseType = dist(gen) % 3;

	switch (noiseType) {
	case 0: // Salt
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				if (dist(gen) < 2) { // 2% chance
					image.at<uchar>(i, j) = 255;
				}
			}
		}
		break;
	case 1: // Pepper
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				if (dist(gen) < 2) { // 2% chance
					image.at<uchar>(i, j) = 0;
				}
			}
		}
		break;
	case 2: // Salt and pepper
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				int random = dist(gen);
				if (random < 1) {
					image.at<uchar>(i, j) = 255;
				}
				else if (random < 2) {
					image.at<uchar>(i, j) = 0;
				}
			}
		}
		break;
	}
}

std::vector<std::string> DatasetGenerator::loadBackgroundImages() {
	std::vector<std::string> files, pngFiles;

	cv::glob(cfg.backgroundDir + "*.jpg", files, true);
	cv::glob(cfg.backgroundDir + "*.png", pngFiles, true);
	files.insert(files.end(), pngFiles.begin(), pngFiles.end());

	if (files.empty())
		throw std::runtime_error("No background PNGs found in: " + cfg.backgroundDir);

	return files;
}

void DatasetGenerator::applyRandomGaussianBlur(cv::Mat& object, cv::Mat& mask, std::mt19937& gen)
{
	std::uniform_int_distribution<> kSizeDist(cfg.minImageBlurKernelSize, cfg.maxImageBlurKernelSize);

	std::uniform_real_distribution<> prob_dist(0.0, 1.0);
	if (prob_dist(gen) < cfg.noBlurProbability) {
		return;
	}

	int ksize = kSizeDist(gen) | 1; // Ensure the kernel size is odd
	cv::GaussianBlur(object, object, cv::Size(ksize, ksize), 0);
	cv::GaussianBlur(mask, mask, cv::Size(ksize, ksize), 0);
	cv::threshold(mask, mask, 150, 255, cv::THRESH_BINARY);
}

std::vector<std::string> DatasetGenerator::getImagesForCollage(int numOfImages, std::vector<std::string> dataset, std::mt19937& gen)
{
	std::vector<std::string> selectedImages;
	std::unordered_set<std::string> includedClasses;

	do {
		selectedImages.clear();
		includedClasses.clear();

		std::vector<std::string> shuffledDataset = dataset;
		std::shuffle(shuffledDataset.begin(), shuffledDataset.end(), gen);

		for (int i = 0; i < numOfImages && i < shuffledDataset.size(); ++i) {
			selectedImages.push_back(shuffledDataset[i]);
			includedClasses.insert(((std::filesystem::path)shuffledDataset[i]).parent_path().filename().string());
		}
	} while (includedClasses.size() == 1 && includedClasses.count(std::string(1, cfg.randomImageSymbol)) == 1);

	return selectedImages;
}

void DatasetGenerator::reflectObject(cv::Mat& mask, cv::Mat& A, std::mt19937& gen)
{
	std::uniform_int_distribution<> dis(0, 3);

	int reflectionCode = dis(gen);

	A = cv::Mat::eye(2, 3, CV_64F);
	switch (reflectionCode) {
	case 0: // x axis
		A.at<double>(1, 1) = -1;
		A.at<double>(1, 2) = mask.rows;
		break;
	case 1: // y axis
		A.at<double>(0, 0) = -1;
		A.at<double>(0, 2) = mask.cols;
		break;
	case 2: // both
		A.at<double>(0, 0) = -1;
		A.at<double>(1, 1) = -1;
		A.at<double>(0, 2) = mask.cols;
		A.at<double>(1, 2) = mask.rows;
		break;
	case 3:
	default:
		break;
	}

	cv::warpAffine(mask, mask, A, mask.size(), cv::INTER_NEAREST);
}

void DatasetGenerator::rotateObject(cv::Mat& A, cv::Mat& mask, std::mt19937& gen)
{
	std::uniform_real_distribution<double> angleDistribution(0., 360.);
	A = cv::getRotationMatrix2D(cv::Point2d((mask.cols - 1) / 2.0, (mask.rows - 1) / 2.0), angleDistribution(gen), 1);
	double absCos = std::abs(A.at<double>(0, 0));
	double absSin = std::abs(A.at<double>(0, 1));
	int newWidth = static_cast<int>((mask.rows * absSin) + (mask.cols * absCos));
	int newHeight = static_cast<int>((mask.rows * absCos) + (mask.cols * absSin));
	cv::Point2f center(static_cast<float>(mask.cols / 2), static_cast<float>(mask.rows / 2));
	A.at<double>(0, 2) += (newWidth / 2) - center.x;
	A.at<double>(1, 2) += (newHeight / 2) - center.y;
	cv::warpAffine(mask, mask, A, cv::Size(newWidth, newHeight), cv::INTER_NEAREST);
}

float DatasetGenerator::calculateScale(float sMin, float sMax, float currentScaleChange)
{
	if (currentScaleChange < sMin) {
		return sMin / currentScaleChange;
	}
	else if (currentScaleChange > sMax) {
		return sMax / currentScaleChange;
	}
	else {
		return 1.0f;
	}
}

void DatasetGenerator::resizeObject(cv::Mat& object, cv::Mat& mask, float sMin, float sMax)
{
	cv::Mat objectOrg = object.clone();
	cv::Mat maskOrg = mask.clone();
	cv::Rect bbox = findBbox(maskOrg);
	float scaleChange;

	scaleChange = 1.0f;

	do {
		object = objectOrg.clone();
		mask = maskOrg.clone();

		float scale = calculateScale(sMin, sMax, scaleChange);

		cv::resize(object, object, cv::Size(), scale, scale);
		cv::resize(mask, mask, cv::Size(), scale, scale);

		cv::Rect newbbox = findBbox(mask);
		scaleChange = std::sqrt(static_cast<float>(newbbox.area()) / static_cast<float>(bbox.area()));
	} while (scaleChange < sMin || scaleChange > sMax);
}

std::vector<cv::Point2f> DatasetGenerator::convertToPoints2f(const std::vector<cv::Point>& points) {
	std::vector<cv::Point2f> points2f;
	for (const auto& pt : points) {
		points2f.push_back(cv::Point2f(pt.x, pt.y));
	}
	return points2f;
}

// magic numbers are left untouched - it is hard to manipulate and perhaps the function should be removed 
void DatasetGenerator::distortObject(cv::Mat& object, cv::Mat& mask, std::mt19937 g, std::vector<cv::Point>& pathPoints)
{
	cv::Mat objectOrg = object.clone();
	cv::Mat maskOrg = mask.clone();
	double f = 1000;
	std::uniform_real_distribution<double> A(0, 0.15);
	std::uniform_real_distribution<double> P(0, 0.0009);
	float sMin = 0.9;
	float sMax = 1.1;
	std::uniform_real_distribution<double> S_increase(sMin, 1.3);
	std::uniform_real_distribution<double> S_decrease(0.7, sMax);
	std::vector<double> D = { 0, 10, 20, 30 };
	std::uniform_int_distribution<> d(0, D.size() - 1);

	cv::Rect bbox = findBbox(maskOrg);
	double widthScale = static_cast<double>(cfg.normalizedObjectSize.height) / bbox.width;

	cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, object.cols / 2, 0, f, object.rows / 2, 0, 0, 1);
	cv::Mat M = (cv::Mat_<double>(3, 3) << 1, A(g), 0, A(g), 1, 0, P(g), P(g), 1);

	std::vector<cv::Point2f> corners = { cv::Point2f(0, 0), cv::Point2f(object.cols, 0), cv::Point2f(0, object.rows), cv::Point2f(object.cols, object.rows) };
	std::vector<cv::Point2f> transformedCorners(4);
	cv::perspectiveTransform(corners, transformedCorners, M);
	cv::Size size = cv::boundingRect(transformedCorners).size();
	cv::warpPerspective(object, object, M, size);
	cv::warpPerspective(mask, mask, M, size);
	cv::bitwise_and(object, mask, object);

	if (!pathPoints.empty()) {
		std::vector<cv::Point2f> pathPointsF = convertToPoints2f(pathPoints);
		cv::perspectiveTransform(pathPointsF, pathPointsF, M);
		pathPoints.assign(pathPointsF.begin(), pathPointsF.end());
	}

}


std::vector<cv::Point> DatasetGenerator::findExtremePointsX(const cv::Mat& mask) {
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	cv::Point minXPoint(mask.cols, 0);
	cv::Point maxXPoint(0, 0);

	for (const auto& contour : contours) {
		for (const auto& point : contour) {
			if (point.x < minXPoint.x) {
				minXPoint = point;
			}
			if (point.x > maxXPoint.x) {
				maxXPoint = point;
			}
		}
	}

	return { minXPoint, maxXPoint };
}

void DatasetGenerator::addObjectToOutput(cv::Mat mask, cv::Mat object, cv::Mat& output, std::mt19937 gen)
{
	std::uniform_real_distribution<double> aDistribution(0.7, 2);
	double a = aDistribution(gen);
	object = 255 - object;
	cv::bitwise_and(object, mask, object);
	output = output - (object * a);
}

void DatasetGenerator::transformObjectAffine(cv::Mat mask, cv::Mat A1, cv::Mat A2, cv::Mat A3, cv::Size size, cv::Mat& object, std::vector<cv::Point>& pathPoints)
{
	cv::warpAffine(object, object, A1, size, cv::INTER_NEAREST);
	cv::warpAffine(object, object, A2, size, cv::INTER_NEAREST);
	cv::warpAffine(object, object, A3, size, cv::INTER_NEAREST);
	if (pathPoints.size() > 0)
	{
		cv::transform(pathPoints, pathPoints, A1);
		cv::transform(pathPoints, pathPoints, A2);
		cv::transform(pathPoints, pathPoints, A3);
	}
	cv::bitwise_and(object, mask, object);
}

bool DatasetGenerator::tryTranslateObject(cv::Mat& maskOrg, cv::Mat maskSum, cv::Size backgroundSize, int maxTx, int minTx, int maxTy, int minTy, cv::Mat& A2)
{
	cv::Mat test;
	int Tx, Ty;
	cv::Mat mask, dilatedMask, kernel;

	kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(cfg.minDistanceBetweenObjects * 2 + 1, cfg.minDistanceBetweenObjects * 2 + 1));
	cv::dilate(maskOrg, dilatedMask, kernel);

	for (int j = 0; j < cfg.maxTrials; j++)
	{
		if (maxTx - minTx == 0) Tx = minTx;
		else if (maxTx - minTx < 0) return false;
		else Tx = rand() % (maxTx - minTx) + minTx;

		if (maxTy - minTy == 0) Ty = minTy;
		else if (maxTy - minTy < 0) return false;
		else Ty = rand() % (maxTy - minTy) + minTy;

		A2 = (cv::Mat_<double>(2, 3) << 1, 0, Tx, 0, 1, Ty);
		cv::warpAffine(dilatedMask, mask, A2, backgroundSize, cv::INTER_NEAREST);
		mask(findBbox(mask)) = 255;
		cv::bitwise_and(maskSum, mask, test);

		if (cv::countNonZero(test) == 0)
		{
			cv::warpAffine(maskOrg, maskOrg, A2, backgroundSize, cv::INTER_NEAREST);
			return true;
		}
	}
	return false;
}

void DatasetGenerator::normalizeSize(cv::Mat& object, cv::Mat& mask, std::vector<cv::Point>& pathPoints, std::string label)
{
	float scale;
	cv::Rect bbox = findBbox(mask);
	if (label == "T")
	{
		scale = (float)cfg.normalizedObjectSize.height / (float)bbox.height * 1.3;
	}
	else
	{
		scale = (float)cfg.normalizedObjectSize.width / (float)bbox.height;
	}
	cv::resize(object, object, cv::Size(0, 0), scale, scale);
	cv::resize(mask, mask, cv::Size(0, 0), scale, scale);
	mask = (mask > 2);

	if (!pathPoints.empty())
	{
		for (auto& point : pathPoints)
		{
			point *= scale;
		}
	}
}

void DatasetGenerator::getObject(std::string& objectPath, cv::Mat& object, std::string& label, std::vector<cv::Point>& points)
{
	object = cv::imread(objectPath, cv::IMREAD_GRAYSCALE);
	label = ((std::filesystem::path)objectPath).parent_path().filename().string();
	if (label == "T")
	{
		cv::Mat objectBGR = cv::imread(objectPath);
		points = findColorBlobs(objectBGR, cv::Vec3b(0, 255, 0), 5);
	}
}

void DatasetGenerator::getMask(cv::Mat object, cv::Mat& maskOrg)
{
	cv::threshold(object, maskOrg, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::morphologyEx(maskOrg, maskOrg, cv::MORPH_CLOSE, kernel);
}

void DatasetGenerator::prepareOutputDirs(std::string& PATHoutputCOLLAGE, int setId)
{
	PATHoutputCOLLAGE = cfg.outputPath + "/" + std::to_string(setId) + "/";
	std::filesystem::create_directories(PATHoutputCOLLAGE);
}

std::vector<std::vector<std::string>> DatasetGenerator::splitDatasets()
{
	std::vector<std::vector<std::string>> datasets(cfg.numOfSets);
	std::vector<std::string> objectsPaths, pngPaths;

	cv::glob(cfg.objectsDir + "*.jpg", objectsPaths, true);
	cv::glob(cfg.objectsDir + "*.png", pngPaths, true);
	objectsPaths.insert(objectsPaths.end(), pngPaths.begin(), pngPaths.end());

	if (objectsPaths.empty())
		throw std::runtime_error("No images found in: " + cfg.objectsDir);


	randomise(objectsPaths);

	std::map<char, std::vector<std::string>> classMap;
	for (const auto& path : objectsPaths) {
		char classLabel = std::toupper(path.back());
		classMap[classLabel].push_back(path);
	}

	// Distribute paths evenly across datasets
	for (auto& classItem : classMap) {
		const auto& classPaths = classItem.second;
		int idx = 0;
		for (const auto& path : classPaths) {
			datasets[idx].push_back(path);
			idx = (idx + 1) % cfg.numOfSets;
		}
	}

	return datasets;
}


std::string DatasetGenerator::getDateString()
{
	struct tm newtime;
	time_t now = time(0);
	localtime_s(&newtime, &now);
	int Month = 1 + newtime.tm_mon;
	char dateChar[15];
	strftime(dateChar, 50, "%d%m%y_%H%S", &newtime);

	return std::string(dateChar);
}

void DatasetGenerator::modulo(cv::Mat& img, int value)
{
	for (int i = 0; i < img.cols; i++)
	{
		for (int j = 0; j < img.rows; j++)
		{
			img.at<uint16_t>(j, i) %= value;
		}
	}
}

void DatasetGenerator::randomise(std::vector<std::string>& str)
{
	unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::vector<int> ids;
	for (int i = 0; i < str.size(); i++)
	{
		ids.push_back(i);
	}
	std::shuffle(std::begin(ids), std::end(ids), generator);
	std::vector<std::string> strCpy = str;

	for (int i = 0; i < str.size(); i++)
	{
		str[i] = strCpy[ids[i]];
	}

}

cv::Rect DatasetGenerator::findBbox(cv::Mat mask)
{
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Point> mergedContours;

	cv::findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	for (const auto& innerContour : contours) {
		mergedContours.insert(mergedContours.end(), innerContour.begin(), innerContour.end());
	}

	cv::Rect bbox = cv::boundingRect(mergedContours);

	return bbox;
}

void DatasetGenerator::findTranslationLimits(cv::Mat mask, int backgroundCols, int backgroundRows, int& maxTx, int& minTx, int& maxTy, int& minTy)
{
	cv::Rect boundingRect = findBbox(mask);

	int minY = boundingRect.y;
	int minX = boundingRect.x;
	int maxY = boundingRect.y + boundingRect.height;
	int maxX = boundingRect.x + boundingRect.width;

	maxTx = backgroundCols - maxX - cfg.spaceX;
	minTx = -minX + cfg.spaceX;
	maxTy = backgroundRows - maxY - cfg.spaceY;
	minTy = -minY + cfg.spaceY;
}