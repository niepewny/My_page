/**
 * @file makeJson.h
 * @brief Helpers to build a minimal COCO-like JSON for detection datasets.
 *
 * The schema contains "images", "annotations" and "categories" arrays,
 * plus basic "info" and "licenses". Category ids are assigned in insertion order.
 */

#pragma once

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

 /**
  * @brief Initialize top-level COCO fields and the static categories array.
  *
  * Fills "info", "licenses", and pushes predefined categories (C, D, L, R, T, Z).
  * Call once per set before appending images/annotations.
  */
void initJson(nlohmann::json& file);

/**
 * @brief Append an image record to the JSON.
 * @param file Root JSON.
 * @param imNumber Image id (unique within the set).
 * @param imName File name (relative to the set folder).
 * @param imWidth Width in pixels.
 * @param imHeight Height in pixels.
 */
void appendImage(nlohmann::json& file, int imNumber, std::string imName, int imWidth, int imHeight);

/**
 * @brief Append a bounding-box annotation to the JSON.
 * @param file Root JSON.
 * @param imNumber Image id this annotation belongs to.
 * @param annotationNumber Unique annotation id (within the set).
 * @param categoryId Category id (index in the predefined categories).
 * @param bbox Axis-aligned bounding box in final output resolution.
 */
void appendAnnotations(nlohmann::json& file, int imNumber, int annotationNumber, int categoryId, cv::Rect bbox);
