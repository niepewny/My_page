/**
 * @file ConfigLoader.h
 * @brief Types and utilities for loading generator configuration from JSON.
 *
 * The configuration defines dataset paths, image sizes, augmentation knobs
 * and various numeric heuristics used by the generator. See example
 * structure in config.json.
 */

#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>


 /**
  * @brief Runtime configuration loaded from a JSON file.
  *
  * All sizes are in pixels. Kernel sizes that control blurs must be odd.
  * Symbols must be 1-character class codes (e.g., "C","D","L","R","T","Z","S").
  * See ConfigLoader.cpp for parsing/validation details.
  */
struct Config {
    /// Folder with background PNG images used as canvases.
    std::string backgroundDir;
    /// Root folder with part images; files are organized in subfolders per class.
    std::string objectsDir;
    /// Root output folder; a timestamped subfolder is created per run.
    std::string outputPath;

    /// How many dataset shards (subfolders) to generate.
    int numOfSets;
    /// Number of images to produce in each set.
    int imagesPerSet;

    /// Minimal free border when placing objects (x direction).
    int spaceX;
    /// Minimal free border when placing objects (y direction).
    int spaceY;

    /// Minimum number of components per generated collage.
    int minComponentsPerImg;
    /// Maximum number of components per generated collage.
    int maxComponentsPerImg;

    /// Max random placement attempts per object before skipping it.
    int maxTrials;

    /// List of allowed class symbols (single characters).
    std::vector<char> symbols;

    /// Final output resolution of generated images.
    cv::Size outputSize;

    /// Minimal spacing used to keep objects apart during placement/line drawing.
    int minDistanceBetweenObjects;

    /// Working background size before downscaling to outputSize.
    cv::Size normalizedBackgroundSize;
    /// Reference object size used by normalization/scaling logic.
    cv::Size normalizedObjectSize;

    /// Symbol for the "random/no-class" images that should be ignored in labels.
    char randomImageSymbol;

    // --- Line drawing (net wiring) parameters ---
    /// Max attempts to find a valid line segment for a contact point.
    int drawingLinesMaxTrials;
    /// Number of initial pixels to skip along a candidate line before checks.
    int lineIteratorSkip;
    /// Minimum rendered wire thickness.
    int minLineThickness;
    /// Maximum rendered wire thickness.
    int maxLineThickness;
    /// Minimum Gaussian blur kernel for wires (must be odd).
    int minLineBlurKernelSize;
    /// Maximum Gaussian blur kernel for wires (must be odd).
    int maxLineBlurKernelSize;

    // --- Random lighting parameters ---
    /// Min per-spot multiplicative intensity.
    float minRandomLightingIntensity;
    /// Max per-spot multiplicative intensity.
    float maxRandomLightingIntensity;
    /// Min spot radius.
    float minRandomLightingRadius;
    /// Max spot radius.
    float maxRandomLightingRadius;
    /// Number of light spots to synthesize.
    int numberOfLightSpots;
    /// Gaussian blur kernel size applied to the lighting mask.
    int lightingBlurKernelSize;

    // --- Random image (object/mask) blur parameters ---
    /// Minimum Gaussian blur kernel for objects/masks (odd).
    int minImageBlurKernelSize;
    /// Maximum Gaussian blur kernel for objects/masks (odd).
    int maxImageBlurKernelSize;
    /// Probability to skip blurring (0..1).
    float noBlurProbability;
};

/**
 * @brief Load and validate configuration from a JSON file on disk.
 *
 * Required keys must be present and have correct types; symbols and
 * random_image_symbol must be 1-character strings.
 *
 * @param filename Path to config.json.
 * @return Config Parsed configuration object.
 * @throws std::runtime_error If the file cannot be opened or validation fails.
 */
Config loadConfig(const std::string& filename);
