#include "nlohmann/json.hpp"
#include <fstream>
#include <stdexcept>
#include "ConfigLoader.h"

using json = nlohmann::json;

Config loadConfig(const std::string& filename) {
    std::ifstream f(filename);
    if (!f) {
        throw std::runtime_error("Cannot open config file: " + filename);
    }

    json j;
    f >> j;

    Config cfg;
    cfg.backgroundDir = j.at("background_dir").get<std::string>();
    cfg.objectsDir = j.at("objects_dir").get<std::string>();
    cfg.outputPath = j.at("output_path").get<std::string>();
    cfg.numOfSets = j.at("num_of_sets").get<int>();
    cfg.imagesPerSet = j.at("images_per_set").get<int>();
    cfg.spaceX = j.at("space_x").get<int>();
    cfg.spaceY = j.at("space_y").get<int>();
    cfg.minComponentsPerImg = j.at("min_components_per_img").get<int>();
    cfg.maxComponentsPerImg = j.at("max_components_per_img").get<int>();
    cfg.maxTrials = j.at("max_trials").get<int>();
    auto symStrs = j.at("symbols").get<std::vector<std::string>>();
    cfg.symbols.clear();
    cfg.symbols.reserve(symStrs.size());
    for (const auto& s : symStrs) {
        if (s.size() != 1)
            throw std::runtime_error("symbols[] entries must be 1-character strings");
        cfg.symbols.push_back(s[0]);
    }
    {
        std::vector<int> vec = j.at("output_size");
        cfg.outputSize = cv::Size(vec[0], vec[1]);
    }
    cfg.minDistanceBetweenObjects = j.at("min_distance_between_objects").get<int>();
    {
        std::vector<int> vec = j.at("normalized_background_size");
        cfg.normalizedBackgroundSize = cv::Size(vec[0], vec[1]);
    }
    {
        std::vector<int> vec = j.at("normalized_object_size");
        cfg.normalizedObjectSize = cv::Size(vec[0], vec[1]);
    }
    {
        std::string s = j.at("random_image_symbol").get<std::string>();
        if (s.size() != 1)
            throw std::runtime_error("random_image_symbol must be a 1-character string");
        cfg.randomImageSymbol = s[0];
    }

    cfg.drawingLinesMaxTrials = j.at("drawing_lines_max_trials").get<int>();
    cfg.lineIteratorSkip = j.at("line_iterator_skip").get<int>();
    cfg.minLineThickness = j.at("min_line_thickness").get<int>();
    cfg.maxLineThickness = j.at("max_line_thickness").get<int>();
    cfg.minLineBlurKernelSize = j.at("min_line_blur_kernel_size").get<int>();
    cfg.maxLineBlurKernelSize = j.at("max_line_blur_kernel_size").get<int>();

    cfg.minRandomLightingIntensity = j.at("min_random_lighting_intensity").get<double>();
    cfg.maxRandomLightingIntensity = j.at("max_random_lighting_intensity").get<double>();
    cfg.minRandomLightingRadius = j.at("min_random_lighting_radius").get<int>();
    cfg.maxRandomLightingRadius = j.at("max_random_lighting_radius").get<int>();
    cfg.numberOfLightSpots = j.at("number_of_light_spots").get<int>();
    cfg.lightingBlurKernelSize = j.at("lighting_blur_kernel_size").get<int>();

    cfg.minImageBlurKernelSize = j.at("min_image_blur_kernel_size").get<int>();
    cfg.maxImageBlurKernelSize = j.at("max_image_blur_kernel_size").get<int>();
    cfg.noBlurProbability = j.at("no_blur_probability").get<double>();


    return cfg;
}
