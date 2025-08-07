/**
 * @file DatasetGenerator.h
 * @brief Image dataset synthesizer that composes collages and COCO-style annotations.
 *
 * The generator:
 *  1) Reads background images and part images from folders,
 *  2) Splits part images into @ref Config::numOfSets shards,
 *  3) For each output image, samples N parts, normalizes, augments (rotation,
 *     flips, perspective, blur), places them on the canvas with spacing,
 *  4) Optionally draws "wires" between contact points, adds lighting & noise,
 *  5) Writes the image and appends a COCO-like annotation entry.
 *
 * Throws std::runtime_error on missing inputs (e.g., no backgrounds/parts).
 */

#include "ConfigLoader.h"
#include <opencv2/core.hpp>
#include <random>

 /**
  * @brief Main synthesis engine. Construct with a validated Config.
  */
class DatasetGenerator {
public:
    /// Initialize the generator with a configuration.
    DatasetGenerator(const Config& config);

    /**
     * @brief Run the full generation pipeline.
     *
     * Creates a timestamped subfolder in @ref Config::outputPath,
     * produces @ref Config::numOfSets subfolders, and writes images
     * plus one COCO-style JSON per set.
     *
     * @throws std::runtime_error If inputs are missing or I/O fails.
     */
    void run();

private:
    Config cfg;
    std::mt19937 gen;

    /// Shuffle utility for vectors of paths.
    void randomise(std::vector<std::string>& str);

    /**
     * @brief Compute legal translation bounds for placing an object mask.
     * @param mask Binary mask of the object in working resolution.
     * @param backgroundCols Width of the working canvas.
     * @param backgroundRows Height of the working canvas.
     * @param minX Out: minimal allowed delta-x.
     * @param minY Out: minimal allowed delta-y.
     * @param maxX Out: maximal allowed delta-x.
     * @param maxY Out: maximal allowed delta-y.
     *
     * Bounds include @ref Config::spaceX and @ref Config::spaceY margins.
     */
    void findTranslationLimits(cv::Mat mask, int backgroundCols, int backgroundRows,
        int& minY, int& minX, int& maxY, int& maxX);

    /// In-place modulo for 16-bit images (utility, currently unused).
    void modulo(cv::Mat& img, int value);

    /// Return a timestamp like "DDMMYY_HHSS" used to name the run folder.
    std::string getDateString();

    /**
     * @brief Split part image paths into cfg.numOfSets shards, roughly balanced per class.
     * @return Vector of sets; each set is a vector of absolute file paths.
     */
    std::vector<std::vector<std::string>> splitDatasets();

    /// Create output subfolder for a given set index.
    void prepareOutputDirs(std::string& setOutDir, int setId);

    /// Derive a clean binary mask from a grayscale part image (Otsu + morphology close).
    void getMask(cv::Mat objectGray, cv::Mat& mask);

    /**
     * @brief Read a part image and determine its class label and optional anchor points.
     *
     * For class "T" the function extracts green anchor pixels as contact points.
     */
    void getObject(std::string& objectPath, cv::Mat& objectGray, std::string& label,
        std::vector<cv::Point>& points);

    /**
     * @brief Try to translate an object to a collision-free position.
     * @return true if a valid translation was found within cfg.maxTrials.
     */
    bool tryTranslateObject(cv::Mat& maskOrg, cv::Mat maskSum, cv::Size backgroundSize,
        int maxTx, int minTx, int maxTy, int minTy, cv::Mat& A2);

    /// Apply A1/A2/A3 (rotate/reflect/translate) to the object; transform anchor points as well.
    void transformObjectAffine(cv::Mat mask, cv::Mat A1, cv::Mat A2, cv::Mat A3,
        cv::Size size, cv::Mat& object, std::vector<cv::Point>& pathPoints);

    /// Composite an object onto the output canvas with random intensity scaling.
    void addObjectToOutput(cv::Mat mask, cv::Mat object, cv::Mat& output, std::mt19937 gen);

    /**
     * @brief Appllies distortion.     
     * Note: the current heuristics include several magic constants.
     * Consider adjusting or removing if you want full determinism.
     */
    void distortObject(cv::Mat& object, cv::Mat& mask, std::mt19937 g, std::vector<cv::Point>& pathPoints);

    /// Random rotation with canvas expansion; updates mask and returns the affine matrix.
    void rotateObject(cv::Mat& A, cv::Mat& mask, std::mt19937& gen);

    /// Random mirror reflection (x/y/both/none); updates mask and returns the affine matrix.
    void reflectObject(cv::Mat& mask, cv::Mat& A, std::mt19937& gen);

    /// Axis-aligned bounding box of the mask region.
    cv::Rect findBbox(cv::Mat mask);

    /// Normalize object scale based on cfg.normalizedObjectSize and class-specific rules.
    void normalizeSize(cv::Mat& object, cv::Mat& mask, std::vector<cv::Point>& pathPoints, std::string label);

    /// Sample N image paths for a collage, ensuring not all belong to the "random" class.
    std::vector<std::string> getImagesForCollage(int numOfImages, std::vector<std::string> dataset, std::mt19937& gen);

    /// Optional Gaussian blur applied to both object and its mask (kernel odd, probability from config).
    void applyRandomGaussianBlur(cv::Mat& object, cv::Mat& mask, std::mt19937& gen);

    /// Load background PNG files from a folder; throws if none are found.
    std::vector<std::string> loadBackgroundImages();

    /// Add salt/pepper noise with small probability per pixel.
    void addNoise(cv::Mat& image, std::mt19937& gen);

    /// Synthesize smooth lighting mask and modulate the image (multiply).
    void applyRandomLighting(cv::Mat& image, std::mt19937& gen);

    /// Return the leftmost and rightmost contour points of a binary mask.
    std::vector<cv::Point> findExtremePointsX(const cv::Mat& mask);

    /**
     * @brief Draw pseudo-wires starting from contact points toward image borders,
     *        avoiding overlaps with objects and other wires. Uses cfg.* line parameters.
     */
    void drawRandomLines(cv::Mat joinedMasks, std::vector<cv::Mat> masks,
        std::vector<std::vector<cv::Point>>& points, cv::Mat& outputImage);

    /// Find pixels of a given BGR color (used for "T" anchor marks); returns sample points.
    std::vector<cv::Point> findColorBlobs(const cv::Mat& image, cv::Vec3b targetColor, int tolerance);

    /// Pick a random point on any of the four canvas edges.
    cv::Point randomExtremePoint(const cv::Mat& img);

    /// Resize object/mask to keep area ratio change within [sMin, sMax].
    void resizeObject(cv::Mat& object, cv::Mat& mask, float sMin, float sMax);

    /// Helper for @ref resizeObject to compute the next scale factor.
    float calculateScale(float sMin, float sMax, float currentScaleChange);

    /// Convert integer points to float points.
    std::vector<cv::Point2f> convertToPoints2f(const std::vector<cv::Point>& points);

};