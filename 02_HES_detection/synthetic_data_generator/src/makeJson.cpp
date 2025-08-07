#include "makeJson.h"

using namespace nlohmann;


void initJson(json& file)
{
    file["info"] = {
        {"year", "2024"},
        {"version", "1"},
        {"description", "annotation of semi-synthetic data"},
        {"contributor", ""},
        {"url", ""},
        {"date_created", "2023-05-X"}
    };

    file["licenses"].push_back({
        {"id", 1},
        {"url", "-"},
        {"name", "-"}
        });

    std::vector<std::string> categoryNames = {
        "C",
        "D",
        "L",
        "R",
        "T",
        "Z",
    };

    for (int i = 0; i < categoryNames.size(); ++i) {
        file["categories"].push_back({
            {"id", i},
            {"name", categoryNames[i]},
            {"supercategory", (i == 0 ? "none" : "circuit-voltages")}
            });
    }
}

void appendImage(json& file, int imNumber, std::string imName, int imWidth, int imHeight)
{
    file["images"].push_back(
        {
            {"id", imNumber},
            {"license", 1},
            {"file_name", imName},
            {"height", imHeight},
            {"width", imWidth},
            {"date_captured", "X"}
        }
    );
}

void appendAnnotations(json& file, int imNumber, int annotationNumber, int categoryId, cv::Rect bbox)
{
    file["annotations"].push_back(
        {
            {"id", annotationNumber},
            {"image_id", imNumber},
            {"category_id", categoryId },
            {"bbox", {bbox.x, bbox.y, bbox.width, bbox.height}},
            {"area", bbox.area()},
            {"segmentation", json::array()},
            {"iscrowd", 0}
        }
    );
}
