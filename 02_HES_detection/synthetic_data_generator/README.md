# Synthetic Circuit-Sketch Dataset Generator

Create semi-synthetic datasets of hand-drawn electronic symbols on paper backgrounds, with COCO annotations.  
This generator was built to stress-test object detectors on ugly, inconsistent sketches rather than polished CAD symbols.

---

## What it produces

- Grayscale images (default **200×200**) with randomly placed symbols.
- Per image:
  - symbols normalized in height, randomly rotated/reflected, optionally blurred,
  - affine/perspective jitter,
  - paper backgrounds + random lighting blobs,
  - salt & pepper noise,
  - optional connection lines that avoid crossing symbols.
- A COCO JSON per split with `bbox` for all classes except the background/no-class bucket.
___
- Adding random lines improved detector performance on this sketchy domain.
- Anchors: leftmost/rightmost points for most classes; for `T` (transistor) anchors can be pre-marked as green pixels on source images.  
- Aditional class `S` is for random shapes.
---

## Input data layout

```
background_dir/
|-- background1.jpg
|-- background2.png
|-- ...
objects_dir/
|-- C/
|   |-- c_01.jpg
|   |-- ...
|-- D/
|   |-- d_01.png
|   |-- ...
|-- L/
|-- R/
|-- S/
|-- T/
|-- Z/
|-- .../
```

- One subfolder per class. `S/` is the **no-class** bucket: objects are composited but **not annotated**.
- Transistor (`T`) images may contain **green** points marking line anchors.

---

## Configuration (`config.json`)

```json
{
  "background_dir": "../../ds/background/",
  "objects_dir": "../../ds/parts/",
  "output_path": "../../ds/results/",

  "num_of_sets": 3,
  "images_per_set": 3,

  "space_x": 10,
  "space_y": 10,
  "min_components_per_img": 5,
  "max_components_per_img": 9,
  "max_trials": 5,

  "symbols": ["C", "D", "L", "R", "T", "Z", "S"],

  "output_size": [200, 200],
  "min_distance_between_objects": 10,
  "normalized_background_size": [1500, 1500],
  "normalized_object_size": [300, 300],

  "random_image_symbol": "S",

  "drawing_lines_max_trials": 5,
  "line_iterator_skip": 20,
  "min_line_thickness": 8,
  "max_line_thickness": 14,
  "min_line_blur_kernel_size": 3,
  "max_line_blur_kernel_size": 23,

  "min_random_lighting_intensity": 0.75,
  "max_random_lighting_intensity": 1.25,
  "min_random_lighting_radius": 50,
  "max_random_lighting_radius": 200,
  "number_of_light_spots": 10,
  "lighting_blur_kernel_size": 601,

  "min_image_blur_kernel_size": 3,
  "max_image_blur_kernel_size": 21,
  "no_blur_probability": 0.3
}
```

## Build
### Requirements
- C++17, CMake >= 3.16
- OpenCV 4.x (core, imgcodecs, imgproc, highgui, calib3d)

### Configure and build
```bash
# from generator/
mkdir -p build && cd build

# If OpenCV is found automatically:
cmake .. -DUSE_OPENCV=ON

# If not found, point CMake to OpenCVConfig.cmake:
cmake .. -DUSE_OPENCV=ON -DOpenCV_DIR="C:/opencv/build"           # Windows
# or
cmake .. -DUSE_OPENCV=ON -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 # Linux/macOS

cmake --build . --config Release
```
If discovery still fails, open the generated solution and link OpenCV manually.
The correct path is the folder containing OpenCVConfig.cmake, matching your compiler/arch.

## Run

```bash
# default: looks for ./config.json next to the binary
./syntheticDataGenerator

# or pass a custom config file
./syntheticDataGenerator path/to/config.json

```

## Output
```
<output_path>/<timestamp>/
  0/ 1/ 2/                  # split ids: 0=train, 1=val, 2=test (number of sets is to choose in config.json)
    <id>.png
  annotation_0.json
  annotation_1.json
  annotation_2.json
```

## Generation pipeline (high level)

1. Split object images into train/val/test.

2. For each target image:

    - pick a background and create a large canvas.

    - for each object:

        - Otsu threshold -> mask; morphological close,

        - normalize height, then random rotation/reflection, affine/perspective distortion, noise, optional Gaussian blur,

        - collision-free placement (random translations with rejection based on dilated masks),

        - record anchor points.

    - draw lines from anchors, avoiding objects; blur the line mask; blend.

    - apply lighting blobs and salt & pepper noise.

    - resize to output size; save image and COCO entry.

## Known limitations

- **Non-representative layout**: the final images **do not resemble actual electrical schematics**. Instead, they consist of hand-drawn components randomly placed and augmented on paper-like backgrounds, optionally connected with lines. This was a **pragmatic compromise made under time constraints**, prioritizing diversity and detector robustness over diagram realism.
- **Synthetic connection lines**: the connection paths are generated programmatically (not hand-drawn), which can introduce visual discrepancies compared to the manually sketched symbols. In some cases, this can make them trivially distinguishable by a well-tuned model.
- **Performance**: the generator is not optimized for speed. Heavy use of OpenCV, large canvas rendering, and repeated object placement attempts (to avoid overlap) result in long generation times.
- **Input assumptions**: the generator expects valid, non-empty class folders in `objects_dir/`. There are no built-in checks or graceful fallbacks for missing or malformed data.