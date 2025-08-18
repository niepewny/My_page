# About me

I'm a software developer specialized in AI/ML and computer vision. I have over two years of commercial experience and around five years of academic experience in related fields.

I hold a degree in Mechatronics (specialization: Photonics) from Warsaw University of Technology and I'm currently completing a Master's in Information Technology (also at WUT). During my studies, I’ve focused on deepening my core interests — machine vision and AI, while also exploring how to apply them across different problem domains.

# About this repository

This repository showcases a selection of non-commercial projects I’ve worked on during my academic and personal development. The projects are ordered (in my opinion) from the most relevant and technically impactful to more exploratory ones.

| ID | Name | Brief | Technology |
|----|------|-------|------------|
| 01 | [SfM Reconstruction](./01_SfM_reconstruction/) | Sparse **Structure-from-Motion** with extra geometric hints (approx. pose/depth/intrinsics) + GUI; reconstructs 3D point cloud from one/two images and masks. | C++17, OpenCV, PCL, Qt, CMake |
| 02 | [HES Detection](./02_HES_detection/) | **Handwritten Electronic Schematics**: synthetic data generator + object detection (components) + connection parsing into a matrix. | Python, PyTorch (**YOLOv8**, Faster R-CNN), OpenCV; C++ (generator, nlohmann/json), COCO JSON |
| 03 | [RNN Raincast](./03_RNN_raincast/) | **Nowcasting** on SEVIR IR069: ConvRNN / ConvLSTM / Peephole-ConvLSTM with a lightweight head; demo sample, checkpoint and notebook included. | Python, PyTorch Lightning, Hydra, NumPy, Matplotlib, Weights&Biases (opt.) |
| 04 | [Mouse Anomalies](./04_Mouse_Anomalies/) | **User verification via mouse dynamics** (Balabit): segmentation → 37 features → classic ML for anomaly detection and supervised classification. | Python, scikit-learn, XGBoost, SVM, Isolation Forest, PCA, SMOTE, Pandas/NumPy |
