# 3D Line Reconstruction

Welcome to the public repository for our paper: *"Clustering, Triangulation, and Evaluation of 3D Lines in Multiple Images. ISPRS,2024"* This repository features:
- Adaptive 3D line clustering for multi-view reconstruction.
- Robust triangulation with geometric consistency.

### Installation

- Use the `CMakeLists.txt` file in the `ELSRPP_MULTI` directory to compile the repository with CMake.
- VCPKG is recommended in Windows.
- Line matching is based on the approach described in the paper:  
  **"ELSR: Efficient Line Segment Reconstruction with Planes and Points Guidance," CVPR 2022.**  

### Test Platform

- **Operating System:** Windows 11
- **IDE:** Visual Studio 2022
- **Processor:** Intel i9-14900K
### Dependencies

- Third-Party Libraries:

|              | Boost  | Eigen | OpenCV | OpenMP | TCLAP | NLOPT |
|--------------|--------|-------|--------|--------|-------|--------|
| **Version**  | 1.79.0 | 3.4.0 | 4.5.5  |        | 1.4.0 | 2.7.1  |


### SFM
The algorithm requires the SfM result from either
-  visualsfm
-  pixel4d.
- **Note that the nvm exported from Colmap is not well supported in our code**  
The usage with TCLAP command line can be find in the "region TCLAP cmd line trans" of "main_visualSFM.cpp" and "main_pixel4d.cpp"

### 2D Line detector
It now supports 
- LSD
- Ag3line
- Edlines (There are currently some bugs when using Edlines)

### Recent reconstructions
<p align="center">
  <img src="https://github.com/user-attachments/assets/abd6b995-910c-4d82-af7e-986e4eaefafd" width="45%" />
  <img src="https://github.com/user-attachments/assets/94b57979-b6eb-4e05-b77c-c918977f11d2" width="45%" />
</p>

<p align="center">
  <figure>
    <img src="example/20250408.png" width="45%" />
    <figcaption align="center">Reconstruct with 259 images (8192 ✖ 5460)</figcaption>
  </figure>
</p>

