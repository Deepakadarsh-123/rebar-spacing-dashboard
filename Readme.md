# Rebar Spacing Analysis Dashboard

A tool for automated analysis of rebar spacing from 3D point cloud data (PLY files). 
The application uses clustering-based layer separation, PCA-based alignment, and 
histogram-based peak detection to identify and measure rebar spacing in construction elements.


## Features

- **Multi-Layer Analysis**: Automatically segment point clouds into layers using KMeans clustering
- **Interactive Dashboard**: Web-based interface built with Plotly Dash for real-time parameter adjustment
- **Rebar Detection**: Histogram-based algorithm with Gaussian smoothing for accurate rebar identification
- **Spacing Measurement**: Automated calculation of inter-rebar spacing from reconstructed point clouds
- **3D Visualization**: Integrated Open3D viewer for point cloud inspection
- **Export Capabilities**: Analysis logging and data export functionality

**Please note that the code contains a dia. analysis part too. PLease ignore that as the same hasn't been tested.**

## System Requirements

- Python 3.7 or higher
- Windows/Linux/MacOS
- Memory requirements depend on point cloud size; larger models may require higher RAM

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rebar-spacing-dashboard.git
cd rebar-spacing-dashboard
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration (IMPORTANT)

Before running the application, update the `PLY_FOLDER` variable in the script 
to point to the directory containing the input PLY files.

```python
PLY_FOLDER = r"path/to/your/ply/files"
```

## Usage

### Starting the Dashboard

Run the application:
```bash
python "Rebar Spacing Dashboard.py"
```

The dashboard will be accessible at `http://127.0.0.1:8050/` in the web browser.

### Workflow

1. **Select Model**: Choose a PLY file from the dropdown menu
2. **Configure Layers**: Adjust the KMeans cluster slider to determine the number of rebar layers
3. **Select Layer**: Choose which layer to analyze in detail
4. **Parameter Tuning**:
   - **Histogram Bins**: Controls resolution of rebar detection (100-1000)
   - **Gaussian Sigma**: Smoothing parameter for noise reduction (0-10)
   - **Peak Distance**: Minimum separation between rebars in bins (1-50)
   - **Min Points**: Threshold for valid peak detection (0-500)
5. **Manual Adjustment**: Fine-tune rotation offset if needed (-10° to +10°)
6. **Swap Axis**: Toggle projection axis if rebars are oriented differently
7. **Run Analysis**: Click to process and visualize results

### Analysis Tab

The Analysis tab provides comparative analysis features:
- View all detected layers simultaneously
- Enter planned/actual spacing values
- Export results for reporting

## Algorithm Overview

### Point Cloud Processing Pipeline

1. **Layer Segmentation**: KMeans clustering on Z-coordinates separates horizontal layers
2. **PCA Alignment**: Principal Component Analysis aligns each layer to canonical orientation
3. **RANSAC Refinement**: Robust regression ensures vertical alignment of rebar patterns
4. **Histogram Generation**: Projects points onto primary axis and builds density distribution
5. **Peak Detection**: Identifies rebar positions using scipy's peak finding with configurable thresholds
6. **Spacing Calculation**: Computes inter-rebar distances from detected peaks

## Output Files

- **analysis_log.csv**: Timestamped log of all analyses with parameters and processing time
- Interactive plots and tables within the dashboard
- Optional exports from the Analysis tab

## Key Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| KMeans Clusters | 2 | 2-10 | Number of horizontal layers to detect |
| Histogram Bins | 1000 | 100-1000 | Resolution of density histogram |
| Gaussian Sigma | 1 | 0-10 | Smoothing strength for noise reduction |
| Peak Distance | 21 | 1-50 | Minimum bins between detected rebars |
| Min Points | 50 | 0-500 | Minimum point count for valid peaks |

## Troubleshooting

**Issue**: "PLY folder not found" error
- **Solution**: Update `PLY_FOLDER` variable to correct path

**Issue**: No rebars detected
- **Solution**: Adjust peak distance, minimum points, or try swapping the projection axis

**Issue**: Too many false positives
- **Solution**: Increase minimum points threshold and peak distance

**Issue**: Poor layer separation
- **Solution**: Adjust KMeans cluster count to match actual number of layers

## Citation

If you use this code, please cite the associated paper:

(A BibTeX entry will be added after publication.)


## License

This repository is shared for academic and research use.

## Acknowledgments

- Open3D library for point cloud processing
- Plotly Dash for interactive visualization
- scikit-learn for machine learning algorithms


## Contact

For questions related to this repository or the CRC paper, please contact:
Kumar Adarsh – Deepakadarshstar@gmail.om



