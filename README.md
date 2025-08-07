# VibeCodedSlicer

Note: I did not write anything in this project, all credits go to Gemini CLI.

## Overview

VibeCodedSlicer is a Python-based tool designed for slicing 3D models (specifically STL files) into a series of 2D images. It allows for custom slicing planes by defining a normal vector, enabling angled slices. The tool also generates a 3D interactive visualization of the model, slicing planes, and cross-sections using Plotly.

## Features

- **3D Model Slicing**: Slice STL models along a custom-defined plane.
- **Angled Slicing**: Define a normal vector to perform angled slices, not just axis-aligned ones.
- **2D Image Output**: Generates a sequence of 2D PNG images representing each slice.
- **Interactive 3D Visualization**: Produces an interactive HTML file showing the original model, the slicing volume, and the generated cross-sections.
- **Debug Mode**: Option to save slice coordinates for debugging purposes.

## Setup

### 1. Clone the Repository (if applicable)

If you obtained this project as a repository, clone it:

```bash
git clone <repository_url>
cd VibeCodedSlicer
```

### 2. Conda Environment Setup

It is highly recommended to use the provided Conda environment for this project to ensure all dependencies are correctly managed.

First, ensure you have Anaconda or Miniconda installed.

Navigate to the project directory:

```bash
cd C:\Users\20202827\VibeCodedSlicer
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate VibeCodedSlicer
```

*(Note: An `environment.yml` file is assumed to exist based on previous interactions. If not, you can create one from `requirements.txt` or install directly.)*

### 3. Install Dependencies

If you are not using the `environment.yml` or need to update, install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

## Usage

To run the slicer, execute the `slicer.py` script from your activated `VibeCodedSlicer` Conda environment.

```bash
C:\Users\20202827\.conda\envs\VibeCodedSlicer\python.exe slicer.py
```

The script is configured to:

- Load `Eiffel.stl` from the `input_stl` directory.
- Rescale and center the model.
- Perform angled slicing (default: 30 degrees rotation around Y-axis for XY plane).
- Generate 40 slices.
- Enable debug mode (outputting slice coordinates).
- Generate the 3D Plotly visualization.

## Output

After running the script, the following outputs will be generated:

- **Sliced Images**: A series of PNG images (e.g., `slice_0000.png`, `slice_0001.png`, etc.) will be saved in the `output_angled/` directory. Each image represents a 2D cross-section of the 3D model.
- **3D Visualization**: An HTML file named `slicer_3d_visualization.html` will be saved in the `output_angled/` directory. Open this file in any web browser to view the interactive 3D representation of the model and its slices.
- **Debug Output**: If debug mode is enabled, text files containing slice coordinates will be saved in the `debug_output/` directory.

## Customization

You can modify the `slicer.py` script to customize its behavior:

- **Input Model**: Change `model_path` to point to a different STL file in the `input_stl` directory.
- **Scaling**: Adjust the `scale_factor` in `slicer.place_and_scale_model_in_volume()`.
- **Slicing Parameters**:
  - `num_slices`: Change the number of slices to generate.
  - `plane`, `angle_deg`, `rotation_axis`: Modify these parameters in `slicer.get_rotated_normal()` to change the slicing plane and angle.
- **Output Directory**: Change the `output_dir` when initializing `VibeCodedSlicer`.
- **Debug/Visualization**: Toggle `debug=True/False` and `visualize=True/False` in `slicer.slice()`.
