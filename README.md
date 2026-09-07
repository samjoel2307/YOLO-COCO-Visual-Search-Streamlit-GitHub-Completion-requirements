# YOLO COCO Visual Search + Streamlit

## 1. Project Title

**YOLO COCO Visual Search using Streamlit**

## 2. Abstract / Introduction

This project is a local computer-vision application that combines YOLO object detection with COCO class labels and visual image search in a Streamlit interface.

The application accepts an input image and detects common objects using a pretrained YOLO model trained on the COCO dataset. It displays bounding boxes, object names, and confidence scores.

A second module provides visual search. Users can upload a query image and a temporary gallery of images. Deep image features are extracted with a pretrained ResNet-50 model and cosine similarity is used to rank the gallery images by visual similarity.

The project is designed to be executed locally in **VS Code using a Conda environment**, as required for submission.

## 3. Dataset & YOLO Model Details (COCO)

### COCO

COCO (Common Objects in Context) is a widely used computer-vision dataset containing images and annotations for object detection, segmentation, and related tasks.

The pretrained YOLO model used in this project recognizes the standard COCO object categories, including examples such as:

- person
- car
- bicycle
- bus
- dog
- cat
- chair
- bottle
- laptop
- cell phone

The exact detections depend on the input image and the confidence threshold.

### YOLO model

This project uses **YOLOv8n (`yolov8n.pt`)**, a lightweight YOLO model suitable for local experimentation and CPU systems.

The model file is downloaded automatically by Ultralytics on the first execution, so the large model weight file does not need to be committed to GitHub.

## 4. Environment Setup

### Prerequisites

Install the following on Windows:

1. Anaconda or Miniconda
2. Visual Studio Code
3. Python support in VS Code
4. Git

Open **Anaconda Prompt** or a VS Code terminal configured for Conda.

Create the project environment:

```bash
conda create -n yolo_visual python=3.11 -y
conda activate yolo_visual
```

Then open the project folder in VS Code and select the `yolo_visual` Python interpreter.

## 5. CPU Installation Steps

For a typical Acer TravelMate with Intel integrated graphics, CPU execution is appropriate.

Activate the environment:

```bash
conda activate yolo_visual
```

Install PyTorch CPU packages:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Install the remaining packages:

```bash
pip install -r requirements.txt
```

> If your course provides a specific installation command, use the course command instead.

### GPU Installation Steps

If the computer has a supported NVIDIA GPU, use the PyTorch installation command recommended for the installed CUDA version from the official PyTorch instructions, then run:

```bash
pip install -r requirements.txt
```

Do not install CUDA packages blindly. The correct PyTorch/CUDA combination depends on the GPU, driver, and current PyTorch release.

## 6. How to Run in VS Code using Conda

Open the project folder in VS Code.

Open **Terminal → New Terminal**.

Activate Conda:

```bash
conda activate yolo_visual
```

Confirm Python:

```bash
python --version
```

Confirm the environment:

```bash
where python
```

The path should point to the `yolo_visual` Conda environment.

Run a basic import check:

```bash
python -c "import torch, torchvision, ultralytics, streamlit; print('Environment OK')"
```

## 7. How to Deploy using Streamlit

From the project root, run:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that address in your browser.

The application has two tabs:

### Object Detection

1. Upload an image.
2. YOLO analyzes the image.
3. Bounding boxes are displayed.
4. COCO object names and confidence scores are shown.

### Visual Search

1. Upload a query image.
2. Upload several gallery images.
3. The application extracts deep image features.
4. Images are ranked using cosine similarity.
5. The top matching images are displayed with similarity scores.

## 8. Output Screenshots

The `Screenshots` folder is compulsory for the assignment.

Add **your own screenshots** after running the application locally.

Required screenshots:

| File | What it should show |
|---|---|
| `01_conda_environment.png` | Conda environment activated in VS Code terminal |
| `02_streamlit_terminal.png` | `streamlit run app.py` running in the terminal |
| `03_streamlit_ui.png` | Streamlit application in the browser |
| `04_object_detection.png` | YOLO detection result with bounding boxes and labels |

Recommended additional screenshot:

| File | What it should show |
|---|---|
| `05_visual_search.png` | Visual-search query and ranked gallery results |

**Do not use screenshots downloaded from the internet or copied from another learner.**

Place your screenshots in:

```text
Screenshots/
```

## 9. Enhancements / Innovations Added

The basic YOLO detector has been extended with a visual-search component.

### Enhancement 1 — Interactive Streamlit UI

The project provides a browser-based interface instead of requiring users to interact with a Python script directly.

### Enhancement 2 — Adjustable confidence threshold

Users can change the YOLO confidence threshold from the sidebar.

### Enhancement 3 — COCO object summary

Detected objects are presented with their COCO class names and confidence values.

### Enhancement 4 — Visual similarity search

A query image can be compared against a user-provided image gallery using deep features extracted from ResNet-50.

### Enhancement 5 — Top-K search

Users can choose how many visually similar results should be displayed.

## 10. Results & Conclusion

The application successfully combines object detection and visual search in one Streamlit application.

YOLO provides real-time-style object detection and identifies objects using COCO categories. The visual-search module provides a complementary way to compare images based on learned visual features.

The project demonstrates an end-to-end computer-vision workflow:

```text
Input Image
    ↓
YOLO Object Detection
    ↓
COCO Class + Confidence
    ↓
Streamlit Visualization

Query Image + Gallery
    ↓
ResNet-50 Feature Extraction
    ↓
Cosine Similarity
    ↓
Ranked Visual Search Results
```

The project can be extended in future work with a persistent FAISS index, a larger image database, CLIP embeddings, database storage, user authentication, and cloud deployment.

## Project Structure

```text
yolo-coco-visual-search/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   └── visual_search.py
│
├── data/
│   └── gallery/
│
└── Screenshots/
    ├── 01_conda_environment.png
    ├── 02_streamlit_terminal.png
    ├── 03_streamlit_ui.png
    ├── 04_object_detection.png
    └── 05_visual_search.png
```

## GitHub Submission Checklist

Before submitting:

- [ ] Repository is **public**
- [ ] `app.py` is present
- [ ] `requirements.txt` is present
- [ ] Supporting folders/files are present
- [ ] `README.md` is complete
- [ ] `Screenshots/` exists
- [ ] Own Conda screenshot is included
- [ ] Own Streamlit terminal screenshot is included
- [ ] Own Streamlit UI screenshot is included
- [ ] Own YOLO detection screenshot is included
- [ ] Application was actually tested locally
- [ ] `streamlit run app.py` works
- [ ] Final LMS submission contains **only the public GitHub repository link**

## Academic Integrity

This repository should be customized and tested by the learner. Screenshots must come from the learner's own computer and execution. Add your own observations, improvements, screenshots, and project details before submission.
