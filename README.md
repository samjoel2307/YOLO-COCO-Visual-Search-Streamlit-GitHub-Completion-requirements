
# 🔎 YOLO COCO Visual Search using Streamlit

A local computer-vision project that combines **YOLOv8 object detection**, **COCO classes**, and **deep-feature visual search** in an interactive **Streamlit** application.

> **Submission note:** This project is intended to be executed locally in **VS Code with a Conda environment**, according to the assignment requirements. The screenshots in `Screenshots/` are clearly marked placeholders until you replace them with screenshots from your own computer.

---

## 1. Project Title

### YOLO COCO Visual Search + Streamlit

---

## 2. Abstract / Introduction

This project demonstrates an end-to-end computer-vision application using a pretrained YOLO model and a Streamlit web interface.

The first part of the application performs object detection. A user uploads an image, YOLOv8 detects objects, and the application displays bounding boxes, COCO class names, and confidence scores.

The second part implements visual search. A user provides a query image and a temporary gallery of images. The application extracts deep visual features using a pretrained ResNet-50 network and compares the query with gallery images using cosine similarity. The gallery is then ranked from most visually similar to least similar.

The application is designed for local execution and can be demonstrated directly from a VS Code terminal using:

```bash
streamlit run app.py
```

---

## 3. Dataset & YOLO Model Details (COCO)

### COCO Dataset

The **Common Objects in Context (COCO)** dataset is a widely used benchmark for computer vision. Its object-detection annotations contain 80 common object categories.

Examples of COCO categories include:

- person
- bicycle
- car
- motorcycle
- bus
- train
- truck
- boat
- traffic light
- dog
- cat
- horse
- chair
- couch
- laptop
- cell phone
- bottle

The pretrained YOLO model used by this application uses COCO-trained weights.

### YOLO Model

This project uses:

**YOLOv8n (`yolov8n.pt`)**

The `n` version is a lightweight model selected because it is practical for local execution, including CPU-based laptops.

The model is automatically downloaded by Ultralytics the first time the application runs. The model weight file should **not** be committed to the GitHub repository.

---

## 4. Environment Setup

### Recommended software

- Windows 10/11
- Visual Studio Code
- Miniconda or Anaconda
- Python 3.11
- Git
- A modern web browser

### Create the Conda environment

Open Anaconda Prompt or the VS Code terminal:

```bash
conda create -n yolo_visual python=3.11 -y
```

Activate it:

```bash
conda activate yolo_visual
```

Verify Python:

```bash
python --version
```

---

## 5. CPU Installation Steps

For computers without a supported NVIDIA GPU, CPU execution can be used.

Activate the environment:

```bash
conda activate yolo_visual
```

Install the CPU version of PyTorch and TorchVision:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import torch, torchvision, ultralytics, streamlit; print('Environment OK')"
```

### GPU Installation

If the computer has a supported NVIDIA GPU, install the PyTorch build recommended for the installed NVIDIA driver/CUDA configuration, then install:

```bash
pip install -r requirements.txt
```

The exact GPU command should be selected from the current PyTorch installation instructions rather than assuming a CUDA version.

---

## 6. How to Run in VS Code using Conda

### Step 1 — Open the project

Open this folder in VS Code:

```text
yolo-coco-visual-search
```

### Step 2 — Select the Python interpreter

In VS Code:

```text
Ctrl + Shift + P
→ Python: Select Interpreter
→ yolo_visual
```

### Step 3 — Activate Conda

In the VS Code terminal:

```bash
conda activate yolo_visual
```

### Step 4 — Verify the environment

```bash
where python
```

The returned path should point to the `yolo_visual` Conda environment.

### Step 5 — Run the application

From the project root:

```bash
streamlit run app.py
```

---

## 7. How to Deploy using Streamlit

The required command is:

```bash
streamlit run app.py
```

After starting the application, Streamlit provides a local browser address.

The application contains two main sections.

### Object Detection

1. Open the **Object Detection** tab.
2. Upload a JPG, JPEG, or PNG image.
3. YOLO analyzes the image.
4. The original image is shown.
5. The detected image is shown with bounding boxes.
6. Detected COCO objects and confidence scores are displayed.

### Visual Search

1. Open the **Visual Search** tab.
2. Upload one query image.
3. Upload several gallery images.
4. The application extracts deep image features.
5. Cosine similarity is calculated.
6. The gallery is ranked by similarity.
7. The top results are displayed.

---

# 8. Output Screenshots

**Important:** The screenshots must be captured by the learner from their own VS Code terminal and browser. The placeholder images currently included in this repository are only there to show the required filenames/layout. **Replace them before submitting the repository.**

## 8.1 Conda Environment Activation

Required evidence:

```text
VS Code
→ Terminal
→ conda activate yolo_visual
```

<img width="1280" height="720" alt="01_conda_environment" src="https://github.com/user-attachments/assets/a401d529-133d-4f73-9c39-7ee7e7d5b173" />


---

## 8.2 Streamlit Running in Terminal

Required evidence:

```bash
streamlit run app.py
```
<img width="1280" height="720" alt="02_streamlit_terminal" src="https://github.com/user-attachments/assets/8fd63c5c-e522-4eb2-97e3-15842a2439ad" />


---

## 8.3 Streamlit Web UI

Capture the running application in your browser.

The screenshot should clearly show the project title and Streamlit interface.

<img width="1280" height="720" alt="03_streamlit_ui" src="https://github.com/user-attachments/assets/04e71dbe-a748-4e08-903c-f45779737255" />


---

## 8.4 YOLO Object Detection Result

Upload an image containing recognizable objects and capture the detection result.

The screenshot should show:

- Input image
- Bounding boxes
- Object labels
- Confidence scores

<img width="1280" height="720" alt="04_object_detection" src="https://github.com/user-attachments/assets/cf6a5c52-4505-4abf-8eca-ce1e652cfd64" />

---

## 8.5 Visual Search Result

This additional screenshot demonstrates the visual-search enhancement.

The screenshot should show:

- Query image
- Gallery images
- Ranked results
- Similarity scores
<img width="1280" height="720" alt="05_visual_search" src="https://github.com/user-attachments/assets/5d65aa35-8080-46b6-84f8-e6d88d02028b" />


---

## 9. Enhancements / Innovations Added

### 9.1 Interactive Streamlit interface

Instead of running detection only from a Python command, users can interact with the application through a browser.

### 9.2 Adjustable confidence threshold

A sidebar slider allows users to change the YOLO detection confidence threshold.

### 9.3 COCO detection summary

Detected objects are presented with class names and confidence values.

### 9.4 Visual similarity search

The project extends object detection with an image-search feature using deep visual features from ResNet-50.

### 9.5 Top-K results

Users can select the number of visually similar gallery images to display.

### 9.6 Temporary user-provided gallery

The application does not require a large dataset to be committed to GitHub. Users can upload gallery images directly through Streamlit.

---

# 10. Results & Conclusion

The completed application demonstrates two related computer-vision tasks in a single interface.

### Object detection result

YOLOv8 successfully identifies objects from uploaded images using COCO-trained classes and displays the detections with bounding boxes and confidence scores.

### Visual search result

The visual-search module uses deep image features and cosine similarity to rank uploaded gallery images according to their visual similarity to a query image.

### Conclusion

The project provides a practical demonstration of:

```text
Image Input
     ↓
YOLOv8 Detection
     ↓
COCO Object Classes
     ↓
Bounding Boxes + Confidence
     ↓
Streamlit UI
```

and:

```text
Query Image + Gallery
          ↓
Deep Feature Extraction
          ↓
Cosine Similarity
          ↓
Ranking
          ↓
Top-K Visual Search Results
```

The project can be extended in future work by using CLIP embeddings, FAISS indexing, a persistent image database, larger galleries, object-specific search, and cloud deployment.

---

# Project Structure

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
│       └── .gitkeep
│
└── Screenshots/
    ├── 01_conda_environment.png
    ├── 02_streamlit_terminal.png
    ├── 03_streamlit_ui.png
    ├── 04_object_detection.png
    └── 05_visual_search.png
```

---

# GitHub Upload Checklist

Before submitting to the LMS:

- [ ] Repository is **PUBLIC**
- [ ] `app.py` uploaded
- [ ] `requirements.txt` uploaded
- [ ] `README.md` uploaded
- [ ] `src/` uploaded
- [ ] `data/` uploaded
- [ ] `Screenshots/` uploaded
- [ ] Conda activation screenshot replaced
- [ ] Streamlit terminal screenshot replaced
- [ ] Streamlit UI screenshot replaced
- [ ] Object detection screenshot replaced
- [ ] Visual-search screenshot replaced
- [ ] Application tested locally
- [ ] `streamlit run app.py` tested successfully
- [ ] README screenshots display correctly on GitHub
- [ ] Repository visibility verified as **Public**
- [ ] Only the GitHub repository link is submitted to LMS

---

# Academic Integrity

This project should be executed and tested by the learner. The screenshots included in the final submission must be the learner's own screenshots.

Customize the README with your own observations, screenshots, results, and any additional features you actually implemented. Do not copy another learner's repository or documentation.

---

# License

This project is intended for educational and academic demonstration purposes.
