import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from ultralytics import YOLO


st.set_page_config(
    page_title="YOLO COCO Visual Search",
    page_icon="🔎",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

@st.cache_resource
def load_yolo():
    # The model downloads automatically on first run if it is not present.
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_feature_model():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    return model, weights.transforms()

def detect(image: Image.Image, confidence: float):
    model = load_yolo()
    results = model.predict(source=np.array(image), conf=confidence, verbose=False)
    result = results[0]
    annotated = result.plot()
    annotated = Image.fromarray(annotated[:, :, ::-1])
    detections = []
    names = result.names

    if result.boxes is not None:
        for box, cls, conf in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            class_id = int(cls)
            detections.append({
                "class": names[class_id],
                "confidence": float(conf),
                "box": [float(x) for x in box],
            })
    return annotated, detections

def embedding(image: Image.Image):
    model, transform = load_feature_model()
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        vec = model(tensor)
        vec = F.normalize(vec, p=2, dim=1)
    return vec.squeeze(0).cpu()

def similarity(query_image, gallery_image):
    q = embedding(query_image)
    g = embedding(gallery_image)
    return float(torch.dot(q, g))

def image_from_bytes(data):
    return Image.open(io.BytesIO(data)).convert("RGB")


st.title("🔎 YOLO COCO Visual Search")
st.caption("Object detection with YOLOv8 + COCO classes, plus image similarity search.")

with st.sidebar:
    st.header("Settings")
    confidence = st.slider(
        "YOLO confidence threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.40,
        step=0.05,
    )
    top_k = st.slider("Number of search results", 1, 10, 5)
    st.info(
        "First run may take longer because YOLO and the ResNet feature model "
        "are downloaded automatically."
    )

tab1, tab2 = st.tabs(["Object Detection", "Visual Search"])

with tab1:
    st.subheader("1. Upload an image")
    uploaded = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        key="detection_upload",
    )

    if uploaded:
        image = image_from_bytes(uploaded.getvalue())
        col1, col2 = st.columns(2)

        with st.spinner("Running YOLO detection..."):
            annotated, detections = detect(image, confidence)

        with col1:
            st.image(image, caption="Original image", use_container_width=True)
        with col2:
            st.image(annotated, caption="YOLO detection result", use_container_width=True)

        st.subheader("Detected COCO objects")
        if detections:
            st.dataframe(
                [
                    {
                        "Object": d["class"],
                        "Confidence": f"{d['confidence']:.2%}",
                    }
                    for d in detections
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("No objects were detected above the selected confidence threshold.")

with tab2:
    st.subheader("2. Build a temporary image gallery")
    st.write(
        "Upload several gallery images. The app uses deep image features from "
        "ResNet-50 and cosine similarity to rank images against your query."
    )

    gallery_files = st.file_uploader(
        "Upload gallery images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="gallery_upload",
    )
    query_file = st.file_uploader(
        "Upload the query image",
        type=["jpg", "jpeg", "png"],
        key="query_upload",
    )

    if query_file and gallery_files:
        query_image = image_from_bytes(query_file.getvalue())

        with st.spinner("Computing visual similarity..."):
            scored = []
            for gf in gallery_files:
                gallery_image = image_from_bytes(gf.getvalue())
                score = similarity(query_image, gallery_image)
                scored.append((score, gf.name, gallery_image))

        scored.sort(reverse=True, key=lambda x: x[0])
        results = scored[:top_k]

        st.image(query_image, caption="Query image", width=300)
        st.subheader(f"Top {len(results)} visually similar images")

        cols = st.columns(min(5, len(results)))
        for i, (score, name, img) in enumerate(results):
            with cols[i % len(cols)]:
                st.image(img, caption=f"{name}\nSimilarity: {score:.3f}", use_container_width=True)
    elif query_file and not gallery_files:
        st.info("Upload at least one gallery image to perform visual search.")


st.divider()
st.caption("Academic project: YOLO + COCO object detection + deep-feature visual search + Streamlit.")
