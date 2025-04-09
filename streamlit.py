import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from ultralytics import SAM
import cv2
import pandas as pd
import re

# Load Models
yolo_model = YOLO("C:\\Users\\ADMIN\\Desktop\\Project-II\\tomato\\yolov8detect.pt")       # Object detection
classify_model = YOLO("C:\\Users\\ADMIN\\Desktop\\Project-II\\tomato\\best.pt")           # Classification
sam_model = SAM("sam2_b.pt")                                                               # Segmentation

# Load Fertilizer Data
fertilizer_df = pd.read_excel("C:\\Users\\ADMIN\\Desktop\\Project-II\\tomato\\Updated_Fertilizers_1.xlsx")

# ----------------------------
# Infection Percentage Function
# ----------------------------
def calculate_infection_percentage(leaf_region):
    hsv = cv2.cvtColor(leaf_region, cv2.COLOR_RGB2HSV)
    lower_bound = np.array([10, 50, 50])
    upper_bound = np.array([30, 255, 255])
    infection_mask = cv2.inRange(hsv, lower_bound, upper_bound)
    infected_pixels = np.count_nonzero(infection_mask)
    total_pixels = leaf_region.shape[0] * leaf_region.shape[1]
    infection_percentage = (infected_pixels / total_pixels) * 100 if total_pixels else 0
    return infection_percentage, infection_mask

# ----------------------------
# Fertilizer Recommendation Function
# ----------------------------
def recommend_fertilizer(disease_name, infection_percentage):
    matched_rows = fertilizer_df[fertilizer_df["Name of the disease"].str.lower() == disease_name.lower()]
    
    if matched_rows.empty:
        return "No specific fertilizer recommendation found."

    recommendations = []
    for _, row in matched_rows.iterrows():
        numeric_part = re.findall(r'\d+\.?\d*', row["Dosage"])
        unit_part = re.findall(r'[a-zA-Z]+', row["Dosage"])
        unit_part = unit_part[0] if unit_part else ""

        if numeric_part:
            dosage_value = float(numeric_part[0])
            infection_range = row["Percentage of Infection"].replace('%', '').strip()
            min_inf, max_inf = map(int, re.findall(r'\d+', infection_range))
            range_span = max_inf - min_inf if max_inf - min_inf > 0 else 1

            scaling_factor = (infection_percentage - min_inf) / range_span
            scaling_factor = max(0.1, min(scaling_factor, 1))

            adjusted_dosage = f"{dosage_value * scaling_factor:.2f} {unit_part}"
            recommendations.append(f"{row['Fertilizer']} - {adjusted_dosage}")

    return "\n".join(recommendations)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Tomato Leaf Disease Analyzer & Fertilizer Recommendation")

uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # -------------------
    # 1. Classification
    # -------------------
    st.write("Classifying disease from image...")
    classification_results = classify_model(img_array)
    class_id = classification_results[0].probs.top1
    predicted_class = classification_results[0].names[class_id]
    st.write(f"**Predicted Disease:** {predicted_class}")

    # -------------------
    # 2. Detection
    # -------------------
    st.write("Detecting leaf region with YOLO...")
    detection_results = yolo_model(img_array)
    yolo_boxes = detection_results[0].boxes.xyxy

    if len(yolo_boxes) == 0:
        st.warning("No leaf detected. Please upload a clear image.")
    else:
        for idx, box in enumerate(yolo_boxes):
            x1, y1, x2, y2 = map(int, box)
            leaf_region = img_bgr[y1:y2, x1:x2]
            leaf_region_rgb = cv2.cvtColor(leaf_region, cv2.COLOR_BGR2RGB)

            st.image(leaf_region_rgb, caption=f"Leaf Region {idx+1}", use_column_width=True)

            # -------------------
            # 3. Segmentation
            # -------------------
            st.write(f"Segmenting region {idx+1} with SAM...")
            sam_results = sam_model.predict(leaf_region_rgb, bboxes=[[0, 0, leaf_region.shape[1], leaf_region.shape[0]]])
            sam_masks = sam_results[0].masks

            if sam_masks is not None:
                for j, mask in enumerate(sam_masks.data.cpu().numpy()):
                    mask = (mask * 255).astype(np.uint8)
                    mask_resized = cv2.resize(mask, (leaf_region.shape[1], leaf_region.shape[0]))

                    # -------------------
                    # 4. Infection Percentage
                    # -------------------
                    st.write(f"Calculating infection percentage for region {idx+1} mask {j+1}...")
                    infection_percentage, infection_mask = calculate_infection_percentage(leaf_region_rgb)

                    st.image(infection_mask, caption=f"Infection Mask - {infection_percentage:.2f}%", use_column_width=True)
                    st.write(f"**Infection Percentage:** {infection_percentage:.2f}%")

                    # -------------------
                    # 5. Fertilizer Recommendation
                    # -------------------
                    st.write("Fertilizer Recommendation:")
                    recommendation = recommend_fertilizer(predicted_class, infection_percentage)
                    st.text(recommendation)

                    break  # Only use first mask per region for now
