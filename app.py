import streamlit as st
import folium
from streamlit_folium import st_folium
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os
import tempfile
import requests
from datetime import datetime
import json

# Page configuration
st.set_page_config(layout="wide", page_title="Plant Analysis System")

# Initialize session state
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'latitude' not in st.session_state:
    st.session_state.latitude = 0.0
if 'longitude' not in st.session_state:
    st.session_state.longitude = 0.0
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = None

# API Keys
MAPBOX_TOKEN = "ENTER-YOUR-MAPBO-API"
GOOGLE_STREET_VIEW_API_KEY = 'ENTER-YOUR-GSV-API'

# Load YOLO models
@st.cache_resource
def load_models():
    model1 = YOLO("best (1).pt")
    model2 = YOLO("best_classify_plants (1).pt")
    return model1, model2

# def process_detection(image_path, model1):
#     """Process image with detection model (Model 1)"""
#     results = model1(image_path)
#     processed_results = []
    
#     for result in results:
#         # Draw bounding boxes on the image
#         img = cv2.imread(image_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         for box in result.boxes:
#             x1, y1, x2, y2 = box.xyxy[0].numpy()
#             conf = box.conf[0].item()
            
#             # Draw rectangle
#             cv2.rectangle(img, 
#                          (int(x1), int(y1)), 
#                          (int(x2), int(y2)), 
#                          (255, 0, 0), 2)
            
#             # Add confidence text
#             cv2.putText(img, 
#                        f'Conf: {conf:.2f}', 
#                        (int(x1), int(y1)-10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 
#                        0.5, 
#                        (255, 0, 0), 
#                        2)
            
#             # Store cropped region
#             crop = img[int(y1):int(y2), int(x1):int(x2)]
#             processed_results.append({
#                 'full_image': img,
#                 'crop': crop,
#                 'confidence': conf,
#                 'bbox': [x1, y1, x2, y2]
#             })
            
#     return processed_results
def process_detection(image_path, model1):
    """Process image with detection model (Model 1)"""
    results = model1(image_path)
    processed_results = []
    
    for result in results:
        # Draw bounding boxes on the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        for box in result.boxes:
            # Ensure tensor is on CPU before converting to NumPy
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].item()
            
            # Draw rectangle
            cv2.rectangle(img, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         (255, 0, 0), 2)
            
            # Add confidence text
            cv2.putText(img, 
                       f'Conf: {conf:.2f}', 
                       (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (255, 0, 0), 
                       2)
            
            # Store cropped region
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            processed_results.append({
                'full_image': img,
                'crop': crop,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })
            
    return processed_results


def process_classification(crops, model2):
    """Process cropped regions with classification model (Model 2)"""
    classification_results = []
    
    for crop in crops:
        # Save crop temporarily
        temp_crop_path = f"temp_crop_{datetime.now().timestamp()}.jpg"
        cv2.imwrite(temp_crop_path, cv2.cvtColor(crop['crop'], cv2.COLOR_RGB2BGR))
        
        # Get classification
        result = model2(temp_crop_path)
        class_name = result[0].names[result[0].probs.top1]
        conf = result[0].probs.top1conf.item()
        
        classification_results.append({
            'crop': crop['crop'],
            'class': class_name,
            'confidence': conf
        })
        
        # Clean up
        os.remove(temp_crop_path)
        
    return classification_results

def get_satellite_image(lat, lon, zoom=18):
    """Get satellite image from Mapbox"""
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom},0/600x300@2x?access_token={MAPBOX_TOKEN}"
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    return None

def get_street_view_image(lat, lon, heading=0):
    """Get street-level image from Google Street View API"""
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    
    # Parameters for the Street View API
    params = {
        'size': '600x300',  # Image size
        'location': f'{lat},{lon}',
        'key': GOOGLE_STREET_VIEW_API_KEY,
        'heading': heading,
        'fov': 90,  # Field of view
        'pitch': 0,  # Camera pitch
        'radius': 100  # Search radius in meters
    }
    
    # First, check if Street View is available at this location
    metadata_url = f"{base_url}/metadata"
    metadata_response = requests.get(metadata_url, params=params)
    metadata = metadata_response.json()
    
    if metadata.get('status') == 'OK':
        # Get the actual street view image
        image_url = base_url
        response = requests.get(image_url, params=params)
        
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
    
    return None

def get_multiple_street_views(lat, lon):
    """Get multiple street view images from different angles"""
    headings = [0, 90, 180, 270]  # Get views from different angles
    images = []
    
    for heading in headings:
        img = get_street_view_image(lat, lon, heading)
        if img:
            images.append(img)
    
    return images

# Create two columns for the main layout
col1, col2 = st.columns(2)

# Left column - Image Upload and Processing
with col1:
    st.header("Plant Detection and Classification")
    
    tab1, tab2 = st.tabs(["Upload Image", "Location Analysis"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state.current_image = image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Uploaded Image"):
                try:
                    model1, model2 = load_models()
                    
                    with st.spinner("Processing image..."):
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_file_path = tmp_file.name
                        
                        # Process with detection model
                        detection_results = process_detection(temp_file_path, model1)
                        
                        if detection_results:
                            # Process with classification model
                            classification_results = process_classification(detection_results, model2)
                            
                            # Display results
                            st.subheader("Detection Results")
                            st.image(detection_results[0]['full_image'], 
                                    caption="Detected Plants", 
                                    use_column_width=True)
                            
                            st.subheader("Classification Results")
                            for i, (det, clf) in enumerate(zip(detection_results, classification_results)):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.image(det['crop'], caption=f"Plant {i+1}")
                                    st.write(f"Detection Confidence: {det['confidence']:.2f}")
                                with col_b:
                                    st.write(f"Classification: {clf['class']}")
                                    st.write(f"Classification Confidence: {clf['confidence']:.2f}")
                        else:
                            st.warning("No plants detected in the image.")
                            
                        os.remove(temp_file_path)
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    
    with tab2:
        if st.session_state.latitude != 0.0 and st.session_state.longitude != 0.0:
            st.write(f"Selected Location: {st.session_state.latitude:.6f}, {st.session_state.longitude:.6f}")
            
            # Get satellite and street view images
            with st.spinner("Loading location images..."):
                satellite_img = get_satellite_image(st.session_state.latitude, st.session_state.longitude)
                street_images = get_multiple_street_views(st.session_state.latitude, st.session_state.longitude)
                
                if satellite_img:
                    st.subheader("Satellite View")
                    st.image(satellite_img, caption="Satellite Image", use_column_width=True)
                    
                    if st.button("Analyze Satellite Image"):
                        try:
                            model1, model2 = load_models()
                            
                            with st.spinner("Processing satellite image..."):
                                # Save satellite image temporarily
                                temp_sat_path = "temp_satellite.jpg"
                                satellite_img.save(temp_sat_path)
                                
                                # Process with both models
                                detection_results = process_detection(temp_sat_path, model1)
                                if detection_results:
                                    classification_results = process_classification(detection_results, model2)
                                    
                                    # Display results
                                    st.image(detection_results[0]['full_image'], 
                                            caption="Detection Results", 
                                            use_column_width=True)
                                    
                                    for i, (det, clf) in enumerate(zip(detection_results, classification_results)):
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.image(det['crop'], caption=f"Plant {i+1}")
                                            st.write(f"Detection Confidence: {det['confidence']:.2f}")
                                        with col_b:
                                            st.write(f"Classification: {clf['class']}")
                                            st.write(f"Classification Confidence: {clf['confidence']:.2f}")
                                else:
                                    st.warning("No plants detected in satellite image.")
                                
                                os.remove(temp_sat_path)
                                
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                
                if street_images:
                    st.subheader("Street-Level Views")
                    tabs = st.tabs([f"View {i+1}" for i in range(len(street_images))])
                    
                    for i, (tab, img) in enumerate(zip(tabs, street_images)):
                        with tab:
                            st.image(img, caption=f"Street View {i+1} ({['North', 'East', 'South', 'West'][i]})", use_column_width=True)
                            
                            if st.button(f"Analyze View {i+1}"):
                                try:
                                    model1, model2 = load_models()
                                    
                                    with st.spinner("Processing street view..."):
                                        temp_path = f"temp_street_{i}.jpg"
                                        img.save(temp_path)
                                        
                                        detection_results = process_detection(temp_path, model1)
                                        if detection_results:
                                            classification_results = process_classification(detection_results, model2)
                                            
                                            st.image(detection_results[0]['full_image'], 
                                                    caption="Detection Results", 
                                                    use_column_width=True)
                                            
                                            for j, (det, clf) in enumerate(zip(detection_results, classification_results)):
                                                col_a, col_b = st.columns(2)
                                                with col_a:
                                                    st.image(det['crop'], caption=f"Plant {j+1}")
                                                    st.write(f"Detection Confidence: {det['confidence']:.2f}")
                                                with col_b:
                                                    st.write(f"Classification: {clf['class']}")
                                                    st.write(f"Classification Confidence: {clf['confidence']:.2f}")
                                        else:
                                            st.warning("No plants detected in this view.")
                                        
                                        os.remove(temp_path)
                                        
                                except Exception as e:
                                    st.error(f"An error occurred: {str(e)}")
                else:
                    st.info("No street-level images available for this location.")
        else:
            st.info("Select a location on the map to view and analyze images.")

# Right column - Map
with col2:
    st.header("Location Map")
    
    # Initialize the map with a default view
    default_location = [20.5937, 78.9629]  # Center of India
    m = folium.Map(
        location=default_location if st.session_state.latitude == 0.0 else [st.session_state.latitude, st.session_state.longitude],
        zoom_start=5 if st.session_state.latitude == 0.0 else 18,
        tiles="OpenStreetMap"
    )
    
    # Add marker for current location
    if st.session_state.latitude != 0.0 and st.session_state.longitude != 0.0:
        folium.Marker(
            [st.session_state.latitude, st.session_state.longitude],
            popup="Selected Location",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
    
    # Display the map
    map_data = st_folium(m, height=600, width=None)
    
    # Update location when map is clicked
    if map_data['last_clicked']:
        st.session_state.latitude = map_data['last_clicked']['lat']
        st.session_state.longitude = map_data['last_clicked']['lng']
        st.experimental_rerun()

# Add custom CSS
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
        }
        .uploadedFile {
            margin-bottom: 2rem;
        }
        .stHeader {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .st-emotion-cache-16idsys p {
            margin-bottom: 0.5rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px;
            padding: 8px 16px;
            font-size: 14px;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e0e2e6;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #1f77b4;
            color: white;
        }
        .stImage {
            margin-bottom: 1rem;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        .stMarkdown {
            font-size: 14px;
        }
        .css-1d391kg {
            padding-top: 1rem;
        }
        .stAlert {
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 4px;
        }
        .element-container {
            margin-bottom: 1rem;
        }
        .css-1544g2n.e1fqkh3o4 {
            padding: 1rem;
            border-radius: 4px;
            background-color: #ffffff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        .stSpinner {
            text-align: center;
            margin: 2rem 0;
        }
        .folium-map {
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        /* Improve table styling */
        .dataframe {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        .dataframe th, .dataframe td {
            padding: 8px;
            border: 1px solid #ddd;
            text-align: left;
        }
        .dataframe th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        .dataframe tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        /* Improve button styling */
        .stButton>button:hover {
            border-color: #1f77b4;
            color: #1f77b4;
        }
        .stButton>button:active {
            background-color: #1f77b4;
            color: white;
        }
        /* Improve file uploader styling */
        .uploadedFile>div {
            padding: 1rem;
            border: 1px dashed #ccc;
            border-radius: 4px;
            text-align: center;
        }
        .uploadedFile>div:hover {
            border-color: #1f77b4;
        }
        /* Improve columns layout */
        .row-widget.stHorizontal {
            gap: 1rem;
        }
        /* Improve header styling */
        h1, h2, h3 {
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        /* Improve text styling */
        p {
            line-height: 1.6;
            color: #424242;
        }
        /* Improve container styling */
        .css-1d391kg {
            background-color: white;
            padding: 2rem;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        /* Improve spacing */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
    </style>
    """, unsafe_allow_html=True)
