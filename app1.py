import streamlit as st
import folium
from streamlit_folium import st_folium
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

# Page configuration
st.set_page_config(layout="wide", page_title="Street View Plant Analysis")

# Initialize session state for map location
if 'latitude' not in st.session_state:
    st.session_state.latitude = 0.0
if 'longitude' not in st.session_state:
    st.session_state.longitude = 0.0

# Load YOLO models
@st.cache_resource
def load_models():
    model1 = YOLO("best (1).pt")
    model2 = YOLO("best_classify_plants (1).pt")
    return model1, model2

# Process image function
def process_image(image, model1, model2):
    results = []
    
    # Get predictions from first model
    result1s = model1(image)
    
    for i, result1 in enumerate(result1s):
        if len(result1.boxes.xywh) > 0:
            x, y, w, h = result1.boxes.xywh[0]
            
            # Convert image to numpy array if it's not already
            if isinstance(image, (str, bytes, io.BytesIO)):
                image = np.array(Image.open(image))
            
            # Crop image
            cropped_image = image[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
            
            # Get prediction from second model
            result2 = model2(cropped_image)
            
            # Store results
            results.append({
                'detection': result1,
                'classification': result2[0].names[result2[0].probs.top1],
                'cropped_image': cropped_image
            })
    
    return results

# Create two columns
col1, col2 = st.columns(2)

# Left column - Map
with col1:
    st.header("Location Selection")
    
    # Initialize the map
    m = folium.Map(location=[st.session_state.latitude, st.session_state.longitude], zoom_start=2)
    
    # Add marker if location is selected
    if st.session_state.latitude != 0.0 and st.session_state.longitude != 0.0:
        folium.Marker(
            [st.session_state.latitude, st.session_state.longitude],
            popup="Selected Location",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
    
    # Display the map
    map_data = st_folium(m, height=400)
    
    # Update location if map is clicked
    if map_data['last_clicked']:
        st.session_state.latitude = map_data['last_clicked']['lat']
        st.session_state.longitude = map_data['last_clicked']['lng']
        
    # Display coordinates
    st.write(f"Selected Location: {st.session_state.latitude:.4f}, {st.session_state.longitude:.4f}")

# Right column - Image Upload and Analysis
with col2:
    st.header("Image Analysis")
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an image for analysis", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Process button
        if st.button("Analyze Image"):
            try:
                # Load models
                model1, model2 = load_models()
                
                # Process image
                results = process_image(uploaded_file, model1, model2)
                
                # Display results
                st.subheader("Analysis Results")
                
                for i, result in enumerate(results):
                    st.write(f"Detection {i+1}:")
                    st.write(f"Classification: {result['classification']}")
                    st.image(result['cropped_image'], caption=f"Detected Region {i+1}", use_column_width=True)
                    
                    # Display confidence scores and bounding box coordinates
                    conf = result['detection'].boxes.conf[0].item()
                    st.write(f"Confidence: {conf:.2f}")
                    
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")

# Add some CSS to improve the layout
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
        }
        .uploadedFile {
            margin-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)