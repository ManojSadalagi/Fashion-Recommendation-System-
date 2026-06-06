import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import zipfile
import requests
import gdown

st.set_page_config(
    page_title="Fashion Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better image quality
st.markdown("""
<style>
    .stButton>button {
        background: linear-gradient(90deg, #FF69B4, #FF1493);
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    /* Improve image quality */
    .stImage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 5px;
    }
    img {
        border-radius: 8px;
        object-fit: cover;
    }
    /* Fix blurry images */
    .element-container img {
        image-rendering: -webkit-optimize-contrast;
        image-rendering: crisp-edges;
        image-rendering: pixelated;
    }
    /* Better card styling */
    .recommendation-card {
        background: white;
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        margin-bottom: 15px;
    }
    .recommendation-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    .similarity-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Google Drive Configuration
GOOGLE_DRIVE_FILE_ID = "18BZUrFg6aY5sujhWlhVsUUtDr0Q24SEP"
GOOGLE_DRIVE_ZIP_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"

@st.cache_resource
def setup_dataset():
    """Downloads and extracts the dataset from Google Drive (only on first run)."""
    zip_path = "myntradataset.zip"
    dataset_folder = "myntradataset"
    
    if os.path.exists(dataset_folder) and os.path.exists(f"{dataset_folder}/images"):
        st.sidebar.info("✅ Dataset already loaded")
        return dataset_folder
    
    if not os.path.exists(zip_path):
        try:
            with st.spinner("📥 Downloading dataset from Google Drive (3GB - first time only)..."):
                gdown.download(GOOGLE_DRIVE_ZIP_URL, zip_path, quiet=False)
                st.sidebar.success("✅ Dataset downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Error downloading dataset: {str(e)}")
            st.info("💡 Make sure your Google Drive link is public and the FILE_ID is correct")
            st.stop()
    
    if not os.path.exists(dataset_folder):
        try:
            with st.spinner("📂 Extracting dataset..."):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall()
                st.sidebar.success("✅ Dataset extracted successfully!")
        except Exception as e:
            st.error(f"❌ Error extracting dataset: {str(e)}")
            st.stop()
    
    if os.path.exists(f"{dataset_folder}/images"):
        image_count = len([f for f in os.listdir(f"{dataset_folder}/images") if f.endswith(('.jpg', '.jpeg', '.png'))])
        st.sidebar.success(f"✅ Dataset ready! {image_count:,} images found")
    else:
        st.warning("⚠️ Dataset structure might be incorrect. Expected: mytradataset/images/")
    
    return dataset_folder

class FeatureExtractor:
    def __init__(self, model_name='resnet18'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preserve original size for display, resize only for feature extraction
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(image_tensor)
        
        return features.squeeze().cpu().numpy()

def load_and_resize_image(image_path, max_size=(400, 400)):
    """Load image with proper resizing for high quality display"""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Use high-quality resampling
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def display_image_card(img, caption="", similarity_score=None, use_container_width=True):
    """Display image with high quality and optional similarity badge"""
    if img is None:
        return
    
    # Create a container for the card
    with st.container():
        # Display image with high quality
        st.image(img, use_container_width=use_container_width)
        
        # Display similarity badge if provided
        if similarity_score is not None:
            st.markdown(f'<div class="similarity-badge">🎯 {similarity_score:.1%} Similar</div>', 
                       unsafe_allow_html=True)
        
        # Display caption
        if caption:
            st.caption(caption)

class FashionRecommender:
    def __init__(self, features, metadata_df):
        self.features = features
        self.metadata = metadata_df
        
        self.knn_model = NearestNeighbors(
            n_neighbors=min(20, len(features)),
            metric='cosine',
            algorithm='brute'
        )
        self.knn_model.fit(features)
    
    def get_recommendations(self, item_index, n_recommendations=6):
        query_features = self.features[item_index].reshape(1, -1)
        distances, indices = self.knn_model.kneighbors(query_features, n_neighbors=n_recommendations+1)
        
        indices = indices[0][1:]
        distances = distances[0][1:]
        
        recommendations = self.metadata.iloc[indices].copy()
        recommendations['similarity_score'] = 1 - distances
        
        return recommendations
    
    def find_similar_to_uploaded(self, uploaded_features, n_recommendations=6):
        uploaded_features = uploaded_features.reshape(1, -1)
        distances, indices = self.knn_model.kneighbors(uploaded_features, n_neighbors=n_recommendations)
        
        indices = indices[0]
        distances = distances[0]
        
        recommendations = self.metadata.iloc[indices].copy()
        recommendations['similarity_score'] = 1 - distances
        
        return recommendations

@st.cache_resource
def load_model():
    if not os.path.exists('models/fashion_recommender.pkl'):
        st.error("Model not found! Please train the model first.")
        st.code("python train_myntra_model.py")
        st.stop()
    
    with open('models/fashion_recommender.pkl', 'rb') as f:
        model_data = joblib.load(f)
    
    features = model_data['features']
    metadata = model_data['metadata']
    
    # Fix paths for cross-platform compatibility
    if 'image_path' in metadata.columns:
        metadata['image_path'] = metadata['image_path'].str.replace('\\', '/', regex=False)
    
    st.sidebar.success("✅ Model Loaded Successfully!")
    st.sidebar.info(f"📊 Dataset Size: {len(metadata):,} items")
    
    recommender = FashionRecommender(features, metadata)
    extractor = FeatureExtractor('resnet18')
    
    return recommender, extractor, metadata

def browse_catalog_mode(recommender, metadata):
    st.header("🛍️ Browse Fashion Catalog")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**Total Items:** {len(metadata):,}")
    with col2:
        n_recommendations = st.slider("Number of Recommendations", 3, 9, 6)
    
    st.subheader("✨ Select an Item to Get Recommendations")
    
    # Initialize session state for sample items
    if 'sample_items' not in st.session_state or st.button("🔄 Shuffle Items", key="shuffle_btn"):
        st.session_state.sample_items = metadata.sample(min(12, len(metadata)))
        st.session_state.selected_idx = None
    
    sample_items = st.session_state.sample_items
    
    # Display items in a grid with better quality
    cols = st.columns(4)
    selected_idx = None
    
    for idx, (_, item) in enumerate(sample_items.iterrows()):
        col = cols[idx % 4]
        with col:
            # Load image with high quality
            img = load_and_resize_image(item['image_path'], max_size=(300, 300))
            if img:
                display_image_card(img, caption=item.get('filename', '')[:30])
                if st.button(f"🎯 Select", key=f"select_btn_{idx}"):
                    selected_idx = item.name
                    st.session_state.selected_idx = selected_idx
    
    # Show recommendations if an item is selected
    if st.session_state.get('selected_idx') is not None:
        st.markdown("---")
        st.subheader("🎯 Recommended Items")
        
        query_item = metadata.iloc[st.session_state.selected_idx]
        
        # Create two columns for query and recommendations
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 📸 Selected Item")
            query_img = load_and_resize_image(query_item['image_path'], max_size=(400, 400))
            if query_img:
                display_image_card(query_img, caption=query_item.get('filename', '')[:30])
        
        with col2:
            recommendations = recommender.get_recommendations(st.session_state.selected_idx, n_recommendations)
            
            if len(recommendations) > 0:
                # Display recommendations in a grid
                rec_cols = st.columns(3)
                for idx, (_, rec) in enumerate(recommendations.iterrows()):
                    col = rec_cols[idx % 3]
                    with col:
                        rec_img = load_and_resize_image(rec['image_path'], max_size=(300, 300))
                        if rec_img:
                            display_image_card(
                                rec_img, 
                                caption=rec.get('filename', '')[:25],
                                similarity_score=rec['similarity_score']
                            )
        
        # Add clear selection button
        if st.button("🗑️ Clear Selection"):
            st.session_state.selected_idx = None
            st.rerun()

def upload_image_mode(recommender, extractor, metadata):
    st.header("📸 Upload Your Fashion Image")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Load and display uploaded image with high quality
        image = Image.open(uploaded_file)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🖼️ Your Uploaded Image")
            # Display uploaded image with high quality
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Similar Items Found")
            n_recommendations = st.slider("Number of Results", 3, 12, 6)
            
            with st.spinner("🔍 Analyzing image features..."):
                features = extractor.extract_features(image)
            
            recommendations = recommender.find_similar_to_uploaded(features, n_recommendations)
            
            # Display recommendations in a grid
            rec_cols = st.columns(3)
            for idx, (_, rec) in enumerate(recommendations.iterrows()):
                col = rec_cols[idx % 3]
                with col:
                    rec_img = load_and_resize_image(rec['image_path'], max_size=(300, 300))
                    if rec_img:
                        display_image_card(
                            rec_img,
                            caption=rec.get('filename', '')[:25],
                            similarity_score=rec['similarity_score']
                        )

def analytics_dashboard(metadata):
    st.header("📊 Dataset Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Items", f"{len(metadata):,}")
    with col2:
        st.metric("Unique Images", len(metadata['image_id'].unique()) if 'image_id' in metadata.columns else "N/A")
    with col3:
        st.metric("Feature Dimension", "512D")
    
    st.markdown("---")
    st.subheader("🖼️ Sample Images from Dataset")
    
    # Display sample images in a grid with high quality
    sample_items = metadata.sample(min(12, len(metadata)))
    
    cols = st.columns(4)
    for idx, (_, item) in enumerate(sample_items.iterrows()):
        col = cols[idx % 4]
        with col:
            img = load_and_resize_image(item['image_path'], max_size=(250, 250))
            if img:
                st.image(img, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📋 Dataset Information")
    with st.expander("View Dataset Sample"):
        st.dataframe(metadata.head(20), use_container_width=True)

def main():
    # Title with custom styling
    st.markdown("""
    <h1 style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
               font-size: 3rem; margin-bottom: 1rem;'>
        👗 Fashion Recommendation System
    </h1>
    """, unsafe_allow_html=True)
    
    # Setup dataset from Google Drive
    dataset_path = setup_dataset()
    
    # Load model and data
    try:
        recommender, extractor, metadata = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure you have trained the model first using: python train_myntra_model.py")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Navigation")
    mode = st.sidebar.radio("", ["Browse Catalog", "Upload Image", "Analytics"], 
                           label_visibility="collapsed")
    
    # Add some information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **How it works:**
    1. 📸 Select or upload a fashion item
    2. 🔍 AI analyzes visual features
    3. 🎯 Get similar fashion recommendations
    """)
    
    # Main content based on selected mode
    if mode == "Browse Catalog":
        browse_catalog_mode(recommender, metadata)
    elif mode == "Upload Image":
        upload_image_mode(recommender, extractor, metadata)
    else:
        analytics_dashboard(metadata)

if __name__ == "__main__":
    main()