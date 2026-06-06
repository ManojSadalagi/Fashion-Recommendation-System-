# Fashion Recommendation System

A deep learning-powered fashion recommendation system that suggests visually similar fashion products based on uploaded images.

## Project Overview

This project leverages **Transfer Learning** using **ResNet18** to extract high-dimensional visual features from over **44,000 fashion product images**. The extracted **512-dimensional feature vectors** are used with **K-Nearest Neighbors (KNN)** and **Cosine Similarity** to identify and recommend similar fashion items.

The system is deployed as an interactive **Streamlit web application**, allowing users to upload fashion images and receive real-time recommendations through a completely offline machine learning pipeline.

## Features

* Feature extraction using pre-trained ResNet18
* 44,000+ fashion products dataset
* 512-dimensional image embeddings
* KNN-based similarity search
* Cosine similarity for accurate recommendations
* Real-time image upload and recommendation
* Interactive Streamlit web interface
* Fully offline recommendation pipeline
* Fast and scalable retrieval system

## Tech Stack

* Python
* PyTorch
* Deep Learning
* Transfer Learning (ResNet18)
* Scikit-learn
* KNN Algorithm
* Cosine Similarity
* Streamlit
* Joblib
* NumPy
* Pandas

## Workflow

1. Collect and preprocess fashion product images.
2. Extract image embeddings using pre-trained ResNet18.
3. Store feature vectors for all products.
4. Apply KNN with cosine similarity for nearest-neighbor search.
5. Upload a new image through the Streamlit interface.
6. Generate recommendations based on visual similarity.
7. Display top similar fashion products to the user.

## Results

* Successfully generated accurate fashion recommendations based on visual appearance.
* Efficient retrieval from a dataset of 44,000+ products.
* User-friendly web interface for seamless interaction.

## Future Enhancements

* Hybrid recommendation system (image + metadata)
* Personalized user recommendations
* Cloud deployment
* Support for multiple fashion categories
* Advanced deep learning models (EfficientNet, Vision Transformers)

⭐ If you found this project useful, feel free to star the repository.
Fashion-Recommendation-System/
│
├── app.py
├── requirements.txt
├── README.md
├── images/
├── model/
│   ├── embeddings.pkl
│   ├── filenames.pkl
│   └── knn_model.pkl
│
└── dataset/
