import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import SimpleITK as sitk
from .predictor import ReferenceMapPredictor
from .analyzer import CellTypeAnalyzer

def load_saved_predictor(model_path):
    return ReferenceMapPredictor.load(model_path)

def test_saved_model(model_path, test_image_path, output_dir='results'):
    analyzer = CellTypeAnalyzer(output_dir=output_dir)
    
    # Load the test volume and create combined view
    volume = sitk.GetArrayFromImage(sitk.ReadImage(test_image_path))
    sagittal = volume[volume.shape[0]//2, :, :].T
    coronal = volume[:, volume.shape[1]//2, :].T
    axial = volume[:, :, volume.shape[2]//2].T
    combined_view = np.hstack([sagittal, coronal, axial])
    img = Image.fromarray(combined_view).convert('RGB').resize((224*3, 224))
    
    features = analyzer.extract_features(img)
    predictor = load_saved_predictor(model_path)
    counts = predictor.predict_counts(features)
    similar_images, distances, similar_counts = predictor.find_similar_training_images(features)
    
    # Generate and save visualization with image file name
    img_name = os.path.basename(test_image_path)
    output_path = os.path.join(output_dir, f'test_prediction_{img_name}.png')
    predictor.visualize_prediction(features, test_image_path, output_path)
    
    # Generate similarity comparison plot with image file name
    analyzer.plot_similarity_comparison(
        counts,
        similar_counts,
        os.path.join(output_dir, f'test_similarity_{img_name}.png'),
        title_suffix=f"\nTest Image: {img_name}"
    )
    
    return counts, similar_images, distances, similar_counts