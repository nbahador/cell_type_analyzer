from cell_type_analyzer import load_saved_predictor
import os
import SimpleITK as sitk
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import tempfile
import zipfile
import shutil
import torch
import datetime
from visualizations import PredictionVisualizer

# Updated Configuration with new paths
image_url = "https://s3.us-west-2.amazonaws.com/allen-genetic-tools/bio_file_finder/Thumbnails/STPT/640002_STPT_Thumbnail.png"
model_path = os.path.join("trained_models", "cell_type_predictor.pkl")  # Updated to relative path
output_dir = "results"

def download_and_process_image(url):
    """Download and process image from URL with improved error handling"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Handle different image types
        if url.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(BytesIO(response.content)).convert('RGB')
            # Create standardized input by replicating the image
            target_size = (224, 224)
            img = img.resize(target_size)
            combined = np.hstack([np.array(img), np.array(img), np.array(img)])
            return Image.fromarray(combined)
        
        # Handle volume files in zip archives
        temp_dir = tempfile.mkdtemp()
        try:
            zip_path = os.path.join(temp_dir, os.path.basename(url))
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Search for volume files
            volume_extensions = ('.nii.gz', '.nii', '.nrrd')
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith(volume_extensions):
                        volume_path = os.path.join(root, file)
                        try:
                            volume = sitk.GetArrayFromImage(sitk.ReadImage(volume_path))
                            # Create standardized multi-view
                            views = [
                                volume[volume.shape[0]//2, :, :].T,  # Sagittal
                                volume[:, volume.shape[1]//2, :].T,  # Coronal
                                volume[:, :, volume.shape[2]//2].T   # Axial
                            ]
                            views = [Image.fromarray(v).resize((224, 224)) for v in views]
                            combined = np.hstack([np.array(v) for v in views])
                            return Image.fromarray(combined)
                        except Exception as e:
                            print(f"Error processing volume file {file}: {str(e)}")
                            continue
            
            raise FileNotFoundError("No supported volume file found in downloaded content")
        finally:
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        raise RuntimeError(f"Error processing image from URL: {str(e)}")

def save_prediction_results(output_dir, predictions, confidence, similar_info=None):
    """Save prediction results to a text file"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"prediction_results_{timestamp}.txt")
    
    with open(output_file, 'w') as f:
        f.write("=== Prediction Results ===\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Overall confidence score: {confidence:.2f}/1.0\n\n")
        f.write("Top predicted cell types:\n")
        
        for i, (cell_type, count) in enumerate(predictions[:10]):  # Show top 10
            f.write(f"{i+1}. {cell_type}: {count:.1f}%\n")
        
        if similar_info:
            f.write("\nSimilar training images info:\n")
            for i, (dist, count) in enumerate(zip(similar_info[1], similar_info[2])):
                f.write(f"Match {i+1}: Distance={dist:.4f}, Counts={count}\n")
    
    return output_file

def main():
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        print("\nLoading pre-trained model...")
        # Verify model exists before loading
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {os.path.abspath(model_path)}")
        
        predictor = load_saved_predictor(model_path)
        print("Model loaded successfully")
        
        # Verify critical components
        required_attrs = ['processor', 'model', 'predict_counts', 
                         'find_similar_training_images', 'calculate_confidence']
        for attr in required_attrs:
            if not hasattr(predictor, attr):
                raise AttributeError(f"Loaded predictor is missing required attribute: {attr}")
        
        print("\nProcessing test image...")
        test_image = download_and_process_image(image_url)
        print("Image processed successfully")
        
        print("\nExtracting features...")
        inputs = predictor.processor(images=test_image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k:v.cuda() for k,v in inputs.items()}
            predictor.model = predictor.model.cuda()
        
        with torch.no_grad():
            outputs = predictor.model(**inputs)
        
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        
        print("\nMaking predictions...")
        counts = predictor.predict_counts(features)
        similar_images, distances, similar_counts = predictor.find_similar_training_images(features)
        confidence = predictor.calculate_confidence(counts, similar_counts)
        
        print("\n=== Prediction Results ===")
        print(f"Overall confidence score: {confidence:.2f}/1.0")
        print("\nTop 5 predicted cell types:")
    
        sorted_predictions = sorted(zip(predictor.cell_types, counts), 
                                  key=lambda x: x[1], reverse=True)
    
        #for i, (cell_type, count) in enumerate(sorted_predictions[:5]):
        #    print(f"{i+1}. {cell_type}: {count:.1f}%")
    
        # Save all results to file
        #results_file = save_prediction_results(
        #    output_dir,
        #    sorted_predictions,
        #   confidence,
        #    similar_info=(similar_images, distances, similar_counts))
    
        #print(f"\nResults saved to: {os.path.abspath(results_file)}")


        for i, (cell_type, count) in enumerate(sorted_predictions[:5]):
            print(f"{i+1}. {cell_type}: {count:.1f}%")
    
        # Create and use visualizer
        visualizer = PredictionVisualizer(predictor, output_dir)
        figure_paths = visualizer.visualize_prediction(features, image_url)
    
        print("\nSaved visualizations:")
        for fig_type, path in figure_paths.items():
            if path:  # Some figures might be None if data wasn't available
                print(f"- {fig_type}: {os.path.abspath(path)}")
    
    
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTroubleshooting Guide:")
        print("1. Verify the model exists at:", os.path.abspath(model_path))
        print("2. Check your internet connection for image download")
        print("3. Ensure all required packages are installed and up-to-date")
        print("4. If problems persist, try retraining the model")
        print("5. Check the image URL is valid and accessible")

if __name__ == "__main__":
    # Initialize torch at the main level
    if not torch.cuda.is_available():
        print("Note: CUDA not available, using CPU")
    main()