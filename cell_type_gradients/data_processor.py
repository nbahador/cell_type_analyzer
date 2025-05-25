import os
import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    def __init__(self, excel_path, sheet_name, image_root_dir, cell_types):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.image_root_dir = image_root_dir
        self.cell_types = cell_types
        
    def load_data(self):
        return pd.read_excel(self.excel_path, sheet_name=self.sheet_name)
        
    def load_image(self, image_path):
        try:
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                full_path = os.path.join(self.image_root_dir, os.path.basename(image_path))
                image = Image.open(full_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return Image.new('RGB', (224, 224))
            
    def process_data(self, feature_extractor):
        embeddings = []
        labels = []
        image_paths = []
        confidences = []
        
        df = self.load_data()
        
        for _, row in df.iterrows():
            if pd.isna(row['STPT Thumbnail Image']):
                continue
                
            image = self.load_image(row['STPT Thumbnail Image'])
            features = feature_extractor.extract_features(image)
            
            separator_col = df.columns.get_loc("|")
            cell_type_cols = df.columns[separator_col + 1:separator_col + 1 + len(self.cell_types)]
            counts = row[cell_type_cols].values.astype(float)
            counts = np.nan_to_num(counts, nan=0.0)
            total = counts.sum()
            
            if total > 0:
                proportions = counts / total
                main_type_idx = np.argmax(proportions)
                main_type_prop = proportions[main_type_idx]
                
                embeddings.append(features)
                labels.append(self.cell_types[main_type_idx])
                image_paths.append(row['STPT Thumbnail Image'])
                confidences.append(main_type_prop)
        
        return np.array(embeddings), np.array(labels), np.array(image_paths), np.array(confidences)