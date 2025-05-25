import os
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import pdist, squareform
from .data_processor import DataProcessor
from .feature_extractor import FeatureExtractor
from .visualizations import (
    plot_distance_matrix,
    plot_brain_region_atlas,
    plot_thumbnail_overview,
    plot_cell_type_gradients
)
from .config import CELL_TYPES, TYPE_TO_CATEGORY, TYPE_PALETTE, CONCENTRATION_PALETTE, BACKGROUND_COLOR, BACKGROUND_ALPHA
from .utils import save_figure, save_excel

class CellTypeGradientVisualizer:
    def __init__(self, excel_path, sheet_name, image_root_dir, output_dir='cell_type_gradients_results'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cell_types = CELL_TYPES
        self.type_to_category = TYPE_TO_CATEGORY
        self.type_palette = TYPE_PALETTE
        self.concentration_palette = CONCENTRATION_PALETTE
        self.background_color = BACKGROUND_COLOR
        self.background_alpha = BACKGROUND_ALPHA
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.data_processor = DataProcessor(excel_path, sheet_name, image_root_dir, self.cell_types)
        self.feature_extractor = FeatureExtractor(self.device)
        
        self.process_data()
        
    def process_data(self):
        self.df = self.data_processor.load_data()
        self.embeddings, self.labels, self.image_paths, self.confidences = self.data_processor.process_data(self.feature_extractor)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.cell_types)
        self.encoded_labels = self.label_encoder.transform(self.labels)
        self.compute_global_umap()
        self.compute_distance_matrix()
        
    def compute_global_umap(self):
        print("Computing global UMAP embedding...")
        self.reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
        self.umap_emb = self.reducer.fit_transform(self.embeddings)
        
        self.global_x_min = np.min(self.umap_emb[:, 0])
        self.global_x_max = np.max(self.umap_emb[:, 0])
        self.global_y_min = np.min(self.umap_emb[:, 1])
        self.global_y_max = np.max(self.umap_emb[:, 1])

    def compute_distance_matrix(self):
        print("Computing distance matrix...")
        centroids = []
        valid_types = []
        
        for ct in self.cell_types:
            mask = self.labels == ct
            if sum(mask) > 0:
                centroid = np.median(self.umap_emb[mask], axis=0)
                centroids.append(centroid)
                valid_types.append(ct)
        
        centroids = np.array(centroids)
        distances = squareform(pdist(centroids, 'euclidean'))
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        self.normalized_distances = (distances - min_dist) / (max_dist - min_dist)
        
        self.distance_matrix = pd.DataFrame(
            distances,
            index=valid_types,
            columns=valid_types
        )

    def run_all_visualizations(self):
        plot_cell_type_gradients(self)
        plot_thumbnail_overview(self)
        plot_distance_matrix(self)
        plot_brain_region_atlas(self)