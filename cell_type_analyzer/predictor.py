import os
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import pickle
import torch
from transformers import ViTImageProcessor, ViTModel

class ReferenceMapPredictor:
    def __init__(self, training_embeddings=None, training_labels=None, training_counts=None, 
                 training_images=None, umap_embeddings=None, cell_type_counts=None, 
                 type_palette=None, border_palette=None, cell_types=None,
                 processor=None, model=None):
        self.tree = None
        self.training_embeddings = training_embeddings
        self.training_labels = training_labels
        self.training_counts = training_counts
        self.training_images = training_images
        self.umap_embeddings = umap_embeddings
        self.kde_models = {}
        self.fixed_radius = None
        self.cell_type_counts = cell_type_counts or {}
        self.type_palette = type_palette or []
        self.border_palette = border_palette or []
        self.cell_types = cell_types or []
        self.regressor = None
        self.scaler = StandardScaler()
        self.pca = None
        self.processor = processor or ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = model or ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        if umap_embeddings is not None:
            self._initialize_umap_structures()

    def _initialize_umap_structures(self):
        """Initialize UMAP-related structures"""
        self.tree = KDTree(self.umap_embeddings)
        self._precompute_kde()
        self._compute_fixed_radius()

    def _precompute_kde(self):
        """Precompute KDE models for each cell type"""
        if self.training_labels is None or self.umap_embeddings is None:
            return
            
        unique_labels = np.unique(self.training_labels)
        for label in unique_labels:
            mask = (self.training_labels == label)
            if sum(mask) > 1:
                try:
                    self.kde_models[label] = gaussian_kde(self.umap_embeddings[mask].T)
                except:
                    pass

    def _compute_fixed_radius(self):
        """Compute the fixed radius for neighbor queries"""
        if self.umap_embeddings is None or len(self.umap_embeddings) < 10:
            return
            
        distances, _ = self.tree.query(self.umap_embeddings, k=10)
        self.fixed_radius = np.median(distances[:, -1])
        print(f"Using fixed neighbor query radius: {self.fixed_radius:.4f}")

    def train_regressor(self):
        if self.training_embeddings is None or self.training_counts is None:
            raise ValueError("Training data not available")
            
        X = self.scaler.fit_transform(self.training_embeddings)
        
        if X.shape[1] > 100:
            self.pca = PCA(n_components=100, random_state=42)
            X = self.pca.fit_transform(X)
        
        base_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.regressor = MultiOutputRegressor(base_model)
        self.regressor.fit(X, self.training_counts)

    def predict_counts(self, query_embedding):
        if self.regressor is None:
            self.train_regressor()
            
        query_embedding = self.scaler.transform(query_embedding.reshape(1, -1))
        if self.pca:
            query_embedding = self.pca.transform(query_embedding)
            
        counts = self.regressor.predict(query_embedding)[0]
        counts = np.maximum(counts, 0)
        counts = np.nan_to_num(counts, 0)
        
        total = counts.sum()
        if total > 0:
            counts = counts / total * 100
            
        return counts

    def find_similar_training_images(self, query_embedding, k=5):
        if self.tree is None:
            raise ValueError("Predictor not initialized. Load training data first.")
            
        query_umap = self.project_to_umap(query_embedding)
        distances, neighbor_indices = self.tree.query(query_umap, k=k)
        
        if isinstance(neighbor_indices, int):
            neighbor_indices = [neighbor_indices]
            
        neighbor_images = [self.training_images[i] for i in neighbor_indices]
        neighbor_distances = distances if isinstance(distances, np.ndarray) else [distances]
        neighbor_counts = [self.training_counts[i] for i in neighbor_indices]
        
        return neighbor_images, neighbor_distances, neighbor_counts

    def calculate_confidence(self, predicted_counts, similar_counts):
        if not similar_counts:
            return np.zeros_like(predicted_counts)
        
        mean_similar = np.mean(similar_counts, axis=0)
        pred_norm = predicted_counts / (np.sum(predicted_counts) + 1e-10)
        similar_norm = mean_similar / (np.sum(mean_similar) + 1e-10)
        
        confidence = np.dot(pred_norm, similar_norm) / (
            np.linalg.norm(pred_norm) * np.linalg.norm(similar_norm)
        )
        return confidence

    def project_to_umap(self, embedding):
        distances = np.linalg.norm(self.training_embeddings - embedding, axis=1)
        nearest_idx = np.argmin(distances)
        return self.umap_embeddings[nearest_idx]

    def save(self, filepath):
        save_data = {
            'training_embeddings': self.training_embeddings,
            'training_labels': self.training_labels,
            'training_counts': self.training_counts,
            'training_images': self.training_images,
            'umap_embeddings': self.umap_embeddings,
            'kde_models': self.kde_models,
            'fixed_radius': self.fixed_radius,
            'cell_type_counts': self.cell_type_counts,
            'type_palette': self.type_palette,
            'border_palette': self.border_palette,
            'cell_types': self.cell_types,
            'regressor': self.regressor,
            'scaler': self.scaler,
            'pca': self.pca,
            'processor_config': self.processor.config.to_dict(),
            'model_config': self.model.config.to_dict()
        }
        
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Create new instance with basic attributes
        predictor = cls(
            training_embeddings=save_data['training_embeddings'],
            training_labels=save_data['training_labels'],
            training_counts=save_data['training_counts'],
            training_images=save_data['training_images'],
            umap_embeddings=save_data['umap_embeddings'],
            cell_type_counts=save_data.get('cell_type_counts', {}),
            type_palette=save_data.get('type_palette', []),
            border_palette=save_data.get('border_palette', []),
            cell_types=save_data.get('cell_types', [])
        )
        
        # Restore additional attributes
        predictor.kde_models = save_data['kde_models']
        predictor.fixed_radius = save_data['fixed_radius']
        predictor.regressor = save_data.get('regressor')
        predictor.scaler = save_data.get('scaler', StandardScaler())
        predictor.pca = save_data.get('pca')
        
        # Reinitialize UMAP structures if needed
        if predictor.umap_embeddings is not None:
            predictor._initialize_umap_structures()
        
        return predictor