import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import SimpleITK as sitk
from sklearn.preprocessing import LabelEncoder
from transformers import ViTImageProcessor, ViTModel
from sklearn.manifold import TSNE
from umap import UMAP
from matplotlib.patches import Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import requests
from io import BytesIO
import tempfile
import zipfile
import shutil
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from .predictor import ReferenceMapPredictor
from .config import CELL_TYPES, setup_colors

class CellTypeAnalyzer:
    def __init__(self, output_dir='results', 
                 annotation_dir=None,  # Add this parameter
                 excel_path=None,
                 sheet_name="Training Set"):
        
        # Set default paths
        self.excel_path = excel_path or r"assets/MapMySections_EntrantData.xlsx"
        self.sheet_name = sheet_name
        self.image_root_dir = "images"
        self.output_dir = output_dir
        
        # Set annotation paths with fallback to default location
        annotation_dir = annotation_dir or os.path.join(os.path.dirname(__file__), "annotation")
        self.allen_annotation_path = os.path.join(annotation_dir, "annotation_25.nii.gz")
        self.allen_boundary_path = os.path.join(annotation_dir, "annotation_boundary_25.nii.gz")
        
        self.cell_types = CELL_TYPES
        
        os.makedirs(output_dir, exist_ok=True)
        self.label_encoder = LabelEncoder().fit(self.cell_types)
        self.type_palette, self.border_palette = setup_colors(self.cell_types)
        self.load_allen_data()
        self.init_model()
        self.load_cell_data()
        self.predictor = None

    def load_allen_data(self):
        self.allen_annotation = sitk.GetArrayFromImage(sitk.ReadImage(self.allen_annotation_path))
        self.allen_boundary = sitk.GetArrayFromImage(sitk.ReadImage(self.allen_boundary_path))
        self.allen_boundaries = np.zeros_like(self.allen_annotation)
        for z in range(self.allen_annotation.shape[0]):
            self.allen_boundaries[z] = np.gradient(self.allen_annotation[z])[0] != 0

    def init_model(self):
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def download_and_extract_volume(self, url):
        """Download and extract a volume from a URL with improved error handling and cleanup"""
        temp_dir = None
        try:
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, os.path.basename(url))
            
            # Download the file with timeout and retry
            print(f"Downloading {url}...")
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except requests.exceptions.RequestException as e:
                print(f"Error downloading file: {e}")
                return None
            
            # Extract the file
            extract_dir = os.path.join(temp_dir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            except zipfile.BadZipFile as e:
                print(f"Error extracting zip file: {e}")
                return None
            
            # Search for the volume file (now looking for resampled_green_25.nii.gz specifically)
            target_file = "resampled_green_25.nii.gz"
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    if file == target_file:
                        volume_path = os.path.join(root, file)
                        # Copy to a persistent location before cleanup
                        persistent_path = os.path.join(self.output_dir, os.path.basename(volume_path))
                        shutil.copy2(volume_path, persistent_path)
                        return persistent_path
            
            print(f"Target file {target_file} not found in the extracted archive")
            return None
            
        except Exception as e:
            print(f"Error in download_and_extract_volume: {e}")
            return None
        finally:
            # Clean up temporary files
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Error cleaning up temporary directory: {e}")

    def load_cell_data(self):
        print("Loading training data...")
        self.train_df = pd.read_excel(self.excel_path, sheet_name='Training Set')
        test_df = pd.read_excel(self.excel_path, sheet_name='Test Set')
        
        self.embeddings = []
        self.labels = []
        self.counts = []
        self.image_paths = []
        self.allen_coords = []
        self._process_dataframe(self.train_df)
        
        self.test_embeddings = []
        self.test_image_paths = []
        self.test_df = test_df
        self._process_dataframe(test_df, is_test=True)
        
        # Convert to numpy arrays only if we have data
        if len(self.embeddings) > 0:
            self.embeddings = np.array(self.embeddings)
            self.labels = np.array(self.labels)
            self.counts = np.array(self.counts)
            self.encoded_labels = self.label_encoder.transform(self.labels)
            self.allen_coords = np.array(self.allen_coords)
        else:
            raise ValueError("No valid training data found")
            
        if len(self.test_embeddings) > 0:
            self.test_embeddings = np.array(self.test_embeddings)
            self.test_image_paths = np.array(self.test_image_paths)

    def _process_dataframe(self, df, is_test=False):
        for _, row in df.iterrows():
            # Skip rows without CCF Registered Image File Path
            if pd.isna(row['CCF Registered Image File Path']):
                continue
                
            try:
                # Handle both local files and URLs
                volume_path = row['CCF Registered Image File Path']
                
                # If the path is a URL, download and extract it
                if volume_path.startswith(('http://', 'https://')):
                    volume_path = self.download_and_extract_volume(volume_path)
                    if volume_path is None:
                        print(f"Failed to download volume from URL: {row['CCF Registered Image File Path']}")
                        continue
                
                # Check if the file exists
                if not os.path.exists(volume_path):
                    print(f"Volume file not found: {volume_path}")
                    continue
                    
                # Load the volume and extract three views
                volume = sitk.GetArrayFromImage(sitk.ReadImage(volume_path))
                
                # Ensure all views have the same dimensions before concatenation
                sagittal = volume[volume.shape[0]//2, :, :].T
                coronal = volume[:, volume.shape[1]//2, :].T
                axial = volume[:, :, volume.shape[2]//2].T
                
                # Resize views to ensure consistent dimensions before concatenation
                target_size = (224, 224)  # Target size for each view
                sagittal = np.array(Image.fromarray(sagittal).resize(target_size))
                coronal = np.array(Image.fromarray(coronal).resize(target_size))
                axial = np.array(Image.fromarray(axial).resize(target_size))
                
                # Combine views into a single image (stacked horizontally)
                combined_view = np.hstack([sagittal, coronal, axial])
                
                # Convert to PIL Image
                img = Image.fromarray(combined_view).convert('RGB')
                
                # Extract features from the combined view
                features = self.extract_features(img)
                
            except Exception as e:
                print(f"Error processing volume {row['CCF Registered Image File Path']}: {e}")
                continue
            
            if is_test:
                self.test_embeddings.append(features)
                self.test_image_paths.append(row['CCF Registered Image File Path'])
                continue
                
            sep_col = df.columns.get_loc("|")
            counts = row[sep_col+1:sep_col+1+len(self.cell_types)].values.astype(float)
            counts = np.nan_to_num(counts, 0)
            total = counts.sum()
            
            if total > 0:
                props = counts / total
                main_idx = np.argmax(props)
                
                self.embeddings.append(features)
                self.labels.append(self.cell_types[main_idx])
                self.counts.append(counts)
                self.image_paths.append(row['CCF Registered Image File Path'])
                
                if all(c in row for c in ['X_coord','Y_coord','Z_coord']):
                    self.allen_coords.append([
                        int(row['X_coord']) if not pd.isna(row['X_coord']) else 0, 
                        int(row['Y_coord']) if not pd.isna(row['Y_coord']) else 0, 
                        int(row['Z_coord']) if not pd.isna(row['Z_coord']) else 0
                    ])
                else:
                    self.allen_coords.append([0, 0, 0])

    def extract_features(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k:v.cuda() for k,v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

    def analyze(self, save_model=True):
        if len(self.embeddings) == 0:
            raise ValueError("No valid embeddings found - cannot proceed with analysis")
            
        features = StandardScaler().fit_transform(self.embeddings)
        umap_emb = UMAP(random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(features)
        tsne_emb = TSNE(random_state=42, perplexity=30).fit_transform(features)
        
        cell_type_counts = {
            cell_type: np.sum(self.counts[:, i]) for i, cell_type in enumerate(self.cell_types)
        }
        
        self.predictor = ReferenceMapPredictor(
            training_embeddings=self.embeddings,
            training_labels=self.labels,
            training_counts=self.counts,
            training_images=self.image_paths,
            umap_embeddings=umap_emb,
            cell_type_counts=cell_type_counts,
            type_palette=self.type_palette,
            border_palette=self.border_palette,
            cell_types=self.cell_types,
            processor=self.processor,
            model=self.model
        )
        
        if save_model:
            model_path = os.path.join(self.output_dir, 'cell_type_predictor.pkl')
            self.predictor.save(model_path)
            print(f"Saved trained model to {model_path}")
        
        self.plot_confidence_umap(umap_emb)
        self.plot_spatial_umap(umap_emb)
        self.plot_spatial_tsne(tsne_emb)
        
        if len(self.test_embeddings) > 0:
            test_predictions = self.predict_test_set(umap_emb)
            self.save_test_predictions(test_predictions)
        
        self.save_training_predictions()
        
        return umap_emb, tsne_emb

    def predict_test_set(self, umap_emb):
        test_predictions = []
        
        for i, (embedding, img_path) in enumerate(zip(self.test_embeddings, self.test_image_paths)):
            counts = self.predictor.predict_counts(embedding)
            similar_images, distances, similar_counts = self.predictor.find_similar_training_images(embedding)
            confidence = self.predictor.calculate_confidence(counts, similar_counts)
            
            # Save detailed prediction plot with image file name in title
            img_name = os.path.basename(img_path)
            pred_output_path = os.path.join(self.output_dir, f'test_prediction_{img_name}.png')
            self.predictor.visualize_prediction(embedding, img_path, pred_output_path)
            
            # Save similarity comparison plot with image file name
            self.plot_similarity_comparison(
                counts, 
                similar_counts, 
                os.path.join(self.output_dir, f'test_similarity_{img_name}.png'),
                title_suffix=f"\nTest Image: {img_name}"
            )
            test_predictions.append({
                'image_path': img_path,
                'image_name': img_name,
                'predicted_counts': counts,
                'confidence': confidence,
                'similar_images': similar_images,
                'similar_distances': distances,
                'similar_counts': similar_counts
            })
        
        test_df = pd.DataFrame(test_predictions)
        test_df.to_csv(os.path.join(self.output_dir, 'test_predictions.csv'), index=False)
        
        return test_predictions

    def plot_similarity_comparison(self, predicted, similar_counts, output_path, title_suffix=""):
        """Create a dedicated plot comparing predicted vs similar training samples"""
        if not similar_counts:
            return
        
        mean_similar = np.mean(similar_counts, axis=0)
        mean_similar = mean_similar / np.sum(mean_similar) * 100  # Normalize
        
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        
        width = 0.35
        x = np.arange(len(self.cell_types))
        
        # Only show top 10 cell types for clarity
        top_indices = np.argsort(predicted)[-10:][::-1]
        
        ax.bar(
            x[:len(top_indices)] - width/2, 
            predicted[top_indices], 
            width, 
            label='Predicted',
            color='#1f77b4',
            edgecolor='black'
        )
        
        ax.bar(
            x[:len(top_indices)] + width/2, 
            mean_similar[top_indices], 
            width, 
            label='Similar Training Avg',
            color='#ff7f0e',
            edgecolor='black'
        )
        
        ax.set_xticks(x[:len(top_indices)])
        ax.set_xticklabels([self.cell_types[i] for i in top_indices], rotation=45, ha='right', fontsize=16)
        ax.set_ylabel('Percentage (%)', fontsize=16)
        ax.set_title(f'Predicted vs Similar Training Samples{title_suffix}', fontsize=16, pad=20)
        ax.legend(fontsize=16)
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor='white')
        plt.close()

    def save_test_predictions(self, test_predictions):
        output_df = self.test_df.copy()
        sep_col = output_df.columns.get_loc("|")
        
        for cell_type in self.cell_types:
            if cell_type not in output_df.columns:
                output_df.insert(sep_col + 1, cell_type, 0)
        
        for i, prediction in enumerate(test_predictions):
            for j, cell_type in enumerate(self.cell_types):
                output_df.at[i, cell_type] = prediction['predicted_counts'][j]
        
        output_path = os.path.join(self.output_dir, 'Test_Set_Predictions.xlsx')
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            self.test_df.to_excel(writer, sheet_name='Original Test Set', index=False)
            output_df.to_excel(writer, sheet_name='Test Set Predictions', index=False)
            
            confidence_data = []
            for pred in test_predictions:
                row = {
                    'image_path': pred['image_path'],
                    'image_name': pred['image_name'],
                    'confidence': pred['confidence']
                }
                for j, cell_type in enumerate(self.cell_types):
                    row[cell_type] = pred['predicted_counts'][j]
                confidence_data.append(row)
            
            pd.DataFrame(confidence_data).to_excel(writer, sheet_name='Predicted Counts', index=False)
        
        print(f"Saved test set predictions to {output_path}")

    def save_training_predictions(self):
        training_results = []
        
        for i, (embedding, img_path) in enumerate(zip(self.embeddings, self.image_paths)):
            predicted_counts = self.predictor.predict_counts(embedding)
            actual_counts = self.counts[i]
            
            img_name = os.path.basename(img_path)
            sample_results = {
                'image_path': img_path,
                'image_name': img_name,
                'actual_label': self.labels[i]
            }
            
            for j, cell_type in enumerate(self.cell_types):
                sample_results[f'actual_{cell_type}'] = actual_counts[j]
                sample_results[f'predicted_{cell_type}'] = predicted_counts[j]
            
            training_results.append(sample_results)
        
        training_df = pd.DataFrame(training_results)
        output_path = os.path.join(self.output_dir, 'Training_Set_Results.xlsx')
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            training_df[['image_path', 'image_name', 'actual_label']].to_excel(writer, sheet_name='Label Summary', index=False)
            
            stats_data = []
            for cell_type in self.cell_types:
                stats_data.append({
                    'cell_type': cell_type,
                    'total_actual': training_df[f'actual_{cell_type}'].sum(),
                    'total_predicted': training_df[f'predicted_{cell_type}'].sum(),
                    'mean_actual': training_df[f'actual_{cell_type}'].mean(),
                    'mean_predicted': training_df[f'predicted_{cell_type}'].mean(),
                    'correlation': training_df[f'actual_{cell_type}'].corr(training_df[f'predicted_{cell_type}'])
                })
            
            pd.DataFrame(stats_data).to_excel(writer, sheet_name='Cell Type Stats', index=False)
            training_df.to_excel(writer, sheet_name='Full Data', index=False)
        
        print(f"Saved training set predictions to {output_path}")

    def plot_confidence_umap(self, embedding):
        fig = plt.figure(figsize=(16, 14), facecolor='white')
        gs = GridSpec(2, 1, height_ratios=[4, 1], figure=fig)
        ax = fig.add_subplot(gs[0])
        cax = fig.add_subplot(gs[1])

        # Use more visually distinct markers
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X']
        
        for i, cell_type in enumerate(self.cell_types):
            mask = (self.labels == cell_type)
            if mask.any():
                ax.scatter(
                    embedding[mask, 0], 
                    embedding[mask, 1],
                    color=self.type_palette[i],
                    s=60,  # Increased marker size
                    alpha=0.7,
                    edgecolor=self.border_palette[i],
                    linewidth=0.8,
                    label=cell_type,
                    marker=markers[i % len(markers)]
                )

        handles = [
            plt.Line2D(
                [0], [0], 
                marker=markers[i % len(markers)], 
                color='w',
                markerfacecolor=self.type_palette[i], 
                markersize=12,  # Increased legend marker size
                label=label,
                markeredgecolor=self.border_palette[i],
                markeredgewidth=1
            )
            for i, label in enumerate(self.cell_types)
        ]

        ncol = min(4, int(np.ceil(len(self.cell_types)/8)))
        cax.legend(
            handles=handles, 
            ncol=ncol, 
            fontsize=16,  # Increased legend font size
            title="Cell Types", 
            title_fontsize=14,  # Increased title size
            frameon=True,
            framealpha=1,
            facecolor='white'
        )
        cax.axis('off')

        ax.set_title('UMAP Projection of Cell Type Embeddings', fontsize=16, pad=20)
        ax.set_xlabel('UMAP 1', fontsize=14)
        ax.set_ylabel('UMAP 2', fontsize=14)
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'confidence_umap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'Saved confidence UMAP visualization to {output_path}')

    def plot_spatial_umap(self, embedding):
        self._plot_spatial(embedding, 'UMAP')

    def plot_spatial_tsne(self, embedding):
        self._plot_spatial(embedding, 't-SNE')

    def _plot_spatial(self, embedding, method):
        self._create_base_plot(embedding, method, include_thumbnails=False)
        self._create_base_plot(embedding, method, include_thumbnails=True)

    def _create_base_plot(self, embedding, method, include_thumbnails):
        fig, ax = plt.subplots(figsize=(20, 16), facecolor='white')
        
        # Plot cell type clusters
        for i, cell_type in enumerate(self.cell_types):
            mask = np.where(self.labels == cell_type)[0]
            if len(mask) > 0:
                ax.scatter(
                    embedding[mask, 0], 
                    embedding[mask, 1],
                    color=self.type_palette[i],
                    label=cell_type,
                    s=100,  # Increased marker size
                    alpha=0.7,
                    edgecolor=self.border_palette[i],
                    linewidth=0.8
                )
        
        # Add Allen Brain boundaries if available
        if len(self.allen_coords) > 0 and np.any(self.allen_coords != 0):
            try:
                boundary_img = self._create_boundary_image(embedding)
                ax.imshow(
                    boundary_img, 
                    extent=[
                        embedding[:,0].min(), 
                        embedding[:,0].max(),
                        embedding[:,1].min(), 
                        embedding[:,1].max()
                    ],
                    alpha=0.2, 
                    origin='lower', 
                    cmap='Greys'
                )
            except Exception as e:
                print(f"Boundary overlay error: {e}")
        
        # Add thumbnails if requested
        if include_thumbnails:
            self._add_thumbnails(ax, embedding)
        
        title_suffix = " with Thumbnails" if include_thumbnails else ""
        ax.set_title(
            f'{method} Projection with Allen Brain Boundaries{title_suffix}', 
            fontsize=18,  # Increased title size
            pad=20
        )
        ax.set_xlabel('Component 1', fontsize=16)
        ax.set_ylabel('Component 2', fontsize=16)
        ax.grid(True, alpha=0.2)
        
        # Create custom legend with larger markers
        handles = [
            Patch(
                color=self.type_palette[i],
                label=label,
                edgecolor=self.border_palette[i],
                linewidth=1
            )
            for i, label in enumerate(self.cell_types)
        ]
        
        ax.legend(
            handles=handles, 
            bbox_to_anchor=(1.05, 1), 
            loc='upper left',
            fontsize=16,  # Increased legend font size
            frameon=True,
            framealpha=1,
            facecolor='white'
        )
        
        suffix = "_with_thumbnails" if include_thumbnails else ""
        plt.savefig(
            os.path.join(self.output_dir, f'{method.lower()}_spatial{suffix}.png'), 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white'
        )
        plt.close()

    def _create_boundary_image(self, embedding):
        x_min, x_max = embedding[:,0].min(), embedding[:,0].max()
        y_min, y_max = embedding[:,1].min(), embedding[:,1].max()
        grid_size = 100
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_size),
            np.linspace(y_min, y_max, grid_size)
        )
        
        boundary_img = np.zeros((grid_size, grid_size))
        
        for i in range(grid_size):
            for j in range(grid_size):
                dists = np.sqrt(
                    (embedding[:,0]-xx[i,j])**2 + 
                    (embedding[:,1]-yy[i,j])**2
                )
                nearest = np.argmin(dists)
                z = int(self.allen_coords[nearest, 2])
                if 0 <= z < len(self.allen_boundaries):
                    boundary_img[i,j] = self.allen_boundaries[z].mean()
        
        return boundary_img

    def _add_thumbnails(self, ax, embedding):
        tree = KDTree(embedding)
        x_min, x_max = embedding[:,0].min(), embedding[:,0].max()
        y_min, y_max = embedding[:,1].min(), embedding[:,1].max()
        
        # Create more grid points for thumbnail placement (increased from 6 to 10)
        grid_x = np.linspace(x_min, x_max, 10)[1:-1]  # Skip edges
        grid_y = np.linspace(y_min, y_max, 10)[1:-1]
        
        # Plot representative thumbnails across the space
        for x in grid_x:
            for y in grid_y:
                radius = (x_max - x_min) / 15  # Smaller radius to get more thumbnails
                indices = tree.query_ball_point([x, y], radius)
                
                if indices:
                    best_idx = indices[0]
                    cell_type = self.labels[best_idx]
                    type_idx = np.where(np.array(self.cell_types) == cell_type)[0][0]
                    
                    img_path = self.image_paths[best_idx]
                    try:
                        # Load the volume and create a combined view for the thumbnail
                        volume = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                        sagittal = volume[volume.shape[0]//2, :, :].T
                        coronal = volume[:, volume.shape[1]//2, :].T
                        axial = volume[:, :, volume.shape[2]//2].T
                        combined_view = np.hstack([sagittal, coronal, axial])
                        img = Image.fromarray(combined_view).convert('RGB').resize((180, 60))  # Smaller thumbnail
                        
                        im = OffsetImage(img, zoom=0.5)  # Reduced zoom from 0.8 to 0.5
                        ab = AnnotationBbox(
                            im, 
                            (embedding[best_idx, 0], embedding[best_idx, 1]),
                            frameon=True,
                            bboxprops=dict(
                                edgecolor=self.border_palette[type_idx],
                                lw=1,  # Reduced line width from 2 to 1
                                boxstyle="round,pad=0.2",  # Reduced padding
                                alpha=0.8  # Slightly more transparent
                            )
                        )
                        ax.add_artist(ab)
                        
                        # Only show cell type label, not image name
                        if len(grid_x) * len(grid_y) < 50:  # Only add labels if not too many points
                            ax.text(
                                embedding[best_idx, 0], 
                                embedding[best_idx, 1] - (y_max-y_min)*0.03,  # Reduced offset
                                cell_type,  # Only show cell type, not image name
                                ha='center',
                                va='top',
                                fontsize=16,  # Adjusted font size
                                bbox=dict(
                                    facecolor='white',
                                    alpha=0.7,
                                    edgecolor='none',
                                    boxstyle='round,pad=0.1'  # Reduced padding
                                )
                            )
                    except Exception as e:
                        print(f"Error loading volume for thumbnail: {e}")
        
        # Add representative thumbnails for each cell type (smaller and fewer)
        for i, cell_type in enumerate(self.cell_types):
            mask = np.where(self.labels == cell_type)[0]
            if len(mask) > 0:
                # Find the sample closest to the centroid
                centroid = np.mean(embedding[mask], axis=0)
                dists = np.linalg.norm(embedding[mask] - centroid, axis=1)
                best_idx = mask[np.argmin(dists)]
                
                img_path = self.image_paths[best_idx]
                try:
                    # Load the volume and create a combined view for the thumbnail
                    volume = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                    sagittal = volume[volume.shape[0]//2, :, :].T
                    coronal = volume[:, volume.shape[1]//2, :].T
                    axial = volume[:, :, volume.shape[2]//2].T
                    combined_view = np.hstack([sagittal, coronal, axial])
                    img = Image.fromarray(combined_view).convert('RGB').resize((240, 80))  # Slightly larger for cell types
                    
                    im = OffsetImage(img, zoom=0.6)  # Reduced zoom from 0.9 to 0.6
                    ab = AnnotationBbox(
                        im, 
                        (embedding[best_idx, 0], embedding[best_idx, 1]),
                        frameon=True,
                        bboxprops=dict(
                            edgecolor=self.border_palette[i],
                            lw=2,  # Reduced from 3
                            boxstyle="round,pad=0.3",  # Reduced padding
                            alpha=0.9  # Slightly more transparent
                        )
                    )
                    ax.add_artist(ab)
                    
                    # Only show cell type label, not image name
                    ax.text(
                        embedding[best_idx, 0], 
                        embedding[best_idx, 1] - (y_max-y_min)*0.04,
                        cell_type,  # Only show cell type, not image name
                        ha='center',
                        va='top',
                        fontsize=16,
                        bbox=dict(
                            facecolor='white',
                            alpha=0.8,
                            edgecolor='none',
                            boxstyle='round,pad=0.2'
                        )
                    )
                except Exception as e:
                    print(f"Error loading cell type volume thumbnail: {e}")