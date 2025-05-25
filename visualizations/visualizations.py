import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from PIL import Image
import seaborn as sns
import pandas as pd

class PredictionVisualizer:
    def __init__(self, predictor, output_dir='results'):
        """
        Initialize the visualizer with a trained predictor

        Args:
            predictor (ReferenceMapPredictor): Trained predictor instance
            output_dir (str): Directory to save visualizations
        """
        self.predictor = predictor
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set up consistent styling
        plt.style.use('default')
        self._set_plot_style()

    def _set_plot_style(self):
        """Configure consistent plot styling"""
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.dpi': 300,
            'figure.autolayout': True,
            'savefig.bbox': 'tight'
        })

    def visualize_prediction(self, query_embedding, query_image=None, img_name=None):
        """
        Create comprehensive visualization of prediction results

        Args:
            query_embedding: Feature vector of query image
            query_image: Path to query image file
            img_name: Name of the image for labeling

        Returns:
            dict: Paths to saved figures
        """
        # Generate base filename
        if img_name:
            base_name = os.path.splitext(img_name)[0]
        else:
            base_name = "prediction_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # Make predictions (now getting actual counts instead of percentages)
        counts = self.predictor.predict_counts(query_embedding)
        similar_images, distances, similar_counts = self.predictor.find_similar_training_images(query_embedding)
        confidence = self.predictor.calculate_confidence(counts, similar_counts)

        # Generate all visualizations
        figure_paths = {
            'main': self._create_main_prediction_figure(
                query_embedding, counts, confidence,
                similar_images, distances, similar_counts,
                query_image, base_name
            ),
            'similarity': self._create_similarity_comparison_figure(
                counts, similar_counts, base_name
            ),
            'umap': self._create_umap_projection_figure(
                query_embedding, base_name
            ),
            'distribution': self._create_distribution_figure(
                counts, base_name
            )
        }

        return figure_paths

    def _create_main_prediction_figure(self, query_embedding, counts, confidence,
                                     similar_images, distances, similar_counts,
                                     query_image, base_name):
        """Create the main prediction visualization figure"""
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(2, 3, width_ratios=[3, 1, 1], height_ratios=[1, 1])

        # Main UMAP plot
        ax1 = fig.add_subplot(gs[:, 0])
        self._plot_umap_with_query(ax1, query_embedding)

        # Confidence and prediction plot
        ax2 = fig.add_subplot(gs[0, 1:])
        self._plot_prediction_bars(ax2, counts, confidence)

        # Similar images plot
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_similar_images(ax3, similar_images, distances, query_image)

        # Similarity comparison plot
        ax4 = fig.add_subplot(gs[1, 2])
        if similar_counts:
            self._plot_similarity_comparison(ax4, counts, similar_counts)

        plt.tight_layout(pad=3.0)

        # Save figure
        output_path = os.path.join(self.output_dir, f"{base_name}_main.png")
        plt.savefig(output_path)
        plt.close()

        return output_path

    def _plot_umap_with_query(self, ax, query_embedding):
        """Plot UMAP projection with query point highlighted"""
        unique_labels = np.unique(self.predictor.training_labels)

        for i, label in enumerate(unique_labels):
            mask = (self.predictor.training_labels == label)
            if mask.any():
                ax.scatter(
                    self.predictor.umap_embeddings[mask, 0],
                    self.predictor.umap_embeddings[mask, 1],
                    color=self.predictor.type_palette[i],
                    alpha=0.4,
                    label=label,
                    s=20,
                    edgecolor=self.predictor.border_palette[i],
                    linewidth=0.5
                )

        query_pos = self.predictor.project_to_umap(query_embedding)
        ax.scatter(
            query_pos[0], query_pos[1],
            c='red', marker='X', s=300,
            label='Query Point',
            edgecolors='black',
            linewidth=1.5
        )

        ax.set_title('UMAP Projection with Query Point')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.grid(True, alpha=0.2)
        ax.legend()

    def _plot_prediction_bars(self, ax, counts, confidence):
        """Plot horizontal bar chart of top predictions with actual counts"""
        top_indices = np.argsort(counts)[-5:][::-1]
        top_labels = [self.predictor.cell_types[i] for i in top_indices]
        top_counts = [counts[i] for i in top_indices]

        # Create colormap for confidence
        cmap = plt.colormaps['RdYlGn']
        confidence_colors = [cmap(confidence) for _ in top_counts]

        bars = ax.barh(
            top_labels,
            top_counts,
            color=confidence_colors,
            edgecolor='black',
            linewidth=0.8
        )

        ax.set_title(f'Predicted Cell Type Counts\n(Confidence: {confidence:.2f})')
        ax.set_xlabel('Cell Count')
        ax.xaxis.grid(True, alpha=0.3)

        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.5,
                bar.get_y() + bar.get_height()/2,
                f'{int(width)}',
                ha='left',
                va='center'
            )

    def _plot_similar_images(self, ax, similar_images, distances, query_image=None):
        """Plot similar training images with distances"""
        ax.set_title('Most Similar Training Images')
        ax.axis('off')

        # Add query image if provided
        if query_image:
            try:
                img = Image.open(query_image).resize((150, 150)) if os.path.exists(query_image) else None
                if img:
                    im = OffsetImage(img, zoom=0.7)
                    ab = AnnotationBbox(
                        im,
                        (0.5, 0.9),
                        frameon=True,
                        xycoords='axes fraction',
                        bboxprops=dict(
                            edgecolor='red',
                            lw=2,
                            boxstyle="round,pad=0.3"
                        )
                    )
                    ax.add_artist(ab)
                    ax.text(
                        0.5, 0.85,
                        "Query Image",
                        ha='center',
                        va='top',
                        transform=ax.transAxes
                    )
            except Exception as e:
                print(f"Error loading query image: {e}")

        # Add similar training images
        for i, (img_path, dist) in enumerate(zip(similar_images, distances)):
            try:
                img = Image.open(img_path).resize((150, 150)) if os.path.exists(img_path) else None
                if img:
                    im = OffsetImage(img, zoom=0.7)
                    ab = AnnotationBbox(
                        im,
                        (0.2 + i*0.2, 0.5),
                        frameon=True,
                        xycoords='axes fraction',
                        bboxprops=dict(
                            edgecolor='blue',
                            lw=2,
                            boxstyle="round,pad=0.3"
                        )
                    )
                    ax.add_artist(ab)
                    ax.text(
                        0.2 + i*0.2, 0.35,
                        f"Dist: {dist:.2f}",
                        ha='center',
                        va='top',
                        transform=ax.transAxes
                    )
            except Exception as e:
                print(f"Error loading neighbor image: {e}")

    def _plot_similarity_comparison(self, ax, predicted_counts, similar_counts):
        """Plot comparison between predicted and similar training samples"""
        if not similar_counts:
            return

        mean_similar = np.mean(similar_counts, axis=0)

        for i, (pred, sim) in enumerate(zip(predicted_counts, mean_similar)):
            ax.plot(
                [0, 1],
                [pred, sim],
                marker='o',
                color=self.predictor.type_palette[i],
                label=self.predictor.cell_types[i] if i < 5 else None
            )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Predicted', 'Similar Avg'])
        ax.set_ylabel('Cell Count')
        ax.set_title('Predicted vs Similar Samples')
        ax.grid(True, alpha=0.2)
        ax.legend()

    def _create_similarity_comparison_figure(self, predicted_counts, similar_counts, base_name):
        """Create dedicated similarity comparison plot"""
        if not similar_counts:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        mean_similar = np.mean(similar_counts, axis=0)

        width = 0.35
        top_indices = np.argsort(predicted_counts)[-10:][::-1]
        x = np.arange(len(top_indices))

        ax.bar(
            x - width/2,
            predicted_counts[top_indices],
            width,
            label='Predicted',
            color='#1f77b4',
            edgecolor='black'
        )

        ax.bar(
            x + width/2,
            mean_similar[top_indices],
            width,
            label='Similar Training Avg',
            color='#ff7f0e',
            edgecolor='black'
        )

        ax.set_xticks(x)
        ax.set_xticklabels([self.predictor.cell_types[i] for i in top_indices],
                          rotation=45, ha='right')
        ax.set_ylabel('Cell Count')
        ax.set_title('Predicted vs Similar Training Samples')
        ax.legend()
        ax.grid(True, alpha=0.2)

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f"{base_name}_similarity.png")
        plt.savefig(output_path)
        plt.close()

        return output_path

    def _create_umap_projection_figure(self, query_embedding, base_name):
        """Create UMAP projection plot with query point highlighted"""
        fig, ax = plt.subplots(figsize=(12, 10))

        unique_labels = np.unique(self.predictor.training_labels)
        for i, label in enumerate(unique_labels):
            mask = (self.predictor.training_labels == label)
            if mask.any():
                ax.scatter(
                    self.predictor.umap_embeddings[mask, 0],
                    self.predictor.umap_embeddings[mask, 1],
                    color=self.predictor.type_palette[i],
                    alpha=0.4,
                    label=label,
                    s=20,
                    edgecolor=self.predictor.border_palette[i],
                    linewidth=0.5
                )

        query_pos = self.predictor.project_to_umap(query_embedding)
        ax.scatter(
            query_pos[0], query_pos[1],
            c='red', marker='X', s=300,
            label='Query Point',
            edgecolors='black',
            linewidth=1.5
        )

        ax.set_title('UMAP Projection with Query Point')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.grid(True, alpha=0.2)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f"{base_name}_umap.png")
        plt.savefig(output_path)
        plt.close()

        return output_path

    def _create_distribution_figure(self, counts, base_name):
        """Create distribution plot of all predictions"""
        fig, ax = plt.subplots(figsize=(12, 8))

        sorted_indices = np.argsort(counts)[::-1]
        sorted_counts = counts[sorted_indices]
        sorted_labels = [self.predictor.cell_types[i] for i in sorted_indices]

        colors = [self.predictor.type_palette[i] for i in sorted_indices]

        bars = ax.bar(
            range(len(sorted_counts)),
            sorted_counts,
            color=colors,
            edgecolor='black'
        )

        ax.set_xticks(range(len(sorted_counts)))
        ax.set_xticklabels(sorted_labels, rotation=90)
        ax.set_ylabel('Cell Count')
        ax.set_title('Predicted Cell Type Distribution')
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f"{base_name}_distribution.png")
        plt.savefig(output_path)
        plt.close()

        return output_path