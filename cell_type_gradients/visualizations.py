import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.spatial import ConvexHull
from .utils import save_figure, save_excel

def plot_distance_matrix(visualizer):
    try:
        print("\n=== Starting Distance Matrix Save ===")
        fig_path = os.path.abspath(os.path.join(visualizer.output_dir, 'cell_type_distance_matrix.png'))
        excel_path = os.path.abspath(os.path.join(visualizer.output_dir, 'cell_type_distance_matrix.xlsx'))

        plt.figure(figsize=(16, 14))
        ax = plt.gca()
        mask = np.triu(np.ones_like(visualizer.normalized_distances, dtype=bool))
        cell_types = visualizer.distance_matrix.columns.tolist()
        reversed_cell_types = cell_types[::-1]
        reversed_distances = visualizer.normalized_distances[::-1, :]
        reversed_distances = reversed_distances[:, ::-1]

        heatmap = sns.heatmap(
            reversed_distances,
            cmap='viridis_r',
            annot=True,
            fmt=".2f",
            mask=mask,
            square=True,
            ax=ax,
            cbar_kws={'label': 'Normalized Distance'}
        )

        ax.set_xticks(np.arange(len(reversed_cell_types)) + 0.5)
        ax.set_xticklabels(
            reversed_cell_types,
            rotation=45,
            ha='right',
            rotation_mode='anchor',
            fontsize=10
        )

        ax.set_yticks(np.arange(len(cell_types)) + 0.5)
        ax.set_yticklabels(
            cell_types,
            rotation=0,
            ha='right',
            va='center',
            fontsize=10
        )

        plt.tight_layout()

        for i, ct1 in enumerate(cell_types):
            for j, ct2 in enumerate(reversed_cell_types):
                if i > j:
                    cat1 = visualizer.type_to_category[ct1]
                    cat2 = visualizer.type_to_category[ct2]
                    if cat1 != cat2:
                        ax.add_patch(
                            plt.Rectangle((j, i), 1, 1, fill=False, 
                                        edgecolor='white', lw=1, clip_on=False)
                        )

        ax.set_title('Pairwise Cell Type Distances in UMAP Space', fontsize=14, pad=20)
        save_figure(plt.gcf(), visualizer.output_dir, 'cell_type_distance_matrix.png', dpi=300)

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            visualizer.distance_matrix.to_excel(writer, sheet_name='Distance Matrix')
            pd.DataFrame(visualizer.normalized_distances,
                       index=cell_types,
                       columns=cell_types).to_excel(writer, sheet_name='Normalized')

    except Exception as e:
        print(f"\n!!! FAILED TO SAVE DISTANCE MATRIX !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise

def plot_brain_region_atlas(visualizer):
    try:
        print("\nGenerating brain region atlas...")
        region_clusters = KMeans(n_clusters=9, random_state=42).fit_predict(visualizer.embeddings)
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle('Brain Region Atlas with Cell Type Composition', fontsize=16, y=1.02)
        
        for i, ax in enumerate(axes.flat):
            cluster_images = np.where(region_clusters == i)[0]
            if len(cluster_images) > 0:
                distances = np.linalg.norm(visualizer.embeddings[cluster_images] - 
                                          np.mean(visualizer.embeddings[cluster_images], axis=0), axis=1)
                rep_img_idx = cluster_images[np.argmin(distances)]
                img = visualizer.data_processor.load_image(visualizer.image_paths[rep_img_idx])
                ax.imshow(img)
                
                pie_ax = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
                cell_counts = pd.Series(visualizer.labels[cluster_images]).value_counts()
                pie_colors = [visualizer.type_palette[visualizer.cell_types.index(ct)] 
                            for ct in cell_counts.index]
                
                if len(cell_counts) > 5:
                    top_counts = cell_counts.nlargest(5)
                    other = cell_counts.sum() - top_counts.sum()
                    top_counts['Other'] = other
                    pie_colors = pie_colors[:5] + [(0.8, 0.8, 0.8)]
                    wedges, texts = pie_ax.pie(top_counts, colors=pie_colors, startangle=90)
                    pie_ax.legend(wedges, top_counts.index, 
                                bbox_to_anchor=(1.5, 1), 
                                loc='upper right', 
                                fontsize=6)
                else:
                    wedges, texts = pie_ax.pie(cell_counts, colors=pie_colors, startangle=90)
                    pie_ax.legend(wedges, cell_counts.index, 
                                bbox_to_anchor=(1.5, 1), 
                                loc='upper right', 
                                fontsize=6)
                
                ax.set_title(f"Region Cluster {i+1}\n{len(cluster_images)} samples", fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        save_figure(fig, visualizer.output_dir, 'brain_region_atlas.png')
        
    except Exception as e:
        print(f"Error generating brain region atlas: {str(e)}")
        raise

def plot_thumbnail_overview(visualizer):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.scatter(
        visualizer.umap_emb[:, 0],
        visualizer.umap_emb[:, 1],
        color=visualizer.background_color,
        s=5,
        alpha=visualizer.background_alpha
    )
    
    samples_per_type = 3
    thumbnail_size = 50
    
    for i, ct in enumerate(visualizer.cell_types):
        mask = visualizer.labels == ct
        if sum(mask) > 0:
            ax.scatter(
                visualizer.umap_emb[mask, 0],
                visualizer.umap_emb[mask, 1],
                color=visualizer.type_palette[i],
                s=10,
                alpha=0.6,
                label=ct
            )
            
            type_indices = np.where(mask)[0]
            confidences = visualizer.confidences[mask]
            top_indices = type_indices[np.argsort(confidences)[-samples_per_type:]]
            
            for idx in top_indices:
                try:
                    img = visualizer.data_processor.load_image(visualizer.image_paths[idx]).resize((thumbnail_size, thumbnail_size))
                    im = OffsetImage(img, zoom=0.7)
                    ab = AnnotationBbox(
                        im,
                        (visualizer.umap_emb[idx, 0], visualizer.umap_emb[idx, 1]),
                        frameon=True,
                        bboxprops=dict(
                            edgecolor=visualizer.type_palette[i],
                            linewidth=2,
                            boxstyle="round,pad=0.2",
                            alpha=0.9
                        )
                    )
                    ax.add_artist(ab)
                except Exception as e:
                    print(f"Error adding thumbnail for {ct}: {str(e)}")
    
    ax.set_xlim(visualizer.global_x_min, visualizer.global_x_max)
    ax.set_ylim(visualizer.global_y_min, visualizer.global_y_max)
    ax.set_title("Representative Thumbnails Across All Cell Types", fontsize=16)
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles, labels,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        fontsize=10,
        title='Cell Types',
        title_fontsize=12
    )
    
    plt.tight_layout()
    save_figure(fig, visualizer.output_dir, 'all_cell_types_thumbnails_overview.png')

def plot_cell_type_gradients(visualizer):
    all_types_fig = plt.figure(figsize=(12, 8))
    ax_all = all_types_fig.add_subplot(111)
    
    for i, ct in enumerate(visualizer.cell_types):
        mask = visualizer.labels == ct
        if sum(mask) > 0:
            ax_all.scatter(
                visualizer.umap_emb[mask, 0],
                visualizer.umap_emb[mask, 1],
                color=visualizer.type_palette[i],
                s=10,
                alpha=0.5,
                label=ct
            )
    
    ax_all.set_xlim(visualizer.global_x_min, visualizer.global_x_max)
    ax_all.set_ylim(visualizer.global_y_min, visualizer.global_y_max)
    ax_all.set_title("All Cell Types in UMAP Space", fontsize=14)
    ax_all.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax_all.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax_all.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_figure(all_types_fig, visualizer.output_dir, 'all_cell_types_colored.png')
    
    for i, ct in enumerate(visualizer.cell_types):
        mask = visualizer.labels == ct
        if sum(mask) > 5:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            
            for j, other_ct in enumerate(visualizer.cell_types):
                if other_ct != ct:
                    other_mask = visualizer.labels == other_ct
                    if sum(other_mask) > 0:
                        ax.scatter(
                            visualizer.umap_emb[other_mask, 0],
                            visualizer.umap_emb[other_mask, 1],
                            color=visualizer.type_palette[j],
                            s=10,
                            alpha=visualizer.background_alpha,
                            label=other_ct if j % 5 == 0 else ""
                        )
            
            ax.scatter(
                visualizer.umap_emb[mask, 0],
                visualizer.umap_emb[mask, 1],
                color=visualizer.type_palette[i],
                s=20,
                alpha=0.8,
                label=ct,
                edgecolors='w',
                linewidth=0.5
            )
            
            sns.kdeplot(
                x=visualizer.umap_emb[mask, 0],
                y=visualizer.umap_emb[mask, 1],
                cmap=visualizer.concentration_palette,
                ax=ax,
                fill=True,
                alpha=0.5,
                thresh=0.1,
                levels=10,
                label=f'{ct} density'
            )
            
            sample_indices = np.random.choice(
                np.where(mask)[0],
                size=min(5, sum(mask)),
                replace=False
            )
            
            for idx in sample_indices:
                img = visualizer.data_processor.load_image(visualizer.image_paths[idx]).resize((50, 50))
                im = OffsetImage(img, zoom=0.5)
                ab = AnnotationBbox(
                    im,
                    (visualizer.umap_emb[idx, 0], visualizer.umap_emb[idx, 1]),
                    frameon=True,
                    bboxprops=dict(
                        edgecolor='white',
                        boxstyle="round,pad=0.2",
                        alpha=0.8
                    )
                )
                ax.add_artist(ab)
            
            if sum(mask) > 10:
                kmeans = KMeans(n_clusters=min(3, sum(mask)//5), random_state=42)
                clusters = kmeans.fit_predict(visualizer.umap_emb[mask])
                
                for cluster_id in np.unique(clusters):
                    cluster_points = visualizer.umap_emb[mask][clusters == cluster_id]
                    if len(cluster_points) > 2:
                        hull = ConvexHull(cluster_points)
                        ax.fill(
                            cluster_points[hull.vertices, 0],
                            cluster_points[hull.vertices, 1],
                            alpha=0.1,
                            label=f'Sub-cluster {cluster_id+1}'
                        )
            
            ax.set_xlim(visualizer.global_x_min, visualizer.global_x_max)
            ax.set_ylim(visualizer.global_y_min, visualizer.global_y_max)
            
            ax.set_title(
                f"{ct} Spatial Distribution\n({sum(mask)} samples)",
                fontsize=14,
                pad=20
            )
            ax.set_xlabel('UMAP Dimension 1', fontsize=12)
            ax.set_ylabel('UMAP Dimension 2', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            safe_ct_name = ct.replace(".", "_")
            save_figure(fig, visualizer.output_dir, f'cell_type_gradient_{safe_ct_name}.png')