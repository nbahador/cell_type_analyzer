import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QTabWidget, QFrame, QScrollArea, 
                            QFileDialog, QMessageBox, QComboBox, QGridLayout, 
                            QGroupBox, QProgressBar, QSizePolicy)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QPixmap, QColor, QFont, QPalette, QImage  # Added QImage here
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import tempfile
import zipfile
import shutil
import torch
import datetime
from cell_type_analyzer import load_saved_predictor
from visualizations import PredictionVisualizer

class CellTypeAnalyzerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cell Type Analyzer")
        self.setMinimumSize(1200, 800)
        
        # Initialize variables
        self.model_path = os.path.join("trained_models", "cell_type_predictor.pkl")
        self.output_dir = "results"
        self.current_image = None
        self.predictor = None
        self.features = None
        
        # Setup main window
        self.init_ui()
        
        # Load model if available
        self.load_model()
        
    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Left panel (controls)
        left_panel = QFrame()
        left_panel.setFixedWidth(380)
        left_panel.setStyleSheet("""
            QFrame {
                background-color: #1e293b;
                border-radius: 12px;
                border: 1px solid #475569;
            }
        """)
        
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.setSpacing(15)
        
        # Right panel (visualizations)
        right_panel = QFrame()
        right_panel.setStyleSheet("""
            QFrame {
                background-color: #0f172a;
                border-radius: 12px;
                border: 1px solid #475569;
            }
        """)
        
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 15, 15, 15)
        right_layout.setSpacing(15)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        # Build left panel (controls)
        self.build_left_panel(left_layout)
        
        # Build right panel (visualizations)
        self.build_right_panel(right_layout)
        
        # Apply dark theme
        self.apply_dark_theme()
        
    def apply_dark_theme(self):
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor('#0a0e1a'))
        palette.setColor(QPalette.WindowText, QColor('#e2e8f0'))
        palette.setColor(QPalette.Base, QColor('#1e293b'))
        palette.setColor(QPalette.AlternateBase, QColor('#334155'))
        palette.setColor(QPalette.ToolTipBase, QColor('#f8fafc'))
        palette.setColor(QPalette.ToolTipText, QColor('#f8fafc'))
        palette.setColor(QPalette.Text, QColor('#e2e8f0'))
        palette.setColor(QPalette.Button, QColor('#334155'))
        palette.setColor(QPalette.ButtonText, QColor('#e2e8f0'))
        palette.setColor(QPalette.BrightText, QColor('#10b981'))
        palette.setColor(QPalette.Highlight, QColor('#3b82f6'))
        palette.setColor(QPalette.HighlightedText, QColor('#f8fafc'))
        self.setPalette(palette)
        
    def build_left_panel(self, layout):
        # Title
        title = QLabel("Cell Type Analyzer")
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #f8fafc;
                margin-bottom: 20px;
            }
        """)
        layout.addWidget(title)
        
        # Model status
        self.model_status = QLabel("Model: Not Loaded")
        self.model_status.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #94a3b8;
                margin-bottom: 15px;
            }
        """)
        layout.addWidget(self.model_status)
        
        # Upload section
        upload_group = QGroupBox("Image Input")
        upload_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                color: #e2e8f0;
                border: 1px solid #475569;
                border-radius: 8px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        upload_layout = QVBoxLayout(upload_group)
        upload_layout.setSpacing(10)
        
        # URL input
        url_label = QLabel("Image URL:")
        url_label.setStyleSheet("color: #e2e8f0;")
        
        self.url_input = QComboBox()
        self.url_input.setEditable(True)
        self.url_input.addItem("https://s3.us-west-2.amazonaws.com/allen-genetic-tools/bio_file_finder/Thumbnails/STPT/640002_STPT_Thumbnail.png")
        self.url_input.setStyleSheet("""
            QComboBox {
                background-color: #334155;
                color: #e2e8f0;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        
        url_button = QPushButton("Load from URL")
        url_button.setStyleSheet(self.get_button_style())
        url_button.clicked.connect(self.load_from_url)
        
        # File upload
        file_button = QPushButton("Load from File")
        file_button.setStyleSheet(self.get_button_style())
        file_button.clicked.connect(self.load_from_file)
        
        upload_layout.addWidget(url_label)
        upload_layout.addWidget(self.url_input)
        upload_layout.addWidget(url_button)
        upload_layout.addWidget(file_button)
        
        layout.addWidget(upload_group)
        
        # Image preview
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setStyleSheet("""
            QLabel {
                background-color: #0f172a;
                border: 1px dashed #475569;
                border-radius: 8px;
                min-height: 150px;
            }
        """)
        layout.addWidget(self.image_preview)
        
        # Analysis button
        self.analyze_button = QPushButton("Analyze Image")
        self.analyze_button.setStyleSheet(self.get_button_style(primary=True))
        self.analyze_button.clicked.connect(self.analyze_image)
        self.analyze_button.setEnabled(False)
        layout.addWidget(self.analyze_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #475569;
                border-radius: 4px;
                text-align: center;
                background-color: #334155;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
            }
        """)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status indicator
        self.status_indicator = QLabel()
        self.status_indicator.setAlignment(Qt.AlignCenter)
        self.status_indicator.setStyleSheet("""
            QLabel {
                font-size: 14px;
                padding: 8px;
                border-radius: 8px;
                background-color: #334155;
            }
        """)
        layout.addWidget(self.status_indicator)
        
        # Add stretch to push everything up
        layout.addStretch()
        
    def build_right_panel(self, layout):
        # Tab widget for visualizations
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #475569;
                border-radius: 8px;
                background: #0f172a;
            }
            QTabBar::tab {
                background: #1e293b;
                color: #94a3b8;
                padding: 8px 16px;
                border: 1px solid #475569;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #3b82f6;
                color: #f8fafc;
                border-color: #1d4ed8;
            }
            QTabBar::tab:hover {
                background: #334155;
            }
        """)
        
        # Prediction tab
        self.prediction_tab = QWidget()
        self.prediction_layout = QVBoxLayout(self.prediction_tab)
        self.prediction_canvas = None
        self.tab_widget.addTab(self.prediction_tab, "Prediction")
        
        # UMAP tab
        self.umap_tab = QWidget()
        self.umap_layout = QVBoxLayout(self.umap_tab)
        self.umap_canvas = None
        self.tab_widget.addTab(self.umap_tab, "UMAP")
        
        # Similarity tab
        self.similarity_tab = QWidget()
        self.similarity_layout = QVBoxLayout(self.similarity_tab)
        self.similarity_canvas = None
        self.tab_widget.addTab(self.similarity_tab, "Similarity")
        
        # Distribution tab
        self.distribution_tab = QWidget()
        self.distribution_layout = QVBoxLayout(self.distribution_tab)
        self.distribution_canvas = None
        self.tab_widget.addTab(self.distribution_tab, "Distribution")
        
        layout.addWidget(self.tab_widget)
        
    def get_button_style(self, primary=False):
        if primary:
            return """
                QPushButton {
                    background-color: #3b82f6;
                    color: #f8fafc;
                    border: none;
                    border-radius: 6px;
                    padding: 10px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #2563eb;
                }
                QPushButton:pressed {
                    background-color: #1d4ed8;
                }
                QPushButton:disabled {
                    background-color: #334155;
                    color: #94a3b8;
                }
            """
        else:
            return """
                QPushButton {
                    background-color: #334155;
                    color: #e2e8f0;
                    border: 1px solid #475569;
                    border-radius: 6px;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #475569;
                }
                QPushButton:pressed {
                    background-color: #1e293b;
                }
                QPushButton:disabled {
                    background-color: #1e293b;
                    color: #94a3b8;
                }
            """
    
    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.predictor = load_saved_predictor(self.model_path)
                self.model_status.setText("Model: Loaded")
                self.model_status.setStyleSheet("color: #10b981;")
            else:
                self.model_status.setText("Model: Not Found")
                self.model_status.setStyleSheet("color: #ef4444;")
        except Exception as e:
            self.model_status.setText(f"Model: Error ({str(e)})")
            self.model_status.setStyleSheet("color: #ef4444;")
    
    def load_from_url(self):
        url = self.url_input.currentText().strip()
        if not url:
            self.show_error("Please enter a valid URL")
            return
            
        self.set_status("Downloading image...", "working")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Use a timer to allow UI to update
        QTimer.singleShot(100, lambda: self._download_image(url))
    
    def _download_image(self, url):
        try:
            self.current_image = self.download_and_process_image(url)
            
            # Display preview
            pixmap = QPixmap()
            if isinstance(self.current_image, Image.Image):
                # Convert PIL Image to QPixmap
                img = self.current_image.convert("RGBA")
                data = img.tobytes("raw", "RGBA")
                qimage = QImage(data, img.size[0], img.size[1], QImage.Format_RGBA8888)
                pixmap = QPixmap.fromImage(qimage)
            else:
                # Handle other image types if needed
                pass
                
            self.image_preview.setPixmap(pixmap.scaled(
                self.image_preview.width(), 
                self.image_preview.height(),
                Qt.KeepAspectRatio
            ))
            
            self.analyze_button.setEnabled(True)
            self.set_status("Image loaded successfully", "success")
            
        except Exception as e:
            self.show_error(f"Error loading image: {str(e)}")
            self.set_status("Image load failed", "error")
            
        finally:
            self.progress_bar.setVisible(False)
    
    def load_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", 
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.nii *.nii.gz)"
        )
        
        if not file_path:
            return
            
        self.set_status("Loading image...", "working")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Use a timer to allow UI to update
        QTimer.singleShot(100, lambda: self._load_file(file_path))
    
    def _load_file(self, file_path):
        try:
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                # Handle 2D images
                self.current_image = Image.open(file_path).convert('RGB')
                
                # Display preview
                img = self.current_image.convert("RGBA")
                data = img.tobytes("raw", "RGBA")
                qimage = QImage(data, img.size[0], img.size[1], QImage.Format_RGBA8888)
                pixmap = QPixmap.fromImage(qimage)
                
                self.image_preview.setPixmap(pixmap.scaled(
                    self.image_preview.width(), 
                    self.image_preview.height(),
                    Qt.KeepAspectRatio
                ))
                
            elif file_path.lower().endswith(('.nii', '.nii.gz')):
                # Handle 3D volumes
                volume = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
                
                # Create standardized multi-view
                views = [
                    volume[volume.shape[0]//2, :, :].T,  # Sagittal
                    volume[:, volume.shape[1]//2, :].T,  # Coronal
                    volume[:, :, volume.shape[2]//2].T   # Axial
                ]
                
                target_size = (224, 224)
                views = [Image.fromarray(v).resize(target_size) for v in views]
                combined = np.hstack([np.array(v) for v in views])
                self.current_image = Image.fromarray(combined)
                
                # Display preview
                img = self.current_image.convert("RGBA")
                data = img.tobytes("raw", "RGBA")
                qimage = QImage(data, img.size[0], img.size[1], QImage.Format_RGBA8888)
                pixmap = QPixmap.fromImage(qimage)
                
                self.image_preview.setPixmap(pixmap.scaled(
                    self.image_preview.width(), 
                    self.image_preview.height(),
                    Qt.KeepAspectRatio
                ))
                
            self.analyze_button.setEnabled(True)
            self.set_status("Image loaded successfully", "success")
            
        except Exception as e:
            self.show_error(f"Error loading image: {str(e)}")
            self.set_status("Image load failed", "error")
            
        finally:
            self.progress_bar.setVisible(False)
    
    def download_and_process_image(self, url):
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
    
    def analyze_image(self):
        if not self.current_image or not self.predictor:
            self.show_error("No image loaded or model not available")
            return
            
        self.set_status("Analyzing image...", "working")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Use a timer to allow UI to update
        QTimer.singleShot(100, self._perform_analysis)
    
    def _perform_analysis(self):
        try:
            # Process image and extract features
            inputs = self.predictor.processor(images=self.current_image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k:v.cuda() for k,v in inputs.items()}
                self.predictor.model = self.predictor.model.cuda()

            with torch.no_grad():
                outputs = self.predictor.model(**inputs)

            self.features = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

            # Make predictions
            counts = self.predictor.predict_counts(self.features)
            similar_images, distances, similar_counts = self.predictor.find_similar_training_images(self.features)
            confidence = self.predictor.calculate_confidence(counts, similar_counts)

            # Generate visualizations
            visualizer = PredictionVisualizer(self.predictor, self.output_dir)
            figure_paths = visualizer.visualize_prediction(self.features, "current_image")

            # Update UI with results
            self.update_results(counts, confidence, figure_paths)
            self.set_status("Analysis complete", "success")
            
        except Exception as e:
            self.show_error(f"Error during analysis: {str(e)}")
            self.set_status("Analysis failed", "error")
            
        finally:
            self.progress_bar.setVisible(False)
    
    def update_results(self, counts, confidence, figure_paths):
        # Update prediction tab
        self.update_prediction_tab(counts, confidence, figure_paths.get('main'))
        
        # Update UMAP tab
        self.update_umap_tab(figure_paths.get('umap'))
        
        # Update similarity tab
        self.update_similarity_tab(figure_paths.get('similarity'))
        
        # Update distribution tab
        self.update_distribution_tab(figure_paths.get('distribution'))
    
    def update_prediction_tab(self, counts, confidence, figure_path):
        # Clear previous content
        for i in reversed(range(self.prediction_layout.count())): 
            self.prediction_layout.itemAt(i).widget().setParent(None)
        
        # Create a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)
    
        # Create a container widget
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(15)
    
        # Add confidence label with improved styling
        confidence_label = QLabel(f"Confidence: {confidence:.2f}/1.0")
        confidence_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #e2e8f0;
                padding: 8px;
                background-color: #334155;
                border-radius: 8px;
            }
        """)
        container_layout.addWidget(confidence_label)
    
        # Add top predictions
        top_indices = np.argsort(counts)[-5:][::-1]
    
        for i, idx in enumerate(top_indices):
            cell_type = self.predictor.cell_types[idx]
            percentage = counts[idx]
        
            group = QGroupBox(f"{i+1}. {cell_type}")
            group.setStyleSheet("""
                QGroupBox {
                    font-size: 16px;
                    color: #e2e8f0;  /* Text color for group title */
                    border: 1px solid #475569;
                    border-radius: 8px;
                    margin-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                    color: #333333;  /* Ensure title text is visible e2e8f0 */  
                }
            """)
        
            group_layout = QVBoxLayout(group)
        
            # Progress bar for percentage with improved styling
            progress = QProgressBar()
            progress.setValue(int(percentage))
            progress.setFormat(f"{percentage:.1f}%")
            progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #475569;
                    border-radius: 4px;
                    text-align: center;
                    background-color: #334155;
                    height: 25px;
                    color: #e2e8f0;  /* Text color inside progress bar */
                }
                QProgressBar::chunk {
                    background-color: #3b82f6;
                }
            """)
            group_layout.addWidget(progress)
        
            container_layout.addWidget(group)
    
        # Add the main figure if available
        if figure_path and os.path.exists(figure_path):
            fig = plt.figure(facecolor='white')  # Set white background for the figure
            img = plt.imread(figure_path)
            plt.imshow(img)
            plt.axis('off')
        
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background-color: white;")  # Ensure canvas has white background
            container_layout.addWidget(canvas)
    
        # Set the container as the scroll area's widget
        scroll.setWidget(container)
        self.prediction_layout.addWidget(scroll)
    
    def update_umap_tab(self, figure_path):
        # Clear previous content
        for i in reversed(range(self.umap_layout.count())): 
            self.umap_layout.itemAt(i).widget().setParent(None)
            
        if figure_path and os.path.exists(figure_path):
            fig = plt.figure()
            img = plt.imread(figure_path)
            plt.imshow(img)
            plt.axis('off')
            
            canvas = FigureCanvas(fig)
            self.umap_layout.addWidget(canvas)
    
    def update_similarity_tab(self, figure_path):
        # Clear previous content
        for i in reversed(range(self.similarity_layout.count())): 
            self.similarity_layout.itemAt(i).widget().setParent(None)
            
        if figure_path and os.path.exists(figure_path):
            fig = plt.figure()
            img = plt.imread(figure_path)
            plt.imshow(img)
            plt.axis('off')
            
            canvas = FigureCanvas(fig)
            self.similarity_layout.addWidget(canvas)
    
    def update_distribution_tab(self, figure_path):
        # Clear previous content
        for i in reversed(range(self.distribution_layout.count())): 
            self.distribution_layout.itemAt(i).widget().setParent(None)
            
        if figure_path and os.path.exists(figure_path):
            fig = plt.figure()
            img = plt.imread(figure_path)
            plt.imshow(img)
            plt.axis('off')
            
            canvas = FigureCanvas(fig)
            self.distribution_layout.addWidget(canvas)
    
    def set_status(self, message, status_type="info"):
        self.status_indicator.setText(message)
        
        if status_type == "success":
            self.status_indicator.setStyleSheet("""
                QLabel {
                    color: #10b981;
                    background-color: #334155;
                    border: 1px solid #10b981;
                }
            """)
        elif status_type == "error":
            self.status_indicator.setStyleSheet("""
                QLabel {
                    color: #ef4444;
                    background-color: #334155;
                    border: 1px solid #ef4444;
                }
            """)
        elif status_type == "working":
            self.status_indicator.setStyleSheet("""
                QLabel {
                    color: #3b82f6;
                    background-color: #334155;
                    border: 1px solid #3b82f6;
                }
            """)
        else:  # info
            self.status_indicator.setStyleSheet("""
                QLabel {
                    color: #e2e8f0;
                    background-color: #334155;
                    border: 1px solid #475569;
                }
            """)
    
    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Initialize torch
    if not torch.cuda.is_available():
        print("Note: CUDA not available, using CPU")
    
    window = CellTypeAnalyzerUI()
    window.show()
    sys.exit(app.exec_())