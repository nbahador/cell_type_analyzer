from cell_type_gradients import CellTypeGradientVisualizer

if __name__ == "__main__":
    try:
        # Configuration - update these paths as needed
        excel_path = "assets/MapMySections_EntrantData.xlsx"
        sheet_name = "Training Set"
        image_root_dir = "assets/sample_images"
        
        # Initialize visualizer
        visualizer = CellTypeGradientVisualizer(
            excel_path=excel_path,
            sheet_name=sheet_name,
            image_root_dir=image_root_dir
        )
        
        # Run all visualizations
        visualizer.run_all_visualizations()
        
        print("All plots and data files saved successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise