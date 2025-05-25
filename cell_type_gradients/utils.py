import os
import matplotlib.pyplot as plt

def save_figure(fig, output_dir, filename, dpi=300, bbox_inches='tight'):
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    print(f"Saved figure: {path}")

def save_excel(df, output_dir, filename):
    path = os.path.join(output_dir, filename)
    df.to_excel(path)
    print(f"Saved Excel file: {path}")