import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors as mcolors  # Add this import
import warnings

# Set global style parameters with white background and larger fonts
plt.style.use('default')  # Reset to default style to ensure white background
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.dpi': 300,
    'figure.autolayout': True,
    'savefig.bbox': 'tight'
})

# Try to load a nicer font if available
try:
    mpl.font_manager.fontManager.addfont('Arial.ttf')
    plt.rcParams['font.family'] = 'Arial'
except:
    pass

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Default cell types
CELL_TYPES = [
    'ABC.NN', 'Astro.TE.NN', 'CLA.EPd.CTX.Car3.Glut', 'Endo.NN',
    'L2.3.IT.CTX.Glut', 'L4.5.IT.CTX.Glut', 'L5.ET.CTX.Glut',
    'L5.IT.CTX.Glut', 'L5.NP.CTX.Glut', 'L6.CT.CTX.Glut',
    'L6.IT.CTX.Glut', 'L6b.CTX.Glut', 'Lamp5.Gaba',
    'Lamp5.Lhx6.Gaba', 'Lymphoid.NN', 'Microglia.NN',
    'OPC.NN', 'Oligo.NN', 'Peri.NN', 'Pvalb.Gaba',
    'Pvalb.chandelier.Gaba', 'SMC.NN', 'Sncg.Gaba',
    'Sst.Chodl.Gaba', 'Sst.Gaba', 'VLMC.NN', 'Vip.Gaba'
]

def setup_colors(cell_types):
    """Setup color palettes for visualization"""
    # Use a more visually distinct colormap
    cmap = plt.cm.get_cmap('tab20', len(cell_types))
    type_palette = [mcolors.rgb2hex(cmap(i)) for i in range(len(cell_types))]
    border_palette = []
    for color in type_palette:
        rgb = mcolors.hex2color(color)
        darker = [max(0, c * 0.6) for c in rgb]  # Darker borders for better contrast
        border_palette.append(mcolors.rgb2hex(darker))
    return type_palette, border_palette