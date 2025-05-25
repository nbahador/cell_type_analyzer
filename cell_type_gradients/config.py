import seaborn as sns

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

TYPE_TO_CATEGORY = {
    'ABC.NN': 'Non-neuronal',
    'Astro.TE.NN': 'Non-neuronal',
    'CLA.EPd.CTX.Car3.Glut': 'Glutamatergic',
    'Endo.NN': 'Non-neuronal',
    'L2.3.IT.CTX.Glut': 'Glutamatergic',
    'L4.5.IT.CTX.Glut': 'Glutamatergic',
    'L5.ET.CTX.Glut': 'Glutamatergic',
    'L5.IT.CTX.Glut': 'Glutamatergic',
    'L5.NP.CTX.Glut': 'Glutamatergic',
    'L6.CT.CTX.Glut': 'Glutamatergic',
    'L6.IT.CTX.Glut': 'Glutamatergic',
    'L6b.CTX.Glut': 'Glutamatergic',
    'Lamp5.Gaba': 'GABAergic',
    'Lamp5.Lhx6.Gaba': 'GABAergic',
    'Lymphoid.NN': 'Non-neuronal',
    'Microglia.NN': 'Non-neuronal',
    'OPC.NN': 'Non-neuronal',
    'Oligo.NN': 'Non-neuronal',
    'Peri.NN': 'Non-neuronal',
    'Pvalb.Gaba': 'GABAergic',
    'Pvalb.chandelier.Gaba': 'GABAergic',
    'SMC.NN': 'Non-neuronal',
    'Sncg.Gaba': 'GABAergic',
    'Sst.Chodl.Gaba': 'GABAergic',
    'Sst.Gaba': 'GABAergic',
    'VLMC.NN': 'Non-neuronal',
    'Vip.Gaba': 'GABAergic'
}

TYPE_PALETTE = sns.color_palette("husl", len(CELL_TYPES))
CONCENTRATION_PALETTE = sns.light_palette("navy", as_cmap=True)
BACKGROUND_COLOR = (0.9, 0.9, 0.9)
BACKGROUND_ALPHA = 0.2