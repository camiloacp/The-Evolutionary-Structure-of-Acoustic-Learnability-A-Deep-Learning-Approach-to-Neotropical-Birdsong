"""
Estilo visual estandarizado para las figuras del proyecto birds-sounds.

Replica la paleta y los rcParams usados en los notebooks 11/18/19 para que las
figuras nuevas (comparación ResNet152V2 vs BirdNET) tengan la misma apariencia
de publicación. Uso típico en un notebook:

    import sys; sys.path.append("../src")
    import viz_style as vs
    vs.apply_style()
    ...
    vs.savefig_dual(fig, "birdnet_vs_resnet_metrics")
"""
from __future__ import annotations

import os
import matplotlib.pyplot as plt

# rcParams estándar (idénticos a los notebooks 18/19/11)
RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "legend.fancybox": False,
    "legend.edgecolor": "gray",
    "legend.framealpha": 0.9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 300,
}

# Colores por modelo: el modelo propio (ResNet) en azul primario del proyecto;
# BirdNET (comparación) en el naranja secundario.
MODEL_COLORS = {
    "EfficientNetV2L": "#0173B2",
    "ResNet152V2": "#0173B2",
    "BirdNET_V2.4": "#DE8F05",
    "resnet": "#0173B2",
    "birdnet": "#DE8F05",
}

# Paleta funcional del proyecto (misma de los notebooks 18/19).
COLORS = {
    "primary": "#0173B2",      # azul principal / correcto
    "secondary": "#DE8F05",    # naranja secundario / alternativo
    "success": "#4CAF50",      # verde aciertos
    "error_intra": "#D32F2F",  # rojo confusión intra-familia
    "error_inter": "#9E9E9E",  # gris confusión inter-familia
    "highlight": "#1565C0",    # azul cobalto (bordes / énfasis)
    "neutral": "#333333",      # texto de anotaciones
}

# Paleta categórica (features PGLS) para agrupar familias/categorías si hace falta.
CATEGORICAL = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

# Colormap estándar para matrices de confusión (row-normalized %, 0-100).
CMAP_CONFUSION = "YlOrRd"

# fig/ está en la raíz del repositorio (este módulo vive en src/).
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fig")


def apply_style():
    """Aplica los rcParams y el tema seaborn estándar del proyecto."""
    try:
        import seaborn as sns
        sns.set_theme(style="white")
    except Exception:
        pass
    plt.rcParams.update(RCPARAMS)


def despine(ax):
    """Oculta spines superior/derecho y coloca el grid por debajo (convención del proyecto)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def panel_label(ax, letter, x=-0.02, y=1.03, fontsize=12, on_color=False):
    """Etiqueta de panel para figuras multipart, según la guía editorial.

    La letra va en cursiva dentro de paréntesis rectos (roman), p. ej. '(a)', y se
    coloca en la esquina superior izquierda —fuera del área del panel para no
    interferir con el contenido y en la misma posición en todos los paneles—.
    Con `on_color=True` la etiqueta se dibuja dentro de un círculo blanco (para
    fondos de color, p. ej. mapas de calor). Las sub-partes usan numerales romanos
    entre paréntesis, p. ej. `panel_label(ax, "ii")`.
    """
    txt = rf"$(\mathit{{{letter}}})$"
    bbox = dict(boxstyle="circle,pad=0.28", fc="white", ec="none") if on_color else None
    ax.text(x, y, txt, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=fontsize, bbox=bbox, zorder=10)


def savefig_dual(fig, name, fig_dir=None, dpi=300):
    """Guarda la figura como PNG (a `dpi`) y PDF en `fig/`, con bbox_inches='tight'.

    `name` es el nombre base sin extensión. Devuelve (ruta_png, ruta_pdf).
    """
    fig_dir = fig_dir or FIG_DIR
    os.makedirs(fig_dir, exist_ok=True)
    png = os.path.join(fig_dir, f"{name}.png")
    pdf = os.path.join(fig_dir, f"{name}.pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    return png, pdf
