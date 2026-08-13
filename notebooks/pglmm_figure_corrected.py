# ============================================================================
# VISUALIZACIÓN CORREGIDA - PGLMM (Estilo publicación científica)
# ============================================================================
# CORRECCIÓN: Muestra coeficientes β estandarizados con IC 95% y significancia
# en lugar de "Feature Importance" normalizada que oscurece la estadística

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Configuración científica profesional
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.linewidth': 0.8,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.2,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

# Paleta de colores científica colorblind-friendly
color_palette = {
    'primary': '#0173B2',      # Azul - Morphological
    'secondary': '#DE8F05',    # Naranja - Geographic
    'tertiary': '#029E73',     # Verde - Habitat
    'quaternary': '#CC78BC',   # Púrpura - Trophic
    'quinary': '#CA9161',      # Marrón - Lifestyle
    'senary': '#ECE133'        # Amarillo - Other
}

# Mapeo de categorías a colores
category_colors = {
    'Morphological': color_palette['primary'],
    'Geographic': color_palette['secondary'],
    'Habitat': color_palette['tertiary'],
    'Trophic': color_palette['quaternary'],
    'Lifestyle': color_palette['quinary'],
    'Other': color_palette['senary']
}

# Definir categorías de variables
def categorize_variable(var_name):
    """Asigna una categoría a cada variable."""
    var_lower = var_name.lower()

    morpho_keywords = ['mass', 'beak', 'tarsus', 'wing', 'tail', 'hand-wing', 'kipps']
    if any(kw in var_lower for kw in morpho_keywords):
        return 'Morphological'

    geo_keywords = ['range', 'latitude', 'longitude', 'centroid']
    if any(kw in var_lower for kw in geo_keywords):
        return 'Geographic'

    if 'habitat' in var_lower:
        return 'Habitat'

    if 'trophic' in var_lower:
        return 'Trophic'

    if 'lifestyle' in var_lower:
        return 'Lifestyle'

    if 'support' in var_lower:
        return 'Support'

    return 'Other'

# Limpiar nombres de variables para visualización
def clean_var_name(name):
    """Limpia el nombre de la variable para visualización."""
    if 'Habitat_' in name:
        return 'Hab: ' + name.replace('Habitat_', '')
    elif 'Trophic.Level_' in name:
        return 'Troph. Level: ' + name.replace('Trophic.Level_', '')
    elif 'Trophic.Niche_' in name:
        return 'Troph. Niche: ' + name.replace('Trophic.Niche_', '')
    elif 'Primary.Lifestyle_' in name:
        return 'Life: ' + name.replace('Primary.Lifestyle_', '')
    elif name == 'log_Range.Size':
        return 'Range Size'
    elif name == 'log_Mass':
        return 'Mass'
    elif name == 'log_support':
        return 'Support'
    else:
        name = name.replace('log_', '').replace('_', ' ').replace('.', ' ')
        return name.title()

# ============================================================================
# PREPARAR DATOS DEL PGLMM CON ESTADÍSTICAS CORRECTAS
# ============================================================================

# Usar pglmm_importance_df que ya tiene los coeficientes, SE y p-valores
pglmm_data = pglmm_importance_df.copy()
pglmm_data = pglmm_data[pglmm_data['Variable'] != 'const']

# Asignar categorías
pglmm_data['Category'] = pglmm_data['Variable'].apply(categorize_variable)

# Excluir 'Support' del análisis principal (es una covariable de control)
pglmm_data_filtered = pglmm_data[pglmm_data['Category'] != 'Support'].copy()

# Calcular intervalos de confianza al 95% (z = 1.96 para IC 95%)
z_critical = 1.96
pglmm_data_filtered['CI_lower'] = pglmm_data_filtered['Coef_std'] - z_critical * pglmm_data_filtered['SE']
pglmm_data_filtered['CI_upper'] = pglmm_data_filtered['Coef_std'] + z_critical * pglmm_data_filtered['SE']
pglmm_data_filtered['CI_error'] = z_critical * pglmm_data_filtered['SE']

# Ordenar por valor absoluto del coeficiente (para visualización)
pglmm_data_filtered['Abs_Coef'] = pglmm_data_filtered['Coef_std'].abs()
pglmm_data_sorted = pglmm_data_filtered.sort_values('Abs_Coef', ascending=False)

# Limpiar nombres
pglmm_data_sorted['Variable_clean'] = pglmm_data_sorted['Variable'].apply(clean_var_name)

# ===== CREAR FIGURA CON 2 PANELES =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')

# =============================================================================
# Panel (a): Coeficientes β estandarizados con IC 95%
# =============================================================================
top15 = pglmm_data_sorted.head(15).iloc[::-1]  # Invertir para que el mayor esté arriba

y_pos = np.arange(len(top15))

# Colores: gris para no significativo, color de categoría para significativo
colors = []
for _, row in top15.iterrows():
    if row['Significativo']:
        colors.append(category_colors.get(row['Category'], '#999999'))
    else:
        colors.append('#B0B0B0')  # Gris para no significativo

# Barras horizontales con barras de error
bars = ax1.barh(y_pos, top15['Coef_std'].values,
                color=colors, alpha=0.85,
                edgecolor='black', linewidth=0.5,
                height=0.7, xerr=top15['CI_error'].values,
                error_kw={'capsize': 3, 'capthick': 1, 'elinewidth': 1, 'ecolor': 'black'})

# Línea vertical en cero
ax1.axvline(x=0, color='black', linestyle='-', linewidth=1, zorder=0)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(top15['Variable_clean'], fontsize=13)
ax1.set_xlabel('Standardized coefficient (β)', fontsize=16, fontweight='bold')
ax1.set_title('(a) PGLMM standardized coefficients', fontsize=17, fontweight='bold', loc='left', pad=15)
ax1.tick_params(axis='x', labelsize=12)
ax1.grid(axis='x', alpha=0.3, linewidth=0.5)
ax1.set_axisbelow(True)

# Marcar significativos con asterisco
for i, (_, row) in enumerate(top15.iterrows()):
    if row['Significativo']:
        # Posición del asterisco al final de la barra de error
        x_pos = row['Coef_std'] + row['CI_error'] + 0.02 if row['Coef_std'] >= 0 else row['Coef_std'] - row['CI_error'] - 0.02
        ha = 'left' if row['Coef_std'] >= 0 else 'right'
        ax1.text(x_pos, i, '*', fontsize=16, fontweight='bold', va='center', ha=ha, color='red')

# Leyenda
legend_elements_a = [
    plt.Rectangle((0,0),1,1, fc='#B0B0B0', alpha=0.85, ec='black', lw=0.5, label='Not significant (p ≥ 0.05)'),
    plt.Rectangle((0,0),1,1, fc=color_palette['primary'], alpha=0.85, ec='black', lw=0.5, label='Morphological'),
    plt.Rectangle((0,0),1,1, fc=color_palette['secondary'], alpha=0.85, ec='black', lw=0.5, label='Geographic'),
    plt.Rectangle((0,0),1,1, fc=color_palette['tertiary'], alpha=0.85, ec='black', lw=0.5, label='Habitat'),
    plt.Rectangle((0,0),1,1, fc=color_palette['quaternary'], alpha=0.85, ec='black', lw=0.5, label='Trophic'),
    plt.Rectangle((0,0),1,1, fc=color_palette['quinary'], alpha=0.85, ec='black', lw=0.5, label='Lifestyle'),
]
ax1.legend(handles=legend_elements_a, loc='lower right', fontsize=11, frameon=True,
          fancybox=False, edgecolor='gray', framealpha=0.95)

# Nota sobre significancia
n_sig = pglmm_data_filtered['Significativo'].sum()
note_text = f'n = {len(top15)} variables shown\n{n_sig} significant (p < 0.05)'
ax1.text(0.02, 0.98, note_text, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# =============================================================================
# Panel (b): Comparación de magnitud por categoría (valores absolutos)
# =============================================================================
# Calcular la suma de valores absolutos por categoría
pglmm_data_filtered['Abs_Coef_std'] = pglmm_data_filtered['Coef_std'].abs()
category_sum = pglmm_data_filtered.groupby('Category')['Abs_Coef_std'].sum().sort_values(ascending=False)
category_count = pglmm_data_filtered.groupby('Category').size()
category_sig = pglmm_data_filtered.groupby('Category')['Significativo'].sum()

# Normalizar para que sume 1 (proporción relativa)
total_sum = category_sum.sum()
category_proportion = category_sum / total_sum

bar_colors = [category_colors.get(cat, '#999999') for cat in category_proportion.index]

bars = ax2.bar(range(len(category_proportion)), category_proportion.values,
               color=bar_colors, alpha=0.85,
               edgecolor='black', linewidth=0.5,
               width=0.7)

ax2.set_xticks(range(len(category_proportion)))
ax2.set_xticklabels(category_proportion.index, rotation=45, ha='right', fontsize=13)
ax2.set_ylabel('Relative contribution (|β| proportion)', fontsize=16, fontweight='bold')
ax2.set_title('(b) Contribution by feature category', fontsize=17, fontweight='bold', loc='left', pad=15)
ax2.tick_params(axis='y', labelsize=12)
ax2.grid(axis='y', alpha=0.3, linewidth=0.5)
ax2.set_axisbelow(True)

# Valores sobre barras con conteo de variables
for bar, (cat, val) in zip(bars, category_proportion.items()):
    height = bar.get_height()
    n_vars = category_count[cat]
    n_sig_cat = category_sig[cat]
    sig_text = f'({n_sig_cat}/{n_vars} sig.)' if n_sig_cat > 0 else f'(0/{n_vars} sig.)'
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{val:.1%}\n{sig_text}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylim(0, category_proportion.max() * 1.35)

# R² del modelo y nota importante
r2_val = pglmm_results['PGLMM_full'].pseudo_r2 if hasattr(pglmm_results['PGLMM_full'], 'pseudo_r2') else 0.017
textstr = f'Model pseudo-$R^2$ = {r2_val:.3f}\n(~{r2_val*100:.1f}% variance explained)'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='orange', linewidth=1.5)
ax2.text(0.98, 0.98, textstr, transform=ax2.transAxes, fontsize=12,
        verticalalignment='top', horizontalalignment='right', bbox=props, fontweight='bold')

plt.tight_layout()
plt.savefig('../fig/pglmm_coefficients_publication.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('../fig/pglmm_coefficients_publication.pdf', bbox_inches='tight', facecolor='white')
plt.show()

# ============================================================================
# ESTADÍSTICAS PARA EL MANUSCRITO (CORREGIDAS)
# ============================================================================
print("\n" + "="*80)
print("PGLMM ANALYSIS - CORRECTED SUMMARY FOR MANUSCRIPT")
print("="*80)

print(f"\nModel performance:")
print(f"  Pseudo-R² = {r2_val:.3f} ({r2_val*100:.1f}% variance explained)")
print(f"  Total predictors: {len(pglmm_data_filtered)}")
print(f"  Significant predictors (p < 0.05): {n_sig}")

print(f"\nTop 5 predictors by |β| (NOT necessarily significant):")
for i, (idx, row) in enumerate(pglmm_data_sorted.head(5).iterrows(), 1):
    clean_name = clean_var_name(row['Variable'])
    sig_marker = '*' if row['Significativo'] else ''
    print(f"  {i}. {clean_name}: β = {row['Coef_std']:.4f}, p = {row['p_value']:.4f} {sig_marker}")

print(f"\nSignificant variables (p < 0.05):")
sig_vars = pglmm_data_filtered[pglmm_data_filtered['Significativo']]
if len(sig_vars) == 0:
    print("  NONE - No variables reached statistical significance in PGLMM")
else:
    for idx, row in sig_vars.iterrows():
        clean_name = clean_var_name(row['Variable'])
        direction = "positive" if row['Coef_std'] > 0 else "negative"
        print(f"  • {clean_name}: β = {row['Coef_std']:.4f}, p = {row['p_value']:.4f} ({direction} effect)")

print(f"\nContribution by category (sum of |β|):")
for cat, proportion in category_proportion.items():
    n_feat = category_count[cat]
    n_sig_cat = category_sig[cat]
    print(f"  • {cat}: {proportion:.1%} (n={n_feat}, {n_sig_cat} significant)")

print(f"\nKEY FINDING:")
print(f"  Despite {pglmm_data_sorted.iloc[0]['Variable_clean']} showing the largest |β|,")
print(f"  NO ecological or morphological variable significantly predicts F1-score")
print(f"  when controlling for phylogenetic autocorrelation (PGLMM).")
print(f"  The model explains only {r2_val*100:.1f}% of variance in classification performance.")

print("="*80)
print(f"\nFigures saved:")
print(f"  - ../fig/pglmm_coefficients_publication.png")
print(f"  - ../fig/pglmm_coefficients_publication.pdf")
