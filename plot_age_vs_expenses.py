import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar datos
df = pd.read_csv('train9.csv')

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Relación entre Edad y Gastos Totales', fontsize=16, fontweight='bold')

# 1. Scatter plot básico
ax1 = axes[0, 0]
scatter = ax1.scatter(df['Age'], df['TotalExpenses'], alpha=0.3, s=20, c='steelblue')
ax1.set_xlabel('Edad (años)', fontsize=11)
ax1.set_ylabel('Gastos Totales ($)', fontsize=11)
ax1.set_title('Scatter Plot: Edad vs Gastos Totales', fontsize=12)
ax1.grid(True, alpha=0.3)

# Agregar línea de tendencia
mask = df['Age'].notna() & df['TotalExpenses'].notna()
if mask.sum() > 0:
    z = np.polyfit(df[mask]['Age'], df[mask]['TotalExpenses'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Age'].min(), df['Age'].max(), 100)
    ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Tendencia: y={z[0]:.2f}x+{z[1]:.2f}')
    ax1.legend()

# 2. Boxplot por grupos de edad
ax2 = axes[0, 1]
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 45, 60, 80],
                         labels=['0-12', '13-18', '19-30', '31-45', '46-60', '61+'])
df.boxplot(column='TotalExpenses', by='AgeGroup', ax=ax2)
ax2.set_xlabel('Grupo de Edad', fontsize=11)
ax2.set_ylabel('Gastos Totales ($)', fontsize=11)
ax2.set_title('Distribución de Gastos por Grupo de Edad', fontsize=12)
plt.sca(ax2)
plt.xticks(rotation=0)

# 3. Heatmap de densidad (hexbin)
ax3 = axes[1, 0]
# Filtrar filas donde ambas columnas tienen valores
mask_hexbin = df['Age'].notna() & df['TotalExpenses'].notna()
df_valid = df[mask_hexbin]
hexbin = ax3.hexbin(df_valid['Age'], df_valid['TotalExpenses'],
                     gridsize=30, cmap='YlOrRd', mincnt=1)
ax3.set_xlabel('Edad (años)', fontsize=11)
ax3.set_ylabel('Gastos Totales ($)', fontsize=11)
ax3.set_title('Densidad: Edad vs Gastos (Hexbin)', fontsize=12)
plt.colorbar(hexbin, ax=ax3, label='Frecuencia')

# 4. Gastos promedio por edad
ax4 = axes[1, 1]
age_stats = df.groupby('Age')['TotalExpenses'].agg(['mean', 'median', 'count']).reset_index()
age_stats = age_stats[age_stats['count'] >= 5]  # Solo edades con 5+ observaciones

ax4.plot(age_stats['Age'], age_stats['mean'], marker='o', linewidth=2,
         markersize=4, label='Media', color='steelblue')
ax4.plot(age_stats['Age'], age_stats['median'], marker='s', linewidth=2,
         markersize=4, label='Mediana', color='coral')
ax4.set_xlabel('Edad (años)', fontsize=11)
ax4.set_ylabel('Gastos Totales ($)', fontsize=11)
ax4.set_title('Gastos Promedio por Edad (≥5 observaciones)', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/age_vs_totalexpenses.png', dpi=300, bbox_inches='tight')
print("✓ Gráfica guardada en: plots/age_vs_totalexpenses.png")

# Estadísticas adicionales
print("\n" + "="*60)
print("ESTADÍSTICAS: Relación Edad vs Gastos Totales")
print("="*60)

# Correlación
correlation = df[['Age', 'TotalExpenses']].corr().iloc[0, 1]
print(f"\nCorrelación (Pearson): {correlation:.4f}")

# Estadísticas por grupo de edad
print("\nGastos promedio por grupo de edad:")
print(df.groupby('AgeGroup')['TotalExpenses'].agg(['mean', 'median', 'std', 'count']))

# Identificar outliers
print("\n\nTop 5 mayores gastos:")
top_spenders = df.nlargest(5, 'TotalExpenses')[['Age', 'TotalExpenses', 'HomePlanet', 'VIP']]
print(top_spenders.to_string(index=False))

print("\n" + "="*60)
