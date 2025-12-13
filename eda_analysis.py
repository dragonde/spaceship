"""
Análisis Exploratorio de Datos - Spaceship Titanic
Este script realiza un análisis exploratorio completo del dataset train.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Cargar datos
print("="*80)
print("ANÁLISIS EXPLORATORIO DE DATOS - SPACESHIP TITANIC")
print("="*80)

df = pd.read_csv('train.csv')

# ============================================================================
# 1. INFORMACIÓN BÁSICA DEL DATASET
# ============================================================================
print("\n1. INFORMACIÓN BÁSICA DEL DATASET")
print("-"*80)
print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"\nPrimeras 5 filas:")
print(df.head())

print("\n\nTipos de datos:")
print(df.dtypes)

print("\n\nInformación general:")
df.info()

# ============================================================================
# 2. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================================
print("\n\n2. ESTADÍSTICAS DESCRIPTIVAS")
print("-"*80)
print("\nVariables numéricas:")
print(df.describe())

print("\n\nVariables categóricas:")
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
print(df[categorical_cols].describe())

# ============================================================================
# 3. VALORES NULOS
# ============================================================================
print("\n\n3. ANÁLISIS DE VALORES NULOS")
print("-"*80)
null_counts = df.isnull().sum()
null_percentages = (null_counts / len(df)) * 100
null_df = pd.DataFrame({
    'Columna': null_counts.index,
    'Valores Nulos': null_counts.values,
    'Porcentaje (%)': null_percentages.values
})
null_df = null_df[null_df['Valores Nulos'] > 0].sort_values('Valores Nulos', ascending=False)
print(null_df.to_string(index=False))

# ============================================================================
# 4. ANÁLISIS DE VARIABLE OBJETIVO (Transported)
# ============================================================================
print("\n\n4. ANÁLISIS DE VARIABLE OBJETIVO - TRANSPORTED")
print("-"*80)
transported_counts = df['Transported'].value_counts()
transported_pct = df['Transported'].value_counts(normalize=True) * 100
print("\nDistribución:")
for value, count in transported_counts.items():
    pct = transported_pct[value]
    print(f"  {value}: {count} ({pct:.2f}%)")

# ============================================================================
# 5. ANÁLISIS DE VARIABLES CATEGÓRICAS
# ============================================================================
print("\n\n5. ANÁLISIS DE VARIABLES CATEGÓRICAS")
print("-"*80)

categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']

for col in categorical_features:
    print(f"\n{col}:")
    print(df[col].value_counts())
    print(f"  Valores únicos: {df[col].nunique()}")

# ============================================================================
# 6. ANÁLISIS DE VARIABLES NUMÉRICAS
# ============================================================================
print("\n\n6. ANÁLISIS DE VARIABLES NUMÉRICAS")
print("-"*80)

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

print("\nEstadísticas por variable:")
for col in numeric_features:
    print(f"\n{col}:")
    print(f"  Media: {df[col].mean():.2f}")
    print(f"  Mediana: {df[col].median():.2f}")
    print(f"  Desviación estándar: {df[col].std():.2f}")
    print(f"  Mínimo: {df[col].min():.2f}")
    print(f"  Máximo: {df[col].max():.2f}")

# ============================================================================
# 7. VISUALIZACIONES
# ============================================================================
print("\n\n7. GENERANDO VISUALIZACIONES...")
print("-"*80)

# Crear carpeta para guardar gráficos
Path('plots').mkdir(exist_ok=True)

# 7.1 Distribución de la variable objetivo
fig, ax = plt.subplots(figsize=(8, 6))
df['Transported'].value_counts().plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
ax.set_title('Distribución de Pasajeros Transportados', fontsize=14, fontweight='bold')
ax.set_xlabel('Transportado', fontsize=12)
ax.set_ylabel('Cantidad', fontsize=12)
ax.set_xticklabels(['No', 'Sí'], rotation=0)
plt.tight_layout()
plt.savefig('plots/01_transported_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: plots/01_transported_distribution.png")
plt.close()

# 7.2 Valores nulos por columna
if not null_df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(null_df['Columna'], null_df['Porcentaje (%)'], color='coral')
    ax.set_xlabel('Porcentaje de Valores Nulos (%)', fontsize=12)
    ax.set_title('Valores Nulos por Columna', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/02_missing_values.png', dpi=300, bbox_inches='tight')
    print("✓ Guardado: plots/02_missing_values.png")
    plt.close()

# 7.3 Distribución de variables categóricas
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, col in enumerate(categorical_features):
    if col in df.columns:
        df[col].value_counts().plot(kind='bar', ax=axes[idx], color='steelblue')
        axes[idx].set_title(f'Distribución de {col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col, fontsize=10)
        axes[idx].set_ylabel('Cantidad', fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('plots/03_categorical_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: plots/03_categorical_distributions.png")
plt.close()

# 7.4 Distribución de edad
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df['Age'].hist(bins=30, ax=axes[0], color='mediumpurple', edgecolor='black')
axes[0].set_title('Histograma de Edad', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Edad', fontsize=10)
axes[0].set_ylabel('Frecuencia', fontsize=10)

df.boxplot(column='Age', ax=axes[1], patch_artist=True)
axes[1].set_title('Boxplot de Edad', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Edad', fontsize=10)

plt.tight_layout()
plt.savefig('plots/04_age_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: plots/04_age_distribution.png")
plt.close()

# 7.5 Distribución de gastos
expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

for idx, col in enumerate(expense_cols):
    df[col].hist(bins=30, ax=axes[idx], color='teal', edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribución de {col}', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel(col, fontsize=9)
    axes[idx].set_ylabel('Frecuencia', fontsize=9)

# Ocultar el subplot extra
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('plots/05_expenses_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: plots/05_expenses_distribution.png")
plt.close()

# 7.6 Matriz de correlación de variables numéricas
numeric_df = df[numeric_features].dropna()
correlation_matrix = numeric_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title('Matriz de Correlación - Variables Numéricas', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/06_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: plots/06_correlation_matrix.png")
plt.close()

# 7.7 Transported vs HomePlanet
if 'HomePlanet' in df.columns and 'Transported' in df.columns:
    ct = pd.crosstab(df['HomePlanet'], df['Transported'], normalize='index') * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    ct.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
    ax.set_title('Tasa de Transporte por Planeta de Origen', fontsize=14, fontweight='bold')
    ax.set_xlabel('Planeta de Origen', fontsize=12)
    ax.set_ylabel('Porcentaje (%)', fontsize=12)
    ax.legend(['No Transportado', 'Transportado'], loc='best')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig('plots/07_transported_by_homeplanet.png', dpi=300, bbox_inches='tight')
    print("✓ Guardado: plots/07_transported_by_homeplanet.png")
    plt.close()

# 7.8 Transported vs CryoSleep
if 'CryoSleep' in df.columns and 'Transported' in df.columns:
    ct = pd.crosstab(df['CryoSleep'], df['Transported'], normalize='index') * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    ct.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
    ax.set_title('Tasa de Transporte por Estado de CryoSleep', fontsize=14, fontweight='bold')
    ax.set_xlabel('CryoSleep', fontsize=12)
    ax.set_ylabel('Porcentaje (%)', fontsize=12)
    ax.legend(['No Transportado', 'Transportado'], loc='best')
    ax.set_xticklabels(['No', 'Sí'], rotation=0)
    plt.tight_layout()
    plt.savefig('plots/08_transported_by_cryosleep.png', dpi=300, bbox_inches='tight')
    print("✓ Guardado: plots/08_transported_by_cryosleep.png")
    plt.close()

# 7.9 Edad por estado de transporte
fig, ax = plt.subplots(figsize=(10, 6))
df.boxplot(column='Age', by='Transported', ax=ax, patch_artist=True)
ax.set_title('Distribución de Edad por Estado de Transporte', fontsize=14, fontweight='bold')
ax.set_xlabel('Transportado', fontsize=12)
ax.set_ylabel('Edad', fontsize=12)
plt.suptitle('')
plt.tight_layout()
plt.savefig('plots/09_age_by_transported.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: plots/09_age_by_transported.png")
plt.close()

# 7.10 Gastos totales
df['TotalExpenses'] = df[expense_cols].sum(axis=1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df['TotalExpenses'].hist(bins=50, ax=axes[0], color='darkgreen', edgecolor='black', alpha=0.7)
axes[0].set_title('Distribución de Gastos Totales', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Gastos Totales', fontsize=10)
axes[0].set_ylabel('Frecuencia', fontsize=10)

df.boxplot(column='TotalExpenses', by='Transported', ax=axes[1], patch_artist=True)
axes[1].set_title('Gastos Totales por Estado de Transporte', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Transportado', fontsize=10)
axes[1].set_ylabel('Gastos Totales', fontsize=10)
plt.suptitle('')

plt.tight_layout()
plt.savefig('plots/10_total_expenses.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: plots/10_total_expenses.png")
plt.close()

# ============================================================================
# 8. INSIGHTS ADICIONALES
# ============================================================================
print("\n\n8. INSIGHTS ADICIONALES")
print("-"*80)

# Crear variable de gasto total si no existe
if 'TotalExpenses' not in df.columns:
    df['TotalExpenses'] = df[expense_cols].sum(axis=1)

print(f"\nGasto total promedio: ${df['TotalExpenses'].mean():.2f}")
print(f"Gasto total mediano: ${df['TotalExpenses'].median():.2f}")

print("\nGasto promedio por planeta de origen:")
print(df.groupby('HomePlanet')['TotalExpenses'].mean().sort_values(ascending=False))

print("\nTasa de transporte por edad (grupos):")
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 50, 100],
                        labels=['Niño', 'Adolescente', 'Joven', 'Adulto', 'Mayor'])
age_transport = df.groupby('AgeGroup')['Transported'].value_counts(normalize=True).unstack() * 100
print(age_transport)

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("Todas las visualizaciones se han guardado en la carpeta 'plots/'")
print("="*80)
