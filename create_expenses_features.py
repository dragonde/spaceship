"""
Script para crear versión modificada de train.csv
con features de gastos consolidadas
"""

import pandas as pd
import numpy as np

# Cargar datos
print("Cargando train.csv...")
df = pd.read_csv('train.csv')

# Columnas de gastos individuales
expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

print(f"\nDimensiones originales: {df.shape}")
print(f"Columnas de gastos: {expense_cols}")

# Crear TotalExpenses (suma de todos los gastos)
df['TotalExpenses'] = df[expense_cols].sum(axis=1)

# Crear HasExpenses (1 si gastó algo, 0 si no gastó nada o todos los gastos son NaN)
df['HasExpenses'] = (df[expense_cols].sum(axis=1) > 0).astype(int)

# Eliminar columnas de gastos individuales
df_modified = df.drop(columns=expense_cols)

print(f"\nDimensiones nuevas: {df_modified.shape}")
print(f"\nNuevas columnas agregadas:")
print(f"  - TotalExpenses (float)")
print(f"  - HasExpenses (int: 0 o 1)")

# Mostrar estadísticas de las nuevas variables
print(f"\n{'='*60}")
print("ESTADÍSTICAS DE NUEVAS VARIABLES")
print(f"{'='*60}")

print(f"\nTotalExpenses:")
print(f"  Media: ${df_modified['TotalExpenses'].mean():.2f}")
print(f"  Mediana: ${df_modified['TotalExpenses'].median():.2f}")
print(f"  Mínimo: ${df_modified['TotalExpenses'].min():.2f}")
print(f"  Máximo: ${df_modified['TotalExpenses'].max():.2f}")
print(f"  Valores nulos: {df_modified['TotalExpenses'].isnull().sum()}")

print(f"\nHasExpenses:")
print(f"  Pasajeros que NO gastaron (0): {(df_modified['HasExpenses'] == 0).sum()} ({(df_modified['HasExpenses'] == 0).sum()/len(df_modified)*100:.2f}%)")
print(f"  Pasajeros que SÍ gastaron (1): {(df_modified['HasExpenses'] == 1).sum()} ({(df_modified['HasExpenses'] == 1).sum()/len(df_modified)*100:.2f}%)")

# Mostrar primeras filas
print(f"\n{'='*60}")
print("PRIMERAS 5 FILAS DEL NUEVO DATASET")
print(f"{'='*60}")
print(df_modified.head())

# Guardar nuevo CSV
output_file = 'train_with_expense_features.csv'
df_modified.to_csv(output_file, index=False)

print(f"\n{'='*60}")
print(f"✓ Archivo guardado exitosamente: {output_file}")
print(f"{'='*60}")

# Mostrar columnas finales
print(f"\nColumnas en el nuevo archivo ({len(df_modified.columns)}):")
for i, col in enumerate(df_modified.columns, 1):
    print(f"  {i:2d}. {col}")
