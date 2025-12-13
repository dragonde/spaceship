"""
Script para convertir Num, Age y TotalExpenses a enteros
Genera train5.csv a partir de train4.csv
"""

import pandas as pd
import numpy as np

# Cargar datos
print("Cargando train4.csv...")
df = pd.read_csv('train4.csv')

print(f"\nDimensiones: {df.shape}")

# Verificar tipos actuales
print(f"\n{'='*60}")
print("TIPOS DE DATOS ANTES DE LA CONVERSIÓN")
print(f"{'='*60}")
print(f"  Num:           {df['Num'].dtype} (nulos: {df['Num'].isnull().sum()})")
print(f"  Age:           {df['Age'].dtype} (nulos: {df['Age'].isnull().sum()})")
print(f"  TotalExpenses: {df['TotalExpenses'].dtype} (nulos: {df['TotalExpenses'].isnull().sum()})")

# Mostrar algunos valores antes de la conversión
print(f"\nEjemplos de valores ANTES:")
print(f"  Num:           {df['Num'].head(5).tolist()}")
print(f"  Age:           {df['Age'].head(5).tolist()}")
print(f"  TotalExpenses: {df['TotalExpenses'].head(5).tolist()}")

# Verificar si TotalExpenses tiene valores decimales
has_decimals = (df['TotalExpenses'].dropna() % 1 != 0).any()
print(f"\n¿TotalExpenses tiene valores con decimales? {has_decimals}")
if has_decimals:
    decimal_values = df[df['TotalExpenses'].dropna() % 1 != 0]['TotalExpenses'].head()
    print(f"Ejemplos de valores con decimales: {decimal_values.tolist()}")

# Convertir a Int64 (nullable integer type)
df['Num'] = df['Num'].astype('Int64')
df['Age'] = df['Age'].astype('Int64')
df['TotalExpenses'] = df['TotalExpenses'].astype('Int64')

print(f"\n{'='*60}")
print("CONVERSIÓN COMPLETADA")
print(f"{'='*60}")

print(f"\n{'='*60}")
print("TIPOS DE DATOS DESPUÉS DE LA CONVERSIÓN")
print(f"{'='*60}")
print(f"  Num:           {df['Num'].dtype} (nulos: {df['Num'].isnull().sum()})")
print(f"  Age:           {df['Age'].dtype} (nulos: {df['Age'].isnull().sum()})")
print(f"  TotalExpenses: {df['TotalExpenses'].dtype} (nulos: {df['TotalExpenses'].isnull().sum()})")

# Mostrar algunos valores después de la conversión
print(f"\nEjemplos de valores DESPUÉS:")
print(f"  Num:           {df['Num'].head(5).tolist()}")
print(f"  Age:           {df['Age'].head(5).tolist()}")
print(f"  TotalExpenses: {df['TotalExpenses'].head(5).tolist()}")

# Verificar rangos de valores
print(f"\nRangos de valores:")
print(f"  Num:           {df['Num'].min()} - {df['Num'].max()}")
print(f"  Age:           {df['Age'].min()} - {df['Age'].max()} (media: {df['Age'].mean():.2f})")
print(f"  TotalExpenses: {df['TotalExpenses'].min()} - {df['TotalExpenses'].max()} (media: {df['TotalExpenses'].mean():.2f})")

# Mostrar primeras filas
print(f"\n{'='*60}")
print("PRIMERAS 5 FILAS DEL NUEVO DATASET")
print(f"{'='*60}")
print(df.head())

# Mostrar tipos de datos de todas las columnas
print(f"\n{'='*60}")
print("TIPOS DE DATOS DE TODAS LAS COLUMNAS")
print(f"{'='*60}")
for col in df.columns:
    print(f"  {col:15s} → {df[col].dtype}")

# Guardar nuevo CSV
output_file = 'train5.csv'
df.to_csv(output_file, index=False)

print(f"\n{'='*60}")
print(f"✓ Archivo guardado exitosamente: {output_file}")
print(f"{'='*60}")
print(f"\nLa columna 'Age' ahora es de tipo Int64 (entero nullable)")
print(f"Esto permite mantener los valores nulos sin convertirlos a números")
