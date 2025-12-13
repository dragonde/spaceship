"""
Script para convertir la columna Num de float a entero
Genera train4.csv a partir de train3.csv
"""

import pandas as pd
import numpy as np

# Cargar datos
print("Cargando train3.csv...")
df = pd.read_csv('train3.csv')

print(f"\nDimensiones: {df.shape}")

# Verificar tipo actual de Num
print(f"\nTipo actual de 'Num': {df['Num'].dtype}")
print(f"Valores nulos en 'Num': {df['Num'].isnull().sum()}")

# Mostrar algunos valores antes de la conversión
print(f"\nEjemplos de valores en 'Num' (antes):")
print(df['Num'].head(10))

# Convertir Num a Int64 (nullable integer type)
# Esto permite mantener los valores nulos como NaN en lugar de convertirlos a un número
df['Num'] = df['Num'].astype('Int64')

print(f"\n{'='*60}")
print("CONVERSIÓN COMPLETADA")
print(f"{'='*60}")

print(f"\nTipo nuevo de 'Num': {df['Num'].dtype}")
print(f"Valores nulos en 'Num': {df['Num'].isnull().sum()}")

# Mostrar algunos valores después de la conversión
print(f"\nEjemplos de valores en 'Num' (después):")
print(df['Num'].head(10))

# Verificar rango de valores
print(f"\nRango de valores:")
print(f"  Mínimo: {df['Num'].min()}")
print(f"  Máximo: {df['Num'].max()}")

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
output_file = 'train4.csv'
df.to_csv(output_file, index=False)

print(f"\n{'='*60}")
print(f"✓ Archivo guardado exitosamente: {output_file}")
print(f"{'='*60}")
print(f"\nLa columna 'Num' ahora es de tipo Int64 (entero nullable)")
print(f"Esto permite mantener los valores nulos sin convertirlos a números")
