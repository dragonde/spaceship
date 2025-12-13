"""
Script para separar la columna Cabin en Deck/Num/Side
Genera train2.csv a partir de train1.csv
"""

import pandas as pd
import numpy as np

# Cargar datos
print("Cargando train1.csv...")
df = pd.read_csv('/home/alber/myrepo/spaceship/train1.csv')

print(f"\nDimensiones originales: {df.shape}")
print(f"Columnas: {list(df.columns)}")

# Verificar formato de Cabin
print(f"\nEjemplos de valores en Cabin:")
print(df['Cabin'].head(10))

# Separar la columna Cabin en Deck/Num/Side
print("\nSeparando columna Cabin en Deck/Num/Side...")

# Usar str.split para separar por '/'
cabin_split = df['Cabin'].str.split('/', expand=True)

# Asignar nombres a las nuevas columnas
df['Deck'] = cabin_split[0]
df['Num'] = cabin_split[1]
df['Side'] = cabin_split[2]

# Convertir Num a entero (manejar valores nulos)
df['Num'] = pd.to_numeric(df['Num'], errors='coerce')

# Eliminar la columna Cabin original
df_new = df.drop(columns=['Cabin'])

# Reordenar columnas para poner Deck/Num/Side donde estaba Cabin
# Obtener índice donde estaba Cabin (era la 4ta columna: 0-indexed = 3)
cols = df_new.columns.tolist()

# Mover Deck, Num, Side a la posición 3, 4, 5 (donde estaba Cabin)
# Primero remover del final
cols.remove('Deck')
cols.remove('Num')
cols.remove('Side')

# Insertar en posición 3 (después de CryoSleep)
cols.insert(3, 'Deck')
cols.insert(4, 'Num')
cols.insert(5, 'Side')

df_new = df_new[cols]

print(f"\nDimensiones nuevas: {df_new.shape}")

# Estadísticas de las nuevas columnas
print(f"\n{'='*60}")
print("ESTADÍSTICAS DE NUEVAS COLUMNAS")
print(f"{'='*60}")

print(f"\nDeck:")
print(f"  Valores únicos: {df_new['Deck'].nunique()}")
print(f"  Valores nulos: {df_new['Deck'].isnull().sum()}")
print(f"  Distribución:")
print(df_new['Deck'].value_counts().head(10))

print(f"\nNum:")
print(f"  Valores únicos: {df_new['Num'].nunique()}")
print(f"  Valores nulos: {df_new['Num'].isnull().sum()}")
print(f"  Mínimo: {df_new['Num'].min()}")
print(f"  Máximo: {df_new['Num'].max()}")

print(f"\nSide:")
print(f"  Valores únicos: {df_new['Side'].nunique()}")
print(f"  Valores nulos: {df_new['Side'].isnull().sum()}")
print(f"  Distribución:")
print(df_new['Side'].value_counts())

# Mostrar primeras filas
print(f"\n{'='*60}")
print("PRIMERAS 5 FILAS DEL NUEVO DATASET")
print(f"{'='*60}")
print(df_new.head())

# Guardar nuevo CSV
output_file = 'train2.csv'
df_new.to_csv(output_file, index=False)

print(f"\n{'='*60}")
print(f"✓ Archivo guardado exitosamente: {output_file}")
print(f"{'='*60}")

# Mostrar columnas finales
print(f"\nColumnas en el nuevo archivo ({len(df_new.columns)}):")
for i, col in enumerate(df_new.columns, 1):
    dtype = df_new[col].dtype
    print(f"  {i:2d}. {col:15s} ({dtype})")
