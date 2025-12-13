"""
Script para mover la columna Transported al final
Genera train3.csv a partir de train2.csv
"""

import pandas as pd

# Cargar datos
print("Cargando train2.csv...")
df = pd.read_csv('train2.csv')

print(f"\nDimensiones: {df.shape}")
print(f"\nColumnas originales:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# Mover Transported al final
cols = df.columns.tolist()

# Remover Transported de su posición actual
cols.remove('Transported')

# Agregar al final
cols.append('Transported')

# Reordenar dataframe
df_new = df[cols]

print(f"\n{'='*60}")
print("COLUMNAS REORDENADAS")
print(f"{'='*60}")
print(f"\nColumnas nuevas:")
for i, col in enumerate(df_new.columns, 1):
    dtype = df_new[col].dtype
    marker = " ← OBJETIVO" if col == 'Transported' else ""
    print(f"  {i:2d}. {col:15s} ({dtype}){marker}")

# Mostrar primeras filas
print(f"\n{'='*60}")
print("PRIMERAS 5 FILAS DEL NUEVO DATASET")
print(f"{'='*60}")
print(df_new.head())

# Guardar nuevo CSV
output_file = 'train3.csv'
df_new.to_csv(output_file, index=False)

print(f"\n{'='*60}")
print(f"✓ Archivo guardado exitosamente: {output_file}")
print(f"{'='*60}")
print(f"\nLa columna 'Transported' ahora está en la última posición (columna {len(df_new.columns)})")
