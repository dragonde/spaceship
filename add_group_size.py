"""
Script para agregar columna GroupSize (tamaño del grupo)
Genera train7.csv a partir de train6.csv
"""

import pandas as pd

# Cargar datos
print("Cargando train6.csv...")
df = pd.read_csv('train6.csv')

print(f"\nDimensiones: {df.shape}")
print(f"\nColumnas actuales: {list(df.columns)}")

# Calcular tamaño de cada grupo
print("\nCalculando tamaño de grupos...")
df['GroupSize'] = df.groupby('Group')['Group'].transform('count')

print(f"\nColumna 'GroupSize' creada")
print(f"  Tipo: {df['GroupSize'].dtype}")

# Reordenar columnas para poner GroupSize después de Group
cols = df.columns.tolist()

# Remover GroupSize del final
cols.remove('GroupSize')

# Insertar en posición 2 (después de Group que está en posición 1)
# Posición 0: Group, Posición 1: NumInGroup, Posición 2: GroupSize (nuevo)
cols.insert(2, 'GroupSize')

df_new = df[cols]

print(f"\n{'='*60}")
print("ESTADÍSTICAS DE GroupSize")
print(f"{'='*60}")

print(f"\nTipo: {df_new['GroupSize'].dtype}")
print(f"Valores únicos: {df_new['GroupSize'].nunique()}")
print(f"Mínimo: {df_new['GroupSize'].min()}")
print(f"Máximo: {df_new['GroupSize'].max()}")
print(f"Media: {df_new['GroupSize'].mean():.2f}")
print(f"Mediana: {df_new['GroupSize'].median():.1f}")

# Distribución de tamaños de grupo
print(f"\nDistribución de tamaños de grupo:")
size_dist = df_new['GroupSize'].value_counts().sort_index()
for size, count in size_dist.items():
    pct = (count / len(df_new)) * 100
    print(f"  Tamaño {size}: {count:4d} pasajeros ({pct:5.2f}%)")

# Validación: GroupSize debe ser igual al máximo NumInGroup en cada grupo
print(f"\n{'='*60}")
print("VALIDACIÓN")
print(f"{'='*60}")
max_num_in_group = df_new.groupby('Group')['NumInGroup'].max()
group_size_check = df_new.groupby('Group')['GroupSize'].first()
validation = (max_num_in_group == group_size_check).all()
print(f"\n¿GroupSize = max(NumInGroup) para cada grupo? {validation}")

# Mostrar ejemplos
print(f"\n{'='*60}")
print("EJEMPLOS DE GRUPOS")
print(f"{'='*60}")

print("\nGrupo 3 (2 personas):")
print(df_new[df_new['Group'] == 3][['Group', 'NumInGroup', 'GroupSize', 'Name', 'Age']])

print("\nGrupo 6 (2 personas):")
print(df_new[df_new['Group'] == 6][['Group', 'NumInGroup', 'GroupSize', 'Name', 'Age']])

# Encontrar un grupo grande
large_groups = df_new[df_new['GroupSize'] >= 6]['Group'].unique()
if len(large_groups) > 0:
    example_group = large_groups[0]
    print(f"\nGrupo {example_group} ({df_new[df_new['Group'] == example_group]['GroupSize'].iloc[0]} personas):")
    print(df_new[df_new['Group'] == example_group][['Group', 'NumInGroup', 'GroupSize', 'Name', 'Age']])

# Mostrar primeras filas
print(f"\n{'='*60}")
print("PRIMERAS 10 FILAS DEL NUEVO DATASET")
print(f"{'='*60}")
print(df_new[['Group', 'NumInGroup', 'GroupSize', 'HomePlanet', 'Age', 'Name']].head(10))

# Mostrar tipos de datos
print(f"\n{'='*60}")
print("TIPOS DE DATOS DE TODAS LAS COLUMNAS")
print(f"{'='*60}")
for i, col in enumerate(df_new.columns, 1):
    dtype = df_new[col].dtype
    marker = " ← NUEVA" if col == 'GroupSize' else ""
    print(f"  {i:2d}. {col:15s} → {dtype}{marker}")

# Guardar nuevo CSV
output_file = 'train7.csv'
df_new.to_csv(output_file, index=False)

print(f"\n{'='*60}")
print(f"✓ Archivo guardado exitosamente: {output_file}")
print(f"{'='*60}")
print(f"\nColumna 'GroupSize' agregada en posición 3 (después de Group y NumInGroup)")
print(f"Dimensiones: {df_new.shape[0]} filas × {df_new.shape[1]} columnas")
