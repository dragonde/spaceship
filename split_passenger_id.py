"""
Script para separar PassengerId en Group y NumInGroup
Genera train6.csv a partir de train5.csv
"""

import pandas as pd

# Cargar datos
print("Cargando train5.csv...")
df = pd.read_csv('train5.csv')

print(f"\nDimensiones: {df.shape}")
print(f"\nColumnas actuales: {list(df.columns)}")

# Mostrar ejemplos de PassengerId
print(f"\nEjemplos de PassengerId:")
print(df['PassengerId'].head(10))

# Separar PassengerId en Group y NumInGroup
print("\nSeparando PassengerId en Group/NumInGroup...")

# Usar str.split para separar por '_'
passenger_split = df['PassengerId'].str.split('_', expand=True)

# Convertir a enteros
df['Group'] = passenger_split[0].astype(int)
df['NumInGroup'] = passenger_split[1].astype(int)

# Eliminar PassengerId original
df_new = df.drop(columns=['PassengerId'])

# Reordenar columnas para poner Group y NumInGroup al principio
cols = df_new.columns.tolist()

# Remover Group y NumInGroup del final
cols.remove('Group')
cols.remove('NumInGroup')

# Insertar al principio
cols.insert(0, 'Group')
cols.insert(1, 'NumInGroup')

df_new = df_new[cols]

print(f"\nDimensiones nuevas: {df_new.shape}")

# Estadísticas de las nuevas columnas
print(f"\n{'='*60}")
print("ESTADÍSTICAS DE NUEVAS COLUMNAS")
print(f"{'='*60}")

print(f"\nGroup:")
print(f"  Tipo: {df_new['Group'].dtype}")
print(f"  Valores únicos: {df_new['Group'].nunique()}")
print(f"  Mínimo: {df_new['Group'].min()}")
print(f"  Máximo: {df_new['Group'].max()}")
print(f"  Valores nulos: {df_new['Group'].isnull().sum()}")

print(f"\nNumInGroup:")
print(f"  Tipo: {df_new['NumInGroup'].dtype}")
print(f"  Valores únicos: {df_new['NumInGroup'].nunique()}")
print(f"  Mínimo: {df_new['NumInGroup'].min()}")
print(f"  Máximo: {df_new['NumInGroup'].max()}")
print(f"  Valores nulos: {df_new['NumInGroup'].isnull().sum()}")

# Distribución de tamaño de grupos
print(f"\nDistribución de tamaño de grupos:")
group_sizes = df_new.groupby('Group').size()
print(f"  Grupos con 1 persona: {(group_sizes == 1).sum()}")
print(f"  Grupos con 2 personas: {(group_sizes == 2).sum()}")
print(f"  Grupos con 3 personas: {(group_sizes == 3).sum()}")
print(f"  Grupos con 4+ personas: {(group_sizes >= 4).sum()}")
print(f"  Tamaño máximo de grupo: {group_sizes.max()}")

# Mostrar ejemplos de transformación
print(f"\n{'='*60}")
print("EJEMPLOS DE TRANSFORMACIÓN")
print(f"{'='*60}")
print("\nPrimeras 10 filas:")
print(df_new[['Group', 'NumInGroup', 'HomePlanet', 'Age', 'Name']].head(10))

# Mostrar primeras filas completas
print(f"\n{'='*60}")
print("PRIMERAS 5 FILAS DEL NUEVO DATASET")
print(f"{'='*60}")
print(df_new.head())

# Mostrar tipos de datos de todas las columnas
print(f"\n{'='*60}")
print("TIPOS DE DATOS DE TODAS LAS COLUMNAS")
print(f"{'='*60}")
for i, col in enumerate(df_new.columns, 1):
    dtype = df_new[col].dtype
    print(f"  {i:2d}. {col:15s} → {dtype}")

# Guardar nuevo CSV
output_file = 'train6.csv'
df_new.to_csv(output_file, index=False)

print(f"\n{'='*60}")
print(f"✓ Archivo guardado exitosamente: {output_file}")
print(f"{'='*60}")
print(f"\nPassengerId eliminado y separado en:")
print(f"  - Group: Número de grupo (entero)")
print(f"  - NumInGroup: Número dentro del grupo (entero)")
