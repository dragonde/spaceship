"""
Script para modificar Surname agregando el número de grupo
Formato: Apellido_Grupo (ej: Upead_16)
Genera train9.csv a partir de train8.csv
"""

import pandas as pd

# Cargar datos
print("Cargando train8.csv...")
df = pd.read_csv('train8.csv')

print(f"\nDimensiones: {df.shape}")
print(f"\nColumnas actuales: {list(df.columns)}")

# Mostrar ejemplos antes de la transformación
print(f"\n{'='*60}")
print("EJEMPLOS ANTES DE LA TRANSFORMACIÓN")
print(f"{'='*60}")
print("\nPrimeras 10 filas:")
print(df[['Group', 'Surname']].head(10))

# Crear nueva columna Surname con formato Apellido_Grupo
print("\nModificando Surname para incluir número de grupo...")

# Función para crear el nuevo formato
def create_surname_group(row):
    if pd.isna(row['Surname']):
        return None
    return f"{row['Surname']}_{row['Group']}"

df['Surname'] = df.apply(create_surname_group, axis=1)

print(f"\nTransformación completada")

# Estadísticas de la nueva columna Surname
print(f"\n{'='*60}")
print("ESTADÍSTICAS DE LA COLUMNA SURNAME MODIFICADA")
print(f"{'='*60}")

print(f"\nTipo: {df['Surname'].dtype}")
print(f"Valores únicos: {df['Surname'].nunique()}")
print(f"Valores nulos: {df['Surname'].isnull().sum()}")

# Mostrar ejemplos de transformación
print(f"\n{'='*60}")
print("EJEMPLOS DESPUÉS DE LA TRANSFORMACIÓN")
print(f"{'='*60}")
print("\nPrimeras 20 filas:")
print(df[['Group', 'NumInGroup', 'GroupSize', 'Surname']].head(20))

# Analizar grupos del mismo apellido original
print(f"\n{'='*60}")
print("ANÁLISIS: FAMILIAS EN DIFERENTES GRUPOS")
print(f"{'='*60}")

# Extraer apellido original (antes del underscore) para análisis
df['SurnameOnly'] = df['Surname'].str.split('_').str[0]

# Contar cuántos grupos diferentes por apellido original
surname_group_counts = df[df['SurnameOnly'].notna()].groupby('SurnameOnly')['Group'].nunique()
multi_group_surnames = surname_group_counts[surname_group_counts > 1].sort_values(ascending=False)

print(f"\nApellidos que aparecen en múltiples grupos: {len(multi_group_surnames)}")
print(f"\nTop 10 apellidos más distribuidos:")
for surname, group_count in multi_group_surnames.head(10).items():
    total_people = (df['SurnameOnly'] == surname).sum()
    print(f"  {surname:20s}: {total_people:2d} personas en {group_count:2d} grupos")

    # Mostrar los grupos específicos
    groups = df[df['SurnameOnly'] == surname]['Surname'].unique()[:5]  # Primeros 5
    print(f"    Ejemplos: {', '.join(groups)}")

# Verificar unicidad de Surname modificado
print(f"\n{'='*60}")
print("VERIFICACIÓN DE DUPLICADOS")
print(f"{'='*60}")

# Contar duplicados en la nueva columna Surname
surname_counts = df['Surname'].value_counts()
duplicates = surname_counts[surname_counts > 1]

if len(duplicates) > 0:
    print(f"\nSurname_Group con duplicados: {len(duplicates)}")
    print(f"\nPrimeros 5 Surname_Group duplicados:")
    for surname, count in duplicates.head().items():
        print(f"  {surname}: {count} veces")

    # Mostrar ejemplo de duplicado
    example = duplicates.index[0]
    print(f"\nEjemplo de duplicado '{example}':")
    print(df[df['Surname'] == example][['Group', 'NumInGroup', 'GroupSize', 'Surname', 'Age', 'HomePlanet']])
else:
    print("\n✓ No hay duplicados - cada Surname_Group es único dentro del mismo grupo")

# Comparación: Antes vs Después
print(f"\n{'='*60}")
print("COMPARACIÓN: VALORES ÚNICOS")
print(f"{'='*60}")

# Necesitamos cargar el original para comparar
df_original = pd.read_csv('train8.csv')
original_unique = df_original['Surname'].nunique()
new_unique = df['Surname'].nunique()

print(f"\nApellidos únicos ANTES (solo apellido): {original_unique}")
print(f"Surname_Group únicos DESPUÉS: {new_unique}")
print(f"Diferencia: +{new_unique - original_unique} identificadores únicos")

# Mostrar tipos de datos
print(f"\n{'='*60}")
print("TIPOS DE DATOS DE TODAS LAS COLUMNAS")
print(f"{'='*60}")
for i, col in enumerate(df.drop(columns=['SurnameOnly']).columns, 1):
    dtype = df[col].dtype
    marker = " ← MODIFICADA" if col == 'Surname' else ""
    print(f"  {i:2d}. {col:15s} → {dtype}{marker}")

# Eliminar columna temporal SurnameOnly
df_final = df.drop(columns=['SurnameOnly'])

# Mostrar primeras filas completas
print(f"\n{'='*60}")
print("PRIMERAS 10 FILAS DEL NUEVO DATASET")
print(f"{'='*60}")
print(df_final[['Group', 'NumInGroup', 'GroupSize', 'HomePlanet', 'Age', 'Surname']].head(10))

# Guardar nuevo CSV
output_file = 'train9.csv'
df_final.to_csv(output_file, index=False)

print(f"\n{'='*60}")
print(f"✓ Archivo guardado exitosamente: {output_file}")
print(f"{'='*60}")
print(f"\nColumna 'Surname' modificada con formato: Apellido_Grupo")
print(f"Ejemplos: Upead_16, Ofracculy_1, Vines_2")
print(f"Dimensiones: {df_final.shape[0]} filas × {df_final.shape[1]} columnas")
