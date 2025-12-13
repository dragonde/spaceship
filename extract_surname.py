"""
Script para extraer apellidos de la columna Name
Transforma Name en Surname (solo segunda palabra)
Genera train8.csv a partir de train7.csv
"""

import pandas as pd

# Cargar datos
print("Cargando train7.csv...")
df = pd.read_csv('train7.csv')

print(f"\nDimensiones: {df.shape}")
print(f"\nColumnas actuales: {list(df.columns)}")

# Mostrar ejemplos de Name antes de la transformación
print(f"\n{'='*60}")
print("EJEMPLOS DE NOMBRES ANTES DE LA TRANSFORMACIÓN")
print(f"{'='*60}")
print(df['Name'].head(20))

# Extraer apellido (segunda palabra)
print("\nExtrayendo apellidos...")

# Función para extraer el apellido
def extract_surname(name):
    if pd.isna(name):
        return None
    parts = str(name).split()
    if len(parts) >= 2:
        return parts[1]
    elif len(parts) == 1:
        return parts[0]  # Si solo hay una palabra, usar esa
    else:
        return None

df['Surname'] = df['Name'].apply(extract_surname)

# Eliminar columna Name original
df_new = df.drop(columns=['Name'])

# Reordenar columnas para poner Surname donde estaba Name
cols = df_new.columns.tolist()

# Remover Surname del final
cols.remove('Surname')

# Insertar en posición 11 (donde estaba Name)
# Posiciones: 0-Group, 1-NumInGroup, 2-GroupSize, 3-HomePlanet, 4-CryoSleep,
#            5-Deck, 6-Num, 7-Side, 8-Destination, 9-Age, 10-VIP, 11-Surname
cols.insert(11, 'Surname')

df_new = df_new[cols]

print(f"\nDimensiones nuevas: {df_new.shape}")

# Estadísticas de la columna Surname
print(f"\n{'='*60}")
print("ESTADÍSTICAS DE LA COLUMNA SURNAME")
print(f"{'='*60}")

print(f"\nTipo: {df_new['Surname'].dtype}")
print(f"Valores únicos: {df_new['Surname'].nunique()}")
print(f"Valores nulos: {df_new['Surname'].isnull().sum()}")

# Apellidos más frecuentes
print(f"\nApellidos más frecuentes:")
surname_counts = df_new['Surname'].value_counts().head(20)
for surname, count in surname_counts.items():
    print(f"  {surname:20s}: {count:3d} pasajeros")

# Mostrar ejemplos de transformación
print(f"\n{'='*60}")
print("EJEMPLOS DE TRANSFORMACIÓN")
print(f"{'='*60}")

# Crear tabla comparativa
comparison = pd.DataFrame({
    'Name_Original': df['Name'].head(20),
    'Surname_Nuevo': df_new['Surname'].head(20)
})
print(comparison.to_string(index=False))

# Verificar casos especiales
print(f"\n{'='*60}")
print("VERIFICACIÓN DE CASOS ESPECIALES")
print(f"{'='*60}")

# Nombres con una sola palabra
single_word_names = df[df['Name'].notna() & (df['Name'].str.split().str.len() == 1)]
if len(single_word_names) > 0:
    print(f"\nNombres con una sola palabra: {len(single_word_names)}")
    print(single_word_names[['Name']].head())
else:
    print("\nNo hay nombres con una sola palabra")

# Nombres con más de dos palabras
multi_word_names = df[df['Name'].notna() & (df['Name'].str.split().str.len() > 2)]
if len(multi_word_names) > 0:
    print(f"\nNombres con más de dos palabras: {len(multi_word_names)}")
    print("Primeros 5:")
    for idx, row in multi_word_names.head().iterrows():
        original = row['Name']
        surname = df_new.loc[idx, 'Surname']
        print(f"  '{original}' → '{surname}'")
else:
    print("\nNo hay nombres con más de dos palabras")

# Mostrar primeras filas completas
print(f"\n{'='*60}")
print("PRIMERAS 10 FILAS DEL NUEVO DATASET")
print(f"{'='*60}")
print(df_new[['Group', 'NumInGroup', 'GroupSize', 'HomePlanet', 'Age', 'Surname']].head(10))

# Analizar familias (mismo apellido)
print(f"\n{'='*60}")
print("ANÁLISIS DE FAMILIAS (MISMO APELLIDO)")
print(f"{'='*60}")

surname_groups = df_new.groupby('Surname').size().sort_values(ascending=False)
families = surname_groups[surname_groups > 1]
print(f"\nApellidos compartidos por múltiples pasajeros: {len(families)}")
print(f"\nFamilias más grandes (top 10):")
for surname, count in families.head(10).items():
    print(f"  {surname:20s}: {count:3d} pasajeros")

# Mostrar tipos de datos
print(f"\n{'='*60}")
print("TIPOS DE DATOS DE TODAS LAS COLUMNAS")
print(f"{'='*60}")
for i, col in enumerate(df_new.columns, 1):
    dtype = df_new[col].dtype
    marker = " ← MODIFICADA" if col == 'Surname' else ""
    print(f"  {i:2d}. {col:15s} → {dtype}{marker}")

# Guardar nuevo CSV
output_file = 'train8.csv'
df_new.to_csv(output_file, index=False)

print(f"\n{'='*60}")
print(f"✓ Archivo guardado exitosamente: {output_file}")
print(f"{'='*60}")
print(f"\nColumna 'Name' transformada en 'Surname'")
print(f"Se extrajo la segunda palabra de cada nombre")
print(f"Dimensiones: {df_new.shape[0]} filas × {df_new.shape[1]} columnas")
