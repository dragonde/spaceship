"""
Script para analizar si las personas con el mismo apellido
tienen el mismo valor de Transported
"""

import pandas as pd
import numpy as np

# Cargar datos
print("Cargando train8.csv...")
df = pd.read_csv('train8.csv')

print(f"\nDimensiones: {df.shape}")
print(f"Total de pasajeros: {len(df)}")
print(f"Total de apellidos únicos: {df['Surname'].nunique()}")

print(f"\n{'='*80}")
print("ANÁLISIS: ¿Las personas con el mismo apellido tienen el mismo 'Transported'?")
print(f"{'='*80}")

# Analizar solo apellidos con múltiples personas
surnames_with_multiple = df[df['Surname'].notna()].groupby('Surname').size()
surnames_multi = surnames_with_multiple[surnames_with_multiple > 1]

print(f"\nPasajeros con apellido compartido: {df[df['Surname'].isin(surnames_multi.index)].shape[0]}")
print(f"Número de apellidos compartidos: {len(surnames_multi)}")

# Para cada apellido, contar cuántos valores únicos de Transported hay
surname_analysis = df[df['Surname'].notna()].groupby('Surname').agg({
    'Transported': lambda x: x.nunique(),  # Cuántos valores únicos
    'Surname': 'count'  # Cuántas personas
}).reset_index(drop=True)

surname_analysis.columns = ['UniqueTransportedValues', 'FamilySize']

# Clasificar apellidos
surname_analysis['AllSame'] = surname_analysis['UniqueTransportedValues'] == 1

print(f"\n{'='*80}")
print("RESULTADOS GENERALES")
print(f"{'='*80}")

total_surnames = len(surname_analysis)
surnames_all_same = surname_analysis['AllSame'].sum()
surnames_different = total_surnames - surnames_all_same

print(f"\nTotal de apellidos: {total_surnames}")
print(f"Apellidos donde TODOS tienen el mismo Transported: {surnames_all_same} ({surnames_all_same/total_surnames*100:.2f}%)")
print(f"Apellidos donde hay valores DIFERENTES:           {surnames_different} ({surnames_different/total_surnames*100:.2f}%)")

# Análisis solo para apellidos con múltiples personas
multi_person_surnames = surname_analysis[surname_analysis['FamilySize'] > 1]
total_multi = len(multi_person_surnames)
multi_all_same = multi_person_surnames['AllSame'].sum()
multi_different = total_multi - multi_all_same

print(f"\n{'='*80}")
print("RESULTADOS PARA APELLIDOS COMPARTIDOS (2+ PERSONAS)")
print(f"{'='*80}")

print(f"\nTotal de apellidos compartidos: {total_multi}")
print(f"Apellidos donde TODOS tienen el mismo Transported: {multi_all_same} ({multi_all_same/total_multi*100:.2f}%)")
print(f"Apellidos donde hay valores DIFERENTES:           {multi_different} ({multi_different/total_multi*100:.2f}%)")

# Análisis por tamaño de familia
print(f"\n{'='*80}")
print("ANÁLISIS POR TAMAÑO DE FAMILIA (APELLIDO)")
print(f"{'='*80}")

for size in sorted(multi_person_surnames['FamilySize'].unique()):
    if size > 1 and size <= 20:  # Limitamos a tamaños razonables
        families_of_size = multi_person_surnames[multi_person_surnames['FamilySize'] == size]
        if len(families_of_size) > 0:
            total = len(families_of_size)
            same = families_of_size['AllSame'].sum()
            different = total - same
            print(f"\nTamaño {size:2d}:")
            print(f"  Total apellidos:  {total:4d}")
            print(f"  Todos iguales:    {same:4d} ({same/total*100:5.2f}%)")
            print(f"  Valores mixtos:   {different:4d} ({different/total*100:5.2f}%)")

# Obtener información detallada de apellidos
surname_details = df[df['Surname'].notna()].groupby('Surname').agg({
    'Transported': [lambda x: x.nunique(), lambda x: (x == True).sum(), 'count'],
    'Group': lambda x: x.nunique()  # Cuántos grupos diferentes
}).reset_index()

surname_details.columns = ['Surname', 'UniqueTransported', 'TransportedCount', 'TotalPeople', 'UniqueGroups']
surname_details['AllSameTransported'] = surname_details['UniqueTransported'] == 1
surname_details['NotTransportedCount'] = surname_details['TotalPeople'] - surname_details['TransportedCount']

# Identificar apellidos con valores mixtos
mixed_surnames = surname_details[(~surname_details['AllSameTransported']) & (surname_details['TotalPeople'] > 1)]
mixed_surnames = mixed_surnames.sort_values('TotalPeople', ascending=False)

print(f"\n{'='*80}")
print("EJEMPLOS DE APELLIDOS CON VALORES MIXTOS")
print(f"{'='*80}")

if len(mixed_surnames) > 0:
    print(f"\nTotal de apellidos con valores mixtos: {len(mixed_surnames)}")
    print(f"\nPrimeros 5 apellidos con valores mixtos:")

    for i, (idx, row) in enumerate(mixed_surnames.head(5).iterrows()):
        surname = row['Surname']
        family_data = df[df['Surname'] == surname][['Group', 'NumInGroup', 'GroupSize', 'Surname', 'Age', 'HomePlanet', 'Transported']].sort_values('Group')
        print(f"\nApellido: {surname} ({row['TotalPeople']} personas, {row['UniqueGroups']} grupos)")
        print(family_data.to_string(index=False))
        print(f"  → {row['TransportedCount']} transportados, {row['NotTransportedCount']} NO transportados")

# Identificar apellidos donde todos tienen el mismo valor
same_surnames = surname_details[surname_details['AllSameTransported'] & (surname_details['TotalPeople'] > 1)]
same_surnames = same_surnames.sort_values('TotalPeople', ascending=False)

print(f"\n{'='*80}")
print("EJEMPLOS DE APELLIDOS DONDE TODOS TIENEN EL MISMO VALOR")
print(f"{'='*80}")

if len(same_surnames) > 0:
    print(f"\nTotal de apellidos con mismo valor: {len(same_surnames)}")
    print(f"\nPrimeros 5 apellidos donde todos tienen el mismo valor:")

    for i, (idx, row) in enumerate(same_surnames.head(5).iterrows()):
        surname = row['Surname']
        family_data = df[df['Surname'] == surname][['Group', 'NumInGroup', 'GroupSize', 'Surname', 'Age', 'Transported']].sort_values('Group')
        print(f"\nApellido: {surname} ({row['TotalPeople']} personas, {row['UniqueGroups']} grupos)")
        print(family_data.to_string(index=False))

        all_value = family_data['Transported'].iloc[0]
        print(f"  → Todos: {'TRANSPORTADOS' if all_value else 'NO TRANSPORTADOS'}")

# Comparación: Grupos vs Apellidos
print(f"\n{'='*80}")
print("COMPARACIÓN: GRUPOS vs APELLIDOS")
print(f"{'='*80}")

# Analizar cuántos apellidos comparten grupo
surname_group_relation = df[df['Surname'].notna()].groupby('Surname')['Group'].apply(
    lambda x: x.nunique() / len(x)  # Ratio: grupos únicos / total personas
).mean()

print(f"\nEn promedio, personas con el mismo apellido pertenecen a:")
print(f"  {surname_group_relation*100:.2f}% grupos diferentes (relativo a su tamaño)")

# Ver si hay apellidos que cruzan múltiples grupos
multi_group_surnames = surname_details[surname_details['UniqueGroups'] > 1].sort_values('UniqueGroups', ascending=False)
print(f"\nApellidos que aparecen en múltiples grupos: {len(multi_group_surnames)}")
if len(multi_group_surnames) > 0:
    print(f"\nTop 5 apellidos más distribuidos:")
    for idx, row in multi_group_surnames.head(5).iterrows():
        print(f"  {row['Surname']:20s}: {row['TotalPeople']} personas en {row['UniqueGroups']} grupos diferentes")

# Conclusiones
print(f"\n{'='*80}")
print("CONCLUSIONES")
print(f"{'='*80}")

print(f"""
1. De los {total_multi} apellidos compartidos (2+ personas):
   - {multi_all_same} apellidos ({multi_all_same/total_multi*100:.2f}%) tienen el MISMO valor de Transported
   - {multi_different} apellidos ({multi_different/total_multi*100:.2f}%) tienen valores DIFERENTES

2. Comparación con análisis de Grupos:
   - Grupos (2+ personas): 43.56% mismo valor
   - Apellidos (2+ personas): {multi_all_same/total_multi*100:.2f}% mismo valor
   - Diferencia: {abs(multi_all_same/total_multi*100 - 43.56):.2f} puntos porcentuales

3. Interpretación:
   - {'Apellidos tienen MAYOR' if multi_all_same/total_multi > 0.436 else 'Apellidos tienen MENOR'} consistencia que grupos de viaje
   - Esto {'sugiere' if multi_all_same/total_multi > 0.5 else 'NO sugiere'} que el apellido es un predictor importante

4. {len(multi_group_surnames)} apellidos aparecen en múltiples grupos
   - Esto indica que el apellido NO es equivalente a grupo de viaje
   - Familias pueden viajar en grupos separados
""")

# Guardar análisis
print(f"\n{'='*80}")
print("GUARDANDO RESULTADOS")
print(f"{'='*80}")

output_file = 'surname_transported_analysis.csv'
surname_details.to_csv(output_file, index=False)
print(f"\n✓ Análisis detallado guardado en: {output_file}")

print(f"\nPrimeras 10 filas del análisis:")
print(surname_details.head(10))
