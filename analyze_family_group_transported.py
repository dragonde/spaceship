"""
Script para analizar si las familias nucleares (mismo apellido + mismo grupo)
tienen el mismo valor de Transported
"""

import pandas as pd
import numpy as np

# Cargar datos
print("Cargando train9.csv...")
df = pd.read_csv('train9.csv')

print(f"\nDimensiones: {df.shape}")
print(f"Total de pasajeros: {len(df)}")
print(f"Total de Surname_Group únicos: {df['Surname'].nunique()}")

print(f"\n{'='*80}")
print("ANÁLISIS: ¿Familias nucleares (mismo apellido + grupo) transportadas juntas?")
print(f"{'='*80}")

# Analizar solo Surname_Group con múltiples personas (familias nucleares)
surname_counts = df[df['Surname'].notna()].groupby('Surname').size()
families_nuclear = surname_counts[surname_counts > 1]

print(f"\nPasajeros en familias nucleares (2+ personas): {df[df['Surname'].isin(families_nuclear.index)].shape[0]}")
print(f"Número de familias nucleares: {len(families_nuclear)}")

# Para cada familia nuclear, contar cuántos valores únicos de Transported hay
family_analysis = df[df['Surname'].notna()].groupby('Surname').agg({
    'Transported': lambda x: x.nunique(),  # Cuántos valores únicos
    'Group': ['first', 'count']  # Grupo y cuántas personas
}).reset_index()

family_analysis.columns = ['Surname_Group', 'UniqueTransportedValues', 'Group', 'FamilySize']

# Clasificar familias
family_analysis['AllSame'] = family_analysis['UniqueTransportedValues'] == 1

print(f"\n{'='*80}")
print("RESULTADOS GENERALES")
print(f"{'='*80}")

total_families = len(family_analysis)
families_all_same = family_analysis['AllSame'].sum()
families_different = total_families - families_all_same

print(f"\nTotal de Surname_Group: {total_families}")
print(f"Donde TODOS tienen el mismo Transported: {families_all_same} ({families_all_same/total_families*100:.2f}%)")
print(f"Donde hay valores DIFERENTES:           {families_different} ({families_different/total_families*100:.2f}%)")

# Análisis solo para familias nucleares (múltiples personas)
multi_person_families = family_analysis[family_analysis['FamilySize'] > 1]
total_multi = len(multi_person_families)
multi_all_same = multi_person_families['AllSame'].sum()
multi_different = total_multi - multi_all_same

print(f"\n{'='*80}")
print("RESULTADOS PARA FAMILIAS NUCLEARES (2+ PERSONAS)")
print(f"{'='*80}")

print(f"\nTotal de familias nucleares: {total_multi}")
print(f"Familias donde TODOS tienen el mismo Transported: {multi_all_same} ({multi_all_same/total_multi*100:.2f}%)")
print(f"Familias donde hay valores DIFERENTES:           {multi_different} ({multi_different/total_multi*100:.2f}%)")

# Análisis por tamaño de familia
print(f"\n{'='*80}")
print("ANÁLISIS POR TAMAÑO DE FAMILIA NUCLEAR")
print(f"{'='*80}")

for size in sorted(multi_person_families['FamilySize'].unique()):
    if size > 1:
        families_of_size = multi_person_families[multi_person_families['FamilySize'] == size]
        total = len(families_of_size)
        same = families_of_size['AllSame'].sum()
        different = total - same
        print(f"\nTamaño {size}:")
        print(f"  Total familias:   {total:4d}")
        print(f"  Todos iguales:    {same:4d} ({same/total*100:5.2f}%)")
        print(f"  Valores mixtos:   {different:4d} ({different/total*100:5.2f}%)")

# Obtener información detallada
family_details = df[df['Surname'].notna()].groupby('Surname').agg({
    'Transported': [lambda x: x.nunique(), lambda x: (x == True).sum(), 'count'],
    'Group': 'first'
}).reset_index()

family_details.columns = ['Surname_Group', 'UniqueTransported', 'TransportedCount', 'TotalPeople', 'Group']
family_details['AllSameTransported'] = family_details['UniqueTransported'] == 1
family_details['NotTransportedCount'] = family_details['TotalPeople'] - family_details['TransportedCount']

# Identificar familias con valores mixtos
mixed_families = family_details[(~family_details['AllSameTransported']) & (family_details['TotalPeople'] > 1)]
mixed_families = mixed_families.sort_values('TotalPeople', ascending=False)

print(f"\n{'='*80}")
print("EJEMPLOS DE FAMILIAS NUCLEARES CON VALORES MIXTOS")
print(f"{'='*80}")

if len(mixed_families) > 0:
    print(f"\nTotal de familias con valores mixtos: {len(mixed_families)}")
    print(f"\nPrimeros 5 familias con valores mixtos:")

    for i, (idx, row) in enumerate(mixed_families.head(5).iterrows()):
        surname_group = row['Surname_Group']
        family_data = df[df['Surname'] == surname_group][['Group', 'NumInGroup', 'GroupSize', 'Surname', 'Age', 'CryoSleep', 'Transported']].sort_values('NumInGroup')
        print(f"\nFamilia: {surname_group} ({row['TotalPeople']} personas)")
        print(family_data.to_string(index=False))
        print(f"  → {row['TransportedCount']} transportados, {row['NotTransportedCount']} NO transportados")
else:
    print("\n¡No hay familias nucleares con valores mixtos!")

# Identificar familias donde todos tienen el mismo valor
same_families = family_details[family_details['AllSameTransported'] & (family_details['TotalPeople'] > 1)]
same_families = same_families.sort_values('TotalPeople', ascending=False)

print(f"\n{'='*80}")
print("EJEMPLOS DE FAMILIAS NUCLEARES DONDE TODOS TIENEN EL MISMO VALOR")
print(f"{'='*80}")

if len(same_families) > 0:
    print(f"\nTotal de familias con mismo valor: {len(same_families)}")
    print(f"\nPrimeros 5 familias donde todos tienen el mismo valor:")

    for i, (idx, row) in enumerate(same_families.head(5).iterrows()):
        surname_group = row['Surname_Group']
        family_data = df[df['Surname'] == surname_group][['Group', 'NumInGroup', 'GroupSize', 'Surname', 'Age', 'Transported']].sort_values('NumInGroup')
        print(f"\nFamilia: {surname_group} ({row['TotalPeople']} personas)")
        print(family_data.to_string(index=False))

        all_value = family_data['Transported'].iloc[0]
        print(f"  → Todos: {'TRANSPORTADOS' if all_value else 'NO TRANSPORTADOS'}")

# Comparación con análisis anteriores
print(f"\n{'='*80}")
print("COMPARACIÓN: GRUPOS vs APELLIDOS vs FAMILIAS NUCLEARES")
print(f"{'='*80}")

print(f"""
Consistencia de Transported (% mismo valor para 2+ personas):

1. GRUPOS (mismo Group):                      43.56%
2. APELLIDOS (mismo Surname original):        23.43%
3. FAMILIAS NUCLEARES (Surname + Group):      {multi_all_same/total_multi*100:.2f}%

Diferencias:
  - Familias Nucleares vs Grupos:    {multi_all_same/total_multi*100 - 43.56:+.2f} puntos
  - Familias Nucleares vs Apellidos: {multi_all_same/total_multi*100 - 23.43:+.2f} puntos
""")

# Conclusiones
print(f"\n{'='*80}")
print("CONCLUSIONES")
print(f"{'='*80}")

print(f"""
1. De las {total_multi} familias nucleares (mismo apellido + mismo grupo):
   - {multi_all_same} familias ({multi_all_same/total_multi*100:.2f}%) tienen el MISMO valor de Transported
   - {multi_different} familias ({multi_different/total_multi*100:.2f}%) tienen valores DIFERENTES

2. Interpretación:
   - Familias nucleares {'SÍ' if multi_all_same/total_multi > 0.7 else 'NO'} tienen alta consistencia
   - Estar en el mismo apellido Y mismo grupo {'aumenta' if multi_all_same/total_multi > 43.56 else 'NO aumenta'} la consistencia
   - La consistencia es {'mayor' if multi_all_same/total_multi > 43.56 else 'menor'} que solo compartir grupo

3. Implicación para modelado:
   - Surname_Group {'ES' if multi_all_same/total_multi > 0.7 else 'NO ES'} un predictor fuerte
   - {'Considerar' if multi_all_same/total_multi > 0.7 else 'NO considerar'} features de familia nuclear
   - El transporte {'sigue siendo' if multi_all_same/total_multi < 0.7 else 'NO es'} principalmente individual
""")

# Guardar análisis
print(f"\n{'='*80}")
print("GUARDANDO RESULTADOS")
print(f"{'='*80}")

output_file = 'family_nuclear_transported_analysis.csv'
family_details.to_csv(output_file, index=False)
print(f"\n✓ Análisis detallado guardado en: {output_file}")

print(f"\nPrimeras 10 filas del análisis:")
print(family_details.head(10))
