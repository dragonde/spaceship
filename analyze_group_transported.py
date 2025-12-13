"""
Script para analizar si los miembros del mismo grupo
tienen el mismo valor de Transported
"""

import pandas as pd
import numpy as np

# Cargar datos
print("Cargando train7.csv...")
df = pd.read_csv('train7.csv')

print(f"\nDimensiones: {df.shape}")
print(f"Total de pasajeros: {len(df)}")
print(f"Total de grupos: {df['Group'].nunique()}")

print(f"\n{'='*80}")
print("ANÁLISIS: ¿Los miembros del mismo grupo tienen el mismo 'Transported'?")
print(f"{'='*80}")

# Analizar solo grupos con más de 1 persona
groups_with_multiple = df[df['GroupSize'] > 1]
print(f"\nPasajeros en grupos de 2+ personas: {len(groups_with_multiple)}")
print(f"Número de grupos de 2+ personas: {groups_with_multiple['Group'].nunique()}")

# Para cada grupo, contar cuántos valores únicos de Transported hay
group_analysis = df.groupby('Group').agg({
    'Transported': lambda x: x.nunique(),  # Cuántos valores únicos
    'GroupSize': 'first'
}).reset_index()

group_analysis.columns = ['Group', 'UniqueTransportedValues', 'GroupSize']

# Clasificar grupos
group_analysis['AllSame'] = group_analysis['UniqueTransportedValues'] == 1

print(f"\n{'='*80}")
print("RESULTADOS GENERALES")
print(f"{'='*80}")

total_groups = len(group_analysis)
groups_all_same = group_analysis['AllSame'].sum()
groups_different = total_groups - groups_all_same

print(f"\nTotal de grupos: {total_groups}")
print(f"Grupos donde TODOS tienen el mismo Transported: {groups_all_same} ({groups_all_same/total_groups*100:.2f}%)")
print(f"Grupos donde hay valores DIFERENTES:           {groups_different} ({groups_different/total_groups*100:.2f}%)")

# Análisis solo para grupos con múltiples personas
multi_person_groups = group_analysis[group_analysis['GroupSize'] > 1]
total_multi = len(multi_person_groups)
multi_all_same = multi_person_groups['AllSame'].sum()
multi_different = total_multi - multi_all_same

print(f"\n{'='*80}")
print("RESULTADOS PARA GRUPOS DE 2+ PERSONAS")
print(f"{'='*80}")

print(f"\nTotal de grupos de 2+ personas: {total_multi}")
print(f"Grupos donde TODOS tienen el mismo Transported: {multi_all_same} ({multi_all_same/total_multi*100:.2f}%)")
print(f"Grupos donde hay valores DIFERENTES:           {multi_different} ({multi_different/total_multi*100:.2f}%)")

# Análisis por tamaño de grupo
print(f"\n{'='*80}")
print("ANÁLISIS POR TAMAÑO DE GRUPO")
print(f"{'='*80}")

for size in sorted(multi_person_groups['GroupSize'].unique()):
    groups_of_size = multi_person_groups[multi_person_groups['GroupSize'] == size]
    total = len(groups_of_size)
    same = groups_of_size['AllSame'].sum()
    different = total - same
    print(f"\nTamaño {size}:")
    print(f"  Total grupos:     {total}")
    print(f"  Todos iguales:    {same:4d} ({same/total*100:5.2f}%)")
    print(f"  Valores mixtos:   {different:4d} ({different/total*100:5.2f}%)")

# Identificar grupos con valores mixtos
mixed_groups = group_analysis[~group_analysis['AllSame'] & (group_analysis['GroupSize'] > 1)]

print(f"\n{'='*80}")
print("EJEMPLOS DE GRUPOS CON VALORES MIXTOS")
print(f"{'='*80}")

if len(mixed_groups) > 0:
    print(f"\nTotal de grupos con valores mixtos: {len(mixed_groups)}")
    print(f"\nPrimeros 5 grupos con valores mixtos:")

    for i, (idx, row) in enumerate(mixed_groups.head(5).iterrows()):
        group_id = row['Group']
        group_data = df[df['Group'] == group_id][['Group', 'NumInGroup', 'GroupSize', 'Name', 'Age', 'CryoSleep', 'Transported']]
        print(f"\nGrupo {group_id} (Tamaño: {row['GroupSize']}):")
        print(group_data.to_string(index=False))

        # Contar transportados y no transportados
        transported_count = group_data['Transported'].sum()
        not_transported_count = len(group_data) - transported_count
        print(f"  → {transported_count} transportados, {not_transported_count} NO transportados")

# Identificar grupos donde todos tienen el mismo valor
same_groups = group_analysis[group_analysis['AllSame'] & (group_analysis['GroupSize'] > 1)]

print(f"\n{'='*80}")
print("EJEMPLOS DE GRUPOS DONDE TODOS TIENEN EL MISMO VALOR")
print(f"{'='*80}")

if len(same_groups) > 0:
    print(f"\nTotal de grupos con mismo valor: {len(same_groups)}")
    print(f"\nPrimeros 5 grupos donde todos tienen el mismo valor:")

    for i, (idx, row) in enumerate(same_groups.head(5).iterrows()):
        group_id = row['Group']
        group_data = df[df['Group'] == group_id][['Group', 'NumInGroup', 'GroupSize', 'Name', 'Age', 'Transported']]
        print(f"\nGrupo {group_id} (Tamaño: {row['GroupSize']}):")
        print(group_data.to_string(index=False))

        # Mostrar si todos son True o False
        all_value = group_data['Transported'].iloc[0]
        print(f"  → Todos: {'TRANSPORTADOS' if all_value else 'NO TRANSPORTADOS'}")

# Conclusiones
print(f"\n{'='*80}")
print("CONCLUSIONES")
print(f"{'='*80}")

print(f"""
1. De los {total_multi} grupos de 2+ personas:
   - {multi_all_same} grupos ({multi_all_same/total_multi*100:.2f}%) tienen el MISMO valor de Transported
   - {multi_different} grupos ({multi_different/total_multi*100:.2f}%) tienen valores DIFERENTES

2. Esto sugiere que {'SÍ' if multi_all_same/total_multi > 0.7 else 'NO'} hay una fuerte tendencia a que los grupos
   sean transportados juntos (o no transportados juntos).

3. La variable 'Group' {'ES' if multi_all_same/total_multi > 0.7 else 'NO ES'} un predictor importante para el modelo.
""")

# Guardar análisis
print(f"\n{'='*80}")
print("GUARDANDO RESULTADOS")
print(f"{'='*80}")

# Crear un resumen por grupo
group_summary = df.groupby('Group').agg({
    'GroupSize': 'first',
    'Transported': lambda x: (x == True).sum(),  # Cuántos True
}).reset_index()

group_summary.columns = ['Group', 'GroupSize', 'TransportedCount']
group_summary['NotTransportedCount'] = group_summary['GroupSize'] - group_summary['TransportedCount']
group_summary['AllSameValue'] = (group_summary['TransportedCount'] == 0) | (group_summary['TransportedCount'] == group_summary['GroupSize'])

output_file = 'group_transported_analysis.csv'
group_summary.to_csv(output_file, index=False)
print(f"\n✓ Análisis detallado guardado en: {output_file}")

print(f"\nPrimeras 10 filas del análisis:")
print(group_summary.head(10))
