import pandas as pd
import numpy as np

# Cargar datos
df = pd.read_csv('train9.csv')

print("="*80)
print("CREANDO COLUMNA SpendingPercentil")
print("="*80)

# Inicializar la columna SpendingPercentil con 0
df['SpendingPercentil'] = 0.0

# Filtrar registros con HasExpenses = True (1)
mask_has_expenses = df['HasExpenses'] == 1

print(f"\nRegistros totales: {len(df)}")
print(f"Registros con HasExpenses = False: {(~mask_has_expenses).sum()}")
print(f"Registros con HasExpenses = True: {mask_has_expenses.sum()}")

# Calcular el percentil solo para los que tienen gastos
# pct=True devuelve valores entre 0 y 1
# method='average' maneja empates tomando el promedio de sus rangos
if mask_has_expenses.sum() > 0:
    df.loc[mask_has_expenses, 'SpendingPercentil'] = df.loc[mask_has_expenses, 'TotalExpenses'].rank(pct=True, method='average')

# Verificar el rango de valores
print(f"\nSpendingPercentil - Rango: [{df['SpendingPercentil'].min():.4f}, {df['SpendingPercentil'].max():.4f}]")
print(f"SpendingPercentil - Media: {df['SpendingPercentil'].mean():.4f}")
print(f"SpendingPercentil - Mediana: {df['SpendingPercentil'].median():.4f}")

# Guardar a train10.csv
df.to_csv('train10.csv', index=False)

print(f"\n✓ Archivo generado: train10.csv")
print(f"✓ Total de columnas: {len(df.columns)}")
print(f"✓ Nueva columna 'SpendingPercentil' agregada en posición {df.columns.get_loc('SpendingPercentil') + 1}")

# Mostrar algunas estadísticas
print("\n" + "="*80)
print("ESTADÍSTICAS DE SpendingPercentil")
print("="*80)

print("\nPara HasExpenses = False:")
no_expenses = df[df['HasExpenses'] == 0]['SpendingPercentil']
print(f"  Cantidad: {len(no_expenses)}")
print(f"  Valores únicos: {no_expenses.unique()}")

print("\nPara HasExpenses = True:")
with_expenses = df[df['HasExpenses'] == 1][['TotalExpenses', 'SpendingPercentil']]
print(f"  Cantidad: {len(with_expenses)}")
print(f"  SpendingPercentil mínimo: {with_expenses['SpendingPercentil'].min():.6f}")
print(f"  SpendingPercentil máximo: {with_expenses['SpendingPercentil'].max():.6f}")

print("\nEjemplos de registros con HasExpenses = True:")
print(with_expenses.sort_values('TotalExpenses').head(5).to_string(index=False))
print("\n...")
print(with_expenses.sort_values('TotalExpenses').tail(5).to_string(index=False))

print("\n" + "="*80)
