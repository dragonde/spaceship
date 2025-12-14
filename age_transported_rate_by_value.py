"""
Tabla y gráfica: porcentaje de Transported por cada valor de Age.

Entrada:
- train9.csv (por defecto)

Salida:
- age_transported_rate_by_age.csv: tabla con columnas
    Age, Total, TransportedTrue, TransportedFalse, TransportedRate
- plots/age_transported_rate_by_age.png: gráfica (% Transported vs Edad)

Notas:
- Se excluyen filas con Age nulo (no se puede asignar a un valor exacto de edad).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Porcentaje de Transported por cada valor de Age.")
    p.add_argument("--input", default="train9.csv", help="CSV de entrada (por defecto: train9.csv).")
    p.add_argument(
        "--output",
        default="age_transported_rate_by_age.csv",
        help="CSV de salida con la tabla de tasas por edad.",
    )
    p.add_argument("--plots-dir", default="plots", help="Directorio para guardar la gráfica.")
    p.add_argument(
        "--min-n",
        type=int,
        default=10,
        help="Mínimo tamaño de muestra para anotar puntos (solo afecta a la gráfica).",
    )
    return p.parse_args()


def _to_transport01(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    if s.dtype == object:
        mapped = s.map({"True": 1, "False": 0, True: 1, False: 0})
        return mapped.astype("Int64")
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")

    df = pd.read_csv(input_path)
    if "Age" not in df.columns:
        raise ValueError("No existe la columna 'Age' en el dataset.")
    if "Transported" not in df.columns:
        raise ValueError("No existe la columna 'Transported' en el dataset (necesaria para el cálculo).")

    # Filtrar edades no nulas y convertir a entero (edades son 0..79 en el dataset)
    d = df.loc[~pd.isna(df["Age"]), ["Age", "Transported"]].copy()
    d["Age"] = pd.to_numeric(d["Age"], errors="coerce")
    d = d.loc[~pd.isna(d["Age"])].copy()
    d["Age"] = d["Age"].astype(int)
    d["_Transport01"] = _to_transport01(d["Transported"])

    table = (
        d.groupby("Age", as_index=False)
        .agg(
            Total=("Age", "size"),
            TransportedTrue=("_Transport01", lambda s: int((s == 1).sum())),
            TransportedFalse=("_Transport01", lambda s: int((s == 0).sum())),
            MissingTransported=("_Transport01", lambda s: int(pd.isna(s).sum())),
        )
        .sort_values("Age")
    )
    table["TransportedRate"] = table["TransportedTrue"] / table["Total"]

    # Guardar tabla
    table.to_csv(args.output, index=False)

    # Gráfica
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = table["Age"].to_numpy()
    y = (table["TransportedRate"] * 100).to_numpy()

    ax.plot(x, y, linewidth=2, color="#4ECDC4")
    ax.scatter(x, y, s=25, color="#4ECDC4", edgecolor="black", linewidth=0.3, alpha=0.9)

    # Añadir tamaño de muestra como anotación (solo donde hay suficiente n)
    min_n = int(args.min_n)
    for age, pct, n in zip(table["Age"], y, table["Total"]):
        if int(n) >= min_n and age % 5 == 0:  # para no saturar, anotamos cada 5 años
            ax.text(int(age), float(pct) + 1.2, f"n={int(n)}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Porcentaje de Transported por Edad (valor exacto)")
    ax.set_xlabel("Edad (Age)")
    ax.set_ylabel("Transported (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_plot = plots_dir / "age_transported_rate_by_age.png"
    fig.savefig(out_plot, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print("✓ Tabla y gráfica generadas")
    print(f"  - input:  {input_path}")
    print(f"  - output: {args.output}")
    print(f"  - plot:   {out_plot}")
    print(f"  - filas usadas (Age no nulo): {len(d)}")


if __name__ == "__main__":
    main()



