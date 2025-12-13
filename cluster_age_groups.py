"""
Clustering de grupos de edad (Age) - Spaceship Titanic

Objetivo:
- Segmentar a los pasajeros en "grupos de edad" usando técnicas de clustering (no bins manuales).
- Evaluar varios k (2..max_k) con:
  - Silhouette score (mayor es mejor)
  - Inercia (elbow; menor es mejor, útil para inspección)

Entradas:
- train9.csv (por defecto)

Salidas:
- train9_with_age_clusters.csv: dataset original + columnas:
    - AgeImputed: edad imputada (si Age es nulo)
    - AgeCluster: id de clúster ordenado por centro (0 = más joven)
    - AgeClusterLabel: etiqueta legible (p.ej. "C0_0-12")
- age_cluster_summary.csv: resumen por clúster (tamaño, rango, media, etc.)
- age_cluster_transported_rate.csv: tasa de Transported por clúster (si existe la columna)
- plots/age_clustering_elbow.png
- plots/age_clustering_silhouette.png
- plots/age_clusters_distribution.png
- plots/age_clusters_transported_rate.png (si existe la columna)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clustering para segmentar grupos de edad (Age).")
    p.add_argument(
        "--input",
        default="train9.csv",
        help="Ruta del CSV de entrada (por defecto: train9.csv).",
    )
    p.add_argument(
        "--output",
        default="train9_with_age_clusters.csv",
        help="Ruta del CSV de salida con clústeres.",
    )
    p.add_argument(
        "--summary",
        default="age_cluster_summary.csv",
        help="Ruta del CSV de resumen por clúster.",
    )
    p.add_argument(
        "--plots-dir",
        default="plots",
        help="Directorio donde guardar las gráficas.",
    )
    p.add_argument(
        "--max-k",
        type=int,
        default=10,
        help="Máximo número de clústeres a evaluar (mínimo real: 2).",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad.",
    )
    return p.parse_args()


def _prepare_age(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve:
    - age_raw: vector con NaNs (float)
    - age_imputed: vector sin NaNs (float)
    """
    if "Age" not in df.columns:
        raise ValueError("No existe la columna 'Age' en el dataset.")

    age_raw = df["Age"].astype(float).to_numpy()
    age_imputed = age_raw.copy()
    median_age = float(np.nanmedian(age_imputed))
    age_imputed[np.isnan(age_imputed)] = median_age
    return age_raw, age_imputed


def _evaluate_kmeans(age_imputed: np.ndarray, max_k: int, random_state: int) -> pd.DataFrame:
    """
    Evalúa KMeans en 1D sobre Age (estandarizado) para k en [2..max_k].
    """
    x = age_imputed.reshape(-1, 1)
    x_scaled = StandardScaler().fit_transform(x)

    rows = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = km.fit_predict(x_scaled)
        sil = silhouette_score(x_scaled, labels)
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(sil),
            }
        )
    return pd.DataFrame(rows).sort_values("k")


def _select_best_k(metrics: pd.DataFrame) -> int:
    """
    Selección automática por silhouette (máximo).
    En empates, elige el k más pequeño (más parsimonioso).
    """
    best_sil = metrics["silhouette"].max()
    best = metrics[metrics["silhouette"] == best_sil].sort_values("k").iloc[0]
    return int(best["k"])


def _fit_kmeans(age_imputed: np.ndarray, k: int, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ajusta KMeans en 1D (Age estandarizado) y devuelve:
    - labels: etiquetas originales de KMeans
    - centers_age: centros en escala de Age (no estandarizada)
    """
    x = age_imputed.reshape(-1, 1)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    km = KMeans(n_clusters=k, n_init=50, random_state=random_state)
    labels = km.fit_predict(x_scaled)
    centers_scaled = km.cluster_centers_
    centers_age = scaler.inverse_transform(centers_scaled).reshape(-1)
    return labels, centers_age


def _relabel_by_center(labels: np.ndarray, centers_age: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Re-etiqueta clústeres para que 0 sea el centro más joven, 1 el siguiente, etc.
    """
    order = np.argsort(centers_age)
    mapping = {int(old): int(new) for new, old in enumerate(order)}
    new_labels = np.vectorize(lambda x: mapping[int(x)])(labels)
    return new_labels.astype(int), mapping


def _make_cluster_labels(age_imputed: np.ndarray, age_cluster: np.ndarray) -> pd.Series:
    """
    Etiqueta legible por clúster, usando rango observado en el dataset imputado.
    Formato: C{cluster}_{min}-{max}
    """
    s = pd.Series(age_imputed)
    out = []
    for c in age_cluster:
        out.append(c)
    out = pd.Series(out, index=s.index, dtype=int)

    labels = {}
    for c in sorted(out.unique()):
        ages = s[out == c]
        a_min = int(np.floor(ages.min()))
        a_max = int(np.ceil(ages.max()))
        labels[int(c)] = f"C{int(c)}_{a_min}-{a_max}"
    return out.map(labels)


def _write_plots(metrics: pd.DataFrame, df_out: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Elbow (inercia)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics["k"], metrics["inertia"], marker="o")
    ax.set_title("KMeans 1D sobre Age - Elbow (Inercia)")
    ax.set_xlabel("k (número de clústeres)")
    ax.set_ylabel("Inercia")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "age_clustering_elbow.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    # Silhouette
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics["k"], metrics["silhouette"], marker="o")
    ax.set_title("KMeans 1D sobre Age - Silhouette")
    ax.set_xlabel("k (número de clústeres)")
    ax.set_ylabel("Silhouette score")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "age_clustering_silhouette.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    # Distribución de edades por clúster
    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot = df_out.copy()
    df_plot = df_plot.sort_values("AgeCluster")
    for c in sorted(df_plot["AgeCluster"].unique()):
        ages = df_plot.loc[df_plot["AgeCluster"] == c, "AgeImputed"].to_numpy()
        ax.hist(ages, bins=20, alpha=0.55, label=f"C{c}")
    ax.set_title("Distribución de Age (imputada) por clúster")
    ax.set_xlabel("Edad")
    ax.set_ylabel("Frecuencia")
    ax.legend(title="Clúster", ncols=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "age_clusters_distribution.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def _transported_rate_by_cluster(df_out: pd.DataFrame) -> pd.DataFrame | None:
    """
    Si existe la columna Transported (bool o 0/1), devuelve tabla con:
    - Total, TransportedTrue, TransportedFalse, TransportedRate
    Caso contrario, devuelve None.
    """
    if "Transported" not in df_out.columns:
        return None

    t = df_out["Transported"]
    if t.dtype == bool:
        transported_01 = t.astype(int)
    else:
        # soportar 0/1 o strings "True"/"False" de forma defensiva
        if t.dtype == object:
            transported_01 = t.map({"True": 1, "False": 0, True: 1, False: 0}).astype("Int64")
        else:
            transported_01 = pd.to_numeric(t, errors="coerce").astype("Int64")

    tmp = df_out.copy()
    tmp["_Transported01"] = transported_01

    rate = (
        tmp.groupby(["AgeCluster", "AgeClusterLabel"], as_index=False)
        .agg(
            Total=("AgeCluster", "size"),
            TransportedTrue=("_Transported01", lambda s: int((s == 1).sum())),
            TransportedFalse=("_Transported01", lambda s: int((s == 0).sum())),
            MissingTransported=("_Transported01", lambda s: int(pd.isna(s).sum())),
        )
        .sort_values("AgeCluster")
    )
    rate["TransportedRate"] = rate["TransportedTrue"] / rate["Total"]
    return rate


def _write_transported_rate_plot(rate_df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = rate_df["AgeClusterLabel"].astype(str).to_list()
    y = (rate_df["TransportedRate"] * 100).to_numpy()
    ax.bar(x, y, color=["#4ECDC4" for _ in x])
    ax.set_title("Tasa de Transported por AgeCluster")
    ax.set_xlabel("AgeCluster")
    ax.set_ylabel("Transported (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    for i, val in enumerate(y):
        ax.text(i, val + 1.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(plots_dir / "age_clusters_transported_rate.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")

    df = pd.read_csv(input_path)
    age_raw, age_imputed = _prepare_age(df)

    max_k = int(args.max_k)
    if max_k < 2:
        raise ValueError("--max-k debe ser >= 2")

    metrics = _evaluate_kmeans(age_imputed, max_k=max_k, random_state=int(args.random_state))
    best_k = _select_best_k(metrics)

    labels, centers_age = _fit_kmeans(age_imputed, k=best_k, random_state=int(args.random_state))
    age_cluster, mapping = _relabel_by_center(labels, centers_age)

    df_out = df.copy()
    df_out["AgeImputed"] = age_imputed
    df_out["AgeCluster"] = age_cluster
    df_out["AgeClusterLabel"] = _make_cluster_labels(age_imputed, age_cluster)

    # Resumen por clúster
    summary = (
        df_out.groupby("AgeCluster", as_index=False)
        .agg(
            ClusterLabel=("AgeClusterLabel", "first"),
            Count=("AgeCluster", "size"),
            AgeMin=("AgeImputed", "min"),
            AgeMax=("AgeImputed", "max"),
            AgeMean=("AgeImputed", "mean"),
            AgeMedian=("AgeImputed", "median"),
            AgeStd=("AgeImputed", "std"),
            MissingAge=("Age", lambda s: int(pd.isna(s).sum())),
        )
        .sort_values("AgeCluster")
    )

    # Guardar archivos
    df_out.to_csv(args.output, index=False)
    summary.to_csv(args.summary, index=False)

    # Gráficas
    _write_plots(metrics, df_out, Path(args.plots_dir))

    # Tasa de Transported por clúster (si aplica)
    rate_df = _transported_rate_by_cluster(df_out)
    if rate_df is not None:
        transported_rate_path = Path(args.summary).with_name("age_cluster_transported_rate.csv")
        rate_df.to_csv(transported_rate_path, index=False)
        _write_transported_rate_plot(rate_df, Path(args.plots_dir))

    # Log final (simple)
    print("✓ Clustering de edad completado")
    print(f"  - input:   {input_path}")
    print(f"  - best_k:  {best_k}")
    print(f"  - output:  {args.output}")
    print(f"  - summary: {args.summary}")
    print(f"  - plots:   {args.plots_dir}/age_clustering_*.png y {args.plots_dir}/age_clusters_distribution.png")
    if rate_df is not None:
        print(f"  - transported_rate: {transported_rate_path}")
        print(f"  - transported_plot: {args.plots_dir}/age_clusters_transported_rate.png")
    print("\nMétricas (k, silhouette):")
    print(metrics[["k", "silhouette"]].to_string(index=False))


if __name__ == "__main__":
    main()


