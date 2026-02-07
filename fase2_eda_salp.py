# ============================================================
# FASE 2 – LIMPIEZA Y ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# SISTEMA DE ALUMBRADO PÚBLICO – SALP
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ------------------------------------------------------------

sns.set(style="whitegrid")

ruta_datos = Path(
    r"C:\Users\arlin\OneDrive\Imágenes\SALP_Bucaramanga\data_raw"
)

ruta_salida = Path(
    r"C:\Users\arlin\OneDrive\Imágenes\SALP_Bucaramanga\data_clean"
)
ruta_salida.mkdir(exist_ok=True)

# ------------------------------------------------------------
# CARGA Y UNIFICACIÓN DE ARCHIVOS EXCEL
# ------------------------------------------------------------

archivos = list(ruta_datos.glob("*.xlsx"))
print(f"Archivos encontrados: {len(archivos)}")

dfs = []

for archivo in archivos:
    print(f"Procesando: {archivo.name}")

    df = pd.read_excel(archivo)

    # Normalizar nombres de columnas
    df.columns = df.columns.str.strip().str.lower()

    dfs.append(df)

df_total = pd.concat(dfs, ignore_index=True)

# ------------------------------------------------------------
# LIMPIEZA DE DATOS
# ------------------------------------------------------------

# Conversión de fechas
for col in ["fecha_lectura_ini", "fecha_lectura_fin"]:
    if col in df_total.columns:
        df_total[col] = pd.to_datetime(df_total[col], errors="coerce")

# Normalizar tarifa
if "tarifa_activa" in df_total.columns:
    df_total["tarifa_activa"] = (
        df_total["tarifa_activa"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

# Conversión a numérico
columnas_numericas = [
    "lectura_actual",
    "lectura_anterior",
    "consumo_act",
    "consumo_act_valor",
    "factor_medidor"
]

for col in columnas_numericas:
    if col in df_total.columns:
        df_total[col] = pd.to_numeric(df_total[col], errors="coerce")

# Eliminar consumos inválidos
df_total = df_total[df_total["consumo_act"] > 0]

# ------------------------------------------------------------
# VARIABLES TEMPORALES
# ------------------------------------------------------------

df_total["año"] = df_total["fecha_lectura_fin"].dt.year
df_total["mes"] = df_total["fecha_lectura_fin"].dt.month
df_total["mes_nombre"] = df_total["fecha_lectura_fin"].dt.month_name(locale="es_ES")

# ------------------------------------------------------------
# TENDENCIA MENSUAL POR AÑO
# ------------------------------------------------------------

consumo_mensual = (
    df_total
    .groupby(["año", "mes"])["consumo_act"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(10, 5))

for año in consumo_mensual["año"].unique():
    data = consumo_mensual[consumo_mensual["año"] == año]
    plt.plot(
        data["mes"],
        data["consumo_act"],
        marker="o",
        label=str(año)
    )

plt.title("Tendencia mensual del consumo energético del SALP")
plt.xlabel("Mes")
plt.ylabel("Consumo total (kWh)")
plt.xticks(range(1, 13))
plt.legend(title="Año")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# ESTACIONALIDAD MENSUAL (PROMEDIO MULTIANUAL)
# ------------------------------------------------------------

estacionalidad = (
    df_total
    .groupby("mes")["consumo_act"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(10, 5))

plt.plot(
    estacionalidad["mes"],
    estacionalidad["consumo_act"],
    marker="o",
    color="#4c72b0"
)

plt.title("Estacionalidad mensual del consumo energético (promedio 2022–2025)")
plt.xlabel("Mes")
plt.ylabel("Consumo promedio (kWh)")
plt.xticks(range(1, 13))
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# TOP 10 CUENTAS CON MAYOR CONSUMO
# ------------------------------------------------------------

top10_cuentas = (
    df_total
    .groupby("cuenta")["consumo_act"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))

top10_cuentas.plot(
    kind="bar",
    color="#4c72b0",
    edgecolor="black"
)

plt.title("Top 10 cuentas con mayor consumo energético acumulado")
plt.xlabel("Número de cuenta")
plt.ylabel("Consumo acumulado (kWh)")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# DISTRIBUCIÓN DEL CONSUMO – HISTOGRAMA (OPCIÓN B)
# ------------------------------------------------------------

consumo = df_total["consumo_act"].dropna()
consumo = consumo[consumo > 0]

plt.figure(figsize=(10, 5))

sns.histplot(
    consumo,
    bins=60,
    kde=True,
    log_scale=True,
    color="#4c72b0"
)

plt.title("Distribución del consumo energético del SALP")
plt.xlabel("Consumo energético (kWh) [escala logarítmica]")
plt.ylabel("Cantidad de registros")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# ORGANIZACIÓN FINAL DEL DATASET (AÑO Y MES)
# ------------------------------------------------------------

columnas_ordenadas = [
    "cuenta",
    "nombre",
    "año",
    "mes",
    "mes_nombre",
    "fecha_lectura_ini",
    "fecha_lectura_fin",
    "consumo_act",
    "consumo_act_valor",
    "tarifa_activa",
    "lectura_actual",
    "lectura_anterior",
    "factor_medidor",
    "latitud",
    "longitud"
]

columnas_ordenadas = [c for c in columnas_ordenadas if c in df_total.columns]
df_total = df_total[columnas_ordenadas]

df_total = df_total.sort_values(by=["año", "mes"]).reset_index(drop=True)

# ------------------------------------------------------------
# EXPORTACIÓN DEL EXCEL FINAL
# ------------------------------------------------------------

archivo_salida = ruta_salida / "salp_consumo_2022_2025_ordenado.xlsx"
df_total.to_excel(archivo_salida, index=False)

print("\nProceso completado correctamente.")
print("Excel final organizado por año y mes generado en:")
print(archivo_salida)
