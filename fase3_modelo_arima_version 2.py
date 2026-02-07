# ============================================================
# FASE 3 – MODELADO PREDICTIVO DEL CONSUMO ENERGÉTICO (SALP)
# MODELO ARIMA 
# PRONÓSTICO POR CUENTA – 6 MESES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

# ------------------------------------------------------------
# 1. CARGA DE DATOS
# ------------------------------------------------------------

RUTA = Path(
    r"C:\Users\arlin\OneDrive\Imágenes\SALP_Bucaramanga\data_clean"
)

ARCHIVO = RUTA / "salp_consumo_2022_2025_ordenado.xlsx"
SALIDA = RUTA / "pronostico_arima_por_cuenta_6_meses.xlsx"

df = pd.read_excel(ARCHIVO)

df["cuenta"] = df["cuenta"].astype(str)
df["año"] = df["año"].astype(int)
df["mes"] = df["mes"].astype(int)
df["consumo_act"] = pd.to_numeric(df["consumo_act"], errors="coerce")
df = df[df["consumo_act"] > 0]

df["fecha"] = pd.to_datetime(
    df["año"].astype(str) + "-" + df["mes"].astype(str) + "-01"
)

# ============================================================
# 2. SERIE TEMPORAL GLOBAL (SISTEMA COMPLETO)
# ============================================================

serie_mensual = (
    df.groupby(["año", "mes"])["consumo_act"]
    .sum()
    .reset_index()
)

serie_mensual["fecha"] = pd.to_datetime(
    serie_mensual["año"].astype(str) + "-" +
    serie_mensual["mes"].astype(str) + "-01"
)

serie_mensual = serie_mensual.sort_values("fecha")
serie_mensual.set_index("fecha", inplace=True)

serie = serie_mensual["consumo_act"]

# ------------------------------------------------------------
# 3. GRÁFICA GLOBAL DEL CONSUMO
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(serie, marker="o", linewidth=2.5)
ax.set_title(
    "Evolución mensual del consumo energético del SALP\nPeriodo 2022–2025",
    fontsize=14,
    fontweight="bold"
)
ax.set_xlabel("Fecha")
ax.set_ylabel("Consumo mensual (kWh)")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 4. GRÁFICAS DESAGREGADAS POR AÑO
# ------------------------------------------------------------

for año in serie_mensual["año"].unique():

    data = serie_mensual[serie_mensual["año"] == año]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data["consumo_act"], marker="o", linewidth=2.5)

    ax.set_title(
        f"Consumo energético mensual – Año {año}",
        fontsize=13,
        fontweight="bold"
    )
    ax.set_xlabel("Mes")
    ax.set_ylabel("Consumo mensual (kWh)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 5. PRUEBA DE ESTACIONARIEDAD (ADF)
# ------------------------------------------------------------

adf = adfuller(serie)

print("\nPrueba Dickey-Fuller Aumentada (ADF)")
print(f"Estadístico ADF: {adf[0]:.3f}")
print(f"p-value: {adf[1]:.4f}")

# ------------------------------------------------------------
# 6. VARIACIÓN MENSUAL
# ------------------------------------------------------------

serie_diff = serie.diff().dropna()

fig, ax = plt.subplots(figsize=(11, 5))

ax.bar(serie_diff.index, serie_diff, width=20)
ax.axhline(0, color="black")

ax.set_title(
    "Variación mensual del consumo energético",
    fontsize=13,
    fontweight="bold"
)
ax.set_xlabel("Fecha")
ax.set_ylabel("Cambio mensual (kWh)")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

# ============================================================
# 7. PRONÓSTICO ARIMA POR CUENTA (6 MESES)
# ============================================================

MESES_A_PREDECIR = 6
resultados = []

for cuenta in df["cuenta"].unique():

    serie_cuenta = (
        df[df["cuenta"] == cuenta]
        .groupby("fecha")["consumo_act"]
        .sum()
        .sort_index()
    )

    if len(serie_cuenta) < 3:
        continue

    try:
        modelo = ARIMA(serie_cuenta, order=(1, 1, 1))
        ajuste = modelo.fit()

        forecast = ajuste.forecast(steps=MESES_A_PREDECIR)
        forecast = forecast.clip(lower=0)

        fechas = pd.date_range(
            start=serie_cuenta.index.max() + pd.offsets.MonthBegin(1),
            periods=MESES_A_PREDECIR,
            freq="MS"
        )

        for f, v in zip(fechas, forecast):
            resultados.append({
                "cuenta": cuenta,
                "año": f.year,
                "mes": f.month,
                "mes_nombre": f.month_name(locale="es_ES"),
                "consumo_pronosticado_kwh": round(v, 2)
            })

    except Exception as e:
        print(f"Cuenta omitida {cuenta}: {e}")

# ------------------------------------------------------------
# 8. EXPORTACIÓN A EXCEL
# ------------------------------------------------------------

df_pronostico = pd.DataFrame(resultados)
df_pronostico.to_excel(SALIDA, index=False)

print("\n✅ PROCESO COMPLETADO")
print(f"📊 Total de registros generados: {len(df_pronostico)}")
print(f"📁 Excel generado en:\n{SALIDA}")
