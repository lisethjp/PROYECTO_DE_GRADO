# ============================================================
# PREDICCIÓN ARIMA OPTIMIZADO – ABRIL 2025 A JUNIO 2025
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. RUTAS
# ------------------------------------------------------------

RUTA_DATOS = Path(
    r"C:\Users\arlin\OneDrive\Imágenes\SALP_Bucaramanga\SEGUNDO ENTRENAMIENTO\Prediccion 3 meses\data_clean"
)

RUTA_SALIDA = RUTA_DATOS.parent

ARCHIVO_ENTRADA = RUTA_DATOS / "salp_consumo_2022_2025_ordenado.xlsx"
ARCHIVO_SALIDA = RUTA_SALIDA / "pronostico_ARIMA_abril2025_junio2025.xlsx"

# ------------------------------------------------------------
# 2. CARGA Y LIMPIEZA
# ------------------------------------------------------------

df = pd.read_excel(ARCHIVO_ENTRADA)

df["cuenta"] = df["cuenta"].astype(str)
df["año"] = pd.to_numeric(df["año"], errors="coerce")
df["mes"] = pd.to_numeric(df["mes"], errors="coerce")
df["consumo_act"] = pd.to_numeric(df["consumo_act"], errors="coerce")

df = df.dropna(subset=["cuenta", "año", "mes", "consumo_act"])
df = df[df["consumo_act"] > 0]

df["fecha"] = pd.to_datetime(
    df["año"].astype(int).astype(str) + "-" +
    df["mes"].astype(int).astype(str) + "-01",
    errors="coerce"
)

df = df.dropna(subset=["fecha"])

# ------------------------------------------------------------
# 3. RANGO A PRONOSTICAR
# ------------------------------------------------------------

FECHA_INICIO = pd.to_datetime("2025-04-01")
FECHA_FIN = pd.to_datetime("2025-06-01")

resultados = []

# ------------------------------------------------------------
# 4. FUNCIÓN PARA ENCONTRAR MEJOR ARIMA POR AIC
# ------------------------------------------------------------

def seleccionar_mejor_arima(serie):

    mejor_aic = np.inf
    mejor_orden = None

    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    modelo = ARIMA(serie, order=(p, d, q))
                    resultado = modelo.fit()
                    if resultado.aic < mejor_aic:
                        mejor_aic = resultado.aic
                        mejor_orden = (p, d, q)
                except:
                    continue

    return mejor_orden

# ------------------------------------------------------------
# 5. MODELADO POR CUENTA
# ------------------------------------------------------------

cuentas = df["cuenta"].unique()
print(f"Total de cuentas detectadas: {len(cuentas)}")

for cuenta in cuentas:

    datos_cuenta = df[df["cuenta"] == cuenta]

    serie_cuenta = (
        datos_cuenta
        .groupby("fecha")["consumo_act"]
        .sum()
        .sort_index()
    )

    serie_cuenta = serie_cuenta.asfreq("MS")
    serie_cuenta = serie_cuenta.interpolate(method="linear")

    if len(serie_cuenta) < 8:
        continue

    try:
        # Selección automática del mejor modelo
        mejor_orden = seleccionar_mejor_arima(serie_cuenta)

        if mejor_orden is None:
            continue

        modelo = ARIMA(serie_cuenta, order=mejor_orden)
        ajuste = modelo.fit()

        ultima_fecha = serie_cuenta.index.max()

        pasos = (FECHA_FIN.year - ultima_fecha.year) * 12 + \
                (FECHA_FIN.month - ultima_fecha.month)

        if pasos <= 0:
            continue

        forecast = ajuste.forecast(steps=pasos)
        forecast = forecast.clip(lower=0)

        fechas_futuras = pd.date_range(
            start=ultima_fecha + pd.offsets.MonthBegin(1),
            periods=pasos,
            freq="MS"
        )

        df_temp = pd.DataFrame({
            "fecha": fechas_futuras,
            "valor": forecast
        })

        df_temp = df_temp[
            (df_temp["fecha"] >= FECHA_INICIO) &
            (df_temp["fecha"] <= FECHA_FIN)
        ]

        for _, row in df_temp.iterrows():
            resultados.append({
                "cuenta": cuenta,
                "año": row["fecha"].year,
                "mes": row["fecha"].month,
                "mes_nombre": row["fecha"].month_name(locale="es_ES"),
                "consumo_pronosticado_kwh": round(float(row["valor"]), 2),
                "modelo_usado": str(mejor_orden)
            })

    except Exception as e:
        print(f"Cuenta omitida {cuenta}: {e}")

# ------------------------------------------------------------
# 6. EXPORTAR
# ------------------------------------------------------------

df_pronostico = pd.DataFrame(resultados)

df_pronostico = df_pronostico.sort_values(
    by=["cuenta", "año", "mes"]
).reset_index(drop=True)

df_pronostico.to_excel(ARCHIVO_SALIDA, index=False)

print("\n✅ PROCESO COMPLETADO (ARIMA OPTIMIZADO)")
print(f"📊 Total de filas generadas: {len(df_pronostico)}")
print(f"📁 Archivo generado en:\n{ARCHIVO_SALIDA}")
