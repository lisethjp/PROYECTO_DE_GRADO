# ============================================================
# PRONÓSTICO ARIMA POR CUENTA
# ABRIL 2024 A JUNIO 2025
# MESES EN ESPAÑOL (SIN DEPENDER DEL SISTEMA)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. RUTAS
# ------------------------------------------------------------

RUTA_DATOS = Path(
    r"C:\Users\arlin\OneDrive\Imágenes\SALP_Bucaramanga\TERCER ENTRENAMIENTO\Prediccion mas de 12 meses\data_clean"
)

ARCHIVO_ENTRADA = RUTA_DATOS / "salp_consumo_2022_2025_ordenado.xlsx"
ARCHIVO_SALIDA = RUTA_DATOS / "pronostico_ARIMA_abril2024_junio2025.xlsx"

# ------------------------------------------------------------
# 2. DICCIONARIO DE MESES EN ESPAÑOL
# ------------------------------------------------------------

MESES_ES = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre"
}

# ------------------------------------------------------------
# 3. CARGA DE DATOS
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
    df["mes"].astype(int).astype(str) + "-01"
)

# ------------------------------------------------------------
# 4. RANGO A PRONOSTICAR
# ------------------------------------------------------------

FECHA_INICIO = pd.to_datetime("2024-04-01")
FECHA_FIN = pd.to_datetime("2025-06-01")

resultados = []

cuentas = df["cuenta"].unique()
print(f"Total cuentas detectadas: {len(cuentas)}")

# ------------------------------------------------------------
# 5. MODELADO POR CUENTA
# ------------------------------------------------------------

for cuenta in cuentas:

    datos_cuenta = df[df["cuenta"] == cuenta]

    serie = (
        datos_cuenta
        .groupby("fecha")["consumo_act"]
        .sum()
        .sort_index()
    )

    serie = serie.asfreq("MS")
    serie = serie.interpolate(method="linear")

    if len(serie) < 8:
        continue

    try:
        modelo = ARIMA(serie, order=(1,1,1))
        ajuste = modelo.fit()

        ultima_fecha = serie.index.max()

        pasos = (
            (FECHA_FIN.year - ultima_fecha.year) * 12 +
            (FECHA_FIN.month - ultima_fecha.month)
        )

        if pasos <= 0:
            continue

        forecast = ajuste.forecast(steps=pasos)
        forecast = np.clip(forecast, 0, None)

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
                "mes_nombre": MESES_ES[row["fecha"].month],  # 👈 AQUÍ ESTÁ EL CAMBIO
                "consumo_pronosticado_kwh": round(float(row["valor"]), 2)
            })

    except Exception as e:
        print(f"Cuenta omitida {cuenta}: {e}")

# ------------------------------------------------------------
# 6. EXPORTACIÓN
# ------------------------------------------------------------

df_pronostico = pd.DataFrame(resultados)

if not df_pronostico.empty:

    df_pronostico = df_pronostico.sort_values(
        by=["cuenta", "año", "mes"]
    ).reset_index(drop=True)

    df_pronostico.to_excel(ARCHIVO_SALIDA, index=False)

    print("\n✅ PROCESO COMPLETADO CORRECTAMENTE")
    print(f"📊 Total filas generadas: {len(df_pronostico)}")
    print(f"📁 Archivo generado en:\n{ARCHIVO_SALIDA}")
else:
    print("⚠ No se generaron resultados.")
