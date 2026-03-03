# ============================================================
# PREDICCIÓN ARIMA – ENERO 2025 A JUNIO 2025 POR CUENTA
# ============================================================

import pandas as pd
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. RUTAS
# ------------------------------------------------------------

RUTA_DATOS = Path(
    r"C:\Users\arlin\OneDrive\Imágenes\SALP_Bucaramanga\Prediccion 6 meses\data_clean"
)

RUTA_SALIDA = Path(
    r"C:\Users\arlin\OneDrive\Imágenes\SALP_Bucaramanga\Prediccion 6 meses"
)

ARCHIVO_ENTRADA = RUTA_DATOS / "salp_consumo_2022_2025_ordenado.xlsx"
ARCHIVO_SALIDA = RUTA_SALIDA / "pronostico_enero_junio_2025.xlsx"

# ------------------------------------------------------------
# 2. CARGA
# ------------------------------------------------------------

df = pd.read_excel(ARCHIVO_ENTRADA)

df["cuenta"] = df["cuenta"].astype(str)
df["consumo_act"] = pd.to_numeric(df["consumo_act"], errors="coerce")
df = df[df["consumo_act"] > 0]

df["fecha"] = pd.to_datetime(
    df["año"].astype(str) + "-" + df["mes"].astype(str) + "-01"
)

# ------------------------------------------------------------
# 3. DEFINIR PERIODO OBJETIVO
# ------------------------------------------------------------

fecha_inicio_prediccion = pd.Timestamp("2025-01-01")
fecha_fin_prediccion = pd.Timestamp("2025-06-01")

resultados = []

# ------------------------------------------------------------
# 4. MODELO POR CUENTA
# ------------------------------------------------------------

for cuenta in df["cuenta"].unique():

    serie = (
        df[df["cuenta"] == cuenta]
        .groupby("fecha")["consumo_act"]
        .sum()
        .sort_index()
    )

    if len(serie) < 4:
        continue

    try:
        modelo = ARIMA(serie, order=(1,1,1))
        ajuste = modelo.fit()

        # Número total de meses a proyectar
        meses_totales = (
            (fecha_fin_prediccion.year - serie.index.max().year) * 12 +
            (fecha_fin_prediccion.month - serie.index.max().month)
        )

        if meses_totales <= 0:
            continue

        forecast = ajuste.forecast(steps=meses_totales)
        forecast = forecast.clip(lower=0)

        fechas_futuras = pd.date_range(
            start=serie.index.max() + pd.offsets.MonthBegin(1),
            periods=meses_totales,
            freq="MS"
        )

        df_temp = pd.DataFrame({
            "fecha": fechas_futuras,
            "valor": forecast.values
        })

        # Filtrar solo enero 2025 a junio 2025
        df_temp = df_temp[
            (df_temp["fecha"] >= fecha_inicio_prediccion) &
            (df_temp["fecha"] <= fecha_fin_prediccion)
        ]

        for _, row in df_temp.iterrows():
            resultados.append({
                "cuenta": cuenta,
                "año": row["fecha"].year,
                "mes": row["fecha"].month,
                "mes_nombre": row["fecha"].strftime("%B"),
                "consumo_pronosticado_kwh": float(round(row["valor"], 2))
            })

    except Exception as e:
        print(f"Cuenta omitida {cuenta}: {e}")

# ------------------------------------------------------------
# 5. EXPORTAR
# ------------------------------------------------------------

df_pronostico = pd.DataFrame(resultados)

df_pronostico = df_pronostico.sort_values(
    by=["cuenta", "año", "mes"]
).reset_index(drop=True)

df_pronostico.to_excel(ARCHIVO_SALIDA, index=False)

print("\n✅ PROCESO COMPLETADO")
print(f"📊 Total de registros generados: {len(df_pronostico)}")
print(f"📁 Excel generado en:\n{ARCHIVO_SALIDA}")
