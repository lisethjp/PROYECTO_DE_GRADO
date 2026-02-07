# ============================================================
# FASE 3 – PRONÓSTICO ARIMA POR CUENTA (6 MESES)
# SISTEMA DE ALUMBRADO PÚBLICO – SALP
# SALIDA EXCLUSIVA A EXCEL (SIN GRÁFICAS)
# ============================================================

import pandas as pd
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. RUTAS
# ------------------------------------------------------------

RUTA_BASE = Path(
    r"C:\Users\arlin\OneDrive\Imágenes\SALP_Bucaramanga\data_clean"
)

ARCHIVO_ENTRADA = RUTA_BASE / "salp_consumo_2022_2025_ordenado.xlsx"
ARCHIVO_SALIDA = RUTA_BASE / "pronostico_arima_por_cuenta_6_meses.xlsx"

MESES_A_PREDECIR = 6

# ------------------------------------------------------------
# 2. CARGA Y LIMPIEZA
# ------------------------------------------------------------

df = pd.read_excel(ARCHIVO_ENTRADA)

df = df.dropna(subset=["cuenta", "año", "mes", "consumo_act"])

df["cuenta"] = df["cuenta"].astype(str)
df["año"] = df["año"].astype(int)
df["mes"] = df["mes"].astype(int)
df["consumo_act"] = pd.to_numeric(df["consumo_act"], errors="coerce")

df = df[df["consumo_act"] > 0]

# Fecha mensual forzada
df["fecha"] = pd.to_datetime(
    df["año"].astype(str) + "-" + df["mes"].astype(str) + "-01",
    errors="coerce"
)

df = df.dropna(subset=["fecha"])

# ------------------------------------------------------------
# 3. PRONÓSTICO POR CUENTA (ARIMA)
# ------------------------------------------------------------

resultados = []

cuentas = df["cuenta"].unique()
print(f"Total de cuentas detectadas: {len(cuentas)}")

for cuenta in cuentas:

    df_cuenta = df[df["cuenta"] == cuenta]

    serie = (
        df_cuenta
        .groupby("fecha")["consumo_act"]
        .sum()
        .sort_index()
    )

    # ARIMA necesita mínimo 3 datos
    if len(serie) < 3:
        continue

    try:
        modelo = ARIMA(
            serie,
            order=(1, 1, 1)
        )

        resultado = modelo.fit()

        forecast = resultado.forecast(steps=MESES_A_PREDECIR)
        forecast = forecast.clip(lower=0)

        fechas_futuras = pd.date_range(
            start=serie.index.max() + pd.offsets.MonthBegin(1),
            periods=MESES_A_PREDECIR,
            freq="MS"
        )

        for fecha, valor in zip(fechas_futuras, forecast):
            resultados.append({
                "cuenta": cuenta,
                "fecha": fecha,
                "año": fecha.year,
                "mes": fecha.month,
                "mes_nombre": fecha.month_name(locale="es_ES"),
                "consumo_pronosticado_kwh": round(valor, 2)
            })

    except Exception as e:
        print(f"Cuenta omitida {cuenta}: {e}")

# ------------------------------------------------------------
# 4. EXPORTACIÓN FINAL
# ------------------------------------------------------------

df_resultado = pd.DataFrame(resultados)

print(f"Total de filas generadas: {len(df_resultado)}")

df_resultado.to_excel(
    ARCHIVO_SALIDA,
    index=False,
    engine="openpyxl"
)

print("\n✅ EXCEL GENERADO CORRECTAMENTE")
print(ARCHIVO_SALIDA)
