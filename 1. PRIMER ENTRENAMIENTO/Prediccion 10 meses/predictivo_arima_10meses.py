# ============================================================
# PREDICCIÓN ARIMA – SEPTIEMBRE 2024 A JUNIO 2025 POR CUENTA
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
    r"C:\Users\arlin\OneDrive\Imágenes\SALP_Bucaramanga\Prediccion 10 meses\data_clean"
)

RUTA_SALIDA = Path(
    r"C:\Users\arlin\OneDrive\Imágenes\SALP_Bucaramanga\Prediccion 10 meses"
)

RUTA_SALIDA.mkdir(exist_ok=True)

ARCHIVO_ENTRADA = RUTA_DATOS / "salp_consumo_2022_2025_ordenado.xlsx"
ARCHIVO_SALIDA = RUTA_SALIDA / "pronostico_sep2024_jun2025.xlsx"

# ------------------------------------------------------------
# 2. CARGA DE DATOS
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
# 3. RANGO EXACTO A PRONOSTICAR
# ------------------------------------------------------------

FECHA_INICIO = pd.to_datetime("2024-09-01")
FECHA_FIN = pd.to_datetime("2025-06-01")

FECHAS_OBJETIVO = pd.date_range(
    start=FECHA_INICIO,
    end=FECHA_FIN,
    freq="MS"
)

resultados = []

# ------------------------------------------------------------
# 4. MODELADO POR CUENTA
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

    # Asegurar frecuencia mensual
    serie_cuenta = serie_cuenta.asfreq("MS")

    # Rellenar meses faltantes
    serie_cuenta = serie_cuenta.fillna(method="ffill")

    # Necesita mínimo historial para ARIMA
    if len(serie_cuenta) < 6:
        continue

    try:
        modelo = ARIMA(serie_cuenta, order=(1, 1, 1))
        ajuste = modelo.fit()

        ultima_fecha = serie_cuenta.index.max()

        # Si la serie termina después de junio 2025 no pronostica
        if ultima_fecha >= FECHA_FIN:
            continue

        # Calcular pasos necesarios
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

        # Filtrar solo septiembre 2024 a junio 2025
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
                "consumo_pronosticado_kwh": round(float(row["valor"]), 2)
            })

    except Exception as e:
        print(f"Cuenta omitida {cuenta}: {e}")

# ------------------------------------------------------------
# 5. EXPORTAR A EXCEL
# ------------------------------------------------------------

df_pronostico = pd.DataFrame(resultados)

df_pronostico = df_pronostico.sort_values(
    by=["cuenta", "año", "mes"]
).reset_index(drop=True)

df_pronostico.to_excel(ARCHIVO_SALIDA, index=False)

print("\n✅ PROCESO COMPLETADO")
print(f"📊 Total de filas generadas: {len(df_pronostico)}")
print(f"📁 Archivo generado en:\n{ARCHIVO_SALIDA}")
