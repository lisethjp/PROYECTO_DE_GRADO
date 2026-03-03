# ============================================================
# MODELO HÍBRIDO REALISTA — SEPTIEMBRE 2024 A JUNIO 2025
# FEBRERO ≥ 380 000 kWh
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# RUTA (NO SE CAMBIA)
# ------------------------------------------------------------

RUTA = Path(
    r"C:\Users\arlin\OneDrive\Imágenes\SALP_Bucaramanga\CUARTO ENTRENAMIENTO\Prediccion 10 meses\data_clean"
)

ARCHIVO_ENTRADA = RUTA / "salp_consumo_2022_2025_ordenado.xlsx"
ARCHIVO_SALIDA = RUTA.parent / "pronostico_realista_enero_junio_2025.xlsx"

# ------------------------------------------------------------
# CARGA
# ------------------------------------------------------------

df = pd.read_excel(ARCHIVO_ENTRADA)

df["cuenta"] = df["cuenta"].astype(str)
df["consumo_act"] = pd.to_numeric(df["consumo_act"], errors="coerce")
df = df[df["consumo_act"] > 0]

df["fecha"] = pd.to_datetime(
    df["año"].astype(str) + "-" + df["mes"].astype(str) + "-01"
)

# ------------------------------------------------------------
# NUEVO RANGO
# ------------------------------------------------------------

FECHA_INICIO = pd.Timestamp("2024-09-01")
FECHA_FIN    = pd.Timestamp("2025-06-01")

meses_es = {
    1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",
    5:"Mayo",6:"Junio",7:"Julio",8:"Agosto",
    9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"
}

factores_mes = {
    1:1.22,
    2:1.50,
    3:1.40,
    4:0.98,
    5:1.05,
    6:0.78,
    7:1.00,
    8:1.00,
    9:1.02,
    10:1.05,
    11:1.03,
    12:1.10
}

resultados = []

# ------------------------------------------------------------
# FUNCIÓN OUTLIERS
# ------------------------------------------------------------

def limpiar_outliers(serie):
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    return serie.clip(q1 - 1.5*iqr, q3 + 1.5*iqr)

# ------------------------------------------------------------
# MODELADO POR CUENTA
# ------------------------------------------------------------

cuentas = df["cuenta"].unique()
print(f"Cuentas detectadas: {len(cuentas)}")

for cuenta in cuentas:

    datos_cuenta = df[df["cuenta"] == cuenta]

    serie = (
        datos_cuenta
        .groupby("fecha")["consumo_act"]
        .sum()
        .sort_index()
    )

    serie = serie.asfreq("MS").interpolate()

    if len(serie) < 12:
        continue

    try:
        serie_limpia = limpiar_outliers(serie)
        serie_suav = serie_limpia.ewm(alpha=0.3).mean()

        X = np.arange(len(serie_suav)).reshape(-1,1)
        y = serie_suav.values

        modelo_lr = LinearRegression().fit(X, y)
        tendencia = modelo_lr.predict(X)

        residuos = y - tendencia

        modelo_arima = ARIMA(residuos, order=(1,0,1)).fit()

        ultima_fecha = serie.index.max()

        pasos = (
            (FECHA_FIN.year - ultima_fecha.year) * 12 +
            (FECHA_FIN.month - ultima_fecha.month)
        )

        if pasos <= 0:
            continue

        X_fut = np.arange(len(serie), len(serie)+pasos).reshape(-1,1)

        pred_lr = modelo_lr.predict(X_fut)
        pred_arima = modelo_arima.forecast(steps=pasos)

        forecast = pred_lr + pred_arima

        # Control crecimiento máximo 15%
        ultimo = serie.iloc[-1]
        for i in range(len(forecast)):
            limite = ultimo * 1.15
            forecast[i] = min(forecast[i], limite)
            ultimo = forecast[i]

        forecast = np.clip(forecast, 0, None)

        fechas = pd.date_range(
            start=ultima_fecha + pd.offsets.MonthBegin(1),
            periods=pasos,
            freq="MS"
        )

        df_temp = pd.DataFrame({
            "cuenta": cuenta,
            "fecha": fechas,
            "valor": forecast
        })

        df_temp = df_temp[
            (df_temp["fecha"] >= FECHA_INICIO) &
            (df_temp["fecha"] <= FECHA_FIN)
        ]

        resultados.append(df_temp)

    except Exception as e:
        print(f"Cuenta omitida {cuenta}: {e}")

# ------------------------------------------------------------
# UNIR RESULTADOS
# ------------------------------------------------------------

df_pron = pd.concat(resultados, ignore_index=True)

# ------------------------------------------------------------
# AJUSTE ESTACIONAL
# ------------------------------------------------------------

df_pron["mes"] = df_pron["fecha"].dt.month

df_pron["valor"] = df_pron.apply(
    lambda x: x["valor"] * factores_mes[x["mes"]],
    axis=1
)

# ------------------------------------------------------------
# AJUSTE TOTALES (FEBRERO ≥ 380K)
# ------------------------------------------------------------

totales = df_pron.groupby("fecha")["valor"].sum()

for fecha, total in totales.items():

    mes = fecha.month

    if mes == 2 and total < 380000:
        factor = 380000 / total
        df_pron.loc[df_pron["fecha"] == fecha, "valor"] *= factor
        total = 380000

    if mes == 3 and total < 360000:
        factor = 365000 / total
        df_pron.loc[df_pron["fecha"] == fecha, "valor"] *= factor
        total = 365000

    if total < 100000:
        factor = 100000 / total
        df_pron.loc[df_pron["fecha"] == fecha, "valor"] *= factor
    elif total > 395000:
        factor = 395000 / total
        df_pron.loc[df_pron["fecha"] == fecha, "valor"] *= factor

# ------------------------------------------------------------
# FORMATO FINAL
# ------------------------------------------------------------

df_pron["año"] = df_pron["fecha"].dt.year
df_pron["mes_nombre"] = df_pron["mes"].map(meses_es)

df_final = df_pron[[
    "cuenta","año","mes","mes_nombre","valor"
]].rename(columns={"valor":"consumo_pronosticado_kwh"})

df_final = df_final.sort_values(
    by=["cuenta","año","mes"]
).reset_index(drop=True)

df_final.to_excel(ARCHIVO_SALIDA, index=False)

print("\n✅ PRONÓSTICO SEPT 2024 – JUN 2025 COMPLETADO")
print(f"Filas generadas: {len(df_final)}")
print(f"Archivo guardado en:\n{ARCHIVO_SALIDA}")
