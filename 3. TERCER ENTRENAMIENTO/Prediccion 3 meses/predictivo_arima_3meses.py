# ============================================================
# PRONÓSTICO HÍBRIDO DE ALTA PRECISIÓN — LR + ARIMA OPTIMIZADO
# ABRIL 2025 A JUNIO 2025
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. RUTA DE DATOS
# ------------------------------------------------------------

RUTA_DATOS = Path(
    r"C:\Users\arlin\OneDrive\Imágenes\SALP_Bucaramanga\TERCER ENTRENAMIENTO\Prediccion 3 meses\data_clean"
)

ARCHIVO_ENTRADA = RUTA_DATOS / "salp_consumo_2022_2025_ordenado.xlsx"
ARCHIVO_SALIDA = RUTA_DATOS / "pronostico_HIBRIDO_PRECISO_abril2025_junio2025.xlsx"

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
    df["mes"].astype(int).astype(str) + "-01"
)

# ------------------------------------------------------------
# 3. RANGO A PRONOSTICAR
# ------------------------------------------------------------

FECHA_INICIO = pd.to_datetime("2025-04-01")
FECHA_FIN = pd.to_datetime("2025-06-01")

resultados = []

# ------------------------------------------------------------
# FUNCIONES AUXILIARES
# ------------------------------------------------------------

def definir_d(serie):
    p_value = adfuller(serie)[1]
    return 0 if p_value < 0.05 else 1


def mejor_arima(serie):
    train = serie[:-3]
    test = serie[-3:]

    mejor_rmse = np.inf
    mejor_orden = None

    d = definir_d(train)

    for p in range(0, 4):
        for q in range(0, 4):
            try:
                modelo = ARIMA(train, order=(p, d, q))
                ajuste = modelo.fit()
                pred = ajuste.forecast(steps=3)

                rmse = np.sqrt(mean_squared_error(test, pred))

                if rmse < mejor_rmse:
                    mejor_rmse = rmse
                    mejor_orden = (p, d, q)

            except:
                continue

    return mejor_orden


# ------------------------------------------------------------
# 4. MODELADO POR CUENTA
# ------------------------------------------------------------

cuentas = df["cuenta"].unique()
print(f"Total de cuentas detectadas: {len(cuentas)}")

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

    if len(serie) < 12:
        continue

    try:

        # =====================================================
        # 1) REGRESIÓN LINEAL (TENDENCIA)
        # =====================================================

        X = np.arange(len(serie)).reshape(-1, 1)
        y = serie.values

        modelo_lr = LinearRegression()
        modelo_lr.fit(X, y)

        pred_lr_hist = modelo_lr.predict(X)

        # =====================================================
        # 2) RESIDUOS
        # =====================================================

        residuos = y - pred_lr_hist

        residuos_log = np.log1p(residuos - residuos.min() + 1)

        # =====================================================
        # 3) ARIMA OPTIMIZADO
        # =====================================================

        orden = mejor_arima(residuos_log)

        if orden is None:
            continue

        modelo_arima = ARIMA(residuos_log, order=orden)
        ajuste_arima = modelo_arima.fit()

        # =====================================================
        # 4) PRONÓSTICO
        # =====================================================

        ultima_fecha = serie.index.max()

        pasos = (
            (FECHA_FIN.year - ultima_fecha.year) * 12 +
            (FECHA_FIN.month - ultima_fecha.month)
        )

        if pasos <= 0:
            continue

        X_futuro = np.arange(len(serie), len(serie) + pasos).reshape(-1, 1)

        forecast_lr = modelo_lr.predict(X_futuro)

        forecast_arima_log = ajuste_arima.forecast(steps=pasos)
        forecast_arima = np.expm1(forecast_arima_log)
        forecast_arima = forecast_arima + residuos.min() - 1

        forecast_total = forecast_lr + forecast_arima
        forecast_total = np.clip(forecast_total, 0, None)

        fechas_futuras = pd.date_range(
            start=ultima_fecha + pd.offsets.MonthBegin(1),
            periods=pasos,
            freq="MS"
        )

        for fecha, valor in zip(fechas_futuras, forecast_total):

            if FECHA_INICIO <= fecha <= FECHA_FIN:

                resultados.append({
                    "cuenta": cuenta,
                    "año": fecha.year,
                    "mes": fecha.month,
                    "consumo_predicho": round(valor, 2)
                })

    except Exception as e:
        print(f"Error en cuenta {cuenta}: {e}")
        continue


# ------------------------------------------------------------
# 5. EXPORTAR RESULTADOS
# ------------------------------------------------------------

df_resultados = pd.DataFrame(resultados)

if not df_resultados.empty:
    df_resultados.to_excel(ARCHIVO_SALIDA, index=False)
    print("Pronóstico generado correctamente.")
else:
    print("No se generaron resultados.")
