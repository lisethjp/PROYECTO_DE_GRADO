# ============================================================
# PRONÓSTICO – ARIMA + REGRESIÓN LINEAL
# JULIO A DICIEMBRE 2025
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# RUTA
# ------------------------------------------------------------

RUTA_DATOS = Path(
r"C:\Users\arlin\OneDrive\Imágenes\SALP_Bucaramanga\data_clean"
)

ARCHIVO_ENTRADA = RUTA_DATOS / "salp_consumo_2022_2025_ordenado.xlsx"
ARCHIVO_SALIDA = RUTA_DATOS / "pronostico_JULIO_DICIEMBRE_2025_AJUSTADO.xlsx"

# ------------------------------------------------------------
# CARGA Y LIMPIEZA
# ------------------------------------------------------------

df = pd.read_excel(ARCHIVO_ENTRADA)

df["cuenta"] = df["cuenta"].astype(str)
df["año"] = pd.to_numeric(df["año"], errors="coerce")
df["mes"] = pd.to_numeric(df["mes"], errors="coerce")
df["consumo_act"] = pd.to_numeric(df["consumo_act"], errors="coerce")

df = df.dropna(subset=["cuenta","año","mes","consumo_act"])
df = df[df["consumo_act"] > 0]

df["fecha"] = pd.to_datetime(
df["año"].astype(int).astype(str) + "-" +
df["mes"].astype(int).astype(str) + "-01"
)

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------

FECHA_INICIO = pd.to_datetime("2025-07-01")

MESES_ES = {
1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",
11:"Noviembre",12:"Diciembre"
}

resultados = []

# ------------------------------------------------------------
# FUNCIÓN PARA DEFINIR d
# ------------------------------------------------------------

def definir_d(serie):
    return 0 if adfuller(serie)[1] < 0.05 else 1

# ------------------------------------------------------------
# MODELADO POR CUENTA
# ------------------------------------------------------------

cuentas = df["cuenta"].unique()

print("Cuentas detectadas:", len(cuentas))

for cuenta in cuentas:

    datos_cuenta = df[df["cuenta"] == cuenta]

    serie = (
        datos_cuenta
        .groupby("fecha")["consumo_act"]
        .sum()
        .sort_index()
    )

    serie = serie.asfreq("MS")
    serie = serie.interpolate()

    if len(serie) < 18:
        continue

    try:

        max_hist = serie.max()
        p95 = np.percentile(serie,95)
        p5 = np.percentile(serie,5)

        # ----------------------------------------------------
        # MODELO ARIMA
        # ----------------------------------------------------

        serie_log = np.log(serie)

        d = definir_d(serie_log)

        mejor_aic = np.inf
        mejor_orden = None

        for p in range(3):
            for q in range(3):

                try:

                    modelo = ARIMA(
                        serie_log,
                        order=(p,d,q)
                    )

                    ajuste = modelo.fit()

                    if ajuste.aic < mejor_aic:
                        mejor_aic = ajuste.aic
                        mejor_orden = (p,d,q)

                except:
                    continue

        if mejor_orden is None:
            continue

        modelo_arima = ARIMA(
            serie_log,
            order=mejor_orden
        ).fit()

        forecast_arima = np.exp(
            modelo_arima.forecast(6)
        )

        # ----------------------------------------------------
        # REGRESIÓN LINEAL
        # ----------------------------------------------------

        df_reg = serie.reset_index()

        df_reg["t"] = np.arange(len(df_reg))

        X = df_reg[["t"]]
        y = df_reg["consumo_act"]

        modelo_rl = LinearRegression()
        modelo_rl.fit(X,y)

        fechas_futuras = pd.date_range(
            start=FECHA_INICIO,
            periods=6,
            freq="MS"
        )

        futuros = pd.DataFrame()
        futuros["t"] = np.arange(len(df_reg),len(df_reg)+6)

        forecast_rl = modelo_rl.predict(futuros)

        # ----------------------------------------------------
        # MODELO FINAL
        # ----------------------------------------------------

        forecast = (forecast_arima + forecast_rl) / 2

        limite_superior = min(max_hist*1.15, p95*1.10)

        forecast = np.clip(
            forecast,
            p5*0.90,
            limite_superior
        )

        # ----------------------------------------------------
        # GUARDAR RESULTADOS
        # ----------------------------------------------------

        ultimo_valor = serie.iloc[-1]

        for fecha, valor in zip(fechas_futuras, forecast):

            if fecha.month == 10:
                valor *= 1.02
            elif fecha.month == 11:
                valor *= 1.03
            elif fecha.month == 12:
                valor *= 1.04

            cambio_max = ultimo_valor * 1.12
            cambio_min = ultimo_valor * 0.88

            valor = np.clip(valor, cambio_min, cambio_max)

            ultimo_valor = valor

            resultados.append({
                "cuenta": cuenta,
                "año": fecha.year,
                "mes": fecha.month,
                "mes_nombre": MESES_ES[fecha.month],
                "consumo_pronosticado_kwh": round(float(valor),2)
            })

    except Exception as e:
        print("Cuenta omitida:", cuenta)

# ------------------------------------------------------------
# EXPORTAR
# ------------------------------------------------------------

df_pronostico = pd.DataFrame(resultados)

if not df_pronostico.empty:

    df_pronostico = df_pronostico.sort_values(
        by=["cuenta","año","mes"]
    ).reset_index(drop=True)

    df_pronostico.to_excel(
        ARCHIVO_SALIDA,
        index=False
    )

    print("\nPRONÓSTICO COMPLETADO")
    print("Filas generadas:", len(df_pronostico))
    print("Archivo guardado en:", ARCHIVO_SALIDA)

else:
    print("No se generaron resultados.")