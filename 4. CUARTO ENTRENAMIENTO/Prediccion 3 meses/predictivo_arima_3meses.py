# ============================================================
# PRONÓSTICO AJUSTADO – ABRIL A JUNIO 2025
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. RUTA ACTUAL
# ------------------------------------------------------------

RUTA_DATOS = Path(
    r"C:\Users\arlin\OneDrive\Imágenes\SALP_Bucaramanga\CUARTO ENTRENAMIENTO\Prediccion 3 meses\data_clean"
)

ARCHIVO_ENTRADA = RUTA_DATOS / "salp_consumo_2022_2025_ordenado.xlsx"
ARCHIVO_SALIDA = RUTA_DATOS / "pronostico_ABRIL_JUNIO_2025_AJUSTADO_FINAL.xlsx"

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
# 3. CONFIGURACIÓN
# ------------------------------------------------------------

FECHA_INICIO = pd.to_datetime("2025-04-01")

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
print(f"Cuentas detectadas: {len(cuentas)}")

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

    if len(serie) < 18:
        continue

    try:

        max_hist = serie.max()
        p95 = np.percentile(serie, 95)
        p5 = np.percentile(serie, 5)

        serie_log = np.log(serie)

        d = definir_d(serie_log)

        mejor_aic = np.inf
        mejor_orden = None

        for p in range(0,3):
            for q in range(0,3):
                for P in range(0,2):
                    for Q in range(0,2):
                        try:
                            modelo = SARIMAX(
                                serie_log,
                                order=(p,d,q),
                                seasonal_order=(P,0,Q,12),
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            ajuste = modelo.fit(disp=False)

                            if ajuste.aic < mejor_aic:
                                mejor_aic = ajuste.aic
                                mejor_orden = (p,d,q,P,Q)
                        except:
                            continue

        if mejor_orden is None:
            continue

        p,d,q,P,Q = mejor_orden

        modelo_final = SARIMAX(
            serie_log,
            order=(p,d,q),
            seasonal_order=(P,0,Q,12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        ajuste_final = modelo_final.fit(disp=False)

        # Solo 3 meses exactos
        forecast_log = ajuste_final.forecast(steps=3)
        forecast = np.exp(forecast_log)

        # Control general
        limite_superior = min(max_hist * 1.15, p95 * 1.10)
        forecast = np.clip(forecast, p5 * 0.90, limite_superior)

        fechas_futuras = pd.date_range(
            start=FECHA_INICIO,
            periods=3,
            freq="MS"
        )

        for fecha, valor in zip(fechas_futuras, forecast):

            # ============================================
            # AJUSTE SOLO ABRIL Y JUNIO
            # ============================================

            if fecha.month == 4:      # Abril
                valor = valor * 0.85

            elif fecha.month == 6:    # Junio
                valor = valor * 0.70
                

            # Mayo no se toca

            resultados.append({
                "cuenta": cuenta,
                "año": fecha.year,
                "mes": fecha.month,
                "mes_nombre": MESES_ES[fecha.month],
                "consumo_pronosticado_kwh": round(float(valor), 2)
            })

    except Exception as e:
        print(f"Cuenta omitida {cuenta}: {e}")


# ------------------------------------------------------------
# EXPORTAR
# ------------------------------------------------------------

df_pronostico = pd.DataFrame(resultados)

if not df_pronostico.empty:

    df_pronostico = df_pronostico.sort_values(
        by=["cuenta","año","mes"]
    ).reset_index(drop=True)

    df_pronostico.to_excel(ARCHIVO_SALIDA, index=False)

    print("\n✅ PRONÓSTICO FINAL AJUSTADO COMPLETADO")
    print(f"Filas generadas: {len(df_pronostico)}")
    print(f"Archivo guardado en:\n{ARCHIVO_SALIDA}")

else:
    print("No se generaron resultados.")