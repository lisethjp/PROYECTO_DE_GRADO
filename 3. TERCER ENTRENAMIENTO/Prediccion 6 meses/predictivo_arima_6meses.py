# ============================================================
# MODELO HÍBRIDO REALISTA – ENERO 2025 A JUNIO 2025
# REGRESIÓN + ARIMA SOBRE RESIDUOS + CONTROL DE CRECIMIENTO
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. RUTA CORRECTA
# ------------------------------------------------------------

RUTA = Path(
    r"C:\Users\arlin\OneDrive\Imágenes\SALP_Bucaramanga\TERCER ENTRENAMIENTO\Prediccion 6 meses\data_clean"
)

ARCHIVO_ENTRADA = RUTA / "salp_consumo_2022_2025_ordenado.xlsx"
ARCHIVO_SALIDA = RUTA.parent / "pronostico_realista_enero_junio_2025.xlsx"

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
# 3. RANGO A PRONOSTICAR
# ------------------------------------------------------------

FECHA_INICIO = pd.Timestamp("2025-01-01")
FECHA_FIN = pd.Timestamp("2025-06-01")

meses_es = {
    1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",
    5:"Mayo",6:"Junio",7:"Julio",8:"Agosto",
    9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"
}

resultados = []

# ------------------------------------------------------------
# 4. FUNCIÓN PARA QUITAR OUTLIERS
# ------------------------------------------------------------

def limpiar_outliers(serie):
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_sup = q3 + 1.5 * iqr
    limite_inf = q1 - 1.5 * iqr
    return serie.clip(lower=limite_inf, upper=limite_sup)

# ------------------------------------------------------------
# 5. MODELADO POR CUENTA
# ------------------------------------------------------------

cuentas = df["cuenta"].unique()
print(f"Total cuentas detectadas: {len(cuentas)}")

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

    if len(serie) < 12:
        continue

    try:
        # -----------------------------------------
        # 1️⃣ Limpiar outliers
        # -----------------------------------------
        serie_limpia = limpiar_outliers(serie)

        # -----------------------------------------
        # 2️⃣ Suavizamiento ligero
        # -----------------------------------------
        serie_suavizada = serie_limpia.ewm(alpha=0.3).mean()

        # -----------------------------------------
        # 3️⃣ Regresión lineal (tendencia)
        # -----------------------------------------
        X = np.arange(len(serie_suavizada)).reshape(-1,1)
        y = serie_suavizada.values

        modelo_lr = LinearRegression()
        modelo_lr.fit(X, y)

        tendencia_hist = modelo_lr.predict(X)

        # -----------------------------------------
        # 4️⃣ Residuos
        # -----------------------------------------
        residuos = y - tendencia_hist

        # -----------------------------------------
        # 5️⃣ ARIMA sobre residuos (ligero)
        # -----------------------------------------
        modelo_arima = ARIMA(residuos, order=(1,0,1))
        ajuste_arima = modelo_arima.fit()

        # -----------------------------------------
        # 6️⃣ Calcular pasos hasta junio 2025
        # -----------------------------------------
        ultima_fecha = serie.index.max()

        pasos = (
            (FECHA_FIN.year - ultima_fecha.year) * 12 +
            (FECHA_FIN.month - ultima_fecha.month)
        )

        if pasos <= 0:
            continue

        X_futuro = np.arange(len(serie), len(serie)+pasos).reshape(-1,1)

        forecast_lr = modelo_lr.predict(X_futuro)
        forecast_arima = ajuste_arima.forecast(steps=pasos)

        forecast_total = forecast_lr + forecast_arima

        # -----------------------------------------
        # 7️⃣ Control de crecimiento máximo mensual
        # -----------------------------------------
        crecimiento_max = 0.15  # máximo 15% mensual

        ultimo_valor_real = serie.iloc[-1]

        for i in range(len(forecast_total)):
            limite_superior = ultimo_valor_real * (1 + crecimiento_max)
            forecast_total[i] = min(forecast_total[i], limite_superior)
            ultimo_valor_real = forecast_total[i]

        forecast_total = np.clip(forecast_total, 0, None)

        fechas_futuras = pd.date_range(
            start=ultima_fecha + pd.offsets.MonthBegin(1),
            periods=pasos,
            freq="MS"
        )

        df_temp = pd.DataFrame({
            "fecha": fechas_futuras,
            "valor": forecast_total
        })

        df_temp = df_temp[
            (df_temp["fecha"] >= FECHA_INICIO) &
            (df_temp["fecha"] <= FECHA_FIN)
        ]

        for _, row in df_temp.iterrows():

            mes_num = row["fecha"].month

            resultados.append({
                "cuenta": cuenta,
                "año": row["fecha"].year,
                "mes": mes_num,
                "mes_nombre": meses_es[mes_num],
                "consumo_pronosticado_kwh": round(float(row["valor"]),2),
                "modelo": "Hibrido_Realista"
            })

    except Exception as e:
        print(f"Cuenta omitida {cuenta}: {e}")

# ------------------------------------------------------------
# 6. EXPORTAR
# ------------------------------------------------------------

df_pronostico = pd.DataFrame(resultados)
df_pronostico = df_pronostico.sort_values(
    by=["cuenta","año","mes"]
).reset_index(drop=True)

df_pronostico.to_excel(ARCHIVO_SALIDA, index=False)

print("\n✅ MODELO REALISTA COMPLETADO")
print(f"📊 Total filas: {len(df_pronostico)}")
print(f"📁 Archivo generado en:\n{ARCHIVO_SALIDA}")
