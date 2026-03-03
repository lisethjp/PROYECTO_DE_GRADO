# ============================================================
# SERIE COMPLETA REAL + PRONÓSTICO POR CUENTA
# ABRIL 2024 → JUNIO 2025
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# RUTA (NO CAMBIAR)
# ------------------------------------------------------------
RUTA = Path(
    r"C:\Users\arlin\OneDrive\Imágenes\SALP_Bucaramanga\CUARTO ENTRENAMIENTO\Prediccion mas de 12 meses\data_clean"
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
# RANGO REQUERIDO
# ------------------------------------------------------------
FECHA_INICIO = pd.Timestamp("2024-04-01")
FECHA_FIN = pd.Timestamp("2025-06-01")

rango = pd.date_range(FECHA_INICIO, FECHA_FIN, freq="MS")

meses_es = {
    1:"Enero",2:"Febrero",3:"Marzo",
    4:"Abril",5:"Mayo",6:"Junio",
    7:"Julio",8:"Agosto",9:"Septiembre",
    10:"Octubre",11:"Noviembre",12:"Diciembre"
}

resultados = []

cuentas = df["cuenta"].unique()
print(f"Cuentas detectadas: {len(cuentas)}")

# ------------------------------------------------------------
# PROCESAR CADA CUENTA
# ------------------------------------------------------------
for cuenta in cuentas:

    datos = df[df["cuenta"] == cuenta]

    serie_real = (
        datos.groupby("fecha")["consumo_act"]
        .sum()
        .sort_index()
    )

    # Serie en rango completo
    serie = serie_real.reindex(rango)

    # Última fecha con dato real
    ultima_real = serie_real.index.max()

    # --------------------------------------------------------
    # PRONOSTICAR SOLO LO QUE FALTA
    # --------------------------------------------------------
    if ultima_real < FECHA_FIN:

        historial = serie_real.asfreq("MS").interpolate()

        if len(historial) >= 6:
            try:
                modelo = ARIMA(historial, order=(2,1,2)).fit()

                pasos = (
                    (FECHA_FIN.year - ultima_real.year) * 12 +
                    (FECHA_FIN.month - ultima_real.month)
                )

                pred = modelo.forecast(steps=pasos)

                fechas_pred = pd.date_range(
                    ultima_real + pd.offsets.MonthBegin(1),
                    periods=pasos,
                    freq="MS"
                )

                serie.loc[fechas_pred] = pred.values

            except:
                pass

    # Relleno final si queda algún NaN
    serie = serie.fillna(method="ffill").fillna(method="bfill")

    df_temp = pd.DataFrame({
        "cuenta": cuenta,
        "fecha": rango,
        "consumo_pronosticado_kwh": serie.values
    })

    resultados.append(df_temp)

# ------------------------------------------------------------
# UNIR
# ------------------------------------------------------------
df_pron = pd.concat(resultados, ignore_index=True)

# ------------------------------------------------------------
# FORMATO FINAL
# ------------------------------------------------------------
df_pron["año"] = df_pron["fecha"].dt.year
df_pron["mes"] = df_pron["fecha"].dt.month
df_pron["mes_nombre"] = df_pron["mes"].map(meses_es)

df_final = df_pron[[
    "cuenta","año","mes","mes_nombre",
    "consumo_pronosticado_kwh"
]]

df_final = df_final.sort_values(
    by=["cuenta","año","mes"]
).reset_index(drop=True)

# ------------------------------------------------------------
# GUARDAR
# ------------------------------------------------------------
df_final.to_excel(ARCHIVO_SALIDA, index=False)

print("\n✅ SERIE COMPLETA REAL + PRONÓSTICO GENERADA")
print(f"Filas generadas: {len(df_final)}")
print(f"Archivo guardado en:\n{ARCHIVO_SALIDA}")