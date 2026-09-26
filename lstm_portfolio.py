
# ============================================================
# 0)  LIBRERÍAS, PARÁMETROS Y **SEMILLA GLOBAL**
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
import matplotlib.pyplot as plt
import tensorflow as tf, random, os
SEED=42; np.random.seed(SEED); tf.random.set_seed(SEED); random.seed(SEED); os.environ["PYTHONHASHSEED"]=str(SEED)

assets = [
   "GOOGL", "BRK-B", "MSFT", "AAPL", "JNJ", "GC=F", "BABA", "ASML",
    "MCD", "UNH", "TSLA", "SAN", "AMZN", "MC.PA", "BTC-USD", "0700.HK" ]




# Fechas de interés
init_date      = '2015-01-01'
end_date_train = '2021-12-31'
end_date_val   = '2022-12-31'
end_date_pred  = '2024-12-31'   # test / out-of-sample


# ============================================================
# 1) DESCARGA Y PREPARACIÓN DE DATOS
# ============================================================

df = yf.download(assets, start=init_date, end=end_date_pred, auto_adjust=False)

print(df.columns)

# Extraer solo el "Adj Close" de cada activo
df = df['Adj Close']
df = df.dropna()

# Verificar las primeras filas
print("Datos descargados desde Yahoo Finance: ")
print(df.head())

# Recortar el DataFrame a las fechas específicas
portfolio = df.loc[init_date:end_date_pred]


# ============================================================
# 2) CONFIGURACIÓN DEL MODELO LSTM
# ============================================================

def lstm_training(data,
                  lookback=120, horizon=1,
                  neurons=64,     
                  epochs=50,
                  batch_size=32):
    """
    Entrena un modelo **Bidirectional LSTM** por activo.
    """
    results = {}

    for asset in assets:
        print(f"Entrenando (Bi-LSTM): {asset}")

        # ---------- 1. Escalado ----------
        scaler = MinMaxScaler()
        train  = data.loc[:end_date_train, asset].to_frame()
        train_scaled = scaler.fit_transform(train)

        # ---------- 2. Ventanas ----------
        X_train, y_train = [], []
        for i in range(lookback, len(train_scaled) - horizon):
            X_train.append(train_scaled[i-lookback:i, 0])
            y_train.append(train_scaled[i:i+horizon, 0])

        X_train = np.array(X_train).reshape((-1, lookback, 1))
        y_train = np.array(y_train)

        # ---------- 3. Arquitectura ----------
        model = Sequential([
            Bidirectional(
                LSTM(neurons, return_sequences=False),
                input_shape=(lookback, 1)
            ),
            Dense(horizon)
        ])
        model.compile(optimizer='adam', loss='mse')

        # ---------- 4. Entrenamiento ----------
        model.fit(X_train, y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=1)

        results[asset] = {"model": model, "scaler": scaler}

    return results


    
# ============================================================
# 3) ENTRENAMIENTO Y VALIDACIÓN DEL MODELO LSTM
# ============================================================


# Entrenamiento del modelo LSTM
lookback = 120
horizon = 1
models = lstm_training(portfolio, lookback, horizon)

print("Entrenamiento completado.")


# Validación del modelo LSTM
validation_results = {}
for asset, res in models.items():
    scaler, model = res["scaler"], res["model"]

    val = portfolio.loc[end_date_train:end_date_val, asset].values.reshape(-1, 1)
    val_scaled = scaler.transform(val)

    X_val = [val_scaled[i - lookback:i, 0]
             for i in range(lookback, len(val_scaled))]
    X_val = np.array(X_val).reshape((-1, lookback, 1))

    preds = scaler.inverse_transform(model.predict(X_val, verbose=0))

    validation_results[asset] = {
        "real":      val[lookback:].flatten(),
        "predicted": preds.flatten()
    }

# Visualización de resultados de validación
plt.style.use("ggplot")
plt.rcParams.update({
    "font.size": 10, "axes.facecolor": "white", "grid.color": "lightgray",
    "axes.edgecolor": "black", "axes.linewidth": .8
})

for t, res in validation_results.items():
    plt.figure(figsize=(10, 5))
    plt.plot(res["real"],      label="Real",       color="#4895EF", lw=1.4)
    plt.plot(res["predicted"], label="Predicción", color="#FF8C42",
             ls="--", lw=1.6)
    plt.title(f"Validación LSTM · {t}", fontsize=12, fontweight="bold")
    plt.xlabel("Día"); plt.ylabel("Precio (USD)")
    plt.grid(axis="both", ls=":", alpha=.7); plt.legend(frameon=False)
    plt.tight_layout(); plt.show()


# ============================================================
# 4) PREDICCIÓN USANDO EL MODELO LSTM
# ============================================================

prediction_results = {}
steps_oos = len(portfolio.loc[end_date_val:end_date_pred])

for asset, res in models.items():
    scaler, model = res["scaler"], res["model"]

    hist = portfolio.loc[:end_date_val, asset].values.reshape(-1, 1)
    hist_scaled = scaler.transform(hist)

    series = hist_scaled.copy()
    preds_scaled = []

    for _ in range(steps_oos):
        X_last = series[-lookback:].reshape(1, lookback, 1)
        y_hat  = model.predict(X_last, verbose=0)     # shape (1, 1)
        preds_scaled.append(y_hat[0, 0])
        series = np.vstack([series, y_hat])

    preds_real = scaler.inverse_transform(
                    np.array(preds_scaled).reshape(-1, 1)
                 ).flatten()
    real_oos = portfolio.loc[end_date_val:end_date_pred, asset].values.flatten()

    prediction_results[asset] = {"real": real_oos, "predicted": preds_real}

# Visualización de la predicción
plt.style.use("ggplot")
plt.rcParams.update({
    "font.size":10, "axes.facecolor":"white", "grid.color":"lightgray",
    "axes.edgecolor":"black", "axes.linewidth":.8
})

for t, res in prediction_results.items():
    plt.figure(figsize=(10, 5))
    plt.plot(res["real"],      label="Real",       color="#4895EF", lw=1.4)
    plt.plot(res["predicted"], label="Pronóstico", color="#FF8C42",
             ls="--", lw=1.6)
    plt.title(f"Pronóstico LSTM · {t}",
              fontsize=12, fontweight="bold")
    plt.xlabel("Día"); plt.ylabel("Precio (USD)")
    plt.grid(axis="both", ls=":", alpha=.7); plt.legend(frameon=False)
    plt.tight_layout(); plt.show()




# Predicciones calculadas por el modelo LSTM
lstm_predictions = {
    asset: (res["predicted"][1:] - res["predicted"][:-1]) / res["predicted"][:-1]
    for asset, res in prediction_results.items()
}




# ============================================================
# 5) PESOS DERIVADOS DE LA LSTM
# ============================================================


risk_free_rate = 0.0248  # Tasa libre de riesgo del 2.48%


# 1) Matriz de retornos predichos (filas = días, columnas = activos)
asset_list = list(lstm_predictions.keys())
ret_matrix = np.column_stack([lstm_predictions[a] for a in asset_list])

# 2) Estadísticos por activo
mu_hat    = ret_matrix.mean(axis=0)  * 252          # retorno anualizado
sigma_hat = ret_matrix.std(axis=0)   * np.sqrt(252) # σ anualizada
cov_hat   = np.cov(ret_matrix, rowvar=False) * 252  # matriz Σ̂

# 3) Genera los pesos (elige un criterio)
alpha = 1.3                                           # agresividad
raw_w = np.exp(alpha * mu_hat)                      # Soft-max sobre retornos


weights = pd.Series(raw_w / raw_w.sum(), index=asset_list)
print("\nPesos LSTM:")
print(weights.round(4))

# 4) Métricas de la cartera LSTM
real_ret_df = (
    portfolio[asset_list]              # precios reales
    .pct_change()                      # → retornos diarios
    .loc[end_date_val:end_date_pred]   # periodo fuera de muestra
    .dropna()
)

# — ② estadísticos de la cartera construida con los PESOS ya fijados —
# vector de pesos en el mismo orden que las columnas de real_ret_df
w = weights.loc[real_ret_df.columns].values

mean_daily   = real_ret_df.mean().values           # μ diarios
cov_daily    = real_ret_df.cov().values            # Σ diaria
mu_lstm_real = np.dot(w, mean_daily) * 252         # anualizado
var_lstm_real= np.dot(w, cov_daily @ w) * 252      # anualizado
std_lstm_real= np.sqrt(var_lstm_real)

# ----------------------------------------------------------------


# ============================================================
# COMPARATIVA CON MARKWOITZ
# ============================================================


# Datos de Markowitz
markowitz_expected_return = 0.1737
markowitz_std_dev = 0.1439


# Calcular ratios de Sharpe
markowitz_sharpe_ratio =(markowitz_expected_return- risk_free_rate) / markowitz_std_dev
lstm_sharpe_ratio = (mu_lstm_real - risk_free_rate) / std_lstm_real

# Resultados impresos
print("Resultados Markowitz:")
print(f"- Rendimiento anualizado: {markowitz_expected_return * 100:.2f}%")
print(f"- Desviación estándar anualizada: {markowitz_std_dev * 100:.2f}%")
print(f"- Ratio de Sharpe: {markowitz_sharpe_ratio:.4f}")

print("\nResultados LSTM:")
print(f"- Rendimiento anualizado: {mu_lstm_real * 100:.2f}%")
print(f"- Desviación estándar anualizada: {std_lstm_real * 100:.2f}%")
print(f"- Ratio de Sharpe: {lstm_sharpe_ratio:.4f}")



# ------------------------------------------------------------
#  MÉTRICAS 
# ------------------------------------------------------------
drl_expected_return = 18.161 / 100       
drl_std_dev         = 11.312 / 100        
drl_sharpe_ratio    = 1.386



labels = ["Markowitz", "LSTM", "DRL"]

expected_returns = [
    markowitz_expected_return * 100,
    mu_lstm_real * 100,
    drl_expected_return * 100
]

std_devs = [
    markowitz_std_dev * 100,
    std_lstm_real * 100,
    drl_std_dev * 100
]

sharpe_ratios = [
    markowitz_sharpe_ratio,
    lstm_sharpe_ratio,
    drl_sharpe_ratio
]

# ============================================================
#  PALETA  ·  ESTILO 
# ============================================================
COL = {
    "Markowitz": "#FF8C42",  # naranja seco
    "LSTM"     : "#2CA58D",  # turquesa suave
    "DRL"      : "#4895EF"   # azul corporativo
}

plt.style.use("ggplot")              
plt.rcParams.update({
    "font.size"      : 11,
    "axes.facecolor" : "white",
    "grid.color"     : "lightgray",
    "axes.edgecolor" : "black",
    "axes.linewidth" : .8,
    "figure.dpi"     : 110
})

# ------------------------------------------------------------
#  GRÁFICAS · Markowitz  ·  LSTM  ·  DRL
# ------------------------------------------------------------
bar_cols = [COL["Markowitz"], COL["LSTM"], COL["DRL"]]

# —— Rendimiento anualizado ——
plt.figure(figsize=(8, 6))
plt.bar(labels, expected_returns, color=bar_cols, edgecolor="black")
plt.title("Rendimiento (%) Anualizado")
plt.ylabel("%")
plt.ylim(0, max(expected_returns) * 1.15)
plt.grid(axis="y", linestyle=":", alpha=.7)
plt.show()

# —— Desviación estándar anualizada ——
plt.figure(figsize=(8, 6))
plt.bar(labels, std_devs, color=bar_cols, edgecolor="black")
plt.title("Desviación Estándar (%) Anualizada")
plt.ylabel("%")
plt.ylim(0, max(std_devs) * 1.15)
plt.grid(axis="y", linestyle=":", alpha=.7)
plt.show()

# —— Ratio de Sharpe ——
plt.figure(figsize=(8, 6))
plt.bar(labels, sharpe_ratios, color=bar_cols, edgecolor="black")
plt.title("Ratio de Sharpe")
plt.ylabel("Valor")
plt.ylim(0, max(sharpe_ratios) * 1.15)
plt.grid(axis="y", linestyle=":", alpha=.7)
plt.show()