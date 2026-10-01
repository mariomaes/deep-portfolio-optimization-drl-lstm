# Inteligencia Artificial y Optimización de Carteras: Markowitz vs. Deep RL (PPO) y Bi-LSTM

Este repositorio contiene la implementación empírica de mi Trabajo de Fin de Grado, centrado en comparar el enfoque clásico de optimización Media-Varianza de Markowitz frente a dos arquitecturas de aprendizaje profundo aplicadas a la asignación dinámica de activos: **Deep Reinforcement Learning (PPO)** y **Redes Neuronales Recurrentes Bidireccionales (Bi-LSTM)**.

## 📊 Universo de Inversión y División Temporal
Se analiza una cartera diversificada geográfica y sectorialmente compuesta por **16 activos** (renta variable de EE. UU., Europa y China, junto con materias primas y criptoactivos): `GOOGL`, `BRK-B`, `MSFT`, `AAPL`, `JNJ`, `BABA`, `ASML`, `MCD`, `UNH`, `TSLA`, `SAN`, `AMZN`, `MC.PA`, `0700.HK`, `GC=F` (Oro) y `BTC-USD`.
* **Entrenamiento:** 2015 – 2021
* **Validación (ajuste de hiperparámetros y control en Bi-LSTM):** 2022
* **Prueba fuera de muestra (Out-of-Sample Backtest):** 2023 – 2024

## 🧠 Metodologías Implementadas
1. **Modelo Media-Varianza de Markowitz (`R` - `fPortfolio`):** Construcción de la frontera eficiente bajo restricción *Long-Only* y selección de la Cartera Tangente maximizando el Ratio de Sharpe sobre la Línea del Mercado de Capitales (usando el bono estadounidense a 10 años `GS10` como tasa libre de riesgo).
2. **Deep Reinforcement Learning (`Python` - `PyTorch`, `OpenAI Gym`, `TA-Lib`):** Diseño de un entorno continuo personalizado (`PortfolioEnv`) cuyo espacio de estados integra indicadores técnicos normalizados (`RSI`, `MACD`, `ADX`). Entrenamiento de un agente Actor-Critic mediante **Proximal Policy Optimization (PPO)** y **Generalized Advantage Estimation (GAE)** para maximizar el log-retorno acumulado.
3. **Predicción Secuencial con Bi-LSTM (`Python` - `TensorFlow/Keras`, `Scikit-learn`):** Entrenamiento de redes *Bidirectional LSTM* independientes por activo (ventanas deslizantes de `lookback = 120` días) con esquema predictivo autorregresivo en el conjunto de test y asignación de pesos mediante transformación *softmax* suavizada ($\alpha = 1.3$).

## 📈 Resultados Fuera de Muestra (2023 – 2024)

| Modelo | Rentabilidad Anualizada (%) | Volatilidad Anualizada (%) | Ratio de Sharpe |
| :--- | :---: | :---: | :---: |
| **Markowitz (Cartera Tangente)** | 17,37% | 14,39% | 1,035 |
| **Deep RL (Agente PPO + GAE)** | 18,16% | **11,31%** | 1,386 |
| **Redes Recurrentes (Bi-LSTM)** | **41,51%** | 17,93% | **2,176** |

* El agente **DRL (PPO)** reduce significativamente la volatilidad anualizada de la cartera al **11,31%**, incrementando el Ratio de Sharpe un **34%** respecto a Markowitz.
* El enfoque guiado por **Bi-LSTM** captura la persistencia tendencial del sector tecnológico y activos alternativos en 2023-2024, alcanzando un Ratio de Sharpe de **2,176**.
