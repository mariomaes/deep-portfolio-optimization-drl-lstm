library(quantmod)
library(timeSeries)
library(fPortfolio)
library(caTools)
library(dplyr)
library(PerformanceAnalytics)
library(ggplot2)
library(zoo)
library(scales)


###########################################################
###################### PASOS PREVIOS ######################
###########################################################

# Vector de TICKERS


tickers <- c("GOOGL", "BRK-B", "MSFT", "AAPL", "JNJ", "GC=F", "BABA", "ASML",
              "MCD", "UNH", "TSLA", "SAN", "AMZN", "MC.PA", "BTC-USD", "0700.HK" )

# Extrayendo la data



ini_date     <- "2015-01-01"
test_start   <- "2023-01-01"                           # inicio ventana test
test_end     <- "2024-12-31"                           # fin ventana test
train_end    <- "2021-12-31"                           # último día de estimación

PrecPort<- NULL
for (Ticker in tickers)
PrecPort<-cbind(PrecPort, getSymbols(Ticker, from = ini_date, to = test_end,
                                     auto.assign=FALSE)[,4])
PrecPort 

# Renombrando los encabezados de las columnas

colnames(PrecPort)<-tickers

# Calculando los retornos

RetPort <- na.omit(ROC(PrecPort, type="discrete"))
RetPort


train_ret <- RetPort[paste0("/", train_end)]           
test_ret  <- RetPort[paste0(test_start, "/", test_end)]
# Copia xts para el evaluador
test_ret_eval <- test_ret          # objeto xts/zoo



# Convirtiendo los retornos en series de tiempo
RetPort <- as.timeSeries(RetPort)

RetPort

train_ret <- as.timeSeries(train_ret)
test_ret <- as.timeSeries(test_ret)



###############################################################################
###################### FRONTERA Y PORTAFOLIOS EFICIENTES ######################
###############################################################################

# Tasa libre de riesgo anual (bono 10Y)
getSymbols("GS10", src = "FRED", from = ini_date, to = test_end)
mean_risk_free_rate <- mean(GS10$GS10, na.rm = TRUE) / 100   # en decimal
cat("Tasa libre de riesgo media 2015-2024: ",
    round(100*mean_risk_free_rate, 2), "%\n\n")

spec <- portfolioSpec()
setRiskFreeRate(spec) <- mean_risk_free_rate/252

# Calculando la frontera eficiente

fronteraEff <- portfolioFrontier(train_ret, spec = spec, constraints = "LongOnly")

# Graficando la frontera
# En el gr?fico, se pueden incorporar los siguientes elementos:
# 1: Frontera Eficiente
# 2: El portafolio con la m?nima varianza global
# 3: L?nea tangente al portafolio
# 4: Riesgo y retorno de cada activo
# 5: Portafolio con activos del mismo peso
# 6: Fronteras de dos activos
# 7: Portafolios de Monte Carlo
# 8: Ratio de Sharpe

plot(fronteraEff,c(1,2,3))


# Riesgos y Retornos de la frontera

Riesgo_Retorno <- frontierPoints(fronteraEff)


# Matrices de correlaci?n y covarianza

MatrizCorr<-cor(train_ret)
MatrizCov<-cov(train_ret)

MatrizCov
MatrizCorr

# Pesos en las fronteras eficientes
fronteraPesos<-getWeights(fronteraEff)
colnames(fronteraPesos)<-tickers



# Puntos interesantes:

      
  # Punto de la L?nea de Mercado de Capitales tangente a la Frontera Eficiente  
  LMC<-tangencyPortfolio(train_ret, spec=spec, constraints="LongOnly")
  LMC
  
    # Reporte de los pesos asignados:
    LMC_Pesos<-getWeights(LMC)
    DF_LMC_Pesos<-data.frame(LMC_Pesos)
    acciones<-colnames(fronteraPesos)
    
    
    # Paleta personalizada de 16 colores sobrios y elegantes
    colores_elegantes <- c(
      "#3E5F8A", "#466B99", "#4E79A7", "#5685B5", "#5E91C3",
      "#669DD1", "#6EA9DF", "#77B5ED", "#7FC1FB", "#88CCFF",
      "#92D1F7", "#9CD6F0", "#A6DBE9", "#B0E0E2", "#BAD5D3",
      "#C4CAC4"
    )
    
    ggplot(data = DF_LMC_Pesos, aes(x = acciones, y = LMC_Pesos * 100, fill = acciones))+
      geom_bar(stat = "identity", colour = "black") +
      geom_text(
        aes(label = sprintf("%.2f %%", LMC_Pesos * 100)), 
        vjust = -0.25, size = 2.5
      ) +
      scale_fill_manual(values = colores_elegantes, name = "Activos") +
      scale_y_continuous(labels = number_format(accuracy = 1)) +  
      ggtitle("Pesos de los activos - Portafolio de mercado") +
      labs(x = "Activos", y = "Pesos") +
      theme_minimal(base_size = 12) +
      theme(
        plot.title         = element_text(hjust = 0.5, face = "bold", size = 12),
        axis.title.x       = element_text(face = "plain"),
        axis.title.y       = element_text(face = "plain"),
        axis.text.x        = element_text(angle = 45, hjust = 1, vjust = 1),
        axis.ticks.x       = element_blank(),
        panel.grid.major.y = element_line(color = "gray80", size = 0.4),
        panel.grid.major.x = element_blank(),
        panel.grid.minor   = element_blank(),
        legend.position    = "none"
      )
    
  
    

# ── 1. Función auxiliar para métricas anuales ────────────────────────────────
eval_port <- function(weights, ret_test) {
  port_ret <- zoo(ret_test %*% weights, order.by = index(ret_test))
  ER_ann   <- mean(port_ret) * 252            # retorno anual
  SD_ann   <- sd  (port_ret) * sqrt(252)      # riesgo anual
  Sharpe   <- (ER_ann - mean_risk_free_rate) / SD_ann
  c(Rendimiento = ER_ann, Riesgo = SD_ann, Sharpe = Sharpe)
}

# ── 2. Evaluar la cartera tangente (LMC) ─────────────────────────────────────
LMC_Pesos <- getWeights(LMC)
perf_LMC <- eval_port(LMC_Pesos, test_ret_eval)

cat("===== Portafolio Tangente (fuera de muestra 2023-2024) =====\n")
cat("Rendimiento anualizado: ", round(100*perf_LMC["Rendimiento"], 2), "%\n")
cat("Riesgo anualizado:      ", round(100*perf_LMC["Riesgo"], 2), "%\n")
cat("Sharpe ratio:           ", round(perf_LMC["Sharpe"], 3), "\n")  



    
 






    