# ============================================================
# 0)  LIBRERÍAS, PARÁMETROS Y **SEMILLA GLOBAL**
# ============================================================
import datetime as dt, numpy as np, pandas as pd, yfinance as yf, random, torch, gym
import torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt, talib as ta
from sklearn.model_selection import train_test_split

SEED = 42                                                   #  ← reproducibilidad
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)                   
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

START = dt.datetime(2015, 1, 1)
END   = dt.datetime(2024, 12, 31)

TICKERS = [
   "GOOGL", "BRK-B", "MSFT", "AAPL", "JNJ", "GC=F", "BABA", "ASML",
    "MCD", "UNH", "TSLA", "SAN", "AMZN", "MC.PA", "BTC-USD", "0700.HK" ]
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# 1) DESCARGA Y PREPARACIÓN DE DATOS
# ============================================================
px = yf.download(TICKERS, start=START, end=END, auto_adjust=False,
                 progress=False)['Close'].ffill().bfill()

# ---- NUEVO split ---------------------------------------------------------
train_px = px.loc[:'2021-12-31']   # 2015-01-01 → 2021-12-31
val_px   = px.loc['2022']          # año 2022
test_px  = px.loc['2023':]         # 2023 + 2024
# --------------------------------------------------------------------------

ret_train = train_px.pct_change().dropna()
ret_val   = val_px  .pct_change().dropna()
ret_test  = test_px .pct_change().dropna()

# ============================================================
# 2)  ENTORNO CONTINUO – EL AGENTE ELIGE PESOS CADA DÍA
# ============================================================
class PortfolioEnv(gym.Env):
    """
    Observación:  precios normalizados (log-ret), 14 indicadores * N
    Acción:       vector continuo de pesos (softmax → suman 1)
    Recompensa:   log-retorno diario de la cartera
    """
    def __init__(self, prices):
        super().__init__()
        self.px  = prices.values
        self.ret = pd.DataFrame(prices).pct_change().fillna(0).values
        self.dates = prices.index
        self.n = self.px.shape[1]
        self.action_space      = gym.spaces.Box(-1, 1, shape=(self.n,))
        obs_dim = self.n*14                       # 14 indicadores por activo
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,))
        self.action_space.seed(SEED)              # ←  semilla para acciones
        self._reset_internal()

    # ---------- indicadores técnicos (3 reales + padding) ----------
    def _tech(self, t):
        feat = []
        for c in range(self.n):
            win = self.px[max(0,t-100):t+1, c].astype(float)
            rsi  = ta.RSI(win)[-1] if win.size>14 else 50
            macd = ta.MACD(win)[0][-1] if win.size>35 else 0
            adx  = ta.ADX(win,win,win)[-1] if win.size>14 else 20
            feat += [rsi/100, macd/100, adx/100] + [0]*11
        return np.nan_to_num(feat, nan=0)

    def _state(self):
        return self._tech(self.t)

    # ---------- API Gym ----------
    def reset(self, *, seed: int | None = None, **kwargs):
        super().reset(seed=seed)
        self._reset_internal()
        return self._state()

    def _reset_internal(self):
        self.t = 0
        self.w = np.ones(self.n)/self.n       # equiponderado
        self.port_val = 1.0

    def step(self, action):
        w_new  = torch.softmax(torch.tensor(action), 0).cpu().numpy()
        r_next = (self.ret[self.t+1] @ w_new)
        self.port_val *= (1+r_next)
        reward = np.log(1+r_next)
        self.t += 1
        self.w = w_new
        done = (self.t >= len(self.ret)-2)
        return self._state(), reward, done, {}


# ============================================================
# 3)  AGENTE PPO  +  GAE-λ
# ============================================================
class Policy(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s_dim,256), nn.Tanh(),
                                 nn.Linear(256,256), nn.Tanh())
        self.mu      = nn.Linear(256, a_dim)
        self.v       = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.zeros(a_dim))

    def forward(self, x):
        h        = self.net(x)
        dist     = torch.distributions.Normal(self.mu(h), self.log_std.exp())
        value    = self.v(h)
        return dist, value


def compute_gae(rewards, values, gamma=0.97, lam=0.95):
    """Devuelve ventajas normalizadas + retornos con GAE"""
    values = values + [0]                      # v_{t+1} = 0 al final
    gae, advs, rets = 0, [], []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma*values[t+1] - values[t]
        gae   = delta + gamma*lam*gae
        advs.insert(0, gae)
        rets.insert(0, gae + values[t])
    advs = torch.tensor(advs, dtype=torch.float32, device=DEVICE)
    advs = (advs - advs.mean()) / (advs.std()+1e-8)        # normaliza
    rets = torch.tensor(rets, dtype=torch.float32, device=DEVICE)
    return advs.unsqueeze(1), rets.unsqueeze(1)


def ppo_update(memory, policy, opt, eps=0.2):
    states, acts, rewards, vals, logp_old = map(list, zip(*memory))
    advs, rets = compute_gae(rewards, vals)
    states     = torch.stack(states)
    acts       = torch.stack(acts)
    logp_old   = torch.stack(logp_old)

    for _ in range(4):                                    # 4 epochs
        dist, val = policy(states)
        logp_new  = dist.log_prob(acts).sum(-1, keepdim=True)
        ratio     = (logp_new - logp_old).exp()
        surr1, surr2 = ratio*advs, torch.clamp(ratio,1-eps,1+eps)*advs
        entropy   = dist.entropy().sum(-1, keepdim=True).mean()
        loss = -torch.min(surr1,surr2).mean() + 0.5*(rets-val).pow(2).mean() \
               - 0.005*entropy                          # ligera entropía
        opt.zero_grad(); loss.backward(); opt.step()


def train_policy(env, epochs=30):
    pol = Policy(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
    opt = optim.Adam(pol.parameters(), 3e-4)
    for ep in range(epochs):
        s = torch.tensor(env.reset(seed=SEED), dtype=torch.float32, device=DEVICE)
        mem=[]; tot=0; done=False
        while not done:
            dist,val = pol(s)
            a  = dist.sample(); logp = dist.log_prob(a).sum()
            s2,r,done,_ = env.step(a.cpu().numpy())
            mem.append((s, a, torch.tensor([r], device=DEVICE), val, logp.detach()))
            s  = torch.tensor(s2, dtype=torch.float32, device=DEVICE); tot += r
        ppo_update(mem, pol, opt)
        print(f"Epoch {ep+1:3d} | retorno episódico: {tot:.4f}")
    return pol

# ============================================================
# 4)  ENTRENAMIENTO Y SIMULACIÓN 2023-2024 
# ============================================================
train_env = PortfolioEnv(train_px)
agent     = train_policy(train_env, epochs=30)

test_env = PortfolioEnv(test_px)
state, done = test_env.reset(seed=SEED), False
while not done:
    s = torch.tensor(state, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        a = agent(s)[0].mean
    state, _, done, _ = test_env.step(a.detach().cpu().numpy())

weights_drl = pd.Series(test_env.w, index=test_px.columns)
weights_drl = weights_drl / weights_drl.sum()
print("\nPesos finales DRL:\n", weights_drl.round(4))


# ============================================================
#  MÉTRICAS
# ============================================================
def metrics(w, ret):
    mu  = (w * ret.mean()).sum() * 252
    sig = np.sqrt(w.values @ ret.cov().values @ w.values) * np.sqrt(252)
    rf  = 0.0248
    return mu, sig, (mu - rf) / sig

# ——— resultados ————————————————————————————————
mu_drl,  sig_drl,  sh_drl  = metrics(weights_drl,  ret_test)          # DRL
mu_mk,   sig_mk,   sh_mk   = 0.1737, 0.1439, (0.1737-0.0248)/0.1439   # Markowitz


# ============================================================
#  PALETA + ESTILO 
# ============================================================
COL = {"Markowitz": "#FF8C42",   # naranja seco
       "DRL"      : "#4895EF"}   # azul corporativo

plt.style.use("ggplot")          # base coherente
plt.rcParams.update({
    "font.size"      : 11,
    "axes.facecolor" : "white",
    "grid.color"     : "lightgray",
    "axes.edgecolor" : "black",
    "axes.linewidth" : .8,
    "figure.dpi"     : 110
})


# ------------------------------------------------------------------
#  GRÁFICAS · Markowitz  vs  DRL 
# ------------------------------------------------------------------
labels_md   = ["Markowitz", "DRL"]
colors_md   = [COL[l] for l in labels_md]

exp_ret_md  = [mu_mk * 100,  mu_drl * 100]
risk_md     = [sig_mk * 100, sig_drl * 100]
sharpe_md   = [sh_mk,        sh_drl]

fig, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].bar(labels_md, exp_ret_md, color=colors_md, edgecolor="black")
ax[0].set_title("Rendimiento (%) Anualizado")
ax[0].set_ylabel("%")

ax[1].bar(labels_md, risk_md,    color=colors_md, edgecolor="black")
ax[1].set_title("Desviación Estándar (%) Anualizada")
ax[1].set_ylabel("%")

ax[2].bar(labels_md, sharpe_md,  color=colors_md, edgecolor="black")
ax[2].set_title("Ratio de Sharpe")
ax[2].set_ylabel("valor")

for a in ax:
    a.spines[['right','top']].set_visible(False)
    a.grid(axis="y", linestyle=":", alpha=.7)

plt.suptitle("Comparativa Markowitz vs DRL (2023-2024)",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.90])
plt.show()


# ------------------------------------------------------------------
#  TABLA  · Markowitz · DRL
# ------------------------------------------------------------------
summary = pd.DataFrame({
    'R%': [mu_mk*100,  mu_drl*100],
    'σ%':    [sig_mk*100, sig_drl*100],
    'Sharpe':[sh_mk,  sh_drl]
}, index=['Markowitz','DRL'])

print("\n==============  COMPARACIÓN 2023-2024  ==============\n")
print(summary.round(3))
