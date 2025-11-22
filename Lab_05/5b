import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import seaborn as sns

sns.set(style='whitegrid')

TICKER = 'AAPL'
START = '2013-01-01'
END = '2023-01-01'
N_STATES_RANGE = range(2, 6)
RANDOM_STATE = 42

# Download data
data = yf.download(TICKER, start=START, end=END, progress=False)

if data.empty:
    raise RuntimeError("No data downloaded; check ticker or internet connection")

# Price and returns
adj = data['Close'].copy().dropna()
returns = np.log(adj).diff().dropna()
returns.name = 'log_return'

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(returns.values.reshape(-1, 1))

# Evaluate models for best # states
results = []
for n_states in N_STATES_RANGE:
    model = GaussianHMM(n_components=n_states, covariance_type='full',
                        n_iter=200, random_state=RANDOM_STATE)
    model.fit(X)
    logL = model.score(X)

    n = n_states
    d = X.shape[1]
    n_params = n*(n-1) + (n-1) + n*d + n*d*(d+1)/2
    aic = -2 * logL + 2 * n_params
    bic = -2 * logL + np.log(len(X)) * n_params

    results.append({'n_states': n_states, 'model': model,
                    'logL': logL, 'AIC': aic, 'BIC': bic})

# Display evaluation metrics
results_df = pd.DataFrame(results)
print(results_df[['n_states', 'logL', 'AIC', 'BIC']])

# Select best model according to BIC
best_entry = min(results, key=lambda r: r['BIC'])
best_model = best_entry['model']
print("\nSelected n_states =", best_entry['n_states'])

# Predict hidden states
hidden_states = best_model.predict(X)

# Construct labeled DataFrame
df = pd.DataFrame(
    {
        'adj_close': adj.loc[returns.index].to_numpy().ravel(),
        'log_return': returns.to_numpy().ravel(),
        'state': hidden_states.astype(int)
    },
    index=returns.index
)

# Posterior probabilities for next state prediction
logprob, posteriors = best_model.score_samples(X)

# Print summary preview
print("\nSample of labeled data:")
print(df.head())

# Model parameters
print("\nState Means (interpreting return behavior):")
print(best_model.means_)

print("\nState Covariances (volatility levels):")
print(best_model.covars_)

print("\nTransition Matrix (regime switching probabilities):")
print(best_model.transmat_)

# Predict next state
next_state = np.argmax(posteriors[-1])
print("\nMost likely next state:", next_state)


###############################
# VISUALIZATION SECTION
###############################

# Plot closing prices colored by state
plt.figure(figsize=(15, 6))
for state in np.unique(hidden_states):
    idx = df['state'] == state
    plt.plot(df.index[idx], df['adj_close'][idx], '.', label=f'State {state}')
plt.title(f"{TICKER} Price Regimes Inferred by HMM")
plt.legend()
plt.show()


# Plot returns colored by state
plt.figure(figsize=(15, 5))
plt.scatter(df.index, df['log_return'], c=df['state'], cmap='tab10', s=10)
plt.title(f"Log Returns Colored by Hidden State")
plt.show()

# Plot histogram of states
plt.figure(figsize=(8,4))
df['state'].value_counts().sort_index().plot(kind='bar')
plt.title("State Distribution")
plt.xlabel("State")
plt.ylabel("Frequency")
plt.show()
