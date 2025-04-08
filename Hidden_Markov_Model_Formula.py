# Practical example using Python's hmmlearn library to model weather patterns (sunny, cloudy, rainy) based on observed temperature data

#
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#

#
# Generate synthetic temperature data
np.random.seed(42)
#

# Define the hidden states (weather conditions)
# 0: Sunny, 1: Cloudy, 2: Rainy
n_states = 3

# True model parameters (unknown in real scenarios)
# Initial state probabilities
true_pi = np.array([0.6, 0.3, 0.1])

# Transition Matrix
# Rows: from state, Columns: to state
true_A = np_array([
    [0.7, 0.2, 0.1], # Sunny -> Sunny, Cloudy, Rainy
    [0.3, 0.5, 0.2], # Cloudy -> Sunny, Cloudy, Rainy
    [0.2, 0.4, 0.4] # Rainy -> Sunny, Cloudy, Rainy
])

# Temperature means for each state
true_means = np.array([
    [27.0], # Sunny, high temperature
    [20.0], # Cloudy, medium temperature
    [15.0]  # Rainy, low temperature
])
#

# Temperature variance for each state
true_covars = np.array([
    [3.0], # Sunny
    [2.0], # Cloudy
    [1.0]  # Rainy 
])
#

# Generate the hidden state sequence
n_days = 100
states = np.zeros(n_days, dtype=int)
states[0] = np.random.choice(n_states, p=true_pi)  # Initial state

for t in range(1, n_days):
    states[t] = np.random.choice(n_states, p=true_A[states[t-1]])
#

# Generate observed temperatures based on hidden weather states
temperatures = np.zeros(n_days)
for t in range(n_days):
    temperatures[t] = np.random.normal(
        true_means[states[t]][0], np.sqrt(true_covars[states[t]][0])
    )
#

# Train HMM model on observed temperatures
model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
model.fit(temperatures.reshape(-1, 1))
#

# Decode the hidden states
hidden_states = model.predict(temperatures.reshape(-1, 1))
#

# Plot results
plt.figure(figsize=(15, 8))
#

# Plot observed temperatures
plt.subplot(211)
plt.plot(temperatures)
plt.title("Observed Temperatures")
plt.ylabel("Temperature (°C)")
#

# Plot true and inferred hidden states
plt.subplot(212)
plt.plot(states, "k-", label="True States")
plt.plot(hidden_states, "r--", label="Inferred States")
plt.legend()
plt.title("Hidden States (Weather Conditions)")
plt.ylabel("State")
plt.xlabel("Day")
plt.yticks([0, 1, 2], ["Sunny", "Cloudy", "Rainy"])
#

# Print learned model parameters
print("Learned transition matrix:")
print(np.round(model.transmat_, 2))

print("\nLearned means and variances:")
for i in range(n_states):
    print(f"State {i}: Mean = {model.means_[i][0]:.1f}, Variance = {np.diag(model.covars_[i])[0]:.1f}")
#

# Calculate model accuracy (state prediction)
accuracy = np.mean(states == hidden_states)
print(f"\nState prediction accuracy: {accuracy:.2f}")
#

# Note: In real applications, we would need to align the identified states with the actual states
# since HMM doesn't preserve the original state labels

plt.tight_layout()
plt.show()


