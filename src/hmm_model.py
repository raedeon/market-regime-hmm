from hmmlearn import hmm
import numpy as np

def train_hmm(data, n_states=3):
    """
    Trains a Gaussian HMM on the provided data.
    Input: data (numpy array of Returns and Volatility)
    Output: model, hidden_states
    """
    # covariance_type="full" allows each state to have its own correlation structure
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
    model.fit(data)
    hidden_states = model.predict(data)
    return model, hidden_states