import cvxpy as cp
import numpy as np

def get_optimal_weights(mu, Sigma, mode):
    """
    Solves for weights using Convex Optimization (cvxpy).
    """
    n = len(mu)
    w = cp.Variable(n)
    
    # Constraints: Fully invested (sum=1), No shorting (w>=0)
    constraints = [cp.sum(w) == 1, w >= 0]
    
    if mode == 'BULL':
        # Maximize Sharpe Ratio (Proxy: Return - Risk)
        # Gamma (Risk Aversion) = 1.0
        risk = cp.quad_form(w, Sigma)
        ret = mu @ w
        obj = cp.Maximize(ret - 1.0 * risk)
    else: 
        # BEAR or CHOP -> Minimize Variance
        risk = cp.quad_form(w, Sigma)
        obj = cp.Minimize(risk)
        
    prob = cp.Problem(obj, constraints)
    
    try:
        # SCS is a robust solver for these types of problems
        prob.solve(solver=cp.SCS, verbose=False)
        
        # Check if solver failed
        if w.value is None:
            return np.ones(n) / n
            
        return w.value
    except:
        # Fallback to Equal Weight if solver crashes
        return np.ones(n) / n