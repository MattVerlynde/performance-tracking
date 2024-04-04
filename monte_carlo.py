import numpy as np
from scipy.stats import chi2
from joblib import Parallel, delayed
from tqdm import trange
import plotly.express as px

def test_omnibus(X: np.ndarray, n: int) -> np.ndarray:
    Sigma_0 = np.zeros((p,p), dtype=complex)
    result_denominator = 0
    for t in range(T):
        Sigma_t = n*X[t]
        Sigma_0 = Sigma_0 + Sigma_t
        result_denominator = result_denominator + np.log(np.abs(np.linalg.det(Sigma_t)))
    return n*(p*T*np.log(T) + result_denominator - T*np.log(np.abs(np.linalg.det(Sigma_0))))


def proba_theorique_omnibus(ln_q, p, n, T):
    f = (T-1)*(p**2)
    rho = 1 - (2*p**2-1)/(6*(T-1)*p)*(T/n-1/(n*T))
    omega_2 = (p**2)*(p**2-1)/(24*rho**2)*(T/(n**2)-1/(n*T)**2) -\
            (p**2)*(T-1)/4 * (1 - 1/rho)**2
    Z = -2*rho*ln_q
    Z = np.sort(Z)
    return Z, chi2(f).cdf(Z) + omega_2*(chi2(f+4).cdf(Z) - chi2(f).cdf(Z))


def test_online(sum_Xjmoins1: np.ndarray, Xj: np.ndarray, n: int, j:int):
    return n*(
        p*(j*np.log(j) - (j-1)*np.log(j-1)) +
        (j-1)*np.log(np.abs(np.linalg.det(sum_Xjmoins1))) +
        np.log(np.abs(np.linalg.det(Xj))) - j*np.log(np.abs(np.linalg.det(sum_Xjmoins1+Xj)))
    )


def one_trial_omnibus(trial_no, p, n, T):
    rng = np.random.default_rng(trial_no)
    X = np.zeros((T,p,p), dtype=complex)
    for t in range(T):
        x = rng.standard_normal((p,n)) + 1j*rng.standard_normal((p,n))
        X[t] = np.cov(x)
    return test_omnibus(X, n)

    
if __name__ == "__main__":

    # Parameters
    p = 3
    T = 4
    n = 121
    n_trials = 10000

    # Monte-Carlo
    ln_q = Parallel(n_jobs=-1)(
            delayed(one_trial_omnibus)(i, p, n, T) for i in trange(n_trials))
    ln_q = np.array(ln_q)

    # Probabilities
    Z, p_theo = proba_theorique_omnibus(ln_q, p, n, T)
    # Compute cdf over Z
    p_emp = np.array([np.sum(Z < z) for z in Z])
    p_emp = p_emp/p_emp.max()

    # Display
    fig = px.line(x=Z, y=p_emp, title="Omnibus test", color_discrete_sequence=["blue"])
    # Add line for theoretical p-value to fig
    fig.add_scatter(x=Z, y=p_theo, mode="lines", line=dict(color="red"))
    fig.write_html("omnibus.html", include_plotlyjs="cdn")