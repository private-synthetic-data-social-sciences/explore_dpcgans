
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from time import time 


def fit_mixture_model(
        x, 
        bayesian=True,
        init_params="random", 
        covariance_type="full", 
        n_init=5, 
        n_components=3,
        weight_threshold=0.005, # from RDT
        max_iter=6000 # for n_components=10, 2000 is usually not enough
        ):
    """Estimate a mixture model and return some output"""
    
    start_time=time()
    if bayesian:
        model = BayesianGaussianMixture()
    else:
        model = GaussianMixture()

    # fixed 
    model.warm_start=True
    if bayesian:
        model.weight_concentration_prior_type='dirichlet_process'
        model.weight_concentration_prior=1e-3

    # from user input
    model.init_params=init_params
    model.covariance_type=covariance_type
    model.n_init=n_init
    model.n_components=n_components
    model.max_iter=max_iter


    # estimate
    model.fit(x.reshape(-1, 1))
    weights = model.weights_

    total_time = time()-start_time
    output = {
        "weights": weights,
        "n valid components": sum(weights > weight_threshold),
        "time (s)": total_time,
        "n_iter": model.n_iter_
    }

    return output

    