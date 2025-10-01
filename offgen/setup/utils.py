'''Setup: various handy utilities.'''

# External modules.
import ipywidgets as ipw
from IPython.display import display, clear_output
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import os
from scipy.linalg import lstsq
from scipy.optimize import minimize_scalar
from scipy.stats import beta, weibull_min


###############################################################################


def make_scatter(X, y, lines=[], colors=[], styles=[], labels=[], to_save=False, img_name=None, img_path=None):
    x_data = X.reshape(-1)
    y_data = y.reshape(-1)
    fig, ax = plt.subplots(1, 1, figsize=(7.5,5))
    ax.scatter(x_data, y_data, color="xkcd:black")
    ax.set_xticks([0.0, np.max(x_data)])
    ax.set_xticklabels(["0", r"$x_{\max}$"])
    ax.set_yticks([0.0, np.max(y_data)])
    ax.set_yticklabels(["0", r"$y_{\max}$"])
    if len(lines) > 0:
        for i in range(len(lines)):
            w_slope, w_intercept = lines[i]
            x_forline = np.linspace(np.min(x_data), np.max(x_data), 100)
            y_regline = x_forline*w_slope + w_intercept
            ax.plot(x_forline, y_regline, color=colors[i], ls=styles[i], label=labels[i])
        ax.legend(loc="upper left")
    plt.tight_layout()
    if to_save:
        fname = path.join(img_dir, img_name+".pdf")
        plt.savefig(fname=fname)
    plt.show()


def make_hist(losses_quad, losses_abs, colors, tops=None, to_save=False, img_name=None, img_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,3), sharey=False)
    ax1.hist(losses_quad,
             bins="auto", color=colors, histtype="barstacked",
             fill=True, hatch=None, orientation="vertical", density=False)
    for i in range(len(losses_quad)):
        ax1.axvline(x=np.mean(losses_quad[i]), color=colors[i], ls="dashed")
        ax1.axvline(x=np.median(losses_quad[i]), color=colors[i], ls="dotted")
    ax2.hist(losses_abs,
             bins="auto", color=colors, histtype="barstacked",
             fill=True, hatch=None, orientation="vertical", density=False)
    for i in range(len(losses_abs)):
        ax2.axvline(x=np.mean(losses_abs[i]), color=colors[i], ls="dashed")
        ax2.axvline(x=np.median(losses_abs[i]), color=colors[i], ls="dotted")
    if tops is not None:
        ax1.set_ylim(top=tops[0])
        ax2.set_ylim(top=tops[1])
        ax1.set_title("Squared error (zoomed-in)")
        ax2.set_title("Absolute error (zoomed-in)")
    else:
        ax1.set_title("Squared error")
        ax2.set_title("Absolute error")
    plt.tight_layout()
    if to_save:
        fname = path.join(img_dir, img_name+".pdf")
        plt.savefig(fname=fname)
    plt.show()


def obj_leastmedloss(w, x, y, loss="abs"):
    residuals = w[0]*x + w[1] - y
    if loss == "abs":
        losses = np.abs(residuals)
    else:
        losses = (residuals)**2
    return np.median(losses)


def test_data_linear(X, y, rg):
    # For reference, get the inlying and outlying Y indices, and various other info.
    y_outlier_idx = y.reshape(-1) >= np.quantile(a=y.reshape(-1), q=0.75)
    y_inlier_idx = y.reshape(-1) <= np.quantile(a=y.reshape(-1), q=0.75)
    x_outlier_range = (np.min(X.reshape(-1)[y_outlier_idx]),
                       np.max(X.reshape(-1)[y_outlier_idx]))
    
    # OLS solution *without* the outlying points.
    X_padded = np.hstack([X, np.ones(X.shape, dtype=X.dtype)])
    regline_removed, res_removed, rnk_removed, s_removed = lstsq(a=X_padded[y_inlier_idx,:],
                                                                 b=y[y_inlier_idx,:])
    # Data for the "essentially linear" scenario.
    x_offsample = rg.uniform(low=0.0, high=np.max(X), size=500)
    y_offsample_lin = x_offsample*regline_removed[0] + regline_removed[1]
    y_offsample_lin += rg.normal(loc=0.0, scale=5.0, size=len(x_offsample))
    num_to_perturb = 10
    can_perturb = np.logical_and(x_offsample > x_outlier_range[0],
                                x_offsample < x_outlier_range[1])
    to_perturb = np.argwhere(can_perturb).reshape(-1)[0:num_to_perturb]
    y_offsample_lin[to_perturb] += rg.normal(loc=150.0, scale=25.0, size=len(to_perturb))
    return x_offsample, y_offsample_lin


def test_data_nonlinear(X, y, rg):
    # Data for the "essentially non-linear" scenario.
    x_offsample = rg.uniform(low=0.0, high=np.max(X), size=500)
    nearest_ys = []
    for i in range(len(x_offsample)):
        x_tocheck = x_offsample[i]
        nearest_x_idx = np.argmin(np.abs(X-x_tocheck))
        nearest_ys += [y[nearest_x_idx,0]]
    y_offsample_nonlin = np.array(nearest_ys) + rg.normal(loc=0.0, scale=15.0, size=len(x_offsample))
    return x_offsample, y_offsample_nonlin


def make_scatter_traintest(X, y, rg, lines=[], colors=[], styles=[], labels=[],
                           to_save=False, img_name=None, img_path=None):
    x_data = X.reshape(-1)
    y_data = y.reshape(-1)
    x_data_linear, y_data_linear = test_data_linear(X=X, y=y, rg=rg)
    x_data_nonlinear, y_data_nonlinear = test_data_nonlinear(X=X, y=y, rg=rg)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,5), sharey=True)
    ax1.scatter(x_data_linear, y_data_linear, color="xkcd:gray")
    ax1.scatter(x_data, y_data, color="xkcd:black")
    ax1.set_xticks([0.0, np.max(x_data)])
    ax1.set_xticklabels(["0", r"$x_{\max}$"])
    ax1.set_yticks([0.0, np.max(y_data)])
    ax1.set_yticklabels(["0", r"$y_{\max}$"])
    ax2.scatter(x_data_nonlinear, y_data_nonlinear, color="xkcd:gray")
    ax2.scatter(x_data, y_data, color="xkcd:black")
    ax2.set_xticks([0.0, np.max(x_data)])
    ax2.set_xticklabels(["0", r"$x_{\max}$"])
    ax2.set_yticks([0.0, np.max(y_data)])
    ax2.set_yticklabels(["0", r"$y_{\max}$"])
    if len(lines) > 0:
        for i in range(len(lines)):
            w_slope, w_intercept = lines[i]
            x_forline = np.linspace(np.min(x_data), np.max(x_data), 100)
            y_regline = x_forline*w_slope + w_intercept
            ax1.plot(x_forline, y_regline, color=colors[i], ls=styles[i], label=labels[i])
            ax2.plot(x_forline, y_regline, color=colors[i], ls=styles[i], label=labels[i])
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper left")
    plt.tight_layout()
    if to_save:
        fname = path.join(img_dir, img_name+".pdf")
        plt.savefig(fname=fname)
    plt.show()


def make_dispersion_plot(to_save=False, img_name=None, img_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(7.5,5))
    x_values = np.linspace(-8.0,8.0,100)
    ax.axhline(y=-np.pi/2, color="xkcd:silver")
    ax.axhline(y=np.pi/2, color="xkcd:silver")
    ax.axhline(y=0.0, color="xkcd:silver")
    ax.axvline(x=0.0, color="xkcd:silver")
    ax.plot(x_values, dispersion_holland(x=x_values), color="xkcd:black", ls="-", label=r"$\rho_{0}$")
    ax.plot(x_values, d1_holland(x=x_values), color="xkcd:black", ls="--", label=r"$\rho_{0}^{\prime}$")
    ax.plot(x_values, d2_holland(x=x_values), color="xkcd:black", ls="-.", label=r"$\rho_{0}^{\prime\prime}$")
    ax.legend(loc="upper center", ncol=3)
    ax.set_yticks([-np.pi/2, 0.0,  np.pi/2])
    ax.set_yticklabels([r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$"])
    plt.tight_layout()
    if to_save:
        fname = path.join(img_dir, img_name+".pdf")
        plt.savefig(fname=fname)
    plt.show()


def dispersion_barron(x, alpha):
    '''
    This is the Barron-type function rho(x;alpha).
    '''

    if alpha == 2.0:
        return x**2/2.0
        
    elif alpha == 0.0:
        return np.log1p(x**2/2.0)
    
    elif alpha == -np.inf:
        return 1.0 - np.exp(-x**2/2.0)
        
    else:
        adiff = np.absolute(alpha-2.0)
        ahalf = alpha/2.0
        return (adiff/alpha) * ((1.0+x**2/adiff)**ahalf - 1.0)


def d1_barron(x, alpha):
    '''
    Returns the first derivative of dispersion_barron,
    taken with respect to the first argument.
    '''

    if alpha == 2.0:
        return x
        
    elif alpha == 0.0:
        return x / (1.0 + x**2/2.0)
    
    elif alpha == -np.inf:
        return np.exp(-x**2/2.0) * x
        
    else:
        adiff = np.absolute(alpha-2.0)
        ahalf = alpha/2.0
        return x * (1.0+x**2/adiff)**(ahalf-1.0)


def d2_barron(x, alpha):
    '''
    Returns the second derivative of dispersion_barron,
    taken with respect to the first argument.
    '''

    if alpha == 2.0:
        return np.ones_like(a=x)
        
    elif alpha == 0.0:
        return 1.0/(1.0+x**2/2.0) - (x/(1.0+x**2/2.0))**2
    
    elif alpha == -np.inf:
        return np.exp(-x**2/2.0) * (1.0 - x**2)
        
    else:
        adiff = np.absolute(alpha-2.0)
        ahalf = alpha/2.0
        innards = 1.0+x**2/adiff
        term1 = innards**(ahalf-1.0)
        term2 = (x**2) * (ahalf-1.0) * innards**(ahalf-2.0) * (2/adiff)
        return term1 + term2


def dispersion_huber(x, beta):
    '''
    This is the penalized Huber function rho(x;beta).
    '''
    return beta + (np.sqrt(1.0+x**2)-1.0)/beta


def d1_huber(x, beta):
    '''
    Returns the first derivative of dispersion_huber,
    taken with respect to the first argument.
    '''
    return x / (beta * np.sqrt(1.0+x**2))


def d2_huber(x, beta):
    '''
    Returns the second derivative of dispersion_huber,
    taken with respect to the first argument.
    '''
    return 1.0 / (beta * (1.0+x**2)**(1.5))


def dispersion_holland(x):
    '''
    This is the dispersion function rho(x) used by Holland (2021).
    '''
    return x * np.arctan(x) - np.log1p(x**2)/2.0


def d1_holland(x):
    '''
    Returns the first derivative of dispersion_holland.
    '''
    return np.arctan(x)


def d2_holland(x):
    '''
    Returns the second derivative of dispersion_holland.
    '''
    return 1.0 / (1.0+x**2)


def dispersion_barron_autoset(x, alpha, sigma, interpolate=False,
                              oce_flag=False, eta_custom=None):
    '''
    Barron-type dispersion with automatic eta settings.
    '''

    ## Use a custom eta value if provided, otherwise set automatically.
    if eta_custom is None:
        eta = eta_setter_barron(sigma=sigma,
                                alpha=alpha,
                                interpolate=interpolate,
                                oce_flag=oce_flag)
    else:
        eta = eta_custom
    
    ## Compute the re-scaled dispersion values.
    if sigma <= 0.0 or sigma == np.inf:
        raise ValueError("Only finite positive sigma are allowed.")
    else:
        return eta * dispersion_barron(x=x/sigma, alpha=alpha)


def d1_barron_autoset(x, alpha, sigma, interpolate=False,
                      oce_flag=False, eta_custom=None):
    '''
    Barron-type dispersion with automatic eta settings.
    '''

    ## Use a custom eta value if provided, otherwise set automatically.
    if eta_custom is None:
        eta = eta_setter_barron(sigma=sigma,
                                alpha=alpha,
                                interpolate=interpolate,
                                oce_flag=oce_flag)
    else:
        eta = eta_custom
    
    ## Compute the re-scaled dispersion values.
    if sigma <= 0.0 or sigma == np.inf:
        raise ValueError("Only finite positive sigma are allowed.")
    else:
        return eta * d1_barron(x=x/sigma, alpha=alpha) / sigma


def dispersion_holland_autoset(x, sigma, interpolate=False,
                               oce_flag=False, eta_custom=None):
    '''
    Holland-type dispersion with automatic eta settings.
    '''
    ## Use a custom eta value if provided, otherwise set automatically.
    if eta_custom is None:
        eta = eta_setter_holland(sigma=sigma,
                                 interpolate=interpolate,
                                 oce_flag=oce_flag)
    else:
        eta = eta_custom

    ## Compute the re-scaled dispersion values.
    if sigma < 0:
        raise ValueError("Only non-negative sigma values are allowed.")
    elif sigma == 0.0:
        return eta * np.absolute(x=x)
    elif sigma == np.inf:
        return eta * x**2
    else:
        return eta * dispersion_holland(x=x/sigma)


def d1_holland_autoset(x, sigma, interpolate=False,
                       oce_flag=False, eta_custom=None):
    '''
    First derivative of dispersion_holland_autoset.
    '''
    ## Use a custom eta value if provided, otherwise set automatically.
    if eta_custom is None:
        eta = eta_setter_holland(sigma=sigma,
                                 interpolate=interpolate,
                                 oce_flag=oce_flag)
    else:
        eta = eta_custom
    
    ## Compute the re-scaled dispersion values.
    if sigma < 0:
        raise ValueError("Only non-negative sigma values are allowed.")
    elif sigma == 0.0:
        return eta * np.sign(x=x)
    elif sigma == np.inf:
        return eta * 2 * x
    else:
        return eta * d1_holland(x=x/sigma) / sigma


def get_dispersion(name, **kwargs):
    '''
    Simplest parser, returns a dispersion function
    and its derivative, no fancy auto-setting.
    Note that the derivatives are computed *before*
    scaling using sigma, thus there is no 1/sigma
    factor here.
    '''
    if name == "barron":
        dispersion = lambda x, sigma: dispersion_barron(
            x=x/sigma, alpha=kwargs["alpha"]
        )
        dispersion_d1 = lambda x, sigma: d1_barron(
            x=x/sigma, alpha=kwargs["alpha"]
        )
    elif name == "barron1way":
        dispersion = lambda x, sigma: dispersion_barron(
            x=np.where(x>0.0, x/sigma, 0.0), alpha=kwargs["alpha"]
        )
        dispersion_d1 = lambda x, sigma: np.where(
            x>0.0,
            d1_barron(x=np.where(x>0.0, x/sigma, 0.0),
                      alpha=kwargs["alpha"]),
            0.0
        )
    else:
        raise ValueError("Please provide a valid dispersion name.")
    
    return (dispersion, dispersion_d1)


def get_dispersion_autoset(name, **kwargs):
    '''
    A parser that returns a dispersion function
    and its derivative, with appropriate scaling
    and weighting.
    '''
    if name == "barron":
        dispersion = lambda x, sigma, eta: dispersion_barron_autoset(
            x=x, alpha=kwargs["alpha"], sigma=sigma,
            interpolate=kwargs["interpolate"],
            oce_flag=kwargs["oce_flag"],
            eta_custom=eta
        )
        dispersion_d1 = lambda x, sigma, eta: d1_barron_autoset(
            x=x, alpha=kwargs["alpha"], sigma=sigma,
            interpolate=kwargs["interpolate"],
            oce_flag=kwargs["oce_flag"],
            eta_custom=eta
        )
    elif name == "barron1way":
        dispersion = lambda x, sigma, eta: dispersion_barron_autoset(
            x=np.where(x>0.0, x, 0.0),
            alpha=kwargs["alpha"], sigma=sigma,
            interpolate=kwargs["interpolate"],
            oce_flag=kwargs["oce_flag"],
            eta_custom=eta
        )
        dispersion_d1 = lambda x, sigma, eta: d1_barron_autoset(
            x=np.where(x>0.0, x, 0.0),
            alpha=kwargs["alpha"], sigma=sigma,
            interpolate=kwargs["interpolate"],
            oce_flag=kwargs["oce_flag"],
            eta_custom=eta
        ) * np.where(x>0.0, 1.0, 0.0)
    elif name == "huber":
        # the penalized pseudo-Huber function case.
        #beta = kwargs["beta"]
        raise NotImplementedError
    elif name == "holland":
        dispersion = lambda x, sigma, eta: dispersion_holland_autoset(
            x=x, sigma=sigma, interpolate=kwargs["interpolate"],
            oce_flag=kwargs["oce_flag"],
            eta_custom=eta
        )
        dispersion_d1 = lambda x, sigma, eta: d1_holland_autoset(
            x=x, sigma=sigma, interpolate=kwargs["interpolate"],
            oce_flag=kwargs["oce_flag"],
            eta_custom=eta
        )
    else:
        raise ValueError("Please provide a valid dispersion name.")

    return (dispersion, dispersion_d1)


def gen_data(n, name, rg, **kwargs):
    '''
    Function for generating data.
    '''
    if name == "bernoulli":
        prob = kwargs["prob"]
        x = rg.uniform(low=0.0, high=1.0, size=(n,1))
        return np.where(x <= prob, 1.0, 0.0)
    elif name == "beta":
        a = 1.0
        b = kwargs["b"]
        return rg.beta(a=a, b=b, size=(n,1))
    elif name == "chisquare":
        df = 3.5
        return rg.chisquare(df=df, size=(n,1))
    elif name == "exponential":
        scale = 1.0
        return rg.exponential(scale=scale, size=(n,1))
    elif name == "gamma":
        shape, scale = (4.0, 1.0)
        return rg.gamma(shape=shape, scale=scale, size=(n,1))
    elif name == "lognormal":
        mean, sigma = (0.0, 0.5)
        return rg.lognormal(mean=mean, sigma=sigma, size=(n,1))
    elif name == "normal":
        loc = 0.0
        scale = kwargs["scale"]
        return rg.normal(loc=loc, scale=scale, size=(n,1))
    elif name == "pareto":
        a = kwargs["a"]
        return rg.pareto(a=a, size=(n,1))
    elif name == "uniform":
        low, high = (-0.5, 0.5)
        return rg.uniform(low=low, high=high, size=(n,1))
    elif name == "wald":
        mean, scale = (1.0, 1.0)
        return rg.wald(mean=mean, scale=scale, size=(n,1))
    elif name == "weibull":
        a = 1.2
        return rg.weibull(a=a, size=(n,1))
    else:
        return None


# Objective functions for each (relevant) criterion class.

def obfn_rrisk(theta, x, paras):
    '''
    For our regularized risk.
    '''
    eta = paras["eta"]
    dispersion = get_disp_barron(alpha=paras["alpha"])
    sigma = paras["sigma"]
    return np.mean(x) + eta * np.mean(dispersion(x=(x-theta)/sigma))


def obfn_trisk(theta, x, paras):
    '''
    For our threshold risk class.
    '''
    etatilde = paras["etatilde"]
    dispersion = get_disp_barron(alpha=paras["alpha"])
    sigma = paras["sigma"]
    return np.mean(dispersion(x=(x-theta)/sigma)) + etatilde * theta


def obfn_trisk1way(theta, x, paras):
    '''
    For our threshold risk class (one-directional).
    '''
    etatilde = paras["etatilde"]
    dispersion = get_disp_barron(alpha=paras["alpha"], oneway=True)
    sigma = paras["sigma"]
    return np.mean(dispersion(x=(x-theta)/sigma)) + etatilde * theta


def obfn_median(theta, x, paras):
    '''
    A trivial guy, just force it to return the median.
    '''
    return np.absolute(theta-np.median(x))


def obfn_cvar(theta, x, paras):
    '''
    For CVaR risk.
    '''
    prob = paras["prob"]
    return theta + np.mean(np.where(x >= theta, x-theta, 0.0)) / (1.0-prob)


def obfn_entropic(theta, x, paras):
    '''
    For entropic risk (assuming positive gamma).
    '''
    gamma = paras["gamma"]
    if gamma == 0.0:
        return np.absolute(theta-np.mean(x))
    elif gamma < 0.0:
        ## Since know the closed form, force it to take this value.
        return np.absolute(theta-get_entropic(x=x, gamma=gamma))
    else:
        ## This OCE-type objective only works for positive gamma.
        return theta + (np.mean(np.exp(gamma*(x-theta)))-1.0)/gamma


def obfn_dro(theta, x, paras):
    '''
    For the DRO risk we consider.
    '''
    shape = 2.0
    bound = 0.5*(1.0/(1.0-paras["atilde"])-1.0)**2
    sstar = shape / (shape-1.0) # shape-star.
    factor = (1.0 + shape*(shape-1.0)*bound)**(1.0/shape)
    return theta + factor * np.mean(np.where(x >= theta, x-theta, 0.0)**sstar)**(1.0/sstar)


def obfn_holland_tomean(theta, x, paras):
    '''
    M-location using Holland dispersion function,
    with re-scaling done to approach the mean as
    sigma gets large.
    '''
    sigma = paras["sigma"]
    eta = 2*sigma**2
    return eta * np.mean(dispersion_holland(x=(x-theta)/sigma))


def obfn_holland_tomed(theta, x, paras):
    '''
    M-location using Holland dispersion function,
    with re-scaling done to approach the mean as
    sigma gets large.
    '''
    sigma = paras["sigma"]
    eta = 2*sigma/np.pi
    return eta * np.mean(dispersion_holland(x=(x-theta)/sigma))


def bracket_prep(x, paras, obfn_name, verbose):
    
    x_init = np.mean(x)
    x_low = np.amin(x)
    x_high = np.amax(x)
    
    ## Prepare the relevant objective function.
    if obfn_name == "rrisk":
        obfn = lambda theta: obfn_rrisk(theta=theta, x=x, paras=paras)
    elif obfn_name == "trisk":
        obfn = lambda theta: obfn_trisk(theta=theta, x=x, paras=paras)
    elif obfn_name == "trisk1way":
        obfn = lambda theta: obfn_trisk1way(theta=theta, x=x, paras=paras)
    elif obfn_name == "trisk-median":
        obfn = lambda theta: obfn_median(theta=theta, x=x, paras=paras)
    elif obfn_name == "cvar":
        obfn = lambda theta: obfn_cvar(theta=theta, x=x, paras=paras)
    elif obfn_name == "entropic":
        obfn = lambda theta: obfn_entropic(theta=theta, x=x, paras=paras)
    elif obfn_name == "dro":
        obfn = lambda theta: obfn_dro(theta=theta, x=x, paras=paras)
    elif obfn_name == "mestToMean":
        obfn = lambda theta: obfn_holland_tomean(theta=theta, x=x, paras=paras)
    elif obfn_name == "mestToMed":
        obfn = lambda theta: obfn_holland_tomed(theta=theta, x=x, paras=paras)
    else:
        raise ValueError("Please pass a valid obfn_name.")
    
    # Compute brackets.
    f_init = obfn(theta=x_init)
    f_low = obfn(theta=x_low)
    f_high = obfn(theta=x_high)
    while f_low < f_init:
        x_low -= np.absolute(x_init) + np.absolute(x_low)
        f_low = obfn(theta=x_low)
        if verbose:
            print("Bracket prep ({}): extending MIN side.".format(obfn_name))
            
    while f_high < f_init:
        x_high += np.absolute(x_init) + np.absolute(x_high)
        f_high = obfn(theta=x_high)
        if verbose:
            print("Bracket prep ({}): extending MAX side.".format(obfn_name))
    
    cond_bracket = (f_low > f_init) and (f_high > f_init)
    if cond_bracket == False:
        print("Warning: bracket condition is", cond_bracket)
        print("Details:", f_low, f_init, f_high)
    
    return (x_low, x_init, x_high)


def get_obfn(name):
    '''
    A simple wrapper for parsing "obfn" functions.
    '''
    if name == "rrisk":
        return obfn_rrisk
    elif name == "trisk":
        return obfn_trisk
    elif name == "trisk1way":
        return obfn_trisk1way
    elif name == "trisk-median":
        return obfn_median
    elif name == "cvar":
        return obfn_cvar
    elif name == "entropic":
        return obfn_entropic
    elif name == "dro":
        return obfn_dro
    elif name == "mestToMean":
        return obfn_holland_tomean
    elif name == "mestToMed":
        return obfn_holland_tomed
    else:
        raise ValueError("Please provide a proper obfn name.")


def get_entropic(x, gamma):
    '''
    A direct computation of the entropic risk.
    '''
    return np.log(np.mean(np.exp(gamma*x))) / gamma


def get_disp_barron(alpha, oneway=False):
    '''
    A simple wrapper for specifying Barron-type dispersion functions.
    Take shape parameter alpha, and return the desired function.
    '''
    if oneway:
        return lambda x: dispersion_barron(x=np.where(x>0.0, x, 0.0),
                                           alpha=alpha)
    else:
        return lambda x: dispersion_barron(x=x, alpha=alpha)


def make_criterion_plot(data, rg, criteria, data_kwargs, to_save=False, img_name=None, img_path=None):
    
    # Clerical parameters.
    n = 10000
    tolerance = 1e-10
    verbose = False
    transparency = 0.1
    flip_data = False # specify whether or not to flip the sign of the data.
    data_all = ["bernoulli", "beta", "chisquare", "exponential", "gamma",
                "lognormal", "normal", "pareto", "uniform",
                "wald", "weibull"]
    data_bounded = ["bernoulli", "beta", "uniform"]
    data_unbounded = ["chisquare", "exponential", "gamma", "lognormal",
                      "normal", "pareto", "wald", "weibull"]
    data_heavytails = ["chisquare", "exponential", "lognormal",
                       "pareto", "wald", "weibull"]
    data_symmetric = ["normal", "uniform"]

    todo_riskparas = {
        "rrisk": ("alpha", np.linspace(1.0, 2.0, 200)),
        "trisk": ("alpha", np.linspace(1.0, 2.0, 200)),
        "trisk-median": ("alpha", np.linspace(-2.0, 2.0, 200)),
        "cvar": ("prob", np.linspace(0.025, 0.975, 200)),
        "entropic": ("gamma", np.concatenate([np.linspace(-2.0, 2.0, 200)])),
        "dro": ("atilde", np.linspace(0.025, 0.975, 200)),
        "mestToMean": ("sigma", np.linspace(0.005, 2.0, 200)),
        "mestToMed": ("sigma", np.linspace(0.005, 2.0, 200))
    }
    aux_riskparas = {
        "rrisk": {"sigma": 0.5, "eta": 1.0},
        "trisk": {"sigma": 0.5, "etatilde": 0.99},
        "trisk-median": {"sigma": 0.5, "etatilde": 1.0},
        "cvar": {},
        "entropic": {},
        "dro": {},
        "mestToMean": {},
        "mestToMed": {}
    }

    # Setup of labels, ticks, and titles.
    xlabels = {"rrisk": r"$\alpha$ value",
               "trisk": r"$\alpha$ value",
               "trisk-median": r"$\alpha$ value",
               "cvar": r"$\beta$ value",
               "entropic": r"$\gamma$ value",
               "dro": r"$\widetilde{a}$ value",
               "mestToMean": r"$\sigma$ value",
               "mestToMed": r"$\sigma$ value"}
    titles = {"rrisk": "M-risk",
              "trisk": "T-risk (minimal)",
              "trisk-median": "T-risk (median)",
              "cvar": "CVaR risk",
              "entropic": "Tilted risk",
              "dro": r"$\chi^{2}$-DRO risk",
              "mestToMean": "M-location (mean scale)",
              "mestToMed": "M-location (median scale)"}
    xticks = {"rrisk": [np.amin(todo_riskparas["rrisk"][1]), np.amax(todo_riskparas["rrisk"][1])],
              "trisk": [np.amin(todo_riskparas["trisk"][1]), np.amax(todo_riskparas["trisk"][1])],
              "trisk-median": [np.amin(todo_riskparas["trisk-median"][1]), np.amax(todo_riskparas["trisk-median"][1])],
              "cvar": [0.0, 1.0],
              "entropic": [np.amin(todo_riskparas["entropic"][1]), 0.0, np.amax(todo_riskparas["entropic"][1])],
              "dro": [0.0, 1.0],
              "mestToMean": [np.amin(todo_riskparas["mestToMean"][1]), np.amax(todo_riskparas["mestToMean"][1])],
              "mestToMed": [np.amin(todo_riskparas["mestToMed"][1]), np.amax(todo_riskparas["mestToMed"][1])]}
    xticklabels = {"rrisk": [str(tick) for tick in xticks["rrisk"]],
                   "trisk": [str(tick) for tick in xticks["trisk"]],
                   "trisk-median": [str(tick) for tick in xticks["trisk-median"]],
                   "cvar": [str(tick) for tick in xticks["cvar"]],
                   "entropic": [str(tick) for tick in xticks["entropic"]],
                   "dro": [str(tick) for tick in xticks["dro"]],
                   "mestToMean": [str(tick) for tick in xticks["mestToMean"]],
                   "mestToMed": [str(tick) for tick in xticks["mestToMed"]]}
    vlines = {"rrisk": xticks["rrisk"],
              "trisk": xticks["trisk"],
              "trisk-median": xticks["trisk-median"],
              "cvar": xticks["cvar"],
              "entropic": xticks["entropic"],
              "dro": xticks["dro"],
              "mestToMean": xticks["mestToMean"],
              "mestToMed": xticks["mestToMed"]}

    # Setup of storage dictionaries.
    fun_values = { crit: {} for crit in criteria }
    sol_values = { crit: {} for crit in criteria }
    
    # Get basic statistics of the data distribution.
    x_values = gen_data(n=n, name=data, rg=rg, **data_kwargs)
    x_values = x_values - np.mean(x_values) # center the data, so we get positive and negative values.
    if flip_data:
        x_values = -x_values # flip the signs if desired.
    x_mean = np.mean(x_values)
    x_quantiles = np.quantile(a=x_values, q=[0.01, 0.25, 0.5, 0.75, 0.99])
    x_q01 = x_quantiles[0]
    x_q25 = x_quantiles[1]
    x_q50 = x_quantiles[2]
    x_q75 = x_quantiles[3]
    x_q99 = x_quantiles[4]
    x_min = np.amin(x_values)
    x_max = np.amax(x_values)
    
    # Ticks on the vertical axis are kept as simple as possible.
    yticks = [0.0]
    ytick_labels = [str(0.0)]

    # Put a bit of effort into ensuring the visuals are clear.
    if data in data_bounded:
        # If bounded, show all the data.
        ylim_top = np.amax(np.absolute(x_values))*(1.05) + 0.1
        ylim_bottom = -ylim_top
    else:
        # Otherwise, use max values as a guide.
        ylim_top = np.amax(np.absolute(x_values))
        ylim_bottom = -ylim_top
        if data in data_heavytails:
            ylim_top = x_quantiles[-1]
            ylim_bottom = -x_quantiles[-1]
    
    # Loop over criterion classes.
    for crit in criteria:
        
        pname, riskparas = todo_riskparas[crit]
        aux_paras = aux_riskparas[crit]
        
        # Prepare storage.
        fun_values[crit][data] = []
        sol_values[crit][data] = []
        
        # Loop over criterion hyperparameters.
        for riskpara in riskparas:
            
            # Set remaining parameter to be passed.
            aux_paras[pname] = riskpara
            
            # Preparation of brackets for optimization.
            bracket_low, bracket_mid, bracket_high = bracket_prep(
                x=x_values, paras=aux_paras, obfn_name=crit, verbose=verbose
            )
            
            # Prepare the objective function.
            obfn = get_obfn(name=crit)
            
            # Run the optimization.
            opt_result = minimize_scalar(fun=obfn,
                                         bracket=(bracket_low, bracket_mid, bracket_high),
                                         bounds=(bracket_low, bracket_high),
                                         args=(x_values, aux_paras),
                                         method="bounded",
                                         options={"xatol": tolerance})
            
            # Store the result.
            sol_values[crit][data] += [opt_result.x]
            if crit == "trisk-median":
                obfn_aux = get_obfn(name="trisk")
                fun_values[risk][data] += [obfn_aux(theta=opt_result.x, x=x_values, paras=aux_paras)]
            elif crit == "entropic":
                if aux_paras["gamma"] <= 0.0:
                    fun_values[crit][data] += [get_entropic(x=x_values, gamma=aux_paras["gamma"])]
                else:
                    fun_values[crit][data] += [opt_result.fun]
            else:
                fun_values[crit][data] += [opt_result.fun]
            
            
        # After this inner loop, store in arrays.
        fun_values[crit][data] = np.copy(np.array(fun_values[crit][data]))
        sol_values[crit][data] = np.copy(np.array(sol_values[crit][data]))
    
    
    # Put together a multi-axis figure.
    fig, axes = plt.subplots(1, 1+len(criteria), figsize=(3*(1+len(criteria)),3),
                             gridspec_kw={"width_ratios": [2]*len(criteria) + [1]},
                             sharey=True)
    
    # Organize the axes.
    axes_risk = axes[0:-1]
    ax_data = axes[-1]
    
    # Plot lines or ticks that appear on all the criteria plots.
    for ax in axes_risk:
        ax.axhline(y=x_q25, color="xkcd:silver")
        ax.axhline(y=x_q75, color="xkcd:silver")
        ax.axhline(y=x_q50, color="xkcd:red")
        ax.axhline(y=x_mean, color="xkcd:blue")
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
    
    # Plot threshold and criteria values for each class.
    for i, crit in enumerate(criteria):
        ax = axes[i]
        ax.plot(todo_riskparas[crit][1], sol_values[crit][data], color="xkcd:black", ls="solid")
        ax.plot(todo_riskparas[crit][1], fun_values[crit][data], color="xkcd:black", ls="dashed")
        ax.fill_between(
            x=todo_riskparas[crit][1], y1=sol_values[crit][data], y2=fun_values[crit][data],
            alpha=transparency, color="xkcd:black", lw=0
        )
        for vline in vlines[crit]:
            ax.axvline(vline, color="xkcd:silver", ls="dotted")
        ax.set_title(titles[crit])
        ax.set_xlabel(xlabels[crit])
        ax.set_xticks(xticks[crit])
        ax.set_xticklabels(xticklabels[crit])
    
    # Plot the data histogram.
    ax_data.hist(x_values,
                 bins="auto", color="black", hatch=None,
                 orientation="horizontal", density=False)
    ax_data.set_xlabel("Frequency")
    ax_data.set_title(data)
    
    # Set the vertical axis limits.
    for ax in axes:
        ax.set_ylim(top=ylim_top, bottom=ylim_bottom)
    
    # Output the figure.
    plt.tight_layout()
    if to_save:
        if flip_data:
            fname = path.join(img_dir, img_name+"_flipped.pdf")
        else:
            fname = path.join(img_dir, img_name+".pdf")
        plt.savefig(fname=fname)
    plt.show()


def make_trisk_plot(data, rg, data_kwargs, to_save=False, img_name=None, img_path=None):
    
    # Clerical parameters.
    criteria = ["trisk_alpha", "trisk_sigma", "trisk_etatilde"]
    n = 10000
    tolerance = 1e-10
    verbose = False
    transparency = 0.1
    flip_data = False # specify whether or not to flip the sign of the data.
    data_all = ["bernoulli", "beta", "chisquare", "exponential", "gamma",
                "lognormal", "normal", "pareto", "uniform",
                "wald", "weibull"]
    data_bounded = ["bernoulli", "beta", "uniform"]
    data_unbounded = ["chisquare", "exponential", "gamma", "lognormal",
                      "normal", "pareto", "wald", "weibull"]
    data_heavytails = ["chisquare", "exponential", "lognormal",
                       "pareto", "wald", "weibull"]
    data_symmetric = ["normal", "uniform"]

    todo_riskparas = {
        "trisk_alpha": ("alpha", np.linspace(1.0, 2.0, 200)),
        "trisk_sigma": ("sigma", np.linspace(0.05, 1.5, 200)),
        "trisk_etatilde": ("etatilde", np.linspace(-1.0, 1.0, 200))
    }
    aux_riskparas = {
        "trisk_alpha": {"sigma": 0.5, "etatilde": 1.0},
        "trisk_sigma": {"alpha": 1.0, "etatilde": (1.0/1.5)-0.01},
        "trisk_etatilde": {"sigma": 0.99, "alpha": 1.0}
    }

    # Setup of labels, ticks, and titles.
    xlabels = {"trisk_alpha": r"$\alpha$ value",
               "trisk_sigma": r"$\sigma$ value",
               "trisk_etatilde": r"$\eta$ value"} # the paper now uses eta, not etatilde.
    titles = {"trisk_alpha": "T-risk (minimal)",
              "trisk_sigma": "T-risk (minimal)",
              "trisk_etatilde": "T-risk (minimal)"}
    xticks = {"trisk_alpha": [np.amin(todo_riskparas["trisk_alpha"][1]),
                              np.amax(todo_riskparas["trisk_alpha"][1])],
              "trisk_sigma": [np.amin(todo_riskparas["trisk_sigma"][1]),
                              np.amax(todo_riskparas["trisk_sigma"][1])],
              "trisk_etatilde": [np.amin(todo_riskparas["trisk_etatilde"][1]),
                                 np.amax(todo_riskparas["trisk_etatilde"][1])]}
    xticklabels = {"trisk_alpha": [str(tick) for tick in xticks["trisk_alpha"]],
                   "trisk_sigma": [str(tick) for tick in xticks["trisk_sigma"]],
                   "trisk_etatilde": [str(tick) for tick in xticks["trisk_etatilde"]]}
    vlines = {"trisk_alpha": xticks["trisk_alpha"],
              "trisk_sigma": xticks["trisk_sigma"],
              "trisk_etatilde": xticks["trisk_etatilde"]}

    # Setup of storage dictionaries.
    fun_values = { crit: {} for crit in criteria }
    sol_values = { crit: {} for crit in criteria }
    
    # Get basic statistics of the data distribution.
    x_values = gen_data(n=n, name=data, rg=rg, **data_kwargs)
    x_values = x_values - np.mean(x_values) # center the data, so we get positive and negative values.
    if flip_data:
        x_values = -x_values # flip the signs if desired.
    x_mean = np.mean(x_values)
    x_quantiles = np.quantile(a=x_values, q=[0.01, 0.25, 0.5, 0.75, 0.99])
    x_q01 = x_quantiles[0]
    x_q25 = x_quantiles[1]
    x_q50 = x_quantiles[2]
    x_q75 = x_quantiles[3]
    x_q99 = x_quantiles[4]
    x_min = np.amin(x_values)
    x_max = np.amax(x_values)
    
    # Ticks on the vertical axis are kept as simple as possible.
    yticks = [0.0]
    ytick_labels = [str(0.0)]

    # Put a bit of effort into ensuring the visuals are clear.
    if data in data_bounded:
        # If bounded, show all the data.
        ylim_top = np.amax(np.absolute(x_values))*(1.05) + 0.1
        ylim_bottom = -ylim_top
    else:
        # Otherwise, use max values as a guide.
        ylim_top = np.amax(np.absolute(x_values))
        ylim_bottom = -ylim_top
        if data in data_heavytails:
            ylim_top = x_quantiles[-1]
            ylim_bottom = -x_quantiles[-1]
    
    # Loop over criterion classes.
    for crit in criteria:
        
        pname, riskparas = todo_riskparas[crit]
        aux_paras = aux_riskparas[crit]
        
        # Prepare storage.
        fun_values[crit][data] = []
        sol_values[crit][data] = []
        
        # Loop over criterion hyperparameters.
        for riskpara in riskparas:
            
            # Set remaining parameter to be passed.
            aux_paras[pname] = riskpara
            
            # Preparation of brackets for optimization.
            obfn_name = crit.split("_")[0] # special parsing here.
            bracket_low, bracket_mid, bracket_high = bracket_prep(
                x=x_values, paras=aux_paras, obfn_name=obfn_name, verbose=verbose
            )
            
            # Prepare the objective function.
            obfn = get_obfn(name=obfn_name)
            
            # Run the optimization.
            opt_result = minimize_scalar(fun=obfn,
                                         bracket=(bracket_low, bracket_mid, bracket_high),
                                         bounds=(bracket_low, bracket_high),
                                         args=(x_values, aux_paras),
                                         method="bounded",
                                         options={"xatol": tolerance})
            
            # Store the result.
            sol_values[crit][data] += [opt_result.x]
            fun_values[crit][data] += [opt_result.fun]
            
            
        # After this inner loop, store in arrays.
        fun_values[crit][data] = np.copy(np.array(fun_values[crit][data]))
        sol_values[crit][data] = np.copy(np.array(sol_values[crit][data]))
    
    
    # Put together a multi-axis figure.
    fig, axes = plt.subplots(1, 1+len(criteria), figsize=(3*(1+len(criteria)),3),
                             gridspec_kw={"width_ratios": [2]*len(criteria) + [1]},
                             sharey=True)
    
    # Organize the axes.
    axes_risk = axes[0:-1]
    ax_data = axes[-1]
    
    # Plot lines or ticks that appear on all the criteria plots.
    for ax in axes_risk:
        ax.axhline(y=x_q25, color="xkcd:silver")
        ax.axhline(y=x_q75, color="xkcd:silver")
        ax.axhline(y=x_q50, color="xkcd:red")
        ax.axhline(y=x_mean, color="xkcd:blue")
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
    
    # Plot threshold and criteria values for each class.
    for i, crit in enumerate(criteria):
        ax = axes[i]
        ax.plot(todo_riskparas[crit][1], sol_values[crit][data], color="xkcd:black", ls="solid")
        ax.plot(todo_riskparas[crit][1], fun_values[crit][data], color="xkcd:black", ls="dashed")
        ax.fill_between(
            x=todo_riskparas[crit][1], y1=sol_values[crit][data], y2=fun_values[crit][data],
            alpha=transparency, color="xkcd:black", lw=0
        )
        for vline in vlines[crit]:
            ax.axvline(vline, color="xkcd:silver", ls="dotted")
        ax.set_title(titles[crit])
        ax.set_xlabel(xlabels[crit])
        ax.set_xticks(xticks[crit])
        ax.set_xticklabels(xticklabels[crit])
    
    # Plot the data histogram.
    ax_data.hist(x_values,
                 bins="auto", color="black", hatch=None,
                 orientation="horizontal", density=False)
    ax_data.set_xlabel("Frequency")
    ax_data.set_title(data)
    
    # Set the vertical axis limits.
    for ax in axes:
        ax.set_ylim(top=ylim_top, bottom=ylim_bottom)
    
    # Output the figure.
    plt.tight_layout()
    if to_save:
        if flip_data:
            fname = path.join(img_dir, img_name+"_flipped.pdf")
        else:
            fname = path.join(img_dir, img_name+".pdf")
        plt.savefig(fname=fname)
    plt.show()


# Functions used in computing dispersions.

def dispersion_cvar(x, beta):
    '''
    CVaR with threshold at the confidence level "beta".
    This function computes dispersion from OCE form.
    '''
    return np.clip(a=x, a_min=0.0, a_max=None) / (1.0-beta)


def dispersion_entropic(x, gamma):
    '''
    Entropic risk with "shape" parameter.
    This function computes dispersion from OCE form.
    '''
    if gamma == 0.0:
        return x
    else:
        return (np.exp(gamma*x)-1.0)/gamma


def dispersion_dro(x, a, c):
    '''
    DRO risk in convenient dual form.
    '''
    cstar = c/(c-1.0)
    factor = (1.0+c*(c-1.0)*a)**(1.0/c)
    return (factor*np.clip(a=x, a_min=0.0, a_max=None))**cstar


def dispersion_dro_repara(x, atilde):
    '''
    Reparametrized DRO risk under c=2.
    '''
    a = 0.5 * (1.0/(1.0-atilde)-1.0)**2.0
    cstar = 2.0
    factor = np.sqrt(1.0+2.0*a)
    return (factor*np.clip(a=x, a_min=0.0, a_max=None))**cstar


def make_dispersion_comparison(to_save=False, img_name=None, img_path=None):

    # Risk parameters up for investigation here.
    barron_todo = np.linspace(-2.0, 2.0, 25)
    cvar_todo = np.linspace(0.0, 0.98, 25)
    entropic_todo = np.linspace(0.0, 5.0, 25)
    dro_todo = np.linspace(0.0, 0.98, 25)
    
    # Colour setup.
    barron_cmap = cm.get_cmap("inferno")
    barron_colours = []
    for i in range(len(barron_todo)):
        barron_colours += [barron_cmap(i/len(barron_todo))]
    
    cvar_cmap = cm.get_cmap("inferno")
    cvar_colours = []
    for i in range(len(cvar_todo)):
        cvar_colours += [cvar_cmap(i/len(cvar_todo))]
    
    entropic_cmap = cm.get_cmap("inferno")
    entropic_colours = []
    for i in range(len(entropic_todo)):
        entropic_colours += [entropic_cmap(i/len(entropic_todo))]
    
    dro_cmap = cm.get_cmap("inferno")
    dro_colours = []
    for i in range(len(dro_todo)):
        dro_colours += [dro_cmap(i/len(dro_todo))]

    x_values = np.linspace(-1.0, 1.0, 250)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12,3), sharey=True)
    
    ax1.axhline(y=0.0, color="xkcd:silver")
    ax1.axvline(x=0.0, color="xkcd:silver")
    ax2.axhline(y=0.0, color="xkcd:silver")
    ax2.axvline(x=0.0, color="xkcd:silver")
    ax3.axhline(y=0.0, color="xkcd:silver")
    ax3.axvline(x=0.0, color="xkcd:silver")
    ax4.axhline(y=0.0, color="xkcd:silver")
    ax4.axvline(x=0.0, color="xkcd:silver")
    
    for i, alpha in enumerate(barron_todo):
        sigma = 0.2
        y_values = dispersion_barron(x=x_values/sigma, alpha=alpha)
        ax1.plot(x_values, y_values, color=barron_colours[i], ls="solid")
        ax1.set_title(r"Barron ($\rho_{\alpha}$)")
    
    for i, beta in enumerate(cvar_todo):
        y_values = dispersion_cvar(x=x_values, beta=beta)
        ax2.plot(x_values, y_values, color=cvar_colours[i], ls="solid")
        ax2.set_title(r"CVaR ($\phi$)")
    
    for i, gamma in enumerate(entropic_todo):
        y_values = dispersion_entropic(x=x_values, gamma=gamma)
        ax3.plot(x_values, y_values, color=entropic_colours[i], ls="solid")
        ax3.set_title(r"Tilted ($\phi$)")
        
    for i, atilde in enumerate(dro_todo):
        y_values = dispersion_dro_repara(x=x_values, atilde=atilde)
        ax4.plot(x_values, y_values, color=dro_colours[i], ls="solid")
        ax4.set_title(r"$\chi^{2}$-DRO ($\phi$)")
    
    ax1.set_ylim(top=3.25, bottom=-1.25)
    plt.tight_layout()
    if to_save:
        fname = path.join(img_dir, img_name+".pdf")
        plt.savefig(fname=fname)
    plt.show()


def make_dispersion_comparison_colorbars(to_save=False, img_name=None, img_path=None):

    # Risk parameters up for investigation here.
    barron_todo = np.linspace(-2.0, 2.0, 25)
    cvar_todo = np.linspace(0.0, 0.98, 25)
    entropic_todo = np.linspace(0.0, 5.0, 25)
    dro_todo = np.linspace(0.0, 0.98, 25)

    # Colour setup.
    barron_cmap = cm.get_cmap("inferno")
    barron_colours = []
    for i in range(len(barron_todo)):
        barron_colours += [barron_cmap(i/len(barron_todo))]
    
    cvar_cmap = cm.get_cmap("inferno")
    cvar_colours = []
    for i in range(len(cvar_todo)):
        cvar_colours += [cvar_cmap(i/len(cvar_todo))]
    
    entropic_cmap = cm.get_cmap("inferno")
    entropic_colours = []
    for i in range(len(entropic_todo)):
        entropic_colours += [entropic_cmap(i/len(entropic_todo))]
    
    dro_cmap = cm.get_cmap("inferno")
    dro_colours = []
    for i in range(len(dro_todo)):
        dro_colours += [dro_cmap(i/len(dro_todo))]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12,0.5))
    fig.subplots_adjust(bottom=0.5)

    # Barron
    cmap = barron_cmap
    bounds = [alpha for alpha in barron_todo]
    norm = BoundaryNorm(bounds, cmap.N, extend="both")
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=ax1, orientation="horizontal")
    cbar.set_ticks([np.min(barron_todo), 0.0, np.max(barron_todo)])
    cbar.set_ticklabels([str(-2.0), str(0), str(2.0)])
    ax1.set_title(r"$\alpha$")
    
    # CVaR
    cmap = cvar_cmap
    bounds = [beta for beta in cvar_todo]
    norm = BoundaryNorm(bounds, cmap.N, extend="both")
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=ax2, orientation="horizontal")
    cbar.set_ticks([np.min(cvar_todo), 0.5, np.max(cvar_todo)])
    cbar.set_ticklabels([str(0.0), str(0.5), str(0.98)])
    ax2.set_title(r"$\beta$")
    
    # Entropic
    cmap = entropic_cmap
    bounds = [gamma for gamma in entropic_todo]
    norm = BoundaryNorm(bounds, cmap.N, extend="both")
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=ax3, orientation="horizontal")
    cbar.set_ticks([np.min(entropic_todo), np.max(entropic_todo)])
    cbar.set_ticklabels([str(0.0), str(5.0)])
    ax3.set_title(r"$\gamma$")
    
    # DRO
    cmap = dro_cmap
    bounds = [atilde for atilde in dro_todo]
    norm = BoundaryNorm(bounds, cmap.N, extend="both")
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=ax4, orientation="horizontal")
    cbar.set_ticks([np.min(dro_todo), 0.5, np.max(dro_todo)])
    cbar.set_ticklabels([str(0.0), str(0.5), str(0.98)])
    ax4.set_title(r"$\widetilde{a}$")
    
    if to_save:
        fname = path.join(img_dir, img_name+".pdf")
        plt.savefig(fname=fname)
    plt.show()


def loss_squared(h, h_star):
    return (h-h_star)**2

def loss_absolute(h, h_star):
    return np.absolute(h-h_star)

def loss_hinge(h, h_star):
    return np.maximum(1-h*h_star,0)

def loss_crossentropy(h, h_star):
    return np.log1p(np.exp(-h_star*h))


def make_loss_tranforms(loss_names, to_save=False, img_name=None, img_path=None):

    # Clerical setup.
    barron_todo = np.linspace(-2.0, 2.0, 25)
    cvar_todo = np.linspace(0.0, 0.98, 25)
    entropic_todo = np.linspace(0.0, 5.0, 25)
    dro_todo = np.linspace(0.0, 0.98, 25)

    # Colour setup.
    barron_cmap = cm.get_cmap("inferno")
    barron_colours = []
    for i in range(len(barron_todo)):
        barron_colours += [barron_cmap(i/len(barron_todo))]
    
    cvar_cmap = cm.get_cmap("inferno")
    cvar_colours = []
    for i in range(len(cvar_todo)):
        cvar_colours += [cvar_cmap(i/len(cvar_todo))]
    
    entropic_cmap = cm.get_cmap("inferno")
    entropic_colours = []
    for i in range(len(entropic_todo)):
        entropic_colours += [entropic_cmap(i/len(entropic_todo))]
    
    dro_cmap = cm.get_cmap("inferno")
    dro_colours = []
    for i in range(len(dro_todo)):
        dro_colours += [dro_cmap(i/len(dro_todo))]

    todo_loss_fns = {"squared": loss_squared,
                     "absolute": loss_absolute,
                     "hinge": loss_hinge,
                     "cross-entropy": loss_crossentropy}
    todo_domain_widths = {"squared": 8.0, "absolute": 15.0, "hinge": 15.0, "cross-entropy": 15.0}
    theta_value = 3.0
    h_star = np.pi

    for loss_fn_name in loss_names:
    
        loss_fn = todo_loss_fns[loss_fn_name]
        domain_width = todo_domain_widths[loss_fn_name]
        h_values = np.linspace(h_star-domain_width/2, h_star+domain_width/2, 100)
        x_values = loss_fn(h=h_values, h_star=h_star)-theta_value
    
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12,4), sharey=True)
    
        ax1.axhline(y=0.0, color="xkcd:silver")
        ax1.axvline(x=h_star, color="xkcd:silver")
        ax2.axhline(y=0.0, color="xkcd:silver")
        ax2.axvline(x=h_star, color="xkcd:silver")
        ax3.axhline(y=0.0, color="xkcd:silver")
        ax3.axvline(x=h_star, color="xkcd:silver")
        ax4.axhline(y=0.0, color="xkcd:silver")
        ax4.axvline(x=h_star, color="xkcd:silver")
    
        for i, alpha in enumerate(barron_todo):
            y_values = dispersion_barron(x=x_values, alpha=alpha)
            ax1.plot(h_values, y_values, color=barron_colours[i], ls="solid")
            ax1.set_title("T-risk")
    
        for i, beta in enumerate(cvar_todo):
            y_values = dispersion_cvar(x=x_values, beta=beta)
            ax2.plot(h_values, y_values, color=cvar_colours[i], ls="solid")
            ax2.set_title("CVaR")
    
        for i, gamma in enumerate(entropic_todo):
            y_values = dispersion_entropic(x=x_values, gamma=gamma)
            ax3.plot(h_values, y_values, color=entropic_colours[i], ls="solid")
            ax3.set_title("Tilted")
    
        for i, atilde in enumerate(dro_todo):
            y_values = dispersion_dro_repara(x=x_values, atilde=atilde)
            ax4.plot(h_values, y_values, color=dro_colours[i], ls="solid")
            ax4.set_title(r"$\chi^{2}$-DRO")
        
        ax1.set_ylim(top=4.25, bottom=-1.25)
        xticks = [h_star]
        xtick_labels = [r"$h^{\ast}$"]
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xtick_labels)
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xtick_labels)
        ax3.set_xticks(xticks)
        ax3.set_xticklabels(xtick_labels)
        ax4.set_xticks(xticks)
        ax4.set_xticklabels(xtick_labels)
        fig.suptitle("Transformed losses ({})".format(loss_fn_name))
        plt.tight_layout()
        if to_save:
            fname = path.join(img_dir, img_name+".pdf")
            plt.savefig(fname=fname)
        plt.show()


weibull_c = 1.5
weibull_mean = weibull_min.stats(c=weibull_c, moments="m")

def weibull(x, mean=weibull_mean, c=weibull_c):
    return weibull_min.pdf(x=x, c=c, loc=-mean, scale=1)

def weibull_flipped(x, mean=weibull_mean, c=weibull_c):
    return weibull(x=-x, mean=mean, c=c)

def beta_wide(x):
    return beta.pdf(x=x, a=1.7, b=1.7, loc=0, scale=1)

def beta_narrow(x):
    return beta.pdf(x=x, a=170, b=170, loc=1, scale=1)


def triangle(x, width, p, left):
    idx_left = x <= left+p*width
    idx_right = x > left+p*width
    density = np.copy(x)
    density[idx_left] = 2*(density[idx_left]-left)/(p*width**2)
    density[idx_right] = 2*(left+width-density[idx_right])/((1-p)*width**2)
    return density


def make_weibull(to_save=False, img_name=None, img_path=None):
    fig, ax = plt.subplots(1, figsize=(4, 2.5))
    x_right = np.linspace(-weibull_mean, 4.0, 100)
    x_left = np.linspace(-4.0, weibull_mean, 100)
    shift = 0.75
    x_shifted = np.linspace(-weibull_mean-shift, 4.0, 100)
    ax.axvline(x=0.0, color="xkcd:black", lw=1)
    ax.axhline(y=0.0, color="xkcd:black", lw=1)
    ax.plot(x_right, weibull(x=x_right), color="xkcd:blue")
    ax.plot(x_shifted, weibull(x=x_shifted+shift), color="xkcd:blue", ls="dashed", label=r"$\mathsf{L}_{1}$")
    ax.plot(x_left, weibull_flipped(x=x_left), color="xkcd:red", label=r"$\mathsf{L}_{2}$")
    plt.legend(loc="best")
    plt.tight_layout()
    if to_save:
        fname = path.join(img_dir, img_name+".pdf")
        plt.savefig(fname=fname)
    plt.show()


def make_beta(to_save=False, img_name=None, img_path=None):
    fig, ax = plt.subplots(1, figsize=(4, 2.5))
    x_right = np.linspace(1.0, 2.0, 100)
    x_left = np.linspace(0.0, 1.0, 100)
    ax.axhline(y=0.0, color="xkcd:black", lw=1)
    ax.plot(x_left, beta_wide(x=x_left), color="xkcd:red", label=r"$\mathsf{L}_{1}$")
    ax.plot(x_right, beta_narrow(x=x_right), color="xkcd:blue", ls="dashed", label=r"$\mathsf{L}_{2}$")
    plt.legend(loc="best")
    plt.tight_layout()
    if to_save:
        fname = path.join(img_dir, img_name+".pdf")
        plt.savefig(fname=fname)
    plt.show()


def make_triangles(to_save=False, img_name=None, img_path=None):
    fig, ax = plt.subplots(1, figsize=(4, 2.5))
    x_right = np.linspace(2.25, 2.75, 100)
    x_left = np.linspace(0, 2.5, 100)
    ax.axhline(y=0.0, color="xkcd:black", lw=1)
    ax.plot(x_left, triangle(x=x_left, width=2.5, p=0.5, left=0.0),
            color="xkcd:red", ls="solid", label=r"$\mathsf{L}_{1}$")
    ax.plot(x_right, triangle(x=x_right, width=0.5, p=0.5, left=2.25),
            color="xkcd:blue", ls="dashed", label=r"$\mathsf{L}_{2}$")
    plt.legend(loc="best")
    plt.tight_layout()
    if to_save:
        fname = path.join(img_dir, img_name+".pdf")
        plt.savefig(fname=fname)
    plt.show()


def make_nocollapse(to_save=False, img_name=None, img_path=None):
    a = 2.0
    X = np.array(([1.,-1.], [-1.,1.], [a,-a]))
    Y = np.array(([0], [1], [1])).flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,2.5))
    
    horiz_points = np.linspace(-2.5, 2.5, 100)
    line_separator = horiz_points
    ax1.plot(horiz_points, line_separator,
            color="xkcd:silver",
            linestyle="dashed")
    idx_0 = Y == 0
    idx_1 = Y == 1
    ax1.scatter(X[idx_0,0], X[idx_0,1], marker="o", color="xkcd:gold")
    ax1.scatter(X[idx_1,0], X[idx_1,1], marker="x", color="xkcd:dark teal")
    ax1.axhline(y=0.0, color="xkcd:silver")
    ax1.axvline(x=0.0, color="xkcd:silver")
    ax1.set_xticks([-1, 1, a])
    ax1.set_xticklabels(["-1", "1", r"$a$"])
    ax1.set_yticks([-a, -1, 1])
    ax1.set_yticklabels([r"-$\mathit{a}$", "-1", "1"])
    xlim_tuple = (-2.5, 2.5)
    ylim_tuple = (-2.5, 2.5)
    ax1.set(xlim=xlim_tuple, ylim=ylim_tuple)
    p = 0.8
    bin_width = 0.25
    bin_starts = np.array([0.25/2, 0.5+0.25/2])
    bar_positions = bin_starts + bin_width/2
    probabilities = np.array([p, 1-p])
    bars = ax2.bar(bar_positions, probabilities, bin_width,
                   color="xkcd:black",
                   edgecolor="xkcd:white")
    ax2.set_xticks([0.25, 0.75])
    ax2.set_xticklabels([r"$\left\{ \Vert{\mathrm{X}}\Vert_{2} \leq \sqrt{2} \right\}$",
                         r"$\left\{ \Vert{\mathrm{X}}\Vert_{2} > \sqrt{2} \right\}$"])
    ax2.set_yticks([0.0, p, 1.0])
    ax2.set_yticklabels(["0", r"$p$", "1"])
    ax2.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    plt.tight_layout()
    if to_save:
        fname = path.join(img_dir, img_name+".pdf")
        plt.savefig(fname=fname)
    plt.show()


def makedir_safe(dirname: str) -> None:
    '''
    A simple utility for making new directories
    after checking that they do not exist.
    '''
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return None


def _dist_param_spec(data):
    '''
    Map distribution name -> (generator kwarg, slider spec)
    '''
    if data == "bernoulli":
        return "prob", dict(cls="FloatSlider", value=0.30, min=0.25, max=0.75, step=0.05, description="mean")
    if data == "beta":
        return "b", dict(cls="FloatSlider", value=0.5, min=0.1, max=3.5, step=0.1, description="shape")
    if data == "pareto":
        return "a", dict(cls="FloatSlider", value=3.5, min=3.25, max=10.0, step=0.25, description="shape")
    if data == "normal":
        return "scale", dict(cls="FloatSlider", value=1.0, min=0.1, max=5.0, step=0.1, description="scale")
    raise ValueError("Unsupported data distribution!")

def _build_slider(spec):
    if ipw is None:
        raise RuntimeError("ipywidgets is not available. Please `pip install ipywidgets`.")
    spec = spec.copy()
    cls = spec.pop("cls")
    return getattr(ipw, cls)(**spec)

def interactive_criterion_plot(data, rg, criteria):
    '''
    Shows a single slider for the chosen data distribution and re-calls
    make_criterion_plot(...) with `data_kwargs={param: slider.value}` on change.
    '''
    if ipw is None:
        raise RuntimeError("ipywidgets is not available. Please `pip install ipywidgets`.")

    param_name, spec = _dist_param_spec(data)
    slider = _build_slider(spec)

    # Dedicated output area just for the figure.
    out = ipw.Output()

    # Layout and drawing function.
    ui = ipw.VBox([slider, out])
    display(ui)
    def _draw(_=None):
        with out:
            out.clear_output(wait=True)
            fig = make_criterion_plot(
                data=data,
                rg=rg,
                criteria=criteria,
                data_kwargs={param_name: slider.value}
            )
            # Ensure something is displayed. If make_criterion_plot draws inline,
            # calling plt.show() is still safe; if it returns a fig, display it.
            if fig is not None:
                display(fig)
            else:
                plt.show()
    slider.observe(_draw, names="value")
    _draw() # initial render


def interactive_trisk_plot(data, rg):
    '''
    Shows a single slider for the chosen data distribution and re-calls
    make_criterion_plot(...) with `data_kwargs={param: slider.value}` on change.
    '''
    if ipw is None:
        raise RuntimeError("ipywidgets is not available. Please `pip install ipywidgets`.")

    param_name, spec = _dist_param_spec(data)
    slider = _build_slider(spec)

    # Dedicated output area just for the figure.
    out = ipw.Output()

    # Layout and drawing function.
    ui = ipw.VBox([slider, out])
    display(ui)
    def _draw(_=None):
        with out:
            out.clear_output(wait=True)
            fig = make_trisk_plot(
                data=data,
                rg=rg,
                data_kwargs={param_name: slider.value}
            )
            # Ensure something is displayed. If make_criterion_plot draws inline,
            # calling plt.show() is still safe; if it returns a fig, display it.
            if fig is not None:
                display(fig)
            else:
                plt.show()
    slider.observe(_draw, names="value")
    _draw() # initial render


###############################################################################
