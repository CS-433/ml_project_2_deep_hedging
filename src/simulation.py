import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm




# ## Simulation


def CallBS(t, T, K, S, r, q, sigma): #[DONE]
    '''
    Calculates the price and the delta of a call option using the Black-Scholes formula.
    Inputs:
        t = current time
        T = expiry time
        K = strike price
        S = current stock price
        r = risk-free rate
        q = dividend yield
        sigma = volatility
    Outputs:
        P = option price
        delta = option delta
    '''
    d1 = (np.log(S/K) + (r -q + 0.5*sigma**2)*(T-t)) / (sigma* np.sqrt(T-t))
    d2 = d1 - (sigma* np.sqrt(T-t))
    
    # price
    P = S * np.exp(-q*(T-t))* norm.cdf(d1) - np.exp(-r*(T-t)) * K * norm.cdf(d2) 
    
    # delta
    delta = np.exp(-q*(T-t))*norm.cdf(d1)

    return P, delta


def GBM_sim(n, T, dt, S0, mu, r, q, sigma, days, freq): #[DONE]
    '''
    Simulates the price of the underlying with a geometric brownian motion model (Black-Scholes model), from time 0 to final time T.
    Inputs:
        n =     [float] number of paths
        T =     [float] time to expiry
        dt =    [float] time step, function of frequency (e.g. freq*0.01)
        S0 =    [float] current (starting) price
        mu =    [float] return on asset
        r =     [float] risk-free rate
        q =     [float] dividend yield
        sigma = [float] volatility
        days =  [int] number of days in year
        freq =  [float] trading frequency (e.g. 2 = every two days, 0.5 = every day twice)
    Output:
        S =     [array] simulated underlying price process
    '''
    T = int(T/freq) # adjust T to frequency
    
    # initialise variables
    S = np.zeros((n, T))  # Underlying price path

    S[:, 0] = S0
    # generate price path based on random component and derive option price and delta   
    for t in tqdm(range(1, T)):  # generate paths
        dW = np.random.normal(0, 1, size=(n)) # standard normal random variable
        S[:, t] = S[:, t-1] * np.exp((mu - 0.5*sigma**2)*dt/days + sigma*np.sqrt(dt/days)*dW) # BS Model of Stock Price 

    
    return S


def SABR_sim(n, days, freq, T, dt, S0, sigma0, v, rho, mu): #[DONE]
    '''
    Simulates the price of the underlying with a special case of the SABR model (beta = 1), from time 0 to final time T.
    Calculates the option price and delta at all time steps using the Black-Scholes option pricing formula.
    Inputs:
        n =           [float] number of simulations
        days =        [int] number of days in a year
        freq =        [float] trading frequency
        T =           [float] end time (expiry)
        dt =          [float] time step
        S0 =          [float] current (starting) price
        sigma0 =     [float] initial volatility
        v =           [float] volatility of underlying volatility
        rho =         [float] correlation of the two Brownian Motions
    Output:
        S =           [array] simulated price process
        sigma_stoch = [array] simulated stochastic volatility process
    '''

    T = int(T/freq) # adjust T to frequency

    # initialise variables
    sigma_stoch = np.zeros((n, T)) # Underlying stochastic volatility path
    S = np.zeros((n, T))  # Underlying price path

    sigma_stoch[:, 0] = sigma0
    S[:, 0] = S0
 
    # generate parameters for creating correlated random numbers
    mean = np.array([0,0])
    Corr = np.array([[1, rho], [rho, 1]]) # Correlation matrix
    STD = np.diag([1,1]) # standard deviation vector
    Cov = STD@Corr@STD # covariance matrix, input of multivariate_normal function


    # generate price path based on random component and derive option price and delta
    for t in tqdm(range(1,T)):  
        dW = np.random.multivariate_normal(mean, Cov, size = n)  # correlated random BM increments
        sigma_stoch[:, t] = sigma_stoch[:, t-1]*np.exp((-0.5*v**2)*dt/days + v*np.sqrt(dt/days)*dW[:, 0]) # GBM model of volatility
        S[:, t] = S[:, t-1]*np.exp((mu - 0.5*sigma_stoch[:, t]**2)*dt/days + sigma_stoch[:, t]*np.sqrt(dt/days)*dW[:, 1]) # Black-Scholdes GBM model of underlying price


    return S, sigma_stoch


def SABR_IV(sigma_stoch, t, T, S, K, r, q, v, rho):
    # future price
    f = S * np.exp((r - q) * (T-t))
    # at the money case
    atm = sigma_stoch * (1+(T-t)*(rho * v * sigma_stoch/4  + v**2 * (2-3 * rho**2)/24))
    xi = (v/ sigma_stoch) * np.log(f / K)
    xi_func = np.log((np.sqrt(1 - 2 * rho * xi + xi**2) + xi - rho) / (1 - rho))

    imp_vol = np.where(f == K, atm, atm * xi / xi_func)

    return imp_vol


def bartlett_delta(T, t, S, K, ivol, ds, rho, v):
    # Find Bartlett's delta using numerical differentiation
    d_volatility = ds * v * rho/S # following Bartlett (2006) Eq. 12 and using 

    i_sigma = SABR_IV(ivol, t, T, S, K, r, q, v, rho)
    i_sigma_plus = SABR_IV(ivol + d_volatility, t, T, S + ds, K, r, q, v, rho)

    p_base, _ = CallBS(t, T, K, S, r, q, i_sigma)
    p_plus, _ = CallBS(t, T, K, S + ds, r, q, i_sigma_plus)

    # finite differences
    bartlett_delta = (p_plus-p_base) / ds

    return bartlett_delta

def simulateGBM(n, T, dt, S0, mu, r, q, sigma, days, freq):
    S_gbm = GBM_sim(n, T, dt, S0, mu, r, q, sigma, days, freq)
    times = np.arange(0,T,freq)
    p_gbm, d_gbm = CallBS(times/days, T/days, K, S_gbm, r, q, sigma)

    return S_gbm, p_gbm, d_gbm

def simulateSABR (n, T, dt, S0, mu, r, q, sigma, days, freq, rho, ds, v):
    S_sabr, s_sabr = SABR_sim(n, days, freq, T, dt, S0, sigma, v, rho, mu)
    times = np.arange(0,T,freq)
    iv_SABR = SABR_IV(s_sabr, times/days, T/days, S_sabr, K, r, q, v, rho)
    p_sabr, delta_sabr= CallBS(times/days, T/days, K, S_sabr, r, q, s_sabr)
    bl_delta_sabr = bartlett_delta(T/days, times/days, S_sabr, K, iv_SABR, ds, rho, v)

    return S_sabr, s_sabr, iv_SABR, p_sabr, delta_sabr, bl_delta_sabr



def OU(X0, beta, alpha, sigmaOU, n, T, freq, days, dt):
    '''Generates an Ornstein-Uhlenbeck simulation
    Inputs:
        dt = freq*0.01 for example or just freq*1
    '''

    T = int(T/freq)
    X = np.zeros((n,T))
    X[:,0] = X0
    
    for t in range(1,T):
        dW = np.random.normal(0, 1, size=(n))
        X[:,t] = (1-beta)*X[:,t-1] + alpha*beta + sigmaOU*np.sqrt(dt/days)*dW
    
    return X

# ## Classical Delta and Bartlett Hedging for Short European Call Option (Benchmark)

def hedgingStrategy(method,notional, n, T, dt, S0, mu, r, q, sigma, days, freq, rho, ds, v):
    '''
    Implements delta hedging for GBM model and delta hedging and bartlett hedging for SABR model.
    '''
    if method == "GBM":
        S_gbm, p_gbm, d_gbm = simulateGBM(n, T, dt, S0, mu, r, q, sigma, days, freq)
        trading_gbm = np.diff(d_gbm, axis = 1)
        trading_gbm = np.concatenate((d_gbm[:,0].reshape(-1,1), trading_gbm), axis=1)
        trading_gbm *= notional
        holding_gbm = d_gbm*notional

        return S_gbm, p_gbm, d_gbm, trading_gbm, holding_gbm

    if method == "SABR":
        # sabr delta hedging
        S_sabr, s_sabr, iv_SABR, p_sabr, delta_sabr, bl_delta_sabr = simulateSABR(n, T, dt, S0, mu, r, q, sigma, days, freq, rho, ds, v)
        trading_sabr = np.diff(delta_sabr, axis = 1)
        trading_sabr = np.concatenate((delta_sabr[:,0].reshape(-1,1), trading_sabr), axis=1)*notional
        trading_sabr *= notional
        holding_sabr = delta_sabr*notional

        # sabr bartlett delta hedging
        trading_sabr_bartlett = np.diff(bl_delta_sabr, axis = 1)
        trading_sabr_bartlett = np.concatenate((bl_delta_sabr[:,0].reshape(-1,1), trading_sabr_bartlett), axis=1)
        trading_sabr_bartlett *= notional
        holding_sabr_bartlett = bl_delta_sabr*notional

        return S_sabr, p_sabr, delta_sabr, bl_delta_sabr, trading_sabr, holding_sabr, trading_sabr_bartlett, holding_sabr_bartlett

# ## Evaluation


def APL_process(S, p, holding):
    '''
    Calculates the Accounting PnL process for a portfolio of an option, the underlying, with proportional trading costs
    Inputs:
        S =              [array] underlying price process
        p =              [array] option price process (adjusted for number of underlying)
        holding =        [array] process of number of the underlying held at each period
    Output:
        APL =            [array] process of Accounting PnL
        holding_lagged = [array] lagged process of number of underlying held at each period
    '''
    kappa = 0.01
    # create lagged variables for APL
    p_lagged = np.roll(p, 1)
    p_lagged[:, 0] = np.nan # the first element was p[-1], this has to be changed to NaN
    S_lagged = np.roll(S, 1)
    S_lagged[:, 0] = np.nan # the first element was S[-1], this has to be changed to NaN
    holding_lagged = np.roll(holding, 1)
    holding_lagged[:, 0] = np.nan # the first element was holding[-1], this has to be changed to NaN

    # accounting PnL
    APL = p - p_lagged + holding_lagged*(S-S_lagged) - kappa* np.abs(S*(holding - holding_lagged)) 

    return APL, holding_lagged


def hedgingCost(kappa, S, holding, holding_lagged):
    '''
    Calculates the total hedging cost from time t onward for all t
    Inputs:
        kappa =           [float] proportional hedging cost parameter
        S =               [array] underlying price process
        holding =         [array] process of amount of the underlying asset held at any given time t
        holding_lagged =  [array] process of amount of the underlying asset held at any given time t-1
    Output:
        C =               [array] total hedging cost from time t onward
    '''
    # Hedging cost at each period
    C = kappa* np.abs(S*(holding - holding_lagged))


    return C


def objective(C, c):
    '''
    Calculates the loss from time t (present) to time T (expiry).
    Input:
        C = [array] total hedging cost
        c = [float] weight of standard deviation
    Output:
        Y = [array] loss function over time
    '''
    Y = np.zeros(C.shape)

    for t in range(C.shape[1]):
        Y[:,t] = \
            np.nanmean(C[:, :(t+1)], axis = 1) + \
            c*np.nanstd(C[:, :(t+1)], axis = 1)

    return Y


