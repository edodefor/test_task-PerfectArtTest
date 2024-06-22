import numpy as np
import scipy as sp   
from sklearn.metrics import r2_score
        
"""
This module provides functions nessesary to process returns series and to perform stochastic simulations

"""

def get_distribution(sample, bins = None, normalized = True):
    """
    Custom method to calculate frequensy statistical distribution of valies in a given sample.
    
    Parameters:
    -----------
    
    :param sample: 1D array of floats or ints, sample to get distribution from
    
    :param bins: numnber of bins to split data. Default is Freedman–Diaconis bins
    
    :param normalized: bool, if True, normalize distribution. Default is True
    
    Returns:
    --------
    
    :return: frequensies, bin_centers, bin_edges 
    """  
    
    sample = np.sort(sample)
    _min, _max = np.min(sample), np.max(sample)
    
    if abs(_min) == np.inf or _max == np.inf:
        raise ValueError("Minimal or maximal value of the sample is infinite")
    
    #Freedman–Diaconis rule    
    if bins == None:
        iqr = np.percentile(sample, 75) - np.percentile(sample,25 )
        N = len(sample)
        h = 2 * iqr * N ** (-1/3)
        bins = (_max - _min) // h
        
    else:
        h = (_max - _min) / bins
    
    #Here I decided to implement my owm pdf instead of using scipy.stats or numpy.histogram
    step = _min
    frequensies, bin_centers, bin_edges = [], [], [_min] 
            
    while step < _max:
        frequensies.append(len(np.where((sample >= step) & (sample < step + h))[0]))
        bin_centers.append(step + h/2)
        bin_edges.append(step + h)
        #Next iteration
        step += h
        
    frequensies, bin_centers, bin_edges = np.array(frequensies), np.array(bin_centers), np.array(bin_edges)
    
    #
    if normalized is True:
        frequensies = frequensies/len(sample)
        
    return frequensies, bin_centers, bin_edges
            

def get_moments(sample, k=1):
    """
    Calculate k-th central moments of a sample
    
    Parameters:
    -----------
    
    :param sample: 1D- or nD-array of floats, sample or array of samples to calculate moments
    
    :param k: int or 1D array of ints, order of moment to calculate
    
    Returns:
    --------
    :return: float or 1D-array of floats, moments, calculated from samples
    """
    
    #check if sampl is 1D or n-D
    try:
        iter(sample[0])
        
        _samples_moments = []
        
        #check if k is iterable 
        try:
            iter(k)
            for _sample in sample:
                
                _moments = []
                for _k in k:
                    #first moment -- expectation
                    if _k == 1:
                        _moments.append(np.mean(_sample))
                    
                    #other moments
                    else:
                        _moments.append(np.mean((np.array(_sample) - np.mean(_sample)) ** _k))
                
                _samples_moments.append(_moments)
            
            return np.array(_samples_moments)
       
        #case of int k
        except TypeError:
            for _sample in sample:
                if k == 1:
                    _samples_moments.append(np.mean(_sample))
                
                else:
                    _samples_moments.append(np.mean((np.array(_sample) - np.mean(_sample)) ** k))
                    
            return np.array(_samples_moments)
     
    #case of 1D sample   
    except TypeError:
        try:
            iter(k)
                            
            _moments = []
            for _k in k:
                if _k == 1:
                    _moments.append(np.mean(sample))
                else:
                    _moments.append(np.mean((np.array(sample) - np.mean(sample)) ** _k))
        
            return np.array(_moments)
        
        except TypeError:
            if k == 1:
                return np.mean(sample)
            
            else:
                return np.mean((np.array(sample) - np.mean(sample) ** k))
            
            
            
                    

def clt_r2_test(samples, bins = None):
    """
    Calculate r-squared value, coefficient of determination between sample means distribution and normal distribution
    to perform Central Limit Theorm test on closeness of distribution of shifted means to distribution to N(0,1)
    
    Parameters:
    -----------
    
    :param sample: N-D array, array of samples generated from some distribution
    
    :param bins: int, number of bins to generate frequency distribution of 
    
    Returns:
    --------
    
    :return: population mean, population variance, r-squared coefficient, array of sample means 
    """
     
    N = len(np.array(samples).T)     #Number of elements in one sample
    
    # calculate set of sample means, total population mean and variance
    sample_means = get_moments(samples)
    total_mean = np.mean(np.concatenate(samples, axis=0))
    total_var = np.sqrt(np.var(np.concatenate(samples, axis=0)))
    
    #obtaining sequence of means S_n = (mean(X_n) - E(X))/(sigma/sqrt(n))
    S = np.sqrt(N)*(sample_means - total_mean)/total_var
    
    #calculate frequency (or prodability in the n->inf limit) distribution of S_n
    #and realization of N(0,1)
    sample_pdf, values, edges = get_distribution(S, bins=bins)
    true_normal = sp.stats.norm.cdf(values, loc=0, scale=1)
    
    #calculate coefficient coefficient of determination.
    #here shifted sample means S_n plays role of true values and values obtained from N(0,1) of the model prediction
    #therefor r2 shows goodness of fit
    r2_val = r2_score(np.cumsum(sample_pdf), true_normal)
    
    return total_mean, total_var, r2_val, sample_means     
      
         
         
            
        
        
        
        
    
    
    
    