import numpy as np
import scipy as sp


class ReturnSeries():
    r""" 1D time series of one- and n-day returns. 
    
    Notes
    -----
    The instance of the Class realizes a series of one day returns :math:'r_{i}^{1}' defined as:
    
    .. math:: 
       
       r_i^1 = \frac{P_{i+1} - P_{i}}{P_{i}} 
       
    where :math: 'P_{i}' is a price at :math:'i^{th}' day. Each element of the series is an independent realisation of
    the random variable with Levi's stable distribution.
        
    """

    one_day_returns_values = np.array([])    #field for one day returns series
    n_days_returns_values = np.array([])     #field for N days returns series
    
    #Class constructor
    def __init__(self, 
                 length = 750, 
                 n_days = 10,
                 alpha = 1.7,
                 beta = 0,
                 gamma = 1,
                 delta = 1):
        
        self.length = length
        self.n_days = n_days
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.distribution_params = self.alpha, self.beta, self.gamma, self.delta       

    
    def generate(self, length = None, n_days = None):
        """
        Generate 1D timeseries 
        
        Parameters
        ----------
        
        :param length: int,
            number of elements of the timeseries (default is 750)
        :param distribution: str,
            Levy's stable distribution realization, can be either 'numpy_levy' or 'castom_levy'
        
        Return:
        -------
        
        :return: numpy.array
            1D array of floats, one day returns 
        """
        
        #handeling method parameters values
        if length == None:
            length = self.length
            
        if n_days == None:
            n_days = self.n_days
        
        self.one_day_returns_values = sp.stats.levy_stable.rvs(*self.distribution_params, length)
        
        
        #make a copy of the one-day returns series to use in in n-days calculatios
        one_day_rs = self.one_day_returns_values.copy()
        one_day_rs += 1  
                
        #Calculate series of N-products of shifted one-day returns and shift the result by -1 to gen N-days returns
        #PEP8 would suggest to make length-n_days its own var but my variant seems more readable
        iteration_number = length - (n_days - 1)
        self.n_days_returns_values = np.array([np.prod(one_day_rs[i : i+n_days-1]) 
                                               for i in range(iteration_number)])
        self.n_days_returns_values -= 1
        
        #delete copy of the series to avoid running out of RAM when creating large samples of long series
        del one_day_rs, iteration_number
    
    

class ReturnsSample(ReturnSeries):
    r"""
    M-size sample of one-day and N-days returns series.
    Successor class of the 'perfect_art.series.ReturnSeries'.  
    
    Fields:
    -------
    
    :param one_days_sample_array: numpy.array, array allocated for a sample of one-day returns series;
            
    :param n_days_sample_array: numpy.array, array allocated for a sample of n-days returns series;
            
    Methods:
    --------
    
    :param samplify: 
        Creates samples of one-day and n-days returns series.
    """
    
    
    one_day_sample_array = np.array([])    #array of one-day returns series
    n_days_sample_array = np.array([])     #array of n-days returns series 
    percentiles_array = np.array([])     
        
    
    #Class constructor
    def __init__(self,
                 sample_size = 10,
                 q = 0.01, 
                 length = 750, 
                 n_days = 10,
                 alpha = 1.7,
                 beta = 0,
                 gamma = 1,
                 delta = 1):
        
        self.sample_size = sample_size
        self.q = q
        
        #invoking parent class constructor 
        ReturnSeries.__init__(self, 
                              length = length, 
                              n_days = n_days,
                              alpha = alpha,
                              beta = beta,
                              gamma = gamma,
                              delta = delta )
                            
    
    def samplify(self, sample_size=None, q=None):
        """
        Generate random samples of 'sample_size' sequencies of one-day and n-day returns.
        
        Parameters:
        -----------
        
        :param sample_size: int, number of elements of each sample.
                            
        Returns:
        --------
        
        :return: None. The result is written in self.one_day_sample_array and self.n_days_sample_array
        """               
        
        #handeling parameters values
        if sample_size == None:
            sample_size = self.sample_size
            
        if q == None:
            q = self.q
        
        #introduce temporary lists for sample arrays because I hate numpy.append
        _one_day_sample_array, _n_days_sample_array, _percentiles_array = [], [], []    
        while sample_size > 0:
            self.generate()     #genetate random series using ReturnSeries.generate()
            
            _one_day_sample_array.append(self.one_day_returns_values)
            _n_days_sample_array.append(self.n_days_returns_values)
            _percentiles_array.append(np.percentile(self.n_days_returns_values, q * 100))
            sample_size -=1
        
        #turning the result into numpy array
        self.one_day_sample_array = np.array(_one_day_sample_array)
        self.n_days_sample_array = np.array(_n_days_sample_array)
        self.percentiles_array  = np.array(_percentiles_array)
        
        #get rid of the temp arrays. After exiting the function local variables will be destroed anyway though... 
        del _one_day_sample_array, _n_days_sample_array, _percentiles_array 
        
        
        
    
        
    

            

            
    