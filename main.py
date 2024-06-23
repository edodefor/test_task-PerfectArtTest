from perfect_art import series
from perfect_art import statistics
from perfect_art import graphics
import numpy as np

from argparse import ArgumentParser
   


def simulation( init_population,   #size of initial population of percentiles from n-days return series
                added_population,   #size of added population of percentiles from n-days return series
                samples_number,   #number of samples taken from population
                init_guess_q, #minimal r2 value for initial gues
                q,  #quantile to calculaye percentile distribution    
                length,   #number of observations, length of one-day returns series
                n_days,   #number of days for n-days return values
                alpha,   #alpha parameter of stable distribution
                beta,   #skewness parameter for stable distribution
                gamma,   #shift  parameter for stable distribution function
                delta,   #scale parameter for stable distribution function
                r2_cutoff,   #cutoff r2 coefficient value at which the simulation terminates
                max_iter,   #maximal number of iterations at which the simulation termibates
                inflator   #coefficient of inflation of added population at each iteration
                ):   
    """
    This function is the principal solution to the test task.
    It performs stochastic simulation to obtain distribution of 1%-percentiles of some sample 10-days return series.
    
    Using ReturnSeries ans ReturnSample classes from series module it generates sample of 750 one-day returns, use it 
    to calculate 741 overlaping 10-daus returns, then gets 0.01 quantile = 1% percentile of each element of 10-days returns 
    series sample. This result is used to generate initial population and additional population at each iteration called here 
    added population or added sample.
    
    Then it calculates sample means S_n distribution shifted on mean of the total population and   scaled with the variance of the total 
    population. The cumulative distribution of such S_n is compared to cdf values of the N(0,1) geterated at the same points.
    R-squared metric is calculated for S_n and N(0,1) cdfs, where S_n plays the role of the true values and sample from N(0,1) is
    some modeled values. 
    
    This set of operations is repeated iteratively. At first iteration initial population is generated. 
    During the following iterations added populations are added to the total populations. If the added populations increases 
    r2 coefficient, it becomes part of the total population. In contrast, if this added population 
    decreases r2 it will not be included in the total population.
    
    The underlying idea that the distribution of the total population (and its moments, e.g. mean) gets closer to the true 
    distribution. At the same time means of samples of N its elements approach the total population mean and the distribution
    of sample means approach normal distribution due to central limit theorem.
    The closer we to the true distribution and the large N gets, the closer empirical distribution of the means gets to the 
    mormal distribution with expextation equal to population mean and sigma squared equal to population variance, thus guaranteeing
    convergence,   
    
    Parameters:
    -----------
    
    :param init_population: int, size of initial population of percentiles from n-days return series
    
    :param added_population: int, size of added population of percentiles from n-days return series
    
    :param samples_number: int, number of samples taken from population
    
    :param init_guess-q: float, minimal r2 value for initial population
    
    :param q: float, quantile to calculaye percentile distribution 
    
    :param length: int, number of observations, length of one-day returns series
    
    :param n_days: number of days for n-days return values
    
    :param alpha: float, alpha parameter of stable distribution
    
    :param beta: floar, skewness parameter for stable distribution
    
    :param gamma: float, shift  parameter for stable distribution function
    
    :param delta: float, scale parameter for stable distribution function
    
    :param r2_cutoff: float, cutoff r2 coefficient value at which the simulation terminates
    
    :param max_iter: int, maximal number of iterations at which the simulation termibates
    
    :param inflator: float, coefficient of inflation of added population at each iteration
    
    Returns:
    --------
    
    :returns: total percentiles population, sample means values, N (size of a sample), array of iteration stepps, 
    array of current maximal r2 coefficients at each iteration, array of total population mens at each iteration, 
    array of total variances at each iteration     
    """
    #create sample of n-days series sample object
    sample = series.ReturnsSample(q=q,
                                  length=length,
                                  alpha=alpha,
                                  beta=beta,
                                  gamma=gamma,
                                  delta=delta)

    #define initial values of tenp variables
    r2, _r2 = -np.inf, -np.inf
    total_mean, total_var = np.nan, np.nan
    step = 1
    
    population_mean_array, population_var_array = [], []
    r2_values = []
    steps = []
    
    percentiles_samples = np.array([])
    

    while r2 < r2_cutoff and step <= max_iter:
        #creating an initial population
        #initial population is an initial gues andconvergence depends on it
        #therefore choose initial population with good enough quality
        if percentiles_samples.size == 0: 
            print(f"Generating initial population. Trying to guess initial sample with R2 >= {init_guess_q}. \n")
            #making guesses until we reach good quality
            populations_dict = {} 
            idx=0
            while _r2 < init_guess_q and idx < 50:
                sample.series_sample(sample_size=init_population)
                percentiles_samples = np.array(sample.percentiles_array)
                
                N = len(percentiles_samples) // samples_number
        
                N_samples = statistics.samplify(percentiles_samples, N, samples_number)
                population_moments = statistics.get_moments(percentiles_samples, k=[1,2])
                population_mean, population_var = population_moments[0], population_moments[1]
        
                _r2, sample_means = statistics.clt_r2_test(N_samples, population_mean, population_var, N)
                
                populations_dict[_r2] = percentiles_samples
                idx += 1
            
            #for the case if we could not guess appropriate population
            if _r2 < init_guess_q:
                max_r2 = np.max(populations_dict.keys)
                percentiles_samples = population_dict[max_r2]
                print("Failed to guess initial population with required minimal quality in 50 iterations. \n")
                print(f"initial population qith qiality R2={max_r2} was taken. \n")
                
            #creating temporal array     
            _percentiles_samples = percentiles_samples
            
            #destroy unnecessary variables  
            del populations_dict
                
        sample.series_sample(sample_size=added_population) #generate random added population 

        #add added sample to a total population
        percentiles_added = sample.percentiles_array
        _percentiles_samples = np.concatenate((_percentiles_samples, percentiles_added), axis=0)    
        
        N = len(_percentiles_samples) // samples_number
        
        N_samples = statistics.samplify(_percentiles_samples, N, samples_number)
        population_moments = statistics.get_moments(_percentiles_samples, k=[1,2])
        population_mean, population_var = population_moments[0], population_moments[1]
        
        _r2, sample_means = statistics.clt_r2_test(N_samples, population_mean, population_var, N) 
        
        #check if new population means samples better fit normal distribution
        if _r2 > r2:
            r2 = _r2
            percentiles_samples = np.concatenate((percentiles_samples, _percentiles_samples), axis=0)
            added_population *= inflator
                
        population_mean_array.append(population_mean)
        population_var_array.append(population_var)
        r2_values.append(r2)
        steps.append(step)
            
        #update temporal array
        _percentiles_samples = percentiles_samples
                
            
        step +=1
        
        print(f"step={step-1}\t current r2={r2}\t added r2={_r2}\t mean={population_mean}\t variance={np.sqrt(population_var)}", end='\r')
    
    
    return (percentiles_samples, sample_means, N, 
            np.array(steps), np.array(r2_values), 
            np.array(population_mean_array), np.array(population_var_array))
    
    
if __name__ == '__main__':
        
    print("enter")
    parser = ArgumentParser()
    
    parser.add_argument('-n', '--n_days', default=10, required=False, type=int,
                        help='number days to calculate n-days return values')
    parser.add_argument('-l', '--length', default=750, required=False, type=int,
                        help='length of one-dar returns series')
    parser.add_argument('-a', '--alpha', default=1.7, required=False, type=float,
                        help='alpha parameter for stable distribution function')
    parser.add_argument('-b', '--beta', default=0, required=False, type=float,
                        help='skewness parameter for stable distribution function')
    parser.add_argument('-g', '--gamma', default=1., required=False, type=float,
                        help='shift  parameter for stable distribution function')
    parser.add_argument('-d', '--delta', default=1., required=False, type=float,
                        help='scale parameter for stable distribution function')
    parser.add_argument('-q', '--q', default=0.01, required=False, type=float,
                        help='quantile value at which quantile (or percentile) distribution is simulated')
    parser.add_argument('-ip', '--init_population', default=1000, required=False, type=int,
                        help='size of initial population of percentiles from n-days return series')
    parser.add_argument('-s', '--samples_number', default=100, required=False, type=int,
                        help='number of samples taken from population')
    parser.add_argument('-ap', '--added_population', default=250, required=False, type=int,
                        help='size of added population of percentiles from n-days return series')
    parser.add_argument('-r', '--r2_cutoff', default=0.991, required=False, type=float,
                        help='value of R-squared test at which simulation is terminated')
    parser.add_argument('-m', '--max_iter', default=np.inf, required=False, type=float,
                        help='maximal number of iterations during simulation. Default if np.inf')
    parser.add_argument('-if', '--inflator', default=1.5, required=False, type=float,
                        help='coefficient of inflation of added population at each iteration')
    parser.add_argument('-iq', '--init_guess_q', default=0.8, required=False, type=float,
                        help='initial guess quality, minimal r2 value for initial population')
    
    #post-simulation params
    parser.add_argument('-o', '--order', default=2, required=False, type=int,
                        help='maximal order of moments of the simulated distribution to calculate')
    parser.add_argument('-af', '--all_figures', default=False, required=False, type=bool,
                        help='If true, convergense plots are generated')
    
    
    args = parser.parse_args()
    
    #simulate
    simulation_params = (args.init_population, args.added_population, args.samples_number, args.init_guess_q,
                         args.q, args.length, args.n_days, args.alpha, args.beta, args.gamma, args.delta, 
                        args.r2_cutoff, args.max_iter, args.inflator)

    percentiles, means, N, steps, r2_t, mean_t, var_t = simulation(*simulation_params)
    
    #generate frequency and cumulative distributions of the simulated percentile population
    pdf, bins, edges = statistics.get_distribution(percentiles)
    cdf = np.cumsum(pdf)
    
    #generating figures
    label_pdf = f"{args.q}-quantile frequencies"
    title_pdf = f"{args.n_days}-returns {args.q}-quantile frequency distribution"
    x_label_pdf = f"{args.q}-quantile values"
    y_label_pdf = "Frequencies"
    
    graphics.save_hist(bins, pdf, label_pdf,x_label_pdf, y_label_pdf, title_pdf)
    
    label_cdf = f"{args.q}-quantile probabilities"
    title_cdf = f"{args.n_days}-returns {args.q}-quantile cumulative distribution"
    x_label_cdf = f"{args.q}-quantile values"
    y_label_cdf = "Cumulative probalility"
    
    graphics.save_hist(bins, cdf, label_cdf,x_label_cdf, y_label_cdf, title_cdf)
    
    moments = statistics.get_moments(percentiles,k=range(1, args.order+1))
    print("Sucsess")
    print(f"Simulation was terminated at r-squared value {r2_t[-1]} and iteration number {steps[-1]}")
    
    print(f"{args.order} central moments of the simulated distribution:\n")
    for i,moment in enumerate(moments):
        print(f"{i+1}-th moment={moment}")
        
    print("Histograms of frequency and cumulative distributions was saved to ./figures")
    
    if args.all_figures == True:
        label_conv = r"$R^2$"
        title_conv = r"$R^2$ vs steps"
        x_label_conv = f"steps"
        y_label_conv = r"$R^2$"
        
        graphics.save_plot(steps, 1-r2_t, label_conv, x_label_conv, y_label_conv, title_conv)
        
        print("Convergence plot was saved to ./figures")
