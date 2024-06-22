from perfect_art import series
from perfect_art import statistics
from perfect_art import graphics
import numpy as np

from argparse import ArgumentParser


def simulation( sample_size,   #size of sample of n-days return samples, do not confuse with number of observations
                q,  #quantile to calculaye percentile distribution    
                length,   #number of observations, length of one-day returns series
                n_days,   #number of days for n-days return values
                alpha,   #alpha parameter of stable distribution
                beta,   #skewness parameter for stable distribution
                gamma,   #shift  parameter for stable distribution function
                delta,   #scale parameter for stable distribution function
                r2_cutoff,   #cutoff r2 coefficient value at which the simulation terminates
                max_iter,   #maximal number of iterations at which the simulation termibates
                min_added_size   #size of random population added to the total population during simulation
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
    
    :param sample size: int, size of sample of n-days return samples, do not confuse with number of observations
    
    :param q: float, quantile to calculaye percentile distribution 
    
    :param length: int, number of observations, length of one-day returns series
    
    :param n_days: number of days for n-days return values
    
    :param alpha: float, alpha parameter of stable distribution
    
    :param beta: floar, skewness parameter for stable distribution
    
    :param gamma: float, shift  parameter for stable distribution function
    
    :param delta: float, scale parameter for stable distribution function
    
    :param r2_cutoff: float, cutoff r2 coefficient value at which the simulation terminates
    
    :param max_iter: int, maximal number of iterations at which the simulation termibates
    
    :param min_added_size: int, size of random population added to the total population during simulation
    
    Returns:
    --------
    
    :returns: total percentiles population, sample means values, N (size of a sample), array of iteration stepps, 
    array of current maximal r2 coefficients at each iteration, array of total population mens at each iteration, 
    array of total variances at each iteration     
    """
    #create sample of n-days series sample object
    levy_dist_params = sample_size, q, length, n_days, alpha, beta, gamma, delta
    sample = series.ReturnsSample(*levy_dist_params)

    #define initial values of tenp variables
    r2, _r2 = -np.inf, -np.inf
    total_mean, total_var = np.nan, np.nan
    step = 1
    
    #define array for total population and temporary array
    percentiles_samples = np.array([[] for i in range(sample_size)])
    _percentiles_samples = percentiles_samples
    
    total_mean_array, total_var_array = [], []
    r2_values = []
    steps = []

    while r2 < r2_cutoff and step <= max_iter:
        sample.samplify() #generate random population (initial at 1 iter, added sample otherwise)

        #add added sample to a total population
        percentiles_added = np.array([sample.percentiles_array]).T
        _percentiles_samples = np.concatenate((_percentiles_samples, percentiles_added), axis=1)    
        
        #just to make initial population and added population larger
        if step % min_added_size == 0:
            total_mean, total_var, _r2, sample_means = statistics.clt_r2_test(_percentiles_samples) 
        
            #check if new population means samples better fit normal distribution
            if _r2 > r2:
                r2 = _r2
                percentiles_samples = np.concatenate((percentiles_samples, _percentiles_samples), axis=1)
                
            total_mean_array.append(total_mean)
            total_var_array.append(total_var)
            r2_values.append(r2)
            steps.append(step)
            
            #update temporal array
            _percentiles_samples = percentiles_samples
                
            
        step +=1
        
        print(f"step={step-1}\t current r2={r2}\t added r2={_r2}\t mean={total_mean}\t variance={total_var}", end='\r')
    
    
    return (np.concatenate(percentiles_samples, axis=0), sample_means, len(percentiles_samples[0]),
            np.array(steps), np.array(r2_values), np.array(total_mean_array), np.array(total_var_array))
    
    
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
    parser.add_argument('-s', '--sample_size', default=100, required=False, type=int,
                        help='length of sample of n-days return series')
    parser.add_argument('-r', '--r2_cutoff', default=.995, required=False, type=float,
                        help='value of R-squared test at which simulation is terminated')
    parser.add_argument('-m', '--max_iter', default=np.inf, required=False, type=float,
                        help='maximal number of iterations during simulation. Default if np.inf')
    parser.add_argument('-ms', '--min_added_size', default=1, required=False, type=int,
                        help='minimal population size of each sample element' 
                        + 'added to the total population during simulation')
    
    #post-simulation params
    parser.add_argument('-o', '--order', default=2, required=False, type=int,
                        help='maximal order of moments of the simulated distribution to calculate')
    parser.add_argument('-af', '--all_figures', default=False, required=False, type=bool,
                        help='If true, convergense plots are generated')
    
    
    args = parser.parse_args()
    
    #simulate
    simulation_params = (args.sample_size, args.q, args.length, args.n_days, 
                        args.alpha, args.beta, args.gamma, args.delta, 
                        args.r2_cutoff, args.max_iter, args.min_added_size)

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
        
        graphics.save_plot(steps, np.log(r2_t), label_conv, x_label_conv, y_label_conv, title_conv)
        
        print("Convergence plot was saved to ./figures")
