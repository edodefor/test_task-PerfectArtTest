import matplotlib.pyplot as plt
import numpy as np
        
                   
def save_hist(x, y, legend, x_label, y_label, title, figsize=(10,10), dpi=500):
    """
    Generates and saves histogram plots to a .png file
    
    Parameters:
    -----------
    
    :param x: 1D- or N-D array, argument values
    
    :param y: 1D or N-D array, function values
    
    :param legend: str or array of strings, labels for datasets
    
    :param x_label: str, x-axis label
    
    :param y_label: str, y-axis label
    
    :param title: str, title of the figure
    
    :param figsize: tuple, figure size parameters
    
    :param dpi: int, pixel resolution
    
    
    Returns:
    --------
    
    :return: None, save vigure to ./figures foler
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.axes()
    
    try:
        iter(x[0])
        iter(y[0])
        iter(legend)
        
        if len(x) != len(y) or len(x) != len(legend):
            raise ValueError("x and y and corresponding legend arrays should have equal length")
        
        alpha = 1/len(x)
        for (_x,_y,l) in zip(x,y,legend):
            width = abs(np.max(_x) - np.min(_x)) / 100
            ax.bar(_x, _y, width=width, alpha=alpha, label=l) 
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        
        plt.savefig("figures/" + title + ".png")
        
    except:
        width = abs(np.max(x) - np.min(x)) / 100
        ax.bar(x, y, width=width, label=legend) 
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        
        plt.savefig("figures/" + title + ".png")
        
        
def save_plot(x, y, labels, x_label, y_label, title, figsize=(10,10), dpi=500):
    """
    Generates and saves line plots to a .png file
    
    Parameters:
    -----------
    
    :param x: 1D- or N-D array, argument values
    
    :param y: 1D or N-D array, function values
    
    :param labels: str or array of strings, labels for datasets
    
    :param x_label: str, x-axis label
    
    :param y_label: str, y-axis label
    
    :param title: str, title of the figure
    
    :param figsize: tuple, figure size parameters
    
    :param dpi: int, pixel resolution
    
    
    Returns:
    --------
    
    :return: None, save vigure to ./figures foler
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.axes()
    
    try:
        iter(x[0])
        iter(y[0])
        iter(labels)
        
        if len(x) != len(y) or len(x) != len(labels):
            raise ValueError("x and y and corresponding legend arrays should have equal length")
        
        for (_x,_y,l) in zip(x,y,labels):
            ax.plot(_x, _y, linewidth=2, label=l) 
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        
        plt.savefig("figures/" + title + ".png")
        
    except:
        ax.plot(x, y, linewidth=2, label=labels) 
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        
        plt.savefig("figures/" + title + ".png")
    