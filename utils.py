from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

class Utils:

    def defineColorPalette():
        """
        Use the seaborn library to define a better looking color palette.
    
        Currently takes no inputs, but that might change as I make more
        palettes
    
        Returns:
        - palette: the seaborn palette defined
        """
        # try using the xkcd_rgb colors
        # want 6 colors
        colors = ['blue', 'cherry red', 'golden yellow', 'vibrant green', 'violet', 'pumpkin orange']
        palette = sns.xkcd_palette(colors)
    
        return palette 

    def plotCorrelationMatrices3GivenCMapRange(fn1, fn2, fn3, cmapMin, cmapMax,
                          title1="Cross Correlation Matrix 1",
                          title2="Cross Correlation Matrix 2",
                          title3="Cross Correlation Matrix 3",
                          outFn=""):
        """    
        Takes 3 filenames, colormap min and max values, 3 graph titles, and an output filename
        
        Read in the correlation ratio matrices from the files
        Plot all 3 correlation ratio matrices in 1 figure following the layout
    
           title1       title2       title3
        +---------+  +---------+  +---------+  +-+
        |         |  |         |  |         |  | |
        |         |  |         |  |         |  | |
        |    1    |  |    2    |  |    3    |  | |
        |         |  |         |  |         |  | |
        |         |  |         |  |         |  | |
        +---------+  +---------+  +---------+  +-+
    
        where the bar on the side is the colorbar
    
        If an output function is specified, save the graph to a file.
    
        Inputs:
        - fn1, fn2, fn3: the filenames of the correlation matrices
        - cmapMin: min value for the colormap
        - cmapMax: max value for the colormap
        - title1, title2, title3 (optional): title to label each correlation ratio matrix
            *** Highly suggested that the user specifies the title as the defaults are nondescriptive
        - outFn (optional): output filename; if specified, save the figure to this file
    
        Returns:
        - nothing
    
        Effects:
        - displays the correlation matrices in the notebook
        - if outFn is specified, save the figure to a file
        """
        # load the matrices
        mat1 = np.loadtxt(open(fn1, 'r'), delimiter=',')
        mat2 = np.loadtxt(open(fn2, 'r'), delimiter=',')
        mat3 = np.loadtxt(open(fn3, 'r'), delimiter=',')
          
        # make a new figure
        fig = plt.figure(figsize=(14,5))
   
        # subplot 1: correlation matrix 1
        ax1 = fig.add_subplot(131)
        cax1 = ax1.matshow(mat1, cmap=plt.cm.gist_heat_r, vmin=cmapMin, vmax=cmapMax) # plasma_r looks cool, gist_heat_r is good
        ax1.xaxis.set_ticks_position('bottom')
        ax1.set_title(title1)
    
        # subplot 2: correlation matrix 2
        ax2 = fig.add_subplot(132)
        cax2 = ax2.matshow(mat2, cmap=plt.cm.gist_heat_r, vmin=cmapMin, vmax=cmapMax) # plasma_r looks cool, gist_heat_r is good
        ax2.xaxis.set_ticks_position('bottom')
        ax2.set_title(title2)
    
        # subplot 3: correlation matrix 3
        ax3 = fig.add_subplot(133)
        cax3 = ax3.matshow(mat3, cmap=plt.cm.gist_heat_r, vmin=cmapMin, vmax=cmapMax) # plasma_r looks cool, gist_heat_r is good
        ax3.xaxis.set_ticks_position('bottom')
        ax3.set_title(title3)
        
        # add the colorbar
        # fig.subplots_adjust(wspace=0, hspace=0.05) # needed to adjust the whitespace at one point, for future reference
        fig.colorbar(cax1, ax=[ax1, ax2, ax3])
    
        # show the figure in the notebook
        plt.show()
    
        # if the outFn argument is specified, save the figure to a file
        if not outFn == "":
            fig.savefig(outFn, bbox_inches='tight', dpi=600)
    
