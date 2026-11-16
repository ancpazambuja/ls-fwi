
"""@package local_data_viewer.py
Visualization of models and data.
"""

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors0

import random

from src.utl.classes import Utils as utl
from src.utl.classes import Graph as graph

from src.utl.logging import Log as log
from src.data.data import Models2D, DeepTomographyModels, SupervisedData, Data


class DataAndModelsViewer:
    """!
    Contains specific visualizations for Data and Models.
    """
    DEFAULT_VELOCITY_CMAP = "turbo"
    DEFAULT_SEISMIC_CMAP = "RdBu_r"
    DEFAULT_DIFF_CMAP = "RdBu_r"

    DEFAULT_LATENT_MEAN_CMAP = "viridis"
    DEFAULT_LATENT_STD_CMAP = "magma"
    DEFAULT_LATENT_FULL_CMAP = "PuOr" #"lisbon", "RdBu_r", "vlag"

    DEFAULT_X_CMAP = "turbo" 
    DEFAULT_QUALITY_CMAP = "seismic" # red - white - blue
    DEFAULT_ERROR_CMAP = "seismic"   # red - white - blue


    @staticmethod
    def zy_view( z=None, z_full=None, y=None, nsamples=1, samples=None, colorbar=False, 
                z_mean_cmap=DEFAULT_LATENT_MEAN_CMAP, global_palette=False,
                z_std_cmap=DEFAULT_LATENT_STD_CMAP, z_full_cmap=DEFAULT_LATENT_FULL_CMAP,
                z_min=None, z_max=None,
                y_min=None, y_max=None, y_cmap=DEFAULT_VELOCITY_CMAP, title=""
        ):
    
        if samples is not None:
            nsamples = len(samples)

        assert z.shape[0] == y.shape[0], f"Shape mismatch between z and y: {z.shape[0]} != {y.shape[0]}"
        assert z.shape[0] == z_full.shape[0], f"Shape mismatch between z and z_full: {z.shape[0]} != {z_full.shape[0]}"
    
        NELEMS_TOTAL = z.shape[0]
    
        COLORBAR = colorbar
    
        NLINES = nsamples
        NCOLS = 4

        if global_palette:
            z_gmin = z_min.min()
            z_gmax = z_max.max()
            y_gmin = y_min.min()
            y_gmax = y_max.max()
            z_min[:] = z_gmin
            z_max[:] = z_gmax
            y_min[:] = y_gmin
            y_max[:] = y_gmax  

        factor_x = 4
        factor_y = factor_x * nsamples / 4
    
        log.info("| Z mean | Z std | Z full | Y |")
    
        fig = plt.figure( figsize=( graph.PLOT_SX * factor_x, graph.PLOT_SY * factor_y), constrained_layout=True)
    
        gs = fig.add_gridspec( NLINES, NCOLS)
    
        for i in range(NLINES):
    
            if samples is None: 
                idx = i
            else:
                idx = samples[i]
    
            title_prefix = " [ " + str(idx) + " ] "
    
            #
            # Expected latent shape:
            # (C,H,W)
            #
            z_sample = z[idx]
    
            if len(z_sample.shape) == 3:
    
                # Average latent image
                z_mean = z_sample.mean(axis=0)
    
                # Std latent image
                z_std = z_sample.std(axis=0)
    
            else:
    
                z_mean = z_sample
                z_std = np.zeros_like(z_sample)
    
            z_full_sample = z_full[idx]

            # Velocity model
            y_sample = y[idx]
    
            # Handle (1,H,W)
            if len(y_sample.shape) == 3:
                y_sample = y_sample[0]
    
            graph.drawImageAttr_grid( fig.add_subplot(gs[i, 0]), z_mean, vmin=z_min[idx], vmax=z_max[idx], cmap=z_mean_cmap, colorbar=COLORBAR, title=title_prefix + "z mean")
    
            graph.drawImageAttr_grid( fig.add_subplot(gs[i, 1]), z_std, vmin=z_min[idx], vmax=z_max[idx], cmap=z_std_cmap, colorbar=COLORBAR, title="z std")


            ax = fig.add_subplot(gs[i, 2])
            graph.drawImageAttr_grid( ax, z_full_sample, vmin=z_min[idx], vmax=z_max[idx], cmap=z_full_cmap, colorbar=COLORBAR, title=f"z full)", aspect=factor_x/factor_y)
            for ih in range(0, 21*32 + 1, 32):
                ax.axhline( ih - 0.5, color='white', linewidth=0.5)
            for iw in range(0, 25*16 + 1, 16):
                ax.axvline( iw - 0.5, color='white', linewidth=0.5) 


            graph.drawImageAttr_grid( fig.add_subplot(gs[i, 3]), y_sample, vmin=y_min[idx], vmax=y_max[idx], cmap=y_cmap, colorbar=COLORBAR, title="y generated")
    
        plt.show()
    
        return fig

    def x_view(data=None, nsamples=1, samples=None, colorbar=False, x_cmap=DEFAULT_X_CMAP, title=""):
        if samples is not None:
            nsamples = len(samples)
        
        NELEMS_TOTAL = data.shape[0]
        COLORBAR = colorbar
        NLINES = 1
        NCOLS = 1
        NFILTERS = data.shape[1]
        
        factor_x = 2
        subfactor = np.ceil(NFILTERS/NCOLS)
        factor_y = factor_x*nsamples*(subfactor/4)

        #factor_x = 10
        #factor_y = 3
        
        log.info(f"| X | {NLINES} x {NCOLS} common offset gathers!")
        
        fig = plt.figure(figsize=(graph.PLOT_SX*factor_x,graph.PLOT_SY*factor_y), constrained_layout=True)
        gs = fig.add_gridspec(NLINES*nsamples, NCOLS)
        
        for i in range(nsamples):
            if samples is None:
                i_rnd = random.randint(0, NELEMS_TOTAL-1)
            else:
                i_rnd = samples[i]
            for filt in range(NFILTERS):
                iline = (i*NLINES)+(filt//NCOLS)
                icol = filt%NCOLS
                if iline%NLINES == 0 and icol == 0:
                    title_  = " [ " + str(i_rnd) + " ] " + title
                else:
                    title_ = ""
                graph.drawImageAttr_grid(fig.add_subplot(gs[iline,icol]), data[i_rnd,filt,:,:], cmap=x_cmap, colorbar=COLORBAR, title=title_)
        #plt.show()
        return fig   

    def yy_view(y=None, y_hat=None, loss=None, nsamples=1, samples=None, vmin=None, vmax=None, colorbar=False, y_cmap=DEFAULT_VELOCITY_CMAP, title=""):
        if samples is not None:
            nsamples = len(samples)

        if loss is not None:
            assert loss.shape[0] == y.shape[0], f"[ ERROR ] Loss vector must have the same size as y and y_hat."
            
        NELEMS_TOTAL = y.shape[0]
        COLORBAR = colorbar
        NLINES = nsamples
        NCOLS = 3
        
        factor_x = 3
        factor_y = factor_x*nsamples/4
        
        log.info(f"| Y | Y_hat - loss | DIFF MAP |")
        
        
        fig = plt.figure(figsize=(graph.PLOT_SX*factor_x,graph.PLOT_SY*factor_y), constrained_layout=True)
        gs = fig.add_gridspec(NLINES, NCOLS)
        
        for i in range(NLINES):
            if samples is None:
                i_rnd = random.randint(0, NELEMS_TOTAL-1)
            else:
                i_rnd = samples[i]
                
            title_prefix  = " [ " + str(i_rnd) + " ] "
            y_hat_title = "y_hat"
            diff_title = "Diff (y-y_hat)"
            
            diff_map = y[i_rnd,:,:] - y_hat[i_rnd,:,:]
            diff_range = max(diff_map.max(), abs(diff_map.min()))
           
            if loss is not None:
                y_hat_title = y_hat_title + " -- " + str(loss[i_rnd])

            graph.drawImageAttr_grid(fig.add_subplot(gs[i,0]), y[i_rnd,:,:], vmin=vmin, vmax=vmax, cmap=y_cmap, colorbar=COLORBAR, title=title_prefix + title + " - y")
            graph.drawImageAttr_grid(fig.add_subplot(gs[i,1]), y_hat[i_rnd,:,:], vmin=vmin, vmax=vmax, cmap=y_cmap, colorbar=COLORBAR, title=y_hat_title)
            graph.drawImageAttr_grid(fig.add_subplot(gs[i,2]), diff_map, vmin=-diff_range, vmax=diff_range, cmap=DataAndModelsViewer.DEFAULT_ERROR_CMAP, colorbar=COLORBAR, title=diff_title)
        plt.show()
        return fig
    
    def yyq_view(y=None, y_hat=None, qmap=None, loss=None, nsamples=1, samples=None, vmin=None, vmax=None, colorbar=False, y_cmap=DEFAULT_VELOCITY_CMAP, title=""):
        if samples is not None:
            nsamples = len(samples)

        if loss is not None:
            assert loss.shape[0] == y.shape[0], f"[ ERROR ] Loss vector must have the same size as y and y_hat."
            
        NELEMS_TOTAL = y.shape[0]
        COLORBAR = colorbar
        NLINES = nsamples
        NCOLS = 4
        
        factor_x = 4
        factor_y = factor_x*nsamples/4
        
        log.info(f"| Y | Y_hat - loss | DIFF MAP | QUALITY MAP |")
        
        
        fig = plt.figure(figsize=(graph.PLOT_SX*factor_x,graph.PLOT_SY*factor_y), constrained_layout=True)
        gs = fig.add_gridspec(NLINES, NCOLS)
        
        for i in range(NLINES):
            if samples is None:
                i_rnd = random.randint(0, NELEMS_TOTAL-1)
            else:
                i_rnd = samples[i]
                
            title_prefix  = " [ " + str(i_rnd) + " ] "
            y_hat_title = "y_hat"
            diff_title = "Diff (y-y_hat)"
            qmap_title = "quality map"

            diff_map = y[i_rnd,:,:] - y_hat[i_rnd,:,:]
            diff_range = max(diff_map.max(), abs(diff_map.min()))

            if loss is not None:
                y_hat_title = y_hat_title + " -- " + str(loss[i_rnd])

            graph.drawImageAttr_grid(fig.add_subplot(gs[i,0]), y[i_rnd,:,:], vmin=vmin, vmax=vmax, cmap=y_cmap, colorbar=COLORBAR, title=title_prefix + title + " - y")
            graph.drawImageAttr_grid(fig.add_subplot(gs[i,1]), y_hat[i_rnd,:,:], vmin=vmin, vmax=vmax, cmap=y_cmap, colorbar=COLORBAR, title=y_hat_title)
            graph.drawImageAttr_grid(fig.add_subplot(gs[i,2]), diff_map, vmin=-diff_range, vmax=diff_range, cmap=DataAndModelsViewer.DEFAULT_ERROR_CMAP, colorbar=COLORBAR, title=diff_title)
            graph.drawImageAttr_grid(fig.add_subplot(gs[i,3]), qmap[i_rnd,:,:], vmin=-1, vmax=1, cmap=DataAndModelsViewer.DEFAULT_QUALITY_CMAP, colorbar=COLORBAR, title=qmap_title)
        plt.show()
        return fig

    def xy_view(data:SupervisedData=None, nsamples=1, samples=None, colorbar=False, x_cmap=DEFAULT_X_CMAP, y_cmap=DEFAULT_VELOCITY_CMAP, title=""):
        if samples is not None:
            nsamples = len(samples)
            
        NELEMS_TOTAL = data.get_nmb_of_elems()
        COLORBAR = colorbar
        NLINES = nsamples
        NCOLS = 3
        NFILTERS = data.get_X().shape[1]
        
        factor_x = 2
        factor_y = factor_x*nsamples/4
        
        log.info(f"| X offset 0 | X offset 1 | Y |")
        
        fig = plt.figure(figsize=(graph.PLOT_SX*factor_x,graph.PLOT_SY*factor_y), constrained_layout=True)
        gs = fig.add_gridspec(NLINES, NCOLS)
        
        for i in range(NLINES):
            if samples is None:
                i_rnd = random.randint(0, NELEMS_TOTAL-1)
            else:
                i_rnd = samples[i]
                
            title_prefix  = " [ " + str(i_rnd) + " ] "
            graph.drawImageAttr_grid(fig.add_subplot(gs[i,0]), data.get_X()[i_rnd,0,:,:], cmap=x_cmap, colorbar=COLORBAR, title=title_prefix + title)
            graph.drawImageAttr_grid(fig.add_subplot(gs[i,1]), data.get_X()[i_rnd,NFILTERS-1,:,:], cmap=x_cmap, colorbar=COLORBAR, title="")
            graph.drawImageAttr_grid(fig.add_subplot(gs[i,2]), data.get_Y()[i_rnd,:,:], cmap=y_cmap, colorbar=COLORBAR, title="")
        plt.show()
        return fig

    def flow_view(data:Data=None, models:DeepTomographyModels=None, nsamples=1, samples=None, i_offset=0, colorbar=False, x_cmap=DEFAULT_X_CMAP, y_cmap=DEFAULT_VELOCITY_CMAP, models_cmap=DEFAULT_VELOCITY_CMAP):
        if samples is not None:
            nsamples = len(samples)
            
        NELEMS_TOTAL = data.get_nmb_of_elems()
        COLORBAR = colorbar
        NLINES = nsamples
        NCOLS = 7
        
        factor_x = 4
        factor_y = factor_x*nsamples/4
        
        fig = plt.figure(figsize=(graph.PLOT_SX*factor_x,graph.PLOT_SY*factor_y), constrained_layout=True)
        gs = fig.add_gridspec(NLINES*2, NCOLS)
        
        for i in range(NLINES):
            
            if samples is None:
                i_rnd = random.randint(0, NELEMS_TOTAL-1)
            else:
                i_rnd = samples[i]
            
            graph.drawImageAttr_grid(fig.add_subplot(gs[i*2,0]), data.get_lowres_data().get_X()[i_rnd,i_offset,:,:], cmap=x_cmap, colorbar=COLORBAR, title=str(i_rnd) + ' - LOWRES X')
            graph.drawImageAttr_grid(fig.add_subplot(gs[i*2,1]), data.get_lowres_data().get_Y()[i_rnd,:,:], cmap=y_cmap, colorbar=COLORBAR, title='LOWMRES Y', vmin=-1.7, vmax=1.7)
            graph.drawImageAttr_grid(fig.add_subplot(gs[i*2+1,0]), models.get_lowres_models().get_models()[i_rnd,:,:], cmap=models_cmap, colorbar=COLORBAR, title='LOWRES MODEL')
            
            graph.drawImageAttr_grid(fig.add_subplot(gs[i*2,2]), data.get_mediumres_data().get_X()[i_rnd,i_offset,:,:], cmap=x_cmap, colorbar=COLORBAR, title='MEDIUMRES X')
            graph.drawImageAttr_grid(fig.add_subplot(gs[i*2,3]), data.get_mediumres_data().get_Y()[i_rnd,:,:], cmap=y_cmap, colorbar=COLORBAR, title='MEDIUMRES Y', vmin=-1.7, vmax=1.7)
            graph.drawImageAttr_grid(fig.add_subplot(gs[i*2+1,2]), models.get_mediumres_models().get_models()[i_rnd,:,:], cmap=models_cmap, colorbar=COLORBAR, title='MEDIUMRES MODEL')
            
            graph.drawImageAttr_grid(fig.add_subplot(gs[i*2,4]), data.get_highres_data().get_X()[i_rnd,i_offset,:,:], cmap=x_cmap, colorbar=COLORBAR, title='HIGHRES X')
            graph.drawImageAttr_grid(fig.add_subplot(gs[i*2,5]), data.get_highres_data().get_Y()[i_rnd,:,:], cmap=y_cmap, colorbar=COLORBAR, title='HIGHRES Y', vmin=-1.7, vmax=1.7)
            graph.drawImageAttr_grid(fig.add_subplot(gs[i*2+1,4]), models.get_highres_models().get_models()[i_rnd,:,:], cmap=models_cmap, colorbar=COLORBAR, title='HIGRES MODEL')
            
            graph.drawImageAttr_grid(fig.add_subplot(gs[i*2+1,6]), models.get_true_models().get_models()[i_rnd,:,:], cmap=models_cmap, colorbar=COLORBAR, title='TRUE MODEL')
        
        plt.show()
        return fig

    def model_view(models:DeepTomographyModels=None, nsamples=1, samples=None, i_offset=0, colorbar=False, models_cmap=DEFAULT_VELOCITY_CMAP):
        if samples is not None:
            nsamples = len(samples)
            
        NMODELS = models.get_nmb_of_elems()
        COLORBAR = colorbar
        NLINES = nsamples
        NCOLS = 4
        
        factor_x = 4
        factor_y = factor_x*nsamples/4
        
        fig = plt.figure(figsize=(graph.PLOT_SX*factor_x,graph.PLOT_SY*factor_y), constrained_layout=True)
        gs = fig.add_gridspec(NLINES, NCOLS)
        
        for i in range(NLINES):
            
            if samples is None:
                i_rnd = random.randint(0, NMODELS-1)
            else:
                i_rnd = samples[i]
                
            if models.get_lowres_models() is not None:
                graph.drawImageAttr_grid(fig.add_subplot(gs[i,0]), models.get_lowres_models().get_models()[i_rnd,:,:], cmap=models_cmap, colorbar=COLORBAR, title=str(i_rnd) + ' - LOWRES MODEL')
            if models.get_mediumres_models() is not None:
                graph.drawImageAttr_grid(fig.add_subplot(gs[i,1]), models.get_mediumres_models().get_models()[i_rnd,:,:], cmap=models_cmap, colorbar=COLORBAR, title=str(i_rnd) + ' - MEDIUMRES MODEL')
            if models.get_highres_models() is not None:
                graph.drawImageAttr_grid(fig.add_subplot(gs[i,2]), models.get_highres_models().get_models()[i_rnd,:,:], cmap=models_cmap, colorbar=COLORBAR, title=str(i_rnd) + ' - HIGRES MODEL')
            graph.drawImageAttr_grid(fig.add_subplot(gs[i,3]), models.get_true_models().get_models()[i_rnd,:,:], cmap=models_cmap, colorbar=True,title='TRUE MODEL')

            #graph.drawImageAttr_grid(fig.add_subplot(gs[i,4]), models.get_water_mask().get_models()[i_rnd,:,:], cmap="seismic", colorbar=True, title='WATER_MASK')
            
        plt.show()
        return fig    
        
        


