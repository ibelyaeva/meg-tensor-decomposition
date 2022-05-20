import mne
import numpy as np
from mne.viz.utils import _prepare_trellis
from mne.viz.utils import _setup_vmin_vmax
from mne.viz.utils import _setup_cmap
from mne.viz.utils import tight_layout
from mne.viz.utils import plt_show
import matplotlib.pyplot as plt
from mne.viz.topomap import _add_colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import save_plots_util as plt_util
import texfig
import math
plt.rcParams['text.usetex'] = True
plt.rcParams["legend.frameon"] = False
from mne import evoked as ev
from scipy import stats
from mne.preprocessing import peak_finder

class Factor(object):
    def __init__(self, tp, n, col_num, component=None, contrast=None, info=None):
        
        self.tp = tp
        self.n_comp = self.tp.data.shape[0]
        self.contrast = contrast
        self.component = component
        self.n = n
        self.col_num = col_num
        self.name = 'IC {:2d}'.format(self.col_num)
        self.fig_name = 'ic{:2d}'.format(self.col_num)
        self.info = info

    def plot_factor(self, condition, k, sp, tc, tp, fig_id = None,title=None, annotate=True, topo_time=None, p_value=None, peaks = None, coord_names=None, ch_type='grad'):
        colors = ["crimson", "yellow" ]
        scalings=1
        
        label_font = {
        'size': 9,
        }
        
        title = "Temporal Factor "  + ' ' + str(k+1)
        
        fig, axes = plt.subplots(1, figsize=(7, 3))
        times = self.tp.times * 1e3
        
        x_ticks = np.arange(-100,1200,100)   
        axes.set(title=title, xlim=times[[0, -1]], xlabel='Time (ms)', ylabel='A.U')
        
        
        x_label = "Time (ms)"
        y_label = "A.U"
        
        tmin = -100
        tmax = np.max(times)
    
        tc = tc*scalings
        ymin = np.min(tc)
        ymax = np.max(tc)
        
        vmin = np.min(np.array(ymin))
        vmax = np.max(np.array(ymax))
        
        t_stim_end = 800
        label = condition.upper()
        
        cnt = 0
        
        self.format_axes(axes)
        data = tc.flatten()
        axes.plot(data, zorder=2, color=colors[cnt], linewidth=3, label=label)
        
        axes.set_xticks(x_ticks)
        axes.set_xlabel(x_label)
            
        axes.set_ylabel(y_label)
            
        axes.set_xlim(tmin, tmax)
        axes.set_ylim(ymin, ymax)
            
        axes.vlines(0, ymin, ymax, linestyles ="dotted", colors ="k")
        axes.vlines(t_stim_end,ymin, ymax, linestyles ="dotted", colors ="k")
            
        axes_title = title
        axes.set_title(axes_title)
            
        axes.legend(loc = 'upper left')
        
        #plot topomap
        divider = make_axes_locatable(axes)
        ax_topo = divider.append_axes('right', size='30%', pad=0.05)
            
        ax_colorbar = divider.append_axes('right', size='2%', pad=0.05)
        
        axes_list = []
        axes_list.append(ax_topo)
        axes_list.append(ax_colorbar)
        
        scalings = {'grad':1}
        vmin_, vmax_ = _setup_vmin_vmax(data, vmin, vmax)
        tp_map = tp.plot_topomap(ch_type='grad', 
                                     sensors=True, colorbar=True,
                                     res=300, size=3, units=dict(grad=''), scalings = scalings,
                             time_unit='ms', contours=6, image_interp='spline16', average=0.05, axes=axes_list,
                              extrapolate='head',
                               outlines='head', sphere=(0., 0., -0.00332, 0.18)
                              )
    
        ax_topo.set_title('Spatial Factor ' + str(k + 1), fontdict=label_font)
        
        if p_value is not None:
            p_fmt = self.format_number(p_value)
            if p_value <= 0.05:
                component_label =r'$^*$' + r'$p=$' + p_fmt
            else:
                component_label = r'$p=$' + p_fmt
            
            ax_topo.set_xlabel(component_label, fontdict=label_font)
        
        if fig_id is not None:
            file_fig_id = fig_id + '_factor' + '_' + condition + '_' + str(cnt + 1)
            tight_layout(fig=fig)
            plt_util.save_fig_pdf_no_dpi(file_fig_id)
            plt.close(fig)
            
    def fmt(self,x):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)
    
    def format_number(self, x, fmt='%1.2e'):
        s = fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
            if significand and exponent:
                s =  r'%s{\times}%s' % (significand, exponent)
            else:
                s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)
    
    def format_axes(self, ax):

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('k')
            ax.spines[spine].set_linewidth(1)

        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('k')
            ax.spines[spine].set_linewidth(1)

            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_tick_params(direction='in', top = True, left = True, right=True, color='k')