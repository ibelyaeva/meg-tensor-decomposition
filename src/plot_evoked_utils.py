import os as os
import file_service as fs
import mne

import matplotlib.pyplot as plt
from mne.viz.utils import _connection_line
from mne.viz.utils import _check_time_unit
from numbers import Integral
import save_plots_util as plt_util
from mne.viz.utils import _setup_vmin_vmax
from mpl_toolkits.axes_grid1 import (make_axes_locatable, ImageGrid,
                                     inset_locator)
from tsmoothie.smoother import *
from mne.viz.utils import tight_layout


def find_peaks(evoked, npeaks, tmax):
    """Find peaks from evoked data.
    Returns ``npeaks`` biggest peaks as a list of time points.
    """
    evoked_bk = evoked.copy()
    evoked_bk = evoked_bk.crop(0,tmax)
    from scipy.signal import argrelmax
    gfp = evoked_bk.data.std(axis=0)
    order = len(evoked_bk.times) // 30
    if order < 1:
        order = 1
    peaks = argrelmax(gfp, order=order, axis=0)[0]
    if len(peaks) > npeaks:
        max_indices = np.argsort(gfp[peaks])[-npeaks:]
        peaks = np.sort(peaks[max_indices])
    times = evoked_bk.times[peaks]
    if len(times) == 0:
        times = [evoked_bk.times[gfp.argmax()]]
    print(times)
    return times

def prepare_joint_axes(n_maps, figsize=None):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize)
    main_ax = fig.add_subplot(212)
    ts = n_maps + 2
    map_ax = [plt.subplot(4, ts, x + 2 + ts) for x in range(n_maps)]
    # Position topomap subplots on the second row, starting on the
    # second column
    return fig, main_ax, map_ax

def set_contour_locator(vmin, vmax, contours):
    """Set correct contour levels."""
    locator = None
    if isinstance(contours, Integral) and contours > 0:
        from matplotlib import ticker
        # nbins = ticks - 1, since 2 of the ticks are vmin and vmax, the
        # correct number of bins is equal to contours + 1.
        locator = ticker.MaxNLocator(nbins=contours + 1)
        contours = locator.tick_values(vmin, vmax)
    return locator, contours

def plot_evoked(evoked, target_folder, k, cond_name, tmax = 0.8, npeaks = 3, p_val = None):
    sphere = (0., 0., -0.00332, 0.18)
    times = find_peaks(evoked, npeaks,tmax)
    print("Component peaks = " + str(times))
    fig, axes = plt.subplots(1, figsize=(7, 3))
    
    format_axes(axes)
    figure = evoked.plot(spatial_colors=True, gfp=True, time_unit = 'ms', show=False,
                       units = dict(grad=''), scalings = {'grad':1}, axes = axes,
                      titles = {'grad':''}, sphere = sphere, window_title = None,
                                   )
    fig_folder = os.path.join(target_folder, 'fig')
    fig_id = 'comp_' + str(k+1) + '_' + cond_name + '.pdf'
    fig_id = os.path.join(fig_folder, fig_id)
    fs.ensure_dir(fig_folder)
    comp_name = 'Component ' + str(k+1)

    x_ticks = np.arange(-100,1100,100) 
    axes.set_xticks(x_ticks)
    axes.set(title=comp_name)
    y_label = "A.U"
    axes.set_ylabel(y_label)
    t_stim_end = 800
    ymin = np.min(evoked.data)
    ymax = np.max(evoked.data)
    axes.vlines(0, ymin, ymax, linestyles ="dotted", colors ="k")
    axes.vlines(t_stim_end,ymin, ymax, linestyles ="dotted", colors ="k")
    
    plt.subplots_adjust(bottom=0.14)
    
    plt.savefig(fig_id)
    plt.close()
    
def plot_evoked_with_figure_id(evoked, target_folder, k, cond_name, fig_id, tmax = 0.8, npeaks = 3, p_val = None):
    sphere = (0., 0., -0.00332, 0.18)
    times = find_peaks(evoked, npeaks,tmax)
    print("Component peaks = " + str(times))
    fig, axes = plt.subplots(1, figsize=(7, 3))
    
    format_axes(axes)
    figure = evoked.plot(spatial_colors=True, gfp=True, time_unit = 'ms', show=False,
                       units = dict(grad=''), scalings = {'grad':1}, axes = axes,
                      titles = {'grad':''}, sphere = sphere, window_title = None,
                                   )
    fig_folder = os.path.join(target_folder, 'fig')
    fig_id = fig_id +  '.pdf'
    fig_id = os.path.join(fig_folder, fig_id)
    fs.ensure_dir(fig_folder)
    comp_name = 'Component ' + str(k+1)
   
    x_ticks = np.arange(-100,1100,100) 
    axes.set_xticks(x_ticks)
    axes.set(title=comp_name)
    y_label = "A.U"
    axes.set_ylabel(y_label)
    t_stim_end = 800
    ymin = np.min(evoked.data)
    ymax = np.max(evoked.data)
    axes.vlines(0, ymin, ymax, linestyles ="dotted", colors ="k")
    axes.vlines(t_stim_end,ymin, ymax, linestyles ="dotted", colors ="k")
    
    plt.subplots_adjust(bottom=0.14)
    
    plt.savefig(fig_id)
    plt.close()
    
def plot_evoked_with_figure_id_limits(evoked, target_folder, k, cond_name, fig_id, tmax = 0.8, npeaks = 3, p_val = None, y_min_ = None, y_max_ = None):
    sphere = (0., 0., -0.00332, 0.18)
    times = find_peaks(evoked, npeaks,tmax)
    print("Component peaks = " + str(times))
    fig, axes = plt.subplots(1, figsize=(7, 3))
    
    format_axes(axes)
    figure = evoked.plot(spatial_colors=True, gfp=True, time_unit = 'ms', show=False,
                       units = dict(grad=''), scalings = {'grad':1}, axes = axes,
                      titles = {'grad':''}, sphere = sphere, window_title = None,
                                   )
    fig_folder = os.path.join(target_folder, 'fig')
    fig_id = fig_id +  '.pdf'
    fig_id = os.path.join(fig_folder, fig_id)
    fs.ensure_dir(fig_folder)
    comp_name = 'Component ' + str(k+1)

    x_ticks = np.arange(-100,1100,100) 
    axes.set_xticks(x_ticks)
    axes.set(title=comp_name)
    y_label = "A.U"
    axes.set_ylabel(y_label)
    t_stim_end = 800
    
    if y_min_:
        ymin = y_min_
    else:
        ymin = np.min(evoked.data)
        
    if y_max_:
        ymax = y_max_
    else:
        ymax = np.max(evoked.data)
        
    axes.vlines(0, ymin, ymax, linestyles ="dotted", colors ="k")
    axes.vlines(t_stim_end,ymin, ymax, linestyles ="dotted", colors ="k")
    
    plt.subplots_adjust(bottom=0.14)
    
    plt.savefig(fig_id)
    plt.close()
    
def plot_evoked_original(evoked, target_folder, k, cond_name, tmax = 0.8, npeaks = 3, p_val = None):
    sphere = (0., 0., -0.00332, 0.18)
    times = find_peaks(evoked, npeaks,tmax)
    print("Component peaks = " + str(times))
    fig, axes = plt.subplots(1, figsize=(7, 3))
    
    format_axes(axes)
    figure = evoked.plot(spatial_colors=True, gfp=True, time_unit = 'ms', show=False,
                       units = dict(grad='fT/cm'), scalings = {'grad':1e13}, axes = axes,
                      titles = {'grad':''}, sphere = sphere, window_title = None,
                                   )
    fig_folder = os.path.join(target_folder, 'fig')
    fig_id = 'input_' + cond_name + '.pdf'
    fig_id = os.path.join(fig_folder, fig_id)
    fs.ensure_dir(fig_folder)
    comp_name = 'Condition (input): ' + cond_name.upper()

    x_ticks = np.arange(-100,1100,100) 
    axes.set_xticks(x_ticks)
    axes.set(title=comp_name)
    y_label = "fT/cm"
    axes.set_ylabel(y_label)
    t_stim_end = 800
    ymin = np.min(evoked.data)
    ymax = np.max(evoked.data)
    axes.vlines(0, ymin, ymax, linestyles ="dotted", colors ="k")
    axes.vlines(t_stim_end,ymin, ymax, linestyles ="dotted", colors ="k")
    
    plt.subplots_adjust(bottom=0.14)
    
    plt.savefig(fig_id)
    plt.close()
    
    fig_id = os.path.join(fig_folder,cond_name + '_condition_joint')
    time_unit = dict(time_unit="ms")
    sphere = (0., 0., -0.00332, 0.18)
    topomap_args = {}
    topomap_args['time_unit'] = "ms"
    topomap_args['scalings'] = {'grad':1e13}
    topomap_args['units'] = dict(grad='fT/cm')
    topomap_args['sphere'] = sphere
    extrapolate='head'
    outlines='head'
    topomap_args['extrapolate'] = extrapolate
    topomap_args['outlines'] = outlines
    
    evoked.plot_joint(title=comp_name, ts_args=time_unit, show=False,
                  topomap_args=topomap_args)  # all evoked
    
    plt.savefig(fig_id)
    plt.close()
    
def plot_evoked_with_topo(evoked, target_folder, k, cond_name, tmax = 0.8, npeaks = 3, p_val = None, ch_type = 'grad', times=None, times_ts = None, title = None):
    
    print("Component times = " + str(times))   
    if times is not None:
        times_sec = times
    else:
        times_sec = find_peaks(evoked, npeaks, tmax)
    
    if  times_ts is None:
        _, times_ts = _check_time_unit('ms', times_sec)

    print("Component Original peaks = " + str(times_sec))  
    print("Component Times Ts = " + str(times_sec))   
    
    fig, ts_ax, map_ax = prepare_joint_axes(len(times_sec),
                                                          figsize=(8.0, 4.2))
    
    format_axes(ts_ax)
    
    x_ticks = np.arange(-100,1200,100) 
    ts_ax.set_xticks(x_ticks)
    #plot evoked  
    sphere = (0., 0., -0.00332, 0.18)
    figure = evoked.plot(spatial_colors=True, gfp=True, time_unit = 'ms', 
                       units = dict(grad=''), scalings = {'grad':1}, axes = ts_ax,
                      titles = {'grad':''}, sphere = sphere, window_title = None,
                                   )
    fig_folder = os.path.join(target_folder, 'fig')
    fig_id = 'comp_' + str(k+1) + '_' + cond_name + '_topo' + '.pdf'
    fig_id = os.path.join(fig_folder, fig_id)
    fs.ensure_dir(fig_folder)
    peaks = find_peaks(evoked, npeaks, tmax)
    print("Component peaks = " + str(peaks))
    comp_name = 'Component ' + str(k+1)

    
    if title is None:
        title = comp_name
    
    ts_ax.set(title=title)
    y_label = "A.U"
    ts_ax.set_ylabel(y_label)
    t_stim_end = 800
    ymin = np.min(evoked.data)
    ymax = np.max(evoked.data)
    ts_ax.vlines(0, ymin, ymax, linestyles ="dotted", colors ="k")
    ts_ax.vlines(t_stim_end,ymin, ymax, linestyles ="dotted", colors ="k")
    
    #topomap
    time_unit = dict(time_unit="ms")
    sphere = (0., 0., -0.00332, 0.18)
    topomap_args = {}
    topomap_args['time_unit'] = "ms"
    topomap_args['sphere'] = sphere
    extrapolate='head'
    outlines='head'
    scalings = {'grad':1}
    topomap_args['extrapolate'] = extrapolate
    topomap_args['outlines'] = outlines
    topomap_args['scalings'] = scalings
    
    vmin, vmax = ts_ax.get_ylim()
    norm = ch_type == 'grad'
    vmin = 0 if norm else vmin
    vmin, vmax = _setup_vmin_vmax(evoked.data, vmin, vmax, norm)
    
    evoked.plot_topomap(times=times_sec, axes=map_ax, show=False,ch_type='grad',
                        colorbar=False, sensors=True,res=300, size=3,
                        time_unit='ms', contours=6, image_interp='spline16', average=0.005,
                              extrapolate='head', units=dict(grad=''), scalings = scalings,
                               outlines='head', sphere=sphere 
                       )
    
    plt.subplots_adjust(left=.1, right=.93, bottom=0.1,
                            top=1.0)
    
    lines = [_connection_line(timepoint, fig, ts_ax, map_ax_)
             for timepoint, map_ax_ in zip(times_ts, map_ax)]
    for line in lines:
        fig.lines.append(line)
    
    # mark times in time series plot
    for timepoint in times_ts:
        ts_ax.axvline(timepoint, color='grey', linestyle="dotted",
                      linewidth=1.5, alpha=.66, zorder=0)
    
    from matplotlib import ticker   
    divider = make_axes_locatable(ts_ax)
    ax_colorbar = divider.append_axes('right', size='2%', pad=0.08)
    cbar = plt.colorbar(map_ax[0].images[0], cax=ax_colorbar)
    
    contours = 6
    if not isinstance(contours, (list, np.ndarray)):
        _, contours = set_contour_locator(vmin, vmax, 6)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.set(title='A.U')
    cbar.update_ticks()
    
    plt.savefig(fig_id)
    plt.close()
    
    
    
def time_mask(times, tmin=None, tmax=None, sfreq=None, raise_error=True,
               include_tmax=True):
    """Safely find sample boundaries."""
    orig_tmin = tmin
    orig_tmax = tmax
    tmin = -np.inf if tmin is None else tmin
    tmax = np.inf if tmax is None else tmax
    if not np.isfinite(tmin):
        tmin = times[0]
    if not np.isfinite(tmax):
        tmax = times[-1]
        include_tmax = True  # ignore this param when tmax is infinite
    if sfreq is not None:
        # Push to a bit past the nearest sample boundary first
        sfreq = float(sfreq)
        tmin = int(round(tmin * sfreq)) / sfreq - 0.5 / sfreq
        tmax = int(round(tmax * sfreq)) / sfreq
        tmax += (0.5 if include_tmax else -0.5) / sfreq
    else:
        assert include_tmax  # can only be used when sfreq is known
    if raise_error and tmin > tmax:
        raise ValueError('tmin (%s) must be less than or equal to tmax (%s)'
                         % (orig_tmin, orig_tmax))
    mask = (times >= tmin)
    mask &= (times <= tmax)
    if raise_error and not mask.any():
        extra = '' if include_tmax else 'when include_tmax=False '
        raise ValueError('No samples remain when using tmin=%s and tmax=%s %s'
                         '(original time bounds are [%s, %s])'
                         % (orig_tmin, orig_tmax, extra, times[0], times[-1]))
    return mask
    
def format_axes(ax):

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
        
def plot_factor(condition, tc, tp, k, fig_id = None,title=None, p_value=None, ch_type='grad'):
        colors = ["crimson", "yellow" ]
        scaling = 1e15
        print ("P VAL = " + str(p_value))
        label_font = {
        'size': 9,
        }
            
        title = "Temporal Factor "  + ' ' + str(k+1)
        
        fig, axes = plt.subplots(1, figsize=(8, 3))
        times = tp.times * 1e3
        
        lines = list()
        
        x_ticks = np.arange(-100,1200,100)   
        axes.set(title=title, xlim=times[[0, -1]], xlabel='Time (ms)', ylabel='A.U')
        
        x_label = "Time (ms)"
        y_label = "A.U"
        
        tmin = -100
        tmax = np.max(times)
    
        ymin = np.min(tc)
        ymax = np.max(tc)
        
        vmin = np.min(np.array(ymin))
        vmax = np.max(np.array(ymax))
        
        t_stim_end = 800
        label =  condition.upper()
        
        cnt = 0
        tc = tc*scaling
        format_axes(axes)
        data = tc.flatten()
        print("data shape = " + str(data.shape))
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
            p_fmt = format_number(p_value)
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
            

def format_number(x, fmt='%1.2e'):
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