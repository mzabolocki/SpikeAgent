import numpy as np
import pylab
import shutil
import os
import matplotlib.pyplot as plt 
from matplotlib import gridspec, rcParams
import spikeinterface.widgets as sw
import spikeinterface.qualitymetrics as qm
from spikeinterface.core.npzsortingextractor import NpzSortingExtractor
import spikeinterface.full as si
import spikeinterface.core as sc 
from probeinterface.plotting import plot_probe

def remove_ax(axs, n_unit):
    for ax in axs.flatten()[n_unit:]:
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.set_xticks([])
        ax.set_yticks([])
        
        
def plot_isi(sorting, params):
    n_columns = params['n_columns']
    n_row = int(np.ceil(sorting.unit_ids.shape[0]/n_columns))
    fig,axs = plt.subplots(n_row,n_columns,figsize=(n_columns*2,n_row))
    sw.plot_isi_distribution(sorting, window_ms=200.0, bin_ms=1.0,axes=axs)
    n_unit = len(sorting.unit_ids)
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    remove_ax(axs,n_unit)
    plt.tight_layout()
    plt.show()
    

def plot_templates(we, params, folder_paths):    
    waveform_folder = folder_paths.waveform_folder
    if params['curated']:
        waveform_folder = folder_paths.curated_waveform_folder
    colored_template = params['colored_template']
    n_columns = params['n_columns']
    figure_height = params['figure_height']
    probe_zoomin = params['probe_zoomin']
    save_figures = params['save_figures']
        
    we._template_cache={}
    
    n_unit = len(we.unit_ids)
    n_row = int(np.ceil(n_unit / n_columns))  # Number of rows based on units
    
    colors = []
    cm = pylab.get_cmap('rainbow')
    for i in range(n_unit):
        if colored_template:
            colors.append(cm(1. * i / n_unit))
        else:
            colors.append('black')

    fig, axs = plt.subplots(n_row, n_columns, figsize=(n_columns*5, figure_height*n_row/10))

    # Ensure axs is a 2D array for consistency when indexing
    if n_row == 1:
        axs = np.array([axs])  # Make sure axs is always 2-dimensional

    sw.plot_unit_templates(we, unit_ids=we.unit_ids, axes=axs.flatten())
    
    extremum_channels_ids = sc.get_template_extremum_channel(we, peak_sign='neg')
    for i, unit_id in enumerate(we.unit_ids):
        ax = axs.flatten()[i]
        if probe_zoomin:
            ec_idx = np.where(we.channel_ids==extremum_channels_ids[unit_id])[0].item()
            y_loc = we.get_channel_locations()[ec_idx][1]
            ax.set_ylim(y_loc-figure_height, y_loc+figure_height)
        for line in ax.lines:
            line.set_color(colors[i])

    remove_ax(axs,n_unit)
    
    if save_figures:
        plt.savefig(waveform_folder / 'templates.png',dpi=300)
        print(f"Templates saved in {waveform_folder / 'templates.png'}")
    plt.show()
    #plt.close()


def sorting_unit_show(we, params, folder_paths):
    waveform_folder = folder_paths.waveform_folder
    if params['curated']:
        waveform_folder = folder_paths.curated_waveform_folder
    colored_template = params['colored_template']
    n_columns = params['n_columns']
    save_figures = params['save_figures']
    
    unit_list = we.unit_ids
    n_unit = len(unit_list)
    
    extremum_channels_ids = sc.get_template_extremum_channel(we, peak_sign='neg')
    
    colors=[]
    cm = pylab.get_cmap('rainbow')
    for i in range(n_unit):
        if colored_template:
            colors.append(cm(1. * i / n_unit))  # color will now be an RGBA tuple
        else:
            colors.append('black')
            
    n_row = int(np.ceil(n_unit/n_columns))
    fig, axs = plt.subplots(n_row, n_columns, figsize=(n_columns*5, 5*n_row))

    for i, unit_id in enumerate(unit_list):
        ax = axs.flatten()[i]
        ec_idx = np.where(we.channel_ids==extremum_channels_ids[unit_id])[0].item()
        template = we.get_template(unit_id)[:, ec_idx].T
        ax.plot(template, lw=3,label=unit_id,color=colors[i])
        ax.set_title(f'Template {unit_id}')
        
    remove_ax(axs,n_unit)
    if save_figures:
        plt.savefig(waveform_folder / 'extremum_templates.png',dpi=300)
        print(f"Extremum templates saved in {waveform_folder / 'extremum_templates.png'}")
    plt.tight_layout()
    plt.show()
    #plt.close()
    
    
def show_rasters(sorting, day_length, params, folder_paths):
    rasters_save_path = folder_paths.pack_folder / 'rasters.png'
    if params['curated']:
        rasters_save_path = folder_paths.pack_folder / 'curated_rasters.png'
    save_figures = params['save_figures']
    
    fig, ax = plt.subplots(1,1,figsize=(8,1+len(sorting.unit_ids)/4))
    sw.plot_rasters(sorting,  time_range=(0, day_length),ax=ax)
    if save_figures:
        plt.savefig(rasters_save_path ,dpi=300)
        print(f"Rasters saved in {rasters_save_path}")
    plt.show()
    #plt.close()
    
    
def show_probe(recording, params):
    probe_zoomin = params['probe_zoomin']
    figure_height = params['figure_height']
    fig, ax = plt.subplots(figsize=(5, figure_height/10))
    si.plot_probe_map(recording, ax=ax, with_channel_ids=True)
    if probe_zoomin:
        y_med = np.median(recording.get_channel_locations()[:,1])
        ax.set_ylim(y_med-figure_height, y_med+figure_height)
    plt.show()
    #plt.close()
    

def show_recording(recording, recording_cmr):
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    si.plot_timeseries(recording, backend='matplotlib',  clim=(-50, 50), ax=axs[0])
    si.plot_timeseries(recording_cmr, backend='matplotlib',  clim=(-50, 50), ax=axs[1])
    for i, label in enumerate(('Raw', 'CMR')):
        axs[i].set_title(label)


def save_waveform(we, params, folder_paths):
    waveform_folder = folder_paths.waveform_folder
    if params['curated']:
        waveform_folder = folder_paths.curated_waveform_folder
    colored_template = params['colored_template']
    
    etm_folder = waveform_folder / 'extremum_templates'
    if etm_folder.exists():
        shutil.rmtree(etm_folder)
    os.mkdir(etm_folder)
        
    unit_list = we.unit_ids
    n_unit = len(unit_list)
    
    extremum_channels_ids = sc.get_template_extremum_channel(we, peak_sign='neg')
    
    colors=[]
    cm = pylab.get_cmap('rainbow')
    for i in range(n_unit):
        if colored_template:
            colors.append(cm(1. * i / n_unit))  # color will now be an RGBA tuple
        else:
            colors.append('black')
            
    
    for i, unit_id in enumerate(unit_list):
        fig = plt.figure(figsize=(4,4))
        ec_idx = np.where(we.channel_ids==extremum_channels_ids[unit_id])[0].item()
        template = we.get_template(unit_id)[:, ec_idx].T
        plt.plot(template, lw=3,label=unit_id,color=colors[i])
        plt.title(f'Template {unit_id}')
        plt.savefig(etm_folder / f'extremum_templates_{unit_id}.png')
        plt.close()
        

def save_all_in_one(we, params, folder_paths):
    waveform_folder = folder_paths.waveform_folder
    if params['curated']:
        waveform_folder = folder_paths.curated_waveform_folder
    probe_zoomin = params['probe_zoomin']
    figure_height = params['figure_height']
    aio_folder = waveform_folder / 'all_in_one'
    if aio_folder.exists():
        shutil.rmtree(aio_folder)
    os.mkdir(aio_folder)
        
    unit_list = we.unit_ids
    
    extremum_channels_ids = sc.get_template_extremum_channel(we, peak_sign='neg')
    isi_rate, _ = qm.compute_isi_violations(we, isi_threshold_ms=1.5 , min_isi_ms=0)
        
    for unit_id in unit_list:
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        
        ec_idx = np.where(we.channel_ids==extremum_channels_ids[unit_id])[0].item()
        template = we.get_template(unit_id)[:, ec_idx].T
        
        axs[0].plot(template, color='black')
        sw.plot_unit_templates(we, unit_ids=[unit_id], axes=[axs[1]])
        for line in axs[1].lines:
            line.set_color('black')
        if probe_zoomin:
            y_loc = we.get_channel_locations()[ec_idx][1]
            axs[1].set_ylim(y_loc-figure_height, y_loc+figure_height)
        sw.plot_isi_distribution(we.sorting.select_units(unit_ids=[unit_id]), window_ms = 150, bin_ms=2.0, axes=[axs[2]])
        axs[2].set_title(f'ISI violation : {isi_rate[unit_id]:0.1f}%')
        plt.tight_layout()
        plt.savefig(aio_folder / f'Allinone_{unit_id}.png')
        plt.close()


def summary(metrics, we, keep_unit_ids, params, folder_paths):
    max_isi = params['max_isi']
    min_snr = params['min_snr']
    save_figures = params['save_figures']
    summary_path = folder_paths.pack_folder / 'summary.png'
    
    fig = plt.figure(figsize=(10, 10))
    grid = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.5)
    
    ax = fig.add_subplot(grid[0,0])
    ax.hist(metrics['firing_rate'], bins=50, color='grey')
    ax.set_xlabel('firing rate (Hz)')
    ax.set_ylabel('# of units')
    
    ax = fig.add_subplot(grid[0,1:])
    freq = we.sampling_frequency
    extremum_channels_ids = sc.get_template_extremum_channel(we, peak_sign='neg')
    for unit_id in we.unit_ids:
        ec = np.where(we.channel_ids==extremum_channels_ids[unit_id])[0].item()
        st = we.sorting.get_unit_spike_train(unit_id)/freq
        ecs = [ec for _ in range(st.shape[0])]
        ax.scatter(st, ecs, s=1,color='black',alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channels')
    ax.set_xlim(0,10)
    ax.set_ylim(0,len(we.channel_ids))
    ax.set_title('Spikes from Channels')
    
    
    ax = fig.add_subplot(grid[1,0])
    ax.hist(metrics['snr'],bins=50, color='grey')
    ax.set_xlabel('snr')
    ax.set_ylabel('# of units')
    ax.axvline(min_snr, color='black', linestyle = '--')
    ax.set_title(f'> {min_snr} = good units')
    
    ax = fig.add_subplot(grid[1,1])
    ax.hist(metrics['isi_violations_ratio'],bins=50, color='grey')
    ax.set_xlabel('isi_violations_ratio')
    ax.set_ylabel('# of units')
    ax.axvline(max_isi, color='black', linestyle = '--')
    ax.set_title(f'< {max_isi} = good units')
    
    ax = fig.add_subplot(grid[1,2])
    ax.hist(metrics['amplitude_median'],bins=50, color='grey')
    ax.set_xlabel('amplitude median')
    ax.set_ylabel('# of units')
    good_i = metrics.loc[keep_unit_ids,'firing_rate']
    good_a = metrics.loc[keep_unit_ids,'amplitude_median']
    bad_i = metrics.loc[~metrics.index.isin(keep_unit_ids),'firing_rate']
    bad_a = metrics.loc[~metrics.index.isin(keep_unit_ids),'amplitude_median']

    ax = fig.add_subplot(grid[2,0])
    ax.scatter(good_i, good_a, color='blue', alpha=0.5, label='good',s=5)
    ax.scatter(bad_i, bad_a, color='red', alpha=0.5, label='bad',s=5)
    ax.set_xlabel('firing rate (Hz)')
    ax.set_ylabel('amplitude')
    ax.legend(loc='lower right')
    
    ax = fig.add_subplot(grid[2,1])
    ax.scatter(np.log10(good_i), np.log10(good_a), color='blue', alpha=0.5, label='good',s=5)
    ax.scatter(np.log10(bad_i), np.log10(bad_a), color='red', alpha=0.5,label='bad',s=5)
    ax.set_xlabel('firing rate (Hz)')
    ax.set_ylabel('amplitude')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('loglog')
    ax.legend(loc='lower right')
    if save_figures:
        plt.savefig(summary_path, dpi=300)
        print(f"Summary saved in {summary_path}")
    plt.show()
    #plt.close()

