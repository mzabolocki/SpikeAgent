import numpy as np
from pathlib import Path
import spikeinterface.sorters as ss
# from kilosort import DEFAULT_SETTINGS

MP_DEFAULT_PARAMS = {
        'sorter_name': 'mountainsort4',
        'raw_data_path': './data/raw_data',
        'processed_path': './data/processed_data',
        'folder_name':'0330',
        'probe_zoomin': False,
        'filter_frequency_minimum': 300,
        'filter_frequency_maximum': 3000,
        'filter_operator': 'average',
        'load_if_exist': True}
NP_DEFAULT_PARAMS = {
        'sorter_name': 'kilosort4',
        'raw_data_path': './data/raw_data',
        'processed_path': './data/raw_data',
        'folder_name':'AL031_2019-12-02',
        'probe_zoomin': True,
        'filter_frequency_minimum': 300,
        'filter_frequency_maximum': None,
        'filter_operator': 'median',
        'load_if_exist': True}
SHOW_DEFAULT_PARAMS = {
        'save_figures': True,
        'curated': False,
        'colored_template': False,
        'n_columns': 10,
        'figure_height': 50,
        'show_metrics': False}
CURATION_DEFAULT_PARAMS = {
        'min_snr': 5.0,
        'max_isi': 1.5}

DEFAULT_PROBE = {'positions': np.array([[150, 250], [150,200],[50, 0], 
                    [50, 50],[50, 100], [0, 100],
                    [0, 50], [0, 0],[650, 0], 
                    [650, 50],[650, 100], [600, 100],
                    [600, 50], [600, 0],[500, 200],
                    [500, 250],[500, 300],[450, 300],
                    [450, 250], [450, 200],[350, 400], 
                    [350, 450],[350, 500], [300, 500],
                    [300, 450], [300, 400], [200, 200],
                    [200, 250],[200, 300],[150, 300]]),
                'radius': 5}


###########################################################################################

class Folder_paths():
    def __init__(self, params):
        folder_name = params['folder_name']
        raw_data_path = Path(params['raw_data_path'])
        processed_path = Path(params['processed_path'])
        self.raw_data_folder = raw_data_path / folder_name
        self.output_folder = processed_path / folder_name
        self.recording_folder = processed_path / folder_name / 'recording'

    def update(self, sorter_name):
        self.pack_folder = self.output_folder / sorter_name
        self.sorting_folder = self.pack_folder / 'sorting'
        self.waveform_folder = self.pack_folder / 'waveform'
        self.firing_save_path = self.sorting_folder / 'sorter_output' / 'firings.npz'
        self.curated_waveform_folder = self.pack_folder / 'curated_waveform'
        self.curated_firing_save_path = self.sorting_folder / 'sorter_output' / 'curated_firings.npz'
        
        
def get_default_params(is_neuropixel:bool, **kwargs):
        params = {'is_neuropixel': is_neuropixel}
        if is_neuropixel:
                params.update(NP_DEFAULT_PARAMS)
        else:
                params.update(MP_DEFAULT_PARAMS)
        params.update(SHOW_DEFAULT_PARAMS)
        params.update(CURATION_DEFAULT_PARAMS)
        params.update(DEFAULT_PROBE)
        params.update(kwargs)
        return params
        
def freqmax_is_valid(sorter_name, fs):
    default_sorter_params = ss.get_default_sorter_params(sorter_name)

    if sorter_name in ['spykingcircus2', 'tridesclous2']:
        freq_max = 8000
    elif sorter_name == 'waveclus':
        freq_max = default_sorter_params['sort_filter_fmax']
    else:
        freq_max = default_sorter_params.get('freq_max', 0)

    return freq_max < fs/2 

def get_sorting_params(sorter_name, recording, **kwargs):
    fs = recording.sampling_frequency
    if sorter_name == 'kilosort4':
        sorting_params = DEFAULT_SETTINGS
        for key in kwargs.keys():
            if key in sorting_params:
                sorting_params[key] = kwargs[key]
        sorting_params['n_chan_bin'] = recording.channel_ids.shape[0]
        sorting_params['fs'] = fs
    else:    
        sorting_params = ss.get_default_sorter_params(sorter_name)
        if sorter_name == 'mountainsort4':
            sorting_params['num_workers'] = 10
            sorting_params['adjacency_radius'] = 120
            if recording.channel_ids.shape[0] > 300:
                sorting_params['detect_threshold'] = 6
                sorting_params['freq_max'] = None
        for key1, value1 in sorting_params.items():
            if isinstance(value1, dict):
                for key2, value2 in value1.items():
                    if 'dtype' in key2:
                        sorting_params[key1]['dtype'] =  recording.dtype.name
                    if key2 in kwargs.keys():
                        sorting_params[key1][key2] = kwargs[key2]
            else:
                if key1 in kwargs.keys():
                    sorting_params[key1] = kwargs[key1]

        if 'dtype' in sorting_params:
            sorting_params['dtype'] =  recording.dtype.name
    for key, value in sorting_params.items():
        if isinstance(value, dict):
            for key2, value2 in value.items():
                print(f'{key2}: {value2}')
        else:
                print(f'{key}: {value}')
    print()
    return sorting_params


def get_jobs(sorting_method, recording, sorting_save_path, **sorting_params):
    jobs = {'sorter_name':sorting_method, 'recording': recording, \
            'remove_existing_folder':True, 'output_folder':sorting_save_path}
    jobs.update(sorting_params)
    return jobs


def get_np_folder_paths(settings):
        folder_paths = {}
        results_dir = Path(settings['results_dir'])
        folder_paths['ops_path'] = results_dir / 'ops.npy'
        folder_paths['camps_path'] = results_dir / 'cluster_Amplitude.tsv'
        folder_paths['contam_pct_path'] = results_dir / 'cluster_ContamPct.tsv'
        folder_paths['chan_map_path'] =  results_dir / 'channel_map.npy'
        folder_paths['templates_path'] = results_dir / 'templates.npy'
        folder_paths['amplitudes_path'] = results_dir / 'amplitudes.npy'
        folder_paths['st_path'] = results_dir / 'spike_times.npy'
        folder_paths['clu_path'] = results_dir / 'spike_clusters.npy'
        folder_paths['result_dir'] = results_dir
        
        return folder_paths

def get_sorter_list():
    return ['mountainsort4', 'mountainsort5', 'kilosort4', 'spykingcircus2', 'tridesclous2', 'tridesclous', 'herdingspikes']
