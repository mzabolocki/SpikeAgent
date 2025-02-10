from kilosort import run_kilosort
from spikeinterface.core.npzsortingextractor import NpzSortingExtractor
import os
import numpy as np
import shutil
import Agent.default as ad

def run_sorter(sorter_name, recording, remove_existing_folder, output_folder, **sorting_params):
    assert sorter_name == 'kilosort4', f'{sorter_name}: wrong sorting method'
    
    if output_folder.exists() and remove_existing_folder:
        shutil.rmtree(output_folder)
    output_folder.mkdir()
            
    recording_folder = output_folder.parent.parent /'recording'
    data_type = recording.dtype
    probe = recording.get_probe()
    kilosort_probe = get_kilosort_probe(probe)
    settings = sorting_params
    settings['data_dir'] = recording_folder
    settings['results_dir'] = output_folder
    extra_settings = {'probe': kilosort_probe, 'data_dtype': data_type, 'do_CAR': True}
    
    run_kilosort(settings = settings, **extra_settings)

    firing_save_path = output_folder / 'sorter_output' / 'firings.npz'
    write_sorting(firing_save_path, settings)
    sorting = NpzSortingExtractor(firing_save_path)
    
    return sorting


def get_kilosort_probe(probe):
    xyc = probe.contact_positions
    chanMap = probe.device_channel_indices
    kcoords = probe.shank_ids
    n_chan = chanMap.shape[0]
    if np.all(kcoords == ''): kcoords[:] = 1

    kilosort_probe = {'xc':xyc[:,0].astype(np.float32),
                    'yc':xyc[:,1].astype(np.float32),
                    'kcoords':kcoords.astype(np.float32),
                    'chanMap':chanMap.astype(np.int32),
                    'n_chan':n_chan
    }
    return kilosort_probe


def write_sorting(firing_save_path, settings):
    if not firing_save_path.parent.exists():
        firing_save_path.parent.mkdir()
        
    np_folder_paths = ad.get_np_folder_paths(settings)
    
    fs = np.load(np_folder_paths['ops_path'], allow_pickle=True).item()['fs']
    st = np.load(np_folder_paths['st_path'])
    clu = np.load(np_folder_paths['clu_path'])
    
    firings = get_firings(fs, st, clu)
    np.savez(firing_save_path, **firings)
    
    
def get_firings(fs, spike_time, spike_label, num_segment=1) -> dict:
    firings = {'unit_ids': np.arange(1,np.max(spike_label)+2,1),
                'num_segment': np.array([num_segment]),
                'sampling_frequency': np.array([fs],dtype=np.float64),
                'spike_indexes_seg0': spike_time,
                'spike_labels_seg0': spike_label.astype(np.int64)}
    
    return firings