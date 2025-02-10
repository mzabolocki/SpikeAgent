import os
import shutil
import numpy as np
import json
from spikeinterface.core.npzsortingextractor import NpzSortingExtractor
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.core as sc
import spikeinterface.full as si
import Agent.run_kilosort as ak
import Agent.default as ad



def get_we(recording, sorting, params, folder_paths):
    load_if_exist = params['load_if_exist']
    waveform_folder = folder_paths.waveform_folder
    if not load_if_exist and waveform_folder.exists():
        shutil.rmtree(waveform_folder)
            
    if not waveform_folder.exists():
        sc.extract_waveforms(recording, sorting, waveform_folder,
                ms_before=1, ms_after=2., max_spikes_per_unit=1000000,
                return_scaled=False,
                n_jobs=-1, chunk_size=30000)
    
    we = si.load_waveforms(waveform_folder)
    return we
    

def run_sorter(recording, params, sorting_params, folder_paths):
    load_if_exist = params['load_if_exist']
    sorting_folder = folder_paths.sorting_folder
    firing_save_path = folder_paths.firing_save_path
    pack_folder = folder_paths.pack_folder
    sorter_name = params['sorter_name']
    
    if not pack_folder.exists(): pack_folder.mkdir()
    if not load_if_exist and sorting_folder.exists():
        shutil.rmtree(sorting_folder)
    if (pack_folder/'sorting_params.json').exists():
        with open (pack_folder/'sorting_params.json', 'r') as f:
            prev_sorting_params = json.load(f)
            if prev_sorting_params != sorting_params:
                shutil.rmtree(pack_folder)
                pack_folder.mkdir()
    with open (pack_folder/'sorting_params.json', 'w') as f:
            json.dump(sorting_params, f)
    if not firing_save_path.exists():
        # Modified
        ###########################################################################################
        jobs = ad.get_jobs(sorter_name, recording, sorting_folder, **sorting_params)
        if sorter_name == 'kilosort4':
            sorting_wave_clus = ak.run_sorter(**jobs)
        else: 
            sorting_wave_clus = ss.run_sorter(**jobs)
        ###########################################################################################
        keep_unit_ids = []
        for unit_id in sorting_wave_clus.unit_ids:
            spike_train = sorting_wave_clus.get_unit_spike_train(unit_id=unit_id)
            n = spike_train.size
            if(n>20):
                keep_unit_ids.append(unit_id)

        curated_sorting = sorting_wave_clus.select_units(unit_ids=keep_unit_ids, renamed_unit_ids=None)
        NpzSortingExtractor.write_sorting(curated_sorting, firing_save_path)

    sorting = se.NpzSortingExtractor(firing_save_path)
    print(f"Sorting saved in {sorting_folder}")
    print(f"Sorting paramters saved in {pack_folder / 'sorting_params.json'}")
    
    
    return sorting
    
def units_merge(recording_cmr, sorting, curated_units, params, folder_paths, merge_unit_ids_pack = []):
    waveform_folder = folder_paths.curated_waveform_folder
    firing_save_path= folder_paths.curated_firing_save_path
    pack_folder = folder_paths.pack_folder
    load_if_exist=params['load_if_exist']
    
    if not load_if_exist:
        if waveform_folder.exists():
            shutil.rmtree(waveform_folder)
        if firing_save_path.exists():
            os.remove(firing_save_path)
    
    if (pack_folder / 'curated_units.npy').exists():
        prev_curated_units = np.load(pack_folder / 'curated_units.npy')
        if not np.array_equal(prev_curated_units, curated_units):
            shutil.rmtree(waveform_folder)
            os.remove(firing_save_path)
    np.save(pack_folder / 'curated_units.npy', curated_units)
    if not firing_save_path.exists():
        delete_unit_ids_pack = [item for item in list(sorting.unit_ids) if item not in curated_units]
        S = sorting._sorting_segments[0]
        merged_sorting = sorting
        remove_ids = []
        
        for idx in range(len(merge_unit_ids_pack)):
            merge_unit_ids = merge_unit_ids_pack[idx]

            for unit_id_id, unit_id in enumerate(merge_unit_ids):
                S.spike_labels[S.spike_labels == unit_id] = merge_unit_ids[0]

            merged_sorting._sorting_segments[0] = S

            remove_ids.extend(merge_unit_ids[1:])

        remove_ids+=delete_unit_ids_pack

        keep_ids = merged_sorting.unit_ids[~np.isin(merged_sorting.unit_ids, remove_ids)]
        np.save(pack_folder / 'curated_units.npy', keep_ids)
        merged_sorting = merged_sorting.select_units(unit_ids=keep_ids, renamed_unit_ids=None)  
        NpzSortingExtractor.write_sorting(merged_sorting, firing_save_path)    

    if not waveform_folder.exists():
        sc.extract_waveforms(recording_cmr, merged_sorting, waveform_folder,
                ms_before=1, ms_after=2., max_spikes_per_unit=1000000,
                return_scaled=False,
                n_jobs=-1, chunk_size=30000)
    
    merged_sorting = se.NpzSortingExtractor(firing_save_path)    
    merged_we = si.load_waveforms(waveform_folder)
    sorting = merged_sorting
    we = merged_we
    #we._template_cache=[]
    #we.run_extract_waveforms()     
    return sorting,we