import numpy as np
import pandas as pd
from probeinterface import Probe
import spikeinterface.core as sc
import spikeinterface.qualitymetrics as qm
from spikeinterface.preprocessing import bandpass_filter, common_reference
import spikeinterface.full as si
import spikeinterface.qualitymetrics.quality_metric_calculator as sqq

def create_probe(recording, params):
    if recording.has_probe(): 
        return recording.get_probe()
    positions = params['positions']
    radius = params['radius']
    
    mesh_probe = Probe(ndim=2, si_units='um')
    mesh_probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': radius})
    ant = {'first_index':0}
    mesh_probe.annotate(**ant)
    channel_indices_raw = recording.get_channel_ids()
    channel_indices = [i for i in channel_indices_raw]
    mesh_probe.set_device_channel_indices(channel_indices)
    recording.set_probe(mesh_probe, in_place=True)

    return mesh_probe
    

def save_extremum(we, folder_paths):
    pack_folder = folder_paths.pack_folder
    extremum_channels_ids = sc.get_template_extremum_channel(we, peak_sign='neg')
    pd.DataFrame.from_dict(extremum_channels_ids, orient='index').to_csv(pack_folder / 'extremum_channels_ids.csv')
    print(f"Extremum channels saved in {pack_folder / 'extremum_channels_ids.csv'}")
    

def auto_curation(we,params):
    max_isi = params['max_isi']
    min_snr = params['min_snr']
    metrics = si.compute_quality_metrics(we, metric_names=['firing_rate', 'snr','isi_violation', 'amplitude_median'])
    our_query = f"(snr > {min_snr}) & (isi_violations_ratio < {max_isi})"
    keep_units = metrics.query(our_query)
    keep_unit_ids = keep_units.index.values
    print(f"Curated ID ({len(keep_unit_ids)}/{len(metrics.index)}): {', '.join(map(str, keep_unit_ids))}")
    if params['show_metrics']:
        print(metrics)
    return keep_unit_ids, metrics

def curatoin_params():
    print(sqq.get_default_qm_params().keys())


def filters(recording, params):
    freq_min = params['filter_frequency_minimum']
    freq_max = params['filter_frequency_maximum']
    operator = params['filter_operator']
    
    if freq_max is None:
        recording_f1 = si.highpass_filter(recording, freq_min=freq_min)
        recording_f = si.phase_shift(recording_f1)
    else:
        recording_f = bandpass_filter(recording, freq_min=freq_min, freq_max=freq_max)
    
    recording_cmr = common_reference(recording_f, reference='global',operator=operator)
    return recording_cmr
