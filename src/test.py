import sys
import Agent.default as ad
import Agent.read as ar

    
# Set default parameters
params = ad.get_default_params(False)
 
# Set folder paths according to params
folder_paths = ad.Folder_paths(params)
print(folder_paths.raw_data_folder)
folder_paths.raw_data_folder = '/Users/liangningyue/Documents/GitHub/Spike_Agent/data/raw_data/0330'
folder_paths.processed_data_folder = '/Users/liangningyue/Documents/GitHub/Spike_Agent/data/processed_data'
recording, day_length = ar.read_data_folder(params, folder_paths)