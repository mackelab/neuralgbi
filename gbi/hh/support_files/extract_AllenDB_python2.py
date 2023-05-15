import inspect
import numpy as np
import os
import pickle

list_cells_AllenDB = np.array([[464212183,33],[485184849,60],[486132712,41],[485466109,36],[485958978,46],[509881736,39],
                               [566517779,46],[567399060,44],[567399060,38],[569469018,44],[532571720,38],[532571720,42],
                               [532571720,47],[532571720,49],[555060623,33],[555060623,34],[534524026,29],[532355382,33],
                               [526950199,37],[518290966,57],[488683425,36],[488683425,43],[488683425,46],
                               [566517779,44],[566517779,56],[566517779,57],[566517779,58],[566517779,59]])

def extract_data(ephys_cell,sweep_number):
    """Data extraction from AllenDB

    Parameters
    ----------
    ephys_cell : int
        Cell identity from AllenDB
    sweep_number : int
        Stimulus identity for cell ephys_cell from AllenDB
    """
    t_offset = 815.
    duration = 1450.
    real_data_path = 'ephys_cell_{}_sweep_number_{}.pkl'.format(ephys_cell,sweep_number)
    if not os.path.isfile(real_data_path):
        from allensdk.core.cell_types_cache import CellTypesCache
        from allensdk.api.queries.cell_types_api import CellTypesApi

        manifest_file = 'cell_types/manifest.json'

        cta = CellTypesApi()
        ctc = CellTypesCache(manifest_file=manifest_file)
        data_set = ctc.get_ephys_data(ephys_cell)
        sweep_data = data_set.get_sweep(sweep_number)  # works with python2 and fails with python3
        sweeps = cta.get_ephys_sweeps(ephys_cell)

        sweep = sweeps[sweep_number]

        index_range = sweep_data["index_range"]
        i = sweep_data["stimulus"][0:index_range[1]+1] # in A
        v = sweep_data["response"][0:index_range[1]+1] # in V
        sampling_rate = sweep_data["sampling_rate"] # in Hz
        dt = 1e3/sampling_rate # in ms
        i *= 1e6 # to muA
        v *= 1e3 # to mV
        v = v[int(t_offset/dt):int((t_offset+duration)/dt)]
        i = i[int(t_offset/dt):int((t_offset+duration)/dt)]


        real_data_obs = np.array(v).reshape(1, -1, 1)
        I_real_data = np.array(i).reshape(-1)
        t_on = int(sweep['stimulus_start_time']*sampling_rate)*dt-t_offset
        t_off = int( (sweep['stimulus_start_time']+sweep['stimulus_duration'])*sampling_rate )*dt-t_offset
        
        f = open(real_data_path, 'wb')
        pickle.dump((real_data_obs,I_real_data,dt,t_on,t_off), f)
        f.close()

#################################################
for i in range(len(list_cells_AllenDB[:,0])):
    print(i)
    extract_data(list_cells_AllenDB[i,0],list_cells_AllenDB[i,1])
