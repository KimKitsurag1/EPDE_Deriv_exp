import numpy as np
from epde.interface.prepared_tokens import CustomTokens, TrigonometricTokens, CacheStoredTokens
from epde.evaluators import CustomEvaluator
import epde.interface.interface as epde_alg
import time

def fitness_value_extractor_1d(history):
    clear_hiss = []
    for i in history:
        if i == []:
            continue
        else:
            clear_hiss.append(i)
    hist = []
    eq_hist = []
    for i in clear_hiss:
        eq_hist=np.array([])
        for j in i:
            x = [u.value for u in j.vals.chromosome.values()][:1]
            x = np.array([u.fitness_value for u in x])
            eq_hist = np.append(eq_hist, x)
        hist.append(np.amin(eq_hist))  
    return np.array(hist).T


t = np.linspace(0, 1, 101)#[:100]
x = np.linspace(0, 1, 101)#[:100]
v = np.loadtxt('wave_sln_100.csv', delimiter = ',')#.T[:100][:,:100]
grids = np.meshgrid(t, x, indexing='ij')
hiss = []
deriv_data = [[],[],[],[]]
for i in np.arange(1.3e4,1e5,1e4)[::-1]:
    print(str(i)*20)
    one_stage_hiss = []
    minn = np.array([])
    for j in range(10):
        print(str(j)*30)
        epde_search_obj4 = epde_alg.epde_search(use_solver = False, dimensionality = 1, boundary = 20 ,coordinate_tensors = grids,
                                                    verbose_params = {'show_moeadd_epochs' : True})
        custom_grid_tokens = CacheStoredTokens(token_type = 'grid',
                                            token_labels = ['t', 'x'],
                                            token_tensors={'t' : grids[0], 'x' : grids[1]},
                                            params_ranges = {'power' : (1, 1)},
                                            params_equality_ranges = None)
        trig_tokens = TrigonometricTokens(dimensionality = 1)
        factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.8, 0.2]}
        epde_search_obj4.set_moeadd_params(population_size=5, training_epochs =15)
        epde_search_obj4.set_preprocessor(default_preprocessor_type='ANN', preprocessor_kwargs={'epochs_max':i})
        epde_search_obj4.fit(data = v, max_deriv_order=(2, 2),  equation_terms_max_number = 4, equation_factors_max_number = factors_max_number,coordinate_tensors = [t,],
                                eq_sparsity_interval = (1e-8, 1e-4),additional_tokens = [custom_grid_tokens,trig_tokens,])
        epde_search_obj4.equation_search_results(only_print = True)
        one_stage_hiss.append(epde_search_obj4.optimizer.history)
        d1s1 = epde_search_obj4.cache[1].memory_default[('du/dx1', (1.0,))]
        d1s2 = epde_search_obj4.cache[1].memory_default[('du/dx2', (1.0,))]
        d2s1 = epde_search_obj4.cache[1].memory_default[('d^2u/dx1^2', (1.0,))]
        d2s2 = epde_search_obj4.cache[1].memory_default[('d^2u/dx2^2', (1.0,))]
        deriv_data[0].append(d1s1)
        deriv_data[1].append(d1s2)
        deriv_data[2].append(d2s1)
        deriv_data[3].append(d2s2)

    for k in one_stage_hiss:
        print(type(fitness_value_extractor_1d(k)))
        print(fitness_value_extractor_1d(k))
        if set(fitness_value_extractor_1d(k)) == set():
            continue
        minn = np.append(minn, np.amin(fitness_value_extractor_1d(k)))
    hiss.append(minn)
    np.save('ann_wave_data.npy', hiss)
    np.save('wave_diff.npy', deriv_data)
    