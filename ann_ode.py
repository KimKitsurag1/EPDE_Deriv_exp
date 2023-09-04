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

deriv_data = [[],[]]
deriv_stage = [[],[]]
start_time = time.time()
t = np.linspace(0,10,1000)
v = np.load('ode_prep.npy')
hiss = []
one_stage_hiss = []
for i in np.arange(1.3e4,1e5,1e4)[::-1]:
    one_stage_hiss = []
    minn = np.array([])
    print(str(i)*10)
    for j in range(10):
        print(str(j)*50)
        epde_search_obj3 = epde_alg.epde_search(use_solver = False, dimensionality = 0, boundary = 10 ,coordinate_tensors = [t,],
                                                verbose_params = {'show_moeadd_epochs' : False})
        custom_grid_tokens = CacheStoredTokens(token_type = 'grid',
                                        token_labels = ['t'],
                                        token_tensors = {'t' : t},
                                        params_ranges = {'power' : (1, 1)},
                                        params_equality_ranges = None)
        trig_tokens = TrigonometricTokens(dimensionality = 0)
        factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.8, 0.2]}
        epde_search_obj3.set_moeadd_params(population_size=4, training_epochs = 15)
        epde_search_obj3.set_preprocessor(default_preprocessor_type='ANN', preprocessor_kwargs={'epochs_max':i})
        epde_search_obj3.fit(data = v, max_deriv_order=2,  equation_terms_max_number = 4, equation_factors_max_number = factors_max_number,coordinate_tensors = [t,],
                            eq_sparsity_interval = (1e-8, 1),additional_tokens = [trig_tokens, custom_grid_tokens])
#         epde_search_obj3.equation_search_results(only_print = True)
        one_stage_hiss.append(epde_search_obj3.optimizer.history)
        deriv_stage[0].append(epde_search_obj3.cache[1].memory_default[('du/dx1', (1.0,))])
        deriv_stage[1].append(epde_search_obj3.cache[1].memory_default[('d^2u/dx1^2', (1.0,))])
    for k in one_stage_hiss:
        print(type(fitness_value_extractor_1d(k)))
        print(fitness_value_extractor_1d(k))
        if set(fitness_value_extractor_1d(k)) == set():
            continue
        minn = np.append(minn, np.amin(fitness_value_extractor_1d(k)))
    hiss.append(minn)
    deriv_data[0].append(deriv_stage[0])
    deriv_data[1].append(deriv_stage[1])
    np.save('ode_diff.npy', deriv_data)
    np.save('ann_ode_data.npy',hiss)
print(time.time()-start_time)