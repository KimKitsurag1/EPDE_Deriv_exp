import numpy as np
from epde.interface.prepared_tokens import CustomTokens, TrigonometricTokens, CacheStoredTokens
from epde.evaluators import CustomEvaluator
import epde.interface.interface as epde_alg
import time

def fitness_value_extractor(history):
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
            x = [u.value for u in j.vals.chromosome.values()][:2]
            x = np.array([u.fitness_value for u in x])
            eq_hist = np.append(eq_hist, x)
        hist.append(eq_hist.reshape((eq_hist.size//2,2)))
    for i in hist:
        if len(i.shape) == 1:
            continue
        else:
            i = i.T
            i[0] = np.amin(i[0])
            i[1] = np.amin(i[1])    
    for i in range(len(hist)):
        hist[i] = hist[i][0]
    return np.array(hist).T

deriv_data = [[],[]]
start_time = time.time()
t = np.load('t.npy')
data = np.load('data.npy')
x = data[:, 0]; y = data[:, 1]
    # x += np.random.normal(0, err_factor*np.min(x), size = x.size)
    # y += np.random.normal(0, err_factor*np.min(y), size = y.size) 
dimensionality = x.ndim - 1
popsize = 7
hiss = []
for i in np.arange(1.3e4,1e5,1e4)[::-1]:
    one_stage_hiss = []
    minn = np.array([])
    print(str(i)*10)
    for j in range(7):
        print(str(j)*50)
        epde_search_obj = epde_alg.epde_search(use_solver = False, dimensionality = dimensionality, boundary = 10,
                                                                       coordinate_tensors = [t,], verbose_params = {'show_moeadd_epochs' : True})    
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
                                                                 preprocessor_kwargs={'epochs_max':i})
        epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=25)
        trig_tokens = TrigonometricTokens(dimensionality = dimensionality)
        factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.8, 0.2]}

        epde_search_obj.fit(data=[x, y], variable_names=['u', 'v'], max_deriv_order=(1,),
                                                    equation_terms_max_number=3, data_fun_pow = 1, additional_tokens=[trig_tokens,], 
                                                    equation_factors_max_number=factors_max_number,
                                                    eq_sparsity_interval=(1e-10, 1e-4), coordinate_tensors=[t, ])
        epde_search_obj.equation_search_results(only_print = True)
        one_stage_hiss.append(epde_search_obj.optimizer.history)
    for k in one_stage_hiss:
        if set(fitness_value_extractor(k)[0]) == set() or set(fitness_value_extractor(k)[1]) == set():
            continue
        print(np.amin(fitness_value_extractor(k)[0]))
        print(np.amin(fitness_value_extractor(k)[1]))
        minn = np.append(minn, [np.amin(fitness_value_extractor(k)[0]),np.amin(fitness_value_extractor(k)[1])])
    hiss.append(minn)
    deriv_data[0].append(epde_search_obj.cache[1].memory_default[('du/dx1', (1.0,))])
    deriv_data[1].append(epde_search_obj.cache[1].memory_default[('dv/dx1', (1.0,))])
    np.save('system_diff.npy' , deriv_data)
    np.save('ann_sde_data.npy', hiss)
print(time.time() - start_time)