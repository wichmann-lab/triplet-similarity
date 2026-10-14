import numpy as np
from scipy import stats

##### LOG ODDS
##### See scripts/felix-code/ for the derivation ########
def logodds(p):
    return np.log(p/(1-p))

# change needed in LogOdds
def change_needed_in_logodds(p=0.75, delta=0.02):
    alpha = logodds(p + delta) - logodds(p)
    return alpha

def delta_p_for_fixed_logoddsincrease(p, alpha):
	""" Derived with Wolfram Mathematica, see Wolfram notebook
	LogOddsIncrease.nb
	"""
	
	num = -p + (np.exp(alpha)*p)/(1-p) - (np.exp(alpha)*p**2)/(1-p)
	denom = 1 + (np.exp(alpha)*p)/(1-p)

	return num/denom


##### SDT utils
def dprime_from_pcorrect(pc, hitrate=None, ps=0.5):
    """ Calculates d' from percentage correct
    
    Function translated from Felix Wichmann's MATLAB function DPrimeFromCorrect
    """
    if hitrate is None:
        hitrate = pc

    # 1. Calculate false alarm rate
    fa_rate = (pc - 1 + ps*(1 - hitrate))/(ps - 1)
    
    # 2. Calculate the criterion lambda
    lmbda = stats.Normal().icdf(1 - fa_rate)

    #% 3. Calculate d-Prime
    d_prime = lmbda - stats.Normal().icdf(1-hitrate)
    
    #% 4. Add values into struct
    return {'percent correct' : pc,
            'hitrate': hitrate,
            'ps': ps,
            'false_alarm_rate': fa_rate,
            'lambda': lmbda,
            'd-prime': d_prime,
           }

def correct_from_dprime(dprime=np.linspace(0, 6, 100), task='2AFC'):
        
    if task=='2AFC':
        # For 2AFC percent correct is the integral from zero to infinity of a
        # Gaussian with SD = sqrt(2) and mean = d'
        p =  1 - stats.norm.cdf(0, loc=dprime, scale=sqrt(2))

    elif task == 'YesNo':
        #  Assuming the same number of signal and no-signal trials, for YesNo
        # percent correct is 1/2 Hits + 1/2 correct rejections. For an optimal
        # observer the criterion is placed at dPrime/2. Given symmetry of criterion
        # and variance of signal and noise distribution, percent correct is the
        # integral from dPrime/2 to infinity of N(dPrime,1), thus:
        p = 1 - stats.norm.cdf(dprime/2, loc=dprime, scale=1);
    else:
        raise('Invalid task specified. Use ''2AFC'' or ''YesNo''.')

    return p

def minimal_pc_change(p, min_dprime_delta = 0.1):

    data = dprime_from_pcorrect(p)
    
    min_dprime_values = data['d-prime'] + min_dprime_delta

    min_p_values = correct_from_dprime(min_dprime_values, 'YesNo')

    return min_p_values


def recalculate_dimensionality(dim_result, min_dprime_delta=None, pval_thr=0.05, criterion='SDT', p=0.75, delta=0.02):
    
    estimated_dimension = dim_result.estimated_dimension
    test_dimensions = dim_result.dimensions
    pvals_corrected = dim_result.stats_result['pvals_corrected']
    test_scores = dim_result.test_scores
    
    # calculating effect size manually from differences
    mean_effect_size = np.mean(np.diff(test_scores, axis=0), axis=1) # mean diffs
    
    # mean scores per dimension
    mean_test_scores = np.mean(test_scores, axis=1) 

    # establish whether to reject adding a new dimension or not
    rejects = []
    min_deltas = []
    for d in range(len(test_dimensions) -1):
        # Criterion 1: output of hypothesis test 
        pval = pvals_corrected[d]
    
        # Criterion 2: minimal change in Pc
        # Case 1: following signal detection theory
        if criterion=='SDT':
            if min_dprime_delta is None:
                raise(ValueError("min_dprime_delta should be set for criterion 'SDT'"))
            min_pc = minimal_pc_change(mean_test_scores[d], min_dprime_delta = min_dprime_delta)
            min_delta = min_pc - mean_test_scores[d]
            
        # Case 2: following change in LogODDS
        elif criterion=='LogOdds':
            alpha = change_needed_in_logodds(p, delta)
            min_delta = delta_p_for_fixed_logoddsincrease(mean_test_scores[d], alpha)
                            
        if pval < pval_thr and mean_effect_size[d]>=min_delta:
            reject=True
        else:
            reject=False
    
        rejects.append(reject)
        min_deltas.append(min_delta)
    
    # recalculate dimensions
    for ix, reject in enumerate(rejects):
        if not reject:
            estimated_dimension = test_dimensions[ix]
            break
    else:
        estimated_dimension = test_dimensions[-1]
    
    return {'pval_corrected': pvals_corrected,
            'estimated_dimension' : estimated_dimension,
            'reject': rejects,
            'min_delta': min_deltas,
            'criterion': criterion}   