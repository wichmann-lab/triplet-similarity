import numpy as np
import pandas as pd
import json

def calculate_triplet_correct(embedding, triplets):
    
    anchor   = embedding[triplets[:, 0], :]
    positive = embedding[triplets[:, 1], :]
    negative = embedding[triplets[:, 2], :]

    # calculates euclidean distance with np.linalg.norm
    ncorrect = np.sum(np.linalg.norm(anchor - positive, axis=1) < np.linalg.norm(anchor - negative, axis=1))
    score = ncorrect / anchor.shape[0]

    return score


def reads_json_file(fname):    
    with open(fname, 'r') as f:
         contents = json.loads(f.read())
        
    #train_scores = np.asarray(contents['train_scores'])
    test_scores = np.asarray(contents['test_scores'])
    stats_results = contents['stats_result']
    dimensions = contents['dimensions']
    estimated_dimension = contents['estimated_dimension']

    
    return dimensions, test_scores, stats_results, estimated_dimension
    
CRITERION_SUFFIX = {'significance': '', 'LogOdds': '_LogOdds'}


def read_dimensionality(obs, task, image_set, criterion='significance', data_root='..'):
    """Estimated dimensionality of one condition, read from its json.

    The json of the significance criterion is written by the fitting notebooks,
    the `_LogOdds` json by analysis-effectsize-criterion.ipynb. These jsons are
    the source of the reported dimensionalities.
    """
    if criterion not in CRITERION_SUFFIX:
        raise ValueError(f"criterion must be one of {list(CRITERION_SUFFIX)}, got {criterion!r}")
    fname = (f"{data_root}/data/triplet_accuracy_vs_dimensionality-{obs}-{task}-{image_set}"
             f"{CRITERION_SUFFIX[criterion]}.json")
    with open(fname, 'r') as f:
        return int(json.load(f)['estimated_dimension'])


def read_info(obs, task, image_set):
    fname = f"../data/triplet_accuracy_vs_dimensionality-{obs}-{task}-{image_set}.json"
    
    dimensions, test_scores, stats_results, estimated_dimension = reads_json_file(fname)
    
    # test_scores are parsed into a dataframe
    data=[]
    for i, d in enumerate(dimensions):
        td = np.vstack((test_scores[i, :], np.repeat(d, test_scores.shape[1]))).T
        data.append(td)
    triplet_accuracy = pd.DataFrame(np.vstack(data), columns=['triplet accuracy', 'dimension'])
    triplet_accuracy['observer'] = obs
    triplet_accuracy['image_set'] = image_set
    triplet_accuracy['task']= task 

    info = {'estimated_dimension': estimated_dimension,
            'stats_results': stats_results,
            'dimensions': dimensions}
    
    return triplet_accuracy, info, test_scores