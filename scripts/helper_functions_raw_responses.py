"""
Helper functions to calculate handle raw responses

"""

import pandas as pd
import numpy as np


def get_inter_intra_dataframe_per_observer(df, ref_observer, obs_names, tasks, image_sets):

    # compute pairwise inter-observer reliability for each observer pair:
    # -------------------------------------------------------------------
    data = dict()

    for obs in obs_names:
        if obs == ref_observer: 
            continue
        d = pairwise_agreement(df, ref_observer, obs, tasks, image_sets)
        data[obs] = d


    # make this into a long DataFrame:
    df_1 = pd.concat({k: pd.DataFrame(v) for k, v in data.items()}, axis=0)

    print("Reference observer is: ", ref_observer)

    df_long = df_1.stack(level=0).reset_index(0).rename(columns={"level_0": "observer", 0: "value"})#.melt(id_vars='observer', value_vars=['classic', 'odd one out'],var_name = 'task')
    imgsets = []
    designs = []
    for tup in df_long.index.values:
        (set, design) = tup
        imgsets.append(set)
        designs.append(design)

    df_long = pd.DataFrame(dict([(x, y) for x, y in zip(['Agreement', 'Observer', 'Task', 'ImageSet'], 
                                                        [df_long.value.to_list(), df_long.observer.to_list(), designs, imgsets])]))
    df_long['Agreement_Type'] = np.repeat("Inter", len(df_long))
    #df_long.head()

    # Getting the intra consistency values:
    # -------------------------------------
    data = dict()

    for task in tasks:
        imagesets = dict()
        for image_set in image_sets: 
            r1, r2, r3 = check_runs_pairwise(df, ref_observer, task, image_set)
            imagesets[image_set] = r1, r2, r3
        data[task] = imagesets 

    # make into a long dataframe:
    df_intra = pd.concat({k: pd.DataFrame(v) for k, v in data.items()}, axis=1).melt()
    df_intra.columns = ['Task','ImageSet','Agreement']
    df_intra['Observer'] = np.repeat(ref_observer, len(df_intra))
    df_intra['Agreement_Type'] = np.repeat("Intra", len(df_intra))
    #df_intra.head()

    # combine inter and intra data frames:
    df_all = pd.concat([df_long, df_intra])
    
    # compute agreement gain (reliability exceeding task's chance level)


    df_all["Scaled Agreement"] = np.where(df_all["Task"] == "classic", 
                                         (df_all["Agreement"] - 0.5) / 0.5, 
                                        (df_all["Agreement"] - 0.33) / (1 - 0.33))

    return df_all

def get_response_matrix_all(df, obs_names, task, image_set, rep):

    sorted_all_responses  = []
    missing               = []

    # get missing observations across all observers:
    for obs in obs_names:
                    
        # get the sorted responses for the comparison observer and check for missing triplets
        _, responses, indices_obs = get_observer_responses_per_rep(df, obs, task, image_set, rep=rep)
        sorted_df_obs             = pd.DataFrame({ "Responses": responses, "triplet_index" : indices_obs}).sort_values(by=["triplet_index"])
        missing_triplets_obs      = np.setdiff1d(range(600), indices_obs)

        # combine missing triplets for reference and comparison observer:
        missing = np.hstack([missing, missing_triplets_obs])
    
    # remove missing observations:
    for obs in obs_names:
        _, responses, indices_obs = get_observer_responses_per_rep(df, obs, task, image_set, rep=rep)
        sorted_df_obs             = pd.DataFrame({ "Responses": responses, "triplet_index" : indices_obs}).sort_values(by=["triplet_index"])
        sorted_df_obs = sorted_df_obs[~sorted_df_obs.triplet_index.isin(missing)]

        sorted_all_responses.append(sorted_df_obs.Responses.to_numpy())
    
    return np.vstack(sorted_all_responses)

def get_observer_responses_per_rep(df, obs, task, image_set, rep=0):
    """
    Extracts the responses, triplet embeddings, and triplet indices for a specific observer, task, and image set.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the experimental data.
    obs : str
        The observer identifier to filter the data.
    task : str
        The task identifier to filter the data.
    image_set : str
        The image set identifier to filter the data.
    rep : int
        for which of the three repetition sessions to get the data. 
    

    Returns
    -------
    tuple
        A tuple containing:
        - triplets : numpy.ndarray
            The stacked embeddings of the triplets from both sessions.
        - responses : numpy.ndarray
            The concatenated responses from both sessions.
        - triplet_indices : numpy.ndarray
            The concatenated indices of the triplets from both sessions.
        
    Notes
    -----
    This function filters the DataFrame `df` based on the observer (`obs`), task (`task`), and image set (`image_set`),
    and then extracts the data for two specific triplet sets ("2" and "2"). The data for each session is sorted by session 
    and the first session's responses, embeddings, and indices are concatenated with those of the second session.

    Example
    -------
    triplets, responses, triplet_indices = get_observer_responses_per_rep(df, 'O2', 'classic', 'greebles', rep=0)
    """
    
    # create a task-, observer-, and image-set-specific data frame:
    df_obs = df[(df.observer==obs) & (df.task==task) & (df.image_set==image_set)] 

    # get the session identifier for the 'rep' session for each set of triplets, i.e 1b and 1a or 2a and 2b
    session_1 = df_obs[df_obs.triplet_set == "1"].reset_index().sort_values(by=["session"]).session[rep]
    session_2 = df_obs[df_obs.triplet_set == "2"].reset_index().sort_values(by=["session"]).session[rep]

    responses = np.hstack([df_obs[df_obs.session == session_1].responses.to_list()[0], 
                           df_obs[df_obs.session == session_2].responses.to_list()[0]])
    
    triplets  = np.vstack([df_obs[df_obs.session == session_1].embedding_ready_triplets.to_list()[0], 
                           df_obs[df_obs.session == session_2].embedding_ready_triplets.to_list()[0]])
    
    triplet_indices = np.hstack([df_obs[df_obs.session == session_1].triplet_index.to_list()[0], 
                                 df_obs[df_obs.session == session_2].triplet_index.to_list()[0]])
    
    return triplets, responses, triplet_indices

def create_random_pooling(response_matrix_rest, seed=42):

    # set seed 
    np.random.seed(seed) 

    n_observers, n_observations = response_matrix_rest.shape
    selected_indices            = np.random.choice(n_observers, n_observations)
    selected_values             = response_matrix_rest[selected_indices, np.arange(n_observations)]

    return selected_values

def pairwise_agreement_with_pooled(df, ref_obs, obs_names, tasks, image_sets, seed=42):
    """
    Calculate the pairwise agreement between a reference observer's responses and pooled responses 
    from the other observers, across multiple tasks and image sets.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the response data for each observer, task, and image set.
    
    ref_obs : string
        string containing the observer name of the reference observer, e.g. 'O1' to 'O5'

    obs_names : list of str
        A list of names of all the observers included in the study.

    tasks : list of str
        A list of task names for which pairwise agreement should be calculated.

    image_sets : list of str
        A list of image set names for which pairwise agreement should be calculated.

    seed : int, optional, default=42
        The random seed used for pooling the responses from the remaining observers.

    Returns
    -------
    data : dict
        A dictionary where keys are task names and values are dictionaries mapping image set names 
        to the calculated pairwise agreement percentage for that task and image set.
    
    Notes
    -----
    - The function computes the pairwise agreement for each repetition (3 repetitions per triplet) 
      between the reference observer's responses and the pooled responses of the other observers.
    - The agreement is calculated as the percentage of identical responses between the reference observer 
      and the pooled responses.
    - If all responses are identical, the percentage will be 1.
    """

    # get integers to slice array based on which observer is reference and which are rest:
    ref_num        = int(ref_obs[1])-1
    other_obs_nums = np.setdiff1d(range(5), ref_num)

    # initialise dict for all data 
    data = dict()
        
    for task in tasks:

        img = dict()
            
        for image_set in image_sets:

            id_responses_all = []
            n_reps           = 3  # there are 3 repetitions for each triplet

            for rep in range(n_reps):

                # get response data for this repetition:
                response_matrix      = get_response_matrix_all(df, obs_names, task, image_set, rep)
                ref_data             = response_matrix[ref_num]
                response_matrix_rest = response_matrix[other_obs_nums]

                # get data pooled from the remaining 4 observers:
                pooled_data = create_random_pooling(response_matrix_rest, seed=seed)

                # compute identical responses
                id_responses = [(x == y) for x, y in zip(ref_data, pooled_data)]
                # add to list with all repetition sessions:
                #id_responses_all = id_responses_all + id_responses

                _, counts    = np.unique(id_responses, return_counts=True)

                # calculate percentage:
                if counts.shape[0] == 2:
                    perc = counts[1] / len(id_responses)
                else:  # else all are identical 
                    perc = 1
                
                id_responses_all.append(perc)
                    
            img[image_set] = id_responses_all
            
        data[task] = img
        
    return data 

def pairwise_agreement(df, ref_observer, obs, tasks, image_sets):

    data = dict()
    
    for task in tasks:

        img = dict()
        
        for image_set in image_sets:

            id_responses_all = []
            n_reps           = 3  

            for rep in range(n_reps):
            
                # get the sorted responses for the reference observer and check for missing triplets
                _, responses, indices_ref = get_observer_responses_per_rep(df, ref_observer, task, image_set, rep=rep)
                sorted_df_ref             = pd.DataFrame({ "Responses": responses, "triplet_index" : indices_ref}).sort_values(by=["triplet_index"])
                missing_triplets_ref      = np.setdiff1d(range(600), indices_ref)
                
                # get the sorted responses for the comparison observer and check for missing triplets
                _, responses, indices_obs= get_observer_responses_per_rep(df, obs, task, image_set, rep=rep)
                sorted_df_obs            =pd.DataFrame({ "Responses": responses, "triplet_index" : indices_obs}).sort_values(by=["triplet_index"])
                missing_triplets_obs     = np.setdiff1d(range(600), indices_obs)

                # combine missing triplets for reference and comparison observer:
                missing = np.hstack([missing_triplets_ref, missing_triplets_obs])
                
                # drop values missing in both
                sorted_df_ref = sorted_df_ref[~sorted_df_ref.triplet_index.isin(missing)]
                sorted_df_obs = sorted_df_obs[~sorted_df_obs.triplet_index.isin(missing)]
                
                # compute identical responses in this repetition:
                id_responses = [(x == y) for x, y in zip(sorted_df_ref.Responses.to_numpy(),
                                                        sorted_df_obs.Responses.to_numpy())]

                # add to list with all repetition sessions:
                id_responses_all = id_responses_all + id_responses

            _, counts    = np.unique(id_responses_all, return_counts=True)

            if counts.shape[0] == 2:
                perc = counts[1] / len(id_responses_all)
            else:  # else all are identical 
                perc = 1

            img[image_set] = perc
            
        data[task] = img
        
    return data 

def check_runs_pairwise(triplet_df, observer, task, image_set):
    '''
    Returns the consistency of responses (as percentage of identical responses) pairwise for each repetition 
    

    Params:
    - triplet_df: pd Dataframe with all the data 
    - observer:   string containing the identifier of the observer, e.g. "O2"
    - task:       string indicating the task, "classic" or "odd one out"
    - image_set:  string indicating the image set, "action", "taxonomic" or "greebles"
    '''
    
    perc_correct_1 = []
    perc_correct_2 = []
    perc_correct_3 = []
    
    for set in ["1", "2"]:

        # split data frame according to observer, image set, task and set of images (0-299 or 300-299):
        df_split = triplet_df[(triplet_df.observer==observer) & (triplet_df.task==task) & 
                              (triplet_df.triplet_set==set) & (triplet_df.image_set==image_set)]
        df_split.reset_index(inplace=True)

        # sort data frames by trial index for each of the 3 repetitions:
        df_split_0 = pd.DataFrame({"Triplet_index": df_split.triplet_index[0],
                                   "Response": df_split.responses[0]}).sort_values(by=["Triplet_index"])
        df_split_1 = pd.DataFrame({"Triplet_index": df_split.triplet_index[1],
                                   "Response": df_split.responses[1]}).sort_values(by=["Triplet_index"])

        if observer == "O5": # observer 5 is identical to P1 => the first session of O5 is P1's first sessions
            df_split_2 = triplet_df[(triplet_df.observer=="P1") & (triplet_df.task==task) & 
                                    (triplet_df.triplet_set==set) & (triplet_df.image_set==image_set)]
            df_split_2.reset_index(inplace=True)
            df_split_2 = pd.DataFrame({"Triplet_index": df_split_2.triplet_index[0],
                                      "Response": df_split_2.responses[0]}).sort_values(by=["Triplet_index"])

        else:    
            df_split_2 = pd.DataFrame({"Triplet_index": df_split.triplet_index[2],
                                      "Response": df_split.responses[2]}).sort_values(by=["Triplet_index"])

        # check that responses exist in all three
        df_split_0, df_split_1, df_split_2 = clean_triplet_indices(df_split_0, df_split_1, df_split_2, set)
        
    
            
        id_responses_1 = [(x == y) for x, y in zip(df_split_0.Response.to_numpy(),
                                                   df_split_1.Response.to_numpy())]
        id_responses_2 = [(y == z) for y, z in zip(df_split_1.Response.to_numpy(), 
                                                   df_split_2.Response.to_numpy())]
        id_responses_3 = [(x == z) for x, z in zip(df_split_0.Response.to_numpy(),
                                                   df_split_2.Response.to_numpy())]
            
        _, counts_1     = np.unique(id_responses_1, return_counts=True)
        _, counts_2     = np.unique(id_responses_2, return_counts=True)
        _, counts_3     = np.unique(id_responses_3, return_counts=True)
        
        # if the shape is 2: take the second entry corresponds to the nr of identical responses
        if counts_1.shape[0] == 2:
            perc_correct_1.append(counts_1[1] / len(id_responses_1))
        else:  # else all are identical 
            perc_correct_1.append(1)
        if counts_2.shape[0] == 2:
            perc_correct_2.append(counts_2[1] / len(id_responses_2))
        else:
            perc_correct_2.append(1)
        if counts_3.shape[0] == 2:
            perc_correct_3.append(counts_3[1] / len(id_responses_3))
        else:
            perc_correct_3.append(1)

    print(perc_correct_1, perc_correct_2, perc_correct_3)
    return np.mean(perc_correct_1), np.mean(perc_correct_2), np.mean(perc_correct_3)

def clean_triplet_indices(df_split_0, df_split_1, df_split_2, set):
    '''
    This function drops triplets which do not occur in all 3 repetitions

    Params: 
    - set:        whether 1st or 2nd set of triplets 
    - df_splits:  pd dataframes containing the 3 repetitions of identical triplets

    Returns: 
    - new dataframes containing only triplets present in each repetition  
    '''
    
    if set == "1": 
        range = np.arange(300)
    if set == "2":
        range = np.arange(300, 600)

    # check whether all the values from range are contained in our triplet indices:
    missing_0 = np.setdiff1d(range, df_split_0.Triplet_index.to_numpy())
    missing_1 = np.setdiff1d(range, df_split_1.Triplet_index.to_numpy())
    missing_2 = np.setdiff1d(range, df_split_2.Triplet_index.to_numpy())

    missing = np.hstack([np.hstack([missing_0, missing_1]), missing_2])
    
    #print("From repetition 0 the following triplets miss responses: ", missing_0)
    #print("From repetition 1 the following triplets miss responses: ", missing_1)
    #print("From repetition 2 the following triplets miss responses: ", missing_2)

    #print(len(df_split_0), len(df_split_1), len(df_split_2))
    
    # drop values missing in repetition x from repetitions y & z:
    df_split_0 = df_split_0[~df_split_0.Triplet_index.isin(missing)]
    df_split_1 = df_split_1[~df_split_1.Triplet_index.isin(missing)]
    df_split_2 = df_split_2[~df_split_2.Triplet_index.isin(missing)]

    #print(len(df_split_0), len(df_split_1), len(df_split_2))

    return df_split_0, df_split_1, df_split_2

def get_observer_responses_per_rep(df, obs, task, image_set, rep=0):
    """
    Extracts the responses, triplet embeddings, and triplet indices for a specific observer, task, and image set.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the experimental data.
    obs : str
        The observer identifier to filter the data.
    task : str
        The task identifier to filter the data.
    image_set : str
        The image set identifier to filter the data.
    rep : int
        for which of the three repetition sessions to get the data. 
    

    Returns
    -------
    tuple
        A tuple containing:
        - triplets : numpy.ndarray
            The stacked embeddings of the triplets from both sessions.
        - responses : numpy.ndarray
            The concatenated responses from both sessions.
        - triplet_indices : numpy.ndarray
            The concatenated indices of the triplets from both sessions.
        
    Notes
    -----
    This function filters the DataFrame `df` based on the observer (`obs`), task (`task`), and image set (`image_set`),
    and then extracts the data for two specific triplet sets ("2" and "2"). The data for each session is sorted by session 
    and the first session's responses, embeddings, and indices are concatenated with those of the second session.

    Example
    -------
    triplets, responses, triplet_indices = get_observer_responses_per_rep(df, 'O2', 'classic', 'greebles', rep=0)
    """
    
    # create a task-, observer-, and image-set-specific data frame:
    df_obs = df[(df.observer==obs) & (df.task==task) & (df.image_set==image_set)] 

    # get the session identifier for the 'rep' session for each set of triplets, i.e 1b and 1a or 2a and 2b
    session_1 = df_obs[df_obs.triplet_set == "1"].reset_index().sort_values(by=["session"]).session[rep]
    session_2 = df_obs[df_obs.triplet_set == "2"].reset_index().sort_values(by=["session"]).session[rep]

    responses = np.hstack([df_obs[df_obs.session == session_1].responses.to_list()[0], 
                           df_obs[df_obs.session == session_2].responses.to_list()[0]])
    
    triplets  = np.vstack([df_obs[df_obs.session == session_1].embedding_ready_triplets.to_list()[0], 
                           df_obs[df_obs.session == session_2].embedding_ready_triplets.to_list()[0]])
    
    triplet_indices = np.hstack([df_obs[df_obs.session == session_1].triplet_index.to_list()[0], 
                                 df_obs[df_obs.session == session_2].triplet_index.to_list()[0]])
    
    return triplets, responses, triplet_indices

