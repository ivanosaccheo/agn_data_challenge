import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn



def ensemble_selector(loss_function, y_hats, y_true, init_size=1,
                      replacement=True, max_iter=100):
    
    """
    see https://towardsdatascience.com/ensembles-the-almost-free-lunch-in-machine-learning-91af7ebe5090
    Implementation of the algorithm of Caruana et al. (2004) 'Ensemble
    Selection from Libraries of Models'. Given a loss function mapping
    predicted and ground truth values to a scalar along with a dictionary of
    models with predicted and ground truth values, constructs an optimal
    ensemble minimizing ensemble loss, by default allowing models to appear
    several times in the ensemble.
    Parameters
    ----------
    loss_function: function
        accepting two arguments - numpy arrays of predictions and true values - 
        and returning a scalar
    y_hats: dict
        with keys being model names and values being numpy arrays of predicted
        values
    y_true: np.array
        numpy array of true values, same for each model
    init_size: int
        number of models in the initial ensemble, picked by the best loss.
        Default is 1
    replacement: bool
        whether the models should be returned back to the pool of models once
        added to the ensemble. Default is True
    max_iter: int
        number of iterations for selection with replacement to perform. Only
        relevant if 'replacement' is True, otherwise iterations continue until
        the dataset is exhausted i.e.
        min(len(y_hats.keys())-init_size, max_iter). Default is 100
    Returns
    -------
    ensemble_loss: pd.Series
        with loss of the ensemble over iterations
    model_weights: pd.DataFrame
        with model names across columns and ensemble selection iterations
        across rows. Each value is the weight of a model in the ensemble
    """
    # Step 1: compute losses
    losses = dict()
    for model, y_hat in y_hats.items():
        losses[model] = loss_function(y_hat, y_true)

    # Get the initial ensemble comprising the best models
    losses = pd.Series(losses).sort_values()
    init_ensemble = losses.iloc[:init_size].index.tolist()

    # Compute its loss
    if init_size == 1:
        # Take the best loss
        init_loss = losses.loc[init_ensemble].values[0]
        y_hat_avg = y_hats[init_ensemble[0]].copy()
    else:
        # Average the predictions over several models
        y_hat_avg = np.array(
            [y_hats[mod] for mod in init_ensemble]).mean(axis=0)
        init_loss = loss_function(y_hat_avg, y_true)

    # Define the set of available models
    if replacement:
        available_models = list(y_hats.keys())
    else:
        available_models = losses.index.difference(init_ensemble).tolist()
        # Redefine maximum number of iterations
        max_iter = min(len(available_models), max_iter)

    # Sift through the available models keeping track of the ensemble loss
    # Redefine variables for the clarity of exposition
    current_loss = init_loss
    current_size = init_size

    loss_progress = [current_loss]
    ensemble_members = [init_ensemble]
    for i in range(max_iter):
        # Compute weights for predictions
        w_current = current_size / (current_size + 1)
        w_new = 1 / (current_size + 1)

        # Try all models one by one
        tmp_losses = dict()
        tmp_y_avg = dict()
        for mod in available_models:
            tmp_y_avg[mod] = w_current * y_hat_avg + w_new * y_hats[mod]
            tmp_losses[mod] = loss_function(tmp_y_avg[mod], y_true)

        # Locate the best trial
        best_model = pd.Series(tmp_losses).sort_values().index[0]

        # Update the loop variables and record progress
        current_loss = tmp_losses[best_model]
        loss_progress.append(current_loss)
        y_hat_avg = tmp_y_avg[best_model]
        current_size += 1
        ensemble_members.append(ensemble_members[-1] + [best_model])

        if not replacement:
            available_models.remove(best_model)

    # Organize the output
    ensemble_loss = pd.Series(loss_progress, name="loss")
    model_weights = pd.DataFrame(index=ensemble_loss.index,
                                 columns=y_hats.keys())
    for ix, row in model_weights.iterrows():
        weights = pd.Series(ensemble_members[ix]).value_counts()
        weights = weights / weights.sum()
        model_weights.loc[ix, weights.index] = weights

    return ensemble_loss, model_weights.fillna(0).astype(float)


def voting_predictions(*probabilities, weights = "equal", vote = "soft", Nclasses=3):

    """"""
    if weights=="equal":
        weights = [1 for i in probabilities]
    else: assert(len(weights) == len(probabilities))

    weights = np.array(weights)/np.sum(weights)

    N_sources = probabilities[0].shape[0]
    
    if vote == "hard":
        predictions = np.zeros((N_sources, len(probabilities)), dtype=int)
        for i, probability in enumerate(probabilities):
            predictions[:,i] = np.argmax(probability, axis = 1)
            Nvotes = np.array([np.sum(predictions==i, axis = 1) for i in range(Nclasses)]).T
            final_predictions =   np.argmax(Nvotes, axis = 1)
    elif vote == "soft":
        weighted_probabilities = np.zeros((N_sources, Nclasses, len(probabilities)))
        for (i,probability)in enumerate(probabilities):
            weighted_probabilities[:,:,i] = weights[i]*probability
        final_predictions = np.argmax(np.sum(weighted_probabilities, axis = -1), axis = 1)        
    return final_predictions



class general_convo2d(nn.Module):
    """
    General class to apply a 2d convolution + a MaxPooling
    """
    def __init__(self, in_channels = 3, out_channels = 5, 
                 conv_kernel = 3, conv_stride = 1, conv_padding = 0,
                 pool_kernel = 1, pool_stride = 1, image_width = 16, 
                 activation_function = nn.LeakyReLU, seed = 24051941):
        super(general_convo2d, self).__init__()
        
        self.seed = seed
        self.activation_function = activation_function
        self.model = nn.Sequential(nn.Conv2d(in_channels= in_channels, out_channels=out_channels,
                                    kernel_size = conv_kernel, stride = conv_stride, padding=conv_padding),
                                    self.activation_function(),
                                    nn.MaxPool2d(kernel_size=pool_kernel, stride = pool_stride))

        self.Nout_conv = np.floor((image_width+ 2*conv_padding - conv_kernel)/conv_stride) +1
        self.Nout_pool = np.floor((self.Nout_conv-pool_kernel)/pool_stride) +1
        self.Nout_tot = int(self.Nout_pool*self.Nout_pool*out_channels)
    
    def forward(self, image):
        image = self.model(image)
        return image

###Define a general Dense Neural network
class general_dense(nn.Module):
    """
    General class to create a NN with arbitrary number of dense layers
    """
    def __init__(self, input_size, output_size,  hidden_sizes =[],
                activation_function = nn.LeakyReLU, seed = 26052013, bias = True):
        super(general_dense, self).__init__()
        N_hidden = len(hidden_sizes) 
        self.activation_function = activation_function

        if N_hidden == 0:
            self.full_sequence = nn.Sequential(nn.Linear(input_size, output_size, bias = bias),
                                               self.activation_function())

        elif N_hidden  == 1:
            assert len(hidden_sizes) == N_hidden
            first_layer_out = hidden_sizes[0]
            self.full_sequence = nn.Sequential(nn.Linear(input_size, hidden_sizes[0], bias = bias),
                                               self.activation_function(),
                                               nn.Linear(hidden_sizes[0],output_size, bias = bias),
                                               self.activation_function())
        else:
            first_layer_out = hidden_sizes[0]
            last_layer_in = hidden_sizes[-1]
        
            first_layer = [nn.Linear(input_size, first_layer_out, bias = bias),
                                self.activation_function()]
        
            hidden_layers =[]
            for N_in, N_out in zip(hidden_sizes[:-1],hidden_sizes[1:]):
                hidden_layers.append(nn.Linear(N_in, N_out, bias = bias))
                hidden_layers.append(self.activation_function())
            
            hidden_layers = hidden_layers

            last_layer = [nn.Linear(last_layer_in, output_size, bias = bias),
                                            self.activation_function()]
            

            self.full_sequence = nn.Sequential(*first_layer, *hidden_layers, *last_layer)
        
        
    def forward(self, x):
        x = self.full_sequence(x)
        return x


