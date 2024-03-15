import pandas as pd
import numpy as np 
import os
import copy
import torch 
from torch.utils.data import DataLoader,  TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.utils import shuffle



def load_table(features = "savic", have_images = False, remove_unlabeled = True,
               separate_AGN = False, remove_missing_features = True,
               directory = os.path.expanduser("~/DATA/data_challenge")):
    
    """"Features can be savic, important, photometry, variability
    """
               
    table = pd.read_parquet(os.path.join(directory, "ObjectTable.parquet"), engine='fastparquet')
    
    print(f"Original Table has {len(table)} sources")
    
    table["label"] = table["class"].copy()
    
    if separate_AGN:
        table["label"].replace({'Star': 0, 'Gal': 1, 'Qso': 2, 'Agn': 3, 'highZQso': 2}, inplace = True)
    else:
        table["label"].replace({'Star': 0, 'Gal': 1, 'Qso': 2, 'Agn': 2, 'highZQso': 2}, inplace = True)
    
    if remove_unlabeled:
       table = table[~np.isnan(table["label"])]
       print(f"Keeping {len(table)} labeled sources")
    else: 
        table.loc[np.isnan(table["label"]), "class"] = "Unlabeled"
        table.loc[np.isnan(table["label"]), "label"] = np.max(table["label"])+1


    features_fname = features.lower()+"_features.dat"

    input_dir = os.path.expanduser("~/WORK/challenge/input_files")
    try:
        with open(os.path.join(input_dir, features_fname), 'r') as f:
            feature_list = [line.strip("\n") for line in f]
        f.close
        table  = table[feature_list]
        print(f"Keeping only the required {len(feature_list)-2} features")
    except FileNotFoundError:
        print(f"Keeping all features")

    if remove_missing_features:
        table.dropna(axis=0, how="any", inplace=True)
        print(f"Keeping {len(table)} with all features available")

    if have_images:
       images_names = os.listdir(os.path.join(directory, "cutouts"))
       images_names = [i.replace(".npy", "") for i in images_names]
       images_names = [int(i) for i in images_names]  
       objectID = [int(i) for i in  table.index]
       has_image = np.isin(objectID, images_names)
       table = table[has_image]
       print(f"Keeping {len(table)} with available cutouts")

    print(table["class"].value_counts())
    
    return table



def load_images(objectID, image_shape = 64, cropped_shape = 16, as_tensor = False,
               normalize = True, 
               homedir = os.path.expanduser("~/DATA/data_challenge/cutouts")):
    """
    Load images and crop them to have  final shape cropped_shape x cropped_shape
    """
    images = []
    to_crop = int((image_shape-cropped_shape)/2)
    for i, name in enumerate(objectID):
        filename = name+ ".npy"
        image = np.load(os.path.join(homedir, filename))
        image = image[to_crop: image_shape-to_crop, to_crop:image_shape-to_crop, :]
        images.append(image)

    if normalize:
        images = [i/255 for i in images]
        
    if as_tensor:
        image_tensor = torch.empty((len(images), cropped_shape, cropped_shape, 3), dtype=torch.float32)
        for i, image in enumerate(images):
            image_tensor[i,:,:,:] = torch.from_numpy(image)
        
        image_tensor = torch.permute(image_tensor, (0, 3, 1, 2))  #shape taken from pytorch CNN
        
        return image_tensor
    else:
        return images


def transform2tensor(x, y, one_hot = False):
    x_t = torch.tensor(x, dtype = torch.float32)
    y_t = torch.tensor(y, dtype = torch.int64)
    y_t = y_t.type(torch.LongTensor)

    if one_hot:
        y_t = torch.nn.functional.one_hot(y_t)
    return x_t, y_t



def get_bins(array_to_bin,  N_objects = None, N_bins = None, specific_cuts = None):
    """ Binna gli oggetti di array_2_bin. La larghezza dei bin varia in modo che ogni bin abbia lo
        stesso numero di oggetti al suo interno.
        Si può specificare il numero di oggetti per ogni bin (N_objects,in tal caso il numero di bin totali varia)
        oppure il numero totale di bins (N_bins, in tal caso il numero di oggetti per bin varia, 
        ma resta lo stesso per tutti i bin).
        N_objects ha la precedenza rispetto a N_bins.
        Può essere passato anche specific_cuts in modo da avere un ulteriore suddivisione
        in bins. In tal caso N_objects o N_bins devono essere liste con N+1 elementi dove N è il numero
        di elementi in specific_cuts.
        Esempio: specific_cuts = [0.6, 1, 3], N_objects = [100, 200, 300, 400] restituisce dei bin in cui ci
        sono esattemante 100 oggetti per valori <= 0.6, 200 tra 0.6 e 1, 300 tra 1 e 3 e 400 per 
        valori superiori.
     """
    def get_quantiles(array_to_bin, Nbins):
        quantiles_cuts = np.arange(Nbins+1)/Nbins
        quantiles = np.quantile(array_to_bin, quantiles_cuts)
        quantiles[0], quantiles[-1] = np.nextafter(quantiles[0], -np.inf), np.nextafter(quantiles[-1], np.inf)
        return quantiles
        
    if specific_cuts is not None:
        specific_cuts = np.array(specific_cuts).flatten()
        Nbins = 0
        quantiles = []
        bins = np.digitize(array_to_bin, specific_cuts)
        try:
            for unique_bin, nobj in zip(np.unique(bins), N_objects, strict = True):
                assert(isinstance(nobj, int))
                idx = bins == unique_bin
                nbins = int(np.ceil(np.sum(idx) / nobj))
                quantiles.append(get_quantiles(array_to_bin[idx], nbins))
                Nbins += nbins
                print(f"Grouping fixed {nobj} QSOs in {nbins} bins")
            
        except TypeError:
            print("Using fixed number of bins for each specific cut")
            for unique_bin, nbins in zip(np.unique(bins), N_bins, strict = True):
                assert(isinstance(nbins, int))
                idx = bins == unique_bin
                quantiles.append(get_quantiles(array_to_bin[idx], nbins))
                Nbins += nbins
                print(f"Grouping {nobj} QSOs in fixed {nbins} bins")
            
            #L'ultimo elemnto di un quantile è equivalente al primo del successivo 
            final_bins = np.hstack([i[:-1] for i in quantiles])
        
    else:
        try:
            assert isinstance(N_objects, int)
            Nbins = int(np.ceil(len(array_to_bin)/N_objects))
            print(f"Returning {Nbins} bins with {N_objects} objects each")
        except AssertionError:
            assert isinstance(N_bins, int)
            Nbins = N_bins
        final_bins = get_quantiles(array_to_bin, Nbins)
    return final_bins, Nbins



def load_variability_table(which_filter = 0, 
               which_columns = ["objectId", "mjd", "psMag", "psMagErr"],
               objectId = None,
               scale_magnitudes = True, magnitudes_scale_range = (-2, 2),
               scale_time = True, time_scale_range = (-2, 2),
               directory = os.path.expanduser("~/DATA/data_challenge")):
    """
    which_filter: int, 0 = u, 1 = g, 2 = r ....
    objectId = None or list of objectID, only elements in the list are returned
    
    """
    
    variability_table = pd.read_parquet(os.path.join(directory, "ForcedSourceTable.parquet"), engine='fastparquet')
    variability_table = variability_table[variability_table["filter"] == which_filter]
    variability_table = variability_table[which_columns]

    if objectId is not None:
        variability_table = variability_table.merge(objectId, how = 'right', on = "objectId") 
        ## keeping only data with corresponding objectId
    
    tables = []   #list of vatiability tables, each containing data for 1 source
    tables_id = []
    tic = time.perf_counter()
    for id_group, data_group in variability_table.groupby("objectId"):
        if not data_group["psMag"].isnull().all():    #some objects have all nan in lightcurves
                tables.append(data_group.sort_values("mjd"))
                tables_id.append(id_group)
    toc = time.perf_counter()
    print(f"Created variability tables for {len(tables)} sources in {toc-tic} seconds")
    
    if scale_magnitudes and scale_time:
        scaler_4_magnitudes = MinMaxScaler(feature_range = magnitudes_scale_range)
        scaler_4_time = MinMaxScaler(feature_range = time_scale_range)
        for table in tables:
            table["psMag"] = scaler_4_magnitudes.fit_transform(table["psMag"].to_numpy().reshape(-1, 1))
            table["mjd"] = scaler_4_time.fit_transform(table["mjd"].to_numpy().reshape(-1, 1))
    
    elif scale_time: 
        scaler_4_time = MinMaxScaler(feature_range = time_scale_range)
        for table in tables:
            table["mjd"] = scaler_4_time.fit_transform(table["mjd"].to_numpy().reshape(-1, 1))

    elif scale_magnitudes:
        scaler_4_magnitudes = MinMaxScaler(feature_range = magnitudes_scale_range)
        for table in tables:
            table["psMag"] = scaler_4_magnitudes.fit_transform(table["psMag"].to_numpy().reshape(-1, 1))
    
    tic = time.perf_counter()
    print(f"Scaled time and fluxes/magnitudes in in {tic-toc} seconds")
    
    return tables, tables_id
    

class sample_container():
    def __init__(self, X, y, objectID):
        self.X = X
        self.y = y
        self.objectID = objectID
        return None
    
    def get_images(self, images):
        self.images = images
    
    def get_tensors(self, one_hot = False):
        self.X_t, self.y_t = transform2tensor(self.X, self.y, one_hot = one_hot)
        return None

    def get_dataset(self):
        if not hasattr(self, "X_t"):
            self.get_tensors()
        if hasattr(self, 'images'):
            self.dataset = TensorDataset(self.X_t, self.images, self.y_t)
            print("Loading images in the dataset")
        else:
            self.dataset = TensorDataset(self.X_t, self.y_t)
        return None
    
    def get_dataloader(self, batch_size = 50, shuffle = True):
        if not hasattr(self, "dataset"):
            self.get_dataset()
        self.dataloader = DataLoader(self.dataset, batch_size = batch_size, shuffle=shuffle)
        return None
        
def prepare_sample(object_table, test_size = 0.1, validation_size = 0.5, 
                   scaler = StandardScaler(),
                   shuffle_random_state = 2121805, 
                   split_random_state_1 = 14101806,
                   split_random_state_2 = 14061807):
    object_table = shuffle(object_table, random_state=shuffle_random_state)
    
    X = object_table.drop(columns=["class", "label"])
    y = object_table["label"].to_numpy().astype('int32')
    objectID = object_table.index
    
    if test_size <= 0:
        X_train, y_train, objectID_train = X, y, objectID
        X_test, y_test, objectID_test = None, None, None
    else:
        X_train, X_test, y_train, y_test, objectID_train, objectID_test = train_test_split(X, y, 
                                objectID, test_size = test_size, random_state = split_random_state_1)
    
    X_train, X_validation, y_train, y_validation, objectID_train, objectID_validation = train_test_split(X_train, y_train, 
                                                                                    objectID_train, test_size = validation_size,
                                                                                    random_state = split_random_state_2)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_validation = scaler.transform(X_validation)
    
    if test_size > 0:
        X_test = scaler.transform(X_test)
    
    train = sample_container(X_train, y_train, objectID_train)
    validation = sample_container(X_validation, y_validation, objectID_validation)
    test = sample_container(X_test, y_test, objectID_test)
    

    print("Returning train, validation and test objects")

    return train, validation, test







    

