import pandas as pd
import numpy as np 
import os
import torch 
from sklearn.preprocessing import MinMaxScaler



def load_table(savic_features = True, have_images = False, remove_unlabeled = True,
               directory = os.path.expanduser("~/DATA/data_challenge")):
               
    table = pd.read_parquet(os.path.join(directory, "ObjectTable.parquet"), engine='fastparquet')
    
    print(f"table has {len(table)} sources")
    print(table["class"].value_counts())
               
    table["label"] = table["class"]
    table["label"].replace({'Star': 0, 'Gal': 1, 'Qso': 2, 'Agn': 2, 'highZQso': 2}, inplace = True)
    
    if remove_unlabeled:
       table = table[~np.isnan(table["label"])]
       print(f"Keeping {len(table)} labeled sources")
    
    if savic_features:
       with open("input_files/savic_features.dat", 'r') as f:
            feature_list = [line.strip("\n") for line in f]
       f.close
       table  = table[feature_list]
       table.dropna(axis=0, how="any", inplace=True)
       print(f"Keeping {len(table)} with all features used in Savic+23")
    
    if have_images:
       images_names = os.listdir(os.path.join(directory, "cutouts"))
       images_names = [i.replace(".npy", "") for i in images_names]
       images_names = [int(i) for i in images_names]  
       objectID = [int(i) for i in  table.index]
       has_image = np.isin(objectID, images_names)
       table = table[has_image]
       print(f"Keeping {len(table)} with available cutouts")
    
    return table



def load_images(objectID, image_shape = 64, cropped_shape = 16, as_tensor = False,
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
        images.append(image/255)
        
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
        print("Non l'ho ancora implementato")
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
               scale = True, scale_range = (-2, 2),
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
    
    for id_group, data_group in variability_table.groupby("objectId"):
        if not data_group["psMag"].isnull().all():    #some objects have all nan in lightcurves
                tables.append(data_group.sort_values("mjd"))
                tables_id.append(id_group)
    
    print(f"Created variability tables for {len(tables)} sources")
    
    if scale:
        scaler = MinMaxScaler(feature_range = scale_range)
        for table in tables:
            table["mjd"] = table["mjd"] - table["mjd"].iloc[0]
            table["psMag"] = scaler.fit_transform(table["psMag"].to_numpy().reshape(-1, 1))
    print("Scaled time and fluxes/magnitudes")
    
    return tables, tables_id
    