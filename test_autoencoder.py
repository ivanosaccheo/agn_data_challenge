import os
import datetime
import configparser
import ast
import shutil
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch.nn as nn
import torch
from torchsummary import summary
import utility_library as ulb
import data_challenge_library as dcl




class autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(autoencoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

def train_routine(dataloader, model, loss_fn, optimizer, verbose = True):
    num_batches = len(dataloader)
    for batch, (features, _ ) in enumerate(dataloader): 
        output = model(features)
        loss = loss_fn(output, features)

        # Backpropagation
        optimizer.zero_grad()    # Clear the gradient
        loss.backward()          # Compute the gradient (??)
        optimizer.step()         # update model weights

        
        if batch == round(num_batches/2):
            train_loss = loss.item()
            if verbose:
                print(f"train loss   :   {loss:>7f}")

    return train_loss

def test_routine(dataloader, model, loss_fn, verbose = True):
    num_batches = len(dataloader)
    test_loss = 0
    model.eval() 
    with torch.no_grad():
        for features, _ in dataloader:
            output = model(features)
            test_loss += loss_fn(output, features).item()

    test_loss /= num_batches
    if verbose:
        print(f"Avg test loss      : {test_loss:>8f}")

    return test_loss


giorno = datetime.date.today()
config_filename = "configuration_files/autoencoder.ini"
save_dir = "trained_models/autoencoder"
save_name = f"autoencoder_{giorno}"
fname = os.path.join(save_dir, save_name)




config = configparser.ConfigParser()
config.read(config_filename)
hidden_sizes = [int(i) for i in ast.literal_eval(config.get("MODEL", "hidden_sizes"))]
latent_size = int(config.get("MODEL", "latent_size"))
batch_size = int(config.get("TRAINING", "batch_size"))
N_epochs = int(config.get("TRAINING", "N_epochs"))
validation_size = float(config.get("SAMPLE", "validation_size"))
test_size = float(config.get("SAMPLE", "test_size"))
shuffle_random_state = int(config.get("SAMPLE", "shuffle_random_state"))
split_random_state_1 = int(config.get("SAMPLE", "split_random_state_1"))
split_random_state_2 = int(config.get("SAMPLE", "split_random_state_2"))




if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

shutil.copy(config_filename, f"{fname}.ini")




object_table = dcl.load_table(have_images = True, features="important")
train, validation, test = dcl.prepare_sample(object_table, 
                                             validation_size=validation_size,
                                             test_size=test_size,
                                             shuffle_random_state=shuffle_random_state,
                                             split_random_state_1=split_random_state_1,
                                             split_random_state_2=split_random_state_2)


train.get_dataloader(batch_size=batch_size)
validation.get_dataloader(batch_size=batch_size)
test.get_dataloader(batch_size=batch_size)



my_encoder = ulb.general_dense(input_size=325, output_size = latent_size, hidden_sizes = hidden_sizes)
my_decoder = ulb.general_dense(input_size = latent_size, output_size = 325, hidden_sizes = hidden_sizes[::-1])

my_autoencoder = autoencoder(my_encoder, my_decoder)

loss_fn = torch.nn.MSELoss()  
optimizer = torch.optim.Adam(my_autoencoder.parameters(), lr=0.001, weight_decay=0.01)

train_loss = []
test_loss  = []
for t in range(N_epochs):
    print(f"Epoch {t+1}/{N_epochs}---------------------")
    train_loss.append(train_routine(train.dataloader, my_autoencoder, loss_fn, optimizer, verbose = False))
    test_loss.append(test_routine(test.dataloader, my_autoencoder, loss_fn, verbose = True))
print("Done!")


torch.save(my_autoencoder.state_dict(), f"{save_name}.pt")

fig, ax = plt.subplots()
ax.plot(np.arange(N_epochs), train_loss, label ='Train')
ax.plot(np.arange(N_epochs), test_loss, label ='Test')
ax.set_xlabel("EPOCH")
ax.set_ylabel("Loss")
ax.legend()
plt.savefig("second_attempt_AE.png", bbox_inches = "tight")