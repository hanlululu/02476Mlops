import torch
import numpy as np 
import click 

@click.group()
def cli():
    pass

@click.command()
@click.option("--training", default=True, help='if accessing training data')
def mnist(training):
    # exchange with the corrupted mnist dataset
    path = "/Users/hanluhe/Documents/MLops/dtu_mlops/data/corruptmnist/"
    if training:
        ## load all training datasets
        train_images = []
        train_labels = []

        for i in range(0,5):
            with np.load(path+'train_' +str(i)+'.npz') as f:
                images_train, labels_train = f['images'], f['labels']
                images_train_scaled = np.array([img/255. for img in images_train])
                train_images.append(torch.from_numpy(images_train_scaled))
                train_labels.append(torch.from_numpy(labels_train))
        print(train_images)
        return torch.cat(train_images,dim=0), train_labels 
        
    else:
        ## load test dataset
        test = np.load(path + "test.npz")

        images_test, labels_test = torch.from_numpy(test['images']), torch.from_numpy(test['labels'])
        test_images = torch.tensor([img/255. for img in images_test])

        return test_images, labels_test

cli.add_command(mnist)

if __name__ == "__main__":
    cli()