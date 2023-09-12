## Torch modules
import torch
from torch.utils.data import Dataset, DataLoader

## Other DS modules 
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

## Visualisation modules
import matplotlib.pyplot as plt
import seaborn as sns

## Other modules
import h5py
from tqdm import tqdm
import os


class EurosatPreloadedDataset(Dataset):
    """
    This class is a subclass of PyTorch's Dataset class, specifically designed for the EuroSAT dataset in HDF5 format.
    The class assumes that the dataset file consists of 'images' and 'labels' datasets, and each image corresponds to a label at the same index.
    The class implements the necessary __len__() and __getitem__() methods for a PyTorch dataset, along with helper methods to load and normalize images. 
    """

    def __init__(self, root_dir, transform=None, classes=None, indices = None) -> None:
      self.root_dir = root_dir
      self.transform = transform
      self.classes = classes
      self.indices = indices
      
      self.images = self.__loadimages__(root_dir)
      self.labels = self.__loadlabels__(root_dir)
 
    def __len__(self):
      return len(self.images)

    def __loadimages__(self, path):
      with h5py.File(path, "r") as f:
        # in case we only want certain images (training/testset)
        if self.indices:
          return f['images'][self.indices]
        else: 
          return list(f['images'])

    def __loadlabels__(self, path):
      with h5py.File(path, "r") as f:
        if self.indices:
          return f['labels'][self.indices]
        else:
          return list(f['labels'])

    def __getitem__(self, idx):
      image = self.images[idx]
      label = self.labels[idx]

      image = self.__normalize__(image).astype(np.float32)
      
      if self.transform is not None:
          image = self.transform(image)
      return (image, label)

    def __normalize__(self, band_data):
      band_data = np.array(band_data)

      # normalize with the 2- and 98-percentiles instead of minimum and maximum of each band
      lower_perc = np.percentile(band_data, 2, axis=(0,1))
      upper_perc = np.percentile(band_data, 98, axis=(0,1))

      band_data = np.clip(band_data, lower_perc, upper_perc)
    
      return (band_data - lower_perc) / (upper_perc - lower_perc)


class EurosatPreloadedTestset(Dataset):
    """
    The EurosatPreloadedTestset class is a subclass of PyTorch's Dataset class, tailored to handle the EuroSAT dataset in HDF5 format for testing.
    This class expects that the dataset file consists of 'images' and 'ids' datasets, where each image corresponds to an id at the same index. 
    Unlike the EurosatPreloadedDataset class, it doesn't deal with labels, as typically labels are not present in a test set.
    """

    def __init__(self, root_dir, transform=None) -> None:
      self.root_dir = root_dir
      self.transform = transform
      
      self.images = self.__loadimages__(root_dir)
      self.ids = self.__loadids__(root_dir)
 
    def __len__(self):
      return len(self.images)

    def __loadimages__(self, path):
      with h5py.File(path, "r") as f:
        return list(f['images'])

    def __loadids__(self, path):
      with h5py.File(path, "r") as f:
        return list(f['ids'])

    def __getitem__(self, idx):
      image = self.images[idx]
      id = self.ids[idx]

      image = self.__normalize__(image).astype(np.float32)
      
      if self.transform is not None:
          image = self.transform(image)
      return (image, id)

    def __normalize__(self, band_data):
      band_data = np.array(band_data)

      # normalize with the 2- and 98-percentiles instead of minimum and maximum of each band
      lower_perc = np.percentile(band_data, 2, axis=(0,1))
      upper_perc = np.percentile(band_data, 98, axis=(0,1))
    
      return (band_data - lower_perc) / (upper_perc - lower_perc)
    
def plot_confusion_matrix(model, valset, class_names, model_name, image_path, device):
    """
    Generates and saves a confusion matrix for the provided model and validation dataset.
    This function iterates through the validation dataset, making predictions using the provided model, and compares those predictions to the true labels. 
    It then generates a confusion matrix and displays it as a heatmap using seaborn.
    """

    predicted_labels = []
    true_labels = []
    model.eval()

    for i in tqdm(range(len(valset.labels))):
        sample, label = valset.__getitem__(i)
        sample = sample.to(device)
        
        with torch.no_grad():
            output = model(sample.unsqueeze(0))
            _, predicted = torch.max(output.data, 1)
        
            true_labels.append(label)
            predicted_labels.append(predicted.cpu().numpy())

            # Generate the confusion matrix

    # print(f'No of true labels {len(true_labels)}, No of predicted labels {len(predicted_labels)}')
    cm = confusion_matrix(true_labels, predicted_labels)

    # Display the confusion matrix using seaborn heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names, fmt='d')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.tight_layout()
    
    plt.savefig(image_path)

# Taken from the Machine Learning lab at University of St.Gallen
def normalize_for_display(band_data):
    """Normalize multi-spectral imagery across bands.
    The input is expected to be in HxWxC format, e.g. 64x64x13.
    To account for outliers (e.g. extremly high values due to
    reflective surfaces), we normalize with the 2- and 98-percentiles
    instead of minimum and maximum of each band.
    """
    band_data = np.array(band_data)
    lower_perc = np.percentile(band_data, 2, axis=(0,1))
    upper_perc = np.percentile(band_data, 98, axis=(0,1))
    
    return (band_data - lower_perc) / (upper_perc - lower_perc)

def plot_histogram_for_rgb(images):
  """
  Plots a histogram for the RGB channels of an image.
  The image must have the bands 0 (red), 1 (green), 2 (blue).
  """
  red_channel = [img[:, :, 0].flatten() for img in images]
  green_channel = [img[:, :, 1].flatten() for img in images]
  blue_channel = [img[:, :, 2].flatten() for img in images]

  ## Plot histograms for each color channel
  fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
  value_min = -1
  value_max = 2

  hist_red, bins_red = np.histogram(red_channel, bins=128, range=(value_min, value_max))  # Adjust the number of bins as needed
  hist_green, bins_green = np.histogram(green_channel, bins=128, range=(value_min, value_max))  # Adjust the number of bins as needed
  hist_blue, bins_blue = np.histogram(blue_channel, bins=128, range=(value_min, value_max))  # Adjust the number of bins as needed

  # Plot the histogram using Matplotlib
  axes[0].bar(bins_red[:-1], hist_red, width=np.diff(bins_red), color='red', alpha=0.7)
  axes[0].set_title('Histogram of Red Channel')
  axes[0].set_xlabel('Pixel Value')
  axes[0].set_ylabel('Frequency')

  axes[1].bar(bins_green[:-1], hist_green, width=np.diff(bins_green), color='green', alpha=0.7)
  axes[1].set_title('Histogram of Green Channel')
  axes[1].set_xlabel('Pixel Value')
  axes[1].set_ylabel('Frequency')

  axes[2].bar(bins_blue[:-1], hist_blue, width=np.diff(bins_blue), color='blue', alpha=0.7)
  axes[2].set_title('Histogram of Blue Channel')
  axes[2].set_xlabel('Pixel Value')
  axes[2].set_ylabel('Frequency')
  plt.tight_layout()
  plt.show()
    


def dataset_first_n(dataset, n, id2label=None, transformed=False):
    # Get unique random indices
    random_idx = np.random.choice(len(dataset), size=n, replace=False)
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 5))

    for i, idx in enumerate(random_idx):
        img = dataset.__getitem__(idx)[0]
        if transformed == True:
          img = img.permute(1, 2, 0)
        
        axs[i].imshow(img)
        
        if id2label:
          classID = dataset.__getitem__(idx)[1]
          label = id2label[classID]
          axs[i].set_title(f"{label}", fontsize=30)

        axs[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_dacnn(model, valset, class_names, model_name, image_path, device):
    """
    Does the same as the plot_confusion_matrix just for the Domain Adaptation-CNN setup
    """

    predicted_labels = []
    true_labels = []
    model.eval()

    for i in tqdm(range(len(valset.labels))):
        sample, label = valset.__getitem__(i)
        sample = sample.to(device)
        
        with torch.no_grad():
            output, _ = model(sample.unsqueeze(0))
            _, predicted = torch.max(output, 1)
        
            true_labels.append(label)
            predicted_labels.append(predicted.cpu().numpy())

            # Generate the confusion matrix

    # print(f'No of true labels {len(true_labels)}, No of predicted labels {len(predicted_labels)}')
    cm = confusion_matrix(true_labels, predicted_labels)

    # Display the confusion matrix using seaborn heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names, fmt='d')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.tight_layout()
    
    plt.savefig(image_path)


def plot_learning_curve(train_losses, val_losses, model_name, formatted_now, visualization_path):
  """
  Generates and saves a learning curve for the model given the training and validation losses.
  This function plots the training and validation losses over the epochs. The resulting plot is saved to the specified path.
  """
  
  # prepare plot
  fig = plt.figure()
  ax1 = fig.add_subplot(111)

  # add grid
  ax1.grid(linestyle='dotted')

  # plot the training epochs vs. the epochs' classification error
  ax1.plot(np.array(range(1, len(train_losses)+1)), train_losses, color='blue', label='train loss (blue)')
  ax1.plot(np.array(range(1, len(val_losses)+1)), val_losses, color='red', label='val loss (red)')

  # add axis legends
  ax1.set_xlabel("[Epoch $e_i$]", fontsize=10)
  ax1.set_ylabel("[Classification Error $\mathcal{L}^{NLL}$]", fontsize=10)

  # set plot legend
  plt.legend(loc="upper right", numpoints=1, fancybox=True)

  # add plot title
  plt.title('Epochs $e_i$ vs. Classification Error $L^{NLL}$', fontsize=10);
  plt.savefig(os.path.join(visualization_path, f'learning_curve_{model_name}_{formatted_now}.jpg'))


def loss_accuracy_curve_dacnn(acc_class, acc_source, acc_target, acc_eval, loss_class, loss_source, loss_target, loss_eval, image_path):
  """
  Generates and displays a line plot for accuracy and loss over epochs, also saves the plot to the specified path.
  The accuracy lines are plotted on the left y-axis, and the loss lines are plotted on the right y-axis.
  The accuracy and loss values are provided as lists where the index in the list represents the epoch.
  """
  
  ## Replacing zeros (skipped epochs) in tracker with previous values
  acc_class = replace_zeros_with_previous(acc_class)
  loss_class = replace_zeros_with_previous(loss_class)

  ## Create a list of epochs
  epoch_list = list(range(1, len(acc_class) + 1))

  ## Create the first line plot for accuracy values and left y-axis
  fig, ax1 = plt.subplots(figsize=(15, 8))
  
  ## Adding all accuracy values
  lns1 = ax1.plot(epoch_list, acc_class, label='Class Acc', color="darkblue")
  lns2 = ax1.plot(epoch_list, acc_source, label='Source Acc', color="darkgreen")
  lns3 = ax1.plot(epoch_list, acc_target, label='Target Acc', color="darkred")
  lns4 = ax1.plot(epoch_list, acc_eval, label='Eval Acc', color="purple")
  
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Accuracy', color='black')
  ax1.tick_params('y', color='black')
  
  plt.xticks(epoch_list)

  ## Create the second line plot for loss values and right y-axis
  ax2 = ax1.twinx()
  
  ## Adding all loss values 
  lns5 = ax2.plot(epoch_list, loss_class, label='Class Loss', color='lightblue')
  lns6 = ax2.plot(epoch_list, loss_source, label='Source Loss', color='lightgreen')
  lns7 = ax2.plot(epoch_list, loss_target, label='Target Loss', color='salmon')
  lns8 = ax2.plot(epoch_list, loss_eval, label='Evaluation Loss', color='lavender')
  
  ax2.set_ylabel('Loss', color='black')
  ax2.tick_params('y', color='black')

  # added these three lines
  lns = lns1+lns2+lns3+lns4+lns5+lns6+lns7+lns8
  labs = [l.get_label() for l in lns]
  ax1.legend(lns, labs, loc='upper right', bbox_to_anchor=(1.2, 1))

  # Add title
  plt.title('Accuracy and Loss per Epoch')
  fig.tight_layout()
  
  # Saving picture to image_path
  plt.savefig(image_path)

  # Show the plot
  plt.show()


def accuracy(prediction, label):
  """
  Functions that takes as input the predictions from the model and ground truth label. 
  Calculates the number of correct predictions
  """
  
  correct_predictions = torch.sum(torch.argmax(prediction, dim=1) == label)
  correct_predictions = correct_predictions.item()
  return correct_predictions


class ForeverDataIterator:
    """
    A data iterator that will never stop producing data
    """

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)


def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.
    """

    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


def replace_zeros_with_previous(array):
  """
  Replaces zero elements in a list with the previous element.
  """

  new_array = []
  
  for k in range(len(array)):
    if array[k] == 0:
        new_array.append(array[k-1])
    else:
      new_array.append(array[k])
    
  return new_array

