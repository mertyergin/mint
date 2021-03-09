import numpy as np
from sklearn.metrics import confusion_matrix
import nibabel as nib
import imageio
from IPython.display import Image
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns
import os

def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    return (img - mi) / (ma - mi)

    return img

def window_image(img, window_center, window_width, intercept, slope, normalize=False):
    img = (img*slope + intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img < img_min] = img_min
    img[img > img_max] = img_max

    if normalize:
        return normalize_minmax(img)

    return img

def visualize_sample_data_gif():
    nii_path = 'data/volume-1.nii'
    gif_path = 'data/gif.gif'

    if not os.path.exists(gif_path):
        data = np.array(nib.load(nii_path).get_data())
        data = window_image(data, -500, 1500, 0, 1, normalize=True)
        data = (data * 255).astype("uint8")
        data = np.rot90(data)

        images = []
        for i in range(data.shape[2]):
            images.append(data[:,:,i])        
        imageio.mimsave(gif_path, images, duration=0.001)
    return Image(filename=gif_path, format='png')

def plot_cm(y_true, y_pred, class_names, figsize=(10,10)):
    
    y_true = [class_names[str(k)] for k in np.squeeze(y_true)]
    y_pred = [class_names[str(k)] for k in np.squeeze(y_pred)]
    labels = [class_names[key] for key in sorted(class_names.keys(), reverse=False)]
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))

    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap= "BuGn", annot=annot, fmt='', ax=ax, cbar=False)

def show_image(img, label, class_names):
    plt.title((class_names[str(label)]))
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    
def show_random_images(images, labels, class_names, n=3, preds = None):
    sample_indexes = random.sample(list(range(0,len(images))), n*n)
    sample_indexes = np.array(sample_indexes).reshape(n,n)
    
    fig, axes = plt.subplots(n,n, figsize=(12,12))
    
    for i in range(n):
        for j in range(n):
            sample_idx = sample_indexes[i,j]
            sample_img = images[sample_idx]
            x_center = sample_img.shape[0] // 2
            
            
            axes[i,j].imshow(sample_img, cmap="gray")
            axes[i,j].axis('off')
            
            class_name = class_names[str(labels[sample_idx][0])]
            axes[i,j].set_title(class_name)
            if preds is not None:
                pred_name = class_names[str(preds[sample_idx][0])]
                if pred_name == class_name:
                    color = "green"
                else:
                    color = "red"

                axes[i,j].text(x_center, 4, pred_name, ha="center", va="center", size="large",color=color, family="monospace", weight="extra bold")

def to_one_hot(y, num_classes):
    return tf.keras.utils.to_categorical(y,num_classes=num_classes)

def show_train_history(history):
    fig, axes = plt.subplots(1,2, figsize=(12,6))

    axes[0].plot(history.history['accuracy'])
    axes[0].plot(history.history['val_accuracy'])
    axes[0].set_title('model accuracy')
    axes[0].set_ylabel('accuracy')
    axes[0].set_xlabel('epoch')
    axes[0].legend(['train', 'val'], loc='upper left')

    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'])
    axes[1].set_title('model loss')
    axes[1].set_ylabel('loss')
    axes[1].set_xlabel('epoch')
    axes[1].legend(['train', 'val'], loc='upper left')
    plt.show()