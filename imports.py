import os
import glob
import pickle
import random
import time
import hashlib

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F

from PIL import Image, ImageDraw
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms, models

from yaml_reader import yaml_reader


config = yaml_reader()

#dirs
train_val_dir = config["dirs"]["train_val_dir"]
test_dir = config["dirs"]["test_dir"]
class_list = os.listdir(train_val_dir)

#training_parameters
base_lr = config["training_parameters"]["base_lr"]
batch_size = config["training_parameters"]["batch_size"] 
num_epochs = config["training_parameters"]["num_epochs"]
random_seed = config["training_parameters"]["random_seed"]

#output_parameters
output_model_dir = config["output_parameters"]["out_model_directory"]
output_inference_dir = config["output_parameters"]["out_inference_directory"]

#init_device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
 