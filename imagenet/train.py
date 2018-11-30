# This is a port of the Pytorch AlexNet training into Tensorflow. 
# I am using the tf.estimator API

import argparse

import tensorflow as tf
import numpy as np


parser = argparse.ArgumentParser(description='Tensorflow ImageNet Training')
parser.add_argument('-dd', '--data_dir', required=True,
                   help='Data directory to load input images')
parser.add_argument('-md', '--model_dir', required=True,
                   help='Model directory to save files related to models')
parser.add_argument('-bs', '--batch_size', default=64,
                   help='Batch size of ImageNet data')
parser.add_argument('--epochs', default=90,
                   help='Number of training epochs')
parser.add_argument('-lr','--learning-rate', default=0.1)

