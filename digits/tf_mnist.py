import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split, learning_curve
import tensorflow as tf

def main():
    train = pd.read_csv("train.csv")
    y = train.values[:,0]
    X = train.values[:,1:]

    # split the training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333)
    
    print('ready')
    
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)





if __name__ == '__main__':
    main()