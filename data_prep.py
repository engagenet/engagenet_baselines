"""Data preparation code 

Author - Ximi
License - MIT
"""
import os
import glob
import utils
import config

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import normalize
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler() 




n_segments = config.N_SEGMENTS

def data_loader_fusion(feature_type, val=False, base_dir='data'):
    labels = pd.read_csv(f'{base_dir}/final_labels.csv')
    Xy = np.load(f'{base_dir}/Xy_{feature_type}.npy', allow_pickle=True)
    
#     Xy_openface = Xy[:, 2]
#     Xy_marlin = Xy[:, 1]
    
    features_label_map = {}
    for xy in Xy:  
        features_label_map[xy[0]] = (xy[1], xy[2], xy[3])
    
    train_x_1 = []
    train_x_2 = []
    train_y = []
    if val:
        val_x_1 = []
        val_x_2 = []
        val_y = []
        
    test_x_1 = []
    test_x_2 = []
    test_y = []
    
    trainXy = utils.read_file(f'{base_dir}/train.txt')
    testXy = utils.read_file(f'{base_dir}/test.txt')
    valXy = utils.read_file(f'{base_dir}/valid.txt')
    
    for e in trainXy:
        try:
            xy = features_label_map[e]
            if xy[2] != config.SNP:
                train_x_1.append(xy[0])
                train_x_2.append(xy[1])
                train_y.append(config.LABEL_MAP[xy[2]])
        except KeyError as k:
            pass
    
    X = np.array(train_x_2)
    scaler.fit(X.reshape(-1, X.shape[-1]))
    for i in range(len(train_x_2)):
        train_x_2[i] = scaler.transform(train_x_2[i])
        
    for e in valXy:
        try:
            xy = features_label_map[e]
            if xy[2] != config.SNP:
#                     
                x = xy[1]
                
                x = scaler.transform(xy[1])
                if val:
                    val_x_2.append(x)
                    val_x_1.append(xy[0])
                    val_y.append(config.LABEL_MAP[xy[2]])
                else:
                    train_x_1.append(xy[0])
                    train_x_2.append(x)
                    train_y.append(config.LABEL_MAP[xy[2]])
        except KeyError as k:
            pass
#                 print ('not found(val): ', k)
            
    for e in testXy:
        try:
            xy = features_label_map[e]
            if xy[2] != config.SNP:
                
                x = xy[1]
                
                x = scaler.transform(xy[1])
                
                test_x_1.append(xy[0])
                test_x_2.append(x)
                test_y.append(config.LABEL_MAP[xy[2]])
        except KeyError as k:
            pass
#             print ('not found(test): ', k)
    if val:
        return (
            ((np.array(train_x_1), np.array(train_x_2)), np.array(train_y)), 
            ((np.array(val_x_1), np.array(val_x_2)), np.array(val_y)), 
            ((np.array(test_x_1), np.array(test_x_2)), np.array(test_y))
        )
    else:
        return (
            ((np.array(train_x_1), np.array(train_x_2)), np.array(train_y)), 
            ((np.array(test_x_1), np.array(test_x_2)), np.array(test_y))
            )
def data_loader_v1(feature_type, val=False, scale=True, base_dir='data'):
    
    """Data load without having separate npy files for splits
    """
    labels = pd.read_csv(f'{base_dir}/final_labels.csv')
    Xy = np.load(f'{base_dir}/Xy_{feature_type}.npy', allow_pickle=True)
    Xy = utils.cleanXy(Xy)
    
    
    features_label_map = {}
    for xy in Xy:  
        features_label_map[xy[0]] = (xy[1], xy[2])
        
    train_x = []
    train_y = []
    if val:
        val_x = []
        val_y = []
        
    test_x = []
    test_y = []
    
    trainXy = utils.read_file(f'{base_dir}/train.txt')
    testXy = utils.read_file(f'{base_dir}/test.txt')
    valXy = utils.read_file(f'{base_dir}/valid.txt')
    
    for e in trainXy:
        try:
            xy = features_label_map[e]
            if xy[1] != config.SNP:
                train_x.append(xy[0])
                train_y.append(config.LABEL_MAP[xy[1]])
        except KeyError as k:
            pass
#             print ('not found(train): ', k)
    if scale:    
        X = np.array(train_x)
        scaler.fit(X.reshape(-1, X.shape[-1]))
        for i in range(len(train_x)):
            train_x[i] = scaler.transform(train_x[i]) 
    
    for e in valXy:
        try:
            xy = features_label_map[e]
            if xy[1] != config.SNP:
#                     
                x = xy[0]
                if scale:
                    x = scaler.transform(xy[0])
                if val:
                    val_x.append(x)
                    val_y.append(config.LABEL_MAP[xy[1]])
                else:
                    train_x.append(x)
                    train_y.append(config.LABEL_MAP[xy[1]])
        except KeyError as k:
            pass
#                 print ('not found(val): ', k)
            
    for e in testXy:
        try:
            xy = features_label_map[e]
            if xy[1] != config.SNP:                
                x = xy[0]
                if scale:
                    x = scaler.transform(xy[0])
                test_x.append(x)
                test_y.append(config.LABEL_MAP[xy[1]])
        except KeyError as k:
            pass
#             print ('not found(test): ', k)
    if val:
        return ((np.array(train_x), np.array(train_y)), 
                (np.array(val_x), np.array(val_y)), 
                (np.array(test_x), np.array(test_y)))
    else:
        return ((np.array(train_x), np.array(train_y)), 
                (np.array(test_x), np.array(test_y)))
    



if __name__ == '__main__':

    print ("testing data prep")
    feature_type = config.FUSION
    train, val, test = data_loader_fusion(feature_type, val=True)
    train_x, train_y = train
    train_x1, train_x2 = train_x
    test_x, test_y = test
#     print (len(train_x1))
#     print ("train shape: ", train_x.shape)
    print ('train y shape: ', train_y.shape)
    print ("train 1 shape: ", train_x1.shape)
    print ('train 2 shape: ', train_x2.shape)
#     print ("val shape: ", val_x.shape)
#     print ("test shape: ", test_x.shape)
  
