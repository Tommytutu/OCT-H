# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:58:54 2024

@author: JC TU
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelBinarizer

def loadData(dataname):
    """
    load training and testing data from different dataset
    """
    if dataname == 'UK':
        x, y = loadUK()
        return x, y
    
    if dataname == 'give':
        x, y = loadgive()
        return x, y
    
    if dataname == 'lending':
        x, y = loadlending()
        return x, y
    
    if dataname == 'fraud':
        x, y = loadfraud()
        return x, y
    
   


def loadUK():
    df=pd.read_csv(r'data\UK_fold_1.csv')
    x=df.drop('BAD',axis=1)
    y=df['BAD']
    x = np.array(x.values)
    y = np.array(y.values)
    return x, y

def loadgive():
    df=pd.read_csv(r'data\give_bin_fold_1.csv')
    x=df.drop('BAD',axis=1)
    y=df['BAD']
    x = np.array(x.values)
    y = np.array(y.values)
    return x, y

def loadlending():
    df=pd.read_csv(r'data\lending_club_fold1_o4.csv')
    x=df.drop('BAD',axis=1)
    y=df['BAD']
    x = np.array(x.values)
    y = np.array(y.values)
    return x, y

def loadfraud():
    df=pd.read_csv(r'data\fraud_fold_1.csv')
    x=df.drop('BAD',axis=1)
    y=df['BAD']
    x = np.array(x.values)
    y = np.array(y.values)
    return x, y