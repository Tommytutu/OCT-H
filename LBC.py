
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:41:09 2024

@author: JC TU
"""


import numpy as np 
import pandas as pd
import gurobipy as gp
from gurobipy import*
from gurobipy import GRB

import time
from scipy import stats


from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer


class LBC:
    def __init__(self,  min_samples_split=1, alpha=0.01,N=5, warmstart=True,objective = 'F1-score', CFN =1, timelimit=60, output=True):
        self.min_samples_split = min_samples_split
        self.alpha = alpha
        self.N = N
        self.warmstart = warmstart
        self.timelimit = timelimit
        self.CFN = CFN
        self.objective =objective
        self.output = output
        self.trained = False
        self.optgap = None
    
    
    def fit(self, x, y):
        
        xy= np.column_stack((x,y))
        
        n, op = x.shape


        len_y=len(y)
        
        sum_y = sum(y)
        
      
      
        xy= np.column_stack((x,y))
        
        unique_rows, counts = np.unique(xy, axis=0, return_counts=True)
        
        
        y =unique_rows[:, -1]

        x = unique_rows[:, :-1]
        
        n, p = x.shape
        if self.output:
            print('Training data include {} instances, {} features.'.format(len_y,p))
            print('unique number of instances:', len(unique_rows), 'number of reduced instances:',len_y-len(unique_rows))

        
       

        n, p=x.shape
        cost = len_y/ sum_y
        weights = {0:1, 1:cost}
        
        a_start=np.zeros(p)
        a_start=np.zeros(p)
        
        clf = DecisionTreeClassifier(max_depth=1,class_weight=weights)
        clf.fit(x,y)
        
        decision_rules = clf.tree_

      
        af= decision_rules.feature[0]
            

      
        leaf_labels = {}

        def get_leaf_labels(node, parent_index=0):
            if node.children_left[parent_index] == -1:
                leaf_labels[parent_index] = clf.classes_[np.argmax(node.value[parent_index])]
            else:
                get_leaf_labels(node, node.children_left[parent_index])
                get_leaf_labels(node, node.children_right[parent_index])

        get_leaf_labels(clf.tree_)
        
        if leaf_labels[1]==1:
            a_start[af] =-1
           
        else:
            a_start[af] =1
           
        # create a model
        m = gp.Model('m')

        # output

        # time limit
        
       
        m.Params.timelimit = self.timelimit
        
        m.params.DegenMoves=0
        
        m.params.ImproveStartTime=0.8*self.timelimit


      
        m.modelSense = GRB.MINIMIZE

       
        a = m.addVars(p,  lb=-1, ub=1, vtype=GRB.CONTINUOUS, name='a') # splitting feature
        bara = m.addVars(p, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='bara') # splitting feature
        s = m.addVars(p,  vtype=GRB.BINARY, name='s') # splitting feature
        b = m.addVar(lb=-1, ub=1, vtype=GRB.CONTINUOUS, name='b') # splitting threshold
        y_pre = m.addVars(n, vtype=GRB.BINARY, name='y_pre') # leaf node assignment
        f1 = m.addVar(lb=0,ub=1, vtype=GRB.CONTINUOUS, name='f1') # leaf node samples
        
        e = 0.005
        

      
        
        
        TP=gp.quicksum((y[i] == 1) * y_pre[i] * counts[i] for i in range(n))
        TN=gp.quicksum((y[i] == 0) * (1-y_pre[i]) * counts[i] for i in range(n))
        FP=gp.quicksum((y[i] == 0) * (y_pre[i])  * counts[i] for i in range(n))
        FN=gp.quicksum((y[i] == 1) * (1-y_pre[i]) * counts[i] for i in range(n))
        
        if self.objective == 'F1-score':
            obj = -f1 + self.alpha * gp.quicksum(s[j] for j in range(p))
            m.setObjective(obj)
            
        elif self.objective == 'acc':
            obj = (FP+FN)/sum(counts)
            m.setObjective(obj)
        
        elif self.objective == 'cost-sensitive':
            obj = (FP+ self.CFN*FN)/sum(counts)
            m.setObjective(obj)
            
        
        
        
        
        m.addConstr(bara.sum()<= 1)


        m.addConstrs(bara[j] >= a[j]  for j  in range(p))

        m.addConstrs(bara[j] >= -a[j]  for j  in range(p))

        m.addConstrs(s[j] >=a[j] for j  in range(p))

        m.addConstrs(-s[j] <=a[j]  for j  in range(p))

        m.addConstr(s.sum() >=1)

      
        m.addConstr(s.sum('*')<=self.N)


        m.addConstrs(gp.quicksum(a[j] * x[i,j] for j in range(p)) >=b - 2*(1 - y_pre[i]) for i in range(n))

                
        m.addConstrs(gp.quicksum(a[j] * x[i,j] for j in range(p)) +e<= b+ 2*y_pre[i] for i in range(n))
        
        # F1_score
        m.addConstr(f1*(2*TP+FP+FN)<=2*TP)

                       

     
        
        

        for j in range(p):
            a[j].start=a_start[j]
           

        b.start = 0

        m.optimize()
        
        


        self.a = {ind:a[ind].x for ind in a}
        self.ap = {ind:a[ind].x for ind in a}
        self.y_pre = {ind:y_pre[ind].x for ind in y_pre}
        self.b = b.x
        self.optgap = m.MIPGap
        self.optgap = m.MIPGap
        self.l_index = [1,2]
        self.top_indices = top_indices
        
        self.a = {i: 0 for i in range(op)}
        
        
        for i in range(len(top_indices)):
            self.a[top_indices[i]] = self.ap[i]
            
        
        

        
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        a_array = np.array([self.a[i] for i in sorted(self.a.keys())])
        dot_products = np.dot(X, a_array)
        y_pred = (dot_products >= self.b).astype(int)
        
        return y_pred
    
    def predict_proba(self, X):
        X = np.array(X)
        a_array = np.array([self.a[i] for i in sorted(self.a.keys())])
        dot_products = np.dot(X, a_array)
        y_pred = (dot_products - self.b)
        
        return y_pred
    
    def predictleaf(self, X):
        X = np.array(X)
        a_array = np.array([self.a[i] for i in sorted(self.a.keys())])
        dot_products = np.dot(X, a_array)
        leaf = (dot_products >= self.b).astype(int) +1
        
        return leaf
        
        
        

    
       
    

    