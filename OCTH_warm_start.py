# -*- coding: utf-8 -*-
"""
Created on Fri May 10 20:00:44 2024

@author: JC TU
"""


from collections import namedtuple
import numpy as np
from scipy import stats
import gurobipy as gp
from gurobipy import GRB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

class OCTH_warm_start:
    """
    optimal classification tree
    """
    
    
    def __init__(self, max_depth=2, min_samples_split=1, alpha=0.01,N=5,timeout=600, warmstart=True, objective ='F1-score', CFN= 1, ax={}, bx={}, timelimit=600, output=True):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.alpha = alpha
        self.N = N
        self.timeout=timeout
        self.warmstart = warmstart
        self.timelimit = timelimit
        self.output = output
        self.trained = False
        self.ax=ax
        self.bx=bx
        self.optgap = None
        self.objective = objective
        self.CFN =CFN

        # node index
        self.n_index = [i+1 for i in range(2 ** (self.max_depth + 1) - 1)]
        self.b_index = self.n_index[:-2**self.max_depth] # branch nodes
        self.l_index = self.n_index[-2**self.max_depth:] # leaf nodes

    def fit(self, x, y):
        """
        fit training data
        """
        xy= np.column_stack((x,y))
        
        sumy = sum(y)


        len_y=len(y)
        
        unique_rows, counts = np.unique(xy, axis=0, return_counts=True)
        
        
        
        
        y =unique_rows[:, -1]

        x = unique_rows[:, :-1]
        
        self.n, self.p = x.shape
        if self.output:
            print('Training data include {} instances, {} features.'.format(len_y,self.p))
            print('unique number of instances:', len(unique_rows), 'number of reduced instances:',len_y-len(unique_rows))



      
        # labels
        self.labels = np.unique(y)

        # scale data
        self.scales = np.max(x, axis=0)
        self.scales[self.scales == 0] = 1

        # solve MIP
        # m, a, b, c, d, l = self._buildMIP(x/self.scales, y)
        m, a, b, c, d, l = self._buildMIP(x, y, counts, sumy)
        if self.warmstart:
            self._setStart(x, y, a, b)
            
        m.optimize()
        
        self.optgap = m.MIPGap
        self.ObjVal = m.ObjVal

        # get parameters
        self._a = {ind:a[ind].x for ind in a}
        self._b = {ind:b[ind].x for ind in b}
        self._c = {ind:c[ind].x for ind in c}
        self._d = {ind:d[ind].x for ind in d}

        self.trained = True
        self.a =self._a
        self.b =self._b
        
        return self
    
    
        

    def predictleaf(self, x):
        """
        model prediction
        """
        if not self.trained:
            raise AssertionError('This optimalDecisionTreeClassifier instance is not fitted yet.')
        
        def getleaf(tb):
            t=tb
            if tb % 2 == 1:
            
                t=2*t+1
            else:
              
                    t=2*t
            
            return t

        # leaf label
        labelmap = {}
        for t in self.l_index:
            for k in self.labels:
                if self._c[k,t] >= 1e-2:
                    labelmap[t] = k

    
        leaf =[]
        for xi in x:
            t = 1
            while t not in self.l_index:
                right = (sum([self._a[j,t] * xi[j] for j in range(self.p)]) + 1e-9 >= self._b[t])
                if self._d[t] == 1:
                    if right:
                       t = 2 * t + 1
                       
                    else:
                         t = 2 * t
                     
                else:
                       t = getleaf(t)
            
            # label
            leaf.append(t)
           
        return np.array(leaf)
    
    
    def predict(self, x):
        """
        model prediction
        """
        if not self.trained:
            raise AssertionError('This optimalDecisionTreeClassifier instance is not fitted yet.')
        
        def getleaf(tb):
            t=tb
            if tb % 2 == 1:
            
                t=2*t+1
            else:
              
                    t=2*t
            
            return t

        # leaf label
        labelmap = {}
        for t in self.l_index:
            for k in self.labels:
                if self._c[k,t] >= 1e-2:
                    labelmap[t] = k

        y_pred = []
        for xi in x:
            t = 1
            while t not in self.l_index:
                right = (sum([self._a[j,t] * xi[j] for j in range(self.p)]) + 1e-9 >= self._b[t])
                if self._d[t] == 1:
                    if right:
                       t = 2 * t + 1
                       
                    else:
                         t = 2 * t
                     
                else:
                       t = getleaf(t)
            
          
            y_pred.append(labelmap[t])

        return np.array(y_pred)

    def _buildMIP(self, x, y,counts,sumy):
        """
        build MIP formulation for Optimal Decision Tree
        """
        def get_left_childrenleaf(s, max_depth):
            ss=s
            d1=0
            while s != 0:
                d1=d1+1
                s = s // 2
            
            l_min= 2*ss*2**(max_depth-d1)
            l_max=l_min+2**(max_depth-d1)
            
            left_childrenleaf=self.n_index[l_min-1: l_max-1]
            return left_childrenleaf
            

        def get_right_childrenleaf(s, max_depth):
            ss=s
            d1=0
            while s != 0:
                d1=d1+1
                s = s // 2
            
            l_max=ss
            d=d1
            while d<=max_depth:
                l_max=2*l_max+1
                d=d+1
            
            
            l_min= l_max  - 2**(max_depth-d1)
            
            
            right_childrenleaf=self.n_index[l_min: l_max]
            return right_childrenleaf
        
        def get_l(t):
            lls=[]
            lrs=[]
            left=(t % 2 == 0)
            right=(t % 2 == 1)
            
            if t>=2 and left:
                while (t % 2 == 0):
                     lls.append(t)
                     t=t//2
                lls.append(t)
                lls.pop(0)
            
            if t>=3 and right:
                while (t % 2 == 1) and (t>=3):
                     lrs.append(t)
                     t=t//2
                lrs.append(t)
                lrs.pop(0)
            
            if left:
                return lls
            else:
                return lrs


        # create a model
        m = gp.Model('m')

        # output
        m.Params.outputFlag = self.output
        m.Params.LogToConsole = self.output
        # time limit
        m.Params.timelimit = self.timelimit
        
        m.params.Heuristics=0.1
        
        #m.params.NonConvex=2
        

        # model sense
        m.modelSense = GRB.MINIMIZE

        # variables
        a = m.addVars(self.p, self.b_index, lb=-1, ub=1, vtype=GRB.CONTINUOUS, name='a') # splitting feature
        b = m.addVars(self.b_index, lb=-1, ub=1, vtype=GRB.CONTINUOUS, name='b') # splitting threshold
        bara = m.addVars(self.p, self.b_index, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='bara') # splitting feature
        s = m.addVars(self.p, self.b_index, vtype=GRB.BINARY, name='s') # splitting feature
        
        c = m.addVars(self.labels, self.l_index, vtype=GRB.BINARY, name='c') # node prediction
        d = m.addVars(self.b_index, vtype=GRB.BINARY, name='d') # splitting option
        l = m.addVars(self.l_index, vtype=GRB.BINARY, name='l') # leaf node activation
        
       
        z = m.addVars(self.n, self.l_index, vtype=GRB.BINARY, name='z') # leaf node assignment
       
        
        f1 = m.addVar(lb=0,ub=1, vtype=GRB.CONTINUOUS, name='f1') # leaf node samples
        
        TP = gp.quicksum(z[i,t] * (y[i]==1) * counts[i] for t in self.l_index for i in range(self.n))
        FN = sumy - TP
        
        TN = gp.quicksum(z[i,t] * (y[i]==0) * counts[i] for t in self.l_index for i in range(self.n))
        
        FP = sum(counts) - sumy - TN
       
        #(counts.sum() -gp.quicksum(counts[i]* z[i,t] for i in range(self.n) for t in self.l_index))/counts.sum() 
        # objective function
        if self.objective == 'F1-score':
            error=-f1 
            obj = error + self.alpha * gp.quicksum(s[j,t] for j in range(self.p) for t in self.b_index)
            m.setObjective(obj)
            
        elif self.objective == 'acc':
            obj = (FP+FN)/sum(counts)
            m.setObjective(obj)
        
        elif self.objective == 'cost-sensitive':
            obj = (FP+ self.CFN*FN)/sum(counts)
            m.setObjective(obj)
            
        
        
        
        mu = 0.005
        
        m.addConstr(d[1]==1)
        

        m.addConstr(l[2**self.max_depth] == 1)

        m.addConstr(l[2**(self.max_depth +1)-1] == 1)
        
       

        m.addConstrs(bara.sum('*', t) <= d[t] for t in self.b_index)
        
       # m.addConstrs(bara[j,t] >= mu + 0.001 for t in self.b_index for j  in range(self.p))

        m.addConstrs(bara[j,t] >= a[j,t] for t in self.b_index for j  in range(self.p))

        m.addConstrs(bara[j,t] >= -a[j,t] for t in self.b_index for j  in range(self.p))

        m.addConstrs(s[j,t] >=a[j,t] for t in self.b_index for j  in range(self.p))

        m.addConstrs(-s[j,t] <=a[j,t] for t in self.b_index for j  in range(self.p))

        m.addConstrs(s[j,t] <=d[t] for t in self.b_index for j  in range(self.p))

        m.addConstrs(s.sum('*',t) >=d[t] for t in self.b_index)
        

        m.addConstrs(d[t] <= d[t//2] for t in self.b_index if t != 1)

        m.addConstrs(c.sum('*', t) == l[t] for t in self.l_index)

        m.addConstrs(z.sum(i, '*') <= 1 for i in range(self.n))
        
        m.addConstrs(z[i,t]  <= l[t] for t in self.l_index for i in range(self.n))
        
        
        m.addConstrs(gp.quicksum(a[j,ta] * x[i,j]  for j in range(self.p)) + mu* d[ta] 
                     <=
                     b[ta] + (2+mu)*(1- gp.quicksum(z[i,t] for t in get_left_childrenleaf(ta, self.max_depth)))
                     for ta in self.b_index for i in range(self.n))
            
       


        m.addConstrs(gp.quicksum(z[i,t] for t in get_right_childrenleaf(ta, self.max_depth))
                      <=
                      (gp.quicksum(a[j,ta] * x[i,j]  for j in range(self.p)) -b[ta] +2)/2
                      for ta in self.b_index for i in range(self.n))

        m.addConstrs(z[i,t] <= c[y[i],t] for t in self.l_index for i in range(self.n))
        

        m.addConstrs(gp.quicksum(z[i,t] *counts[i] for i in range(self.n)) >= self.min_samples_split * l[t] for t in self.l_index)
        
        m.addConstrs(l[t]<=gp.quicksum(d[s] for s in get_l(t)) for t in self.l_index)

        m.addConstrs(self.max_depth*l[t]>=gp.quicksum(d[s] for s in get_l(t)) for t in self.l_index)
        
        if self.objective == 'F1-score':
            m.addConstr(f1*(2* sumy +FP-FN)<=2*(sumy -FN))


        
        
        # m.addConstr(f1<=0.9)
        
        #m.addConstr(f1>=0.1)

        return m, a, b, c, d, l

    @staticmethod
    def _calBaseline(y):
        """
        obtain baseline accuracy by simply predicting the most popular class
        """
        mode = stats.mode(y)[0][0]
        return np.sum(y == mode)

    @staticmethod
    def _calMinDist(x):
        """
        get the smallest non-zero distance of features
        """
        min_dis = []
        for j in range(x.shape[1]):
            xj = x[:,j]
            # drop duplicates
            xj = np.unique(xj)
            # sort
            xj = np.sort(xj)[::-1]
            # distance
            dis = [1]
            for i in range(len(xj)-1):
                dis.append(xj[i] - xj[i+1])
            # min distance
            min_dis.append(np.min(dis) if np.min(dis) else 1)
        return min_dis

    def _setStart(self, x, y, a, b):
        """
        set warm start from CART
        """
        
        if self.max_depth<=1:
            # train with CART
            if self.min_samples_split > 1:
                clf = tree.DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            else:
                clf = tree.DecisionTreeClassifier(max_depth=self.max_depth)
            clf.fit(x, y)

            # get splitting rules
            rules = self._getRules(clf)

            # fix branch node
            for t in self.b_index:
                b[t].start=0
                # not split
                if rules[t].feat is None or rules[t].feat == tree._tree.TREE_UNDEFINED:
                   # d[t].start = 0
                    for f in range(self.p):
                        a[f,t].start = 0
                # split
                else:
                    #d[t].start = 1
                    for f in range(self.p):
                        if f == int(rules[t].feat):
                            a[f,t].start = 1
                        else:
                            a[f,t].start = 0
        
        else:
            
            md=self.max_depth-1
            nn_index = [i+1 for i in range(2 ** (md + 1) - 1)]
            bn_index = nn_index[:-2**md] # branch nodes
            ln_index = nn_index[-2**md:] # leaf nodes
            
            for t in self.b_index:
                if t <= 2**(self.max_depth-1) - 1:
                    b[t].start=self.bx[t]
                    for f in range(self.p):
                        a[f,t].start = self.ax[f,t]
                
                else:
                    b[t].start= 0
                    for f in range(self.p):
                        a[f,t].start = 0
                    
                    
            
            

    def _getRules(self, clf):
        """
        get splitting rules
        """
        # node index map
        node_map = {1:0}
        for t in self.b_index:
            # terminal
            node_map[2*t] = -1
            node_map[2*t+1] = -1
            # left
            l = clf.tree_.children_left[node_map[t]]
            node_map[2*t] = l
            # right
            r = clf.tree_.children_right[node_map[t]]
            node_map[2*t+1] = r

        # rules
        rule = namedtuple('Rules', ('feat', 'threshold', 'value'))
        rules = {}
        # branch nodes
        for t in self.b_index:
            i = node_map[t]
            if i == -1:
                r = rule(None, None, None)
            else:
                r = rule(clf.tree_.feature[i], clf.tree_.threshold[i], clf.tree_.value[i,0])
            rules[t] = r
        # leaf nodes
        for t in self.l_index:
            i = node_map[t]
            if i == -1:
                r = rule(None, None, None)
            else:
                r = rule(None, None, clf.tree_.value[i,0])
            rules[t] = r

        return rules


