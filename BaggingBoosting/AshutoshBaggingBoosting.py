from __future__ import division
#! /usr/bin/python

'''
This is a template outlining the functions we are expecting for us to be able to
interface with an call your code. This is not all of the functions you need. You
will also need to make sure to bring your decision tree learner in somehow
either by copying your code into a learn_decision_tree function or by importing
your decision tree code in from the file your wrote for PA#1. You will also need
some functions to handle creating your data bags, computing errors on your tree,
and handling the reweighting of data points.

For building your bags numpy's random module will be helpful.
'''

# This is the only non-native library to python you need
import numpy as np;
import csv 
import random
import math
from math import log
import sys, os;
import pandas as pd

from numpy import *
'''
Function: load_and_split_data(datapath)
datapath: (String) the location of the UCI mushroom data set directory in memory

This function loads the data set. datapath points to a directory holding
agaricuslepiotatest1.csv and agaricuslepiotatrain1.csv. The data from each file
is loaded and returned. All attribute values are nomimal. 30% of the data points
are missing a value for attribute 11 and instead have a value of "?". For the
purpose of these models, all attributes and data points are retained. The "?"
value is treated as its own attribute value.

Two nested lists are returned. The first list represents the training set and
the second list represents the test set.
'''
def load_data(datapath,fileName):
    data = open(datapath+fileName,"rb")
    o= csv.reader(data,delimiter=",")
    training_data = []

    o.next();
    for i in o:
        training_data.append(i)
    
    return training_data
#################CUSTOM FUNCTIONS##############
#Random sample of the data. Retrun 80% training data and 20% tuning data
def sample(data,parts):
    #Shuffle the data points
    shuffle_data = random.shuffle(data)
    random.shuffle(data)
    #We multiply with 0.8 to get first 80% of shuffled data as 
    size_dataset = int(len(data)*0.8)
    
    training_set = data[0:size_dataset]
    test_set = data[size_dataset:len(data)]

    return training_set,test_set

#Function One : Create_bag
#This function creates the bag of input data. It creates bag with replacement of data.
def create_bag(training_data):
    bag = []
    for r in range(0,len(training_data)):
        idx = np.random.randint(0,len(training_data));
        bag.append(training_data[idx])
    return bag
    
    
#Find the entropy of the datapoints.
# Param : 1> data_points : Datapoints
#         2> target_index : The column number of the target label in the data.
def entropy1(data_points,target_index):

    #Since its a  binary class I am calculating the number of datapoints with 0 and 1 i,e yes/no class
    yes=0
    no=0

    for data_point in data_points:
        #These are the negative points
        if int(data_point[target_index])==0:
            no=no+1
        #These are the positive points
        elif int(data_point[target_index])==1:
            yes=yes+1

    entropy_val=0.0
    entropy_child1 = 0
    entropy_child2 = 0
    entropy_yes = 0
    entropy_no =0
    if(len(data_points)>0):
        if(yes>0):
            #Find entropy of the left child
            entropy_yes=float(yes)/len(data_points)
            entropy_child1 = entropy_yes*math.log(entropy_yes,2)
        if(no>0):
            #Find entropy of the right child
            entropy_no=float(no)/len(data_points)
            entropy_child2 = entropy_no*math.log(entropy_no,2)
        entropy_val=entropy_val-( entropy_child1 + entropy_child2)

    return entropy_val 
    
#This function Creates a node of a tree.
class CreateNode:
  def __init__(self,
               #THe element column_name stores the name of the node
               column_name=-1,
               #THe element split stores the the splitting condition
               split=None,
               #This stores the items in "yes" class at that node
               yes=None,
               #This stores the items in "no" class at that node
               no=None,
               #This stores the column name of the left node
               left_node=None,
               #This stores the column name of right node
               right_node=None):
    self.column_name=column_name
    self.split=split
    self.yes=yes
    self.no=no
    self.left_node=left_node
    self.right_node=right_node
    
    
#This function creates the Tree by calculating the entropy and returns a built decision tree.
def createTree(datapoints,depth): 
    #1. When all the datapoints have been classified
    #2. When the depth has been reached
    if(len(datapoints)==0 or depth==0) : 
        return CreateNode() 
    target_index = 20
    node_entropy = entropy1(datapoints,target_index)
    #Drop the column with the heading as "id"
    #print("node entropy is " + str(node_entropy))

    #Convert the dataset into columns instead of rows to find the best split at each feature.
    #1. Convert rows to columns
    training_cols =  [list(col) for col in zip(*datapoints)]

    #Stores best information gain
    parent_gain = 0
    decision_column = 0
    decision_value = 0
    #Stores the left tree data 
    decision_data1 = 0
    #Stores the right tree data 
    decision_data2 = 0
    
    #Stores the information gain of all the features.
    node_gain = []
    best_column = []
    best_value = []
    best_data1 = []
    best_data2 = []
    
    #First iterate over all the features and find the best feature at this node
    #We neglect the first column as the first feature is the target column
    for feature in range(0,len(training_cols)) :
        if(feature==target_index):
            node_gain.append(node_gain[len(node_gain)-1])
            best_column.append(best_column[len(best_column)-1])
            best_value.append(best_value[len(best_value)-1])
            best_data1.append(best_data1[len(best_data1)-1])
            best_data2.append(best_data2[len(best_data2)-1])
            pass;
        else:
             
            #Now iterate over all the values of each feature.
            #1. First sort all the values.
            #training_cols[feature] = sorted(training_cols[feature])

            value = 0;
            data1 = []
            data2 = []

            #Find the left and right tree data 
            for training_point in datapoints:
                    if(training_point[feature]==value):
                        data1.append(training_point)
                    else:
                        data2.append(training_point)
                        
            data1_size = float(float(len(data1))/float(len(datapoints)))
            data2_size = float(float(len(data2))/float(len(datapoints)))
            #Calculate entropy of child 1
            entropy_data1 = data1_size*entropy1(data1,target_index)
            #Calculate entropy of child 2
            entropy_data2 = data2_size*entropy1(data2,target_index)
            #sum children's entropy
            child_entropy = entropy_data1+entropy_data2
            information_gain=node_entropy-child_entropy

            if len(data1)>=0 and len(data2)>=0:
                if information_gain > parent_gain:
                    parent_gain = information_gain
                    decision_column = feature
                    decision_value = value
                    decision_data1 = data1
                    decision_data2 = data2

            node_gain.append(parent_gain)
            best_column.append(decision_column)
            best_value.append(decision_value)
            best_data1.append(decision_data1)
            best_data2.append(decision_data2)
        
    #Find the index of the best gain from the list.
    if(len(node_gain)>=1):
        index = node_gain.index(max(node_gain))
        best_node_gain=node_gain[index]
    else:
        best_node_gain=0

    #print(index, " ",best_column[index])
    #If the gain is greater than 0 then only create the new node and left and right node.
    if(best_node_gain>0 ):
        #Create left node
        #1. Remove the existing parent node feature from the consideration at next levels.
        #1a. Convert rows to columns

        #print("Depth of the left tree is " + str(depth-1))
        left_depth = depth-1
        left_node = createTree(best_data1[index],left_depth)
        
        #Create right node
        #1. Remove the existing parent node feature from the consideration at next levels.
        #1a. Convert rows to columns
        #print("Create Right node")
        #print("Depth of the right tree is " + str(depth-1))
        right_depth = depth-1
        right_node = createTree(best_data2[index],right_depth)
        return CreateNode(
               column_name=best_column[index],
               #THe element split stores the the splitting condition
               split=value,
               #This stores the column name of the left node
               left_node=left_node,
               #This stores the column name of right node
               right_node=right_node
               )
    else:
        #print("Feature sorted at depth" )
        #print(depth)
        
        temp_y = 0
        temp_n = 0
        for i in training_cols[target_index]:
            if(i==0):
                temp_n = temp_n + 1
            if(i==1):
                temp_y = temp_y + 1
        #print("Yes ->",str(temp_y))
        #print("No ->",str(temp_n))

        return CreateNode(yes=temp_y,no=temp_n)

#THis function checks tree and each datapoint.
#It returns the prediction for the input observation.
def classify(observation,tree):

  if tree.yes!=None and tree.no!=None:
    if(tree.no>tree.yes):
        return 0
    else:
        return 1
  else:
    v=observation[tree.column_name]
    branch=None
    #Goes to left tree
    if v<=tree.split: 
        branch=tree.left_node
    #Goes to right tree
    else: 
        branch=tree.right_node
    
    return classify(observation,branch)
    
#Create Bag of data with replacement.
def create_bag(training_data):
    bag = []
    for r in range(0,len(training_data)):
        idx = np.random.randint(0,len(training_data));
        bag.append(training_data[idx])
    return bag 
    


def mydataset(o):
    training_set, test_set = o[:int(len(o)*0.7)], o[int(len(o)*0.7):]
    training_cols =  [list(col) for col in zip(*training_set)]
    del training_cols[21]
    training_points =  [list(col) for col in zip(*training_cols)]

    training_set = training_points
    #CHANGE THE DATATYPE OF THE NUMBERS FROM STRING TO INTEGER AND KEEP THE STRINGS AS STRING.
    for index,datapoint in enumerate(training_set):
        training_set[index]=map(int,training_set[index][0:len(training_set[index])])

    #1. Convert rows to columns
    training_cols1 =  [list(col) for col in zip(*test_set)]
    final_target = map(int,training_cols1[20])
    del training_cols1[21]
    #3. Again convert the data to rows
    test_set =  [list(col) for col in zip(*training_cols1)]

    for index,datapoint in enumerate(test_set):
        test_set[index]=map(int,test_set[index][0:len(test_set[index])])

    return training_set, test_set, final_target

#This class is used to create the Adaboost Model.
class AdaBoost:

    def __init__(self, data):
        building_set, training_set, final_target = mydataset(data)
        self.data = data
        self.training_set = training_set
        self.building_set = building_set
        self.final_target = final_target
        self.N = len(self.training_set)
        self.weights = ones(self.N)/self.N
        self.treeList = []
        self.RULES = []
        self.ALPHA = []

    def set_rule(self, func, dpt, numofmodels):
        for repeat in range(numofmodels):
            self.building_set = create_bag(self.building_set)
            Tree = createTree(self.building_set,dpt)
            errors = array([func(self.training_set[t], Tree)==self.final_target[t] for t in range(len(self.training_set))])
            e = (errors*self.weights).sum()
            
            alpha = 0.5 * log((1-e)/e)
            #print 'e=%.2f a=%.2f'%(e, alpha)
            w = zeros(self.N)
            for i in range(self.N):
                if errors[i] == 1: w[i] = self.weights[i] * exp(alpha)
                else: w[i] = self.weights[i] * exp(-alpha)
            self.weights = w / w.sum()
            self.treeList.append(Tree)
            self.ALPHA.append(alpha)

    def evaluate(self,test_points1,final_target1):
        NR = len(self.treeList)
        tru = []
        #print(len(final_target1))
        lit = len(test_points1)+700
        for x in range(len(test_points1)):
            hx = []
            for i in range(NR):
                val = classify(test_points1[x],self.treeList[i])
                hx.append(self.ALPHA[i]*val)
                
            if(sign(final_target1[x]) == sign(sum(hx))):
                tru.append(1)
            else:
                tru.append(0)

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        predicted_label=tru
        test_points_size = lit
        actual = map(lambda x:x if x== 1 else 0,final_target1)
        
        print("The accuracy of the boosting model is ")
        print(float(sum(tru))/(test_points_size)*100)
        


#This method is used to generate the bagged and boosted trees.  
def generate_tree(training_set,dpt):

    #Preprocess the data
    training_cols =  [list(col) for col in zip(*training_set)];
    #Delete the column 21 which will degrade the performance of the tree. It is one of the repetitive data column. 
    del training_cols[21];
    if(dpt<=3): dpt=4;
    training_set =  [list(col) for col in zip(*training_cols)]
    
    #CHANGE THE DATATYPE OF THE NUMBERS FROM STRING TO INTEGER AND KEEP THE STRINGS AS STRING.
    for index,datapoint in enumerate(training_set):
        training_set[index]=map(int,training_set[index][0:len(training_set[index])])
    
    Tree = createTree(training_set,dpt)
    
    return Tree;
'''
Function: learn_bagged(tdepth, numbags, datapath)
tdepth: (Integer) depths to which to grow the decision trees
numbags: (Integer)the number of bags to use to learn the trees
datapath: (String) the location in memory where the data set is stored

This function will manage coordinating the learning of the bagged ensemble.

Nothing is returned, but the accuracy of the learned ensemble model is printed
to the screen.
'''
def learn_bagged(tdepth, numbags, datapath):
    
    Input_fileName = "\\agaricuslepiotatrain1.csv"
    training_data = load_data(datapath,Input_fileName)
    
    Bags = []
    bag = []
    
    #Iteratively create multiple bags and store the bag into the List of Bags.
    
    for bag_size in range(1,numbags+1):
        u = create_bag(training_data)
        Bags.append(create_bag(training_data))
    
    Trees = []

    #Iteratively create multiple Trees and store the trees into the List of Trees.
    it = 0
    for bag in Bags:
        Tree  = generate_tree(bag,tdepth)
        it = it+1
        print("Tree "+str(it)+" built!")
        Trees.append(Tree)
       
        
        
    Input_test_fileName = "\\agaricuslepiotatest1.csv"
    test_data = load_data(datapath,Input_test_fileName)

    training_cols1 =  [list(col) for col in zip(*test_data)]
    final_target = map(int,training_cols1[20])
    #Delete the bruises?no column 
    del training_cols1[21]
    #3. Again convert the data to rows
    test_points1 =  [list(col) for col in zip(*training_cols1)]
    
    for index,datapoint in enumerate(test_points1):
        test_points1[index]=map(int,test_points1[index][0:len(test_points1[index])])
    
    df = pd.DataFrame()
    actual_id = []
    actual_label = []
    predicted_id = []
    predicted_label = []
    actual = {}
    prediction = {}
    for j,tune in enumerate(test_points1):
        current_prediction_label = []
        current_prediction_id = [] 
        for Tree in Trees:
            val = classify(tune,Tree)
            current_prediction_id.append(j)
            current_prediction_label.append(val)
        #print(tune, current_prediction_label,max(set(current_prediction_label), key=current_prediction_label.count))
        predicted_id.append(j)
        actual_label.append(final_target[j])
        predicted_label.append(max(set(current_prediction_label), key=current_prediction_label.count))
        prediction[j] = max(set(current_prediction_label), key=current_prediction_label.count)
        actual[j]=final_target[j]
 
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(0,len(predicted_label)):
        if(actual[i]==1 and predicted_label[i]==1):
            tp = tp +1
        elif(actual[i]==1 and predicted_label[i]==0):
            fn = fn + 1 
        elif(actual[i]==0 and predicted_label[i]==1):
            tn = tn + 1 
        elif(actual[i]==0 and predicted_label[i]==0):
            fp = fp + 1
    print("For depth "+ str(tdepth) + " and number  of bags is "+str(numbags) )
    same_elements =  set(actual.items()) & set(prediction.items())
    print("The accuracy of the bagging model is ")
    print(float(len(same_elements))/float(len(test_points1))*100)
    print("The confusion Matrix is as follows:");
    print(str(tp) + "\t" + str(fn) + "\n" + str(fp) + "\t" + str(tn))

'''
Function: learn_boosted(tdepth, numtrees, datapath)
tdepth: (Integer) depths to which to grow the decision trees
numtrees: (Integer) the number of boosted trees to learn
datapath: (String) the location in memory where the data set is stored

This function wil manage coordinating the learning of the boosted ensemble.

Nothing is returned, but the accuracy of the learned ensemble model is printed
to the screen.
'''
def learn_boosted(tdepth, numtrees, datapath):
    data = open(datapath+"\\agaricuslepiotatrain1.csv","rb")
    training_data = csv.reader(data,delimiter=",")
    dpt = tdepth
    o = []
    #Ignore first row of column name by moving to second row  and adding it to the list.
    
    training_data.next();
    for i in training_data:
        for it,u in enumerate(i):
            if i[it]=='0': i[it]= '-1'
        o.append(i)
        
    training_set, test_set = sample(o,2)
    training_cols =  [list(col) for col in zip(*training_set)]
    if(dpt<=3): dpt=4;
    del training_cols[21]
    training_points =  [list(col) for col in zip(*training_cols)]
    
    training_set = training_points
    #CHANGE THE DATATYPE OF THE NUMBERS FROM STRING TO INTEGER AND KEEP THE STRINGS AS STRING.
    for index,datapoint in enumerate(training_set):
        training_set[index]=map(int,training_set[index][0:len(training_set[index])])
        
    #1. Convert rows to columns
    training_cols1 =  [list(col) for col in zip(*test_set)]
    final_target = map(int,training_cols1[20])
    del training_cols1[21]
    #3. Again convert the data to rows
    test_set =  [list(col) for col in zip(*training_cols1)]
    
    for index,datapoint in enumerate(test_set):
        test_set[index]=map(int,test_set[index][0:len(test_set[index])])    

           
    m = AdaBoost(o)
    m.set_rule(lambda x,y: classify(x,y),dpt, numtrees)
    
    Input_test_fileName = "\\agaricuslepiotatest1.csv"
    test_data = load_data(datapath,Input_test_fileName)
    x = []
    for i in test_data:
        for it,u in enumerate(i):
            if i[it]=='0': i[it]= '-1'
        x.append(i)
    test_data1 = x
    training_cols1 =  [list(col) for col in zip(*test_data1)]
    final_test_target = map(int,training_cols1[20])
    #Delete the bruises?no column 
    del training_cols1[21]
    #3. Again convert the data to rows
    test_points1 =  [list(col) for col in zip(*training_cols1)]
    
    for index,datapoint in enumerate(test_points1):
        test_points1[index]=map(int,test_points1[index][0:len(test_points1[index])])


    print("For depth "+ str(tdepth) + " and number of trees "+str(numtrees) )
    m.evaluate(test_points1,final_test_target)

    pass;


if __name__ == "__main__":
    # The arguments to your file will be of the following form:
    # <ensemble_type> <tree_depth> <num_bags/trees> <data_set_path>
    # Ex. bag 3 10 mushrooms
    # Ex. boost 1 10 mushrooms

    # Get the ensemble type
    entype = sys.argv[1];
    
    # Get the depth of the trees
    #tdepth = int(sys.arg[2]);
    tdepth = 2
    # Get the number of bags or trees
    nummodels = int(sys.argv[3]);
    
    
    # Get the location of the data set
    datapath = sys.argv[4];
    # Check which type of ensemble is to be learned
    if entype == "bag":
        # Learned the bagged decision tree ensemble
        learn_bagged(tdepth, nummodels, datapath);
    else:
        # Learned the boosted decision tree ensemble
        learn_boosted(tdepth, nummodels, datapath);
