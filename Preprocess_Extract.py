import numpy as np
from numpy import linalg as LA
from sets import Set
from Symbol import *
from loadData import *
import matplotlib.pyplot as plt

class Preprocess_Extract():

    '''
    The class is used to preprocess and extract features from the training data
    '''
    def __init__(self):
        pass

    def extract_data_without_pca(self,file_set,file_path_till_Traininkml):
        '''
        The function is used to extract data from inkml files
        '''
        load_obj = loadData()
        count = 0
        X_train= []
        y_train=[]
        stroke_to_pixel=[]
        count_fName=0
        for fileName in file_set:
            count_fName=count_fName+1
            print ("File No=%d ") % (count_fName)
            fileName=fileName.strip("\'")
            fileName=fileName.replace('/home/sbp3624/PatternRecog/TrainINKML_v3/',file_path_till_Traininkml)
            root_obj, trace_obj_dict = load_obj.loadInkml(fileName)
            symbols = load_obj.get_symbol(root_obj,trace_obj_dict)

            
            for symbol in symbols:
                features = symbol.get_features()
                X_train.append(features)
                y_train.append(symbol.symbol_class)
        N=len(X_train) 

        return X_train,y_train

    def extract_data_without_pca_relationship(self,file_set,file_path_till_Traininkml):
        '''
        The function is used to extract data from inkml files
        '''
        load_obj = loadData()
        count = 0
        X_train= []
        y_train=[]
        stroke_to_pixel=[]
        count_fName=0
        for fileName in file_set:
            count_fName=count_fName+1
            print ("File No=%d ") % (count_fName)
            fileName=fileName.strip("\'")
            fileName=fileName.replace('/home/sbp3624/PatternRecog/TrainINKML_v3/',file_path_till_Traininkml)
            root_obj, trace_obj_dict = load_obj.loadInkml(fileName)
            symbols = load_obj.get_symbol(root_obj,trace_obj_dict)
            # get relationship data
            
            
            for symbol in symbols:
                features = symbol.get_features()
                X_train.append(features)
                y_train.append(symbol.symbol_class)
        N=len(X_train) 
        
        return X_train,y_train


    
if __name__=="__main__":
    pass
