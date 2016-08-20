import numpy as np
import pdb
import csv
import math
import sys
from  sets import Set
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from KNN import *
from loadData import *
from os.path import basename
from os.path import dirname
from os.path import stat
from os import mkdir
from Preprocess_Extract import *
from Distribution_symbols import *
from FileWrite import *
from Segmentation import *
import time
import cPickle as cp
from RelationShip_Classifier import * 
class Classification:
    '''
    Classification is the main class that calls the classification ,segmentation  and parsing  methods.

    '''
    def __init__(self):
        pass

    def classification(self,file_path_till_Traininkml,classifier_obj,str_opt):
        '''
        The function is used to classify the data.
        Input:
        file_path_till_Traininkml : Path of inkml files
        classifier_obj : Classifier object
        str_opt: Test or Train 

        '''
        load_obj = loadData()
        lg_folder_name="classification_"+str_opt
        file_write_obj=FileWrite(lg_folder_name)
        flag=False
        with open('split_files.txt','r') as f:
            for line in f:
                if flag:
                    files=line
                    files=files.strip("Set([")
                    files=files.strip("])\n")            
                    list_files=files.split(', ')
                    break
                
                elif line.startswith(str_opt):
                    flag=True
                    continue
        
        count_traces=0
        count=0
        print 'Classification started'
        count=0
        for fileName in list_files:
            count=count+1
            fileName=fileName.strip("'")
            print "count= %d" % (count)
            fileName=fileName.replace("/home/sbp3624/PatternRecog/TrainINKML_v3/",file_path_till_Traininkml)
            print fileName+"\n"
            root_obj, trace_obj_dict = load_obj.loadInkml(fileName)
            symbols = load_obj.get_symbol(root_obj,trace_obj_dict)
            symbol_list=symbols
            
            X_test=[]
            count_traces=0
            for symbol in symbol_list:
                features=symbol.get_features()
                X_test.append(features)
                count_traces=count_traces+len(symbol.symbol_list)
            X_test_final=np.asarray(X_test)
            predict_labels=classifier_obj.predict(X_test_final)
            #Write this to lg file.
            file_write_obj.write_to_lg(predict_labels,fileName,symbol_list,count_traces,lg_folder_name)              


def main(argv):
    '''
    main method
    '''
    start_time = time.time()
    
    # Extract features from the given data.

    p=Preprocess_Extract() 
   
    # This contaions list of files for training and testing .2/3 for training and 1/3rd for testing
    # Get list of train files
    f=open('split_files.txt','r')

    # Get files for train data
    files=f.readline()
    files=f.readline()
    files=files.strip("Set([")
    files=files.strip("])\n")
    list_files=files.split(', ')
    train_files=Set(list_files)

    #Get files for test data    
    files=f.readline()
    files=f.readline()
    files=files.strip("Set([")
    files=files.strip("])\n")
    list_files=files.split(', ')
    test_files=Set(list_files)

    # Load Classifier 
    print 'Loading Classifier\n'
    f_read_Classifier= open ('f_classifier','rb')
    rfc = cp.load(f_read_Classifier)

    # load relationship object
    print 'Loading Relational Classifier\n'
    f_read_rel=open('f_rel_classifier','rb')
    rel_rfc= cp.load(f_read_rel)
    
    # Need to specify path where the inkml files are located.
    file_path_till_Traininkml='/home/sbp3624/PatternRecog/TrainINKML_v3/'
    s= Segmentation()
    print "Classification , Segmentation and Parsing for Testing data started \n"
    str_opt="Test"

    # First segmenting the data 
    s.sym_segmentation(rfc,file_path_till_Traininkml,str_opt,rel_rfc)
    
    print 'Done!!!!'
    print "Total Time Taken= %f" %(time.time()-start_time)

if __name__=="__main__":
    main(sys.argv[1:])
