from MinimumSpanningTree import *
from loadData import *
from Symbol import *
from FileWrite import *
import numpy as np


class Segmentation:
    '''
    The class class methods to segment strokes
    '''
    
    def __init__(self):
        pass

    def sym_segmentation(self,classifier_obj,file_path_till_traininkml,str_opt,rel_classifier_obj):
        '''
        The function calls methods from MinimumSpanningTree to segment,classify and parse symbols
        Input 
        classifier_obj - Classifier pretrained model
        file_path_till_traininkml - path to inkml file
        str_opt - Train or Test
        rel_classifier_obj - Realationship classifier pretrained model.   
        '''
        load_obj = loadData()
        m=MinimumSpanningTree()
        symbol_obj=Symbol()
        
       
      
        lg_folder_name="parsing_"+str_opt
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

        count=0


        for fileName in list_files:
            count=count+1
            fileName=fileName.strip("'")
            print "count= %d" % (count)
            fileName=fileName.replace("/home/sbp3624/PatternRecog/TrainINKML_v3/",file_path_till_traininkml)
            root_obj, trace_obj_dict = load_obj.loadInkml(fileName)
          
         
            m.get_segmentation(trace_obj_dict,classifier_obj,symbol_obj,file_write_obj,fileName,lg_folder_name,rel_classifier_obj)
