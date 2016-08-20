import numpy as np
from loadData import *
from Symbol import *
from os.path import basename
from Segmentation import *
from sklearn.ensemble import RandomForestClassifier
from Edmonds import *
from scipy.sparse.csgraph import minimum_spanning_tree
import cPickle as cp
class RelationShipClassifier:
    '''
    The class is used for parsing and training relational classifier.
    '''
    def __init__(self):
        pass

    def get_relationship_data(self,file_path_till_traininkml,str_opt,file_path_lg_train):
        '''
        The method extracts the data required for training relationship classifier.
        '''

        load_obj=loadData()
        symbol_obj=Symbol()
        X_rel_train=[]
        y_rel_train=[]
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
            fileName_lg=basename(fileName)
            pos=fileName_lg.find(".")
            fileName_sub=fileName_lg[:pos] + ".lg"
            fileName_lg=file_path_lg_train+fileName_sub
            root_obj, trace_obj_dict = load_obj.loadInkml(fileName)
            dict_sym={}
            list_Obj=[]
            list_R=[]
      
            with open(fileName_lg,"r") as f_read:
                for line in f_read:
                    line=line.strip("\n")
                    line=line.replace(" ","")
                    if line.startswith("O"):
                        list_obj=line.split(",")
                        dict_sym[list_obj[1]]=list_obj[4:]
                    elif line.startswith("R"):
                        list_R=line.split(",")
                        list_1=dict_sym[list_R[1]]
                        list_1+=dict_sym[list_R[2]]
                        rel_label=list_R[3]
                        list_traceobj_rel=[]           
                        total_points=[]
                        for trace_id in list_1:
                            #list_traceobj_rel.append(trace_obj_dict[int(trace_id)])
                            total_points+=trace_obj_dict[int(trace_id)].original_points
                        #First get the original points then normalize
                        trace_obj=Trace(points_float=total_points)
                        trace_obj.normalization()
                        list_traceobj_rel.append(trace_obj)
                        symbol_obj.symbol_list=list_traceobj_rel
                        features=symbol_obj.get_features()
                        X_rel_train.append(features)
                        y_rel_train.append(rel_label)
                        list_traceobj_rel.remove(trace_obj)


                        
        return X_rel_train,y_rel_train


    def sym_parsing(self,rel_classifier_obj,file_path_till_traininkml,str_opt):
      
        '''
        The function calls methods from MinimumSpanningTree class to segment and classify symbols and then parses the symbols, finally
        writing it to an lg files.
        '''
        load_obj = loadData()
        e=Graph()
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
            symbols = load_obj.get_symbol(root_obj,trace_obj_dict)
            adj_matrix,dict_mapping_Symbol_index,index_to_symbol=e.LineOfSight(symbols,rel_classifier_obj)
            dict_map_rel_to_syms=self.get_parse_layout(adj_matrix,dict_mapping_Symbol_index,index_to_symbol,rel_classifier_obj)
            dict_map_rel_to_syms=dict_map_rel_to_syms[0]  # because the funtion returns a tuple
            self.write_to_lg(fileName,symbols,dict_map_rel_to_syms,lg_folder_name)

    def get_parse_layout(self,adj_matrix,dict_mapping_Symbol_index,index_to_symbol,rel_classifier_obj):
        '''
        The function returns a dictionary of relation corresponding to two symbols, which is later used for writing it to a lg files.
        '''

        symbol_obj=Symbol()
        dict_map_rel_to_syms={}
        
        tree=minimum_spanning_tree(adj_matrix)
        mst=tree.toarray()
        
        for i in xrange(mst.shape[0]):
            eye_obj=index_to_symbol[i]
            arr=np.where(mst[i]>0)[0]
            if arr.shape[0]==0:
                continue
            else:
                trace_list=[]
                points_eye=[]
                for k in xrange(len(eye_obj.symbol_list)):
                        points_eye+=eye_obj.symbol_list[k].original_points

                for j in xrange(arr.shape[0]):
    
                    other_obj=index_to_symbol[arr[j]]
                    total_points=[]
                    for l in xrange(len(other_obj.symbol_list)):
                        total_points+=other_obj.symbol_list[l].original_points
               
                    total_points=points_eye+total_points
                    trace_obj=Trace(points_float=total_points)
                    trace_obj.normalization()
                    trace_list.append(trace_obj)
                 
                    symbol_obj.symbol_list=trace_list
                    features=symbol_obj.get_features()
                    X_test=np.asarray(features)
                    label=rel_classifier_obj.predict(X_test.reshape(1,-1))
                
                    if label[0] not in dict_map_rel_to_syms:
                        dict_map_rel_to_syms[label[0]] = []
                        dict_map_rel_to_syms[label[0]].append((eye_obj,other_obj))
                    else:
                        dict_map_rel_to_syms[label[0]].append((eye_obj,other_obj) )           
                 
                    trace_list.remove(trace_obj)
                    
        return dict_map_rel_to_syms,
    
    
    def write_to_lg(self,fileName,symbol_list,dict_map_rel_to_syms,str_task):
        
        '''
        The function is used to write the symbol objects and their relationships to lg files
        '''

     
        dict_map_temp={} # For parsing
        fileName_lg=basename(fileName)
        dot_pos = fileName_lg.find(".")
        fileName_lg=fileName_lg[:dot_pos]+".lg"
        dir_name=str_task+"_lg_files"
        f = open(dir_name+"/"+fileName_lg,"w") 
        f.write('# IUD '+fileName+'\n'+'# [ OBJECTS ]')
        count_traces=0
        for sym_object in symbol_list:
            count_traces=count_traces + len(sym_object.symbol_list)
            
        f.write('\n# Primitive Nodes (N): '+str(count_traces))
        f.write('\n#    Objects (O): '+ str(len(symbol_list)))
        dict_map_of_symbol_count={}
        for i in xrange(len(symbol_list)):
            
            
            label=symbol_list[i].symbol_class
            if label in dict_map_of_symbol_count:
                dict_map_of_symbol_count[label]=dict_map_of_symbol_count[label]+1
            else:
                dict_map_of_symbol_count[label]=0
                dict_map_of_symbol_count[label]=dict_map_of_symbol_count[label]+1
           
            
            sym_class=symbol_list[i].symbol_class
            if sym_class==',':
                sym_class='COMMA'
                
            sym_count=sym_class+"_"+str(dict_map_of_symbol_count[label])
            
            sym_obj=symbol_list[i]
           
            dict_map_temp[sym_obj]=sym_count    # For parsing
            str_traces=""
            for trace_obj in sym_obj.symbol_list:
                str_traces+=" ,"+str(trace_obj.trace_id)

            prob = "1.0"
            f.write('\nO, '+sym_count+", "+sym_class+", "+prob+str_traces)

        f.write('\n')
         
        for k in  dict_map_rel_to_syms:
            rel=k
            list_rel= dict_map_rel_to_syms[k]
            for val in list_rel:
           
                eye_obj,other_obj=val
                eye_label=dict_map_temp[eye_obj]
                other_label=dict_map_temp[other_obj]
                f.write(' \nR, '+ eye_label +', '+other_label+', '+rel+', '+'1.0')

     
        f.close() 
    
def main() :
    '''
    main function 
    '''
  
    print 'START!!!!!!\n'
    r=RelationShipClassifier()        
    file_path_lg_train='/home/sbp3624/PatterRecognition/segmentation_Train_lg_files_out/'
    e=Graph()  # Need to give a classifier there.
    '''
    f_rfc_rel=open("f_rel_classifier",'wb')
    X_rel_train_arr=np.asarray(X_rel_train)
    y_rel_train_arr=np.asarray(y_rel_train)
    print X_rel_train_arr.shape
    print '\n'
    rfc = RandomForestClassifier(n_estimators=60)
    rfc.fit(X_rel_train,y_rel_train)
    print "dumping!!!!!\n"
    cp.dump(rfc,f_rfc_rel)
    f_rfc_rel.close()
    '''
    print 'Loading Realtional Classifier \n'
    f_rel_read=open('f_rel_classifier','rb')
    rfc=cp.load(f_rel_read)
    str_opt="Test"
    file_path_till_traininkml='/home/sbp3624/PatterRecognition/TrainINKML_v3/'
    r.sym_parsing(rfc,file_path_till_traininkml,str_opt)

    print "Done!!!"

if __name__=="__main__":
    main()
    
