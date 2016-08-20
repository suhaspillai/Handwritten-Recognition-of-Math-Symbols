import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from Symbol import *
from Edmonds import *
from RelationShip_Classifier import * 
import pdb


class MinimumSpanningTree:
    '''
    This class is used for segmentation, it create a minimum spanning tree, which is then used by dynamic programining algorithm.   
    ''' 
    def __init__main():
        pass

    
    def get_spanning_tree(self, trace_objs):
        '''
        The function creates minimum spanning tree using Euclidean distance
        input :
        trace_objs: These are strokes

        return : an matrix which represents minimum spanning tree.
        
        '''
        N=len(trace_objs)   # trace_objs  is a dictionary
        adj_matrix=np.zeros((N,N))
        for i in xrange(N):
           
            a_x,a_y=trace_objs[i].get_centroids()
            for j in xrange(i+1,N):
                b_x,b_y=trace_objs[j].get_centroids()
                adj_matrix[i,j]=np.sum(np.square(a_x-b_x)+np.square(a_y-b_y))
                
        tree=minimum_spanning_tree(adj_matrix)
        return (tree.toarray().astype(int))


    def get_spanning_tree_symbol_level(self, symbol_objs):
        '''
        The function creates minimum spanning tree using Euclidean distance
        input :
        trace_objs: These are strokes

        return : an matrix which represents minimum spanning tree.
        
        '''
        N=len(trace_objs)   # trace_objs  is a dictionary
        adj_matrix=np.zeros((N,N))
        for i in xrange(N):  
            a_x,a_y=trace_objs[i].get_centroids()
            for j in xrange(i+1,N):
                b_x,b_y=trace_objs[j].get_centroids()
                adj_matrix[i,j]=np.sum(np.square(a_x-b_x)+np.square(a_y-b_y))
                
        tree=minimum_spanning_tree(adj_matrix)
        return (tree.toarray().astype(int))





    
    def memoized_initialization(self,n):
        '''
        The function is used for initialization of r,s arrays
        input : size of the array

        returns:
        r: stores the distance for every possible split
        s: stroes the index of what is the correct split, this is used for back tracking.
        '''
        r = np.zeros(n)
        s=np.zeros(n)
        for i in xrange(n):
            s[i]=0
            r[i]=-1000000
        return r,s

    # To check if there is an edge between 2 nodes.
    def check_edge(self,list_strokes,minimum_span_tree):

        '''
        The function check whether there is an edge between the two vertices (strokes) of the graph.
        input:
        list_strokes : List of strokes which are vertices of a graph.
        minimum_span_tree: The minimum spanning tree of the graph.

        returns:
        true is there is a direct edge between the two vertices.
        '''
        
        flag=True
        for i in xrange(len(list_strokes)-1):
            index_1=list_strokes[i]
            index_2=list_strokes[i+1]
            if (minimum_span_tree[index_1,index_2]==True) or (minimum_span_tree[index_2,index_1]==True):
                continue
            else:
                flag=False
                break
        return flag


    def optimal_segmentation(self,total_iter,start,r,s,classifier_obj,X_dict_test,min_span_tree,dict_mapping_strokes,symbol_obj):
        '''
        The function is used to create optimal segmentation of strokes suing minimum spanning tree and Dyanmic Programming.      
        input :
        total_iter : Total number of iterations , i.e total number of strokes.
        r: It is used lookup array, to get the values of already computed distance .
        s: It is used get the index , which gives the best split of segemantation of strokes. i.e the iterartion, number which gives max distance for that set of strokes.
        classifier_obj : Classifier's Object used for classification.
        X_dict_test :  Samples for segmenatation
        min_span_tree : minimum spanning tree matrix
        dict_mapping_strokes: This stores the list of strokes to be grouped together for the best split .
        symbol_obj : Object of symbol class


        returns:
        r[j]: It is used lookup array, to get the values of already computed distance .This returns the max values for est possible combination of strokes.
        s: It is used get the index , which gives the best split of segemantation of strokes. i.e the iterartion, number which gives max distance for that set of strokes.
        dict_mapping_strokes: This stores the list of strokes to be grouped together for the best split .
        '''
        
        if r[total_iter-start]>=0:
            return r[total_iter-start],s,dict_mapping_strokes
        elif total_iter-start==0:
            r[total_iter-start]=1
            return r[total_iter-start],s,dict_mapping_strokes
        else:
            q=-10000  # initialized to some -ve value
            stroke_set_list=[]
            list_strokes=[]
            counter =1
            for i in xrange(start,total_iter):
                list_strokes.append(i)
                if self.check_edge(list_strokes,min_span_tree):
                    
                    stroke_set=X_dict_test[i-1]
                    stroke_set_list.append(stroke_set)        # appending the strokes/trace object , if the symbol contains multiple strokes.
                    symbol_obj.symbol_list=stroke_set_list
                    features_list=symbol_obj.get_features()
                    stroke_set_array=np.asarray(features_list)
                    prob_stroke_set=np.max(classifier_obj.predict_proba(stroke_set_array.reshape(1,-1)))
                    start_next_iter=i+1
                    val,s,dict_mapping_strokes=self.optimal_segmentation(total_iter,start_next_iter,r,s,classifier_obj,X_dict_test,min_span_tree,dict_mapping_strokes,symbol_obj)
                    val=(prob_stroke_set+val)
                    #val=val*prob_stroke_set
                    if q<val:
                        q=val
                        s[total_iter-start]=counter       # This is back track for the best path or optimal partioning
                        dict_mapping_strokes[total_iter-start]=[k for k in  list_strokes]
                    r[total_iter-start]=q
                    counter=counter+1
                else:
                    continue
            return r[total_iter-start],s,dict_mapping_strokes




    def get_segmentation(self,trace_obj_dict,classifier_obj,symbol_obj,File_write_obj,fileName,lg_folder_name,rel_classifier_obj):
        '''
        The function is calls optimal segmenattation and then call method to write lg files
        Input
        trace_obj_dict: trace objects for that file
        classifier_obj: classifier object
        symbol_obj: Symbol class object
        File_write_obj:File write object
        lg_folder_name: Foldername to store lg files
        '''
        min_span_tree=self.get_spanning_tree(trace_obj_dict)
        demo_min_span_tree=min_span_tree
        temp=np.zeros(demo_min_span_tree.shape[0])
        temp=temp.reshape(temp.shape[0],1)
        demo_min_span_tree=np.hstack((temp,demo_min_span_tree))
        temp=np.zeros(demo_min_span_tree.shape[1])
        demo_min_span_tree=np.vstack((temp,demo_min_span_tree))
        demo_min_span_tree_bool=demo_min_span_tree>0
        # Now get best probability distribution for set of strokes  
        dict_trace_map={}
        count=0
        s=[]
        # This will help in mapping which index belong to which trace
        for i in xrange(1,len(trace_obj_dict)+1):
            dict_trace_map[i]=trace_obj_dict[i-1]

        n=len(trace_obj_dict)+1
        r,s = self.memoized_initialization(n)
        dict_mapping_strokes={}
        r,s,dict_mapping_strokes=self.optimal_segmentation(n,1,r,s,classifier_obj,trace_obj_dict,demo_min_span_tree_bool,dict_mapping_strokes,symbol_obj)
        index=n-1
        list_predict_symbol=[]
        symbol_list=[]
        count_traces=0

        #Back Track and get the list of strokes
        while index>0:
            
            x=s[index]
            list_strokes=dict_mapping_strokes[index]
            symbol_object=Symbol()
            symbol_list.append(symbol_object)  # List of symbol object , which will be used to extract features            
            strokes_symbol=[]    
            for k in list_strokes:
                trace_obj=dict_trace_map[k]
                strokes_symbol.append(trace_obj)
                count_traces=count_traces+1
                
            symbol_object.symbol_list=strokes_symbol
            index=index-int(x)

        stroke_to_pixel=[]
        X_test=[]
        
        for symbol in symbol_list:
            features=symbol.get_features()
            X_test.append(features)          
            
        X_test_final=np.asarray(X_test)
        
        predict_labels=classifier_obj.predict(X_test_final)
       
        e=Graph()
        r=RelationShipClassifier()
     

        adj_matrix,dict_mapping_Symbol_index,index_to_symbol=e.LineOfSight(symbol_list,rel_classifier_obj)
       
        dict_map_rel_to_syms=r.get_parse_layout(adj_matrix,dict_mapping_Symbol_index,index_to_symbol,rel_classifier_obj)
        dict_map_rel_to_syms=dict_map_rel_to_syms[0]
       
        File_write_obj.write_to_lg(predict_labels,fileName,symbol_list,dict_map_rel_to_syms,count_traces,lg_folder_name)     
