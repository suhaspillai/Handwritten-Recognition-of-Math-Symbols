import numpy as np
import pdb
import sys
import operator
import math
import operator
from Symbol import *
from Trace import *
class Graph:
    
    def __init__(self):
        self.dict_adj={}

    def add_edge(self,v, n):
        if v not in self.dict_adj:
            pdb.set_trace()
            self.dict_adj[v]=[]
            self.dict_adj[v].append(n)
        else:
            self.dict_adj[v].append(n)

    def iscycle(self,visited,node,adj_mat_bool,list_cycle_nodes,adj_mat,path_dist):
        #visited=np.zeros(adj_mat_bool.shape[0])
        if visited [node]==1:
            return True,list_cycle_nodes,path_dist
        else:
            if np.sum(adj_mat_bool[node]==True)==0:
                return False,list_cycle_nodes,path_dist
            else:
                visited[node]=True
                list_cycle_nodes.append(node)
                adj_nodes=np.where(adj_mat_bool[node]==True)[0]
                for i in xrange(adj_nodes.shape[0]):
                    path_dist=path_dist+adj_mat[node,adj_nodes[i]]
                    val,list_cycle_nodes,path_dist=self.iscycle(visited,adj_nodes[i],adj_mat_bool,list_cycle_nodes,adj_mat,path_dist)
                    if val==True:
                        if path_dist==0:
                            break
                        else:
                            val=False  # Because it is not a zero cyclic path
                    #substract path dist previously added,as we are going back again.
                    path_dist=path_dist-adj_mat[node,adj_nodes[i]]
                        
                #list_cycle_nodes.remove(node)
                if val==False:
                    visited[node]=val # This is to make the visited nodes unvisited
                    list_cycle_nodes.remove(node)
                return val,list_cycle_nodes,path_dist

    def check_cycle(self,new_adj_mat_bool,adj_mat):
        pdb.set_trace()
        ret=False
        for i in xrange(new_adj_mat_bool.shape[0]):
            path_dist=0
            visited=np.zeros(new_adj_mat_bool.shape[0])
            list_cycle_nodes=[]
            val,list_cycle_nodes,path_dist=self.iscycle(visited,i,new_adj_mat_bool,list_cycle_nodes,adj_mat,path_dist)
            if val==True and path_dist==0:
                ret=True
                break
        return ret,list_cycle_nodes

    def contract (self,new_adj_mat_bool,list_nodes,adj_mat):
        new_Graph_size=(new_adj_mat_bool.shape[0]-len(list_nodes))+1
        new_Graph=np.zeros((new_Graph_size,new_Graph_size))
        new_Graph[:]=sys.maxint
        new_graph_row, new_graph_col=new_Graph.shape
        new_node=0
        # Keep track of original nodes to the new nodes mapping
        dict_mapping_org_new={}
        dict_mapping_new_org={}
        count_map=0
        for it_map in xrange(new_adj_mat_bool.shape[0]):
            if it_map in list_nodes:
                continue
            else:
                dict_mapping_org_new[it_map]=count_map
                dict_mapping_new_org[count_map]=it_map
                count_map=count_map+1
        pdb.set_trace()
        for node in xrange(new_adj_mat_bool.shape[0]):
            col=0
            flag=True
            
            if node in list_nodes:
                arr=np.where(new_adj_mat_bool[node]==True)[0]
                for i in xrange(arr.shape[0]):
                    if arr[i] in list_nodes:
                        continue
                    new_Graph[new_graph_row-1,dict_mapping_org_new[arr[i]]]= adj_mat[node,arr[i]]
                
            else:
                arr=np.where(new_adj_mat_bool[node]==True)[0]
                #for i in xrange(new_adj_mat_bool.shape[0]):
                for i in xrange(arr.shape[0]):
                    if arr[i] in list_nodes:
                        # If there are two patah from A ---> I , then choose the path with minimum distance.
                        if flag==True:
                            val= adj_mat[node,arr[i]]
                            new_Graph[new_node,new_graph_col-1]= val
                            flag=False
                            continue
                        else:
                            if val>new_adj_mat[node,arr[i]]:
                                val=new_adj_mat[node,arr[i]]
                                new_Graph[new_node,new_graph_col-1]= val
                    else:
                        new_Graph[new_node,dict_mapping_org_new[arr[i]]] = adj_mat[node,arr[i]]
                        col=col+1
            #dict_mapping_org_new[node]=new_node  # To keep track of orginal mapping--> new mapping
                new_node=new_node+1

        #new_Graph[sys.maxint]=0
        return new_Graph,dict_mapping_new_org
        
    def Edmond_mst(self,adj_mat):
        dict_shortdist={}
        adj_mat_bool=adj_mat<sys.maxint
        
        for i in xrange(adj_mat.shape[0]):
            arr=np.where(adj_mat_bool[:,i]==True)[0]
            if arr.shape[0]==0:
                dict_shortdist[i]=0
            else:
                dict_shortdist[i]=np.min(adj_mat[:,i][arr])
                
        new_adj_mat=np.zeros(adj_mat.shape)
        
        for i in xrange(adj_mat.shape[0]):      # Going columns wise
            arr=np.where(adj_mat[:,i]<sys.maxint)[0]  # get all the edges poinitng to th vertex
            new_adj_mat[:,i]=sys.maxint
            if arr.shape[0]==0:
                continue
            for j in xrange(arr.shape[0]):
                new_adj_mat[arr[j],i] = adj_mat[arr[j],i]-dict_shortdist[i]

        new_adj_mat_bool=np.zeros(new_adj_mat.shape)
        # make all the values in the adj_mat greater than 0 to False,because we want to zero cycle in a graph.
        #new_adj_mat_bool[new_adj_mat==0]=True
        pdb.set_trace()
        new_adj_mat_bool[new_adj_mat<sys.maxint]=True
        val,list_nodes= self.check_cycle(new_adj_mat_bool,new_adj_mat)
        if val==True:
            new_graph,dict_mapping_new_org=self.contract (new_adj_mat_bool,list_nodes,new_adj_mat)
            print new_graph
            child_uncontract=self.Edmond_mst(new_graph)
            uncontract=np.zeros(new_adj_mat.shape[0])
            for i in xrange(child_uncontract.shape[0]):
                org=dict_mapping_new_org[i]
                arr=np.where(child_uncontract[i]==True)[0]
                if arr.shape[0]!=0:
                    uncontract[org][arr[0]]=True
            list_nodes.append(list_nodes[0]) # adding first to last ---> to get a cycle
            for i in xrange(len(list_nodes)):
                arr=np.where(uncontract[:][i+1]==True)[0]
                if arr.shape[0]!=0:
                    # Do not put an edge this edge from the cycle from i---> i+1
                    continue
                else:
                    uncontract[list_nodes[i]][list_nodes[i+1]]=True
            return uncontract 
        else:
            list_nodes=[]
            #list_nodes.append(0)
            list_nodes_from_root=self.find_path_from_root(new_adj_mat,new_adj_mat_bool,0,list_nodes)
            print list_nodes_from_root
            uncontract=np.zeros(new_adj_mat.shape)
            for i in xrange(list_nodes_from_root)-1:
                uncontract[list_nodes_from_root[i]][list_nodes_from_root[i+1]]=True
            return uncontract
            
       
    def find_path_from_root(self, new_adj_mat,new_adj_mat_bool, node,list_nodes):
        if len(list_nodes)==new_adj_mat.shape[0]:
            return list_nodes
        else:
            
            list_nodes.append(node)
            arr=np.where(new_adj_mat_bool[node]==True)[0]
            if arr.shape[0]==0:
                return list_nodes
            else:
                
                node_val=np.argmin(new_adj_mat[node])
                #list_nodes.append(node_val)
                list_nodes=self.find_path_from_root(new_adj_mat,new_adj_mat_bool, node_val,list_nodes)

            return list_nodes



    def calculate_BAR(self,other_obj,BBC_x,BBC_y):
        #pdb.set_trace()
        vector_v2=np.array([1,0]).reshape(1,2)
        flag=True
        min_val=0
        max_val=0
        points=[]
        for k in xrange(len(other_obj.symbol_list)):
            #total_points+= other_obj.symbol_list[i].poinst_float
            #points = other_obj.symbol_list[k].points_float
            points += other_obj.symbol_list[k].original_points
        points_sorted_y=sorted(points,key=operator.itemgetter(1))
        points_sorted_x=sorted(points,key=operator.itemgetter(0))
        list_points_LOS=[]
        list_points_LOS_val=[]
        #list_points_LOS.append(points_sorted_x[0])
        #list_points_LOS.append(points_sorted_x[-1])
        list_points_LOS.append(points_sorted_y[0])
        list_points_LOS.append(points_sorted_y[-1])
        for pt in list_points_LOS:
            pt_x,pt_y=pt
            vector_v1=np.array([pt_x-BBC_x,pt_y-BBC_y]).reshape(2,1)
            if pt_y>=BBC_y:
                v1=np.sqrt(np.sum(np.square(vector_v1)))
                v2=np.sqrt(np.sum(np.square(vector_v2)))
                val=(vector_v2.dot(vector_v1))/ (v1*v2)
                x_rad=math.acos(val[0,0])
                deg=math.degrees(x_rad)
                list_points_LOS_val.append(deg)                    
            else:
                v1=np.sqrt(np.sum(np.square(vector_v1)))
                v2=np.sqrt(np.sum(np.square(vector_v2)))
                val=(vector_v2.dot(vector_v1))/(v1*v2)
                x_rad=math.acos(val[0,0])
                deg=math.degrees(x_rad)
                deg=360-deg
                list_points_LOS_val.append(deg)
                    
        min_val=min(list_points_LOS_val)
        max_val=max(list_points_LOS_val)
        #pdb.set_trace() 
        return min_val, max_val

    def get_sorted_symbol_list(self,BBC_x,BBC_y,Symbol_list,index):
        #pdb.set_trace()
        dict_sorted_symbols={}
        for i in xrange(len(Symbol_list)):
            if index==i:
                continue
            else:
                other_obj=Symbol_list[i]
                #x_min,x_max,y_min,y_max=other_obj.get_BoundingBox_values()
                x_centroid, y_centroid=other_obj.get_centroids()
                other_BBC_x= x_centroid #(x_min+x_max)/2
                other_BBC_y= y_centroid #(y_min+y_max)/2
                dist=math.pow((BBC_x-other_BBC_x),2) + math.pow((BBC_y - other_BBC_y),2)
                dict_sorted_symbols[other_obj]=dist
        # sort based on distance
        sorted_basedon_eye=sorted(dict_sorted_symbols.items(),key=operator.itemgetter(1))
        #pdb.set_trace()
        return sorted_basedon_eye

    def check_overlap(self,min_BAR,max_BAR,bin_array):
        #pdb.set_trace()
        ret =True
        if (min_BAR>=0 and min_BAR<180) and  (max_BAR>=0 and max_BAR<180):
            for i in xrange(int(min_BAR),int(max_BAR)+1):
                if bin_array[i]==1:
                    ret=False
                    break
                            
            #if any(bin_array[int(min_BAR):int(max_BAR)+1])==0
            if ret==True:
                bin_array[int(min_BAR):int(max_BAR)+1]=1
                ret=True
        elif (min_BAR>=180 and  min_BAR<360)and (max_BAR>=180 and max_BAR<360):
            #if any(bin_array[int(min_BAR):int(max_BAR)+1])==0:
            for i in xrange(int(min_BAR),int(max_BAR)+1):
                if bin_array[i]==1:
                    ret=False
                    break
            if ret==True:
                bin_array[int(min_BAR):int(max_BAR)+1]=1
                ret=True
        elif (min_BAR>=90 and min_BAR <180)  and (max_BAR>=180 and max_BAR<240):
            for i in xrange(int(min_BAR),180):
                if bin_array[i]==1:
                    ret=False
                    break
            for i in xrange(180,int(max_BAR)):
                if bin_array[i]==1:
                    ret=False
                    break
            if ret==True:
                #if bin_array[int(min_BAR):180]==0 and bin_array[180:int(max_BAR)+1]==0:
                bin_array[int(min_BAR):180]=1
                bin_array[180:int(max_BAR)+1]=1
                ret =True
        elif (min_BAR>=0 and min_BAR<90) and (max_BAR>=240 and max_BAR<360) :
            for i in xrange(0,int(min_BAR)):
                if bin_array[i]==1:
                    ret=False
                    break

            for i in xrange(int(max_BAR),360):
                if bin_array[i]==1:
                    ret=False
                    break
                
            #if bin_array[0:int(min_BAR)]==0 and bin_array[int(max_BAR):360]==0:
            if ret==True:
                bin_array[0:int(min_BAR)]=1
                bin_array[int(max_BAR):360]=1
                ret =True
        #pdb.set_trace()
        return ret,bin_array
    
            

    def LineOfSight(self,Symbol_list,relation_classfier_obj):
        '''
        The fucntion contructs a line of sight graph , returning an adjacency matrix
        '''
       
        vector_v2=np.array([1,0])
        # This keeps ytrack of sym_obj-->index, which helps in filling adj_matrix
        dict_mapping_Symbol_index={}  
        N=len(Symbol_list)
        adj_matrix=np.zeros((N,N))
        index_to_symbol={}
        for i in xrange(len(Symbol_list)):
            sym_object_map=Symbol_list[i]
            dict_mapping_Symbol_index[sym_object_map]=i
            index_to_symbol[i]= sym_object_map
             
        for i in xrange(len(Symbol_list)):
            # Start line of sight
            #pdb.set_trace()
            eye_obj=Symbol_list[i]
            bin_array=np.zeros(360)
            dict_eye_to_others_angmap={}
             # Flag: To eneter value in dict dict_eye_to_others_angmap for the 1st time, without checking overlap
            flag=True
            list_sorted_symbols=[]
            #x_min,x_max,y_min,y_max=eye_obj.get_BoundingBox_values()
            x_centroid,y_centroid =eye_obj.get_centroids()    
            BBC_x= x_centroid
            BBC_y = y_centroid
            # get the list of all the symbols and sort them
            sorted_basedon_eye = self.get_sorted_symbol_list(BBC_x,BBC_y,Symbol_list,i)
            # Now calculate BAR angle
            for k in xrange(len(sorted_basedon_eye)):
                sym_obj,dist=sorted_basedon_eye[k]
                min_BAR,max_BAR=self.calculate_BAR(sym_obj,BBC_x,BBC_y)
                flag_val,bin_array=self.check_overlap(min_BAR,max_BAR,bin_array)
                #if self.check_overlap(min_BAR,max_BAR,bin_array):
                if flag_val==True:
                    dict_eye_to_others_angmap[sym_obj]=[min_BAR,max_BAR]
                    other_index=dict_mapping_Symbol_index[sym_obj]
                    eye_index=dict_mapping_Symbol_index[eye_obj]
                    #Need to give pronbaility from relationship classifier
                    #get probability
                    dist_prob=self.get_relation_probability(eye_obj,sym_obj,relation_classfier_obj)
                    adj_matrix[eye_index][other_index]=dist_prob
        #print adj_matrix
        #print "\n*****\n"
        #print index_to_symbol
        return adj_matrix,dict_mapping_Symbol_index,index_to_symbol

    def get_relation_probability(self,eye_obj,other_obj,relation_classfier_obj):
        #pdb.set_trace()
        sym_obj=Symbol()
        trace_list=[]
        total_points=[]
        #First get all the points together
        for i in xrange(len(eye_obj.symbol_list)):
            #trace_list.append(eye_obj.symbol_list[i])
            total_points+=eye_obj.symbol_list[i].original_points 
        for i in xrange(len(other_obj.symbol_list)):
            total_points+=other_obj.symbol_list[i].original_points 
            #trace_list.append(other_obj.symbol_list[i])

        trace_obj=Trace(points_float=total_points)
        trace_obj.normalization()
        trace_list.append(trace_obj)       
        sym_obj.symbol_list=trace_list
        X=sym_obj.get_features()
        X=np.asarray(X)
        prob = np.max(relation_classfier_obj.predict_proba(X.reshape(1,-1)))
        #(1-prob) this is to get the minimum spanning tree.
        trace_list.remove(trace_obj)
        return(1-prob)
        
        
def main():
    g=Graph()
    '''
    g.add_edge(0,1)
    g.add_edge(1,2)
    g.add_edge(2,3)
    g.add_edge(4,1)
    '''
    print g.dict_adj

    adj_mat=np.array([[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,0],[0,1,0,1,0]])
    adj_mat=np.array([[0,0,0,0,1],[1,0,1,0,0],[0,0,0,1,0],[0,0,0,0,0],[0,1,0,0,0]])
    adj_mat=np.array([[0,1,0,0,0,0],[0,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,1,0],[1,0,0,0,0,1],[0,0,0,1,0,0]])
    adj_mat=np.array([[0,5,0,0,0,0,0,11],[0,0,3,0,0,13,0,0],[0,0,0,12,0,9,0,0,],[0,0,0,0,1,0,0,0],[0,0,4,0,0,0,0,0],[0,0,0,0,8,0,7,0],[0,2,0,0,0,0,0,10],[0,6,0,0,0,0,0,0,]])
    adj_mat[adj_mat[:]==0] =sys.maxint   
    print adj_mat
    pdb.set_trace()
    g.Edmond_mst(adj_mat)
    #print sym_object_map
if __name__=="__main__":
    #main()
    pass
