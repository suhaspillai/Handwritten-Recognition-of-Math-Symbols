import numpy as np
import pdb
import csv
import math
from  sets import Set

class Node:
     # Keeps a mapping of discrete and continuos attributes
    
    def __init__(self, best_attribute,best_list_set,best_attribute_val_dict,best_prob_distribution):
        self.best_attribute = best_attribute
        self.best_list_set=best_list_set
        self.best_attribute_val_dict=best_attribute_val_dict
        self.leafnode  =False
        self.newnode =None
        self.prob_distribution= best_prob_distribution
        
class DescisionTree:
    noOfClasses =2
    dict_discrete_continous={}
    def __init__(self):
        pass

    def cal_entropy(self,data):
        #pdb.set_trace()
        set_class_values=Set()
        #N=data.shape[0]
        N=len(data)
        for i in xrange(len(data)):
            set_class_values.add(data[i][-1])   # This will add distinct values for numbers of ground truth

        entropy=0

        for set_val in set_class_values:
            class_label_count=0
            for k in xrange(len(data)):
                if (data[k][-1] == set_val):
                    class_label_count =class_label_count+1
                #class_label_count=np.count(data[:,-1]== set_class_values[j])
            val = float(class_label_count)/N
            #pdb.set_trace()
            entropy = entropy - (val * math.log(val,2))
            
        return entropy

    # This will check whether the data belongs to same class or not ,if yes, then this will return true.
    def check_distribution(self,data):
        #pdb.set_trace()
        #val = data[0,-1]
        if bool(data)==False:           # To check if the dictionary is empty or not
            return False,None
        else:
            val = data[0][-1]
            count=0
            for k in xrange(len(data)):
                if (data[k][-1] == val):
                    count =count+1
            #count = np.sum(data[data[:,-1]==val])
            #if (count==data.shape[0]):
            if (count==len(data)):
                return True,val       # This will also give the label, which will be used of testing
            else:
                return False,None


     # Get the probability distribution for that node
    def get_prob_distribution(self,data):
        dict_prob = {i:0 for i in xrange(self.noOfClasses)}
        for  j in xrange(len(data)):
            class_val = data[j][-1]
            if class_val in dict_prob:
                dict_prob[class_val] = dict_prob[class_val] + 1
            
        #print (dict_prob)         
        list_prob=[]
        for k in dict_prob:
            list_prob.append(float(dict_prob[k]) / len(data))

        del dict_prob       # Free memory  
        return list_prob
             
         
     #Suppose you have a dictionary for continuous attributes   
    def  cal_gain_ratio(self,data,column):
        
        #Calculate Entropy of the full set
        list_sets=[]  # this will be used to store data when split on a particular attribute
        check_distribution={}
        Info_T = self.cal_entropy(data)
        prob_distribution= self.get_prob_distribution(data)
        #N = data.shape[0]
        N=len(data)
        val_to_split_cont=0
        
        #For now just take the discrete values , leave the continuous values
        #For continuous /  values
        #if  dict_discrete_continous
        if self.dict_discrete_continous[column]=="continuous":
            
            cal_to_split_cont=None
            #if isinstance(data[0,column],int) or  isinstance(data[0,column],float):
            sorted_column = [data[i][column] for i in data ]
            #sorted_column = np.sort(data[:,column])
            sorted_column.sort()
            
            if (N%2==0 or N==1):  # for even len
                #print ("***%d") % (N)
                mid_point = (sorted_column[N/2] )
            else:
                #print ("***%d") % (N)
                mid_point = (sorted_column[N/2] + sorted_column[(N/2+1)])/2
            
            for iter_i in xrange(len(sorted_column)):
                if sorted_column[iter_i]>mid_point:
                    cal_to_split_cont=sorted_column[iter_i-1]   # Since, it is already sorted, take the value less than the midpoint but greater than all others till midpoint.
                    break
                
            if cal_to_split_cont is None:
                cal_to_split_cont = mid_point
            #split_function=lambda data:data[column]>=cal_to_split_cont
            #sub_data_1 = data[data[:,column]<=cal_to_split_cont]
            #sub_data_2 = data[data[:,column]>cal_to_split_cont]
            sub_data_1={}
            sub_data_2={}
            count_1=0
            count_2 = 0
            
            for k in xrange(len(data)):

                if data[k][column]<=cal_to_split_cont:
                    sub_data_1[count_1] = data[k]
                    count_1=count_1+1
                else:
                    sub_data_2[count_2] = data[k]
                    count_2=count_2+1
            list_sets.append(sub_data_1)
            list_sets.append(sub_data_2)
            #pdb.set_trace()
            check_distribution["<="] = [self.check_distribution(sub_data_1), Node,cal_to_split_cont]
            check_distribution[">"] = [self.check_distribution(sub_data_2), Node,cal_to_split_cont] # if all samples belong to same class then it will return true. 
            sub_data_temp=(sub_data_1,sub_data_2)
            Info_x=0
            split_ratio=0
            #pdb.set_trace()
            for i in [0,1] :
                
                sub_data=sub_data_temp[i]
                if bool(sub_data) ==False:     # Fpr dictionary being empty
                    break
                else:
                    #sub_N = sub_data.shape[0]
                    sub_N = len(sub_data)
                    sub_N_val = float(sub_N)/float(N)
                    Info_x = Info_x+ (sub_N_val*self.cal_entropy(sub_data))
                    split_ratio = split_ratio +   (-sub_N_val) * math.log((sub_N_val),2)  # calculating split ratio
        # calculate Split ratio
                
            gain = (Info_T-Info_x)
            if  split_ratio==0:
                gain_ratio=gain
            else:
                gain_ratio = gain/split_ratio

         # For nominal values   
        else:
            # Now entropy for the attribute i.e Inof(x)
            #set_distinct_values=(row[column] for row in data)
            set_distinct_values=Set(data[row][column] for row in data)  # This should give distinct values
            Info_x=0
            split_ratio=0
            for col_val in set_distinct_values:
                sub_data={}   
                #check this
                #sub_data = data[data[:,column]==col_val]
                count_sub_data=0
                for k in data:
                    if data[k][column]==col_val:
                        sub_data[count_sub_data] = data[k]
                        count_sub_data = count_sub_data + 1
                        
                check_distribution[col_val] = [self.check_distribution(sub_data), Node]   # if all samples belong to same class then it will return true.
                list_sets.append(sub_data)
                #sub_N = sub_data.shape[0]
                sub_N = len(sub_data)   
                #pdb.set_trace()
                sub_N_val = float(sub_N)/float(N)
                Info_x = Info_x+ (sub_N_val*self.cal_entropy(sub_data))
                split_ratio = split_ratio +   (-sub_N_val) * math.log((sub_N_val),2)     # calculating split ratio
                del sub_data
                
            gain = (Info_T-Info_x)
            if split_ratio==0:
                gain_ratio = gain
            else:
                gain_ratio = gain/split_ratio

        return gain_ratio,list_sets,check_distribution,prob_distribution                 
        #return gain_ratio
        
    #####################BuildTree######################
    def  buildDecisionTree(self,data,list_col):
        #list_col ---> this will contain the column which we have to split the data on     
        #pdb.set_trace()
        if(len(list_col)==0 or len(data)==0):    # Till all the attributes / columns are checked
            node = Node(None,None,None,None)
            node.leafnode = True
            return node
        else:
            #pdb.set_trace()
            list_col_gng_frwd=[]
            best_gain_ratio=0
            best_attribute=None
            best_list_set=[]
            best_attribute_val_dict={}    # For chrcking at test time which attribute is true and which one is false
            best_prob_distribution=[]
            #cols = len(data)-1
            for column in list_col:
                
                list_col_gng_frwd.append(column)
                gain_ratio,list_sets,check_distribution, prob_distribution = self.cal_gain_ratio(data,column)
                if gain_ratio>=best_gain_ratio:
                    best_gain_ratio=gain_ratio
                    best_attribute=column
                    best_list_set=list_sets
                    best_attribute_val_dict = check_distribution
                    best_prob_distribution = prob_distribution       
                del list_sets                       # free memory
                del check_distribution     # free memory
                       
             # so that we do not calculate gain ratio for this attribute,becuase we have split the node on that attribute   
            list_col_gng_frwd.remove(best_attribute)  
            #pdb.set_trace()    
            # call the next subtree
           # pdb.set_trace()
            node = Node(best_attribute,best_list_set,best_attribute_val_dict,best_prob_distribution)
            if self.dict_discrete_continous[best_attribute]=="continuous":
                for i,j in enumerate (best_attribute_val_dict):
                    #check_bool_distribution this is a (tuple,function)
                    check_bool_distribution,newnode,threshold=best_attribute_val_dict[j]
                    if(check_bool_distribution[0]==False):
                        #node = Node(best_attribute,best_list_set,best_attribute_val_dict)
                        newdata = best_list_set[i]
                        if len(newdata)!=0:
                            newnode = self.buildDecisionTree(newdata,list_col_gng_frwd) #making change
                            best_attribute_val_dict[j][1]=newnode
                        else:
                            best_attribute_val_dict[j][1]=None
                    #    node.leafnode=False
                    
                    else:
                        list_gng_frwd_leaf=[]
                        newdata = best_list_set[i]
                        newnode = self.buildDecisionTree(newdata, list_gng_frwd_leaf)
                        best_attribute_val_dict[j][1] = newnode
                    
            else:
                for i,j in enumerate (best_attribute_val_dict):
                    check_bool_distribution,newnode=best_attribute_val_dict[j]
                    if(check_bool_distribution[0]==False):
                        newdata = best_list_set[i]
                        if len(newdata)!=0:
                            newnode = self.buildDecisionTree(newdata,list_col_gng_frwd)
                            best_attribute_val_dict[j][1]=newnode
                        else:
                            best_attribute_val_dict[j][1]=None
                    #    node.leafnode=False
                    else:
                        list_gng_frwd_leaf=[]
                        newdata = best_list_set[i]
                        newnode = self.buildDecisionTree(newdata, list_gng_frwd_leaf)
                        best_attribute_val_dict[j][1] = newnode
            return node
                    
    def classify(self,test_data,ROOT):
        pred_label={i:0 for i in xrange(len(test_data))}
        #pdb.set_trace()
        
        for i in xrange(len(test_data)):
            root=ROOT
            flag=True
            row = test_data[i]
            #print (row)
           # print ("ietartion", i)
           # while root.newnode not None:     # need to change this
            while bool(root.best_attribute_val_dict) :
                #pdb.set_trace()
                col = root.best_attribute
                if self.dict_discrete_continous[col]=="continuous":
                    check_distribution, new_root, threshold = root.best_attribute_val_dict["<="]
                    if row[col]<=threshold:
                        if check_distribution[0]==True:
                            pred_label[i] = check_distribution[1]
                            flag=False
                            break
                        else:
                            prob_distribution = root.prob_distribution
                            root=new_root
                            # Now get the probability distribution of the node.
                            if root is None:
                                class_val = prob_distribution.index(max(prob_distribution))  # give the prob of max lass in thsoe samples
                                pred_label[i] = class_val  # Class label for the class with highest probability distribution
                                break
                    else:
                        check_distribution, new_root, threshold = root.best_attribute_val_dict[">"]
                        if check_distribution[0]==True:
                            pred_label[i] = check_distribution[1]
                            flag=False
                            break
                        else:
                            prob_distribution = root.prob_distribution
                            root=new_root
                            # Now get the probability distribution of the node, since its a null node
                            if root is None:
                                class_val = prob_distribution.index(max(prob_distribution))  # give the prob of max lass in thsoe samples
                                pred_label[i] = class_val  # Class label for the class with highest probability distribution
                                break
                 # For discrete attributes           
                else:
                    val = row[col]
                    key_val=None
                    check_flag =False
                    #Check if the key is there or not in the dictionary
                    for key in root.best_attribute_val_dict:
                        if key == val:
                            check_flag=True
                            break
                        else:
                            check_flag = False
                            key_val=key
                    
                    prob_distribution = root.prob_distribution       
                    # If the key is there then check the distribution
                    if check_flag==True:
                        check_distribution,new_root=root.best_attribute_val_dict[val]  # How will you check for continuos attribute
                        if check_distribution[0]==True:
                            pred_label[i]= check_distribution[1]
                            flag=False
                            break
                        else:
                            
                            root=new_root
                            # Now get the probability distribution of the node, since its a null node
                            if root is None:
                                class_val = prob_distribution.index(max(prob_distribution))  # give the prob of max lass in thsoe samples
                                pred_label[i] = class_val  # Class label for the class with highest probability distribution
                                break
                    # If the key is not there, then just take the class with highest distribution for taht node and retun        
                    else:
                        class_val = prob_distribution.index(max(prob_distribution))  # give the prob of max lass in thsoe samples
                        pred_label[i] = class_val  # Class label for the class with highest probability distribution
                        break
                
                    
                    
        return pred_label
    #def train(self, training_set):

    #def  classify(self, test_instance):

def run_decision_tree():
    # Load the dataset
    
    with open("hw4-task1-data.csv","r") as file_csv:
        data = [tuple(line) for line in csv.reader(file_csv,delimiter="\t")]
    print "Total number of entries: %d" % (len(data))    

    d_tree = DescisionTree()
    
    d_tree.dict_discrete_continous[0]="continuous "       #age
    d_tree.dict_discrete_continous[1]="discrete "             #workclass
    d_tree.dict_discrete_continous[2]="continuous "       #fnlwgt
    d_tree.dict_discrete_continous[3]="discrete "             #education
    d_tree.dict_discrete_continous[4]="continuous"            #education-num
    d_tree.dict_discrete_continous[5]="discrete "             #martial -status
    d_tree.dict_discrete_continous[6]="discrete"              #occupation
    d_tree.dict_discrete_continous[7]="discrete"              #relationship
    d_tree.dict_discrete_continous[8]="discrete "            #race
    d_tree.dict_discrete_continous[9]="discrete "            #sex
    d_tree.dict_discrete_continous[10]="continuous "       #capital gain
    d_tree.dict_discrete_continous[11]="continuous"      #capitalloss 
    d_tree.dict_discrete_continous[12]="continuous"     #hrs per week
    d_tree.dict_discrete_continous[13]="discrete "# native-country
    d_tree.dict_discrete_continous[14]="discrete"
    
    #print data[0]
    columns=15
    X_train = {}
    y_train = {}

    for i in xrange(len(data)):
        line = data[i][0]
        list_data=[]
        line_list = line.split(",")
        if line_list[-1]==">50K":
            y_train[i]=1
        else:
            y_train[i] = 0
        for j in xrange(columns-1):
           
            if d_tree.dict_discrete_continous[j] =="continuous":
                list_data.append(float(line_list[j]))
            else:
                list_data.append(line_list[j])
        list_data.append(y_train[i])
        X_train[i]=list_data


    list_col = len(X_train[0]) -1
    print (list_col)
    #print (type(X_train))
    sample_dict = {i: X_train [i] for i in xrange(10) }
    #print (sample_dict)
    #d_tree.buildDecisionTree(sample,list_col)

        
if __name__=="__main__":
    run_decision_tree()
