from sets import Set
from loadData import *
import pdb


class Distribution_Symbol:
        '''
        Class Distribution_symbols distributes symbols into 2/3 training and 1/3 testing,based on spliting training and test files.
        '''
        def __init__(self):
                pass
        
        def get_file_symbol_info(self,set_of_files):
                '''
                The function is used for calculating symbol count  
                '''
                count_no_files = len(set_of_files)
                load_obj = loadData()
                dict_sym_to_file={}
                count_sym={}
                total_count_sym=0
                for fileName in set_of_files:   
                    root_obj, trace_obj_dict = load_obj.loadInkml(fileName)    
                    symbols = load_obj.get_symbol(root_obj,trace_obj_dict)

                    for sym_obj in symbols:
                        sym_id=sym_obj.symbol_class
                        if  sym_id in dict_sym_to_file:
                            dict_sym_to_file[sym_id].append(fileName)
                        else:
                           dict_sym_to_file[sym_id]=[]
                           dict_sym_to_file[sym_id].append(fileName)
                            
                        if sym_id in count_sym:
                            count_sym[sym_id] = count_sym[sym_id] + 1
                        else:
                            count_sym[sym_id] = 1
                        total_count_sym = total_count_sym +  1
                        
                return  dict_sym_to_file,count_sym,total_count_sym   
        
        def get_symbol_distribution(self,path):
                '''
                The function calculates symbol distribution.
                '''
                load_obj = loadData()
                count = 0
                list_all_files=[]
                count_break=0          
                filePath=path
                count=0 
                with open(filePath,"r") as file_read:
                        counnt=count+1
                        for line in file_read:
                                fileName = "/home/sbp3624/PatternRecog/TrainINKML_v3/" + line
                                fileName = fileName.strip('\n')
                                list_all_files.append(fileName)
                                count_break=count_break+1
                     

                dict_file_to_sym_train={}
                count_no_files = len(list_all_files)
                count=0
                train_count = 2*count_no_files/3
                test_count = count_no_files - train_count
                train_file = Set()
                test_file=Set()
                
                for i in xrange(count_no_files):
                        if i < train_count:
                                fileName = list_all_files[i]
                                train_file.add(fileName)
                        else:
                                fileName = list_all_files[i]    
                                test_file.add(fileName)
                        

                dict_sym_to_file_train={}
                dict_sym_to_file_test={}
                count_train={}
                count_test={}
                total_no_sym_train=0
                total_no_sym_test=0
                no_iter = 0
                   

                while (no_iter<30):
                        print "Start iteration = %d" % (no_iter)
                        dict_sym_to_file_train,count_train,total_no_sym_train = self.get_file_symbol_info(train_file)
                        dict_sym_to_file_test,count_test,total_no_sym_test = self.get_file_symbol_info(test_file)
                        
                        count_terminate = 0            
                        for sym in count_test:
                                if sym in count_train:
                                        prob_sym_train =float (count_train[sym])/(count_train[sym]+count_test[sym])
                                        prob_sym_test = 1-prob_sym_train

                                        if prob_sym_train>=0.63 and prob_sym_train <0.67 and prob_sym_test >=0.3 and prob_sym_test<=0.4 :
                                                count_terminate = count_terminate + 1
                                        else:
                                                if prob_sym_train>0.67:
                                                        #swap file from train -> test
                                                        random_no= np.random.randint(len(dict_sym_to_file_train[sym])) 
                                                        swap_file = dict_sym_to_file_train[sym][random_no]    # randomly choose a file that belong to a symbol
                                                        #Remove that file from all the symbols
                                                        train_file.discard(swap_file)
                                                        #add this to the test dict
                                                        test_file.add(swap_file)
                                                else:
                                                        random_no= np.random.randint(len(dict_sym_to_file_test[sym]))
                                                        swap_file = dict_sym_to_file_test[sym][random_no]
                                                        test_file.discard(swap_file)
                                                        train_file.add(swap_file)
                        no_iter = no_iter + 1                            


                 #Write the distribution of files in a txt

                file_write = open("split_files.txt","a")
                file_write.write("Train Files \n")
                file_write.write(str(train_file))
                file_write.write("\n")
                file_write.write("Test Files \n")
                file_write.write(str(test_file))
                
                file_prob_dist=open('file_prob_dist.txt','a')        
                # get the distribution
                dict_sym_to_file_train,count_train,total_no_sym_train = self.get_file_symbol_info(train_file)
                dict_sym_to_file_test,count_test,total_no_sym_test = self.get_file_symbol_info(test_file)
                    
                
                for sym in count_test:
                        if sym in count_train:
                            prob_sym_train =float(count_train[sym])/(count_train[sym]+count_test[sym])
                            prob_sym_test = 1-prob_sym_train
                            file_prob_dist.write(sym+"    "+str(prob_sym_train)+"    "+str(prob_sym_test)+"\n")
                            

                file_write.close()
                file_prob_dist.close()
                return train_file,test_file

if __name__=="__main__":
        pass





