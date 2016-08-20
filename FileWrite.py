import numpy as np
from os.path import basename
from os.path import dirname
from os.path import stat
from os import mkdir
from os import remove
from os import path
import subprocess
import shutil
import pdb

class FileWrite:
    '''
    This class is used to write lg files.
    
    '''
    dir_name_new=""
    
    def __init__(self,str_dir_name):
            
        dir_name ="." #dirname(filePath)
        dir_name_new = str_dir_name+'_lg_files'
        dir_name_new2=str_dir_name+'_lg_files_out'
        if path.exists(dir_name_new):
   
                shutil.rmtree(dir_name_new)
        if path.exists(dir_name_new2):
                shutil.rmtree(dir_name_new2)
        try:
            
            stat(dir_name_new)
            stat(dir_name_new2)
        except:
            mkdir(dir_name_new)
            mkdir(dir_name_new2)
        

    #def write_to_lg(self,predict_labels,fileName,symbol_list,count_traces,str_task):
    def write_to_lg(self,predict_labels,fileName,symbol_list,dict_map_rel_to_syms,count_traces,str_task):
        #print 'Inside write file\n'
        #pdb.set_trace()
        '''
        The function is used to write lg files.
        '''
        dict_map_temp={}
        fileName_lg=basename(fileName)
        dot_pos = fileName_lg.find(".")
        fileName_lg=fileName_lg[:dot_pos]+".lg"
        dir_name=str_task+"_lg_files"
        f = open(dir_name+"/"+fileName_lg,"w") 
        f.write('# IUD '+fileName+'\n'+'# [ OBJECTS ]')
        f.write('\n# Primitive Nodes (N): '+str(count_traces))
        f.write('\n#    Objects (O): '+ str(len(symbol_list)))
        dict_map_of_symbol_count={}
        for i in xrange(len(symbol_list)):
            label=predict_labels[i]
            if label in dict_map_of_symbol_count:
                dict_map_of_symbol_count[label]=dict_map_of_symbol_count[label]+1
            else:
                dict_map_of_symbol_count[label]=0
                dict_map_of_symbol_count[label]=dict_map_of_symbol_count[label]+1
            sym_class= predict_labels[i]
            
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


        # For Relationship
        for k in  dict_map_rel_to_syms:
            rel=k
            list_rel= dict_map_rel_to_syms[k]
            for val in list_rel:
                #eye_obj,other_obj=dict_map_rel_to_syms[k].pop()
                eye_obj,other_obj=val
                eye_label=dict_map_temp[eye_obj]
                other_label=dict_map_temp[other_obj]
                f.write(' \nR, '+ eye_label +', '+other_label+', '+rel+', '+'1.0')

        #pdb.set_trace()
        f.close() 

#        dir_name_new2=str_task+"_lg_files_out"           
#        subprocess.Popen(['/home/suhaspillai/PatternRecog/Assignments/Project1/lgeval/bin/crohme2lg',fileName,'./'+dir_name_new2+"/"+fileName_lg])







