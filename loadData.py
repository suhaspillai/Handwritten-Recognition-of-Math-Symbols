import numpy as np
from BeautifulSoup import *
from Symbol import *
from Trace import *
import pdb
class loadData:
    def __init__(self):
        pass

    def loadInkml(self, fileName):
        '''
        The function is used to loadinkml files
        '''

        soup = BeautifulSoup(open(fileName))
           # To store floating point numbers
        points_str=[]       
        trace_obj_dict={}
        # Will loop through all the trace ids
        loop_all_trace = soup.findAll('trace')
        for trace in loop_all_trace:
            points_float = []
            points_str = trace.string.strip("\n").split(",")
            for  pt in points_str:
                coord = pt.split()
    
                points_float.append((float(coord[0]),float(coord[1])))

            # Gets the trace id  
            trace_id = int(trace.attrs[0][1])
            #print ("\nTrace id is : %s") % (trace_id)
            #print ("****************************")
            #print (points_float)
            # create an object for every trace and store in the dictionary
            trace_object=Trace(trace_id,points_float)
            
            trace_obj_dict[ trace_id] = trace_object
            trace_object.normalization()    
            
        return soup,trace_obj_dict

    def  get_symbol(self,soup,trace_obj_dict):
        '''
        The function is used to get symbol information and traceObjects
        '''
        flag = True    # to ignore the first tracegroup
        symbols=[]    # This will store the objects for every symbol in the file.  
       # pdb.set_trace()
        for  trace_g in soup.findAll('tracegroup'):
            if flag==True:
                flag=False
                continue
            else:
                symbol_class = trace_g.annotation.text
                # Get symbol id
                try:
                    symbol_id = int(trace_g.attrs[0][1])
                except:
                    #Conversion to int nit possible
                    symbol_id = int(trace_g.attrs[0][1].split(":")[0])

                symbol_list=[]
                #print ("Symbol_id: %d" ) % (symbol_id)
                trace_view = trace_g.findAll('traceview')
                #pdb.set_trace()
                for i in xrange(len(trace_view)):
                    trace_id = int(trace_view[i]['tracedataref'])
                    symbol_list.append(trace_obj_dict[trace_id])

                symbol_obj = Symbol(symbol_id,symbol_list,symbol_class)

            symbols.append(symbol_obj)

        return symbols    
                    

    def  get_symbol_test(self,soup,trace_obj_dict):
        flag = True    # to ignore the first tracegroup
        symbols=[]    # This will store the objects for every symbol in the file.  
       # pdb.set_trace()
        for  trace_g in soup.findAll('tracegroup'):
            if flag==True:
                flag=False
                continue
            else:
              #  symbol_class = trace_g.annotation.text   # No ground truth available
                # Get symbol id
                try:
                    symbol_id = int(trace_g.attrs[0][1])
                except:
                    #Conversion to int nit possible
                    symbol_id = int(trace_g.attrs[0][1].split(":")[0])

                symbol_list=[]
                #print ("Symbol_id: %d" ) % (symbol_id)
                trace_view = trace_g.findAll('traceview')
                #pdb.set_trace()
                for i in xrange(len(trace_view)):
                    trace_id = int(trace_view[i]['tracedataref'])
                    symbol_list.append(trace_obj_dict[trace_id])

                symbol_obj = Symbol(symbol_id,symbol_list,symbol_class)

            symbols.append(symbol_obj)

        return symbols    



                

if __name__=="__main__":
    loadInkml()
else:
    pass
