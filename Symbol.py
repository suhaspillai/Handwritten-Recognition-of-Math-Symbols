import numpy as np
import matplotlib.pyplot as plt
import math
from  sets import Set
import pdb
from scipy import ndimage
from PIL import Image
from pylab import *
from scipy import misc
class Symbol:
    '''
    The class has methods that extract features from raw data . 
    '''
    symbol_direction={"left":1,"right":2,"up":3,"down":4}
    
    def __init__(self, symbol_id=None, symbol_list=None,symbol_class=None):
        self.symbol_id = symbol_id
        self.symbol_list = symbol_list
        self.symbol_class = symbol_class

    '''
    def symbol_visualize(self):
        symbol_point_list = self.symbol_list[0].points_float
        list_x=[]
        list_y =[]
        for i in xrange(len(symbol_point_list)):
            a = symbol_point_list[i]
            x,y  =a
            list_x.append(x)
            list_y.append(y)

        fig =plt.figure()
        ax = fig.add_subplot(111)
        ncolors = len(plt.rcParams['axes.prop_cycle'])
        ax.plot(list_x,list_y,'o-')
        plt.axis([min(list_x),max(list_x),max(list_y),min(list_y)])
        plt.savefig("Before_Smoothing.png",format='png')
        plt.show()
     '''   

    def symbol_visualize(self,x,y):
        
        #symbol_point_list = self.symbol_list[0].points_float
        list_x=x
        list_y=y
        fig =plt.figure()
        ax = fig.add_subplot(111)
        ncolors = len(plt.rcParams['axes.prop_cycle'])
        ax.plot(list_x,list_y,'o')
        plt.axis([min(list_x)-2,max(list_x)+2,max(list_y)+2,min(list_y)-2])
        plt.axis('off')
        plt.savefig("symbol.png",format='png')
        plt.close()
        #plt.show()

    
        
    #1
    def find_curvature_count(self):
        set_curvature = Set()  # calculate trace for single strokes or a combination of strokes.
        dict_curvature = {}
        trace_list = self.symbol_list
        count=0
        for trace_obj in trace_list:
            original_points = trace_obj.original_points
            '''
            x_max,y_max,x_min,y_min = trace_obj.get_max_min()
            # Need to shift the origin for x and y to get right directions
            points_shift_origin = trace_obj.shift_origin(x_min,y_min,1)
            '''
            for i in xrange(len(original_points)-1):
                if len(set_curvature) < 4:
                    x,y = original_points[i]
                    x_next, y_next = original_points[i+1]

                    if math.fabs(x-x_next)>=math.fabs(y-y_next):
                        #moving in left or right direction.
                        if x_next-x >=0:   # for right in x direction
                            set_curvature.add("right")
                            if "right" not in dict_curvature:
                                dict_curvature["right"] = count
                                count=count+1
                            #set_curvature.add("right")
                            #print "RIGHT--------->>"
                            #print ("x=%f, y=%f, x_next=%f,y_next=%f") % (x,y,x_next,y_next)
                        else:   # moving left in x direction
                            set_curvature.add("left")
                            if "left" not in dict_curvature:
                                dict_curvature["left"] = count
                                count = count +1
                            #print "LEFT <<---------"
                            #print ("x=%f, y=%f, x_next=%f,y_next=%f") % (x,y,x_next,y_next)
                            
                    else :
                        if y_next-y >= 0:   # moving up in y direction
                            set_curvature.add("down")
                            if "down" not in dict_curvature:
                                dict_curvature["down"] = count
                                count = count +1
                            #print "DOWN "
                            #print ("x=%f, y=%f, x_next=%f,y_next=%f") % (x,y,x_next,y_next)
  
                        else:       #moving down in Y direction
                            set_curvature.add("up")
                            if "up" not in dict_curvature:
                                dict_curvature["up"] = count
                                count = count +1
                            #print ("x=%f, y=%f, x_next=%f,y_next=%f") % (x,y,x_next,y_next)
 

                else :
                    break
        # 1: left 2:right 3:up 4:down
        first= 1
        last = 2
        for k in dict_curvature:
            if dict_curvature[k]==0:
                first = self.symbol_direction[k]
            if dict_curvature[k]==len(dict_curvature)-1:
                last = self.symbol_direction[k]
        
                
        return len(set_curvature),first,last
    
    #2    
    def get_aspectRatio(self):
        x_min=0
        x_max=0
        y_min = 0
        y_max = 0
        trace_obj_list = self.symbol_list
        for trace_obj in trace_obj_list:
            x_max_t,y_max_t,x_min_t,y_min_t = trace_obj.get_max_min()
            if x_max_t>x_max:
                x_max = x_max_t
            if y_max_t>y_max:
                y_max = y_max_t
            if x_min_t < x_min:
                x_min = x_min_t
            if y_min_t < y_min:
                y_min =y_min_t

        width = x_max - x_min
        height = y_max - y_min

        if width <= 0:
            width = 0.01
        if height <= 0:
            height = 0.01
            
        aspect_ratio = width/height
        
        #print "x_max= %f, y_max= %f, x_min= %f, y_min= %f" % (x_max,y_max,x_min,y_min)
        return aspect_ratio

    #3
    def get_noOfTraces(self):
        trace_obj_list = self.symbol_list
        count_trace = len(trace_obj_list)
        return count_trace
    #4    
    def get_mean(self):
        mean_x=0
        mean_y=0
        trace_obj_list = self.symbol_list
        for trace_obj in trace_obj_list:
            
            m_x, m_y =  trace_obj.cal_mean()
            #print ("m_x= %d m_y= %d") % (m_x,m_y)
            mean_x = mean_x + m_x
            mean_y = mean_y + m_y

        mean_x = mean_x / len(trace_obj_list)
        mean_y = mean_y/ len(trace_obj_list)
        
        return mean_x,mean_y


    def get_var_covar(self):
        
        trace_obj_list =self.symbol_list
        mean_x,mean_y = self.get_mean()

        var_x = 0
        var_y = 0
        cov_xy=0
        count_points=0
        for trace_obj in trace_obj_list:
            points = trace_obj.points_float
            for pt in points:
                x,y = pt
                var_x+= (x-mean_x)**2
                var_y += (y-mean_y)**2
                cov_xy+=(x-mean_x) * (y-mean_y)
            count_points=count_points + len(points)

        if count_points !=0:
            var_x = var_x/count_points
            var_y = var_y/count_points
            cov_xy = cov_xy/count_points

        return var_x, var_y, cov_xy
         
    #5
    def get_delta_position(self):
        trace_obj_list = self.symbol_list
        delta_all_traces=[]  # This will be used for all the delta across traces in that symbol.
        val_delta_x_sum = 0
        val_delta_y_sum=0
        count_val_delta_sum = 0
        for trace_obj in trace_obj_list:
            #N = len(trace_obj.points_float)-2
            N = len(trace_obj.points_float)
            delta=[] # For every trace store delta features, because point may vary across storkes.
            val_delta_x=0
            val_delta_y=0
            #print ("Value of N is ******* %d") % (N)
            if  N>4:   # Calculate only when the number of points in the tarce greater than or equal to 4
                count_val_delta_sum = count_val_delta_sum + N
                for i in xrange(N):
                    x,y  = trace_obj.points_float[i]   
                    if i <2:
                        x_next,y_next = trace_obj.points_float[i+2]
                        val_delta_x = x_next-x
                        val_delta_y = y_next-y
                        if val_delta_x !=0:
                            delta.append([val_delta_x,val_delta_y])
                        
                    elif i>=N-2:
                        x_prev,y_prev = trace_obj.points_float[i-2]
                        val_delta_x = x-x_prev
                        val_delta_y = y-y_prev
                        if val_delta_x !=0:
                            delta.append([val_delta_x,val_delta_y])
                    else:
                        x_prev,y_prev = trace_obj.points_float[i-2]
                        x_next,y_next = trace_obj.points_float[i+2]
                        val_delta_x = x_next-x_prev
                        val_delta_y = y_next-y_prev
                        if val_delta_x !=0:
                            delta.append([val_delta_x,val_delta_y])

                    # This will sum deltas across all the traces.
                    val_delta_x_sum = val_delta_x_sum + val_delta_x
                    val_delta_y_sum = val_delta_y_sum + val_delta_y 

                delta_all_traces.append(delta)

        #N_trace_obj = len(trace_obj_list)
        #print ("trace objects %d  %d %d") % (N_trace_obj, N,len(trace_obj.points_float) )
        # avearge across all the traces for x.
        if count_val_delta_sum==0:
            count_val_delta_sum = 1
            
        val_delta_x_sum = val_delta_x_sum/(count_val_delta_sum)        
        #average across all the traces for y.
        val_delta_y_sum = val_delta_y_sum / (count_val_delta_sum)

        return val_delta_x_sum,val_delta_y_sum, delta_all_traces 

            
    def get_writing_angle(self,delta_all_traces):
        delta_writ_angle_all_trace=[]
        writ_angle_sum = 0
        N = len(delta_all_traces)
        count_delta_x_y = 0
        for i in xrange(N):
            delta_writ_angle_per_trace=[]
            delta_x_y = delta_all_traces[i]
            #print ("******************")
            #print (len(delta_x_y))
            count_delta_x_y = count_delta_x_y + len(delta_x_y) # count of delta's for that trace. 
            for j in xrange(len(delta_x_y)):
                x = delta_x_y[j][0]
                y=  delta_x_y[j][1]
                #print ("x= %f, y= %f") % (x,y)
                delta_theta = math.atan(y/x)
                writ_angle_sum = writ_angle_sum + delta_theta
                delta_writ_angle_per_trace.append(delta_theta)
            
            delta_writ_angle_all_trace.append(delta_writ_angle_per_trace)
            
        if count_delta_x_y==0:
            count_delta_x_y = 1
        writ_angle_avg = writ_angle_sum /(count_delta_x_y)
        return writ_angle_avg,delta_writ_angle_all_trace
        
    def get_delta_writing_angle(self, delta_writ_angle_all_trace):
        delta_writ_angle_sum=0
        count_writ_theta=0
        N = len(delta_writ_angle_all_trace)
        for i in xrange(N):
            writ_theta = delta_writ_angle_all_trace[i]
            count_writ_theta = count_writ_theta + len(writ_theta) 
            for j in xrange(len(writ_theta)):
                if j==0:
                    continue
                else:
                    theta_prev = writ_theta[j-1]
                    theta_curr = writ_theta[j]
                    delta_writ = theta_curr-theta_prev
                    delta_writ_angle_sum= delta_writ_angle_sum + delta_writ

        if count_writ_theta == 0:
            count_writ_theta = 1
        delta_writ_angle_avg = delta_writ_angle_sum /(count_writ_theta)

        return delta_writ_angle_avg

    def addpoints(self,list_points):
        list_x=[]
        list_y=[]
        N=len(list_points)
        for i in xrange(N-1):
            x,y = list_points[i]
            x_next,y_next=list_points[i+1]
            x_new_pt = (float(x) + float(x_next))/2
            y_new_pt = (float(y) + float(y_next))/2
            list_x.append(x)
            list_x.append(x_new_pt)
            list_y.append(y)
            list_y.append(y_new_pt)
        x,y=list_points[-1]
        list_x.append(x)
        list_y.append(y)
        return list_x,list_y

    '''       
    def get_stroke_to_pixel(self):
        trace_obj_list = self.symbol_list
        points=[]
        # Getting strokes for the entire symbol
        for stroke in trace_obj_list:
            points+=stroke.points_float

        #Add new points to the existing points
        list_x,list_y=self.addpoints(points)
        self.symbol_visualize(list_x,list_y)
        im=array(Image.open('symbol.png').convert('L'))
        #im.save('symbol_dim.png')
        #im = array(Image.open('symbol_dim.png'))
        # Convert to binary
        im = 1*(im<128)
        # filter that window over the and if atleast 1 one is overlapped with the filter and input image, the center pixel is marked one 
        struct1 = ndimage.generate_binary_structure(2,2)
        new_binary=ndimage.binary_dilation(im,structure =struct1,iterations=2).astype(im.dtype)
        #Image resized to 128 * 128
        #size=128,128
        size=64,64
        new_binary_resize=misc.imresize(new_binary,size,interp='bilinear')
        #pdb.set_trace()
        dim = new_binary_resize[0]
        #plt.imshow(new_binary_resize)
        #plt.show()            
        #new_binary_resize=new_binary_resize.reshape(dim*dim)
        new_binary_resize=new_binary_resize.flatten()
        #imshow(new_binary_resize)
        return (new_binary_resize.tolist())
    '''

    def get_stroke_to_pixel(self):
        trace_obj_list = self.symbol_list
        points=[]
        # Getting strokes for the entire symbol
        for stroke in trace_obj_list:
            points+=stroke.points_float

        list_pt_x,list_pt_y=self.addpoints(points)
        #points_add=zip(list_x,list_y)
        xx=np.round(np.linspace(-3,3,720),3)
        yy=np.round(np.linspace(-3,3,526),3)
        #x=np.round(np.asarray(list_x),3)
        #y=np.round(np.asarray(list_y),3)
        new_arr=np.zeros((720,526))
        list_x=[]
        list_y=[]
        for i in xrange(len(list_pt_x)):
            x=list_pt_x[i]
            y=list_pt_y[i]
            l_x= np.argwhere(abs(xx-x)<0.09)
            l_y=np.argwhere(abs(yy-y)<0.09)
            
            if x>=0:
                for k in l_x:
                    if xx[k]>=0:
                        list_x.append(k)
                     
            else:
                for k in l_x:
                    if xx[k]<0:
                        list_x.append(k)
            
            if y>=0:
                for k in l_y:
                    if yy[k]>=0:
                        list_y.append(k)
            else:
                for k in l_y:
                    if yy[k]<0:
                        list_y.append(k)
            
            l_zip = zip(list_x,list_y)
            for pt in l_zip:
                new_arr[pt[0],pt[1]]=1

            l_zip[:]=[]
            list_x[:]=[]
            list_y[:]=[]

        size=64,64
        new_binary_resize=misc.imresize(new_arr,size,interp='bilinear')
        #plt.imshow(new_binary_resize.T)
        # For smoothing, since it is pixelated
        gaussian_im=ndimage.gaussian_filter(new_binary_resize,1)
        return gaussian_im.flatten().tolist() 


    #Get histogram of points feature.

    def hist_of_points(self,split_value):
        '''
        The fucntion extracts histogram of points features
        '''

        #pdb.set_trace()
        
        trace_obj_list = self.symbol_list
        points=[]
        # Getting strokes for the entire symbol
        for stroke in trace_obj_list:
            points+=stroke.points_float

        #print points
        x_min,y_min = points[0]
        x_max,y_max=points[0]
        for i in xrange(1,len(points)):
            x,y = points[i]
            if  x>x_max:
                x_max =x
            elif x<x_min:
                x_min=x
            if y>y_max:
                y_max=y
            elif y<y_min:
                y_min=y

       
        hist_of_points=np.zeros((split_value,split_value))
        x_array=np.linspace(x_min,x_max,split_value+1)
        y_array=np.linspace(y_max,y_min,split_value+1)
        for pt in points:
            x,y=pt
            #pdb.set_trace()
            for i in xrange(split_value):
                
                if i==split_value-1:
                    index_x=i
                    break
                elif x>=x_array[i] and x<x_array[i+1]:
                    index_x=i
                    break
                                
            for j in xrange(split_value):
                if j==split_value-1:
                    index_y=j
                    break
                elif y<=y_array[j] and y>y_array[j+1]:
                    index_y=j
                    break
                    
            hist_of_points[index_x][index_y]=hist_of_points[index_x][index_y]+1

        return hist_of_points.flatten().tolist()


    def get_crossing_line(self):
 
        '''
        The function is used to extract crossing line features
        ''' 
        # Get bounding box points
        trace_obj_list = self.symbol_list
        points=[]
        # Getting strokes for the entire symbol
        for stroke in trace_obj_list:
            points+=stroke.points_float

        #print points
        x_min,y_min = points[0]
        x_max,y_max=points[0]
        for i in xrange(1,len(points)):
            x,y = points[i]
            if  x>x_max:
                x_max =x
            elif x<x_min:
                x_min=x
            if y>y_max:
                y_max=y
            elif y<y_min:
                y_min=y
                
     
        
        list_x=np.linspace(x_min,x_max,6)
        list_y=np.linspace(y_max,y_min,6)

        #print len(list_x),len(list_y)
        list_x_y=[]
        list_x_y.append(list_x)
        list_x_y.append(list_y)
        count=0
        list_crossing_features=[]
        for list_axis in list_x_y:
          
            for i in xrange(len(list_axis)-1):
                
            
                boundary_start=list_axis[i]
                boundary_end = list_axis[i+1]
                #get the 9 lines
                mid_pt = (boundary_start + boundary_end)/2
                left_to_mid=np.linspace(boundary_start,mid_pt,6)
                right_to_mid = np.linspace(mid_pt,boundary_end,6)
                lines_cor=left_to_mid.tolist()[1:-1]
                lines_cor+=right_to_mid.tolist()[0:-1]
                #print "\n lines_corr =%d" % (len(lines_cor))
                if count==0:
                    if x_min-x_max==0:
                        #print "\n Inside min max for x"
                        list_crossing_features+=[0 for i in xrange(3)]
                    else:
                        list_crossing_features +=self.get_crossing_features(lines_cor,points,boundary_start,boundary_end,'x')
               
                    
                else:
                    if y_min-y_max==0:
                        #print "\n Inside min max for y"
                        list_crossing_features+=[0 for i in xrange(3)]
                    else:
                        list_crossing_features+=self.get_crossing_features(lines_cor,points,boundary_start,boundary_end,'y')    
                  
                    
            count=count+1
            
        return list_crossing_features     
                
                
    def get_crossing_features(self,lines_cor,points,boundary_start,boundary_end,var):
        '''
        The fucntion is used to extract crossing features for symbol
        '''

        sum_start_pos=0
        sum_end_pos=0
        count_intersections=0
               
        for cor in lines_cor:
            list_intersection=[]
           
            for i in xrange(len(points)-1):
                x_1,y_1=points[i]
                x_2,y_2=points[i+1]
                if var=='x':
                    x_cordinate=cor
                           #y_cordinate=max_val
                    if x_1>=boundary_start and x_1<boundary_end and x_2>=boundary_start and x_2<boundary_end:
                        if x_cordinate>=x_1 and x_cordinate<=x_2:
                            #print x_1,x_2,(x_2-x_1)

                            if abs(x_2-x_1)<1e-8:
                                y_new= y_1
                                list_intersection.append(y_new)                              
                            else:
                                slope= float((y_2-y_1))/(x_2-x_1)
                                y_new=slope*(x_cordinate-x_1)+ y_1
                                list_intersection.append(y_new)

                            ''''         
                            if abs(slope)<=1e-8:
                                y_new= y_1
                                list_intersection.append(y_new)
                            #find intersection of y
                            else:
                                #slope= float((y_2-y_1))/(x_2-x_1)
                                y_new=slope*(x_cordinate-x_1)+ y_1
                                list_intersection.append(y_new)
                            '''  
                elif var=='y':
                    y_cordinate=cor
                    if y_1<=boundary_start and y_1>boundary_end and y_2<=boundary_start and y_2>boundary_end:
                        if y_1>y_2:
                            if y_cordinate<=y_1 and y_cordinate>=y_2 :
                                if abs(x_2-x_1)<1e-8 or abs(y_2-y_1)<1e-8:
                                    x_new=x_1
                                    list_intersection.append(x_new)
                                else:
                                    slope=float((y_2-y_1))/(x_2-x_1)
                                    x_new=float((y_cordinate-y_1)) * (1/slope) + x_1
                                    list_intersection.append(x_new)
                        else:
                            if y_cordinate >=y_1 and y_cordinate <=y_2:
                                if abs(x_2-x_1)<1e-8 or abs(y_2-y_1)<1e-8:
                                    x_new=x_1
                                    list_intersection.append(x_new)
                                else:
                                    slope=float((y_2-y_1))/(x_2-x_1) 
                                    x_new=float((y_cordinate-y_1)) * (1/slope) + x_1
                                    list_intersection.append(x_new)

                   #pdb.set_trace()
            if len(list_intersection)!=0:
                sum_start_pos=sum_start_pos+list_intersection[0]
                sum_end_pos=sum_start_pos+list_intersection[-1]
                count_intersections = count_intersections + len(list_intersection)
               
        #pdb.set_trace()
        avg_start_pos=float(sum_start_pos)/9
        avg_end_pos=float(sum_end_pos)/9
        avg_count_intersections=float(count_intersections)/9
        list_avg=[avg_start_pos,avg_end_pos,avg_count_intersections]
        #print len(list_avg)
        #print "\n"
        return list_avg
        
            
    def get_BoundingBox_values(self):
        '''
        The fucntion gets the bounding box value for a symbol
        '''
        trace_obj_list = self.symbol_list
        points=[]
        # Getting strokes for the entire symbol
        for stroke in trace_obj_list:
            points+=stroke.points_float

        #print points
        x_min,y_min = points[0]
        x_max,y_max=points[0]
        for i in xrange(1,len(points)):
            x,y = points[i]
            if  x>x_max:
                x_max =x
            elif x<x_min:
                x_min=x
            if y>y_max:
                y_max=y
            elif y<y_min:
                y_min=y
                
        return x_min,x_max,y_min,y_max


    def get_centroids(self):
        '''
        The function gets centroid values
        '''
        #pdb.set_trace()
        trace_obj_list = self.symbol_list
        points=[]
        # Getting strokes for the entire symbol
        for stroke in trace_obj_list:
            points+=stroke.original_points
        
        x_min,y_min = points[0]
        x_max,y_max=points[0]
        for i in xrange(1,len(points)):
            x,y = points[i]
            if x>x_max:
                x_max =x
            elif x<x_min:
                x_min=x
            if y>y_max:
                y_max=y
            elif y<y_min:
                y_min=y

        #print x_min,y_min,x_max,y_max
        
        x_centroid=(float(x_min)+float(x_max))/2
        y_centroid=(float(y_min)+float(y_max))/2
        return (x_centroid, y_centroid)

    
    def get_features(self):
        features=[]
        # Get curavature_count
        count_curvature,left_dir,rigth_dir = self.find_curvature_count()
        features += [count_curvature,left_dir,rigth_dir]
        
        '''    
        # Just taking the frst direction.
        for k in dict_curvature:
            if dict_curvature[k]==0:
                features +=[k]
        '''
        
        # Get aspect ratio
        aspect_ratio = self.get_aspectRatio()
        features += [aspect_ratio]

        #Get no of traces 
        
        noOfTrace = self.get_noOfTraces()
        features+=[noOfTrace]

        # Get delta_x and delta_y positions
        delta_x,delta_y,delta_list_across_traces = self.get_delta_position()
        features +=[delta_x,delta_y]

        # Get avearge writing angle

        writ_angle,delta_writ_angle_all_trace = self.get_writing_angle(delta_list_across_traces)
        features += [writ_angle]

        # Get delta writing angle
        delta_writ_angle = self.get_delta_writing_angle(delta_writ_angle_all_trace)
        features += [delta_writ_angle]

        # get covariance between x and y points
        var_x,var_y,cov_xy = self.get_var_covar()
        features += [cov_xy]

        
        #get histogram of points
        features+=self.hist_of_points(4)

        #get crossing line
   
        features+=self.get_crossing_line()
      
        return features
                    
        
            

    
