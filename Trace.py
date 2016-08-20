import numpy as np
import math
class Trace:

    '''
    The class is used for preprocessing the raw data.
    '''
    
    def __init__(self,trace_id=None,points_float=None):
        self.trace_id = trace_id
        self.points_float = points_float
        self.original_points = points_float   # This can be used for calculating curvature.
        self.mean_x = 0
        self.mean_y = 0
            
    def normalization(self):
        norm_points_float=[]
        noOfcord = len(self.points_float[0])
        N = len(self.points_float)

        if noOfcord==2:
            sum_x = 0
            sum_y=0
            for i in xrange(N):
                x,y =   self.points_float[i]
                sum_x = sum_x +x
                sum_y = sum_y +y

            mean_x = sum_x/N
            mean_y = sum_y/N

            var_x=0
            var_y = 0
            for i in xrange(N):
                x,y = self.points_float[i]
                var_x = var_x + np.square(x-mean_x)
                var_y = var_y + np.square(y-mean_y)
               
            SD_x = np.sqrt(var_x/N)
            SD_y = np.sqrt(var_y/N)
            
            if SD_x==0:
                SD_x =mean_x
            if SD_y ==0:
                SD_y= mean_y
                
            #print (SD_x,SD_y,var_x,var_y)
            for i in xrange(N):
                x,y = self.points_float[i]
                norm_points_float.append(((x-mean_x)/SD_x,(y-mean_y)/SD_y))       

        self.points_float = norm_points_float
        self.mean_x = mean_x
        self.mean_y = mean_y
        #print "Normalization Done!!!!"
       

        
    def smoothing(self):

        for i in xrange(1,len(self.points_float)):
            x_prev,y_prev = self.points_float[i-1]
            x_curr,y_curr = self.points_float[i]
            x_new = 0.75 * x_prev + 0.25 * x_curr
            y_new = 0.75 * y_prev + 0.25 * y_curr
            a=self.points_float[i]
            list_a = list(a)
            list_a[0]=x_new
            list_a[1] = y_new
            self.points_float[i] = tuple(list_a)
    
    def thinning(self):
        thinning_float_list=[]
        # Insert the first point
        thinning_float_list.append(self.points_float[0])
        i=0
        while i < len(self.points_float):
            flag=False
            print "value of i  :%d" % (i)
            x_curr,y_curr = self.points_float[i]
            for j in xrange(i+1,len(self.points_float)):
                x_next, y_next = self.points_float[j]
                dist_x = math.fabs(x_curr-x_next)
                dist_y = math.fabs(y_curr-y_next)
                if dist_x>=0.15 and dist_y >=0.15:      # 0.15 still gives a better estimate of the symbol.
                    thinning_float_list.append(self.points_float[j])
                    print "Changed!!!!!!"
                    flag = True
                    break

                else:
                    continue
            if flag==True:
                i=j
            else:
                i=i+1

        self.points_float=thinning_float_list


    def get_max_min(self):
        x_max=0
        x_min=0
        y_max = 0
        y_min = 0
        
        for i in xrange(len(self.points_float)):
            x,y = self.points_float[i]
            if x>x_max:
                x_max =x
            elif x<x_min:
                x_min=x
            if y>y_max:
                y_max=y
            elif y<y_min:
                y_min=y

        return x_max,y_max,x_min,y_min
    
    def get_centroids(self):
        x_min,y_min = self.original_points[0]
        x_max,y_max=self.original_points[0]
        for i in xrange(1,len(self.original_points)):
            x,y = self.original_points[i]
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


    def  cal_mean(self):
        mean_x=0
        mean_y=0
        sum_x = 0
        sum_y = 0
        for i in xrange(len(self.points_float)):
            x,y = self.points_float[i]
            
            sum_x =sum_x + x
            sum_y = sum_y + y
            #print ("x= %f, y =%f") % (sum_x,sum_y)

        #print ("sum_x = %f and sum_y= %f"  % (sum_x,sum_y))
        mean_x = sum_x/len(self.points_float)
        mean_y = sum_y / len(self.points_float)

        return  mean_x, mean_y
                
    def shift_origin(self,min_x,min_y,delta):
        points_shift_origin =[]
        x_abs = math.fabs(min_x)
        y_abs = math.fabs(min_y)
        for i in xrange(len(self.points_float)):
            x,y=self.points_float[i]
            x_shift = x+x_abs+delta
            y_shift = y + y_abs+delta
            points_shift_origin.append((x_shift,y_shift))
            
        return points_shift_origin
            
            
        
