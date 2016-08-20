from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import sys
from sets import Set
import pdb

class Classifier:
    ''' Classifier class with Linear classifier and KNearest Neighbor'''
    
    def __init__(self):
        pass
    
                
    #def  predict_KNN (self,data, k_1=1):
    def predict_KNN(self, X_train, y_train, X_test,y_test,k_1):
        
        '''
        Predict classification rate using KNN

        Input : Training data

       '''
        
        X_train = X_train
        y_train = y_train
        X_test = X_test
        y_test= y_test     
        y_predict = np.zeros(y_test.shape)
        K = k_1
        for i in xrange(X_test.shape[0]):
            print "Starting sample %d" % (i)
            x = X_test[i]
            dict_dist = {}
            list_dist=[]
            for j in xrange(X_train.shape[0]):
                dist = np.sum(np.square(x-X_train[j]))
                dict_dist[dist] = j
                list_dist.append(dist)
               
            list_dist.sort()
            count_1 = 0
            for k1 in xrange(K):
                val = list_dist[k1]
                pos = dict_dist[val]
                y_predict[i] = y_train[pos]
        print 'Classification rate for KNN_%d : %f percent' % (K,np.mean(y_predict==y_test)*100)


    def KNN_fast(self, X_train, y_train, X_test,k_1):
        X_train = X_train
        y_train = y_train
        X_test = X_test
    
        y_predict = np.zeros(X_test.shape[0])
        K = k_1
        dist_test = np.zeros((X_test.shape[0],X_train.shape[0]))
        for i in xrange(X_test.shape[0]):
            if i%1000==0:
                print "Starting sample %d" % (i)
            x_test= X_test[i]
            dist = np.sum(np.square(X_train-x_test),axis=1)
            dist_test[i]=dist

        min_dist = dist_test.argmin(axis=1)
        y_predict = y_train[min_dist]
        return y_predict 

    
    def Confusion_Matrix(self,yPredict, classifier_name):
        '''
        Plot Confusion matrix

        Input :
        yPredict: Predict labels for the training data
        classifier_name: Name of the classifier
        '''
        y_predict = yPredict
        b_right = np.sum(y_predict[:100]==0)
        b_wrong = 100-b_right
        o_right = np.sum(y_predict[100:]==1)
        o_wrong = 100-o_right
        print 'Display Confusion Matrix'
        confusion_matrix  = np.array([[o_right,o_wrong],[b_wrong,b_right]])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        width, height = confusion_matrix.shape
        ax.matshow(confusion_matrix,cmap=plt.cm.Blues)
        ax.set_xlabel('Predicted',fontsize=25)
        ax.set_ylabel('Actual',fontsize=25)
        labels=['Orange','Blue']
        ax.annotate(str(confusion_matrix[0][0]),(0,0),fontsize=25)
        ax.annotate(str(confusion_matrix[0][1]),(1,0),fontsize=25)
        ax.annotate(str(confusion_matrix[1][0]),(0,1),fontsize=25)
        ax.annotate(str(confusion_matrix[1][1]),(1,1),fontsize=25)
        labels_x = [item.get_text() for item in ax.get_xticklabels()]
        labels_x[2]='Blue'
        labels_x[1]='Orange'
        ax.set_xticklabels(labels_x,fontsize=15)
        labels_y = [item.get_text() for item in ax.get_yticklabels()]
        labels_y[2]='Blue'
        labels_y[1]='Orange'
        ax.set_yticklabels(labels_y,fontsize=15)
        savefile = 'Confusion_matrix'+classifier_name+'.png'
        #Save the plot
        plt.savefig(savefile,format='png')
        plt.show()

    def Decision_Boundary_Linear(self,data,W):
        '''
        Plotting Decision Boundary for Linear Model

        Input:
        data - Training data
        W - Weight matrix
        '''
        n_ones = np.ones((data.shape[0],1))
        data=np.concatenate((n_ones,data),axis=1)
        X_data = X_data = data[:,:3]
        b_data=data[data[:,3]==0]
        o_data=data[data[:,3]==1]
        X_1=np.linspace(np.min(data[:,1]), np.max(data[:,1]),1000)
        X_2 = np.linspace(np.min(data[:,2]),np.max(data[:,2]),1000)
        x_1,x_2 = np.meshgrid(X_1,X_2)
        decision_points_Z = np.zeros((x_1.shape))
        n_ones_con=np.ones((x_1.shape[0],1))
        decision_points=np.zeros((x_1.shape[0],X_data[0,1:].shape[0]))

        for i in xrange(x_1.shape[0]):
            x_1_temp=x_1[:,i]
            x_2_temp= x_2[:,0]
            x_concat = np.r_['1,2,0',x_1_temp,x_2_temp]   
            x_concat = np.concatenate((n_ones_con,x_concat),axis=1)
            val_array = np.dot(x_concat,W.transpose())   
            decision_points_Z[i] = val_array[:] 
            val = val_array[0]
            if val > 0.5:        
                for j in xrange(1,val_array.shape[0]):
                    if val_array[j]<=0.5:
                        decision_points[i] = x_concat[j,1:]
                        break
            else:
                for j in xrange(1,val_array.shape[0]):
                    if val_array[j] > 0.5:
                        decision_points[i] = x_concat[j,1:]
                        break
        color = '#FFA500'
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(b_data[:,1],b_data[:,2],'bo')
        ax.plot(o_data[:,1],o_data[:,2],color,linestyle='None',marker = 'o')
        ax.contour(x_1,x_2,decision_points_Z.transpose(),cmap=plt.cm.gray, levels=[.5])
        # Save the plot
        plt.savefig('Linearplot.png',fomat='png')

    def Decision_Boundary_KNN(self,data,k_1=1):
        '''
        Plotting decision boundary for KNN

        Input:
        data: Training data
        k_1: Number of Nearest Neighbors.
        '''
        X_data= data[:,:2]
        y_data=data[:,2]
        K_val=k_1
        x_1 = np.linspace(np.min(data[:,0]),np.max(data[:,0]),200)
        x_2 = np.linspace(np.min(data[:,1]),np.max(data[:,1]),200)
        x_k1,x_k2 = np.meshgrid(x_1,x_2)
        Z = np.zeros(x_k1.shape)
        print 'Will execute for 200 iterations'
        for i in xrange(x_k1.shape[0]):
            x_1_temp=x_k1[:,i]
            x_2_temp= x_k2[:,0]
            x_concat = np.r_['1,2,0',x_1_temp,x_2_temp]
            for j in xrange(x_concat.shape[0]):
                x = x_concat[j]
                dict_dist = {}
                list_dist=[]
                for k in xrange(X_data.shape[0]):
                        dist = np.sum(np.square(x-X_data[k]))
                        list_dist.append(dist)
                        dict_dist[dist]=k
                list_dist.sort()
                count_1=0
                for k1 in xrange(K_val):
                    val = list_dist[k1]
                    pos = dict_dist[val]
                    if y_data[pos] == 1:
                        count_1=count_1+1
               
                prob_count_1 = float(count_1)/K_val
                if prob_count_1>0.5:
                    Z[i][j] = 1
                else:
                    Z[i][j] = 0
            print 'done iteration %d' % (i)

        color = '#FFA500'
        fig = plt.figure()
        ax = fig.add_subplot(111)
        b_data=data[data[:,2]==0]
        o_data=data[data[:,2]==1]
        ax.plot(b_data[:,0],b_data[:,1],'bo')
        ax.plot(o_data[:,0],o_data[:,1],color,linestyle='None',marker = 'o')
        ax.contour(x_k1,x_k2,Z.transpose(),cmap=plt.cm.gray)
        #Save the plot
        plt.savefig('KNN_'+str(K_val)+'.png',format='png')   

def main():
    #Load the training data
    X_data = {}
    count = 0
    set_classes = Set()
    with open("/home/suhaspillai/PatternRecog/Assignments/Project1/RandomForestcode/data_math_symbol.txt","r") as file_read_data:
        for line in file_read_data:
            s=line
            l_s=s.strip("\n").strip("\t").split("\t")
            X_data[count]=l_s
            set_classes.add(l_s[-1])
            count=count+1

    print (set_classes)
    print ("\n count of all the classes is %d") % (len(set_classes))


    dict_class_mapping  = {}
    for i in xrange(10):
        dict_class_mapping[str(i)]= i

    count_class = 10 
    # For others
    for k in  set_classes:
        k = k.strip("\\")
        if k in dict_class_mapping:
            continue
        dict_class_mapping[k] = count_class
        count_class = count_class + 1

    #print dict_class_mapping['gt']
    
    row = count
    cols = len(X_data[0])

    print "row are %d" % (row)
    print "cols are %d" % (cols)

    data_mat = np.zeros((row,cols))
    count_row = 0
    count_col=0
    for key in X_data:
        list_sample = X_data[key]   
        for j in xrange(len(list_sample)-1):
            data_mat[count_row][j] = float(list_sample[j])
        
        key_class_label = list_sample[-1]
        key_class_label=key_class_label.strip("\\")
        data_mat[count_row][j+1] = dict_class_mapping[key_class_label]      
        count_row = count_row+1
        
    c =  Classifier()
    c.predict_KNN(data_mat)
    print 'Done!!!!'
    
if __name__=="__main__":
    pass
    main()
