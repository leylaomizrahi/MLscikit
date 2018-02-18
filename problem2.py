import csv
import sys
import numpy as np 
import matplotlib.pyplot as plt

#Funtions 

def f(x): 
     return np.dot(x,b)

def R(x):  
     sum = 0.0
     n = len(x)    
     for i in range(n): 
          sum = sum + ((f(x[i]) - y[i])** 2)

     sum = (sum / (2 * n))  
     return sum                             

def scale(x, stdev):  
     xscaled = [] 
     for i in range(len(x)):
          xscaled.append((x[i] - np.mean(x))/ stdev)
     xscaled = np.array(xscaled, dtype='float64')
     return xscaled
 
#Set up variables 
x1 = [] 
x2 = [] 
x = []
y = []
b = [0.0,0.0,0.0]
a = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5 , 1.0, 5.0, 10.0 , 0]
number = 100
no = 0 


# Retrive Data from input 
file_reader = csv.reader(open(sys.argv[1], 'rb'), delimiter=',')
file_writer = csv.writer(open(sys.argv[2], 'wb'), delimiter=',')

for row in file_reader:
     y.append( float(row[2]))
     x1.append(float(row[0]))
     x2.append(float(row[1]))

#Normalize data 

x1 = np.array(x1, dtype='float64')
x2 = np.array(x2, dtype='float64')
stdevx1 = np.std(x1 , ddof = 1)
stdevx2 = np.std(x2, ddof = 1)
x1 = scale(x1, stdevx1)
x2 = scale(x2, stdevx2)



for i in range(len(x1)): 
     x.append((1,x1[i],x2[i]))

x = np.array(x, dtype='float64')


#Update b's 
     
min = 10000
for k in range(len(a)):
     b = [0.0,0.0,0.0]
     no = 0 
     while (no < number):
          for j in range(len(b)):
               sum = 0.0 
               for i in range(len(x)):                    
                    sum = sum + ((f(x[i]) - y[i]) * x[i][j])
               b[j] = b[j] - ((sum /len(x)) * a[k])         
          
          if (R(x) < min):
               min = a[k]
               
          no = no +1
          
     file_writer.writerow([a[k], no, b[0], b[1], b[2]])       
     a[9] = min 
     



