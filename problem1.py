import csv
import sys
import matplotlib.pyplot as plt

x = []
y = []
w = [0,0,0]
x1 = [] 
x2 = [] 
file_reader = csv.reader(open(sys.argv[1], 'rb'), delimiter=',')
file_writer = csv.writer(open(sys.argv[2], 'wb'), delimiter=',')

for row in file_reader:
     x.append((1,int(row[0]), int(row[1])))
     x1.append(int(row[0]))
     x2.append(int(row[1]))
     y.append(int(row[2]))

plt.plot( x1, x2, y, 'ro')
plt.show()




def f(x):
     sum = 0
     for i in range(len(x)):
          sum = sum + w[i] * x[i] 
     if (sum > 0): 
          return 1
     return -1 

convergence = True 

while(convergence):
     prev = list(w)
     for i in range(len(x)): 
          if f(x[i]) * y[i] <= 0: 
               for j in range(len(w)):
                    w[j] = w[j] + (y[i] * x[i][j])
     if prev == w: 
          convergence = False

     file_writer.writerow([w[1], w[2],w[0]])               
     


