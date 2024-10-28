import math
import numpy as np
import matplotlib.pyplot as plt
rl=1.5
ru=6
def adapFD(f,x,level):
  g=np.zeros(len(x))
  count=0;
  for i in range(len(x)):
    h=1.15*math.sqrt(level)
    l=0
    u=9999
    x4plus=x.copy()
    xplus=x.copy()
    x4plus[i]=x4plus[i]+4*h
    xplus[i]=xplus[i]+h
    rt=np.abs((f(x4plus)-4*f(xplus)+3*f(x))/(8*level));
    count+=3;
    while (rt>ru or rt<rl):
      if rt<rl:
        l=h;
      elif rt>ru:
        u=h;
      if u==9999:
        h=4*h;
      elif l==0:
        h=h/4;
      else:
        h=(l+u)/2
      x4plus=x.copy()
        
      xplus=x.copy()
      x4plus[i]=x4plus[i]+4*h
      xplus[i]=xplus[i]+h
      rt=np.abs((f(x4plus)-4*f(xplus)+3*f(x))/(8*level));
      count+=2;
    xplus=x.copy()
    xplus[i]=xplus[i]+h
    g[i]=(f(xplus)-f(x))/h
    count+=1;   
  return g,count
def adapGD(f,ftrue,x_init,maxcount,level):
  countall=[0];
  count=0;
  x=x_init
  funall=[ftrue(x)];
  x_all=[x];  
  while count<maxcount:
    g,c=adapFD(f,x,level)
    count+=c
    t=1;
    while f(x-t*g)>f(x)-0.2*t*np.linalg.norm(g)**2 and t>=1e-15:
      count+=1;
      t=t/2;
    x=x-t*g
    funall.append(ftrue(x));
    x_all.append(x);
    countall.append(count)
  return x,funall,countall,x_all
def adapGDBDL(f,ftrue,x_init,maxcount,level):
  countall=[0];
  count=0;
  x=x_init
  funall=[ftrue(x)];
  x_all=[x];  
  t=1;
  while count<maxcount:
    g,c=adapFD(f,x,level)
    grad=g;
    count+=c
    t_init=t;
    loop=0;
    while f(x-t*grad) >f(x)-0.08*t*np.dot(grad, grad):
        loop=loop+1;
        count=count+1;
        if loop%2==1:
            t=t_init*2**(-loop//2);
        else:
            t=t_init*2**(-loop//2); 
    x=x-t*g;
    funall.append(ftrue(x));
    x_all.append(x);
    countall.append(count)
  return x,funall,countall,x_all
def ftrue(x):
  return (np.exp(2*x[0]+3*x[1]-1)+np.exp(3*x[0]-x[1])+np.exp(x[0]-x[1]-6)-3)**2
def f(x):
  noise=np.random.uniform(-level,level)
  return ftrue(x)+noise