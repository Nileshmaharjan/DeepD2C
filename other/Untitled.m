Fs = 150.0;  
Ts = 1.0/Fs;
t = np.arange(0,1,Ts) 

ff1 = 5;   
ff2 = 10;  
y = np.sin(2*np.pi*ff1*t) + np.sin(3*np.pi*ff2*t)