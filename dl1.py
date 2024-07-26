###Implement the following functions in Python from scratch. Do not use any library functions. You are allowed to use numpy and matplotlib.
###Generate 100 equally spaced values between -10 and 10. Call this list as z. Implement the following functions and its derivative.
###Use class notes to find the expression for these functions. Use z as input and plot both the function outputs and its derivative outputs.
###Upload your code into Github and share it with me.
###a. Sigmoid
###b. Tanh
###c. ReLU (Rectified Linear Unit)
###d. Leaky ReLU
###e. Softmax


import numpy as np
from matplotlib import pyplot as plt

z = np.linspace(-10, 10, 100)
#print(z)
#print(z.shape)



###1.  sigmoid function is given by: g(z) = 1/1+e^-z
e_power_minus_z = np.exp(-z)
g_z = 1/(1+e_power_minus_z)
#print(g_z)
minvalofg_z = min(g_z)
maxvalofg_z = max(g_z)
print("Minimum value of sigmoid function is: ", minvalofg_z)
print("Maximum value of sigmoid function is: ", maxvalofg_z)
###Sigmoid function is NOT zero centered.
### derivative of sigmoid function is given by: (1/1+e^-z)(e^-z/1+e^-z)
derivative1 = (1/(1+e_power_minus_z))*(e_power_minus_z/(1+e_power_minus_z))
#print(derivative1)
###plotting z and g(z)
plt.plot(z, g_z)
plt.xlabel("z")
plt.ylabel("g(z)")
##plt.show()
###plotting z and its derivative
plt.plot(z, derivative1)
plt.xlabel("z")
plt.ylabel("Derivative")
plt.title("Sigmoid function and Derivative")
#plt.show()



###2. Tanh function is given by: sinh(z)/cosh(z) or (e^z - e^-z)/(e^z + e^-z)
e_power_z = np.exp(z)
tanh_z = (e_power_z - e_power_minus_z) / (e_power_z + e_power_minus_z)
print(tanh_z)
minvaloftanh_z = min(tanh_z)
maxvaloftanh_z = max(tanh_z)
###Tanh function is zero centered.
print("Minimum value of tanh function is: ", minvaloftanh_z)
print("Maximum value of tanh function is: ", maxvaloftanh_z)
###derivative of tanh function is given by: 1-((e^x-e^-x)^2/(e^x + e^-x)^2) or 1-tanh^2(z)
derivative2 = 1-((e_power_z-e_power_minus_z)**2/(e_power_z + e_power_minus_z)**2)
#print(derivative2)
plt.plot(z, tanh_z)
plt.xlabel("z")
plt.ylabel("tanh(z)")
##plt.show()
plt.plot(z, derivative2)
plt.xlabel("z")
plt.ylabel("Derivative")
plt.title("Tanh function and Derivative")
#plt.show()



###3. ReLU (Rectified Linear Unit) function is given by: max(0, z)
f_x = np.maximum(0, z)
#print(f_x)
minvaloff_x = min(f_x)
maxvaloff_x = max(f_x)
print("Minimum value of ReLU function is: ", minvaloff_x)
print("Maximum value of ReLU function is: ", maxvaloff_x)
###ReLU function is NOT zero centerd.
derivative3 = np.where(z > 0, 1, 0)
#print(derivative3)
plt.plot(z, f_x)
plt.xlabel("z")
plt.ylabel("f_x")
###plt.show()
plt.plot(z, derivative3)
plt.title("ReLU function and Derivative")
plt.xlabel("z")
plt.ylabel("Derivative")
#plt.show()


###4. Leaky ReLU function is given by:  ax if z<0 else z
a = 0.05
lr_f_x = np.where(z < 0, a*z, z)
#print(lr_f_x)
minvaloflr_f_x = min(lr_f_x)
maxvaloflr_f_x = max(lr_f_x)
print("Minimum value of Leaky ReLU function is: ", minvaloflr_f_x)
print("Maximum value of Leaky ReLU function is: ", maxvaloflr_f_x)
###Leaky ReLU function is NOT zero centered.
derivative4 = np.where(z > 0, 1, a)
#print(derivative4)
plt.plot(z, lr_f_x)
plt.xlabel("z")
plt.ylabel("Leaky ReLU function")
##plt.show()
plt.plot(z, derivative4)
plt.title("Leaky ReLU function and Derivative")
plt.xlabel("z")
plt.ylabel("Derivative")
#plt.show()



###5. Softmax function is given by:
#z = np.linspace(-10, 10, 100)
softmax_f_x = np.exp(z) / np.sum(np.exp(z))
print(softmax_f_x)
minvalofsoftmax_f_x = min(softmax_f_x)
maxvalofsoftmax_f_x = max(softmax_f_x)
print("Minimum value of Softmax function is: ", minvaloflr_f_x)
print("Maximum value of Softmax function is: ", maxvaloflr_f_x)
###Softmax function is NOT zero centered.
plt.plot(z, softmax_f_x)
plt.title("Softmax function")
plt.xlabel("z")
plt.ylabel("Softmax function")
#plt.show()
