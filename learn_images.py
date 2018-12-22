from scipy import misc
import glob
from neural import *

# for image_path in glob.glob("./A/hsf_0/hsf_0_00000.png"):
#     image = misc.imread(image_path)
#     image = image[:][:][0]

image_path = "./A/hsf_0/hsf_0_00000.png"
image = misc.imread(image_path)
image = (1 - image.T[0]/255).T.reshape(-1)


num_inputs = len(image)
num_outputs = 2
HL = [num_inputs+1, 32, 32, num_outputs]

# while flag:
#     flag = False        
#     for i in range(1000):
#         L = get_lin(syn)
#         Y = L[-1]

#         d_syn = get_back_prop(y, L)

#         for i in range(len(HL) - 1):
#             syn[i] += d_syn[i]

#     error = np.mean(np.abs(y - L[-1]))

#     print("Error:", str(error))

#     if error>0.45:
#         syn = new_syn()
#     if error>0.01:
#         flag = True
