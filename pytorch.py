
# Pytorch **101**


import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print(torch.__version__)

intro to **Tensors**

## scalar
scalar = torch.tensor(4)
scalar

scalar.ndim

## tensor as normal in
scalar.item()

## vector
vector = torch.tensor([4 , 4])
vector

vector.ndim

vector.shape

## matrix
matrix = torch.tensor([[1, 2],
                           [2, 3]])
matrix

matrix.ndim

matrix[0]

matrix.shape

# Tensor
Tensor = torch.tensor([[[1, 2, 3],
                       [4, 5, 6],
                        [7, 8, 9]]])
Tensor

Tensor.ndim

Tensor.shape

Tensor[0]

Random **Tensor**

### why random tensors?
## rand numbers --> look at data --> update rand numbers --> repate
### create a rand tensor of size(3, 4)
rand_tensor = torch.rand( 2, 3, 4)
rand_tensor

rand_tensor.ndim

rand_tensor.shape

# rand tensor similar shape to an image
rand_image_size_tensor = torch.rand(size = (224, 224, 3)) # height, width, colour channels (R, G, B)
rand_image_size_tensor.shape, rand_image_size_tensor.ndim

## zeros and ones
zeros = torch.zeros(3, 4)
zeros , zeros.ndim

ones = torch.ones(3, 4)
ones , ones.ndim

zeros * rand_tensor

ones * rand_tensor

ones.dtype

 **Range** **of** **tensors** **&** **tensors**-**like**

## torch.arange
arange0 = torch.arange(1,11,2)

## tensors-like
Talike = torch.zeros_like(input = arange0) ## (arange0)
Talike

**Tensors** **data** **type**

float32= torch.tensor([1, 2, 3, 4], dtype=None)
float32

int16= torch.tensor([1, 2, 3, 4], dtype=int)
int16

T= torch.tensor([1, 2, 3, 4],
                dtype=None, ## data type
                    device=None, ## cpu, gpu, tpu
                          requires_grad=False ## trace the gradients
                                                      )
T

### Getting info about tensors

## 1. tensors are not on the right data type
## 2. tensors are not on the right shape
## 3. tensors are not on the right device

some_tensor =  torch.rand(3, 4)
some_tensor
print(f"Datatype:{some_tensor.dtype}")
print(f"Shape:{some_tensor.shape}") ## shape = size()
print(f"Device:{some_tensor.device}")

**Manipulating Tensors** (tensors operations)
- addtion/sub
- multipliction (element- wise)/div
- matrix multipliction

Tensor = torch.tensor([1, 2, 3])
Tensor + 10

Tensor - 10

Tensor * 10

Tensor / 10

Tensor % 2

**Matrix Multipliction**

Main ways of matrix mul in neural networks and deep learing:
1.  Element- wise
2.  Matrix Mul (dot product)

# Element wise mul
print(Tensor * Tensor)

# the sum of the tensor element
torch.matmul(Tensor, Tensor)

# by hand (matrix mul)
# 1*1 + 2*2 + 3*3 =14

%%time
val = 0
for i in range(len(Tensor)):
  val+= Tensor[i] * Tensor[i]
val

%%time
val0 = torch.matmul(Tensor, Tensor)

%%time
randT = torch.rand([10, 10, 10])

%%time
val2 = torch.matmul(randT, randT)
val2

There are two main rules that performing matrix mul need to be satisfy:
1. the **inner imensions** must match
- '(3, 2) @ (3, 2)' won't work
- '(2, 3) @ (3, 2)' will work
- '(3, 2) @ (2, 3)' will work
2.  the  resulting matrix has the shape of the **outer dimensions**:
- '(2, 3) @ (3, 2) = (2, 2)'

##torch.matmul(torch.rand(3, 2), torch.rand(3, 2))

torch.matmul(torch.rand(3, 2), torch.rand(2, 3))

# shapes for mat mul
Ta= torch.tensor([[1, 2],
                   [3, 4],
                      [5, 6]])
Tb =  torch.tensor([[1, 2],
                      [3, 4],
                        [5, 6]])
Ta.shape, Tb.shape

##torch.mm(Ta, Tb)

# adjust the shape
# we use tanspose
Tb, Tb.shape, Tb.T, Tb.T.shape

torch.mm(Ta, Tb.T)

Min, Max, Mean, Sum, etc..(tensor aggregation)

X = torch.rand(1, 3, 4)
X

X.min(), torch.min(X), X.max(), torch.max(X)

## mean() dose not work with long int
## we could use if we encountered a data type error
##  torch.mean(I/p.type(torch.datatype(float32 usully)))
torch.mean(X), X.mean()

torch.sum(X), X.sum()

** Positional min, max**

torch.argmin(X), torch.argmax(X)

Reshaping, stacking, squeezing and unsequezzing tensores
- Reshaping : reshae the input tensor to a defned shape
- View : Returns a view of an input tensor of a certain shape but keep the same memory as the original tensor
- Stacing : combine multiple tensors on top of each other (vstack) or side by side (hstack)
- Squeeze : remove all '1' dimensions from a tensor
- Unsequeeze : add '1' dimensions to a target tensot
- permute : Returns a view of the input with dimensions permute(swapped) in a certain way

import torch
x = torch.arange(1.,11.)
x, x.shape

## add an extra dimension
x_reshape = x.reshape((5, 2))
x_reshape, x_reshape.shape

## view
z = x.view(5, 2)
z, z.shape


## chainging z changing x they shar the memory
z[:, 0] = 5
z, x

(views) require the data to be stored contiguously in memory if the data dose not stored continuously it dose not share the memory== (reshape smart)

# stack tensor on top each other
x_stack = torch.stack([x, x, x, x, x], dim = 1)
x_stack,x_stack.shape

# torch.squeeze() - removes all single dimensions from a target tensor
x_reshape, x_reshape.shape

x_reshape.squeeze(),x_reshape.squeeze().shape

Yy = torch.rand(1 ,9)
Yy, Yy.shape

Yy.squeeze(),Yy.squeeze().shape

## torch.unsqueeze() - adds a single dimension to traget tensor  at a specific dim
Yy_unsq = Yy.squeeze().unsqueeze(dim = 0)
Yy_unsq,Yy_unsq.shape

## torch.permute() - rearrange the dimesions of a target tensor in a specificed order
x_img = torch.rand(size=(224, 224, 3)) # [height, width, colour channels]
x_permuted = x_img.permute(2, 0, 1) #shift the dim[0, 1, 2] rearranging
x_img, x_permuted

Indexing (select data from tensors) using Numpy

import torch 
X = torch.arange(1, 10).reshape(1, 3, 3)
X, X.shape

print(X[0][1][2])

H = torch.arange(1, 19).reshape(2, 3, 3)
H, H.shape

print(H[1][1][2])

# ":" for selecting all dimensions
print(H[:, :,:])

# Pytorch tensors & Numpy
torch to numpy --> torch.from_numpy(ndarray)

numpy to torch --> torch.Tensors.nump()

# numpy to tensors
import torch as rh
import numpy as np

array = np.arange(1., 8.)
tensor = rh.from_numpy(array) # .type(float32)
array, array.shape, tensor, tensor.shape


array.dtype # dflt tensor(torch) is float32

 # tensors to numpy
import torch as rh
import numpy as np
tensor = torch.ones(7)
Numpy_tens = tensor.numpy()
tensor, tensor.shape, Numpy_tens, Numpy_tens.shape

# Reproducbility 
"take random out fo a random"

- Shortly, how the neural network learns:
start with random numbers -> tensors operations -> update random numbers to try and make them better representations of the data -> again -> again - again....
- To reduce the randomness in neural network and pytorch comes the concept of a **Random seed**.

Essentially what **Random seed** dose is "flavour" the randomness

import torch as ch
randA = ch.rand(3, 4)
randB = ch.rand(3, 4)
randA, randB, randA == randB


# random but reproducible
import torch as m
# set random seed
seed = 42
m.manual_seed(seed)
randA = m.rand(3, 4)
m.manual_seed(seed)
randB = m.rand(3, 4)m.manual_seed(seed)
randA, randB, randA == randB

