# 📘 PyTorch 101

# Setup
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("PyTorch Version:", torch.__version__)

# --- Introduction to Tensors ---

# Scalars
scalar = torch.tensor(4)
print("Scalar:", scalar, "| Dims:", scalar.ndim, "| Item:", scalar.item())

# Vectors
vector = torch.tensor([4, 4])
print("Vector:", vector, "| Dims:", vector.ndim, "| Shape:", vector.shape)

# Matrices
matrix = torch.tensor([[1, 2], [2, 3]])
print("Matrix:", matrix, "| Dims:", matrix.ndim, "| Shape:", matrix.shape, "| First row:", matrix[0])

# 3D Tensor
tensor3D = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
print("3D Tensor Shape:", tensor3D.shape, "| Dims:", tensor3D.ndim, "| Slice:", tensor3D[0])

# Random Tensors
rand_tensor = torch.rand(2, 3, 4)
rand_image_tensor = torch.rand(224, 224, 3)
print("Random Tensor:", rand_tensor.shape, "| Image-like Tensor:", rand_image_tensor.shape)

# Zeros and Ones
zeros = torch.zeros(3, 4)
ones = torch.ones(3, 4)
print("Zeros:", zeros)
print("Ones:", ones)
print("Element-wise Mul:", ones * rand_tensor)

# Range and Like-Tensors
arange0 = torch.arange(1, 11, 2)
Talike = torch.zeros_like(arange0)
print("Range:", arange0)
print("Zeros Like Range:", Talike)

# Data Types
float32 = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
int16 = torch.tensor([1, 2, 3, 4], dtype=torch.int16)
print("float32 Tensor:", float32.dtype)
print("int16 Tensor:", int16.dtype)

# Tensor Info
some_tensor = torch.rand(3, 4)
print(f"Tensor Info -> DType: {some_tensor.dtype}, Shape: {some_tensor.shape}, Device: {some_tensor.device}")

# Tensor Operations
Tensor = torch.tensor([1, 2, 3])
print("Addition:", Tensor + 10)
print("Subtraction:", Tensor - 10)
print("Multiplication:", Tensor * 10)
print("Division:", Tensor / 10)
print("Modulo:", Tensor % 2)

# Matrix Multiplication
print("Element-wise Mul:", Tensor * Tensor)
print("Dot Product:", torch.matmul(Tensor, Tensor))

# Manual Dot Product
val = 0
for i in range(len(Tensor)):
    val += Tensor[i] * Tensor[i]
print("Manual Dot Product:", val)

# Using matmul
val0 = torch.matmul(Tensor, Tensor)
print("Matmul Result:", val0)

# Large Matrix Multiplication
randT = torch.rand([10, 10, 10])
val2 = torch.matmul(randT, randT)
print("Random Tensor Matmul Result:", val2.shape)

# Matrix Mul Rules
Ta = torch.tensor([[1, 2], [3, 4], [5, 6]])
Tb = torch.tensor([[1, 2], [3, 4], [5, 6]])
print("Ta Shape:", Ta.shape, "| Tb Shape:", Tb.shape, "| Tb Transposed:", Tb.T.shape)

# Matmul after transpose
result = torch.mm(Ta, Tb.T)
print("Matmul Result:", result)

# Aggregation Functions
X = torch.rand(1, 3, 4)
print("Min:", X.min(), "| Max:", X.max(), "| Mean:", X.mean(), "| Sum:", X.sum())

# Positional Min/Max
print("ArgMin:", torch.argmin(X), "| ArgMax:", torch.argmax(X))

# Reshape, View, Stack
x = torch.arange(1., 11.)
x_reshape = x.reshape(5, 2)
z = x.view(5, 2)
z[:, 0] = 5
print("Reshape:", x_reshape.shape, "| View:", z.shape, "| Shared Memory:", x)

x_stack = torch.stack([x, x, x], dim=1)
print("Stacked:", x_stack.shape)

# Squeeze & Unsqueeze
Yy = torch.rand(1, 9)
print("Squeezed:", Yy.squeeze().shape)
Yy_unsq = Yy.squeeze().unsqueeze(0)
print("Unsqueezed:", Yy_unsq.shape)

# Permute
x_img = torch.rand(224, 224, 3)
x_permuted = x_img.permute(2, 0, 1)
print("Original Shape:", x_img.shape, "| Permuted Shape:", x_permuted.shape)

# Indexing
X = torch.arange(1, 10).reshape(1, 3, 3)
print("Index [0][1][2]:", X[0][1][2])
H = torch.arange(1, 19).reshape(2, 3, 3)
print("Index [1][1][2]:", H[1][1][2])
print("All:", H[:, :, :])

# NumPy <-> Torch
array = np.arange(1., 8.)
tensor_from_np = torch.from_numpy(array)
print("From NumPy:", tensor_from_np)

tensor_to_np = torch.ones(7).numpy()
print("To NumPy:", tensor_to_np)

# Reproducibility
torch.manual_seed(42)
randA = torch.rand(3, 4)
torch.manual_seed(42)
randB = torch.rand(3, 4)
print("Are randA and randB equal?", torch.equal(randA, randB))
#1. 🧠 What is a Tensor in PyTorch?
#A tensor in PyTorch is a multi-dimensional array that supports GPU acceleration and automatic differentiation.
#At the lowest level, tensors are abstractions of memory blocks with defined shapes, data types, and strides.
#Unlike NumPy arrays, PyTorch tensors are memory-aware and tightly coupled with autograd, which makes them fundamental to all model training.
#Contiguous memory: PyTorch tensors must often be contiguous in memory to be eligible for certain operations (like view()), meaning the underlying data layout in RAM is sequential.
#📘 Documentation: https://pytorch.org/docs/stable/tensors.html

#2. 🏗️ Tensor Shape Hierarchy
#Scalar: 0D — simplest tensor, stores a single value
#Vector: 1D — list of numbers (activations, weights)
#Matrix: 2D — typical for layer connections or batches
#Higher-order tensors (3D, 4D, etc.): used in image, audio, video, etc.
#These shapes reflect how information is structured in models, e.g. a 4D tensor (B, C, H, W) is used in CNNs (Batch, Channel, Height, Width).

#3. ⚙️ Tensor Initialization
#A. Deterministic vs Stochastic Initialization
#torch.zeros, torch.ones, torch.full are deterministic → predictable memory layout
#torch.rand, torch.randn, torch.randint are stochastic → used for simulating entropy or initial model weights
#Why Random?
#In ML, weights must begin with random but reasonable values:
#Prevent symmetry: random weights ensure neurons learn distinct features
#Avoid zero gradients: especially in deep networks, deterministic zero initialization halts learning
#Emulate entropy: randomness injects exploration
#Why Control Random?
#Random without control = reproducibility chaos.
#To make experiments deterministic, use:
torch.manual_seed(42)
#This sets the internal state of the RNG (random number generator).
#It ensures that calls like torch.rand(...) return the same values across runs.
# https://pytorch.org/docs/stable/generated/torch.manual_seed.html

# 4. 📐 Memory Layout & Tensor Manipulations
# A. Reshape vs View
reshape()#: attempts to return a tensor with the desired shape and reuses data if possible

view()#: same idea, but requires the tensor to be contiguous in memory
#If a tensor has been permuted or sliced in a way that breaks memory contiguity, view() will fail.
#✅ Use .contiguous() before view() to make memory linear again.
#B. Stack vs Concat vs Unsqueeze/Squeeze
# stack: Adds new dimension and stacks tensors on it (e.g. 3 x [3] → [3, 3])
# concat: Merges tensors along existing dimension
# unsqueeze: Add a dimension of size 1
# squeeze: Remove all dimensions of size 1
#These allow control over tensor ranks, useful in model input reshaping or CNN channel alignment.
# https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor.contiguous

#5. 🧮 Matrix Multiplication in ML
#🔁 Two Modes:
#Element-wise Multiplication – Used for broadcasting, Hadamard products
#Dot Product / Matrix Multiplication – Foundation of all neural computation
# Dot product of 1D tensor
torch.matmul(x, x)
#Rules:
#Inner dimensions must match: (A, B) @ (B, C) = (A, C)
#Transpose as needed using .T
#https://pytorch.org/docs/stable/generated/torch.matmul.html

#6. 📊 Aggregations & Reductions
#Used in both loss calculation and evaluation metrics:
#mean, sum, min, max
#argmin, argmax → position of the value
torch.mean(x.float())  # must convert to float if int
#📘 Reduction Ops: https://docs.pytorch.org/docs/stable/tensors.html#reduction-ops

#7. 🔁 NumPy ↔ PyTorch Interop
#PyTorch allows seamless conversion with NumPy:
torch.from_numpy(ndarray)
tensor.numpy()
#⚠️ Tensors created from NumPy share memory. If you modify one, the other changes too.
#📘 NumPy Bridge

#8. 🔬 Device and Type Management
#tensor.dtype → e.g., torch.float32, torch.int64
#tensor.device → usually "cpu" or "cuda"
#Efficient training requires careful planning of:
#Precision (float32 vs float16) for speed/accuracy tradeoff
#Device transfers (.to('cuda')) to maximize GPU use
#📘 Data Types : https://docs.pytorch.org/docs/stable/tensors.html#data-types

#9. 🧪 Reproducibility: The "Flavor" of Randomness
#In a training pipeline, randomness appears in:
#Weight initialization
#Dropout layers
#Data shuffling
#Augmentations
#To debug, benchmark, or compare models, we need determinism:
torch.manual_seed(seed) #– global PyTorch RNG
np.random.seed(seed) #– for NumPy
#random.seed(seed) #– Python’s built-in RNG
#Additionally:
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#These ensure deterministic behavior on CUDA for CNNs, trading performance for reproducibility.
