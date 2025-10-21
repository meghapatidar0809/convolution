import time
import random



# Below are parameters and their explanation (following standard notation as discussed in class)
# N - Number of input fmaps/output fmaps (batch size)
# C - Number of 2-D input fmaps /filters (channels)
# H - Height of input fmap (activations)
# W - Width of input fmap (activations)
# R - Height of 2-D filter (weights)
# S - Width of 2-D filter (weights)
# M - Number of 2-D output fmaps (channels)
# E - Height of output fmap (activations)
# F - Width of output fmap (activations)

# Initializing dimensions of inputs & filters
N, C, H, W, R, S, M = 8, 4, 32, 32, 5, 5, 32
stride_length = 2 # stride length(as mentioned in the question)
padding = 0 # padding(as mentioned in the question)



# Initializing input filters & fmaps ranging between -1 to 1
random.seed(0)

# Generating 4D list(tensor) of random values ranging between -1 to 1 for input feature maps(fmaps)
input_fmaps = []
for _ in range(N):  # For batch size
    batch_array = []
    for _ in range(C):  # For channels
        channels = []
        for _ in range(H):  # For height
            row = [random.uniform(-1, 1) for _ in range(W)]  # For width (innermost)
            channels.append(row)
        batch_array.append(channels)
    input_fmaps.append(batch_array)

# Generating 4D list(tensor) of random values ranging between -1 to 1 for filters
filters = []
for _ in range(M):  # Number of output filters
    filter_array = []
    for _ in range(C):  # Number of channels
        channels = []
        for _ in range(R):  # For filter height
            row = [random.uniform(-1, 1) for _ in range(S)]  # For filter width (innermost)
            channels.append(row)
        filter_array.append(channels)
    filters.append(filter_array)



# Computating output dimensions
E = ((H - R) // stride_length) + 1  # height of output
F = ((W - S) // stride_length) + 1  # width of output



################################################### PART_A ###################################################

# Initializing output fmaps with 0
output_fmaps = []
for _ in range(N):  # For batch size
    batch_array = []
    for _ in range(M):  # Number of output filters
        fmap = []
        for _ in range(E):  # For height of the output
            row = [0 for _ in range(F)]  # For width of the output
            fmap.append(row)
        batch_array.append(fmap)
    output_fmaps.append(batch_array)

# Naive convolution implementation (using 7 nested loops)
def naive_convolution():
    for n in range(N):  # Iterates over batch_size
        for m in range(M):  # Iterates over output filters
            for e in range(E):  # Iterates over output_fmaps height
                for f in range(F):  # Iterates over output_fmaps width
                    for r in range(R):  # Iterates over height of filters 
                        for s in range(S):  # Iterates over width of filters 
                            for c in range(C):  # Iterates over number of channels
                                h_in = e * stride_length + r  # height index in input fmaps
                                w_in = f * stride_length + s  # width index in input fmaps
                                if 0 <= h_in < H and 0 <= w_in < W:   # For ensuring that index lies in range of input_fmaps dimensions
                                    output_fmaps[n][m][e][f] += (
                                        input_fmaps[n][c][h_in][w_in] * filters[m][c][r][s]      # Data Assignment
                                    )

    return output_fmaps



################################################### PART_B ###################################################

# Flattened convolution implementation
def flattened_convolution():
    # Initializing Flattened input matrix with CRS(inputChannels_heightOfFilter_widthOfFilter) rows and 
    # EF(heightOfOutputFmap_widthOfOutputFmap) columns with 0
    flattened_input = []
    for _ in range(C * R * S):  # For each filter position
        row = [0 for _ in range(E * F * N)]  # Creating a row of zeros
        flattened_input.append(row)
   
   
   # Preparing Flattened input_fmaps
    for n in range(N):  # Iterates over batch_size
        for e in range(E):  # Iterates over output_fmaps height
            for f in range(F):  # Iterates over output_fmaps width
                for c in range(C):  # Iterates over number of channels
                    for r in range(R):  # Iterates over height of filters
                        for s in range(S):  # Iterates over width of filters
                            h_in = e * stride_length + r   # Calculates row index in input_fmap
                            w_in = f * stride_length + s   # Calculates column index in input_fmap
                            if 0 <= h_in < H and 0 <= w_in < W:   # For ensuring that index lies in range of input_fmaps dimensions
                                flattened_input[c * R * S + r * S + s][n * E * F + e * F + f] = input_fmaps[n][c][h_in][w_in]  # Data Assignment


    # Initializing Flattened filter matrix with M(outputChannels) rows and CRS(channels_heightOfFilter_widthOfFilter) columns
    flattened_filters = []
    for _ in range(M): # Iterates over output filters
        row = [0] * (C * R * S)  # Creating a row of zeros
        flattened_filters.append(row)
    
    
    # Flattening the filters
    for m in range(M):  # Iterates over output filters
        for c in range(C):  # Iterates over number of channels
            for r in range(R):  # Iterates over height of filters
                for s in range(S):  # Iterates over width of filters
                    flattened_filters[m][R * S * c + S * r + s] = filters[m][c][r][s]

 
    total_count = E * F * N  # counting total no. of elements in Flattened output matrix.


    # Initializing Flattened output matrix with (EFN) elements
    flattened_output = []
    for _ in range(M):  # Iterates over output filters
        row = [0] * total_count  # Creating a row of zeros for each output filter
        flattened_output.append(row)


    # Convolution operation performs matrix multiplication of Flattened input and Flattened filter
    for m in range(M):  # Iterates over output filters
        for i in range(total_count):
            for j in range(C*R*S):
                flattened_output[m][i] += flattened_filters[m][j] * flattened_input[j][i]  # Matrix multiplication operation

    return flattened_output



################################################### PART_C ###################################################

# Implementation for checking that both PART_A and PART_B produce same results 
def verify_outputs(naive_convolution_output, flattened_convolution_output, tolerance_value=1e-6):   # tolerance value (acceptable deviation) will be 0.000001
    
    # Flattening of Naive output (4D to 1D)
    flattened_Naive = []
    for n in range(N):  # Iterates over batch_size
        for m in range(M):  # Iterates over output filters
            for e in range(E):  # Iterates over output_fmaps height
                for f in range(F):  # Iterates over output_fmaps width
                    flattened_Naive.append(round(naive_convolution_output[n][m][e][f], 5))  
                      
    # Flattening of Flattened output (2D to 1D)
    flattened_of_Flattened = []
    for row in flattened_convolution_output:    # Iterate over each row of flattened_convolution_output 
        for value in row:   # Iterate over each value
            flattened_of_Flattened.append(round(value, 5))
    
    
# Flattening might change the order of elements therefore rearrangement of elements is required.


    # Creating a dictionary to map values of flattened_Naive with their index values
    index_dict_for_Naive = {value: index for index, value in enumerate(flattened_Naive)}
    
    # Rearranging flattened_Naive to match the order of values in flattened_of_Flattened
    rearranged_Naive = [flattened_Naive[index_dict_for_Naive[value]] for value in flattened_of_Flattened]

    # Comparison of the rearranged flattened_Naive output with flattened_of_Flattened output
    for i in range(len(flattened_of_Flattened)):
        # Checking if, absolute difference of values of matrices (Naive and Flattened) > tolerance, if yes then we will say the outputs are different
        if abs(rearranged_Naive[i] - flattened_of_Flattened[i]) > tolerance_value: 
            print(f"Rearranged Naive convolution output: {rearranged_Naive[i]}, Flattened convolution output: {flattened_of_Flattened[i]}")
            print(f"It is evitable that output of these two convolution matrices are not same. These are not same at index {i}")
            return False

    return True



################################################### ANALYSIS ###################################################


#Calculating execution time of both convolutions and verifying their correctness

# Execution time of Naive convolution
start_time = time.time()
naive_convolution_output =  naive_convolution() # PART_A
naive_time = time.time() - start_time

# Execution time of Flattened convolution
start_time = time.time()
flattened_convolution_output = flattened_convolution() # PART_B
flattened_time = time.time() - start_time

# Printing execution time
print(f"Naive convolution execution time: {naive_time:.6f} seconds")
print(f"Flattened convolution execution time: {flattened_time:.6f} seconds")

# Verifying whether both methods are generating same output or not
result = verify_outputs(naive_convolution_output,flattened_convolution_output) # PART_C
if result:
    print("Both methods produced same output matrices")
else:

    print("Produced different output matrices")




