import time
import numpy as np
# vectorized dot product between two vectors
x, y = np.random.random(100000), np.random.random(100000)
def unvectorized_dot():
    prod = 0
    arr_len = x.shape[0]
    for i in range(arr_len):
        prod += x[i] * y[i]
    return prod

def vectorized_dot():
    return x.dot(y)

def unvectorized_matmul(A, B):
    # A = N * M
    # B = M * P
    # out = N * P
    # cij = sum_k aik bkj, 0 <=k <= M
    C = np.zeros((A.shape[0], B.shape[1]))
    # iterate across rows and columns
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C[i][j]= sum([A[i][k] * B[k][j] for k in range(A.shape[1])])
    return C

# think about the above code. How are we computing C[i][j]? Could we use vectorization to make this go faster?
def partially_vectorized_matmul(A, B):
    C = np.zeros((A.shape[0], B.shape[1]))
    # iterate across rows and columns
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ai = A[i, :] # access the ith row of A
            bj = B[:,j] # access the jth col of B
            C[i][j] = ai.dot(bj)
    return C

def vectorized_matmul(A, B):
    C = A.dot(B)
    return C
    


        
now = time.time()
unvec = unvectorized_dot()
diff_unvec = time.time() - now
print('Unvectorized version took {} seconds'.format(diff_unvec))

now = time.time()
vec = vectorized_dot()
diff_vec = time.time() - now
print('Vectorized version took {} seconds'.format(diff_vec))

print('Unvectorized is {} times slower than vectorized'.format(diff_unvec/diff_vec))

# create 2 random vectors
A, B = np.random.random((1000,1000)), np.random.random((1000, 2000)) * 2

now = time.time()
unvectorized_prod = partially_vectorized_matmul(A, B)
unvec_time = time.time() - now
print('Unvectorized matrix mult: {}'.format(unvec_time))
now = time.time()
partially_vectorized_prod = partially_vectorized_matmul(A, B)
partial_vec_time = time.time() - now
print('Partially vectorized matrix mult took {}'.format(partial_vec_time))
now = time.time()
vectorized_prod = vectorized_matmul(A, B)
vec_time = time.time() - now
print('Fully vectorized matrix multiplication took {}'.format(vec_time))
assert np.isclose(vectorized_prod, unvectorized_prod).all()
assert np.isclose(vectorized_prod, partially_vectorized_prod).all()

print('Unvectorized matrix multiplication is {} times slower than vectorized'.format(unvec_time/vec_time))