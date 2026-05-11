#include <cuda_runtime.h>
#include <cmath>
#include <cstring>

__device__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    return val;
}


__global__ void query_mag_kernel(const float *query, float *out, int N) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        sum += query[i] * query[i];
    sum = warp_reduce(sum);
    if (threadIdx.x == 0) *out = sqrtf(sum);
}

__global__ void similarity_kernel(
    const float *matrix,
    const float *query,
    float *query_mag,
    float *scores,
    int M,
    int N) {
    int row_idx = blockIdx.x;

    if (row_idx >= M)
        return;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    int num_warps = blockDim.x / 32;

    extern __shared__ float smem[];

    float *sscore = smem;
    float *smag = smem + num_warps;

    const float *matrix_row = matrix + row_idx * N;

    float score = 0.0f;
    float mag = 0.0f;

    for (int col = tid; col < N; col += blockDim.x) {
        float val = matrix_row[col];

        score += val * query[col];
        mag += val * val;
    }

    score = warp_reduce(score);
    mag = warp_reduce(mag);

    if (lane == 0) {
        sscore[wid] = score;
        smag[wid] = mag;
    }

    __syncthreads();

    score = (tid < num_warps) ? sscore[tid] : 0.0f;
    mag = (tid < num_warps) ? smag[tid] : 0.0f;

    score = warp_reduce(score);
    mag = warp_reduce(mag);

    if (tid == 0)
        scores[row_idx] = score / (sqrtf(mag) * query_mag[0] + 1e-8f);
}

// distance functions
__global__ void cosine_distance(
    const float *matrix,
    const float *query,
    float *query_mag,
    float *scores,
    int M,
    int N) {
    int row_idx = blockIdx.x;

    if (row_idx >= M)
        return;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    int num_warps = blockDim.x / 32;

    extern __shared__ float smem[];

    float *sscore = smem;
    float *smag = smem + num_warps;

    const float *matrix_row = matrix + row_idx * N;

    float score = 0.0f;
    float mag = 0.0f;

    for (int col = tid; col < N; col += blockDim.x) {
        float val = matrix_row[col];

        score += val * query[col];
        mag += val * val;
    }

    score = warp_reduce(score);
    mag = warp_reduce(mag);

    if (lane == 0) {
        sscore[wid] = score;
        smag[wid] = mag;
    }

    __syncthreads();

    score = (tid < num_warps) ? sscore[tid] : 0.0f;
    mag = (tid < num_warps) ? smag[tid] : 0.0f;

    score = warp_reduce(score);
    mag = warp_reduce(mag);

    if (tid == 0)
        scores[row_idx] = 1 - score / (sqrtf(mag) * query_mag[0] + 1e-8f);
}


__global__ void euclidean_distance(
    const float *matrix,
    const float *query,
    float *scores,
    int M,
    int N) {
    int row_idx = blockIdx.x;

    if (row_idx >= M)
        return;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    int num_warps = blockDim.x / 32;

    extern __shared__ float sdata[];
    const float *matrix_row = matrix + row_idx * N;

    float score = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        float val = matrix_row[col] - query[col];

        score += val * val;
    }
    score = warp_reduce(score);

    if (lane == 0) {
        sdata[wid] = score;
    }
    __syncthreads();
    score = (tid < num_warps) ? sdata[tid] : 0.0f;
    score = warp_reduce(score);

    if (tid == 0)
        scores[row_idx] = sqrtf(score);
}

__global__ void cityblock_distance(
    const float *matrix,
    const float *query,
    float *scores,
    int M,
    int N) {
    int row_idx = blockIdx.x;

    if (row_idx >= M)
        return;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    int num_warps = blockDim.x / 32;

    extern __shared__ float sdata[];
    const float *matrix_row = matrix + row_idx * N;

    float score = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        float val = matrix_row[col] - query[col];

        score += abs(val);
    }
    score = warp_reduce(score);

    if (lane == 0) {
        sdata[wid] = score;
    }
    __syncthreads();
    score = (tid < num_warps) ? sdata[tid] : 0.0f;
    score = warp_reduce(score);

    if (tid == 0)
        scores[row_idx] = score;
}


extern "C" {
void *create_index(const float *h_matrix, int M, int N) {
    float *d_matrix;

    cudaMalloc(&d_matrix, M * N * sizeof(float));

    cudaMemcpy(
        d_matrix,
        h_matrix,
        M * N * sizeof(float),
        cudaMemcpyHostToDevice
    );

    return (void *) d_matrix;
}


void search_index(
    void *index_ptr,
    const float *h_query,
    float *h_scores,
    int M,
    int N) {
    float *d_matrix = (float *) index_ptr;

    float *d_query, *d_scores, *d_query_mag;

    cudaMalloc(&d_query, N * sizeof(float));
    cudaMalloc(&d_scores, M * sizeof(float));
    cudaMalloc(&d_query_mag, sizeof(float));

    cudaMemcpy(
        d_query,
        h_query,
        N * sizeof(float),
        cudaMemcpyHostToDevice
    );

    query_mag_kernel<<<1,32>>>(d_query, d_query_mag, N);
    cudaDeviceSynchronize();
    int threads = 128;

    int shmem = 2 * (threads / 32) * sizeof(float);

    similarity_kernel<<<M, threads, shmem>>>(
        d_matrix,
        d_query,
        d_query_mag,
        d_scores,
        M,
        N
    );
    cudaDeviceSynchronize();
    cudaMemcpy(
        h_scores,
        d_scores,
        M * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    cudaFree(d_query);
    cudaFree(d_scores);
    cudaFree(d_query_mag);
}

void knn_search(
    void *index_ptr,
    const float *h_query,
    float *h_scores,
    int M,
    int N,
    const char *metric) {
    float *d_matrix = (float *) index_ptr;

    float *d_query, *d_scores;

    cudaMalloc(&d_query, N * sizeof(float));
    cudaMalloc(&d_scores, M * sizeof(float));

    cudaMemcpy(
        d_query,
        h_query,
        N * sizeof(float),
        cudaMemcpyHostToDevice
    );

    int threads = 128;

    int shmem = (threads / 32) * sizeof(float);
    if (std::strcmp(metric, "euclidean") == 0) {
        euclidean_distance<<<M, threads, shmem>>>(
            d_matrix,
            d_query,
            d_scores,
            M,
            N
        );
    } else if (std::strcmp(metric, "cosine") == 0) {
        float *d_query_mag;
        cudaMalloc(&d_query_mag, sizeof(float));
        query_mag_kernel<<<1,32>>>(d_query, d_query_mag, N);
        cudaDeviceSynchronize();
        cosine_distance<<<M, threads, 2 * shmem>>>(
            d_matrix,
            d_query,
            d_query_mag,
            d_scores,
            M,
            N
        );
        cudaFree(d_query_mag);
    } else if (std::strcmp(metric, "cityblock") == 0) {
        cityblock_distance<<<M, threads, shmem>>>(
            d_matrix,
            d_query,
            d_scores,
            M,
            N
        );
    }
    cudaDeviceSynchronize();
    cudaMemcpy(
        h_scores,
        d_scores,
        M * sizeof(float),
        cudaMemcpyDeviceToHost
    );
    cudaFree(d_query);
    cudaFree(d_scores);
}


void destroy_index(void *index_ptr) {
    cudaFree(index_ptr);
}
}


// #include <cuda_runtime.h>


// __global__ void query_mag_kernel(const float* query, float* out, int N) {
//     float sum = 0.0f;
//     for (int i = 0; i < N; i++) sum += query[i] * query[i];
//     *out = sqrtf(sum);
// }

// __global__ void similarity_kernel(const float* matrix, const float* query,
//                                    float query_mag, float* scores, int M, int N)
// {
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     if (row >= M) return;

//     float score = 0.0f, mag = 0.0f;
//     for (int col = 0; col < N; col++) {
//         float val = matrix[row * N + col];
//         score += val * query[col];
//         mag   += val * val;
//     }
//     scores[row] = score / (sqrtf(mag) * query_mag);
// }
// __global__ void similarity_kernel2(const float* matrix, const float* query,
//                                    float query_mag, float* scores, int M, int N)
// {
//     int row_idx=blockIdx.x;
//     if (row_idx >= M) return;
//     extern __shared__ float smem[];
//     float* sscore = smem;
//     float* smag   = smem + blockDim.x;
//     const float* matrix_row=matrix+row_idx*N;
//     float score = 0.0f, mag = 0.0f;
//     for (int col = threadIdx.x; col < N; col+=blockDim.x) {
//         float val = matrix_row[col];
//         score += val * query[col];
//         mag   += val * val;
//     }
//     sscore[threadIdx.x]=score;
//     smag[threadIdx.x]=mag;
//     __syncthreads();
//     for(int stride=blockDim.x/2;stride>0;stride>>=1)
//     {
//         if(threadIdx.x<stride)
//         {
//             sscore[threadIdx.x]+=sscore[threadIdx.x+stride];
//             smag[threadIdx.x]+=smag[threadIdx.x+stride];

//         }
//         __syncthreads();
//     }
//     if(threadIdx.x==0)
//         scores[row_idx] = sscore[0] / (sqrtf(smag[0]) * query_mag);
// }

// __device__ float warp_reduce(float val)
// {
//     for (int offset = 16; offset > 0; offset >>= 1)
//         val += __shfl_down_sync(0xffffffff, val, offset);
//     return val;
// }

// __global__ void similarity_kernel3(const float* matrix, const float* query,
//                                     float query_mag, float* scores, int M, int N)
// {
//     int row_idx   = blockIdx.x;
//     if (row_idx >= M) return;

//     int tid       = threadIdx.x;
//     int lane      = tid % 32;
//     int wid       = tid / 32;
//     int num_warps = blockDim.x / 32;

//     extern __shared__ float smem[];
//     float* sscore = smem;
//     float* smag   = smem + num_warps;   // only need num_warps slots

//     const float* matrix_row = matrix + row_idx * N;

//     float score = 0.0f, mag = 0.0f;
//     for (int col = tid; col < N; col += blockDim.x) {
//         float val = matrix_row[col];
//         score += val * query[col];
//         mag   += val * val;
//     }

//     // stage 1: reduce within each warp
//     score = warp_reduce(score);
//     mag   = warp_reduce(mag);

//     // stage 2: warp leaders write to shared memory
//     if (lane == 0) {
//         sscore[wid] = score;
//         smag[wid]   = mag;
//     }
//     __syncthreads();  // ← ensure all warp leaders have written

//     // stage 3: warp 0 reduces across warp partial sums
//     score = (tid < num_warps) ? sscore[tid] : 0.0f;
//     mag   = (tid < num_warps) ? smag[tid]   : 0.0f;
//     score = warp_reduce(score);
//     mag   = warp_reduce(mag);

//     // stage 4: thread 0 writes final result directly from register
//     if (tid == 0)
//         scores[row_idx] = score / (sqrtf(mag) * query_mag);
// }

// extern "C" void solve(const float* matrix, const float* query, float* scores, int M, int N)
// {
//     // Step 1: compute query magnitude
//     float* d_query_mag;
//     cudaMalloc(&d_query_mag, sizeof(float));
//     query_mag_kernel<<<1, 1>>>(query, d_query_mag, N);

//     float h_query_mag;
//     cudaMemcpy(&h_query_mag, d_query_mag, sizeof(float), cudaMemcpyDeviceToHost);
//     cudaFree(d_query_mag);

//     // Step 2: compute cosine scores
//     // int threads = 256;
//     // int blocks  = (M + threads - 1) / threads;
//     // similarity_kernel<<<blocks, threads>>>(matrix, query, h_query_mag, scores, M, N);
//     // cudaDeviceSynchronize();
//     // int threads = 256;
//     // int blocks  = M;
//     // similarity_kernel2<<<blocks, threads,2*threads*sizeof(float)>>>(matrix, query, h_query_mag, scores, M, N);
//     // cudaDeviceSynchronize();

//     int threads = 128;
//     int shmem   = 2 * (threads / 32) * sizeof(float);  // 2 * 8 floats = 64 bytes
//     similarity_kernel3<<<M, threads, shmem>>>(matrix, query, h_query_mag, scores, M, N);
//     cudaDeviceSynchronize();
// }
