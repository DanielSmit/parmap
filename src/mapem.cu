#include "mapem.h"


/* All the CUDA kernels are implemented in 'cups' namespace. */

namespace cups{

/*
 * The small number of values, which do not chaned during iteration
 * are stored in constant memory.
 */
__constant__ float cm_p2t[277];
__constant__ float cm_facinv[10];
__constant__ int cm_ri[10];
__constant__ float cm_lambdaArr[10];
__constant__ float cm_alphaArr[10];
__constant__ float cm_pArr[100];

__device__
int ii(int ofs, int bc, int row, int col, int bth, int bthc) {
  return ofs + (row*bc + col)*bthc + bth;
}

__device__
int mi(int ofs, int bc, int i, int row, int col, int bth, int bthc) {
  return ofs + (i*bc*bc + row*bc + col)*bthc+bth;
}

__device__
int mij(int ofs, int bc, int i, int j, int row, int col, int bth, int bthc) {
  return ofs + (i*bc*bc*bc + j*bc*bc + row*bc + col)*bthc+bth;
}

__device__
int vi(int ofs, int bc, int i, int idx, int bth, int bthc) {
  return ofs + (i*bc + idx)*bthc + bth;
}

__device__
int vij(int ofs, int bc, int i, int j, int idx, int bth, int bthc) {
  return ofs + (i*bc*bc + j*bc + idx )*bthc + bth;
}

__device__
int factorial(int a) {
  if (a == 0 || a == 1)
    return 1;
  return a * factorial(a - 1);
}

__device__
float f(int r, float lambda, float x) {
  float factor = (float) powf(lambda * x, r - 1) * cm_facinv[r-1];
  float value = factor * lambda * (float) expf(-lambda * x);
  return value;
}

#define MAX_ACTIVE_BLOCKS__MAT_O2_O3(bc, bc2) \
 cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, \
                                                 cups::kernel_mat_o2_o3<bc, bc2>, h,\
                                                 0);

#define KERNEL_MAT_O2_O3(bc, bc2, grid_dim, block_dim) cups::kernel_mat_o2_o3<bc, bc2><<<grid_dim, block_dim, 0>>>(\
    T,\
    L,\
    K,\
    dv_o2mat, dv_o2matex,\
    dv_o3mat, dv_o3matex,\
    dv_umat, dv_umat2, dv_umatex,\
    dv_mmat,\
    dv_vec,\
    dv_timeArr\
    );

template<int bc, int bc2>
__global__ void
kernel_mat_o2_o3(
    int T,
    int L,
    int K,
    float *o2mat, int *o2matex,
    float *o3mat, int *o3matex,
    float *umat, float *umat2, int *umatex,
    float *mmat,
    float *vec,
    float *timeArr
){

	volatile int l = threadIdx.x + blockDim.x*blockIdx.x;
	if(l >= L)
		return;

	int bth = threadIdx.x;
	int bthc = blockDim.x;

	int o3ofs = bthc * blockIdx.x * bc * bc * bc * bc;
	int o2ofs = bthc * blockIdx.x * bc * bc * bc;
	int uofs = bthc * blockIdx.x * bc * bc;
	int tofs = bthc * blockIdx.x * K;

	float rvec[bc];
	float rmmat[bc2];

	int zi = l * K;
	int ze = zi + K;
	ze = (ze > T ? T : ze);

	float max = 0;

	// U = I
	int uex = 0;
	int prev_uex = 0;
	for (int r = 0; r < bc; r++)
		for (int c = 0; c < bc; c++)
			umat[ii(uofs, bc, r, c, bth, bthc)] = r == c ? 1.0f : 0.0f;

	for (int z = zi, zz=0; z < ze; z++, zz++) {
		int idx = tofs + zz*bthc + bth;
		float x = timeArr[idx];
		float addf = (z + 1 != T ? 1.0f : 0.0f );

		// calc M
		max = 0;
		#pragma unroll
		for (int r = 0; r < bc; r++) {
			float fval = f(cm_ri[r], cm_lambdaArr[r], x);
			#pragma unroll
			for (int c = 0; c < bc; c++){
				float val = cm_pArr[r*bc+c]*fval;
				rmmat[r*bc+c] = val;
			}
		}

		int mex = 0;

		int exA = prev_uex + mex;
		int exB = uex + mex;

		int exA_diff = exA - uex;
		int exB_diff = exB - uex;

		float exAf = POW2(exA_diff);
		float exBf = POW2(exB_diff);

		// update o2
		for (int i = 0; i < bc; i++) {

			// o2 = o2 * M
			for (int r = 0; r < bc; r++) {
				#pragma unroll
				for (int c = 0; c < bc; c++)
					rvec[c] = o2mat[mi(o2ofs, bc, i, r, c, bth, bthc)];

				#pragma unroll
				for (int c = 0; c < bc; c++) {
					float sum = 0;
					#pragma unroll
					for (int k = 0; k < bc; k++)
						sum = fmaf(rvec[k],  rmmat[k*bc+c], sum);

					float add = exBf * umat[ii(uofs, bc, r, i, bth, bthc)] * rmmat[i*bc+c] * x;
					o2mat[mi(o2ofs, bc, i, r, c, bth, bthc)] = fmaf(sum, exAf, add);
				}
			}
		}

		// update o3
		for (int i = 0; i < bc; i++) {
			for (int j = 0; j < bc; j++) {

				// o3 = o3 * M
				for (int r = 0; r < bc; r++) {

					#pragma unroll
					for (int c = 0; c < bc; c++)
						rvec[c] = o3mat[mij(o3ofs, bc, i, j, r, c, bth, bthc)];

					#pragma unroll
					for (int c = 0; c < bc; c++) {
						float sum = 0;
						#pragma unroll
						for (int k = 0; k < bc; k++)
							sum = fmaf( rvec[k], rmmat[k*bc+c], sum);

						float add = exBf*umat[ii(uofs, bc, r, i, bth, bthc)] * rmmat[i*bc+j]*addf * (c==j ? 1.0f : 0.0f);
						o3mat[mij(o3ofs, bc, i, j, r, c, bth, bthc)] = fmaf(sum, exAf, add);
					}
				}
			}
		}

		// update u
		max = 0.0f;
		for (int r = 0; r < bc; r++) {
			#pragma unroll
			for (int c = 0; c < bc; c++)
				rvec[c] = umat[ii(uofs, bc, r, c, bth, bthc)];

			for (int c = 0; c < bc; c++) {
				float sum = 0.0f;
				#pragma unroll
				for (int k = 0; k < bc; k++)
					sum = fmaf( rvec[k],  rmmat[k*bc+c], sum);

				umat[ii(uofs, bc, r, c, bth, bthc)] = sum;
				max = MAXX(max, sum);
			}
		}
		int ex = LOG2(max);
		float exf = POW2(-ex);
		for (int r = 0; r < bc; r++)
			for (int c = 0; c < bc; c++)
				 umat[ii(uofs, bc, r, c, bth, bthc)] *= exf;

		prev_uex = uex;
		uex = uex + mex + ex;
	}

	for (int r = 0; r < bc; r++)
		for (int c = 0; c < bc; c++)
	    umat2[l*bc*bc + r*bc + c] = umat[ii(uofs, bc, r, c, bth, bthc)];

	o2matex[l] = prev_uex;
	o3matex[l] = prev_uex;

	umatex[l] = uex;
}


#define MAX_ACTIVE_BLOCKS__VEC_O2_O3(bc, bc2) \
 cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, \
                                                 cups::kernel_vec_o2_o3<bc, bc2>, h,\
                                                 0);

#define KERNEL_VEC_O2_O3(bc, bc2, grid_dim, block_dim) cups::kernel_vec_o2_o3<bc, bc2><<<grid_dim, block_dim, 0>>>(\
    T,\
    K,\
    L,\
    dv_la, dv_laex,\
    dv_uvec, dv_vec, dv_mmat,\
    dv_timeArr,\
    storef, dv_fvec,\
    dv_o2vec, dv_o2vecex,\
    dv_o3vec, dv_o3vecex);

#define MAX_ACTIVE_BLOCKS__VEC_O2_O3_F(bc, bc2) \
 cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, \
                                                 cups::kernel_vec_o2_o3_f<bc, bc2>, h,\
                                                 0);

#define KERNEL_VEC_O2_O3_F(bc, bc2, grid_dim, block_dim) cups::kernel_vec_o2_o3_f<bc, bc2><<<grid_dim, block_dim, 0>>>(\
    T,\
    K,\
    L,\
    dv_la, dv_laex,\
    dv_uvec, dv_vec, dv_mmat,\
    dv_timeArr,\
    storef, dv_fvec,\
    dv_o2vec, dv_o2vecex,\
    dv_o3vec, dv_o3vecex);

template<int bc, int bc2>
__global__ void
kernel_vec_o2_o3(
    int T,
    int K,
    int L,
    float *la, int *laex,
    float *uvec_, float *vec, float *mmat,
    float *timeArr,
    bool storef, float *fvec,
    float *o2vec, int *o2vecex,
    float *o3vec, int *o3vecex
){

	int l = threadIdx.x + blockDim.x*blockIdx.x;
	if(l >= L)
		return;

	int bth = threadIdx.x;
	int bthc = blockDim.x;

	int v3ofs = bthc * blockIdx.x * bc * bc * bc;
	int v2ofs = bthc * blockIdx.x * bc * bc;
	int tofs = bthc * blockIdx.x * K;

	float rvec[bc];
	float ruvec[bc];
	float rmmat[bc2];

	float max, val, exf;
	int ex, mex;

	int prev_uex;
	int uex = laex[l];
	#pragma unroll
	for (int k = 0; k < bc; k++)
		ruvec[k] = la[l*bc+k];

	int zi = l * K;
	int ze = zi + K;
	ze = (ze > T ? T : ze);

	for (int z = zi, zz=0; z < ze; z++,zz++) {
		float x = timeArr[tofs + zz*bthc + bth];
		float addf = (z+1 != T ? 1.0f : 0.0f);
		// calc M
		max = 0.0f;
		#pragma unroll
		for (int r = 0; r < bc; r++) {
//        float fval =  (storef ? fvec[fofs + (zz*bc+r)*bthc + bth] : f(cm_ri[r], cm_lambdaArr[r], x));
			float fval =  f(cm_ri[r], cm_lambdaArr[r], x);

			#pragma unroll
			for (int c = 0; c < bc; c++){
				val = cm_pArr[r * bc + c] * fval;
				/*mmat[ii(uofs, bc, r, c, bth, bthc)]*/ rmmat[r*bc+c] = val;
//          max = MAXX(max, val);
		 }
		}
		mex = 0;

		int exA = prev_uex + mex;
		int exB = uex + mex;

		int exA_diff = exA - uex;
		int exB_diff = exB - uex;

		float exAf = POW2(exA_diff);
		float exBf = POW2(exB_diff);

		// update o2
		for (int i = 0; i < bc; i++) {

			// o2 = o2 * M
			#pragma unroll
			for (int c = 0; c < bc; c++)
				rvec[c] = o2vec[vi(v2ofs, bc, i, c, bth, bthc)];

			#pragma unroll
			for (int c = 0; c < bc; c++) {
				float sum = 0.0f;
				#pragma unroll
				for (int k = 0; k < bc; k++)
					sum = fmaf(rvec[k], rmmat[k*bc+c], sum);

				float add = exBf * ruvec[i] * rmmat[i*bc+c] * x;
				o2vec[vi(v2ofs, bc, i, c, bth, bthc)] = fmaf(sum, exAf, add);
			}
  	}

		// update o3
		for (int i = 0; i < bc; i++) {
			for (int j = 0; j < bc; j++) {

				// o3 = o3 * M
				#pragma unroll
				for (int c = 0; c < bc; c++)
					rvec[c] = o3vec[vij(v3ofs, bc, i, j, c, bth, bthc)];

				#pragma unroll
				for (int c = 0; c < bc; c++) {
					float sum = 0.0f;
					#pragma unroll
					for (int k = 0; k < bc; k++)
						sum = fmaf(rvec[k], rmmat[k*bc+c], sum);

					float add = exBf * ruvec[i] * rmmat[i*bc+j] * addf * (c == j ? 1.0f : 0.0f);
					o3vec[vij(v3ofs, bc, i, j, c, bth, bthc)] = fmaf(sum, exAf, add);
				}
			}
		}
		#pragma unroll
		for (int c = 0; c < bc; c++)
			rvec[c] = ruvec[c];

		max = 0.0f;
		#pragma unroll
		for (int c = 0; c < bc; c++) {
			float sum = 0.0f;
			#pragma unroll
			for (int k = 0; k < bc; k++)
				sum = fmaf(rvec[k], rmmat[k*bc+c], sum);

			ruvec[c] = sum;
			max = MAXX(max, sum);
		}
		ex = LOG2(max);
		exf = POW2(-ex);
		#pragma unroll
		for(int c = 0; c < bc; c++)
			ruvec[c] *= exf;

		prev_uex = uex;
		uex = uex + mex + ex;
	}
	o2vecex[l] = prev_uex;
	o3vecex[l] = prev_uex;

}

template<int bc, int bc2>
__global__ void
kernel_vec_o2_o3_f(
    int T,
    int K,
    int L,
    float *la, int *laex,
    float *uvec_, float *vec, float *mmat,
    float *timeArr,
    bool storef, float *fvec,
    float *o2vec, int *o2vecex,
    float *o3vec, int *o3vecex
){

	int l = threadIdx.x + blockDim.x*blockIdx.x;
	if(l >= L)
		return;

	int bth = threadIdx.x;
	int bthc = blockDim.x;

	int v3ofs = bthc * blockIdx.x * bc * bc * bc;
	int v2ofs = bthc * blockIdx.x * bc * bc;
	int fofs = bthc * blockIdx.x * K * bc;
	int tofs = bthc * blockIdx.x * K;

	float rvec[bc];
	float ruvec[bc];
	float rmmat[bc2];

	float max, val, exf;
	int ex, mex;

	int prev_uex;
	int uex = laex[l];
	#pragma unroll
	for (int k = 0; k < bc; k++)
		ruvec[k] = la[l*bc+k];

	int zi = l * K;
	int ze = zi + K;
	ze = (ze > T ? T : ze);

	for (int z = zi, zz=0; z < ze; z++,zz++) {
		float x = timeArr[tofs + zz*bthc + bth];
		float addf = (z+1 != T ? 1.0f : 0.0f);
		// calc M
		max = 0.0f;
		#pragma unroll
		for (int r = 0; r < bc; r++) {
			float fval =  fvec[fofs + (zz*bc+r)*bthc + bth];
			#pragma unroll
			for (int c = 0; c < bc; c++){
				val = cm_pArr[r * bc + c] * fval;
				rmmat[r*bc+c] = val;
		 }
		}
		mex = 0;

		int exA = prev_uex + mex;
		int exB = uex + mex;

		int exA_diff = exA - uex;
		int exB_diff = exB - uex;

		float exAf = POW2(exA_diff);
		float exBf = POW2(exB_diff);

		// update o2
		for (int i = 0; i < bc; i++) {

			// o2 = o2 * M
			#pragma unroll
			for (int c = 0; c < bc; c++)
				rvec[c] = o2vec[vi(v2ofs, bc, i, c, bth, bthc)];

			#pragma unroll
			for (int c = 0; c < bc; c++) {
				float sum = 0.0f;
				#pragma unroll
				for (int k = 0; k < bc; k++)
					sum = fmaf(rvec[k], rmmat[k*bc+c], sum);

				float add = exBf * ruvec[i] * rmmat[i*bc+c] * x;
				o2vec[vi(v2ofs, bc, i, c, bth, bthc)] = fmaf(sum, exAf, add);
			}
  	}

		// update o3
		for (int i = 0; i < bc; i++) {
			for (int j = 0; j < bc; j++) {

				// o3 = o3 * M
				#pragma unroll
				for (int c = 0; c < bc; c++)
					rvec[c] = o3vec[vij(v3ofs, bc, i, j, c, bth, bthc)];

				#pragma unroll
				for (int c = 0; c < bc; c++) {
					float sum = 0.0f;
					#pragma unroll
					for (int k = 0; k < bc; k++)
						sum = fmaf(rvec[k], rmmat[k*bc+c], sum);

					float add = exBf * ruvec[i] * rmmat[i*bc+j] * addf * (c == j ? 1.0f : 0.0f);
					o3vec[vij(v3ofs, bc, i, j, c, bth, bthc)] = fmaf(sum, exAf, add);
				}
			}
		}
		#pragma unroll
		for (int c = 0; c < bc; c++)
			rvec[c] = ruvec[c];

		max = 0.0f;
		#pragma unroll
		for (int c = 0; c < bc; c++) {
			float sum = 0.0f;
			#pragma unroll
			for (int k = 0; k < bc; k++)
				sum = fmaf(rvec[k], rmmat[k*bc+c], sum);

			ruvec[c] = sum;
			max = MAXX(max, sum);
		}
		ex = LOG2(max);
		exf = POW2(-ex);
		#pragma unroll
		for(int c = 0; c < bc; c++)
			ruvec[c] *= exf;

		prev_uex = uex;
		uex = uex + mex + ex;
	}
	o2vecex[l] = prev_uex;
	o3vecex[l] = prev_uex;

}

#define MAX_ACTIVE_BLOCKS__STORE_F(bc, bc2) \
 cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, \
                                                 cups::kernel_store_f<bc, bc2>, h,\
                                                 0);

#define KERNEL_STORE_F(bc, bc2, grid_dim, block_dim) cups::kernel_store_f<bc,bc2><<<grid_dim, block_dim>>>(\
    T,\
    L,\
    K,\
    dv_umat, dv_umatex,\
    dv_mmat,\
    dv_vec,\
    dv_timeArr,\
    storef, dv_fvec\
  );

template<int bc, int bc2>
__global__ void
kernel_store_f(
    int T,
    int L,
    int K,
    float *umat, int *umatex,
    float *mmat,
    float *vec,
    float *timeArr,
    bool storef, float *fvec
){

	int l = threadIdx.x + blockDim.x*blockIdx.x;

	if(l >= L)
		return;

	int bth = threadIdx.x;
	int bthc = blockDim.x;
	int fofs = bthc * blockIdx.x * K * bc;
	int tofs = bthc * blockIdx.x * K;

	int zi = l * K;
	int ze = zi + K;
	ze = (ze > T ? T : ze);

	for (int z = zi, zz = 0; z < ze; z++, zz++) {
		float x = timeArr[tofs + zz*bthc + bth];

		int idx = fofs + zz*bc*bthc + bth;
		#pragma unroll
		for(int r = 0; r < bc; r++){
			float lambda = cm_lambdaArr[r];
			int ri = cm_ri[r]-1;
			fvec[idx] = (powf(lambda * x, ri) * cm_facinv[ri]) * lambda * expf(-lambda * x);
			idx += bthc;
		}
	}
}

#define MAX_ACTIVE_BLOCKS__UMAT(bc, bc2) \
 cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, \
                                                 cups::kernel_umat<bc, bc2>, h,\
                                                 0);


#define KERNEL_UMAT(bc, bc2, grid_dim, block_dim) cups::kernel_umat<bc,bc2><<<grid_dim, block_dim>>>(\
    T,\
    L,\
    K,\
    dv_umat, dv_umat2, dv_umatex,\
    dv_mmat,\
    dv_vec,\
    dv_timeArr,\
    storef, dv_fvec\
  );

#define MAX_ACTIVE_BLOCKS__UMAT_F(bc, bc2) \
 cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, \
                                                 cups::kernel_umat_f<bc, bc2>, h,\
                                                 0);


#define KERNEL_UMAT_F(bc, bc2, grid_dim, block_dim) cups::kernel_umat_f<bc,bc2><<<grid_dim, block_dim>>>(\
    T,\
    L,\
    K,\
    dv_umat, dv_umat2, dv_umatex,\
    dv_mmat,\
    dv_vec,\
    dv_timeArr,\
    storef, dv_fvec\
  );



template<int bc, int bc2>
__global__ void
kernel_umat(
    int T,
    int L,
    int K,
    float *umat, float *umat2, int *umatex,
    float *mmat,
    float *vec,
    float *timeArr,
    bool storef, float *fvec
){

	int bth = threadIdx.x;
	int bthc = blockDim.x;

	int uofs = bthc * blockIdx.x * bc * bc;
	int tofs = bthc * blockIdx.x * K;

	float rvec[bc];
	float rumat[bc2];
	float rfvec[bc];

	int l = threadIdx.x + blockDim.x*blockIdx.x;
	if(l >= L)
		return;

	int zi = l * K;
	int ze = zi + K;
	ze = (ze > T ? T : ze);

	float max;

	// U = I
	int uex = 0;
	#pragma unroll
	for (int r = 0; r < bc; r++)
		#pragma unroll
		for (int c = 0; c < bc; c++)
			rumat[r*bc+c] = r == c ? 1.0f : 0.0f;

	for (int z = zi, zz = 0; z < ze; z++, zz++) {
		float x = timeArr[tofs + zz*bthc + bth];

	  #pragma unroll
		for (int r = 0; r < bc; r++)
		  rfvec[r] = f(cm_ri[r], cm_lambdaArr[r], x);

		// update u
		max = 0.0f;
		#pragma unroll
		for (int r = 0; r < bc; r++) {
			#pragma unroll
			for (int c = 0; c < bc; c++)
				rvec[c] = rumat[r*bc+c];

	  	#pragma unroll
			for (int c = 0; c < bc; c++) {
				float sum = 0.0f;
				#pragma unroll
				for (int k = 0; k < bc; k++)
					sum = fmaf(rvec[k], rfvec[k]*cm_pArr[k*bc+c], sum);

				rumat[r*bc+c] = sum;
				max = MAXX(max, sum);
			}
		}

		int ex = LOG2(max);
		float exf = POW2(-ex);

		#pragma unroll
		for (int r = 0; r < bc; r++)
			#pragma unroll
			for (int c = 0; c < bc; c++)
				 rumat[r*bc+c] *= exf;

		uex = uex + ex;
	}

	#pragma unroll
	for (int r = 0; r < bc; r++)
		#pragma unroll
		for (int c = 0; c < bc; c++)
		  umat2[l*bc*bc+r*bc+c] = umat[ii(uofs, bc, r, c, bth, bthc)] = rumat[r*bc+c];

	umatex[l] = uex;
}

template<int bc, int bc2>
__global__ void
kernel_umat_f(
    int T,
    int L,
    int K,
    float *umat, float *umat2, int *umatex,
    float *mmat,
    float *vec,
    float *timeArr,
    bool storef, float *fvec
){

	int bth = threadIdx.x;
	int bthc = blockDim.x;

	int uofs = bthc * blockIdx.x * bc * bc;
	int fofs = bthc * blockIdx.x * K * bc;

	float rvec[bc];
	float rumat[bc2];
	float rfvec[bc];

	int l = threadIdx.x + blockDim.x*blockIdx.x;
	if(l >= L)
		return;

	int zi = l * K;
	int ze = zi + K;
	ze = (ze > T ? T : ze);

	float max;

	// U = I
	int uex = 0;
	#pragma unroll
	for (int r = 0; r < bc; r++)
		#pragma unroll
		for (int c = 0; c < bc; c++)
			rumat[r*bc+c] = r == c ? 1.0f : 0.0f;

	for (int z = zi, zz = 0; z < ze; z++, zz++) {

		#pragma unroll
		for (int r = 0; r < bc; r++)
			rfvec[r] = fvec[fofs + (zz*bc+r)*bthc + bth];

		// update u
		max = 0.0f;
		#pragma unroll
		for (int r = 0; r < bc; r++) {
			#pragma unroll
			for (int c = 0; c < bc; c++)
				rvec[c] = rumat[r*bc+c];

			#pragma unroll
			for (int c = 0; c < bc; c++) {
				float sum = 0.0f;
				#pragma unroll
				for (int k = 0; k < bc; k++)
					sum = fmaf(rvec[k], rfvec[k]*cm_pArr[k*bc+c], sum);

				rumat[r*bc+c] = sum;
				max = MAXX(max, sum);
			}
		}

		int ex = LOG2(max);
		float exf = POW2(-ex);

		#pragma unroll
		for (int r = 0; r < bc; r++)
			#pragma unroll
			for (int c = 0; c < bc; c++)
				 rumat[r*bc+c] *= exf;

		uex = uex + ex;
	}
	#pragma unroll
	for (int r = 0; r < bc; r++)
		#pragma unroll
		for (int c = 0; c < bc; c++)
			 umat2[l*bc*bc+r*bc+c] = umat[ii(uofs, bc, r, c, bth, bthc)] = rumat[r*bc+c];

	umatex[l] = uex;
}

#define MAX_ACTIVE_BLOCKS__VEC_A(bc, bc2) \
 cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, \
                                                 cups::kernel_vec_a<bc, bc2>, h,\
                                                 0);


#define KERNEL_VEC_A(bc, bc2, grid_dim, block_dim) cups::kernel_vec_a<bc,bc2><<<grid_dim, block_dim, 0>>>(\
    K,\
    T,\
    L,\
    dv_la, dv_laex,\
    dv_lb, dv_lbex,\
    dv_mmat,\
    dv_vec,\
    dv_timeArr,\
    storef, dv_fvec,\
    dv_a, dv_aex,\
    dv_b, dv_bex,\
    dv_last_a, dv_last_a_ex);

#define MAX_ACTIVE_BLOCKS__VEC_A_F(bc, bc2) \
 cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, \
                                                 cups::kernel_vec_a_f<bc, bc2>, h,\
                                                 0);


#define KERNEL_VEC_A_F(bc, bc2, grid_dim, block_dim) cups::kernel_vec_a_f<bc, bc2><<<grid_dim, block_dim, 0>>>(\
    K,\
    T,\
    L,\
    dv_la, dv_laex,\
    dv_lb, dv_lbex,\
    dv_mmat,\
    dv_vec,\
    dv_timeArr,\
    storef, dv_fvec,\
    dv_a, dv_aex,\
    dv_b, dv_bex, \
    dv_last_a, dv_last_a_ex);

#define MAX_ACTIVE_BLOCKS__VEC_B(bc, bc2) \
 cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, \
                                                 cups::kernel_vec_b<bc, bc2>, h,\
                                                 0);


#define KERNEL_VEC_B(bc, bc2, grid_dim, block_dim) cups::kernel_vec_b<bc,bc2><<<grid_dim, block_dim, 0>>>(\
    K,\
    T,\
    L,\
    dv_la, dv_laex,\
    dv_lb, dv_lbex,\
    dv_mmat,\
    dv_vec,\
    dv_timeArr,\
    storef, dv_fvec,\
    dv_a, dv_aex,\
    dv_b, dv_bex);

#define MAX_ACTIVE_BLOCKS__VEC_B_F(bc, bc2) \
 cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, \
                                                 cups::kernel_vec_b_f<bc, bc2>, h,\
                                                 0);


#define KERNEL_VEC_B_F(bc, bc2, grid_dim, block_dim) cups::kernel_vec_b_f<bc, bc2><<<grid_dim, block_dim, 0>>>(\
    K,\
    T,\
    L,\
    dv_la, dv_laex,\
    dv_lb, dv_lbex,\
    dv_mmat,\
    dv_vec,\
    dv_timeArr,\
    storef, dv_fvec,\
    dv_a, dv_aex,\
    dv_b, dv_bex);



__device__
int vab(int ofs, int bc, int vi, int i, int bth, int bthc){
  return ofs + (vi*bc + i)*bthc + bth;
}

__device__
int sab(int ofs, int vi, int bth, int bthc){
  return ofs + vi*bthc + bth;
}

__device__
int vabn(int K, int ofs, int bc, int vi, int i, int bth, int bthc){
  vi++;
  if(vi == K){
    vi = 0;
    bth++;
    if(bth == bthc){
      bth = 0;
      ofs += bthc*K*bc;
    }
  }
  return ofs + (vi*bc + i)*bthc + bth;
}


__device__
int sabn(int K, int ofs, int vi, int bth, int bthc){
  vi++;
  if(vi == K){
    vi = 0;
    bth++;
    if(bth == bthc){
      bth = 0;
      ofs += bthc*K;
    }
  }
  return ofs + vi*bthc + bth;
}

template<int bc, int bc2>
__global__ void
kernel_vec_a(
    int K,
    int T,
    int L,
    float *la, int *laex,
    float *lb, int *lbex,
    float *mmat,
    float *vec,
    float *timeArr,
    bool storef, float *fvec,
    float *a, int *aex,
    float *b, int *bex,
    float *last_a, int *last_a_ex
){

	int l = threadIdx.x + blockDim.x*blockIdx.x;
	if(l >= L)
		return;

	int bth = threadIdx.x;
	int bthc = blockDim.x;

	int fofs = bthc * blockIdx.x * K * bc;
	int tofs = bthc * blockIdx.x * K;

	float rab[bc];
	float prev_rab[bc];

	int ex;
	float exf, max;

	int zi = l * K;
	int ze = zi + K;
	ze = (ze > T ? T : ze);

	#pragma unroll
	for(int c = 0; c < bc; c++)
		a[vab(fofs, bc, 0, c, bth, bthc)] = la[l*bc+c];
  aex[sab(tofs, 0, bth, bthc)] = laex[l];

  #pragma unroll
	for (int c = 0; c < bc; c++)
	  prev_rab[c] = a[vab(fofs, bc, 0, c, bth, bthc)];
	int prev_aex = laex[l];

	// va
	for (int z = zi+1, zz = 0; z < ze; z++, zz++) {
		float x = timeArr[tofs + zz*bthc + bth];

		max = 0.0f;
		#pragma unroll
		for (int c = 0; c < bc; c++) {
			float sum = 0.0f;
      #pragma unroll
		  for (int k = 0; k < bc; k++)
		    sum = fmaf(prev_rab[k], cm_pArr[k*bc + c]*f(cm_ri[k], cm_lambdaArr[k], x), sum);

			rab[c] = sum;
			max = MAXX(max, sum);
		}
		ex = LOG2(max);
		exf = POW2(-ex);
		#pragma unroll
		for(int c = 0; c < bc; c++)
			a[vab(fofs, bc, zz+1, c, bth, bthc)] = prev_rab[c] = rab[c]*exf;
		aex[sab(tofs, zz+1, bth, bthc)] = prev_aex = prev_aex + ex;

	}

	if(l == L-1){
		int idx = ze - (zi+1);
		for(int c = 0; c < bc; c++)
			last_a[c] = a[vab(fofs, bc, idx, c, bth, bthc)];
    last_a_ex[0] = aex[sab(tofs, idx, bth, bthc)];
  }
}

template<int bc, int bc2>
__global__ void
kernel_vec_a_f(
    int K,
    int T,
    int L,
    float *la, int *laex,
    float *lb, int *lbex,
    float *mmat,
    float *vec,
    float *timeArr,
    bool storef, float *fvec,
    float *a, int *aex,
    float *b, int *bex,
    float *last_a, int *last_a_ex
){

	int l = threadIdx.x + blockDim.x*blockIdx.x;
	if(l >= L)
		return;

	int bth = threadIdx.x;
	int bthc = blockDim.x;

	int fofs = bthc * blockIdx.x * K * bc;
	int tofs = bthc * blockIdx.x * K;

	float rab[bc];
	float prev_rab[bc];

	int ex;
	float exf, max;

	int zi = l * K;
	int ze = zi + K;
	ze = (ze > T ? T : ze);

	#pragma unroll
	for(int c = 0; c < bc; c++)
		a[vab(fofs, bc, 0, c, bth, bthc)] = la[l*bc+c]/*la[l*bc+c]*/;
	  aex[sab(tofs, 0, bth, bthc)] = laex[l];

	#pragma unroll
	for (int c = 0; c < bc; c++)
	  prev_rab[c] = a[vab(fofs, bc, 0, c, bth, bthc)];
	int prev_aex = laex[l];

	// va
	for (int z = zi+1, zz = 0; z < ze; z++, zz++) {

		max = 0.0f;
		#pragma unroll
		for (int c = 0; c < bc; c++) {
			float sum = 0.0f;
			#pragma unroll
			for (int k = 0; k < bc; k++)
				sum = fmaf(prev_rab[k], cm_pArr[k*bc + c]*fvec[fofs + (zz*bc+k)*bthc + bth], sum);

			rab[c] = sum;
			max = MAXX(max, sum);
		}
		ex = LOG2(max);
		exf = POW2(-ex);
		#pragma unroll
		for(int c = 0; c < bc; c++)
			a[vab(fofs, bc, zz+1, c, bth, bthc)] = prev_rab[c] = rab[c]*exf;
		aex[sab(tofs, zz+1, bth, bthc)] = prev_aex = prev_aex + ex;

	}

  /* This is for P_3, P_3_D. */
	if(l == L-1){
		int idx = ze - (zi+1);
		for(int c = 0; c < bc; c++)
			last_a[c] = a[vab(fofs, bc, idx, c, bth, bthc)];
		last_a_ex[0] = aex[sab(tofs, idx, bth, bthc)];
  }
}



template<int bc, int bc2>
__global__ void
kernel_vec_b(
    int K,
    int T,
    int L,
    float *la, int *laex,
    float *lb, int *lbex,
    float *mmat,
    float *vec,
    float *timeArr,
    bool storef, float *fvec,
    float *a, int *aex,
    float *b, int *bex
) {

	int l = threadIdx.x + blockDim.x*blockIdx.x;
	if(l >= L)
		return;

	int bth = threadIdx.x;
	int bthc = blockDim.x;

	int fofs = bthc * blockIdx.x * K * bc;
	int tofs = bthc * blockIdx.x * K;

	float rab[bc];
	float prev_rab[bc];

	int ex;
	float exf, max;

	int zi = l * K;
	int ze = zi + K;
	ze = (ze > T ? T : ze);

	#pragma unroll
	for(int r = 0; r < bc; r++)
		prev_rab[r] = lb[(l+1)*bc+r];
	int prev_bex = lbex[l+1];

	// vb
	for (int z = ze - 1; z >= zi; z--) {
		int zz = z%K;
		float x = timeArr[tofs + zz*bthc + bth];

		max = 0.0f;
		#pragma unroll
		for (int r = 0; r < bc; r++) {
			float sum = 0.0f;
			float fval =  f(cm_ri[r], cm_lambdaArr[r], x);

			#pragma unroll
			for (int k = 0; k < bc; k++)
				sum = fmaf( cm_pArr[r * bc + k]*fval, prev_rab[k] /*vec[v(l, bc, k)]*/, sum);
			rab[r] = sum;
			max = MAXX(sum, max);
		}
		ex = LOG2(max);
		exf = POW2(-ex);

		#pragma unroll
		for(int c = 0; c < bc; c++)
			b[vab(fofs, bc, zz, c, bth, bthc)] = prev_rab[c] = rab[c]*exf;
		bex[sab(tofs, zz, bth, bthc)] = prev_bex = prev_bex + ex;
	}
}

template<int bc, int bc2>
__global__ void
kernel_vec_b_f(
    int K,
    int T,
    int L,
    float *la, int *laex,
    float *lb, int *lbex,
    float *mmat,
    float *vec,
    float *timeArr,
    bool storef, float *fvec,
    float *a, int *aex,
    float *b, int *bex
){

	int l = threadIdx.x + blockDim.x*blockIdx.x;
	if(l >= L)
		return;

	int bth = threadIdx.x;
	int bthc = blockDim.x;

	int fofs = bthc * blockIdx.x * K * bc;
	int tofs = bthc * blockIdx.x * K;

	float rab[bc];
	float prev_rab[bc];

	int ex;
	float exf, max;

	int zi = l * K;
	int ze = zi + K;
	ze = (ze > T ? T : ze);

	#pragma unroll
	for(int r = 0; r < bc; r++)
		prev_rab[r] = lb[(l+1)*bc+r];
	int prev_bex = lbex[l+1];


	// vb
	for (int z = ze - 1; z >= zi; z--) {
		int zz = z%K;

		max = 0.0f;
		#pragma unroll
		for (int r = 0; r < bc; r++) {
			float sum = 0.0f;
			float fval =  fvec[fofs + (zz*bc+r)*bthc + bth];

			#pragma unroll
			for (int k = 0; k < bc; k++)
				sum = fmaf( cm_pArr[r * bc + k]*fval, prev_rab[k] /*vec[v(l, bc, k)]*/, sum);
			rab[r] = sum;
			max = MAXX(sum, max);
		}
		ex = LOG2(max);
		exf = POW2(-ex);

		#pragma unroll
		for(int c = 0; c < bc; c++)
			b[vab(fofs, bc, zz, c, bth, bthc)] = prev_rab[c] = rab[c]*exf;
		bex[sab(tofs, zz, bth, bthc)] = prev_bex = prev_bex + ex;
	}
}

#define MAX_ACTIVE_BLOCKS__ARR_S2_S3(bc, bc2) \
 cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, \
                                                 cups::kernel_arr_s2_s3<bc, bc2>, h,\
                                                 0);


#define KERNEL_ARR_S2_S3(bc, bc2, grid_dim, block_dim) cups::kernel_arr_s2_s3<bc,bc2><<<grid_dim, block_dim, 0>>>(\
    T,\
    K,\
    L,\
    dv_a, dv_aex,\
    dv_b, dv_bex,\
    dv_mmat,\
    dv_timeArr,\
    dv_s2arr, dv_s2arrex,\
    dv_s3arr, dv_s3arrex,\
    storef, dv_fvec);

#define MAX_ACTIVE_BLOCKS__ARR_S2_S3_F(bc, bc2) \
 cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, \
                                                 cups::kernel_arr_s2_s3_f<bc, bc2>, h,\
                                                 0);


#define KERNEL_ARR_S2_S3_F(bc, bc2, grid_dim, block_dim) cups::kernel_arr_s2_s3_f<bc,bc2><<<grid_dim, block_dim, 0>>>(\
    T,\
    K,\
    L,\
    dv_a, dv_aex,\
    dv_b, dv_bex,\
    dv_mmat,\
    dv_timeArr,\
    dv_s2arr, dv_s2arrex,\
    dv_s3arr, dv_s3arrex,\
    storef, dv_fvec);


template<int bc, int bc2>
__global__ void
kernel_arr_s2_s3(
    int T,
    int K,
    int L,
    float *a, int *aex,
    float *b, int *bex,
    float *mmat,
    float *timeArr,
    float *s2arr, int *s2arrex,
    float *s3arr, int *s3arrex,
    bool storef, float *fvec
){

	int exv;
	int exm;

	float exfv;
	float exfm;

	int l = threadIdx.x + blockDim.x*blockIdx.x;
	if(l >= L)
		return;

	int bth = threadIdx.x;
	int bthc = blockDim.x;

	int fofs = bthc * blockIdx.x * K * bc;
	int tofs = bthc * blockIdx.x * K;

	float rsarrm[bc2];
	float rsarrv[bc];

	float rvec[bc];

	int zi = l * K;
	int ze = zi + K;
	ze = (ze > T ? T : ze);


	exv = INT_MIN;
	exm = INT_MIN;
	for (int z = zi, zz=0; z < ze; z++, zz++){
		int aexval = aex[sab(tofs, zz, bth, bthc)];
		exv = MAXX(exv, aexval+bex[sab(tofs, zz, bth, bthc)]);
		if(z<T-1)
			exm = MAXX(exm, aexval+bex[sabn(K, tofs, zz, bth, bthc)]);
	}
	s2arrex[l] = exv;
	s3arrex[l] = exm;

	#pragma unroll
	for (int i = 0; i < bc; i++)
		#pragma unroll
		for (int j = 0; j < bc; j++)
			rsarrm[i*bc+j] = 0.0f;

	#pragma unroll
	for (int i = 0; i < bc; i++)
			rsarrv[i] = 0.0f;

	for (int z = zi, zz = 0; z < ze; z++, zz++) {
		float x = timeArr[tofs + zz*bthc + bth];
		int aexval = aex[sab(tofs, zz, bth, bthc)];
		exfm = POW2(aexval + bex[sabn(K, tofs, zz, bth, bthc)] - exm);
		exfv = POW2(aexval + bex[sab(tofs, zz, bth, bthc)] - exv);


		#pragma unroll
		for(int j = 0; j < bc; j++)
			 rvec[j] = b[vabn(K, fofs, bc, zz, j, bth, bthc)];

		#pragma unroll
	  for (int i = 0; i < bc; i++) {
		  float fval = f(cm_ri[i], cm_lambdaArr[i], x);

		  float aval = a[vab(fofs, bc, zz, i, bth, bthc)];
		  rsarrv[i] = fmaf(exfv, aval  * x * b[vab(fofs, bc, zz, i, bth, bthc)], rsarrv[i]);

			if(z < T-1){
		  	#pragma unroll
				for (int j = 0; j < bc; j++)
		      rsarrm[i*bc+j] = fmaf(exfm, (fval*aval) * cm_pArr[i * bc + j] * rvec[j], rsarrm[i*bc+j]);
		  }
		}
	}

	#pragma unroll
	for (int i = 0; i < bc; i++)
		#pragma unroll
		for (int j = 0; j < bc; j++)
			s3arr[l*bc*bc + i*bc + j] = rsarrm[i*bc+j];

	#pragma unroll
	for (int i = 0; i < bc; i++)
		s2arr[l*bc + i] = rsarrv[i];

}

template<int bc, int bc2>
__global__ void
kernel_arr_s2_s3_f(
    int T,
    int K,
    int L,
    float *a, int *aex,
    float *b, int *bex,
    float *mmat,
    float *timeArr,
    float *s2arr, int *s2arrex,
    float *s3arr, int *s3arrex,
    bool storef, float *fvec
) {

	int exv;
	int exm;

	float exfv;
	float exfm;

	int l = threadIdx.x + blockDim.x*blockIdx.x;
	if(l >= L)
		return;

	int bth = threadIdx.x;
	int bthc = blockDim.x;

	int fofs = bthc * blockIdx.x * K * bc;
	int tofs = bthc * blockIdx.x * K;


	float rsarrm[bc2];
	float rsarrv[bc];

	float rvec[bc];

	int zi = l * K;
	int ze = zi + K;
	ze = (ze > T ? T : ze);

	exv = INT_MIN;
	exm = INT_MIN;
	for (int z = zi, zz=0; z < ze; z++, zz++){
			int aexval = aex[sab(tofs, zz, bth, bthc)];
			exv = MAXX(exv, aexval+bex[sab(tofs, zz, bth, bthc)]);
			if(z<T-1)
				exm = MAXX(exm, aexval+bex[sabn(K, tofs, zz, bth, bthc)]);
	}
	s2arrex[l] = exv;
	s3arrex[l] = exm;

	#pragma unroll
	for (int i = 0; i < bc; i++)
		#pragma unroll
		for (int j = 0; j < bc; j++)
			rsarrm[i*bc+j] = 0.0f;

	#pragma unroll
	for (int i = 0; i < bc; i++)
			rsarrv[i] = 0.0f;


	for (int z = zi, zz = 0; z < ze; z++, zz++) {
		float x = timeArr[tofs + zz*bthc + bth];
		int aexval = aex[sab(tofs, zz, bth, bthc)];
		exfm = POW2(aexval + bex[sabn(K, tofs, zz, bth, bthc)] - exm);
		exfv = POW2(aexval + bex[sab(tofs, zz, bth, bthc)] - exv);

		#pragma unroll
		for(int j = 0; j < bc; j++)
			 rvec[j] = b[vabn(K, fofs, bc, zz, j, bth, bthc)];

		#pragma unroll
		for (int i = 0; i < bc; i++) {
			float fval = fvec[fofs + (zz*bc+i)*bthc + bth];

			float aval = a[vab(fofs, bc, zz, i, bth, bthc)];
			rsarrv[i] = fmaf(exfv, aval  * x * b[vab(fofs, bc, zz, i, bth, bthc)], rsarrv[i]);

			if(z < T-1){
				#pragma unroll
				for (int j = 0; j < bc; j++)
					 rsarrm[i*bc+j] = fmaf(exfm, (fval*aval) * cm_pArr[i * bc + j] * rvec[j], rsarrm[i*bc+j]);
			}
		}
	}

	#pragma unroll
	for (int i = 0; i < bc; i++)
		#pragma unroll
		for (int j = 0; j < bc; j++)
			s3arr[l*bc*bc + i*bc + j] = rsarrm[i*bc+j];

	#pragma unroll
	for (int i = 0; i < bc; i++)
		s2arr[l*bc + i] = rsarrv[i];
}

} /*namespace cups*/

/* Additional methods for ErChmmEmCuda. */

int tt(int k, int h, int K){
  int l = (k - k%K) / K;
  int bthc = h;
  int bth = l%h;
  int bidx = (l-l%h)/h;
  int tofs = bidx * bthc * K;

  int kk = k%K;

  return tofs + kk*bthc + bth;
}

int mi(int ofs, int bc, int i, int row, int col, int bth, int bthc) {
  return ofs + (i*bc*bc + row*bc + col)*bthc+bth;
}

int mij(int ofs, int bc, int i, int j, int row, int col, int bth, int bthc) {
  return ofs + (i*bc*bc*bc + j*bc*bc + row*bc + col)*bthc+bth;
}

int vi(int ofs, int bc, int i, int idx, int bth, int bthc) {
  return ofs + (i*bc + idx)*bthc + bth;
}

int vij(int ofs, int bc, int i, int j, int idx, int bth, int bthc) {
  return ofs + (i*bc*bc + j*bc + idx )*bthc + bth;
}

int factorial(int a) {
  if (a == 0 || a == 1)
    return 1;
  return a * factorial(a - 1);
}

float f(int r, float lambda, float x) {
  float factor = (float) powf(lambda * x, r - 1) / (float) factorial(r - 1);
  float value = factor * lambda * (float) expf(-lambda * x);
  return value;
}

void vec_la(
    int h, int L, int bc,
    float *la, int *laex,
    float *umat, int *umatex,
    float *alphaArr, float *cm_p2t
){

  int ex;
  float exf;
  float max = 0.0f;

  // forward vectors
  for (int c = 0; c < bc; c++){
    la[0*bc + c] = alphaArr[c];
    max = MAXX(max, la[c])
  }
  ex = LOG2(max);
  exf = POW2(-ex);
  for(int c = 0; c < bc; c++)
    la[0*bc+c] *= exf;
  laex[0] = ex;

  for (int l = 0; l < L; l++) {
    max = 0;
    for (int c = 0; c < bc; c++) {
      float sum = 0;
      for (int k = 0; k < bc; k++)
        sum += la[l*bc+k]  * umat[ l*bc*bc + k*bc + c];
      la[(l+1)*bc + c] = sum;
      max = MAXX(max, sum);
    }
    ex = LOG2(max);
    exf = POW2(-ex);
    for(int c = 0; c < bc; c++)
      la[(l+1)*bc+c]  *= exf;

    laex[l+1] = laex[l] + ex + umatex[l];
  }
}


void vec_lb(
    int h, int L, int bc,
    float *lb, int *lbex,
    float *umat, int *umatex,
    float *alphaArr, float *cm_p2t
){

  int ex;
  float exf;
  float max = 0.0f;

  // backward vectors
  for (int c = 0; c < bc; c++)
    lb[L*bc+c] = 1;
  lbex[L] = 0;

  for (int l = L - 1; l >= 0; l--) {
    max = 0;
    for (int r = 0; r < bc; r++) {
      float sum = 0;
      for (int k = 0; k < bc; k++)
        sum += umat[l*bc*bc + r*bc + k] * lb[(l+1)*bc+k];
      lb[l*bc+r] = sum;
      max = MAXX(max, sum);
    }
    ex = LOG2(max);
    exf = POW2(-ex);
    for(int r = 0; r < bc; r++)
      lb[l*bc+r]  *= exf;
    lbex[l] = lbex[l+1] + ex + umatex[l];
  }
}

void calc_lh(
    int h, int L, int bc,
    float *la, int *laex,
    float &lh, int &lhex, float *cm_p2t
){

  float val = 0;

  for (int i = 0; i < bc; i++)
    val += la[L*bc+i];

  lhex = LOG2(val);
  lh = val*POW2(-lhex);
  lhex += laex[L];
}

void s2_from_o2mat(
    int h, int L, int bc,
    float *la, int *laex,
    float *lb, int *lbex,
    float *o2mat, int *o2matex,
    float *s2, int &s2ex, float *cm_p2t
){

  int minex = INT_MAX;
  int maxex = INT_MIN;

  s2ex = INT_MIN;
  for(int l = 0; l < L; l++){
    int ex = laex[l] + lbex[l+1] + o2matex[l];
    s2ex = MAXX(s2ex, ex);

    if(ex < minex) minex = ex;
    if(ex > maxex) maxex = ex;
  }

  // s2i
  float exf;
  for (int i = 0; i < bc; i++) {

   s2[i] = 0;

    for (int l = 0; l < L; l++) {

      // for accessing o2mat[..]
      int bthc = h;
      int bth = l % bthc;
      int bidx = (l-bth) / bthc;
      int o2ofs = bthc * bidx * bc * bc * bc;

      int ex = laex[l] + lbex[l+1] + o2matex[l];
      exf = POW2(ex - s2ex);

      float sum2 = 0;
      for (int c = 0; c < bc; c++) {
        float sum = 0;
        for (int k = 0; k < bc; k++)
          sum += la[l*bc+k] * o2mat[mi(o2ofs, bc, i, k, c, bth, bthc)];
        sum2 += sum * lb[(l+1)*bc+c];
      }
      s2[i] += sum2*exf;
    }
  }
}

void s3_from_o3mat(
    int h, int L, int bc,
    float *la, int *laex,
    float *lb, int *lbex,
    float *o3mat, int *o3matex,
    float *s3, float *S3, int &s3ex, float *cm_p2t
){

  s3ex = INT_MIN;
  for(int l = 0; l < L; l++){
    int ex = laex[l] + lbex[l+1] + o3matex[l];
    s3ex = MAXX(s3ex, ex);
  }

  // s3
  float exf;
  for (int i = 0; i < bc; i++) {
    s3[i] = 0;
    for (int j = 0; j < bc; j++) {
      S3[i * bc + j] = 0;

      for (int l = 0; l < L; l++) {
        int ex = laex[l] + lbex[l+1] + o3matex[l];
        exf = POW2(ex - s3ex);

        float sum2 = 0;
        S3[i * bc + j] += sum2*exf;
      }
      s3[i] += S3[i * bc + j];
    }
  }
}

void sums_s1(
    int h, int T, int K, int L, int bc,
    float *la, int *laex, float *last_ab,
    float *s3, int &s3ex, float *s1, int &s1ex,
    float *timeArr,
    float *pArr, float *lambdaArr, int *ri,
    float *vec, float *cm_p2t
){

  int zi = (L - 1) * K;
  int ze = zi + K;
  ze = (ze > T ? T : ze);
  ze--;

  float x;

  for (int c = 0; c < bc; c++)
    last_ab[c] = la[(L-1)*bc+c];
  int last_ab_ex = laex[L-1];

  float max;
  int ex;
  float exf;

  for (int z = zi; z < ze; z++) {
    x = timeArr[tt(z, h, K)];

    for (int c = 0; c < bc; c++)
      vec[c] = last_ab[c];

    max = 0;
    for (int c = 0; c < bc; c++) {
      float sum = 0;
      for (int k = 0; k < bc; k++)
        sum += vec[k] * pArr[k * bc + c] * f(ri[k], lambdaArr[k], x);
      last_ab[c] = sum;
      max = MAXX(max, sum);
    }
    ex = LOG2(max);
    exf = POW2(-ex);
    for(int c = 0; c < bc; c++)
      last_ab[c] *= exf;
    last_ab_ex += ex;
  }

  x = timeArr[tt(T-1, h, K)];

  for (int r = 0; r < bc; r++) {
    float sum = 0;
    for (int k = 0; k < bc; k++)
      sum += pArr[r * bc + k] * f(ri[r], lambdaArr[r], x);
    last_ab[r] *= sum;
  }

  int exA = s3ex;
  int exB = last_ab_ex;
  ex = MAXX(exA, exB);
  float factorA = POW2(exA-ex);
  float factorB = POW2(exB-ex);

  for (int i = 0; i < bc; i++) {
    s1[i] = factorA*s3[i] + factorB*last_ab[i];
  }
  s1ex = ex;
}

void sums_s1_given_last_a(
    int h, int T, int K, int L, int bc,
    float *last_a, int last_a_ex, float *last_ab,
    float *s3, int &s3ex,
    float *s1, int &s1ex,
    float *timeArr,
    float *pArr, float *lambdaArr, int *ri,
    float *vec, float *cm_p2t
){

  float x = timeArr[tt(T-1, h, K)];

  for (int r = 0; r < bc; r++) {
    float sum = 0;
    for (int k = 0; k < bc; k++)
      sum += pArr[r * bc + k] * f(ri[r], lambdaArr[r], x);
    last_ab[r] = last_a[r] * sum;
  }

  int exA = s3ex;
  int exB = last_a_ex;
  int ex = MAXX(exA, exB);
  float factorA = POW2(exA-ex);
  float factorB = POW2(exB-ex);

  for (int i = 0; i < bc; i++)
    s1[i] = factorA*s3[i] + factorB*last_ab[i];

  s1ex = ex;
}

void s2_from_o2vec(
    int h, int L, int bc,
    float *lb, int *lbex,
    float *o2vec, int *o2vecex,
    float *s2, int &s2ex, float *cm_p2t
){

  s2ex = INT_MIN;
  for(int l = 0; l < L; l++){
    int ex = o2vecex[l] + lbex[l+1];
    s2ex = MAXX(s2ex, ex);
  }

  // s2
  for (int i = 0; i < bc; i++) {

    s2[i] = 0;
    for (int l = 0; l < L; l++) {

      // for accessing 'o2vec'
      int bthc = h;
      int bth = l % bthc;
      int bidx = (l-bth) / bthc;
      int v2ofs = bthc * bidx * bc * bc;

      int ex = o2vecex[l] + lbex[l+1];
      float exf = POW2(ex - s2ex);

      float sum = 0;
      for (int k = 0; k < bc; k++)
        sum += o2vec[vi(v2ofs, bc, i, k, bth, bthc)] * lb[(l+1)*bc+k];

      s2[i] += sum*exf;
    }
  }
}

void s3_from_o3vec(
    int h, int L, int bc,
    float *lb, int *lbex,
    float *o3vec, int *o3vecex,
    float *s3, float *S3, int &s3ex, float *cm_p2t
){

  s3ex = INT_MIN;
  for(int l = 0; l < L; l++){
    int ex = o3vecex[l] + lbex[l+1];
    s3ex = MAXX(s3ex, ex);
  }

    // s3
  for (int i = 0; i < bc; i++) {
    for (int j = 0; j < bc; j++) {
      S3[i * bc + j] = 0;

      for (int l = 0; l < L; l++) {

				// for accessing 'o3vec'
				int bthc = h;
				int bth = l % bthc;
				int bidx = (l-bth) / bthc;
				int v3ofs = bthc * bidx * bc * bc * bc;

        int ex = o3vecex[l] + lbex[l+1];
        float exf = POW2(ex - s3ex);

        float sum = 0;
        for (int k = 0; k < bc; k++)
          sum += o3vec[vij(v3ofs, bc, i, j, k, bth, bthc)] * lb[(l+1)*bc+k];

        S3[i * bc + j] += sum*exf;
      }
    }
  }

  for (int i = 0; i < bc; i++) {
    s3[i] = 0;
    for (int j = 0; j < bc; j++)
      s3[i] += S3[i * bc + j];
  }
}

void sums_s2_s3(
    int h, int L, int bc,
    float *s2arr, int *s2arrex,
    float *s3arr, int *s3arrex,
    float *s2, int &s2ex,
    float *s3, float *S3, int &s3ex, float *cm_p2t
){


  s2ex = INT_MIN;
  for (int l = 0; l < L; l++)
    s2ex = MAXX(s2ex, s2arrex[l]);

  s3ex = INT_MIN;
  for (int l = 0; l < L; l++)
    s3ex = MAXX(s3ex, s3arrex[l]);

  for(int i = 0; i < bc; i++)
    s2[i] = 0;

  for (int l = 0; l < L; l++){
    float exf = POW2(s2arrex[l] - s2ex);
    for (int i = 0; i < bc; i++)
     s2[i] += s2arr[l*bc+i] * exf;
  }

  for (int i = 0; i < bc; i++)
    for (int j = 0; j < bc; j++)
      S3[i*bc + j] = 0;

  for (int l = 0; l < L; l++){
    float exf = POW2(s3arrex[l] - s3ex);

    for (int i = 0; i < bc; i++)
      for (int j = 0; j < bc; j++)
        S3[i * bc + j] += s3arr[l*bc*bc + i*bc + j] * exf;
  }

  for (int i = 0; i < bc; i++){
    s3[i] = 0;
    for (int j = 0; j < bc; j++)
      s3[i] += S3[i * bc + j];
  }
}

void param_estim(
    int T, int bc,
    float lh, int lhex,
    float *s1, int s1ex,
    float *s2, int s2ex,
    float *s3, float *S3, int s3ex,
    int *ri, float *alphaArr, float *lambdaArr,
    float *pArr, float *cm_p2t
){

  float factor = POW2(s1ex - s2ex);

  for (int i = 0; i < bc; i++)
    lambdaArr[i] = factor * ( (ri[i] * s1[i]) / (s2[i]));

  for (int i = 0; i < bc; i++)
    for (int j = 0; j < bc; j++)
      pArr[i * bc + j] = S3[i * bc + j] / s3[i];

  factor = POW2(s1ex - lhex);
  for (int i = 0; i < bc; i++)
    alphaArr[i] = factor * (s1[i] / (T * lh));
}

/* The implementation of ErChmmEmCuda. */

void ErChmmEmCuda::prepare(
    int impl,
    int bc,
    int *ri,
    float *alphaArr,
    float *lambdaArr,
    float *pArr,
    int timeCount,
    float *timeArr,
    int h,
    int parSize,
    float eps,
    int minIterCount,
    int maxIterCount
){

  mEps = eps;
  mMinIterCount = minIterCount;
  mMaxIterCount = maxIterCount;

  mImpl = impl;
  mBc = bc;

  // derived
  int parCount; // number of partitions
  int parCountEx; // number of partitions, granularity of 'h'
  int timeCountEx; // number of inter-arrivals, granularity of 'parSize'

  parCount = (int) ceil( (float) timeCount / parSize);

  int parCountRem = parCount % h;
  parCountEx = parCount;
  if(parCountRem != 0)
    parCountEx = parCount + (h - parCountRem);

  int timeCountRem = timeCount % (h * parSize);
  timeCountEx = timeCount;
  if(timeCountRem != 0)
    timeCountEx = timeCount + (h * parSize - timeCountRem);

  mTimeCount = timeCount;
  mParCount = parCount;
  mParSize = parSize;
  mH = h;

  int T = timeCount;
  int K = parSize;
  int L = parCount;
  int Lex = parCountEx;
  int Tex = timeCountEx;

  // take parameters
  mRi = new int[bc];
  mLambdaArr = new float[bc];
  mAlphaArr = new float[bc];
  mPArr = new float[bc*bc];

  for(int i = 0; i < bc; i++){
    for(int j = 0; j < bc; j++)
      mPArr[i*bc+j] = pArr[i*bc+j];
    mLambdaArr[i] = lambdaArr[i];
    mAlphaArr[i] = alphaArr[i];
    mRi[i] = ri[i];
  }

  mTimeArr = new float[Tex];
  for(int i = 0; i < T; i++){
    mTimeArr[tt(i, h, K)] = timeArr[i];
  }

  // facinv
  mMaxR = 0;
  for(int i = 0; i < bc; i++)
    mMaxR = ( mRi[i] > mMaxR ? mRi[i] : mMaxR);

  mfacinv = new float[mMaxR];
  mfacinv[0] = 1;
  for(int i = 1; i < mMaxR; i++)
    mfacinv[i] = i*mfacinv[i-1];

  for(int i = 0; i < mMaxR; i++)
    mfacinv[i] =  1.0 / mfacinv[i];

  // mp2t
  int minEx = -149;
  int maxEx = 127;
  mp2t = new float[maxEx - minEx + 1];
  for (int i = minEx; i <= maxEx; i++)
    mp2t[i - minEx] = powf(2.0f, i);


  int deviceCount = 0;
  checkCudaErrors( cudaGetDeviceCount(&deviceCount));
  checkCudaErrors( cudaSetDevice(0) );
  if(deviceCount == 0){
    printf("*** there is no CUDE device\n");
    return;
  }

  /* We are not using shared memory, therefore configure for maximum L1. */
  checkCudaErrors( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );

  bc_riArr = sizeof(int) * bc;
  bc_alphaArr = sizeof(float) * bc;
  bc_lambdaArr = sizeof(float) * bc;
  bc_pArr = sizeof(float) * bc * bc;
  bc_timeArr = sizeof(float) * Tex;

  checkCudaErrors( cudaMalloc((void **)&dv_riArr, bc_riArr) );
  checkCudaErrors( cudaMalloc((void **)&dv_alphaArr, bc_alphaArr) );
  checkCudaErrors( cudaMalloc((void **)&dv_lambdaArr, bc_lambdaArr) );
  checkCudaErrors( cudaMalloc((void **)&dv_pArr, bc_pArr) );
  checkCudaErrors( cudaMalloc((void **)&dv_timeArr, bc_timeArr) );

  mla = new float[(L+1) * bc]; mlaex = new int[L+1];
  mlb = new float[(L+1) * bc]; mlbex = new int[L+1];
  mlast_ab = new float[bc];

  ms1 = new float[bc];
  ms2 = new float[bc];
  ms3 = new float[bc];
  mS3 = new float[bc * bc];

  mumat = new float[Lex * bc * bc]; mumatex = new int[L];
  mvec = new float[Lex * bc];

  bc_umat = sizeof(float) * Lex * bc * bc;  bc_umatex = sizeof(int) * L;
  bc_mmat = sizeof(float) * Lex * bc * bc;
  bc_vec = sizeof(float) * Lex * bc;

  checkCudaErrors( cudaMalloc((void **)&dv_umat, bc_umat) );
  checkCudaErrors( cudaMalloc((void **)&dv_umat2, bc_umat) );

  checkCudaErrors( cudaMalloc((void **)&dv_umatex, bc_umatex) );
  checkCudaErrors( cudaMalloc((void **)&dv_mmat, bc_mmat) );
  checkCudaErrors( cudaMalloc((void **)&dv_vec, bc_vec) );

  if (impl == P_1) {
    mo2mat = new float[Lex * bc * bc * bc];        mo2matex = new int[L];
    mo3mat = new float[Lex * bc * bc * bc * bc];   mo3matex = new int[L];

    bc_o2mat = sizeof(float) * Lex * bc * bc * bc;        bc_o2matex = sizeof(int) * L;
    bc_o3mat = sizeof(float) * Lex * bc * bc * bc * bc;   bc_o3matex = sizeof(int) * L;

		checkCudaErrors( cudaMalloc((void **)&dv_o2mat, bc_o2mat) );
		checkCudaErrors( cudaMalloc((void **)&dv_o2matex, bc_o2matex) );
		checkCudaErrors( cudaMalloc((void **)&dv_o3mat, bc_o3mat) );
		checkCudaErrors( cudaMalloc((void **)&dv_o3matex, bc_o3matex) );

	}
	if (impl == P_2 || impl == P_2_D) {

		mo2vec = new float[Lex * bc * bc];      mo2vecex = new int[L];
		mo3vec = new float[Lex * bc * bc * bc]; mo3vecex = new int[L];

		bc_uvec = sizeof(float) * Lex * bc;
		bc_o2vec = sizeof(float) * Lex * bc * bc;       bc_o2vecex = sizeof(int) * L;
		bc_o3vec = sizeof(float) * Lex * bc * bc * bc;  bc_o3vecex = sizeof(int) * L;

		checkCudaErrors( cudaMalloc((void **)&dv_uvec, bc_uvec) );
		checkCudaErrors( cudaMalloc((void **)&dv_o2vec, bc_o2vec) );
		checkCudaErrors( cudaMalloc((void **)&dv_o2vecex, bc_o2vecex) );
		checkCudaErrors( cudaMalloc((void **)&dv_o3vec, bc_o3vec) );
		checkCudaErrors( cudaMalloc((void **)&dv_o3vecex, bc_o3vecex) );

		 if (impl == P_2_D){
			 bc_fvec = sizeof(float) * Tex * bc;
			 checkCudaErrors( cudaMalloc((void **)&dv_fvec, bc_fvec) );
		 }

		 bc_la = sizeof(float) * (L+1) * bc;  bc_laex = sizeof(int) * (L+1);
		 bc_lb = sizeof(float) * (L+1) * bc;  bc_lbex = sizeof(int) * (L+1);

		 checkCudaErrors( cudaMalloc((void **)&dv_la, bc_la) );
		 checkCudaErrors( cudaMalloc((void **)&dv_laex, bc_laex) );
		 checkCudaErrors( cudaMalloc((void **)&dv_lb, bc_lb) );
		 checkCudaErrors( cudaMalloc((void **)&dv_lbex, bc_lbex) );
	}

	if (impl == P_3 || impl == P_3_D) {

		ms2arr = new float[Lex * bc];      ms2arrex = new int[L];
		ms3arr = new float[Lex * bc * bc]; ms3arrex = new int[L];

		bc_uvec = sizeof(float) * Lex * bc;
		bc_a = sizeof(float) * Tex * bc;            bc_aex = sizeof(int)*(Tex);
		bc_b = sizeof(float) * Tex * bc;            bc_bex = sizeof(int)*(Tex);
		bc_s2arr = sizeof(float) * Lex * bc;       bc_s2arrex = sizeof(int) * L;
		bc_s3arr = sizeof(float) * Lex * bc * bc;  bc_s3arrex = sizeof(int) * L;

		checkCudaErrors( cudaMalloc((void **)&dv_uvec, bc_uvec) );

		checkCudaErrors( cudaMalloc((void **)&dv_a, bc_a) );
		checkCudaErrors( cudaMalloc((void **)&dv_aex, bc_aex) );
		checkCudaErrors( cudaMalloc((void **)&dv_b, bc_b) );
		checkCudaErrors( cudaMalloc((void **)&dv_bex, bc_bex) );

		checkCudaErrors( cudaMalloc((void **)&dv_s2arr, bc_s2arr) );
		checkCudaErrors( cudaMalloc((void **)&dv_s2arrex, bc_s2arrex) );
		checkCudaErrors( cudaMalloc((void **)&dv_s3arr, bc_s3arr) );
		checkCudaErrors( cudaMalloc((void **)&dv_s3arrex, bc_s3arrex) );

		 if (impl == P_3_D){
			 bc_fvec = sizeof(float) * Tex * bc;
			 checkCudaErrors( cudaMalloc((void **)&dv_fvec, bc_fvec) );
		 }

		 bc_la = sizeof(float) * (L+1) * bc;  bc_laex = sizeof(int) * (L+1);
		 bc_lb = sizeof(float) * (L+1) * bc;  bc_lbex = sizeof(int) * (L+1);

		 checkCudaErrors( cudaMalloc((void **)&dv_la, bc_la) );
		 checkCudaErrors( cudaMalloc((void **)&dv_laex, bc_laex) );
		 checkCudaErrors( cudaMalloc((void **)&dv_lb, bc_lb) );
		 checkCudaErrors( cudaMalloc((void **)&dv_lbex, bc_lbex) );

		 last_a = new float[bc]; last_a_ex = new int[1];

		 bc_last_a = sizeof(float)*bc;

		 bc_last_a_ex = sizeof(int);

		 checkCudaErrors( cudaMalloc((void **)&dv_last_a, bc_last_a) );
		 checkCudaErrors( cudaMalloc((void **)&dv_last_a_ex, bc_last_a_ex) );
	}
}

void ErChmmEmCuda::calc(){

  int impl = mImpl;
  int T = mTimeCount;
  int K = mParSize;
  int bc = mBc;
  int h = mH;
  int L = mParCount;

  // parameters
  float *alphaArr = mAlphaArr;
  float *pArr = mPArr;
  float *lambdaArr = mLambdaArr;
  int *ri = mRi;

  float *cm_p2t = mp2t;

  // data
  float *timeArr = mTimeArr;

  float *o2mat = mo2mat; int *o2matex = mo2matex;
  float *o3mat = mo3mat; int *o3matex = mo3matex;

  float *o2vec = mo2vec; int *o2vecex = mo2vecex;
  float *o3vec = mo3vec; int *o3vecex = mo3vecex;

  float *umat = mumat; int *umatex = mumatex;
  float *vec = mvec;

  float *la = mla; int *laex = mlaex;
  float *lb = mlb; int *lbex = mlbex;

  float *s2arr = ms2arr; int *s2arrex = ms2arrex;
  float *s3arr = ms3arr; int *s3arrex = ms3arrex;

  float *last_ab = mlast_ab;

  float *s1 = ms1;  int s1ex;
  float *s2 = ms2;  int s2ex;
  float *s3 = ms3;  int s3ex;
  float *S3 = mS3;

  float logli = -FLT_MAX, ologli = -FLT_MAX;
  float log2 = log(2.0);
  float stopCriteria = log(1 + mEps);
  bool storef = impl == P_2_D || impl == P_3_D;

  // copy to device
  checkCudaErrors( cudaMemcpy(dv_timeArr, timeArr, bc_timeArr, cudaMemcpyHostToDevice) );

  checkCudaErrors( cudaMemcpyToSymbol( cups::cm_p2t, mp2t, sizeof(float)*277) );
  checkCudaErrors( cudaMemcpyToSymbol( cups::cm_facinv, mfacinv, sizeof(float)*mMaxR) );
  checkCudaErrors( cudaMemcpyToSymbol( cups::cm_ri, ri, bc_riArr ) );

  checkCudaErrors( cudaMemset(dv_umat, 0, bc_umat) );

  if (impl == P_3 || impl == P_3_D) {
    checkCudaErrors( cudaMemset(dv_s2arr, 0, bc_s2arr) );
    checkCudaErrors( cudaMemset(dv_s3arr, 0, bc_s3arr) );
  }

  int iterCounter = 0;
  for (int iter = 0; iter < mMaxIterCount + 1; iter++) {

		checkCudaErrors( cudaMemcpyToSymbol( cups::cm_lambdaArr, lambdaArr, bc_lambdaArr ) );
		checkCudaErrors( cudaMemcpyToSymbol( cups::cm_pArr, pArr, bc_pArr ) );
		checkCudaErrors( cudaMemcpyToSymbol( cups::cm_alphaArr, alphaArr, bc_alphaArr ) );

		int gridSize = ceil((double) L / (double)h);

		dim3 grid_dim(gridSize, 1, 1);
		dim3 block_dim(h, 1, 1);

    if (impl == P_1) {

		 checkCudaErrors( cudaMemset(dv_o2mat, 0, bc_o2mat) );
		 checkCudaErrors( cudaMemset(dv_o3mat, 0, bc_o3mat) );

		 if(bc == 2) KERNEL_MAT_O2_O3(2, 4,  grid_dim, block_dim);
		 if(bc == 3) KERNEL_MAT_O2_O3(3, 9, grid_dim, block_dim);
		 if(bc == 4) KERNEL_MAT_O2_O3(4, 16, grid_dim, block_dim);
		 if(bc == 5) KERNEL_MAT_O2_O3(5, 25, grid_dim, block_dim);
		 if(bc == 6) KERNEL_MAT_O2_O3(6, 36, grid_dim, block_dim);
		 if(bc == 7) KERNEL_MAT_O2_O3(7, 49, grid_dim, block_dim);
		 if(bc == 8) KERNEL_MAT_O2_O3(8, 64, grid_dim, block_dim);
		 if(bc == 9) KERNEL_MAT_O2_O3(9, 81, grid_dim, block_dim);
		 if(bc == 10) KERNEL_MAT_O2_O3(10, 100, grid_dim, block_dim);

		 checkCudaErrors( cudaMemcpy(o2mat, dv_o2mat, bc_o2mat, cudaMemcpyDeviceToHost) );
		 checkCudaErrors( cudaMemcpy(o2matex, dv_o2matex, bc_o2matex, cudaMemcpyDeviceToHost) );
		 checkCudaErrors( cudaMemcpy(o3mat, dv_o3mat, bc_o3mat, cudaMemcpyDeviceToHost) );
		 checkCudaErrors( cudaMemcpy(o3matex, dv_o3matex, bc_o3matex, cudaMemcpyDeviceToHost) );

		 checkCudaErrors( cudaMemcpy(umat, dv_umat2, bc_umat, cudaMemcpyDeviceToHost) );
		 checkCudaErrors( cudaMemcpy(umatex, dv_umatex, bc_umatex, cudaMemcpyDeviceToHost) );

    }
    if(impl == P_2 || impl == P_2_D || impl == P_3 || impl == P_3_D){

      if(storef) {

        if(bc == 2) KERNEL_STORE_F(2, 4, grid_dim, block_dim);
        if(bc == 3) KERNEL_STORE_F(3, 9, grid_dim, block_dim);
        if(bc == 4) KERNEL_STORE_F(4, 16, grid_dim, block_dim);
        if(bc == 5) KERNEL_STORE_F(5, 25, grid_dim, block_dim);
        if(bc == 6) KERNEL_STORE_F(6, 36, grid_dim, block_dim);
        if(bc == 7) KERNEL_STORE_F(7, 49, grid_dim, block_dim);
        if(bc == 8) KERNEL_STORE_F(8, 64, grid_dim, block_dim);
        if(bc == 9) KERNEL_STORE_F(9, 81, grid_dim, block_dim);
        if(bc == 10) KERNEL_STORE_F(10, 100, grid_dim, block_dim);

        if(bc == 2) KERNEL_UMAT_F(2, 4, grid_dim, block_dim);
        if(bc == 3) KERNEL_UMAT_F(3, 9, grid_dim, block_dim);
        if(bc == 4) KERNEL_UMAT_F(4, 16, grid_dim, block_dim);
        if(bc == 5) KERNEL_UMAT_F(5, 25, grid_dim, block_dim);
        if(bc == 6) KERNEL_UMAT_F(6, 36, grid_dim, block_dim);
        if(bc == 7) KERNEL_UMAT_F(7, 49, grid_dim, block_dim);
        if(bc == 8) KERNEL_UMAT_F(8, 64, grid_dim, block_dim);
        if(bc == 9) KERNEL_UMAT_F(9, 81, grid_dim, block_dim);
        if(bc == 10) KERNEL_UMAT_F(10, 100, grid_dim, block_dim);

      } else {

        if(bc == 2) KERNEL_UMAT(2, 4, grid_dim, block_dim);
        if(bc == 3) KERNEL_UMAT(3, 9, grid_dim, block_dim);
        if(bc == 4) KERNEL_UMAT(4, 16, grid_dim, block_dim);
        if(bc == 5) KERNEL_UMAT(5, 25, grid_dim, block_dim);
        if(bc == 6) KERNEL_UMAT(6, 36, grid_dim, block_dim);
        if(bc == 7) KERNEL_UMAT(7, 49, grid_dim, block_dim);
        if(bc == 8) KERNEL_UMAT(8, 64, grid_dim, block_dim);
        if(bc == 9) KERNEL_UMAT(9, 81, grid_dim, block_dim);
        if(bc == 10) KERNEL_UMAT(10, 100, grid_dim, block_dim);

      }

      checkCudaErrors( cudaMemcpy(umat, dv_umat2, bc_umat, cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaMemcpy(umatex, dv_umatex, bc_umatex, cudaMemcpyDeviceToHost) );
    }

    if (impl == P_1 || impl == P_2 || impl == P_2_D) {

			vec_la(h, L, bc, la, laex, umat, umatex, alphaArr, cm_p2t);
			vec_lb(h, L, bc, lb, lbex, umat, umatex, alphaArr, cm_p2t);

    }else if( impl == P_3 || impl == P_3_D ){

			vec_la(h, L, bc, la, laex, umat, umatex, alphaArr, cm_p2t);
			vec_lb(h, L, bc, lb, lbex, umat, umatex, alphaArr, cm_p2t);

			checkCudaErrors( cudaMemcpy(dv_la, la, bc_la, cudaMemcpyHostToDevice) );
			checkCudaErrors( cudaMemcpy(dv_laex, laex, bc_laex, cudaMemcpyHostToDevice) );

			checkCudaErrors( cudaMemcpy(dv_lb, lb, bc_lb, cudaMemcpyHostToDevice) );
			checkCudaErrors( cudaMemcpy(dv_lbex, lbex, bc_lbex, cudaMemcpyHostToDevice) );

      if(storef == false ){

        if(bc == 2) KERNEL_VEC_A(2, 4, grid_dim, block_dim);
        if(bc == 3) KERNEL_VEC_A(3, 9, grid_dim, block_dim);
        if(bc == 4) KERNEL_VEC_A(4, 16, grid_dim, block_dim);
        if(bc == 5) KERNEL_VEC_A(5, 25, grid_dim, block_dim);
        if(bc == 6) KERNEL_VEC_A(6, 36, grid_dim, block_dim);
        if(bc == 7) KERNEL_VEC_A(7, 49, grid_dim, block_dim);
        if(bc == 8) KERNEL_VEC_A(8, 64, grid_dim, block_dim);
        if(bc == 9) KERNEL_VEC_A(9, 81, grid_dim, block_dim);
        if(bc == 10) KERNEL_VEC_A(10, 100, grid_dim, block_dim);

      }else{

        if(bc == 2) KERNEL_VEC_A_F(2, 4, grid_dim, block_dim);
        if(bc == 3) KERNEL_VEC_A_F(3, 9, grid_dim, block_dim);
        if(bc == 4) KERNEL_VEC_A_F(4, 16, grid_dim, block_dim);
        if(bc == 5) KERNEL_VEC_A_F(5, 25, grid_dim, block_dim);
        if(bc == 6) KERNEL_VEC_A_F(6, 36, grid_dim, block_dim);
        if(bc == 7) KERNEL_VEC_A_F(7, 49, grid_dim, block_dim);
        if(bc == 8) KERNEL_VEC_A_F(8, 64, grid_dim, block_dim);
        if(bc == 9) KERNEL_VEC_A_F(9, 81, grid_dim, block_dim);
        if(bc == 10) KERNEL_VEC_A_F(10, 100, grid_dim, block_dim);

      }

      if(storef == false ){

        if(bc == 2) KERNEL_VEC_B(2, 4, grid_dim, block_dim);
        if(bc == 3) KERNEL_VEC_B(3, 9, grid_dim, block_dim);
        if(bc == 4) KERNEL_VEC_B(4, 16, grid_dim, block_dim);
        if(bc == 5) KERNEL_VEC_B(5, 25, grid_dim, block_dim);
        if(bc == 6) KERNEL_VEC_B(6, 36, grid_dim, block_dim);
        if(bc == 7) KERNEL_VEC_B(7, 49, grid_dim, block_dim);
        if(bc == 8) KERNEL_VEC_B(8, 64, grid_dim, block_dim);
        if(bc == 9) KERNEL_VEC_B(9, 81, grid_dim, block_dim);
        if(bc == 10) KERNEL_VEC_B(10, 100, grid_dim, block_dim);

      }else{

        if(bc == 2) KERNEL_VEC_B_F(2, 4, grid_dim, block_dim);
        if(bc == 3) KERNEL_VEC_B_F(3, 9, grid_dim, block_dim);
        if(bc == 4) KERNEL_VEC_B_F(4, 16, grid_dim, block_dim);
        if(bc == 5) KERNEL_VEC_B_F(5, 25, grid_dim, block_dim);
        if(bc == 6) KERNEL_VEC_B_F(6, 36, grid_dim, block_dim);
        if(bc == 7) KERNEL_VEC_B_F(7, 49, grid_dim, block_dim);
        if(bc == 8) KERNEL_VEC_B_F(8, 64, grid_dim, block_dim);
        if(bc == 9) KERNEL_VEC_B_F(9, 81, grid_dim, block_dim);
        if(bc == 10) KERNEL_VEC_B_F(10, 100, grid_dim, block_dim);

      }

    }


    // log-likelihood computation
    float lh;
    int lhex;

    calc_lh(h, L, bc, la, laex, lh, lhex, cm_p2t);
    ologli = logli;
    logli = log(lh) + lhex*log(2);

    // checking for stop conditions
    if (iter > mMinIterCount + 1)
      if ((logli - ologli) < stopCriteria)
        break;
    if (iter == mMaxIterCount)
       break;
    iterCounter++;

    if (impl == P_1) {
      s2_from_o2mat(h, L, bc, la, laex, lb, lbex, o2mat, o2matex, s2, s2ex, cm_p2t);
      s3_from_o3mat(h, L, bc, la, laex, lb, lbex, o3mat, o3matex, s3, S3, s3ex, cm_p2t);
      sums_s1(h, T, K, L, bc, la, laex, last_ab, s3, s3ex, s1, s1ex, timeArr, pArr, lambdaArr, ri, vec, cm_p2t);
    }

    if (impl == P_2 || impl == P_2_D) {

			checkCudaErrors( cudaMemcpy(dv_la, la, bc_la, cudaMemcpyHostToDevice) );
			checkCudaErrors( cudaMemcpy(dv_laex, laex, bc_laex, cudaMemcpyHostToDevice) );

			checkCudaErrors( cudaMemset(dv_o2vec, 0, bc_o2vec) );
			checkCudaErrors( cudaMemset(dv_o3vec, 0, bc_o3vec) );

			if(storef){

				if(bc == 2) KERNEL_VEC_O2_O3_F(2, 4, grid_dim, block_dim);
				if(bc == 3) KERNEL_VEC_O2_O3_F(3, 9, grid_dim, block_dim);
				if(bc == 4) KERNEL_VEC_O2_O3_F(4, 16, grid_dim, block_dim);
				if(bc == 5) KERNEL_VEC_O2_O3_F(5, 25, grid_dim, block_dim);
				if(bc == 6) KERNEL_VEC_O2_O3_F(6, 36, grid_dim, block_dim);
				if(bc == 7) KERNEL_VEC_O2_O3_F(7, 49, grid_dim, block_dim);
				if(bc == 8) KERNEL_VEC_O2_O3_F(8, 64, grid_dim, block_dim);
				if(bc == 9) KERNEL_VEC_O2_O3_F(9, 81, grid_dim, block_dim);
				if(bc == 10) KERNEL_VEC_O2_O3_F(10, 100, grid_dim, block_dim);

      }else {

				if(bc == 2) KERNEL_VEC_O2_O3(2, 4, grid_dim, block_dim);
				if(bc == 3) KERNEL_VEC_O2_O3(3, 9, grid_dim, block_dim);
				if(bc == 4) KERNEL_VEC_O2_O3(4, 16, grid_dim, block_dim);
				if(bc == 5) KERNEL_VEC_O2_O3(5, 25, grid_dim, block_dim);
				if(bc == 6) KERNEL_VEC_O2_O3(6, 36, grid_dim, block_dim);
				if(bc == 7) KERNEL_VEC_O2_O3(7, 49, grid_dim, block_dim);
				if(bc == 8) KERNEL_VEC_O2_O3(8, 64, grid_dim, block_dim);
				if(bc == 9) KERNEL_VEC_O2_O3(9, 81, grid_dim, block_dim);
				if(bc == 10) KERNEL_VEC_O2_O3(10, 100, grid_dim, block_dim);

      }

      checkCudaErrors( cudaMemcpy(o2vec, dv_o2vec, bc_o2vec, cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaMemcpy(o2vecex, dv_o2vecex, bc_o2vecex, cudaMemcpyDeviceToHost) );

			checkCudaErrors( cudaMemcpy(o3vec, dv_o3vec, bc_o3vec, cudaMemcpyDeviceToHost) );
			checkCudaErrors( cudaMemcpy(o3vecex, dv_o3vecex, bc_o3vecex, cudaMemcpyDeviceToHost) );

			s2_from_o2vec(h, L, bc, lb, lbex, o2vec, o2vecex, s2, s2ex, cm_p2t);
			s3_from_o3vec(h, L, bc, lb, lbex, o3vec, o3vecex, s3, S3, s3ex, cm_p2t);
			sums_s1(h, T, K, L, bc, la, laex, last_ab, s3, s3ex, s1, s1ex, timeArr, pArr, lambdaArr, ri, vec, cm_p2t);

    }
    if (impl == P_3 || impl == P_3_D) {

      if(storef){

        if(bc == 2) KERNEL_ARR_S2_S3_F(2, 4, grid_dim, block_dim);
        if(bc == 3) KERNEL_ARR_S2_S3_F(3, 9, grid_dim, block_dim);
        if(bc == 4) KERNEL_ARR_S2_S3_F(4, 16, grid_dim, block_dim);
        if(bc == 5) KERNEL_ARR_S2_S3_F(5, 25, grid_dim, block_dim);
        if(bc == 6) KERNEL_ARR_S2_S3_F(6, 36, grid_dim, block_dim);
        if(bc == 7) KERNEL_ARR_S2_S3_F(7, 49, grid_dim, block_dim);
        if(bc == 8) KERNEL_ARR_S2_S3_F(8, 64, grid_dim, block_dim);
        if(bc == 9) KERNEL_ARR_S2_S3_F(9, 81, grid_dim, block_dim);
        if(bc == 10) KERNEL_ARR_S2_S3_F(10, 100, grid_dim, block_dim);

      } else {

        if(bc == 2) KERNEL_ARR_S2_S3(2, 4, grid_dim, block_dim);
        if(bc == 3) KERNEL_ARR_S2_S3(3, 9, grid_dim, block_dim);
        if(bc == 4) KERNEL_ARR_S2_S3(4, 16, grid_dim, block_dim);
        if(bc == 5) KERNEL_ARR_S2_S3(5, 25, grid_dim, block_dim);
        if(bc == 6) KERNEL_ARR_S2_S3(6, 36, grid_dim, block_dim);
        if(bc == 7) KERNEL_ARR_S2_S3(7, 49, grid_dim, block_dim);
        if(bc == 8) KERNEL_ARR_S2_S3(8, 64, grid_dim, block_dim);
        if(bc == 9) KERNEL_ARR_S2_S3(9, 81, grid_dim, block_dim);
        if(bc == 10) KERNEL_ARR_S2_S3(10, 100, grid_dim, block_dim);

      }

			checkCudaErrors( cudaMemcpy(s2arr, dv_s2arr, bc_s2arr, cudaMemcpyDeviceToHost) );
			checkCudaErrors( cudaMemcpy(s2arrex, dv_s2arrex, bc_s2arrex, cudaMemcpyDeviceToHost) );

			checkCudaErrors( cudaMemcpy(s3arr, dv_s3arr, bc_s3arr, cudaMemcpyDeviceToHost) );
			checkCudaErrors( cudaMemcpy(s3arrex, dv_s3arrex, bc_s3arrex, cudaMemcpyDeviceToHost) );

			checkCudaErrors( cudaMemcpy(last_a, dv_last_a, bc_last_a, cudaMemcpyDeviceToHost) );
			checkCudaErrors( cudaMemcpy(last_a_ex, dv_last_a_ex, bc_last_a_ex, cudaMemcpyDeviceToHost) );

       sums_s2_s3(h, L, bc, s2arr, s2arrex, s3arr, s3arrex, s2, s2ex, s3, S3, s3ex, cm_p2t);
       sums_s1_given_last_a(h, T, K, L, bc, last_a, last_a_ex[0], last_ab, s3, s3ex, s1, s1ex, timeArr, pArr, lambdaArr, ri, vec, cm_p2t);

    }

    param_estim(T, bc, lh, lhex, s1, s1ex, s2, s2ex, s3, S3, s3ex, ri, alphaArr, lambdaArr, pArr, cm_p2t);

  }

  mLogLikelihood = logli;

}

void ErChmmEmCuda::finish(){

  delete [] mRi;
  delete [] mAlphaArr;
  delete [] mLambdaArr;
  delete [] mPArr;

  delete [] mTimeArr;

  delete [] mla;  delete [] mlaex;
  delete [] mlb;  delete [] mlbex;
  delete [] mlast_ab;

  delete [] ms1;
  delete [] ms2;
  delete [] ms3;
  delete [] mS3;

  delete [] mumat; delete [] mumatex;
  delete [] mvec;

  delete [] mp2t;
  delete [] mfacinv;

  checkCudaErrors( cudaFree(dv_riArr) );
  checkCudaErrors( cudaFree(dv_alphaArr) );
  checkCudaErrors( cudaFree(dv_lambdaArr) );
  checkCudaErrors( cudaFree(dv_pArr) );
  checkCudaErrors( cudaFree(dv_timeArr) );

  checkCudaErrors( cudaFree(dv_umat) );
  checkCudaErrors( cudaFree(dv_umat2) );

  checkCudaErrors( cudaFree(dv_umatex) );
  checkCudaErrors( cudaFree(dv_mmat) );
  checkCudaErrors( cudaFree(dv_vec) );

  if (mImpl == P_1) {
    delete [] mo2mat; delete [] mo2matex;
    delete [] mo3mat; delete [] mo3matex;

		 checkCudaErrors( cudaFree(dv_o2mat) );
		 checkCudaErrors( cudaFree(dv_o2matex) );
		 checkCudaErrors( cudaFree(dv_o3mat) );
		 checkCudaErrors( cudaFree(dv_o3matex) );
  }

  if (mImpl == P_2 || mImpl == P_2_D) {

    delete [] mo2vec; delete [] mo2vecex;
    delete [] mo3vec; delete [] mo3vecex;

    checkCudaErrors( cudaFree(dv_uvec) );
    checkCudaErrors( cudaFree(dv_o2vec) );
    checkCudaErrors( cudaFree(dv_o2vecex) );
    checkCudaErrors( cudaFree(dv_o3vec) );
    checkCudaErrors( cudaFree(dv_o3vecex) );

    if (mImpl == P_2_D)
      checkCudaErrors( cudaFree(dv_fvec) );

    checkCudaErrors( cudaFree(dv_la) );
    checkCudaErrors( cudaFree(dv_laex) );
    checkCudaErrors( cudaFree(dv_lb) );
    checkCudaErrors( cudaFree(dv_lbex) );
  }

  if (mImpl == P_3 || mImpl == P_3_D) {

    if (mImpl == P_3_D)
      checkCudaErrors( cudaFree(dv_fvec) );


    checkCudaErrors( cudaFree(dv_a) );
    checkCudaErrors( cudaFree(dv_aex) );
    checkCudaErrors( cudaFree(dv_b) );
    checkCudaErrors( cudaFree(dv_bex) );

    delete [] ms2arr; delete [] ms2arrex;
    delete [] ms3arr; delete [] ms3arrex;

    checkCudaErrors( cudaFree(dv_s2arr) );
    checkCudaErrors( cudaFree(dv_s2arrex) );
    checkCudaErrors( cudaFree(dv_s3arr) );
    checkCudaErrors( cudaFree(dv_s3arrex) );

    checkCudaErrors( cudaFree(dv_uvec) );

    checkCudaErrors( cudaFree(dv_la) );
    checkCudaErrors( cudaFree(dv_laex) );
    checkCudaErrors( cudaFree(dv_lb) );
    checkCudaErrors( cudaFree(dv_lbex) );

    delete [] last_a;
    delete [] last_a_ex;

    checkCudaErrors( cudaFree(dv_last_a) );
    checkCudaErrors( cudaFree(dv_last_a_ex) );
  }

  checkCudaErrors(cudaDeviceReset());
}

float ErChmmEmCuda::getLogLikelihood(){
  return mLogLikelihood;
}

float* ErChmmEmCuda::getAlphaArr(){
  return mAlphaArr;
}

float* ErChmmEmCuda::getLambdaArr(){
  return mLambdaArr;
}

float* ErChmmEmCuda::getPArr(){
  return mPArr;
}
