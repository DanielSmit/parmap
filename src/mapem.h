#ifndef MAP_EM_HEADER
#define MAP_EM_HEADER

#include <helper_cuda.h>
#include <cfloat>
#include <cmath>

#define P_1 1
#define P_2 2
#define P_2_D 3
#define P_3 4
#define P_3_D 5

#define MAXX(arg1, arg2) ( arg1 > arg2 ? arg1 : arg2 );
#define POW2(arg) ((arg) > 127 || (arg) < -149 ? 0 : cm_p2t[arg + 149] )
#define LOG2(arg) (logf(arg) * 1.442695040888963f )

/* Serial implementation .*/
class ErChmmEm {

  private:

    int mMinIterCount;
    int mMaxIterCount;
    float mEps;

    int mTimeCount;
    int mBc;
    int mSumVarCount;  
    float mMaxPhi;
 
    int *mRi;
    float *mLambdaArr;
    float *mPArr;
    float *mAlphaArr;
    float *mTimeArr;
 
  
    float *msumqk;
    float *msumxqk;

    float *mfMat;
    float *maMat;
    float *mbMat;
    int *maScale;
    int *mbScale;
    float *mnewP;
 
    float *mqVecCurr;

    float mLogLikelihood;

  public:

    void prepare(
      int bc,
      int *ri,
      float *alphaArr,
      float *lambdaArr,
      float *pArr,
      int timeCount,
      float *timeArr, 
      int minIterCount,
      int maxIterCount,
      float eps,
      int sumVarCount,
      float maxPhi
    );

    void calc();
    void finish();

    float getLogLikelihood();
    float* getAlphaArr();
    float* getLambdaArr();
    float* getPArr();

  private:

    float computeBranchDensity(float x, float lambda, int branchSize);
    double llhf(float *lambdaArrF, float *pArrF, float *alphaArrF);

    /* Methods for summation technique. */
    void add_vec_value(float value, float *vecArr, int i, int to);
    void sum_vec_values(float *vecArr);
    void add_mat_value(float value, float *matArr, int i, int j,  int to);
    void sum_mat_values(float *matArr);
 
};

/* Parallel implementations using CUDA. */
class ErChmmEmCuda  {

  private:

    /* Iteration stop conditions. */    

    int mMinIterCount;
    int mMaxIterCount;
    float mEps;

    /* Initial ER-CHMM struture & parameters. */

    int mBc;          // number of branches in ER-CHMM structure

    int *mRi;   
    float *mLambdaArr;
    float *mPArr;
    float *mAlphaArr;

    /* Inter-arrivals. */

    int mTimeCount;   // the number of inter-arrivals
    float *mTimeArr;  // array of inter-arrivals

    /* Execution configuration.  */

    int mParSize;     // the size of partition 
    int mImpl;        // the implementation, valid values are P_1, P_2, P_2_D, P_3, P_3_D 
    int mH;           // number of threads per block, multiple of 32 

    /* Some derived charasteristics. */

    int mMaxR;        // the number of states in the longest branch
    int mParCount;    // the number of partitions


    /* Data for execution on CPU. */

    float *mo2mat;
    float *mo3mat;

    int *mo2matex;
    int *mo3matex;

    float *mo2vec;
    float *mo3vec;

    int *mo2vecex;
    int *mo3vecex;

    float *mumat;
    int *mumatex;

    float *mvec;

    float *mla;
    float *mlb;

    int *mlaex;
    int *mlbex;

    float *ms2arr;
    float *ms3arr;

    int *ms2arrex;
    int *ms3arrex;

    float *mlast_ab;

    float *ms1;
    float *ms2;
    float *ms3;
    float *mS3;

    float *mp2t;
    float *mfacinv;

    /* Data for exectuion on GPU. */

    int *dv_riArr;       int bc_riArr;
    float *dv_alphaArr;  int bc_alphaArr;
    float *dv_lambdaArr; int bc_lambdaArr;
    float *dv_pArr;      int bc_pArr;
    float *dv_timeArr;   int bc_timeArr;
    float *dv_o2mat;     int bc_o2mat;
    float *dv_o3mat;     int bc_o3mat;
    int *dv_o2matex;     int bc_o2matex;
    int *dv_o3matex;     int bc_o3matex;
   
    float *dv_o2vec;     int bc_o2vec;
    float *dv_o3vec;     int bc_o3vec;
    int *dv_o2vecex;     int bc_o2vecex;
    int *dv_o3vecex;     int bc_o3vecex;

    float *dv_uvec;      int bc_uvec;
    float *dv_fvec; 	 int bc_fvec;

    float *dv_la;        int bc_la;
    int *dv_laex;        int bc_laex;
    float *dv_lb;        int bc_lb;
    int *dv_lbex;        int bc_lbex;

    float *dv_a;         int bc_a;
    float *dv_b;         int bc_b;
    int *dv_aex;         int bc_aex;
    int *dv_bex;         int bc_bex;


    float *dv_s2arr;        int bc_s2arr;
    float *dv_s3arr;        int bc_s3arr;
    int *dv_s2arrex;        int bc_s2arrex;
    int *dv_s3arrex;        int bc_s3arrex;
    
    float *dv_umat;      int bc_umat;     int *dv_umatex;      int bc_umatex;
    float *dv_umat2;  // for storing partition density matrices in natural indexing, for faster access on CPU
    float *dv_mmat;      int bc_mmat;
    float *dv_vec;       int bc_vec; 

    float *last_a; int *last_a_ex;
    float *dv_last_a;  int bc_last_a; int *dv_last_a_ex; int bc_last_a_ex;

    /* The log-likelihood value of obtained solution.  */

    float mLogLikelihood;
    
  public:

    void prepare(
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
    );

    void calc();
    void finish();
   
    float getLogLikelihood();
    float* getAlphaArr();
    float* getLambdaArr();
    float* getPArr();

};



#endif /* MAP_EM_HEADER */
