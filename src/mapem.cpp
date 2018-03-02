#include "mapem.h"


/** Prapares for fitting.
*
* bc - number of ER-CHMM branches,
* ri - array of number of states in ER-CHMM branches,
* alphaArr - array of initial branch probabilities,
* lambdaArr - array of transition rates in each ER-CHMM branch,
* timeCount - number of inter-arrivals,
* timeArr - array of inter-arrivals,
* minIterCount/maxIterCount - min/max number of iterations,
* eps - min relative change in likelihood values before stopping iterations,
* sumVarCount - number of variables used in summation technique.
* maxPhi - a parameter in summation technique.
*/
void ErChmmEm::prepare(
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
  ){

  mMinIterCount = minIterCount;
  mMaxIterCount = maxIterCount;
  mEps = eps;
  mMaxPhi = maxPhi;
  mSumVarCount = sumVarCount;
  mTimeCount = timeCount;
  mBc = bc;

  mRi = new int[bc];
  mLambdaArr = new float[bc];
  mAlphaArr = new float[bc];
  mPArr = new float[bc*bc];
  mTimeArr = new float[timeCount];

  for(int i = 0; i < bc; i++)
    mRi[i] = ri[i];

  for(int i = 0; i < bc; i++){
    mLambdaArr[i] = lambdaArr[i];
    mAlphaArr[i] = alphaArr[i];
    for(int j = 0; j < bc; j++)
      mPArr[i*bc+j] = pArr[i*bc+j];
  }

  for(int i = 0; i < timeCount; i++)
    mTimeArr[i] = timeArr[i];

  msumqk = new float[sumVarCount*bc];
  msumxqk = new float[sumVarCount*bc];
  mnewP = new float[sumVarCount*bc*bc];

  mfMat = new float[bc*timeCount];
  maMat = new float[bc*timeCount];
  mbMat = new float[bc*timeCount];

  maScale = new int[timeCount];
  mbScale = new int[timeCount];

  mqVecCurr = new float[bc];
}

void ErChmmEm::add_vec_value(float value, float *vecArr, int i,  int to){
  if(to+1 == mSumVarCount){
    vecArr[to*mBc + i] += value;
    return;
  }

  float mag = vecArr[to*mBc + i] / value;
  if(mag < mMaxPhi){
    vecArr[to*mBc + i] += value;
    return;
  }

  float value2 = vecArr[to*mBc + i];
  add_vec_value(value2, vecArr, i, to+1);
  vecArr[to*mBc + i] = value;
}

void ErChmmEm::sum_vec_values(float *vecArr){
  for(int sp = 1; sp < mSumVarCount; sp++)
    for(int i = 0; i < mBc; i++)
      vecArr[i] += vecArr[sp*mBc + i];
}

void ErChmmEm::add_mat_value(float value, float *matArr, int i, int j,  int to){

  if(to+1 == mSumVarCount){
    matArr[to*mBc*mBc + i*mBc + j] += value;
    return;
  }

  float mag = matArr[to*mBc*mBc + i*mBc + j] / value;
  if(mag < mMaxPhi){
    matArr[to*mBc*mBc + i*mBc + j] += value;
    return;
  }

  float value2 = matArr[to*mBc*mBc + i*mBc + j];
  add_mat_value(value2, matArr, i, j, to+1);
  matArr[to*mBc*mBc + i*mBc + j] = value;

}

void ErChmmEm::sum_mat_values(float *matArr){
  for(int sp = 1; sp < mSumVarCount; sp++)
    for(int i = 0; i < mBc; i++)
      for(int j = 0; j < mBc; j++)
        matArr[i*mBc + j] += matArr[sp*mBc*mBc + i*mBc + j];
}


void ErChmmEm::calc(){

  // parameters
  float *alphaArr = mAlphaArr;
  float *pArr = mPArr;
  float *lambdaArr = mLambdaArr;

  // data
  float *sumqk = msumqk;
  float *sumxqk = msumxqk;

  float *fMat = mfMat;
  float *aMat = maMat;
  float *bMat = mbMat;
  int *aScale = maScale;
  int *bScale = mbScale;
  float *newP = mnewP;
  float *qVecCurr = mqVecCurr;

  // structure
  int bc = mBc;
  int *ri = mRi;

  float logli = -FLT_MAX, ologli = -FLT_MAX;
  float log2 = log(2.0);
  float stopCriteria = log(1 + mEps);
  int iterCounter = 0;

  for(int iter = 0; iter < mMaxIterCount + 1; iter++) {

    // Calculate vector f (branch densities)
    for(int k = 0; k < mTimeCount; k++) {
      float x = mTimeArr[k];
      for(int m=0; m < bc; m++)
        fMat[k*bc+m] = computeBranchDensity (x, lambdaArr[m], ri[m]);
    }

    // calculate vectors a and vector b (forward and backward likelihood vectors)
    memset(aMat, 0, bc*mTimeCount*sizeof(float));
    memset(bMat, 0, bc*mTimeCount*sizeof(float));
    memset(aScale, 0, mTimeCount*sizeof(int));
    memset(bScale, 0, mTimeCount*sizeof(int));

    // vectors a
    for(int i = 0; i < bc; i++)
      for(int j=0; j < bc; j++)
        aMat[i] += alphaArr[j]*fMat[j]*pArr[j*bc+i];

    for(int k = 1; k < mTimeCount; k++) {
      int ofs = k*bc;
      float asum = 0.0;
      for(int i = 0; i < bc; i++){
        for(int j = 0; j < bc; j++)
          aMat[ofs+i] += aMat[ofs+j-bc]*fMat[ofs+j]*pArr[j*bc+i];
        asum += aMat[ofs+i];
      }
      aScale[k] = aScale[k-1];
      if(asum==0)
        break;


      int scaleDiff = ceil(log(asum) / log(2));
      aScale[k] += scaleDiff;

      float factor = pow(2.0, -scaleDiff);
      for(int i = 0; i < bc; i++)
        aMat[ofs+i] *= factor;
    }

    // vectors b
    int ofs = (mTimeCount-1)*bc;
    for(int j = 0; j < bc; j++)
      for(int i = 0; i < bc; i++)
        bMat[ofs+j] += fMat[ofs+j]*pArr[j*bc+i];

    for(int k = mTimeCount - 2; k >= 0; k--) {
      ofs = k*bc;
      float bsum = 0.0;
      for(int j = 0; j < bc; j++){
        for(int i = 0; i < bc; i++)
          bMat[ofs+j] += bMat[ofs+i+bc]*fMat[ofs+j]*pArr[j*bc+i];
        bsum += bMat[ofs+j];
      }
      bScale[k] = bScale[k+1];
      if(bsum==0)
        break;

      int scaleDiff = ceil(log(bsum) / log(2));
      bScale[k] += scaleDiff;

      float factor = pow(2.0, -scaleDiff);
      for(int i = 0; i < bc; i++)
        bMat[ofs+i] *= factor;
    }

    // store the previous log-likelihood value
    ologli = logli;

    // calculate log-likelihood
    float llhval = 0.0;
    for(int i = 0; i < bc; i++){
      llhval += alphaArr[i]*bMat[i];
    }
    logli = log(llhval) + bScale[0] * log2;

    // check for stop conditions
    if(iter > mMinIterCount + 1)
      if((logli - ologli) < stopCriteria)
        break;
    if(iter == mMaxIterCount)
      break;
    iterCounter++;

    float illh = 1.0 / llhval;

    memset(newP, 0, mSumVarCount*bc*bc*sizeof(float));
    memset(sumqk, 0, mSumVarCount*bc*sizeof(float));
    memset(sumxqk, 0,mSumVarCount*bc*sizeof(float));

    // calculate new estimates for the parameters
    for(int k = 0; k < mTimeCount; k++) {

      int ofs = k*bc;
      float x = mTimeArr[k];

      float qVecSum = 0.0;
      float normalizer;
      if(k==0)
        normalizer = ldexp (illh, bScale[1] - bScale[0]);
      else if (k < mTimeCount - 1)
        normalizer = ldexp(illh, aScale[k-1] + bScale[k+1] - bScale[0]);

      for(int m = 0; m < bc; m++) {
        float pMul = (k==0 ? alphaArr[m] : aMat[ofs+m-bc]);
        qVecSum += qVecCurr[m] = pMul * bMat[ofs+m];
        if(k < mTimeCount - 1) {
          pMul *= fMat[ofs+m] * normalizer;
          for(int j = 0; j < bc; j++){
            float value =  pMul * pArr[m*bc+j] * bMat[ofs+bc+j];
            add_mat_value(value, newP, m, j, 0);
          }
        }
      }

      for(int m = 0; m < bc; m++) {
        float value, mag;
        value = qVecCurr[m] / qVecSum;

        add_vec_value(value, sumqk, m, 0);

        value = x * qVecCurr[m] / qVecSum;
        add_vec_value(value, sumxqk, m, 0);
      }
    }

    sum_mat_values(newP);
    sum_vec_values(sumqk);
    sum_vec_values(sumxqk);

    // store new estimates
    for(int i = 0; i < bc; i++) {
      lambdaArr[i] = ri[i] * sumqk[i] / sumxqk[i];
      alphaArr[i] = sumqk[i] / mTimeCount;

      float rowSum = 0.0;
      for(int j = 0; j < bc; j++)
        rowSum += newP[i*bc+j];
      for(int j = 0; j < bc; j++)
        pArr[i*bc+j] = newP[i*bc+j] / rowSum;
    }

  }

  mImplLogLikelihood = logli;
}

void ErChmmEm::finish(){
  int bc = mBc;
  double *alphaArr = new double[bc];
  double *lambdaArr = new double[bc];
  double *pArr = new double[bc*bc];
  for(int i = 0; i < bc; i++){
    alphaArr[i] = (double)mAlphaArr[i];
    lambdaArr[i] = (double)mLambdaArr[i];

    for(int j = 0; j < bc; j++)
      pArr[i*bc+j] = mPArr[i*bc+j];
  }

  mLogLikelihood = llh(bc, mRi, lambdaArr, pArr, alphaArr, mTimeCount, mTimeArr);

  delete [] alphaArr;
  delete [] lambdaArr;
  delete [] pArr;
}

void ErChmmEm::destroy(){

  delete [] mRi;
  delete [] mLambdaArr;
  delete [] mAlphaArr;
  delete [] mPArr;

  delete [] mTimeArr;

  delete [] msumqk;
  delete [] msumxqk;
  delete [] mnewP;

  delete [] mfMat;
  delete [] maMat;
  delete [] mbMat;

  delete [] maScale;
  delete [] mbScale;


  delete [] mqVecCurr;
}

double ErChmmEm::getCpuMemoryUsage(){
  int R = mBc;
  int T = mTimeCount;
  return (T*(3*R+3) + 2*R*R + 6*R + 4)*(4.0 / 1048576.0);
}

double ErChmmEm::getGpuMemoryUsage(){
  return 0;
}

double ErChmmEm::getMemoryUsage(){
  return getCpuMemoryUsage() + getGpuMemoryUsage();
}

float ErChmmEm::computeBranchDensity(float x, float lambda, int branchSize){
  float erlFact = lambda;
  for (int n=1; n < branchSize; n++)
    erlFact *= lambda * x / n;
  return exp (-lambda * x) * erlFact;
}

float ErChmmEm::getImplLogLikelihood(){
  return mImplLogLikelihood;
}

float ErChmmEm::getLogLikelihood(){
  return mLogLikelihood;
}

float* ErChmmEm::getAlphaArr(){
  return mAlphaArr;
}

float* ErChmmEm::getLambdaArr(){
  return mLambdaArr;
}

float* ErChmmEm::getPArr(){
  return mPArr;
}

int factorial(int x){
  if(x == 0 || x == 1) return 1;
  return x*factorial(x-1);
}

double llh(int bc, int *ri, double *lambdaArr, double *pArr, double *alphaArr, int T, float *timeArr){

  int size_ik = bc * T;

  double *farr = new double[size_ik];
  double *barr = new double[size_ik];
  int *bExp = new int[size_ik];

  int timeCount = T;



  double loglikelihood = 0;

  double x;
  int r;
  double lambda;

  double runlikelihood;


  // calc farr
  double factor;
  for (int k = 0; k < timeCount; k++) {
      x = (double)timeArr[k];

      for (int i = 0; i < bc; i++) {
          lambda = lambdaArr[i];
          r = ri[i];
          factor = pow(lambda * x, r - 1) / (double) factorial(r - 1);
          farr[i*T+k/*ik(i, k)*/] = factor * lambda * exp(-lambda * x);
          //printf("farr[%d,%d] = %e\n", i, k, farr[i*T+k/*ik(i, k)*/]);
      }
  }

  // calc aarr
  int power_b;
  double sum_b;
  double tmp_b;

  // calc barr
  for (int k = timeCount - 1; k >= 0; k--) {
      sum_b = 0;

      for (int j = 0; j < bc; j++) {
          double b = 0;
          if (k == timeCount - 1) {
              b = farr[j*T+k/*ik(j, k)*/];
          } else {
              for (int i = 0; i < bc; i++)
                  b += farr[j*T+k/*ik(j, k)*/] * pArr[j*bc+i/*ij(j, i)*/] * barr[i*T+k+1/*ik(i, k + 1)*/];
          }
          barr[j*T+k/*ik(j, k)*/] = b;
          sum_b += b;
      }

      // normalize
      power_b = (int) ceil(log(sum_b) / log(2));
      bExp[k] = (k == timeCount - 1 ? power_b : bExp[k + 1] + power_b);

      tmp_b = pow(2.0, -power_b);
      for (int i = 0; i < bc; i++)
          barr[i*T+k/*ik(i, k)*/] *= tmp_b;

  }

  runlikelihood = 0;
  for (int i = 0; i < bc; i++){
      runlikelihood += alphaArr[i] * barr[i*T+0/*ik(i, 0)*/];
  }

  loglikelihood = log(runlikelihood) + bExp[0] * log(2);

  delete [] barr;
  delete [] farr;
  delete [] bExp;

  return loglikelihood;
}
