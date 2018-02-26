
#include <stdio.h>
#include "mapem.h"

/* We will test each CUDA implementation with a small example.
*  The results will be compared with ones obtained by a serial implementation.
*
* impl - the CUDA implementation to test, valid values : P_1, P_2, P_2_D, P_3, P_3_D.
*/
bool test(int impl){

  /* Number of branches. */
  int bc = 3;
  
  /* ER-CHMM structure */
  int *ri = new int[bc];
  ri[0] = 1; ri[1] = 2; ri[2] = 3;

  /* ER-CHMM initial parameters. */
  float *alphaArr = new float[bc];
  alphaArr[0] = 0.1f;
  alphaArr[1] = 0.3f;
  alphaArr[2] = 0.6f;

  float *lambdaArr = new float[bc];
  lambdaArr[0] = 1.0f;
  lambdaArr[1] = 3.0f;
  lambdaArr[2] = 7.0f;

  float *pArr = new float[bc*bc];
  pArr[0*bc+0] = 0.1f; pArr[0*bc+1] = 0.6f; pArr[0*bc+2] = 0.3f;
  pArr[1*bc+0] = 0.4f; pArr[1*bc+1] = 0.3f; pArr[1*bc+2] = 0.3f;
  pArr[2*bc+0] = 0.5f; pArr[2*bc+1] = 0.2f; pArr[2*bc+2] = 0.3f;

  /* Inter-arrivals. */
  int T = 11;
  float *timeArr = new float[T];
  timeArr[0] = 6.0f;
  timeArr[1] = 1.0f;
  timeArr[2] = 25.0f;
  timeArr[3] = 17.0f;
  timeArr[4] = 31.0f;
  timeArr[5] = 8.0f;
  timeArr[6] = 15.0f;
  timeArr[7] = 3.0f;
  timeArr[8] = 13.0f;
  timeArr[9] = 2.0f;
  timeArr[10] = 11.0f;

  /* Procedure configuration. */
  int minIterCount = 5;
  int maxIterCount = 5;
  float eps = 0.001;
  int partitionSize = 3;
  int h = 32; // number of threads per block, should be multiple of 32
  int sumVarCount = 2;
  float maxPhi = 10000;

  /* ER-CHMM fitting using serial algorithm. */

  ErChmmEm *em = new ErChmmEm();
  em->prepare(
      bc,
      ri,
      alphaArr,
      lambdaArr,
      pArr,
      T,
      timeArr, 
      minIterCount,
      maxIterCount,
      eps,
      sumVarCount,
      maxPhi
    );
 
    em->calc();

  /* ER-CHMM fitting using CUDA */

  ErChmmEmCuda *emCuda = new ErChmmEmCuda();

  emCuda->prepare(
      impl,
      bc,
      ri,
      alphaArr,
      lambdaArr,
      pArr,
      T,
      timeArr,
      h,
      partitionSize,
      eps, 
      minIterCount,
      maxIterCount
    );

  emCuda->calc();

 

  /* For results comparison, we find the max  absolute error. */


  float maxAbsError = 0; 
  float absError;

  for(int i = 0; i < bc; i++){
    absError = fabs(emCuda->getAlphaArr()[i] - em->getAlphaArr()[i]);
    if(absError > maxAbsError) maxAbsError = absError;
  }

  for(int i = 0; i < bc; i++){
    absError = fabs(emCuda->getLambdaArr()[i] - em->getLambdaArr()[i]);
    if(absError > maxAbsError) maxAbsError = absError;
  }


  for(int i = 0; i < bc*bc; i++){
    absError = fabs(emCuda->getPArr()[i] - em->getPArr()[i]);
    if(absError > maxAbsError) maxAbsError = absError;
  }

//  printf("max absolute error : %e\n", maxAbsError);

  bool passed = true;
  if(maxAbsError > 5e-7) passed = false;


  /* Clean up. */
  em->finish();      delete em;
  emCuda->finish();  delete emCuda;
  delete [] pArr;
  delete [] lambdaArr;
  delete [] alphaArr;
  delete [] timeArr;
  delete [] ri;

  return passed;

}

int main(int argc, char *argv[]){
 
  printf("Parallel algorithms for fitting Markov Arrial process fitting, 2017\n");
  printf("\n");

  
  printf("P_1   ...  %s\n", (test(P_1)?"passed":"failed"));
  printf("P_2   ...  %s\n", (test(P_2)?"passed":"failed"));
  printf("P_2_D ...  %s\n", (test(P_2_D)?"passed":"failed"));
  printf("P_3   ...  %s\n", (test(P_3)?"passed":"failed"));
  printf("P_3_D ...  %s\n", (test(P_3_D)?"passed":"failed"));


  printf("\n");

  return 0;
}
