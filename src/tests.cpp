#include "tests.h"

bool test_sole_gauss(){
  int n = 3;
  double *matA = new double[n*n];
  double *vecb = new double[n];
  double *vecx = new double[n];

  matA[0*n+0] = 1; matA[0*n+1] = 2; matA[0*n+2] = 3;
  matA[1*n+0] = -2; matA[1*n+1] = 1; matA[1*n+2] = 2;
  matA[2*n+0] = -1; matA[2*n+1] = 3; matA[2*n+2] = 1;

  vecb[0] = 14;
  vecb[1] = 6;
  vecb[2] = 8;

  sole_gauss(n, matA, vecx, vecb);

  if(vecx[0] != 1.0) return false;
  if(vecx[1] != 2.0) return false;
  if(vecx[2] != 3.0) return false;


  delete [] matA;
  delete [] vecb;
  delete [] vecx;

  return true;
}

bool test_matrix_inverse(){
  int n = 3;
  double *matA = new double[n*n];
  double *matInv = new double[n*n];


  matA[0*n+0] = 1; matA[0*n+1] = 2; matA[0*n+2] = 3;
  matA[1*n+0] = -2; matA[1*n+1] = 1; matA[1*n+2] = 2;
  matA[2*n+0] = -1; matA[2*n+1] = 3; matA[2*n+2] = 1;


  mat_inv(n, matA, matInv);

  double *matU = new double[n*n];
  mul_mat_mat(n, matInv, matA, matU);

  double absErr = 0;
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      double target = 0;
      if(i == j) target = 1;
      absErr += abs(matU[i*n+j]-target);
    }
  }

  if(absErr != 3.33066907387546962127e-16) return false;

  delete [] matA;
  delete [] matInv;
  delete [] matU;

  return true;


}

bool test_erchmm_structure_generation(){

  vector<Structure*>* allStructures = generate_all_structures(4);

  //for(int i = 0; i < allStructures->size(); i++)
  //  (*allStructures)[i]->print_out();

  // structure : 4
  // structure : 3, 1
  // structure : 2, 2
  // structure : 2, 1, 1
  // structure : 1, 1, 1, 1

  if(allStructures->size() != 5) return false;

  Structure *st = (*allStructures)[0];
  if(st->getBc() != 1) return false;
  if(st->getRi()[0] != 4) return false;

  st = (*allStructures)[1];
  if(st->getBc() != 2) return false;
  if(st->getRi()[0] != 3) return false;
  if(st->getRi()[1] != 1) return false;

  st = (*allStructures)[2];
  if(st->getBc() != 2) return false;
  if(st->getRi()[0] != 2) return false;
  if(st->getRi()[1] != 2) return false;

  st = (*allStructures)[3];
  if(st->getBc() != 3) return false;
  if(st->getRi()[0] != 2) return false;
  if(st->getRi()[1] != 1) return false;
  if(st->getRi()[2] != 1) return false;

  st = (*allStructures)[4];
  if(st->getBc() != 4) return false;
  if(st->getRi()[0] != 1) return false;
  if(st->getRi()[1] != 1) return false;
  if(st->getRi()[2] != 1) return false;
  if(st->getRi()[3] != 1) return false;

  return true;
}

bool test_stationary_prob_computation(){

  int bc = 3;
  int *ri = new int[bc];
  double *lambda = new double[bc];
  double *P = new double[bc*bc];
  ri[0] = 1;
  ri[1] = 2;
  ri[2] = 3;

  lambda[0] = 1.5;
  lambda[1] = 2.5;
  lambda[2] = 3.0;


  P[0*bc+0] = 0.2; P[0*bc+1] = 0.3; P[0*bc+2] = 0.5;
  P[1*bc+0] = 0.1; P[1*bc+1] = 0.8; P[1*bc+2] = 0.1;
  P[2*bc+0] = 0.6; P[2*bc+1] = 0.2; P[2*bc+2] = 0.2;

  ErChmm *erChmm = new ErChmm();
  erChmm->set(bc, ri, lambda, P);

  double *alpha = erChmm->obtainAlpha();

  if(alpha[0] != 0.22950819672131150817) return false;
  if(alpha[1] != 0.55737704918032793255) return false;
  if(alpha[2] != 0.21311475409836061479) return false;

  double *alpha2 = new double[bc];
  mul_vec_mat(bc, alpha, erChmm->getP(), alpha2);

  if(alpha2[0] != 0.22950819672131145266) return false;
  if(alpha2[1] != 0.55737704918032793255) return false;
  if(alpha2[2] != 0.21311475409836067030) return false;


  delete [] ri;
  delete [] lambda;
  delete [] P;
  delete erChmm;
  delete [] alpha;
  delete [] alpha2;

  return true;
}

bool test_stationary_prob_computation_for_general_map(){
  int bc = 3;
  int *ri = new int[bc];
  double *lambda = new double[bc];
  double *P = new double[bc*bc];
  ri[0] = 1;
  ri[1] = 2;
  ri[2] = 3;

  lambda[0] = 1.5;
  lambda[1] = 2.5;
  lambda[2] = 3.0;


  P[0*bc+0] = 0.2; P[0*bc+1] = 0.3; P[0*bc+2] = 0.5;
  P[1*bc+0] = 0.1; P[1*bc+1] = 0.8; P[1*bc+2] = 0.1;
  P[2*bc+0] = 0.6; P[2*bc+1] = 0.2; P[2*bc+2] = 0.2;

  ErChmm *erChmm = new ErChmm();
  erChmm->set(bc, ri, lambda, P);

  Map *map = new Map();
  map->set(erChmm);

  double *alpha = map->obtainAlpha();

  if(alpha[0] != 0.22950819672131139715) return false;
  if(alpha[1] != 0.55737704918032793255) return false;
  if(alpha[2] != -0.00000000000000000000) return false;
  if(alpha[3] != 0.21311475409836064254) return false;
  if(alpha[4] != -0.00000000000000000000) return false;
  if(alpha[5] != -0.00000000000000000000) return false;


  delete [] ri;
  delete [] lambda;
  delete [] P;
  delete erChmm;
  delete [] alpha;
  delete map;

  return true;
}

bool test_erchmm_mean_computation(){
  int bc = 3;
  int *ri = new int[bc];
  double *lambda = new double[bc];
  double *P = new double[bc*bc];
  ri[0] = 1;
  ri[1] = 2;
  ri[2] = 3;

  lambda[0] = 1.5;
  lambda[1] = 2.5;
  lambda[2] = 3.0;


  P[0*bc+0] = 0.2; P[0*bc+1] = 0.3; P[0*bc+2] = 0.5;
  P[1*bc+0] = 0.1; P[1*bc+1] = 0.8; P[1*bc+2] = 0.1;
  P[2*bc+0] = 0.6; P[2*bc+1] = 0.2; P[2*bc+2] = 0.2;

  ErChmm *erChmm = new ErChmm();
  erChmm->set(bc, ri, lambda, P);

  Map *map = new Map();
  map->set(erChmm);

  double erChmmMean = erChmm->obtainMean();
  double mapMean = map->obtainMean();

  double absErr = abs(mapMean - erChmmMean);
  if(absErr != 1.11022302462515654042e-16) return false;

  return true;
}

bool test_interarrival_generation(){
  int bc = 3;
  int *ri = new int[bc];
  double *lambda = new double[bc];
  double *P = new double[bc*bc];
  ri[0] = 1;
  ri[1] = 2;
  ri[2] = 3;

  lambda[0] = 1.5;
  lambda[1] = 2.5;
  lambda[2] = 3.0;


  P[0*bc+0] = 0.2; P[0*bc+1] = 0.3; P[0*bc+2] = 0.5;
  P[1*bc+0] = 0.1; P[1*bc+1] = 0.8; P[1*bc+2] = 0.1;
  P[2*bc+0] = 0.6; P[2*bc+1] = 0.2; P[2*bc+2] = 0.2;

  ErChmm *erChmm = new ErChmm();
  erChmm->set(bc, ri, lambda, P);

  Interarrivals *interarrivals = new Interarrivals();
  interarrivals->generate(erChmm, 100000);

  Map *map = new Map();
  map->set(erChmm);

  double meanAbsDiff = abs(interarrivals->getMean() - map->obtainMean());
  if(meanAbsDiff != 1.46993157244312833143e-04) return false;

  delete [] ri;
  delete [] lambda;
  delete [] P;
  delete erChmm;
  delete interarrivals;

  return true;
}

bool test_erchmm_to_general_map(){
  int bc = 2;

  int *ri = new int[bc];
  double *lambda = new double[bc];
  double *P = new double[bc*bc];
  ri[0] = 2;
  ri[1] = 1;

  int n = 3;

  lambda[0] = 10;
  lambda[1] = 20;


  P[0*bc+0] = 0.8; P[0*bc+1] = 0.2;
  P[1*bc+0] = 0.3; P[1*bc+1] = 0.7;

  ErChmm *erChmm = new ErChmm();
  erChmm->set(bc, ri, lambda, P);

  Map *map = new Map();
  map->set(erChmm);

  double absErr = 0;
  double *D0 = map->getD0();
  double *D1 = map->getD1();


  absErr += abs(D0[0*n+0] - (-10.0));
  absErr += abs(D0[0*n+1] - ( 10.0));
  absErr += abs(D0[0*n+2] - (  0.0));

  absErr += abs(D0[1*n+0] - (  0.0));
  absErr += abs(D0[1*n+1] - (-10.0));
  absErr += abs(D0[1*n+2] - (  0.0));

  absErr += abs(D0[2*n+0] - (  0.0));
  absErr += abs(D0[2*n+1] - (  0.0));
  absErr += abs(D0[2*n+2] - (-20.0));



  absErr += abs(D1[0*n+0] - (  0.0));
  absErr += abs(D1[0*n+1] - (  0.0));
  absErr += abs(D1[0*n+2] - (  0.0));

  absErr += abs(D1[1*n+0] - (  8.0));
  absErr += abs(D1[1*n+1] - (  0.0));
  absErr += abs(D1[1*n+2] - (  2.0));

  absErr += abs(D1[2*n+0] - (  6.0));
  absErr += abs(D1[2*n+1] - (  0.0));
  absErr += abs(D1[2*n+2] - ( 14.0));

  if(absErr != 0.0) return false;


  return true;

}

bool test_erchmm_generation(){
  int bc = 3;
  int *ri = new int[bc];
  ri[0] = 1;
  ri[1] = 3;
  ri[2] = 2;

  Structure *st = new Structure(bc, ri);

  Random *rnd = new Random();
  ErChmm *erChmm = new ErChmm();
  double mean = 5;
  erChmm->set(st, mean, rnd);

  double erChmmMean = erChmm->obtainMean();

  double absErr = abs(erChmmMean - mean);

  if(absErr != 0.0) return false;

  delete rnd;
  delete erChmm;
  delete st;

  return true;
}

/* We will test each CUDA implementation with a small example.
*  The results will be compared with ones obtained by a serial implementation.
*
* impl - the CUDA implementation to test, valid values : P_1, P_2, P_2_D, P_3, P_3_D.
*/
bool test_em(int impl){

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

    em->finish();

    //printf("serial : impl-llh = %e\n", em->getImplLogLikelihood());
    //printf("serial :      llh = %e\n", em->getLogLikelihood());



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

  emCuda->finish();

  //printf("   par : impl-llh = %e\n", emCuda->getImplLogLikelihood());
  //printf("   par :      llh = %e\n", emCuda->getLogLikelihood());



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
  em->destroy();      delete em;
  emCuda->destroy();  delete emCuda;
  delete [] pArr;
  delete [] lambdaArr;
  delete [] alphaArr;
  delete [] timeArr;
  delete [] ri;

  return passed;

}


void run_all_tests(){
  printf("\n");

  printf("test_sole_gauss ... %s\n", (test_sole_gauss()?"PASSED":"FAILED"));
  printf("test_matrix_inverse ... %s\n", (test_matrix_inverse()?"PASSED":"FAILED"));

  printf("test_erchmm_structure_generation ... %s\n", (test_erchmm_structure_generation()?"PASSED":"FAILED"));
  printf("test_stationary_prob_computation ... %s\n", (test_stationary_prob_computation()?"PASSED":"FAILED"));
  printf("test_stationary_prob_computation_for_general_map ... %s\n", (test_stationary_prob_computation_for_general_map()?"PASSED":"FAILED"));
  printf("test_erchmm_mean_computation ... %s\n", (test_erchmm_mean_computation()?"PASSED":"FAILED"));



  printf("test_interarrival_generation ... %s\n", (test_interarrival_generation()?"PASSED":"FAILED"));
  printf("test_erchmm_to_general_map ... %s\n", (test_erchmm_to_general_map()?"PASSED":"FAILED"));
  printf("test_erchmm_generation ... %s\n", (test_erchmm_generation()?"PASSED":"FAILED"));

  printf("test_em P_1 ... %s\n", (test_em(P_1)?"PASSED":"FAILED"));
  printf("test_em P_2 ... %s\n", (test_em(P_2)?"PASSED":"FAILED"));
  printf("test_em P_2_D ... %s\n", (test_em(P_2_D)?"PASSED":"FAILED"));
  printf("test_em P_3 ... %s\n", (test_em(P_3)?"PASSED":"FAILED"));
  printf("test_em P_3_D ... %s\n", (test_em(P_3_D)?"PASSED":"FAILED"));


  printf("\n");
}
