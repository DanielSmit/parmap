#include "prog.h"
#include <stdio.h>

//
// General math
//

void mat_print_out(int n, double *mat){
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++)
      printf("%f, \t", mat[i*n+j]);
    printf("\n");
  }
}

void transpose(int n, double *mat){
  double t;
  for(int i = 0; i < n; i++)
    for(int j = 0; j < i; j++){
      t = mat[i*n+j];
      mat[i*n+j] = mat[j*n+i];
      mat[j*n+i] = t;
    }
}

void mul_vec_mat(int n, double *vec, double *mat, double *out){
  for(int i = 0; i < n; i++){
    double sum = 0;
    for(int j = 0; j < n; j++)
      sum += vec[j]*mat[j*n+i];
    out[i] = sum;
  }
}

void mul_mat_mat(int n, double *matA, double *matB, double *matC){
  double sum;

  for (int row = 0; row < n; row++) {
    for (int col = 0; col < n; col++) {
      sum = 0;
      for (int k = 0; k < n; k++)
        sum += matA[row*n+k] * matB[k*n+col];
      matC[row*n+col] = sum;
    }
  }
}

void mat_unit(int n, double *mat){
  for (int row = 0; row < n; row++)
    for (int col = 0; col < n; col++)
      mat[row*n+col] = 0;

  for (int i = 0; i < n; i++)
    mat[i*n+i] = 1;
}

void mat_inv(int n, double *matA, double *matInv){
  double *matB = new double[n*n];
  mat_unit(n, matB);
  sole_gauss_mat(n, matA, matInv, matB);
  delete [] matB;
}

void mat_scale(int n, double *mat, double scale){
  for(int i = 0; i < n*n; i++)
    mat[i] *= scale;
}

double vec_sum_up(int n, double *vec){
  double sum = 0;
  for(int i = 0; i < n; i++)
    sum += vec[i];

  return sum;
}

bool sole_gauss(int n, double *matA, double *vecx, double *vecb){

  double *A = new double[n*n];
  double *x = new double[n];
  double *b = new double[n];

  int i = 0, k = 0, l = 0, j = 0;

	double rawNormA = 0;
	double rawNormB = 0;
	double s = 0;
	double z = 0;

  for(i = 0; i < n; i++){
    b[i] = vecb[i];
    for(j = 0; j < n; j++)
      A[i*n+j] = matA[i*n+j];
  }

	for (k = 0; k < n - 1; k++) {
		l = k;

		for (i = k; i < n; i++) {
			rawNormA = abs(A[l*n+k]);
			rawNormB = abs(A[i*n+k]);
			if (rawNormA < rawNormB)
				l = i;
		}

		if (abs(A[l*n+k]) == 0.)
			return false;

		if (k != l) {

			for (j = k; j < n; j++) {
				z = A[k*n+j];
				A[k*n+j] = A[l*n+j];
				A[l*n+j] = z;
			}

			z = b[k];
			b[k] = b[l];
			b[l] = z;
		}

		for (i = k + 1; i < n; i++) {
			s = A[i*n+k] / A[k*n+k];

			for (j = k + 1; j < n; j++)
				A[i*n+j] -= s * A[k*n+j];
			b[i] -= s * b[k];
		}
	}

	if (abs(A[(n - 1)*n+(n - 1)]) == 0.)
		return false;

	x[n - 1] = b[n - 1] / A[(n-1)*n+(n-1)];

	for (i = n - 2; i >= 0; i--) {
		s = 0;
		for (j = i + 1; j < n; j++)
			s += A[i*n+j] * x[j];

		x[i] = (b[i] - s) / A[i*n+i];
	}



  for(int i = 0; i < n; i++) vecx[i] = x[i];

  delete [] A;
  delete [] x;
  delete [] b;

  return true;

}

bool sole_gauss_mat(int n, double *matA, double *matX, double *matB){
  double *A = new double[n*n];
  double *X = new double[n*n];
  double *B = new double[n*n];




  int i = 0;
  int k = 0;
  int l = 0;
  int j = 0;
  int col = 0;

  double z = 0;
  double rawNormA = 0;
  double rawNormB = 0;
  double s = 0;

  for(i = 0; i < n*n; i++){
    A[i] = matA[i];
    B[i] = matB[i];
  }

  for (k = 0; k < n - 1; k++) {
    l = k;

    for (i = k; i < n; i++) {
      rawNormA = abs(A[l*n+k]);
      rawNormB = abs(A[i*n+k]);
      if (rawNormA < rawNormB)
        l = i;
    }

    if (abs(A[l*n+k]) == 0)
      return false;

    if (k != l) {
      for (j = k; j < n; j++) {
        z = A[k*n+j];
        A[k*n+j] = A[l*n+j];
        A[l*n+j] = z;
      }

      for (j = 0; j < n; j++) {
        z = B[k*n+j];
        B[k*n+j] = B[l*n+j];
        B[l*n+j] = z;
      }
    }

    for (i = k + 1; i < n; i++) {
      s = A[i*n+k] / A[k*n+k];

      for (j = k + 1; j < n; j++)
        A[i*n+j] -= s * A[k*n+j];

      for (j = 0; j < n; j++)
        B[i*n+j] -= s * B[k*n+j];
    }
  }

  if (abs(A[(n-1)*n+(n-1)]) == 0)
    return false;

  for (col = 0; col < n; col++) {
    X[(n - 1)*n+col] = B[(n - 1)*n+col] / A[(n - 1)*n+(n - 1)];

    for (i = n - 2; i >= 0; i--) {
      s = 0;
      for (j = i + 1; j < n; j++)
        s += A[i*n+j] * X[j*n+col];

      X[i*n+col] = (B[i*n+col] - s) / A[i*n+i];
    }
  }


  for(int i = 0; i < n*n; i++) matX[i] = X[i];

  delete [] A;
  delete [] X;
  delete [] B;
}


int prob_selection(int n, double *probArr, double prob){

  double probSum = 0;
  for (int i = 0; i < n; i++)
    probSum += probArr[i];
  if (prob >= probSum)
    return -1;

  if (prob == 0.0)
    return 0;

  double csum = 0;
  int index = 0;
  while (csum < prob) {
    if (index >= n)
      csum = 1.0;
    else
      csum += probArr[index];
    index++;
  }
  index--;

  while (index < n && probArr[index] == 0.0)
    index++;

  if (index == n)
    return -1;

  return index;
}

double exp_dist_cdf_inv(double lambda, double arg){
  return log(1-arg) / (-lambda);
}

double erlang_rnd(int r, double lambda, Random *rnd){
  double sum = 0;
  for(int i = 0; i < r; i++)
    sum += exp_dist_cdf_inv(lambda, rnd->next());

  return sum;
}

//
// Structure
//
Structure::Structure(int bc, int *ri){
  set(bc, ri);
}

void Structure::set(int bc, int *ri){
  mBc = bc;
  mRi = new int[bc];
  for(int i = 0; i < bc; i++) mRi[i] = ri[i];
}

int Structure::getBc(){
  return mBc;
}

int* Structure::getRi(){
  return mRi;
}

double* obtainInitialProbs(){

}

void Structure::print_out(){
  printf("structure : ");
  for(int i = 0; i < mBc-1; i++)
    printf("%d, ", mRi[i]);
  printf("%d\n", mRi[mBc-1]);
}

//
// ER-CHMM structure generation
//
// n - number of states
//
void first_structure(int n, int &bc, int *ri){
  bc = 1;
  ri[0] = n;
}

void next_structure(int n, int &bc, int *ri){
  if(last_structure(n, bc, ri)){
    first_structure(n, bc, ri);
    return;
  }

  if(bc == 1){
    bc = 2;
    ri[0] = n-1;
    ri[1] = 1;
    return;
  }

  for(int i = 0; i < bc-1; i++){
    if(ri[i] - ri[i+1] >= 2){
      ri[i]--;
      ri[i+1]++;
      return;
    }
  }

  bc++;
  ri[0] = n-(bc-1);
  for(int i = 1; i < bc; i++)
    ri[i] = 1;
}

bool last_structure(int n, int &bc, int *ri){
  return bc == n;
}

vector<Structure*>* generate_all_structures(int n){

  vector<Structure*>* allStructures = new vector<Structure*>();

  int *ri = new int[n];
  int bc;
  first_structure(n, bc, ri);
  allStructures->push_back(new Structure(bc, ri));

  do {
    next_structure(n, bc, ri);
    allStructures->push_back(new Structure(bc, ri));

  }while(last_structure(n, bc, ri) == false);



  return allStructures;

}

//
// Map
//
void Map::set(ErChmm *erChmm){
  int bc = erChmm->getBc();
  int *ri = erChmm->getRi();
  double *lambda = erChmm->getLambda();
  double *P = erChmm->getP();




  int size = ri[0];
  int *s = new int[bc];
  s[0] = 0;
  for(int i = 1; i < bc; i++){
    s[i] = s[i-1] + ri[i-1];
    size += ri[i];
  }

  double *D0 = new double[size*size];
  double *D1 = new double[size*size];

  for(int i = 0; i < size*size; i++){
    D0[i] = 0;
    D1[i] = 0;
  }
  int idx;
  for(int i = 0; i < bc; i++){
    for(int j = 0; j < ri[i]-1; j++){
      idx = s[i]+j;
      D0[idx*size + idx] = -lambda[i];
      D0[idx*size + idx+1] = lambda[i];
    }
    idx = s[i] + ri[i] - 1;
    D0[idx*size + idx] = -lambda[i];
  }

  for(int i = 0; i < bc; i++)
    for(int j = 0; j < bc; j++)
      D1[(s[i]+ri[i]-1)*size+s[j]] = P[i*bc+j]*lambda[i];

  mSize = size;
  mD0 = D0;
  mD1 = D1;
}

int Map::getSize(){
  return mSize;
}

double* Map::getD0(){
  return mD0;
}

double* Map::getD1(){
  return mD1;
}

double* Map::obtainAlpha(){
  double *mat = new double[mSize*mSize];
  double *mat2 = new double[mSize*mSize];

  double *inv = new double[mSize*mSize];
  for(int i = 0; i < mSize*mSize; i++)
    mat[i] = -mD0[i];

  mat_inv(mSize, mat, inv);

  mul_mat_mat(mSize, inv, mD1, mat2);

  transpose(mSize, mat2);

  for(int i = 0; i < mSize; i++)
    mat2[i*mSize+i] -= 1.0;


  for(int i = 0; i < mSize; i++)
    mat2[0*mSize+i] = 1;

  double *b = new double[mSize];
  b[0] = 1.;
  for(int i = 1; i < mSize; i++) b[i] = 0.;

  double *alpha = new double[mSize];

  sole_gauss(mSize, mat2, alpha, b);

  delete [] b;
  delete [] mat;
  delete [] mat2;
  delete [] inv;

  return alpha;
}

double Map::obtainMean(){
  double *mat = new double[mSize*mSize];
  double *inv = new double[mSize*mSize];

  for(int i = 0; i < mSize*mSize; i++)
    mat[i] = -mD0[i];

  mat_inv(mSize, mat, inv);

  double *alpha = obtainAlpha();
  double *vec = new double[mSize];

  mul_vec_mat(mSize, alpha, inv, vec);

  double mean = vec_sum_up(mSize, vec);

  delete [] mat;
  delete [] inv;
  delete [] alpha;
  delete [] vec;

  return mean;
}

void Map::print_out(){
  printf("Map, size=%d\n", mSize);
  printf("D0 : \n");
  mat_print_out(mSize, mD0);
  printf("D1 : \n");
  mat_print_out(mSize, mD1);
  printf("\n");
}

//
// ErChmm
//
ErChmm::ErChmm(){

}

void ErChmm::set(int bc, int *ri, double *lambda, double *P){
  mBc = bc;
  mRi = new int[bc];
  mLambda = new double[bc];
  mP = new double[bc*bc];

  for(int i = 0; i < bc; i++){
    mRi[i] = ri[i];
    mLambda[i] = lambda[i];
    for(int j = 0; j < bc; j++)
      mP[i*bc+j] = P[i*bc+j];
  }
}

int ErChmm::getBc(){
  return mBc;
}

int* ErChmm::getRi(){
  return mRi;
}

double* ErChmm::getLambda(){
  return mLambda;
}

double* ErChmm::getP(){
  return mP;
}

double* ErChmm::obtainAlpha(){
  int bc = mBc;
  double *A = new double[bc*bc];
  double *b = new double[bc];
  double *x = new double[bc];

  for(int i = 0; i < bc*bc; i++)
    A[i] = mP[i];

  transpose(bc, A);

  for(int i = 0; i < bc; i++)
    A[i*bc+i] -=1.0;

  for(int i = 0; i < bc; i++)
    A[0*bc+i] = 1;

  for(int i = 0; i < bc; i++)
    b[i] = 0;
  b[0] = 1;

  sole_gauss(bc, A, x, b);

  delete [] A;
  delete [] b;

  return x;
}

void ErChmm::print_out(){
  printf("# number of branches\n");
  printf("%d\n", mBc);

  printf("# structure\n");
  for(int i = 0; i < mBc-1; i++)
    printf("%d, ", mRi[i]);
  printf("%d\n", mRi[mBc-1]);

  printf("# lambda \n");
  for(int i = 0; i < mBc; i++)
    printf("%f\n", mLambda[i]);

  printf("# P \n");
  for(int i = 0; i < mBc; i++){
    for(int j = 0; j < mBc-1; j++)
      printf("%f, ", mP[i*mBc+j]);
    printf("%f\n", mP[i*mBc+mBc-1]);
  }

}

void ErChmm::write_to_file(const char *fileName){
  FILE *file = fopen(fileName, "w");

  fprintf(file, "# number of branches\n");
  fprintf(file, "%d\n", mBc);

  fprintf(file, "# structure\n");
  for(int i = 0; i < mBc-1; i++)
    fprintf(file,"%d, ", mRi[i]);
  fprintf(file,"%d\n", mRi[mBc-1]);

  fprintf(file, "# lambda \n");
  for(int i = 0; i < mBc; i++)
    fprintf(file, "%f\n", mLambda[i]);

  fprintf(file, "# P \n");
  for(int i = 0; i < mBc; i++){
    for(int j = 0; j < mBc-1; j++)
      fprintf(file, "%f, ", mP[i*mBc+j]);
    fprintf(file, "%f\n", mP[i*mBc+mBc-1]);

  }

  fclose(file);
}

void ErChmm::read_from_file(const char *fileName){
  FILE *file = fopen(fileName, "r");

  char *line = NULL;
  char *token = NULL;
  size_t len = 0;
  ssize_t read;

  read = getline(&line, &len, file); // # number of branches
  read = getline(&line, &len, file);
  token = strtok(line, " \n");
  int bc = strtol(token, NULL, 10);
  int *ri = new int[bc];
  double *lambda = new double[bc];
  double *P = new double[bc*bc];

  read = getline(&line, &len, file); // # structure
  read = getline(&line, &len, file);
  for(int i = 0; i < bc; i++){
    token = strtok(line, ", \n");
    ri[i] = strtol(token, NULL, 10);
    line = NULL;
  }


  read = getline(&line, &len, file); // # lambda
  for(int i = 0; i < bc; i++){
    read = getline(&line, &len, file);
    token = strtok(line, ", \n");
    lambda[i] = strtod(token, NULL);
  }

  read = getline(&line, &len, file); // # P
  for(int i = 0; i < bc; i++){
    read = getline(&line, &len, file);
    for(int j = 0; j < bc; j++){
      token = strtok(line, ", \n");
      P[i*bc+j] = strtod(token, NULL);
      line = NULL;
    }
  }

  fclose(file);

  set(bc, ri, lambda, P);

  delete [] ri;
  delete [] lambda;
  delete [] P;
}

//
// Random
//
Random::Random(){
  mGen = new std::mt19937();
  mDist = new std::uniform_real_distribution<double>(0., 1.);
}

double Random::next(){
  return (*mDist)(*mGen);
}

//
// Interarrivals
//
void Interarrivals::generate(ErChmm *erChmm, int count){
  int bc = erChmm->getBc();
  int *ri = erChmm->getRi();
  double *lambda = erChmm->getLambda();
  double *P = erChmm->getP();

  double *alpha = erChmm->obtainAlpha();
  Random *rnd = new Random();

  int cb; // current branch
  cb = prob_selection(bc, alpha, rnd->next());

  float *arr = new float[count];

  double sum = 0;
  for(int k = 0; k < count; k++){
    float interarrival = (float)erlang_rnd(ri[cb], lambda[cb], rnd);
    arr[k] = interarrival;
    sum += (double) interarrival;

    cb = prob_selection(bc, (double*)&P[cb*bc], rnd->next());
  }

  mCount = count;
  mMean = sum / (double)count;
  mArr = arr;

  delete rnd;
  delete [] alpha;
}

int Interarrivals::getCount(){
  return mCount;
}

double Interarrivals::getMean(){
  return mMean;
}

float* Interarrivals::getArr(){
  return mArr;
}

void Interarrivals::print_out(){
  printf("Interarrivals: count=%d, mean=%f\n", mCount, mMean);
  for(int i = 0; i < mCount; i++)
    printf("  [%d] : %f\n", i, mArr[i]);
}

void Interarrivals::write_to_file(const char *fileName){

}

void Interarrivals::read_from_file(const char *fileName){

}

int main(int argc, char *argv[]){

  printf("Parallel algorithms for fitting Markov Arrial process fitting, 2017-2018\n");
  printf("\n");

  run_all_tests();


  // int bc = 2;
  // int *ri = new int[bc];
  // double *lambda = new double[bc];
  // double *P = new double[bc*bc];
  // ri[0] = 1;
  // ri[1] = 2;
  //
  // lambda[0] = 1.5;
  // lambda[1] = 2.5;
  //
  // P[0*bc+0] = 0.2;
  // P[0*bc+1] = 0.8;
  // P[1*bc+0] = 0.7;
  // P[1*bc+1] = 0.3;
  //
  // ErChmm *erChmm = new ErChmm();
  // erChmm->set(bc, ri, lambda, P);
  //
  // erChmm->print_out();
  // erChmm->write_to_file("params.erchmm");
  //
  // ErChmm *erChmm2 = new ErChmm();
  // erChmm2->read_from_file("params.erchmm");
  // erChmm2->print_out();


  return 0;

}
