#include "prog.h"
#include <stdio.h>


void int_arr_to_str(char *str, int n, int *arr){
  char tmp[32] = "";
  strcpy(str, "");
  for(int i = 0; i < n-1; i++){
    sprintf(tmp, "%d, ", arr[i]);
    strcat(str, tmp);
  }
  sprintf(tmp, "%d", arr[n-1]);
  strcat(str, tmp);
}

void int_arr_to_str_compact(char *str, int n, int *arr){
  char tmp[32] = "";
  strcpy(str, "");
  for(int i = 0; i < n; i++){
    sprintf(tmp, "%d", arr[i]);
    strcat(str, tmp);
    if(i < n-1)
      strcat(str, "-");
  }
}

void impl_str(char *str, int impl){
  switch(impl){
    case SERIAL:
      strcpy(str, "ser");
      break;
    case P_1:
      strcpy(str, "p1");
      break;
    case P_2:
      strcpy(str, "p2");
      break;
    case P_2_D:
      strcpy(str, "p2d");
      break;
    case P_3:
      strcpy(str, "p3");
      break;
    case P_3_D:
      strcpy(str, "p3d");
      break;
  }
}

int impl_id(char *str){
  if(strcmp(strtrm(str), "ser") == 0) return SERIAL;
  if(strcmp(strtrm(str), "p1") == 0) return P_1;
  if(strcmp(strtrm(str), "p2") == 0) return P_2;
  if(strcmp(strtrm(str), "p2d") == 0) return P_2_D;
  if(strcmp(strtrm(str), "p3") == 0) return P_3;
  if(strcmp(strtrm(str), "p3d") == 0) return P_3_D;


}


char *strtrm(char *str){
    size_t len = 0;
    char *frontp = str;
    char *endp = NULL;

    if( str == NULL ) { return NULL; }
    if( str[0] == '\0' ) { return str; }

    len = strlen(str);
    endp = str + len;

    /* Move the front and back pointers to address the first non-whitespace
     * characters from each end.
     */
    while( isspace((unsigned char) *frontp) ) { ++frontp; }
    if( endp != frontp )
    {
        while( isspace((unsigned char) *(--endp)) && endp != frontp ) {}
    }

    if( str + len - 1 != endp )
            *(endp + 1) = '\0';
    else if( frontp != str &&  endp == frontp )
            *str = '\0';

    /* Shift the string so that it starts at str so that if it's dynamically
     * allocated, we can still free it on the returned pointer.  Note the reuse
     * of endp to mean the front of the string buffer now.
     */
    endp = str;
    if( frontp != str )
    {
            while( *frontp ) { *endp++ = *frontp++; }
            *endp = '\0';
    }


    return str;
}

void read_line(char *str){
  int idx = 0;
  char c = getchar();
  while(c != '\n'){
    str[idx++] = c;
    c = getchar();
  }
  str[idx] = '\0';
}

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

void Structure::str(char *str){
  int_arr_to_str(str, mBc, mRi);
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


void Map::write_to_file(const char *fileName){
  FILE *file = fopen(fileName, "w");

  fprintf(file, "# number of states\n");
  fprintf(file, "%d\n", mSize);

  fprintf(file, "# matrix D0 \n");
  for(int i = 0; i < mSize; i++){
    for(int j = 0; j < mSize-1; j++)
      fprintf(file, "%.16f, ", mD0[i*mSize+j]);
    fprintf(file, "%.16f\n", mD0[i*mSize+mSize-1]);
  }

  fprintf(file, "# matrix D1 \n");
  for(int i = 0; i < mSize; i++){
    for(int j = 0; j < mSize-1; j++)
      fprintf(file, "%.16f, ", mD1[i*mSize+j]);
    fprintf(file, "%.16f\n", mD1[i*mSize+mSize-1]);
  }

  fclose(file);
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

void ErChmm::set(Structure *st, double mean, Random *rnd){
  int bc = st->getBc();
  int *ri = st->getRi();

  // generating initial lambda sum_mat_value
  double *lambda = new double[bc];
  for(int i = 0; i < bc; i++)
    lambda[i] = 1.;//rnd->next();

  // generating switching probabilities
  double *P = new double[bc*bc];
  for(int i = 0; i < bc; i++){
    double sum = 0;
    for(int j = 0; j < bc; j++){
      double p = rnd->next();
      P[i*bc+j] = p;
      sum += p;
    }
    for(int j = 0; j < bc; j++)
      P[i*bc+j] /= sum;
  }

  // computing stationary probabilities
  double *alpha = calculateStationaryProb(bc, P);
  double erChmmMean = calculateMean(bc, ri, alpha, lambda);


  // rescaling lambda
  double factor = erChmmMean / mean;
  for(int i = 0; i < bc; i++)
    lambda[i] *= factor;

  mBc = bc;
  mLambda = lambda;
  mRi = ri;
  mP = P;

}

void ErChmm::set(int bc, int *ri, float *lambdaArr, float *PArr){
  mBc = bc;
  mRi = new int[bc];
  mLambda = new double[bc];
  mP = new double[bc*bc];

  for(int i = 0; i < bc; i++){
    mRi[i] = ri[i];
    mLambda[i] = (double)lambdaArr[i];
    for(int j = 0; j < bc; j++)
      mP[i*bc+j] = (double)PArr[i*bc+j];
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

double ErChmm::obtainMean(){
  double *alpha = obtainAlpha();
  double mean = calculateMean(mBc, mRi, alpha, mLambda);
  delete [] alpha;
  return mean;
}

double* ErChmm::obtainAlpha(){
  return calculateStationaryProb(mBc, mP);
}

double ErChmm::obtainLogLikelihood(Interarrivals *interarrivals){
  int bc = mBc;
  int *ri = mRi;
  double *lambda = mLambda;
  double *alpha = obtainAlpha();
  double *P = mP;
  int T = interarrivals->getCount();
  float *timeArr = interarrivals->getArr();

  double llhval =  llh(bc, ri, lambda, P, alpha, T, timeArr);

  delete [] alpha;

  return llhval;
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
    fprintf(file, "%.16f\n", mLambda[i]);

  fprintf(file, "# P \n");
  for(int i = 0; i < mBc; i++){
    for(int j = 0; j < mBc-1; j++)
      fprintf(file, "%.16f, ", mP[i*mBc+j]);
    fprintf(file, "%.16f\n", mP[i*mBc+mBc-1]);

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

double ErChmm::calculateMean(int bc, int *ri, double *alpha, double *lambda){
  double mean = 0;
  for(int i = 0; i < bc; i++)
    mean += (alpha[i]*ri[i]) / lambda[i];
  return mean;
}

double* ErChmm::calculateStationaryProb(int bc, double *P){

  double *A = new double[bc*bc];
  double *b = new double[bc];
  double *x = new double[bc];

  for(int i = 0; i < bc*bc; i++)
    A[i] = P[i];

  transpose(bc, A);

  for(int i = 0; i < bc; i++)
    A[i*bc+i] -= 1.0;

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

  for(int k = 0; k < count; k++){
    float interarrival = (float)erlang_rnd(ri[cb], lambda[cb], rnd);
    arr[k] = interarrival;

    cb = prob_selection(bc, (double*)&P[cb*bc], rnd->next());
  }

  mCount = count;
  mMean = computeMean(count, arr);
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
  FILE *file = fopen(fileName, "wb");

  fwrite(&mCount, sizeof(int), 1, file);
  fwrite(mArr, sizeof(float), mCount, file);

  fclose(file);
}

void Interarrivals::read_from_file(const char *fileName){
  FILE *file = fopen(fileName, "rb");

  if(file == 0){
    printf("*** Interarrival::read_from_file('%s') : file == 0\n", fileName);
    return;
  }

  size_t r;

  int count;
  double mean;
  r = fread(&count, sizeof(int), 1, file);

  float *arr = new float[count];
  r = fread(arr, sizeof(float), count, file);

  fclose(file);

  mCount = count;
  mMean = computeMean(count, arr);
  mArr = arr;


}

double Interarrivals::computeMean(int count, float *arr){
  double sum = 0.0;
  for(int i = 0; i < count; i++)
    sum += arr[i];

    return sum / (double)count;
}
//
// FittingOutput
//
void FittingOutput::setTag(char *tag){
  strcpy(mTag, tag);
}

void FittingOutput::setInitialErChmm(ErChmm *erChmm){
  mInitialErChmm = erChmm;
}

void FittingOutput::setResultErChmm(ErChmm *erChmm){
  mResultErChmm = erChmm;
}

void FittingOutput::setImpl(int impl){
  mImpl = impl;
}

void FittingOutput::setLlh(double llh){
  mLlh = llh;
}

void FittingOutput::setImplLlh(double implLlh){
  mImplLlh = implLlh;
}

void FittingOutput::setMem(double  cpuMem, double gpuMem){
  mCpuMem = cpuMem;
  mGpuMem = gpuMem;
  mTotalMem = cpuMem + gpuMem;
}

void FittingOutput::setRuntime(double runtime){
  mRuntime = runtime;
}

void FittingOutput::setL(int L){
  mL = L;
}

void FittingOutput::setIterCount(int iterCount){
  mIterCount = iterCount;
}

char* FittingOutput::tag(){
  return mTag;
}

ErChmm* FittingOutput::initialErChmm(){
  return mInitialErChmm;
}

ErChmm* FittingOutput::resultErChmm(){
  return mResultErChmm;
}

int FittingOutput::getBc(){
  return mInitialErChmm->getBc();
}

int FittingOutput::getL(){
  return mL;
}

int FittingOutput::getImpl(){
  return mImpl;
}

double FittingOutput::getRuntime(){
  return mRuntime;
}

double FittingOutput::getCpuMem(){
  return mCpuMem;
}

double FittingOutput::getGpuMem(){
  return mGpuMem;
}

double FittingOutput::getLlh(){
  return mLlh;
}

void FittingOutput::append_to_file(const char *filePath){
  FILE *file = fopen(filePath, "a");

  char stStr[64]=""; int_arr_to_str(stStr, mInitialErChmm->getBc(), mInitialErChmm->getRi());
  char implStr[6] =""; impl_str(implStr, mImpl);

  fprintf(file, "%s; %d; %d; %s; %s; %e; %e; %.3f; %.3f; %.3f; %.3f\n",
    mTag,
    mL,
    mInitialErChmm->getBc(),
    implStr,
    stStr,
    mLlh,
    mResultErChmm->obtainMean(),
    mRuntime,
    mCpuMem,
    mGpuMem,
    mTotalMem
  );

  fclose(file);
}

void FittingOutput::append_to_file_bin(const char *filePath){
  FILE *file = fopen(filePath, "ab");

  int bc;
  int *ri;
  double *lambda;
  double  *P;

  bc = mInitialErChmm->getBc();
  ri = mInitialErChmm->getRi();
  lambda = mInitialErChmm->getLambda();
  P = mInitialErChmm->getP();

  fwrite(&bc, sizeof(int), 1, file);
  fwrite(ri, sizeof(int), bc, file);
  fwrite(lambda, sizeof(double), bc, file);
  fwrite(P, sizeof(double), bc*bc, file);

  //
  bc = mResultErChmm->getBc();
  ri = mResultErChmm->getRi();
  lambda = mResultErChmm->getLambda();
  P = mResultErChmm->getP();

  fwrite(&bc, sizeof(int), 1, file);
  fwrite(ri, sizeof(int), bc, file);
  fwrite(lambda, sizeof(double), bc, file);
  fwrite(P, sizeof(double), bc*bc, file);

  //
  fwrite(&mImpl, sizeof(int), 1, file);
  fwrite(&mLlh, sizeof(double), 1, file);
  fwrite(&mImplLlh, sizeof(double), 1, file);
  fwrite(&mRuntime, sizeof(double), 1, file);
  fwrite(&mCpuMem, sizeof(double), 1, file);
  fwrite(&mGpuMem, sizeof(double), 1, file);
  fwrite(&mL, sizeof(int), 1, file);
  fwrite(&mIterCount, sizeof(int), 1, file);


  fclose(file);
}

void FittingOutput::read_from_file_bin(FILE * file){

  size_t r;

  int bc;
  int *ri;
  double *lambda;
  double  *P;



  r = fread(&bc, sizeof(int), 1, file);

  ri = new int[bc];
  lambda = new double[bc];
  P = new double[bc*bc];
  r = fread(ri, sizeof(int), bc, file);
  r = fread(lambda, sizeof(double), bc, file);
  r = fread(P, sizeof(double), bc*bc, file);


  mInitialErChmm = new ErChmm();
  mInitialErChmm->set(bc, ri, lambda, P);


  //
  r = fread(&bc, sizeof(int), 1, file);
  r = fread(ri, sizeof(int), bc, file);
  r = fread(lambda, sizeof(double), bc, file);
  r = fread(P, sizeof(double), bc*bc, file);


  mResultErChmm = new ErChmm();
  mResultErChmm->set(bc, ri, lambda, P);


  //
  r = fread(&mImpl, sizeof(int), 1, file);
  r = fread(&mLlh, sizeof(double), 1, file);
  r = fread(&mImplLlh, sizeof(double), 1, file);
  r = fread(&mRuntime, sizeof(double), 1, file);
  r = fread(&mCpuMem, sizeof(double), 1, file);
  r = fread(&mGpuMem, sizeof(double), 1, file);
  r = fread(&mL, sizeof(int), 1, file);
  r = fread(&mIterCount, sizeof(int), 1, file);


  build_tag(mTag, mImpl, bc, ri, mL);
  mTotalMem = mCpuMem + mGpuMem;


}

//tag; L; R; impl; struct; llh; mean; runtime; cpu mem; gpu mem; total
void build_tag(char *str, int impl, int bc, int *ri, int L){

  char stStr[32]="";
  char implStr[6]="";
  int_arr_to_str_compact(stStr, bc, ri);
  impl_str(implStr, impl);

  sprintf(str, "L%d-R%d-%s-%s",
    L,
    bc,
    implStr,
    stStr
  );
}

//
// Research
//
int caseIdx(int L, int bc, int impl){
  int Lidx = 0;
  if(L == 1920) Lidx = 1;
  if(L == 1920*2) Lidx = 2;
  if(L == 1920*4) Lidx = 3;

  int bcidx = bc - 2;

  int idx = Lidx*NUM_STATES*NUM_IMPL + bcidx*NUM_IMPL + impl;
  return idx;
}


void Research::run(){

  // double startTime = clock();
  // double endTime; // = clock();
  // double runtime; // = (double)(endTime-startTime)/CLOCKS_PER_SEC;


  int dir_err = 0;

  char str[256]="";
  printf("\nResearch started. \n\n");

  int step = -1;

  char metaFilePath[64] = "";
  sprintf(metaFilePath, ".%c%s%c%s", kPathSeparator, mFolder, kPathSeparator, mMetaFileName);
  char interarrivalsPath[64] = "";
  sprintf(interarrivalsPath, ".%c%s%c%s", kPathSeparator, mFolder, kPathSeparator, mInterarrivalFile);
  char genErChmmPath[64] = "";
  sprintf(genErChmmPath, ".%c%s%c%s", kPathSeparator, mFolder, kPathSeparator, mGenErChmmFile);
  char infoPath[64] = "";
  sprintf(infoPath, ".%c%s%c%s", kPathSeparator, mFolder, kPathSeparator, mInfoFile);
  char infoBinPath[64] = "";
  sprintf(infoBinPath, ".%c%s%c%s", kPathSeparator, mFolder, kPathSeparator, mInfoFileBin);

  char initialErChmmFolderPath[128] = "";
  char resultErChmmFolderPath[128] = "";
  sprintf(initialErChmmFolderPath, ".%c%s%c%s", kPathSeparator, mFolder, kPathSeparator, mInitialErChmmFolder);
  sprintf(resultErChmmFolderPath, ".%c%s%c%s", kPathSeparator, mFolder, kPathSeparator, mResultErChmmFolder);


  char summaryPath[64] = "";
  sprintf(summaryPath, ".%c%s%c%s", kPathSeparator, mFolder, kPathSeparator, mSummaryFile);

  char llhPath[64] = "";
  sprintf(llhPath, ".%c%s%c%s", kPathSeparator, mFolder, kPathSeparator, mLlhFile);


  // checking for meta file
  printf("Checking for meta file '%s' ...", metaFilePath);

  FILE *metaFile = fopen(metaFilePath, "rb");
  if(metaFile == NULL){
    printf("... not found. Initializing a new research.\n");

    sprintf(str, "rm -rf %s", mFolder);
    dir_err = system(str);

    printf("Creating meta file '%s' ...", metaFilePath);

    sprintf(str, "mkdir %s", mFolder);
    dir_err = system(str);

    sprintf(str, "cd %s  && mkdir %s", mFolder, mInitialErChmmFolder);
    dir_err = system(str);

    sprintf(str, "cd %s  && mkdir %s", mFolder, mResultErChmmFolder);
    dir_err = system(str);


    metaFile = fopen(metaFilePath, "wb");
    step = -1;
    fwrite(&step, sizeof(int), 1, metaFile);
    fclose(metaFile);
    printf("... done.\n");

  }else{
    printf("... found. Resuming research.\n");
  }

  step = read_meta_file();

  if(step == -1) {

    printf("\n( step : -1 )\n\n");

    ErChmm *genErChmm = erChmmForGeneratingInterarrivals();



    printf("Writing '%s' ...", genErChmmPath);
    genErChmm->write_to_file(genErChmmPath);
    printf("... done.\n");

    printf("Generating %d interarrivals ...", mInterarrivalCount);
    Interarrivals* interarrivals = new Interarrivals();
    interarrivals->generate(genErChmm, mInterarrivalCount);
    printf("... done.\n");



    printf("Writing interarrivals to '%s' ...", interarrivalsPath);
    interarrivals->write_to_file(interarrivalsPath);
    printf("... done.\n");

    write_meta_file(0);

  }



  printf("Reading ER-CHMM from '%s' ...", genErChmmPath);
  ErChmm *genErChmm = new ErChmm();
  genErChmm->read_from_file(genErChmmPath);
  printf("... done.\n");

  printf("Reading interarrivals from '%s' ...", interarrivalsPath);
  Interarrivals *interarrivals = new Interarrivals();
  interarrivals->read_from_file(interarrivalsPath);
  printf("... done.\n");


  step = read_meta_file();


  if(step == 0) {

    printf("\n( step : 0 )\n\n");

    printf("Initializing '%s' ...", infoPath);
    FILE *infoFile = fopen(infoPath, "w");
    fprintf(infoFile, "tag; L; R; impl; struct; llh; mean; runtime; cpu mem; gpu mem; total mem\n");

    char structStr[128] = "";
    int_arr_to_str(structStr, genErChmm->getBc(), genErChmm->getRi());


    fprintf(infoFile, "gen-erchmm; - ; %d; - ; %s; %e; %e; - ; - ; - ; -\n",
      genErChmm->getBc(),
      structStr,
      genErChmm->obtainLogLikelihood(interarrivals),
      genErChmm->obtainMean());
    fprintf(infoFile, "sample; - ; - ; - ; - ; - ; %e; - ; - ; - ; - \n", interarrivals->getMean() );

    fclose(infoFile);
    printf("... done.\n");
  }

  int LCount = 3;
  int *LArr = new int[LCount];
  LArr[0] = 1920;
  LArr[1] = 1920*2;
  LArr[2] = 1920*4;

  int curr_step = 1;

  printf("Generating ER-CHMM structures ...");
  vector<Structure*> *allSt =  generate_all_structures(10);

  // removing the first structure, with R = 1
  allSt->erase(allSt->begin());
  printf("... done.\n");

  int stCount = allSt->size();
  int fittingCount = stCount*(LCount*5+1);
  int  analysis_step_at = fittingCount + 1;

  char path[128]="";
  for(int sti = 0; sti < stCount; sti++){
    Structure *st = (*allSt)[sti];

    for(int l = 0; l < LCount; l++){
      int L = LArr[l];

      for(int impl = 0; impl <= 5; impl++){

        if(l != 0 && impl == 0 ) continue;
        if(step >= curr_step){
          curr_step++;
          continue;
        }

        printf("\n( step : %d / %d )\n\n", curr_step, fittingCount);

        ErChmm *initialErChmm = NULL;
        FittingOutput *fo = runFitting(impl, L, initialErChmm, st, interarrivals);

        printf("Adding fitting results to '%s' ...", infoPath);
        fo->append_to_file(infoPath);
        printf("... done.\n");

        printf("Adding fitting results to '%s' ...", infoBinPath);
        fo->append_to_file_bin(infoBinPath);
        printf("... done.\n");


        sprintf(path, "%s%c%s.txt", initialErChmmFolderPath, kPathSeparator, fo->tag());
        printf("Writing initial ER-CHMM to '%s' ...", path);
        fo->initialErChmm()->write_to_file(path);
        printf("... done.\n");

        sprintf(path, "%s%c%s.txt", resultErChmmFolderPath, kPathSeparator, fo->tag());
        printf("Writing result ER-CHMM to '%s' ...", path);
        fo->resultErChmm()->write_to_file(path);
        printf("... done.\n");

        write_meta_file(curr_step);
        curr_step++;

      }
    }

  }

  step = read_meta_file();

  if(step == fittingCount){

    printf("\n( step : %d )\n\n", step);

    printf("Reading fitting results from '%s' ...", infoBinPath);

    FILE *file = fopen(infoBinPath, "rb");
    vector<FittingOutput*> *fos = new vector<FittingOutput*>();
    for(int i = 0; i < fittingCount; i++){
      FittingOutput *fo = new FittingOutput();
      fo->read_from_file_bin(file);
      fos->push_back(fo);
      //printf("tag: %s, runtime = %f\n", fo->tag(), fo->getRuntime());
    }
    fclose(file);
    printf("... done.\n");

    printf("Analysing runtimes and memory usage ...");
    int caseCount = (LCount+1)*NUM_STATES*NUM_IMPL;
    int *stCountArr = new int[caseCount];
    double *rtArr = new double[caseCount];

    // actually, memory usage for R,L is the same
    double *cpuMemArr = new double[caseCount];
    double *gpuMemArr = new double[caseCount];

    for(int i = 0; i < caseCount; i++){
      stCountArr[i] = 0;
      rtArr[i] = 0;
      cpuMemArr[i] = 0;
      gpuMemArr[i] = 0;
    }

    for(int i = 0; i < fittingCount; i++){
      FittingOutput *fo = (*fos)[i];

      int bc = fo->getBc();
      int L = fo->getL();
      int impl = fo->getImpl();
      double runtime = fo->getRuntime();
      double cpuMem = fo->getCpuMem();
      double gpuMem = fo->getGpuMem();

      int idx = caseIdx(L, bc, impl);
      stCountArr[idx]++;
      rtArr[idx] += runtime;
      cpuMemArr[idx] += cpuMem;
      gpuMemArr[idx] += gpuMem;
    }

    int L = 1;
    for(int bc = 2; bc <= NUM_STATES; bc++){
      //for(int impl = 0; impl < NUM_IMPL; impl++){
      int impl = 0; // SERIAL
        int idx = caseIdx(L, bc, impl);
        rtArr[idx] /= (double)stCountArr[idx];
        cpuMemArr[idx] /= (double)stCountArr[idx];
        gpuMemArr[idx] /= (double)stCountArr[idx];
      //}
    }

    L = 1920;
    for(int bc = 2; bc <= NUM_STATES; bc++){
      for(int impl = 1; impl < NUM_IMPL; impl++){
        int idx = caseIdx(L, bc, impl);
        rtArr[idx] /= (double)stCountArr[idx];
        cpuMemArr[idx] /= (double)stCountArr[idx];
        gpuMemArr[idx] /= (double)stCountArr[idx];
      }
    }

    L = 1920*2;
    for(int bc = 2; bc <= NUM_STATES; bc++){
      for(int impl = 1; impl < NUM_IMPL; impl++){
        int idx = caseIdx(L, bc, impl);
        rtArr[idx] /= (double)stCountArr[idx];
        cpuMemArr[idx] /= (double)stCountArr[idx];
        gpuMemArr[idx] /= (double)stCountArr[idx];
      }
    }

    L = 1920*4;
    for(int bc = 2; bc <= NUM_STATES; bc++){
      for(int impl = 1; impl < NUM_IMPL; impl++){
        int idx = caseIdx(L, bc, impl);
        rtArr[idx] /= (double)stCountArr[idx];
        cpuMemArr[idx] /= (double)stCountArr[idx];
        gpuMemArr[idx] /= (double)stCountArr[idx];
      }
    }


    printf("... done.\n");

    printf("Writing runtimes and memory usage to '%s' ...", summaryPath);
    file = fopen(summaryPath, "w");
    for(int l = -1; l < LCount; l++){
      int L = 1;
      if(l >= 0) L = LArr[l];
      char implStr[6]="";
      for(int impl = 0; impl < NUM_IMPL; impl++){
        if(L == 1 && impl != 0) continue;
        if(L != 1 && impl == 0) continue;
        impl_str(implStr, impl);

        fprintf(file, "L = %d, %s\n", L, implStr);
        fprintf(file, "-----------------------------\n");
        fprintf(file, "R; runtime (seconds); total mem; cpu mem; gpu mem\n");
        for(int bc = 2; bc <= NUM_STATES; bc++){
          int idx = caseIdx(L, bc, impl);
          fprintf(file, "%d; %.3f; %.3f; %.3f; %.3f\n", bc, rtArr[idx], (cpuMemArr[idx]+gpuMemArr[idx]), cpuMemArr[idx], gpuMemArr[idx]);
        }
        fprintf(file, "\n");

      }

    }


    fprintf(file, "\n");

    fclose(file);
    printf("... done.\n");

    printf("Sorting fitting results by log-likelihood ...");
    std::sort(fos->begin(), fos->end(),
    []( FittingOutput* lhs,  FittingOutput* rhs)
    {
        return lhs->getLlh() > rhs->getLlh();
    });
    printf("... done. \n");

    printf("Writing sorted results to '%s' ...", llhPath);
    file = fopen(llhPath, "w");
    fprintf(file, "tag; llh\n");
    for(int i = 0; i < fittingCount; i++){
      FittingOutput *fo = (*fos)[i];
      fprintf(file, "%s; %.16e\n", fo->tag(), fo->getLlh());

    }

    fclose(file);
    printf("... done.\n");
  }




  printf("\nResearch finished. \n\n");

}

ErChmm* Research::erChmmForGeneratingInterarrivals(){
  int bc = 5;
  int *ri = new int[bc];
  ri[0] = 3; ri[1] = 3; ri[2] = 2; ri[3] = 1; ri[4] = 1;

  /* ER-CHMM initial parameters. */

  double *lambdaArr = new double[bc];
  lambdaArr[0] = 1.0f;
  lambdaArr[1] = 3.0f;
  lambdaArr[2] = 2.0f;
  lambdaArr[3] = 1.0f;
  lambdaArr[4] = 3.0f;


  double *pArr = new double[bc*bc];
  pArr[0*bc+0] = 0.1f; pArr[0*bc+1] = 0.2f; pArr[0*bc+2] = 0.1f;  pArr[0*bc+3] = 0.4f;  pArr[0*bc+4] = 0.2f;
  pArr[1*bc+0] = 0.3f; pArr[1*bc+1] = 0.1f; pArr[1*bc+2] = 0.2f;  pArr[1*bc+3] = 0.1f;  pArr[1*bc+4] = 0.3f;
  pArr[2*bc+0] = 0.2f; pArr[2*bc+1] = 0.3f; pArr[2*bc+2] = 0.1f;  pArr[2*bc+3] = 0.1f;  pArr[2*bc+4] = 0.3f;
  pArr[3*bc+0] = 0.4f; pArr[3*bc+1] = 0.1f; pArr[3*bc+2] = 0.2f;  pArr[3*bc+3] = 0.2f;  pArr[3*bc+4] = 0.1f;
  pArr[4*bc+0] = 0.2f; pArr[4*bc+1] = 0.2f; pArr[4*bc+2] = 0.1f;  pArr[4*bc+3] = 0.3f;  pArr[4*bc+4] = 0.2f;

  // for(int i = 0; i < bc; i++){
  //   double sum = 0;
  //   for(int j = 0; j < bc; j++)
  //   sum += pArr[i*bc+j];
  //   printf("sum[%d]=%e\n", i, sum);
  //
  //
  // }

  ErChmm *erChmm = new ErChmm();
  erChmm->set(bc, ri, lambdaArr, pArr);
  return erChmm;

}

int Research::read_meta_file(){

  char metaFilePath[64] = "";
  sprintf(metaFilePath, "%s%c%s", mFolder, kPathSeparator, mMetaFileName);

  printf("Reading meta file '%s' ...", metaFilePath);

  FILE *metaFile = fopen(metaFilePath, "rb");
  int step;
  int r = fread(&step, sizeof(int), 1, metaFile);
  fclose(metaFile);

  printf("... done.\n");

  return step;
}

void Research::write_meta_file(int step){
  char metaFilePath[64] = "";
  sprintf(metaFilePath, "%s%c%s", mFolder, kPathSeparator, mMetaFileName);

  printf("Updating meta file '%s' ...", metaFilePath);

  FILE *metaFile = fopen(metaFilePath, "wb");
  fwrite(&step, sizeof(int), 1, metaFile);

  printf("... done.\n");

  fclose(metaFile);
}

//
// Research2
//
void Research2::run(){
  char str[128]="";
  sprintf(str, "mkdir %s", mResultErChmmFolder);
  int dir_err = system(str);

  ErChmm *initialErChmm = new ErChmm();
  initialErChmm->read_from_file(mInitialErChmm);

  int bc = 5;
  int *ri = new int[bc];
  ri[0] = 3;
  ri[1] = 3;
  ri[2] = 2;
  ri[3] = 1;
  ri[4] = 1;

  Structure *st = new Structure(bc, ri);

  Interarrivals *interarrivals = new Interarrivals();
  interarrivals->read_from_file(mInterarrivalFile);

  FILE *infoFile = fopen(mInfoFile, "w");
  fprintf(infoFile, "tag; L; R; impl; struct; llh; mean; runtime; cpu mem; gpu mem; total mem\n");

  char structStr[128] = "";
  int_arr_to_str(structStr, initialErChmm->getBc(), initialErChmm->getRi());


  fprintf(infoFile, "initial-erchmm; - ; %d; - ; %s; %e; %e; - ; - ; - ; -\n",
    initialErChmm->getBc(),
    structStr,
    initialErChmm->obtainLogLikelihood(interarrivals),
    initialErChmm->obtainMean());
  fprintf(infoFile, "sample; - ; - ; - ; - ; - ; %e; - ; - ; - ; - \n", interarrivals->getMean() );

  fclose(infoFile);

  int LCount = 3;
  int *LArr = new int[LCount];
  LArr[0] = 1920;
  LArr[1] = 1920*2;
  LArr[2] = 1920*4;

  vector<FittingOutput*> *fos = new vector<FittingOutput*>();
  for(int l = 0; l < LCount; l++){
    int L = LArr[l];

    for(int impl = 0; impl <= 5; impl++){

      if(l != 0 && impl == 0 ) continue;

      FittingOutput *fo = runFitting(impl, L, initialErChmm, st, interarrivals);
      fos->push_back(fo);

      fo->append_to_file(mInfoFile);

      char path[128]="";
      sprintf(path, "%s%c%s.txt", mResultErChmmFolder, kPathSeparator, fo->tag());
      fo->resultErChmm()->write_to_file(path);

    }
  }

  std::sort(fos->begin(), fos->end(),
  []( FittingOutput* lhs,  FittingOutput* rhs)
  {
      return lhs->getLlh() > rhs->getLlh();
  });

  FILE *file = fopen(mLlhFile, "w");
  fprintf(file, "tag; llh\n");
  for(int i = 0; i < fos->size(); i++){
    FittingOutput *fo = (*fos)[i];
    fprintf(file, "%s; %.16e\n", fo->tag(), fo->getLlh());

  }

  fclose(file);

  delete interarrivals;
  delete initialErChmm;
  delete st;

}

FittingOutput* runFitting(int impl, int L, ErChmm *erChmm, Structure *st, Interarrivals *interarrivals){
  if(impl == SERIAL) return runSerFitting(erChmm, st, interarrivals);
  return runParFitting(impl, L, erChmm, st, interarrivals);
}

FittingOutput* runSerFitting(ErChmm *erChmm, Structure *st, Interarrivals *interarrivals){

  char tag[128]="";
  build_tag(tag, 0, st->getBc(), st->getRi(), 1);

  printf("Fitting research for [%s] :\n", tag);




  Random *rnd = new Random();
  ErChmm *initialErChmm = new ErChmm();
  if(erChmm == NULL){
    printf("Generating initial ER-CHMM ...");
    initialErChmm->set(st, interarrivals->getMean(), rnd);
  }else {
    printf("Setting up initial ER-CHMM ...");
    int bc = erChmm->getBc();
    int *ri = erChmm->getRi();
    double *lambda = erChmm->getLambda();
    double  *P = erChmm->getP();
    initialErChmm->set(bc, ri, lambda, P);
  }

  int bc = initialErChmm->getBc();
  int *ri = st->getRi();
  double *alpha = initialErChmm->obtainAlpha();

  float *alphaArr = new float[bc];
  float *lambdaArr = new float[bc];
  float *pArr = new float[bc*bc];

  for(int i = 0; i < bc; i++){
    alphaArr[i] = alpha[i];
    lambdaArr[i] = initialErChmm->getLambda()[i];
    for(int j = 0; j < bc; j++)
      pArr[i*bc+j] = initialErChmm->getP()[i*bc+j];
  }

  printf("... done.\n");

  /* Procedure configuration. */
  int T = interarrivals->getCount();
  float *timeArr = interarrivals->getArr();
  int minIterCount = 100;
  int maxIterCount = 100;
  float eps = 0.001;
  int sumVarCount = 4;
  float maxPhi = 10000;


  ErChmmEm *em = new ErChmmEm();
  printf("Allocating memory ...");
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
  printf("... done.\n");

  printf("Fitting ...");
  double startTime = clock();
  em->calc();
  double endTime = clock();
  double runtime = (double)(endTime-startTime)/CLOCKS_PER_SEC;
  printf("... done.\n");

  printf("Computing log-likelihood ...");
  em->finish();
  printf(" ... done.\n");

  printf("Collecting results ...");
  ErChmm *resultErChmm = new ErChmm();
  resultErChmm->set(bc, ri, em->getLambdaArr(), em->getPArr());

  FittingOutput *fo = new FittingOutput();
  fo->setTag(tag);
  fo->setInitialErChmm(initialErChmm);
  fo->setResultErChmm(resultErChmm);
  fo->setImpl(0);
  fo->setLlh(em->getLogLikelihood());
  fo->setImplLlh(em->getImplLogLikelihood());
  fo->setMem(em->getCpuMemoryUsage(), em->getGpuMemoryUsage());
  fo->setRuntime(runtime);
  fo->setL(1);
  fo->setIterCount(100);

  printf("... done.\n");


  printf("Deallocating memory ...");
  em->destroy(); delete em;
  delete [] alpha;
  delete [] alphaArr;
  delete [] lambdaArr;
  delete [] pArr;

  delete rnd;
  printf("... done.\n");

  return fo;

}

FittingOutput* runParFitting(int impl, int L, ErChmm *erChmm, Structure *st, Interarrivals *interarrivals){
  char tag[128]="";
  build_tag(tag, impl, st->getBc(), st->getRi(), L);

  printf("Fitting research for [%s] :\n", tag);




  Random *rnd = new Random();
  ErChmm *initialErChmm = new ErChmm();
  if(erChmm == NULL){
    printf("Generating initial ER-CHMM ...");
    initialErChmm->set(st, interarrivals->getMean(), rnd);
  }else {
    printf("Setting up initial ER-CHMM ...");
    int bc = erChmm->getBc();
    int *ri = erChmm->getRi();
    double *lambda = erChmm->getLambda();
    double  *P = erChmm->getP();
    initialErChmm->set(bc, ri, lambda, P);
  }

  int bc = initialErChmm->getBc();
  int *ri = st->getRi();
  double *alpha = initialErChmm->obtainAlpha();

  float *alphaArr = new float[bc];
  float *lambdaArr = new float[bc];
  float *pArr = new float[bc*bc];

  for(int i = 0; i < bc; i++){
    alphaArr[i] = alpha[i];
    lambdaArr[i] = initialErChmm->getLambda()[i];
    for(int j = 0; j < bc; j++)
      pArr[i*bc+j] = initialErChmm->getP()[i*bc+j];
  }

  printf("... done.\n");

  /* Procedure configuration. */
  int T = interarrivals->getCount();
  float *timeArr = interarrivals->getArr();
  int minIterCount = 100;
  int maxIterCount = 100;
  float eps = 0.001;

  int partitionSize = (int)ceil((double)T / (double)L);
  int h = 32; // threads per block

  ErChmmEmCuda *em = new ErChmmEmCuda();
  printf("Allocating memory ...");
  em->prepare(
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
  printf("... done.\n");

  printf("Fitting ...");
  double startTime = clock();
  em->calc();
  double endTime = clock();
  double runtime = (double)(endTime-startTime)/CLOCKS_PER_SEC;
  printf("... done.\n");

  printf("Computing log-likelihood ...");
  em->finish();
  printf(" ... done.\n");

  printf("Collecting results ...");
  ErChmm *resultErChmm = new ErChmm();
  resultErChmm->set(bc, ri, em->getLambdaArr(), em->getPArr());

  FittingOutput *fo = new FittingOutput();
  fo->setTag(tag);
  fo->setInitialErChmm(initialErChmm);
  fo->setResultErChmm(resultErChmm);
  fo->setImpl(impl);
  fo->setLlh(em->getLogLikelihood());
  fo->setImplLlh(em->getImplLogLikelihood());
  fo->setMem(em->getCpuMemoryUsage(), em->getGpuMemoryUsage());
  fo->setRuntime(runtime);
  fo->setL(L);
  fo->setIterCount(100);

  printf("... done.\n");


  printf("Deallocating memory ...");
  em->destroy(); delete em;
  delete [] alpha;
  delete [] alphaArr;
  delete [] lambdaArr;
  delete [] pArr;

  delete rnd;
  printf("... done.\n");

  return fo;
}


//
// Command line interface
//

void acquire_string(const char *promptStr, const char *defaultStr, char *output){
  printf("%s ( %s ) : ", promptStr, defaultStr);
  char input[256]="";

  read_line(input);
  char *tinput = strtrm(input);

  if(strlen(tinput) > 0)
    strcpy(output, tinput);
  else
    strcpy(output, defaultStr);

}

void acquire_file_name(const char *promptStr, const char *defaultStr, char *output){
  bool acquired;
  char input[256]="";
  char *tinput;
  do{
    acquired = true;
    printf("%s ( %s ) : ", promptStr, defaultStr);


    read_line(input);
    char *tinput = strtrm(input);

    if(strlen(tinput) > 0)
      strcpy(output, tinput);
    else
      strcpy(output, defaultStr);

    FILE *file =fopen(output,"r");
    if(file == NULL){
      printf("  *** file \'%s\' do not exist!\n", output);
      printf("\n");
      acquired = false;
    }else{
      fclose(file);
    }
  }while(acquired == false);
}

Structure* acquire_structure(const char *promptStr, const char *defaultStr){
  char input[256]="";
  char structStr[256]="";

  char *tinput = NULL;
  bool acquired = false;
  Structure *structure = NULL;

  do{
    acquired = true;
    printf("%s ( %s ) : ", promptStr, defaultStr);

    read_line(input);
    tinput = strtrm(input);

    if(strlen(tinput) > 0)
      strcpy(structStr, tinput);
    else
      strcpy(structStr, defaultStr);

    vector<int> *vecRi = new vector<int>();
    char *token = strtok(structStr, ",; ");
    while(token != NULL){
      int s = strtol(token, NULL, 10);

      if(s == 0){
        printf("  *** parse error : \'%s\' is not a valid number of states!\n", token);
        printf("\n");
        acquired = false;
        break;
      }
      vecRi->push_back(s);
      token = strtok(NULL, ",; ");
    }

    int bc = vecRi->size();
    int *ri = new int[bc];

    for(int i = 0; i < bc; i++)
      ri[i] = (*vecRi)[i];

    structure = new Structure(bc, ri);


  }while(acquired == false);

  return structure;
}

int acquire_int(const char *promptStr, const char *defaultStr){
  bool acquired = true;
  int value = 0;
  do{
    acquired = true;
    printf("%s ( %s ) : ", promptStr, defaultStr);
    char input[256]="";
    char str[256]="";

    read_line(input);
    char *tinput = strtrm(input);

    if(strlen(tinput) > 0)
      strcpy(str, tinput);
    else
      strcpy(str, defaultStr);

    value = strtol(str, NULL, 10);
    if(value == 0 ){
      printf(" *** parse error : \'%s\' is not a valid integer!\n", str);
      printf("\n");
      acquired = false;
    }
    if(value < 0){
      printf(" *** value has to be > 0!\n");
      printf("\n");
      acquired = false;
    }

  }while(acquired == false);

  return value;
}

void acquire_int_range(const char *promptStr, const char *defaultStr, int min, int max, int &outMin, int &outMax){
  bool acquired = true;

  char input[256]="";
  char str[256]="";

  do{
    acquired = true;
    printf("%s ( %s ) : ", promptStr, defaultStr);


    read_line(input);
    char *tinput = strtrm(input);

    if(strlen(tinput) > 0)
      strcpy(str, tinput);
    else
      strcpy(str, defaultStr);

    char *tokenA = strtok(str, ",; ");
    char *tokenB = strtok(NULL, ",; ");

    if(tokenA == NULL || tokenB == NULL){
      printf("  *** please, enter two integers separated by comma!\n");
      printf("\n");
      acquired = false;
      continue;
    }

    int valA = strtol(tokenA, NULL, 10);
    int valB = strtol(tokenB, NULL, 10);

    if(valA == 0){
      printf("  *** failed to parse \'%s\'!\n", tokenA);
      printf("\n");
      acquired = false;
      continue;
    }

    if(valB == 0){
      printf("  *** failed to parse \'%s\'!\n", tokenB);
      printf("\n");
      acquired = false;
      continue;
    }

    if(valA > valB){
      printf("  *** you have entered an invalid range!\n");
      printf("\n");
      acquired = false;
      continue;
    }

    if(valA < min || valA > max || valB < min || valB > max){
      printf("  *** enter a valid range from [ %d, %d ]!\n", min, max);
      printf("\n");
      acquired = false;
      continue;
    }

    outMin = valA;
    outMax = valB;

  }while(acquired == false);

}

double acquire_double(const char *promptStr, const char *defaultStr){
  bool acquired = true;
  double value = 0;
  do{
    acquired = true;
    printf("%s ( %s ) : ", promptStr, defaultStr);
    char input[256]="";
    char str[256]="";

    read_line(input);
    char *tinput = strtrm(input);

    if(strlen(tinput) > 0)
      strcpy(str, tinput);
    else
      strcpy(str, defaultStr);

    value = strtod(str, NULL);
    if(value == 0.0){
      printf(" *** parse error : \'%s\' is not a valid number!\n", str);
      printf("\n");
      acquired = false;
    }

    if(value < 0.0){
      printf(" *** value has to be > 0!\n");
      printf("\n");
      acquired = false;
    }

  }while(acquired == false);

  return value;
}

void acquire_option(const char *promptStr, const char *defaultStr, vector<char*> *options, char *output){

  bool acquired = true;
  char input[256]="";
  char str[256]="";

  do{
    acquired = true;
    printf("%s ( %s ) : ", promptStr, defaultStr);


    read_line(input);
    char *tinput = strtrm(input);

    if(strlen(tinput) > 0)
      strcpy(str, tinput);
    else
      strcpy(str, defaultStr);

    // check if this @str option is valid
    bool found = false;
    for(int i = 0; i < options->size(); i++){
      char* opt = (*options)[i];
      if(strcmp(opt, str) == 0){
        found = true;
        break;
      }
    }

    if(found == false){
      printf("  *** \'%s\' is not a valid option!\n", str);
      printf("\n");
      acquired = false;
    }else{
      strcpy(output, str);
    }

  }while(acquired == false);

}

struct Option {
  char *opt;
  int id;
  Option(char *opt, int id){
    this->opt = opt;
    this->id = id;
  }
};

vector<int>* acquire_options(const char *promptStr, const char *defaultStr, vector<Option*> *options){

  vector<int> *output = new vector<int>();

  bool acquired = true;
  char input[256]="";
  char str[256]="";

  do{
    output->clear();

    acquired = true;
    printf("%s ( %s ) : ", promptStr, defaultStr);


    read_line(input);
    char *tinput = strtrm(input);

    if(strlen(tinput) > 0)
      strcpy(str, tinput);
    else
      strcpy(str, defaultStr);


    char *token = strtok(str, ",; ");
    while(token != NULL){
      // check if this @str option is valid
      int id = -1;
      for(int i = 0; i < options->size(); i++){
        char* opt = (*options)[i]->opt;
        if(strcmp(opt, token) == 0){
          id = (*options)[i]->id;
          break;
        }
      }
      if(id == -1){
        printf("  *** \'%s\' is not a valid option!\n", token);
        printf("\n");
        acquired = false;
        break;
      }

      output->push_back(id);

      token = strtok(NULL, ",; ");
    }





  }while(acquired == false);

  return output;
}



void cmd_gen(){
  printf("\n");
  printf("/ ER-CHMM parameter generation /\n");
  printf("\n");

  char input[256];
  char *tinput = NULL; // trimmed input

  // ER-CHMM file name
  char erchmmFileName[256] = "";
  acquire_string("Enter ER-CHMM file name", "erchmm.txt", erchmmFileName);
  printf("  >> file name : \'%s\'\n", erchmmFileName);
  printf("\n");


  // ER-CHMM structure
  Structure *structure = acquire_structure("Enter number of states in each branch", "1, 2, 3");
  char structStr[256]="";
  structure->str(structStr);
  printf("  >> structure : %s\n", structStr);
  printf("\n");

  // mean
  double mean = acquire_double("Enter sample mean", "1.0");
  printf("  >> sample mean : %f\n", mean);
  printf("\n");

  // Generate
  Random *rnd = new Random();

  ErChmm *erChmm = new ErChmm();
  erChmm->set(structure, mean, rnd);

  erChmm->write_to_file(erchmmFileName);

  printf("The generated ER-CHMM parameters are written to \'%s\'.\n", erchmmFileName);
  printf("\n");

  delete rnd;
  delete erChmm;
  delete structure;

}

void cmd_sim(){
  printf("\n");
  printf("/ ER-CHMM simulation / \n");
  printf("\n");

  char input[256];
  char *tinput;


  // ER-CHMM file name
  char erchmmFileName[256] = "";
  acquire_file_name("Enter ER-CHMM file name", "erchmm.txt", erchmmFileName);
  printf("  >> file name : \'%s\'\n", erchmmFileName);
  printf("\n");

  // output file
  char outputFileName[256]="interarrivals.bin";
  acquire_string("Enter file for saving interarrivals into", "interarrivals.bin", outputFileName);
  printf("  >> file name : \'%s\'\n", outputFileName);
  printf("\n");

  // number of interarrivals

  int count = acquire_int("Enter number of interarrivals", "100000");
  printf("  >> number of interarrivals : %d\n", count);
  printf("\n");

  //
  ErChmm *erChmm = new ErChmm();
  erChmm->read_from_file(erchmmFileName);

  Interarrivals *interarrivals = new Interarrivals();
  interarrivals->generate(erChmm, count);
  interarrivals->write_to_file(outputFileName);

  printf("  The generated interarrivals (mean : %f) are written to \'%s\'.\n", interarrivals->getMean(), outputFileName);
  printf("\n");

  delete  interarrivals;
  delete erChmm;

}

void cmd_llh(){
  printf("\n");
  printf("/ ER-CHMM log-likehooh computation / \n");
  printf("\n");

  char input[256];
  char *tinput;

  // ER-CHMM parameter file
  char erchmmFileName[256] = "";
  acquire_file_name("Enter ER-CHMM file", "erchmm.txt", erchmmFileName);
  printf("  >> file name \'%s\'\n", erchmmFileName);
  printf("\n");

  // interarrivals file
  char interarrivalsFileName[256] = "interarrivals.bin";
  acquire_file_name("Enter interarrivals file", "interarrivals.bin", interarrivalsFileName);
  printf("  >> file name : \'%s\'\n", interarrivalsFileName);
  printf("\n");

  // compute
  ErChmm *erChmm = new ErChmm();
  erChmm->read_from_file(erchmmFileName);

  Interarrivals *interarrivals = new Interarrivals();
  interarrivals->read_from_file(interarrivalsFileName);
  printf("Number of interarrivals : %d\n", interarrivals->getCount());
  double llh = erChmm->obtainLogLikelihood(interarrivals);

  printf("\n");
  printf("Log-likelihood : %.7e\n", llh);
  printf("\n");


  delete erChmm;
  delete interarrivals;


}

void cmd_fit(){
  printf("\n");
  printf("/ Fitting / \n");
  printf("\n");

  char input[256]="";
  char *tinput;
  int minBc, maxBc;

  // interarrivals file
  char interarrivalsFileName[256] = "interarrivals.bin";
  acquire_file_name("Enter interarrivals file", "interarrivals.bin", interarrivalsFileName);
  printf("  >> file name : \'%s\'\n", interarrivalsFileName);
  printf("\n");

  Interarrivals *interarrivals = new Interarrivals();
  interarrivals->read_from_file(interarrivalsFileName);

  printf("Number of interarrivals : %d\n", interarrivals->getCount());
  printf("\n");


  vector<ErChmm *> *erChmmVec = new vector<ErChmm*>();

  // from file or randomly generated ?
  char optionParams[16]="ff";
  printf("Initial ER-CHMM parameters : \n");
  printf("  ff - from file\n");
  printf("  rg - randomly generated \n");

  vector<char*> *options = new vector<char*>();
  options->push_back("ff");
  options->push_back("rg");

  acquire_option("Enter your choice", "ff", options, optionParams);
  printf("  >> option : %s\n", optionParams);
  printf("\n");
  delete options;


  if(strcmp(optionParams, "ff") == 0){

    // ER-CHMM parameter file
    char erchmmFileName[256] = "";
    acquire_file_name("Enter ER-CHMM file", "erchmm.txt", erchmmFileName);
    printf("  >> file name \'%s\'\n", erchmmFileName);
    printf("\n");

    ErChmm *erChmm = new ErChmm();
    erChmm->read_from_file(erchmmFileName);

    erChmmVec->push_back(erChmm);
  }else if(strcmp(optionParams, "rg") == 0){

    strcpy(optionParams, "st");
    printf("Randomly generated ER-CHMM parameters : \n");
    printf("  st - for given structure \n");
    printf("  ss - for given number of states (multiple structures) \n");

    vector<char*> *options = new vector<char*>();
    options->push_back("st");
    options->push_back("ss");

    acquire_option("Enter your choice", "st", options, optionParams);
    printf("  >> option : %s\n", optionParams);
    printf("\n");
    delete options;

    // random ER-CHMM parameters for given structure
    if(strcmp(optionParams, "st") == 0){

      // ER-CHMM structure
      Structure *structure = acquire_structure("Enter number of states in each branch", "1, 2, 3");
      char structStr[256]="";
      structure->str(structStr);
      printf("  >> structure : %s\n", structStr);
      printf("\n");

      Random *rnd = new Random();
      ErChmm *erChmm = new ErChmm();
      erChmm->set(structure, interarrivals->getMean(), rnd);
      erChmmVec->push_back(erChmm);

      delete rnd;
    } else if(strcmp(optionParams, "ss") == 0) {

      // number of states
      int numOfStates = acquire_int("Enter number of states", "5");
      printf("  >> number of states : %d\n", numOfStates);
      printf("\n");

      // number of branches



      minBc = 2;
      maxBc = numOfStates;
      char tmp[256]="";
      sprintf(tmp, "%d, %d", 2, numOfStates);
      acquire_int_range("Enter range for number of branches, R", tmp, 2, numOfStates, minBc, maxBc);
      printf("  >> range : [ %d, %d ]\n", minBc, maxBc);
      printf("\n");

      vector<Structure*> *allStructures = generate_all_structures(numOfStates);
      vector<Structure*> *selStructures = new vector<Structure*>();

      for(int i = 0; i < allStructures->size(); i++){
        Structure *st = (*allStructures)[i];
        if(st->getBc()>= minBc && st->getBc() <= maxBc)
          selStructures->push_back(st);
      }

      Random *rnd = new Random();
      for(int i = 0; i < selStructures->size(); i++){
        Structure *st = (*selStructures)[i];
        ErChmm *erChmm = new ErChmm();
        erChmm->set(st, interarrivals->getMean(), rnd);
        erChmmVec->push_back(erChmm);
      }

      printf("Number of generated structures : %d\n", (int)selStructures->size());
      char structStr[256]="";
      for(int i = 0; i < selStructures->size(); i++){
        Structure *st = (*selStructures)[i];
        st->str(structStr);
        printf("  [ %d ] : %s\n", (i+1), structStr);
      }
      printf("\n");




    }

  }

  // number of iterations
  int iterationCount = acquire_int("Enter number of iterations", "100");
  printf("  >> number of iterations : %d\n", iterationCount);
  printf("\n");

  // number of partitions
  int partitionCount = acquire_int("Enter number of partitions", "7680");
  printf("  >> number of partitions : %d\n", partitionCount);
  printf("\n");


  // the algorithm implementations
  char implOptions[256]="ser,p3d";
  printf("Enter EM algorithm implementations, available : \n");
  printf("  ser - serial implementation\n");
  printf("  p1 - parallel implementation with one pass\n");
  printf("  p2, p2d - parallel implementation with two passes\n");
  printf("  p3, p3d - parallel implementation with three passes\n");

  vector<Option*> *moptions = new vector<Option*>();
  moptions->push_back(new Option("ser", 0));
  moptions->push_back(new Option("p1", 1));
  moptions->push_back(new Option("p2", 2));
  moptions->push_back(new Option("p2d", 3));
  moptions->push_back(new Option("p3", 4));
  moptions->push_back(new Option("p3d", 5));

  vector<int>* implVec = acquire_options("Enter your choice", "ser, p3d", moptions);
  delete moptions;

  printf("  >> options : ");
  char tmp[256]="";
  for(int i = 0; i < implVec->size(); i++){
    impl_str(tmp, (*implVec)[i] );
    printf("%s", tmp);
    if(i < implVec->size() - 1)
    printf(", ");
  }
  printf("\n\n");



  //
  printf("\n/ Fitting has been started. / \n\n");

  char infoFileName[64]="info.txt";
  char llhFileName[64]="llh.txt";
  char summaryFileName[64]="summary.txt";

  char initialErChmmFolderName[256] = "initial-erchmm";
  char resultErChmmFolderName[256] = "result-erchmm";

  // creating folders
  char syscmd[128]="";
  sprintf(syscmd, "mkdir -p %s", initialErChmmFolderName);
  int dir_err = system(syscmd);

  sprintf(syscmd, "mkdir -p %s", resultErChmmFolderName);
  dir_err = system(syscmd);


  FILE *infoFile = fopen(infoFileName, "w");
  fprintf(infoFile, "tag; L; R; impl; struct; llh; mean; runtime; cpu mem; gpu mem; total mem\n");


  // char structStr[128] = "";
  // int_arr_to_str(structStr, initialErChmm->getBc(), initialErChmm->getRi());
  //
  //
  // fprintf(infoFile, "initial-erchmm; - ; %d; - ; %s; %e; %e; - ; - ; - ; -\n",
  //   initialErChmm->getBc(),
  //   structStr,
  //   initialErChmm->obtainLogLikelihood(interarrivals),
  //   initialErChmm->obtainMean());
  // fprintf(infoFile, "sample; - ; - ; - ; - ; - ; %e; - ; - ; - ; - \n", interarrivals->getMean() );

  fclose(infoFile);




  vector<FittingOutput*> *fos = new vector<FittingOutput*>();

  char path[128]="";
  for(int i = 0; i < erChmmVec->size(); i++){
    ErChmm *initialErChmm = (*erChmmVec)[i];
    for(int impl_i = 0; impl_i < implVec->size(); impl_i++){
      int impl = (*implVec)[impl_i];


      Structure *st = new Structure(initialErChmm->getBc(), initialErChmm->getRi());
      FittingOutput *fo = runFitting(impl, partitionCount, initialErChmm, st, interarrivals);


      fos->push_back(fo);

      fo->append_to_file(infoFileName);


      sprintf(path, "%s%c%s.txt", initialErChmmFolderName, kPathSeparator, fo->tag());
      fo->initialErChmm()->write_to_file(path);

      sprintf(path, "%s%c%s.txt", resultErChmmFolderName, kPathSeparator, fo->tag());
      fo->resultErChmm()->write_to_file(path);



    }
  }


  std::sort(fos->begin(), fos->end(),
  []( FittingOutput* lhs,  FittingOutput* rhs)
  {
      return lhs->getLlh() > rhs->getLlh();
  });

  FILE *file = fopen(llhFileName, "w");
  fprintf(file, "tag; llh\n");
  for(int i = 0; i < fos->size(); i++){
    FittingOutput *fo = (*fos)[i];
    fprintf(file, "%s; %.16e\n", fo->tag(), fo->getLlh());

  }

  fclose(file);


  // summary
  minBc = maxBc = (*fos)[0]->getBc();
  for(int i = 1; i < fos->size(); i++){
    int bc = (*fos)[i]->getBc();
    if(bc < minBc) minBc = bc;
    if(bc > maxBc) maxBc = bc;
  }

  int implCount = implVec->size();
  int bcc = maxBc - minBc + 1;
  int caseCount = implCount*bcc;



  int *strCountArr = new int[caseCount];
  double *rtArr = new double[caseCount];
  double *gpuMemArr = new double[caseCount];
  double *cpuMemArr = new double[caseCount];

  for(int i = 0; i < caseCount; i++){
    strCountArr[i] = 0;
    rtArr[i] = 0;
    gpuMemArr[i] = 0;
    cpuMemArr[i] = 0;
  }

  for(int i = 0; i < fos->size(); i++){
    FittingOutput *fo = (*fos)[i];
    int bc_idx = fo->initialErChmm()->getBc() - minBc;
    int impl_idx = -1;
    for(int impl_i = 0; impl_i < implVec->size(); impl_i++){
      if( (*implVec)[impl_i] == fo->getImpl())
        impl_idx = impl_i;
    }
    int idx = impl_idx * bcc + bc_idx;

    strCountArr[idx]++;
    rtArr[idx] += fo->getRuntime();
    gpuMemArr[idx] += fo->getGpuMem();
    cpuMemArr[idx] += fo->getCpuMem();
  }

  for(int i = 0; i < caseCount; i++){
    rtArr[i] /= (double)strCountArr[i];
    gpuMemArr[i] /= (double)strCountArr[i];
    cpuMemArr[i] /= (double)strCountArr[i];

  }

  file = fopen(summaryFileName, "w");

    int L;
    char implStr[6]="";
    for(int impl_i = 0; impl_i < implVec->size(); impl_i++){
      int impl = (*implVec)[impl_i];
      if(impl == SERIAL)
        L = 1;
      else
        L = partitionCount;

      impl_str(implStr, impl);

      fprintf(file, "L = %d, %s\n", L, implStr);
      fprintf(file, "-----------------------------\n");
      fprintf(file, "R; runtime (seconds); total mem; cpu mem; gpu mem\n");
      for(int i = 0; i < bcc; i++){
        int bc = i + minBc;
        int idx = impl_i*bcc + i;
        fprintf(file, "%d; %.3f; %.3f; %.3f; %.3f\n", bc, rtArr[idx], (cpuMemArr[idx]+gpuMemArr[idx]), cpuMemArr[idx], gpuMemArr[idx]);
      }
      fprintf(file, "\n");

    }

    fclose(file);




  printf("\n/ Fitting is done. / \n\n");

  printf("  Results written: \n");
  printf("    %s%c - initial ER-CHMM parameters,\n", initialErChmmFolderName, kPathSeparator);
  printf("    %s%c - result ER-CHMM parameters,\n", resultErChmmFolderName, kPathSeparator);
  printf("    %s - all information about fitting,\n", infoFileName);
  printf("    %s - summary of runtimes and memory usage,\n", summaryFileName);
  printf("    %s - log-likelihood values, in descending order.\n", llhFileName);
  printf("\n");





}

void cmd_cnv(){
  printf("\n");
  printf("/ Convert ER-CHMM parameters to MAP. / \n");
  printf("\n");

  // ER-CHMM parameter file
  char erchmmFileName[256] = "";
  acquire_file_name("Enter ER-CHMM file", "erchmm.txt", erchmmFileName);
  printf("  >> file name \'%s\'\n", erchmmFileName);
  printf("\n");

  // MAP parameter file
  char mapFileName[256] = "";
  acquire_string("Enter MAP file", "map.txt", mapFileName);
  printf("  >> file name \'%s\'\n", mapFileName);
  printf("\n");

  //
  ErChmm *erChmm = new ErChmm();
  erChmm->read_from_file(erchmmFileName);

  Map *map = new Map();

  map->set(erChmm);

  map->write_to_file(mapFileName);

  printf("The MAP parameters have been written to \'%s\'.\n\n", mapFileName);

  delete erChmm;
  delete map;

}

void cmd_res(){

  printf("\n");
  printf("/ Research computations for paper. / \n");
  printf("\n");
  printf("  ! The research can be interrupted (by closing application). The next time it will be launched it will resume the computations.\n\n");

  Research *research = new Research();
  research->run();
  delete research;

}


int main(int argc, char *argv[]){

  printf("Parallel algorithms for fitting Markov Arrival Processes, 2016-2018\n");
  printf("Copyright: Mindaugas Bražėnas, Gábor Horváth, Miklós Telek\n\n");
  printf("\n");
  printf("This software is a supplimentary material to the paper\n");
  printf("\'Parallel algorithms for fitting Markov Arrival Processes\'\n\n");



  if(run_all_tests() == false){
    return 0;
  }

//   printf(" > ");
//   char input[256] = "";
// fgets(input, sizeof(input), stdin);

  char input[256];




  do {
    printf("Available options: \n");
    printf("  gen - generate ER-CHMM parameters\n");
    printf("  sim - simulate ER-CHMM process\n");
    printf("  llh - compute log-likelihood\n");
    printf("  fit - perform fitting\n");
    printf("  cnv - convert ER-CHMM parameters to MAP\n");
    printf("  res - perform paper research\n");
    printf("  q   - quit\n");
    printf("Select your option : ");
    read_line(input);
    printf("\n");

    if(strcmp(input, "gen") == 0)
      cmd_gen();
    else if(strcmp(input, "sim") == 0)
      cmd_sim();
    else if(strcmp(input, "llh") == 0)
      cmd_llh();
    else if(strcmp(input, "fit") == 0)
      cmd_fit();
    else if(strcmp(input, "cnv") == 0)
      cmd_cnv();
    else if(strcmp(input, "res") == 0)
      cmd_res();

  }while(strcmp(input, "q") != 0);

  return 0;

}
