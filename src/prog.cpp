#include "prog.h"
#include <stdio.h>
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
  printf("Generate all\n");

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





int main(int argc, char *argv[]){

  printf("Parallel algorithms for fitting Markov Arrial process fitting, 2017-2018\n");
  printf("\n");

  run_all_tests();

  int bc = 2;
  int *ri = new int[bc];
  double *lambda = new double[bc];
  double *P = new double[bc*bc];
  ri[0] = 1;
  ri[1] = 2;

  lambda[0] = 1.5;
  lambda[1] = 2.5;

  P[0*bc+0] = 0.2;
  P[0*bc+1] = 0.8;
  P[1*bc+0] = 0.7;
  P[1*bc+1] = 0.3;

  ErChmm *erChmm = new ErChmm();
  erChmm->set(bc, ri, lambda, P);

  erChmm->print_out();
  erChmm->write_to_file("params.erchmm");

  ErChmm *erChmm2 = new ErChmm();
  erChmm2->read_from_file("params.erchmm");
  erChmm2->print_out();


  return 0;

}
