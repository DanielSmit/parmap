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



int main(int argc, char *argv[]){

  printf("Parallel algorithms for fitting Markov Arrial process fitting, 2017-2018\n");
  printf("\n");

  run_all_tests();




  return 0;

}
