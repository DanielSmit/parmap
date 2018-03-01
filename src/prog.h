#ifndef PROG_HEADER
#define PROG_HEADER


#include <stdio.h>
#include <vector>

#include "tests.h"

using namespace std;

//
// General math
//
void mat_print_out(int n, double *mat);
void mul_vec_mat(int n, double *vec, double *mat, double *out);

void transpose(int n, double *mat);
bool sole_gauss(int n, double *matA, double *vecx, double *vecb);

//
// Structure
//
// The representation of ER-CHMM structure.
//
class Structure {
  private:
    int mBc;   // number of Erlang branches
    int *mRi;  // number of states in each Erlang branch

  public:
    Structure(int bc, int *ri);

    void set(int bc, int *ri);
    int getBc();
    int* getRi();
    void print_out();



};

// ER-CHMM structure generation
void first_structure(int n, int &bc, int *ri);
void next_structure(int n, int &bc, int *ri);
bool last_structure(int n, int &bc, int *ri);
vector<Structure*>* generate_all_structures(int n);

//
// ErChmm
//
// ER-CHMM parameters
//
class ErChmm {
  private:
    int mBc;
    int *mRi;
    double *mLambda; // Erlang branch transition rates
    double *mP;      // Erlang branch switching probabilities

  public:
    ErChmm();
    void set(int bc, int *ri, double *lambda, double *P);

    int getBc();
    int *getRi();
    double* getLambda();
    double* getP();
    double* obtainAlpha();


    void print_out();
    void write_to_file(const char *fileName);
    void read_from_file(const char *fileName);

};

#endif /* PROG_HEADER */
