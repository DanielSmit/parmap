#ifndef PROG_HEADER
#define PROG_HEADER


#include <stdio.h>
#include <vector>
#include <cmath>
#include <random>

#include "tests.h"

using namespace std;

class Random;
class ErChmm;

//
// General math
//
void mat_print_out(int n, double *mat);
void mul_vec_mat(int n, double *vec, double *mat, double *out);
void mul_mat_mat(int n, double *matA, double *matB, double *matC);
void mat_unit(int n, double *mat);
void mat_inv(int n, double *matA, double *matInv);
void mat_scale(int n, double *mat, double scale);
double vec_sum_up(int n, double *vec);



void transpose(int n, double *mat);
bool sole_gauss(int n, double *matA, double *vecx, double *vecb);
bool sole_gauss_mat(int n, double *matA, double *matX, double *matB);


int prob_selection(int n, double *probArr, double prob);

double exp_dist_cdf_inv(double lambda, double arg);
double erlang_rnd(int r, double lambda, Random *rnd);


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
// Map
//
class Map {
  private:
    int mSize;
    double *mD0;
    double *mD1;

  public:
    void set(ErChmm *erChmm);
    int getSize();
    double *getD0();
    double *getD1();

    double *obtainAlpha();
    double obtainMean();


    void print_out();

};

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

//
// Random
//
class Random {
  private:
    std::mt19937 *mGen;
    std::uniform_real_distribution<double> *mDist;

  public:
    Random();
    double next();

};

//
// Interarrivals
//
class Interarrivals {
  private:

    int mCount;
    double mMean;
    float *mArr;

  public:
    void generate(ErChmm *erChmm, int count);

    int getCount();
    double getMean();
    float* getArr();

    void print_out();
    void write_to_file(const char *fileName);
    void read_from_file(const char *fileName);

};

#endif /* PROG_HEADER */
