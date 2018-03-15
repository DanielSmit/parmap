#ifndef PROG_HEADER
#define PROG_HEADER



#include <stdio.h>
#include <vector>
#include <cmath>
#include <random>

#include <cstdlib>
#include <algorithm>

#include "tests.h"

using namespace std;

const char kPathSeparator =
#if defined _WIN32 || defined __CYGWIN__
    '\\';
#else
    '/';
#endif

#define NUM_STATES 10
#define NUM_IMPL 6

class Random;
class ErChmm;
class Interarrivals;

void int_arr_to_str(char *str, int n, int *arr);
void int_arr_to_str_compact(char *str, int n, int *arr);
void impl_str(char *str, int impl);
int impl_id(char *str);

char *strtrm(char *str);
void read_line(char *str);

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
// Client
//
void cmd_gen();
void cmd_sim();
void cmd_llh();
void cmd_fit();
void cmd_cnv();
void cmd_res();

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
    void str(char *str);




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
    void write_to_file(const char *fileName);

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
    void set(int bc, int *ri, float *lambdaArr, float *PArr);

    void set(Structure *st, double mean, Random *rnd);

    int getBc();
    int *getRi();
    double* getLambda();
    double* getP();
    double obtainMean();
    double* obtainAlpha();
    double obtainLogLikelihood(Interarrivals *interarrivals);


    void print_out();
    void write_to_file(const char *fileName);
    void read_from_file(const char *fileName);

  private:
    double calculateMean(int bc, int *ri, double *alpha, double *lambda);
    double* calculateStationaryProb(int bc, double *P);


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

  private:
    double computeMean(int count, float *arr);

};

//
// FittingOutput
//
class FittingOutput {
  private:
    char mTag[64]="";
    ErChmm *mInitialErChmm;
    ErChmm * mResultErChmm;
    int mImpl; // SERIAL / P_1 / P_2 / P_2_D / ..
    double mLlh; // log-likelihood
    double mImplLlh; // log-likelihood obtained from running the fitting algorithm
    double mCpuMem;
    double mGpuMem;
    double mTotalMem;
    double mRuntime;
    int mL; // number of partitions
    int mIterCount;

  public:
    void setTag(char *tag);
    void setInitialErChmm(ErChmm *erChmm);
    void setResultErChmm(ErChmm *erChmm);
    void setImpl(int impl);
    void setLlh(double llh);
    void setImplLlh(double implLlh);
    void setMem(double  cpuMem, double gpuMem);
    void setRuntime(double runtime);
    void setL(int L);
    void setIterCount(int iterCount);

    char* tag();
    ErChmm* initialErChmm();
    ErChmm* resultErChmm();

    int getBc();
    int getL();
    int getImpl();
    double getRuntime();
    double getCpuMem();
    double getGpuMem();
    double getLlh();

    void append_to_file(const char *filePath);
    void append_to_file_bin(const char *filePath);
    void read_from_file_bin(FILE * file);

    bool operator < (const FittingOutput*& fo) const
    {
        return (mLlh < fo->mLlh);
    }

  private:

};

void build_tag(char *str, int impl, int bc, int *ri, int L);

//
// Research
//
// Performs research computations.
//
class Research {
  private:
    const char* mFolder = "research";
    const char* mMetaFileName = ".meta";
    const int mInterarrivalCount = 50000000;
    const char* mGenErChmmFile = "gen-erchmm.txt";
    const char* mInterarrivalFile = "interarrivals.bin";
    const char* mInfoFile = "info.txt";
    const char* mInfoFileBin = "info.bin";
    const char* mInitialErChmmFolder = "initial-erchmm";
    const char* mResultErChmmFolder = "result-erchmm";
    const char* mSummaryFile = "summary.txt";
    const char* mLlhFile = "llh.txt";




  public:
    void run();

  private:
    int read_meta_file();
    void write_meta_file(int step);
    ErChmm* erChmmForGeneratingInterarrivals();



};

class Research2 {
  private:
    const char* mInterarrivalFile = "interarrivals.bin";
    const char* mInfoFile = "info.txt";
    const char* mInfoFileBin = "info.bin";
    const char* mLlhFile = "llh.txt";
    const char* mInitialErChmm = "initial-erchmm.txt";
    const char* mResultErChmmFolder = "result-erchmm";

  public:
    void run();

};

FittingOutput* runFitting(int impl, int L, ErChmm *erChmm, Structure *st, Interarrivals *interarrivals);

FittingOutput* runSerFitting(ErChmm *erChmm, Structure *st, Interarrivals *interarrivals);
FittingOutput* runParFitting(int impl, int L, ErChmm *erChmm, Structure *st, Interarrivals *interarrivals);


#endif /* PROG_HEADER */
