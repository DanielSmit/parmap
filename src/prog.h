#ifndef PROG_HEADER
#define PROG_HEADER


#include <stdio.h>
#include <vector>

#include "tests.h"

using namespace std;

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

#endif /* PROG_HEADER */
