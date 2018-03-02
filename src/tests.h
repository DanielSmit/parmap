#ifndef TESTS_HEADER
#define TESTS_HEADER

#include <stdio.h>
#include "mapem.h"
#include "prog.h"

bool test_sole_gauss();

bool test_matrix_inverse();


bool test_erchmm_structure_generation();

bool test_stationary_prob_computation();

bool test_stationary_prob_computation_for_general_map();

bool test_erchmm_mean_computation();


bool test_interarrival_generation();

bool test_erchmm_to_general_map();


bool test_em(int impl);





void run_all_tests();

#endif /* TESTS_HEADER */
