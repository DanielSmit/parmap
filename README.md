**# parmap**

We (Mindaugas Bražėnas, Gábor Horváth, Mikós Telek) have developed three parallel algorithms for fitting MAPs of ER-CHMM structure class. The procedures are implemented to be executed on GPU card using CUDA library. More details will be available in a paper 'Parallel algorithms for fitting Markov Arrival Processes'.


To build the program and and launch it, execute 
```
$ cmake ..
$ make
$ ./prog
```

Upon start, the program performs some tests to check the code. Your these tests if you are modifying the source code. 

```
/ Testing code ... 

  test_sole_gauss ... PASSED
  test_matrix_inverse ... PASSED
  test_erchmm_structure_generation ... PASSED
  test_stationary_prob_computation ... PASSED
  test_stationary_prob_computation_for_general_map ... PASSED
  test_erchmm_mean_computation ... PASSED
  test_interarrival_generation ... PASSED
  test_erchmm_to_general_map ... PASSED
  test_erchmm_generation ... PASSED
  test_em P_1 ... PASSED
  test_em P_2 ... PASSED
  test_em P_2_D ... PASSED
  test_em P_3 ... PASSED
  test_em P_3_D ... PASSED

/ All tests passed.
```

If all the tests passes, program shows available options:
```
Available options: 
  gen - generate ER-CHMM parameters
  sim - simulate ER-CHMM process
  llh - compute log-likelihood
  fit - perform fitting
  cnv - convert ER-CHMM parameters to MAP
  res - perform paper research
  q   - quit
Select your option : 
```

Next, we will go through a few of these options.

**Generation of ER-CHMM parameters**
```
Select your option : gen

/ ER-CHMM parameter generation /

Enter ER-CHMM file name ( erchmm.txt ) : 
  >> file name : 'erchmm.txt'

Enter number of states in each branch ( 1, 2, 3 ) : 3,1
  >> structure : 3, 1

Enter sample mean ( 1.0 ) : 2.0
  >> sample mean : 2.000000

The generated ER-CHMM parameters are written to 'erchmm.txt'
```
The Erlang branch transition rates are assigned value of 1, and then rescaled to match the given sample mean. The contents of generated file 'erchmm.txt':
```
# number of branches
2
# structure
3, 1
# lambda 
0.9862176464158663
0.9862176464158663
# P 
0.1395971306464416, 0.8604028693535585
0.8142417799836712, 0.1857582200163287
```
**Interarrival generation**
```
Select your option : sim

/ ER-CHMM simulation / 

Enter ER-CHMM file name ( erchmm.txt ) : 
  >> file name : 'erchmm.txt'

Enter file for saving interarrivals into ( interarrivals.bin ) : 
  >> file name : 'interarrivals.bin'

Enter number of interarrivals ( 100000 ) :  
  >> number of interarrivals : 100000

  The generated interarrivals (mean : 1.996757) are written to 'interarrivals.bin'.
```
The generated interarrivals, obtained by simulating ER-CHMM process, are written into a binary file. The first 32bit interger denotes number of interarrivals, then follows an array of interarrival values (32bit single precision floating point numbers).

**Fitting**
We have implemented various ways to select initial ER-CHMM parameters for fitting. Here the most general will be shown.
```
Select your option : fit


/ Fitting / 

Enter interarrivals file ( interarrivals.bin ) : 
  >> file name : 'interarrivals.bin'

Number of interarrivals : 100000

Initial ER-CHMM parameters : 
  ff - from file
  rg - randomly generated 
Enter your choice ( ff ) : rg
  >> option : rg

Randomly generated ER-CHMM parameters : 
  st - for given structure 
  ss - for given number of states (multiple structures) 
Enter your choice ( st ) : ss
  >> option : ss

Enter number of states ( 5 ) : 
  >> number of states : 5

Enter range for number of branches, R ( 2, 5 ) : 2, 4
  >> range : [ 2, 4 ]

Number of generated structures : 5
  [ 1 ] : 4, 1
  [ 2 ] : 3, 2
  [ 3 ] : 3, 1, 1
  [ 4 ] : 2, 2, 1
  [ 5 ] : 2, 1, 1, 1

Enter number of iterations ( 100 ) : 
  >> number of iterations : 100

Enter number of partitions ( 7680 ) : 
  >> number of partitions : 7680

Enter EM algorithm implementations, available : 
  ser - serial implementation
  p1 - parallel implementation with one pass
  p2, p2d - parallel implementation with two passes
  p3, p3d - parallel implementation with three passes
Enter your choice ( ser, p3d ) : ser, p2, p3d      
  >> options : ser, p2, p3d


/ Fitting has been started. / 

Fitting research for [L1-R2-ser-4-1] :
Setting up initial ER-CHMM ...... done.
Allocating memory ...... done.
Fitting ...... done.
Computing log-likelihood ... ... done.
Collecting results ...... done.
Deallocating memory ...... done.
Fitting research for [L7680-R2-p2-4-1] :
Setting up initial ER-CHMM ...... done.
Allocating memory ...... done.
Fitting ...... done.
Computing log-likelihood ... ... done.
Collecting results ...... done.
Deallocating memory ...... done.

  [ ... omitted some lines here ... ] 

Fitting research for [L7680-R4-p3d-2-1-1-1] :
Setting up initial ER-CHMM ...... done.
Allocating memory ...... done.
Fitting ...... done.
Computing log-likelihood ... ... done.
Collecting results ...... done.
Deallocating memory ...... done.


/ Fitting is done. / 

```
The results are written into several files. 

'info.txt' - contains information about each fitting procedure. Memory usage is in megabytes, and runtime is measured in seconds. 
```
tag; L; R; impl; struct; llh; mean; runtime; cpu mem; gpu mem; total mem
L1-R2-ser-4-1; 1; 2; ser; 4, 1; -1.628860e+05; 1.996752e+00; 2.110; 3.433; 0.000; 3.433
L7680-R2-p2-4-1; 7680; 2; p2; 4, 1; -1.628860e+05; 1.996749e+00; 0.142; 1.063; 1.063; 2.126
L7680-R2-p3d-4-1; 7680; 2; p3d; 4, 1; -1.628860e+05; 1.996756e+00; 0.091; 0.899; 3.951; 4.850
L1-R2-ser-3-2; 1; 2; ser; 3, 2; -1.676135e+05; 1.996752e+00; 2.221; 3.433; 0.000; 3.433
L7680-R2-p2-3-2; 7680; 2; p2; 3, 2; -1.676135e+05; 1.996748e+00; 0.143; 1.063; 1.063; 2.126
L7680-R2-p3d-3-2; 7680; 2; p3d; 3, 2; -1.676135e+05; 1.996751e+00; 0.091; 0.899; 3.951; 4.850
L1-R3-ser-3-1-1; 1; 3; ser; 3, 1, 1; -1.624684e+05; 1.996753e+00; 2.594; 4.578; 0.000; 4.578
L7680-R3-p2-3-1-1; 7680; 3; p2; 3, 1, 1; -1.624684e+05; 1.996754e+00; 0.261; 1.908; 1.907; 3.815
L7680-R3-p3d-3-1-1; 7680; 3; p3d; 3, 1, 1; -1.624684e+05; 1.996753e+00; 0.111; 1.254; 5.450; 6.703
L1-R3-ser-2-2-1; 1; 3; ser; 2, 2, 1; -1.633232e+05; 1.996753e+00; 2.580; 4.578; 0.000; 4.578
L7680-R3-p2-2-2-1; 7680; 3; p2; 2, 2, 1; -1.633232e+05; 1.996756e+00; 0.259; 1.908; 1.907; 3.815
L7680-R3-p3d-2-2-1; 7680; 3; p3d; 2, 2, 1; -1.633232e+05; 1.996753e+00; 0.112; 1.254; 5.450; 6.703
L1-R4-ser-2-1-1-1; 1; 4; ser; 2, 1, 1, 1; -1.633241e+05; 1.996753e+00; 3.117; 5.722; 0.000; 5.722
L7680-R4-p2-2-1-1-1; 7680; 4; p2; 2, 1, 1, 1; -1.633241e+05; 1.996750e+00; 0.484; 3.352; 3.352; 6.703
L7680-R4-p3d-2-1-1-1; 7680; 4; p3d; 2, 1, 1, 1; -1.633241e+05; 1.996751e+00; 0.141; 1.717; 7.057; 8.774
```
A particular search is identified by a tag, for example, the tag ```L7680-R3-p2-3-1-1``` is used for a parallel search ('P2') using 7680 partitions, ER-CHMM structure {3, 1, 1} with 3 branches. 

In order to easier judge about runtimes, results are summarized in 'summary.txt' for every R:
```
L = 1, ser
-----------------------------
R; runtime (seconds); total mem; cpu mem; gpu mem
2; 2.166; 3.433; 3.433; 0.000
3; 2.587; 4.578; 4.578; 0.000
4; 3.117; 5.722; 5.722; 0.000

L = 7680, p2
-----------------------------
R; runtime (seconds); total mem; cpu mem; gpu mem
2; 0.142; 2.126; 1.063; 1.063
3; 0.260; 3.815; 1.908; 1.907
4; 0.484; 6.703; 3.352; 3.352

L = 7680, p3d
-----------------------------
R; runtime (seconds); total mem; cpu mem; gpu mem
2; 0.091; 4.850; 0.899; 3.951
3; 0.111; 6.703; 1.254; 5.450
4; 0.141; 8.774; 1.717; 7.057

```
Also, in order to analyse which structures are the best for fitting, the sorted tags are given in 'llh.txt' file:
```
tag; llh
L1-R3-ser-3-1-1; -1.6246840625000000e+05
L7680-R3-p3d-3-1-1; -1.6246840625000000e+05
L7680-R3-p2-3-1-1; -1.6246842187500000e+05
L1-R2-ser-4-1; -1.6288598437500000e+05
L7680-R2-p2-4-1; -1.6288598437500000e+05
L7680-R2-p3d-4-1; -1.6288598437500000e+05
L1-R3-ser-2-2-1; -1.6332318750000000e+05
L7680-R3-p2-2-2-1; -1.6332318750000000e+05
L7680-R3-p3d-2-2-1; -1.6332318750000000e+05
L1-R4-ser-2-1-1-1; -1.6332409375000000e+05
L7680-R4-p2-2-1-1-1; -1.6332409375000000e+05
L7680-R4-p3d-2-1-1-1; -1.6332409375000000e+05
L1-R2-ser-3-2; -1.6761348437500000e+05
L7680-R2-p2-3-2; -1.6761348437500000e+05
L7680-R2-p3d-3-2; -1.6761348437500000e+05
```
Finally, the initial ER-CHMM parameters used for fitting are written to 'initial-erchmm/' folder, and the resulting ER-CHMM parameters are written to 'result-erchmm/' folder.

**Research**
To replicate all the results presented in the paper execute:
```
Select your option : res


/ Research computations for paper. / 

  ! The research can be interrupted (by closing application). The next time it will be launched it will resume the computations.


Research started. 

Checking for meta file './research/.meta' ...... found. Resuming research.
Reading meta file 'research/.meta' ...... done.
Reading ER-CHMM from './research/gen-erchmm.txt' ...... done.
Reading interarrivals from './research/interarrivals.bin' ...... done.
Reading meta file 'research/.meta' ...... done.

( step : 0 )

  [ ... omitted many lines here ... ]
```

All the research results are written to 'research/' folder. All the research computations, depending on hardware, can take a long time to complete, therefore we have implemented interruption handling. The computations can be stoped (forcefully) at any time and will be resume upon next launch.

