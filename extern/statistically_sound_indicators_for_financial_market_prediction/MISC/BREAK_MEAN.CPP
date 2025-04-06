/******************************************************************************/
/*                                                                            */
/*  BREAK_MEAN - Stationarity test: break in mean                             */
/*                                                                            */
/******************************************************************************/

// Insert all required includes here

#define DEBUG_THREADS 0

/*
--------------------------------------------------------------------------------

   Routine to compute break in means statistic - multiple-comparisons version
   This is a complete routine that should compile correctly standing alone.

--------------------------------------------------------------------------------
*/

double compute_break_mean(int ncases, // Number of rows in database
                          int n_vars, // Number of columns in database
                          int varnum, // Column index of variable to be tested
                          int min_recent,     // Minimum size of recent history
                          int max_recent,     // Maximum size of recent history
                          int comparisons,    // Number of multiple comparisons
                          int *shuffle_index, // Indices for shuffling rows;
                                              // NULL if handling original data
                          double *database,   // Full database
                          int *ibreak,        // Returns location of break
                          int *iwork,         // Work vector ncases long
                          double *dwork       // Work vector 2 * ncases long
) {
  int i, icomp, irecent, istart1, istart2, ntot, n1, n2;
  double *xwork, crit, max_crit;

  // Get the data
  xwork = dwork + ncases;
  if (shuffle_index == NULL) { // Original data?
    for (i = 0; i < ncases; i++)
      xwork[i] = database[i * n_vars + varnum];
  } else {
    for (i = 0; i < ncases; i++)
      xwork[i] = database[shuffle_index[i] * n_vars + varnum];
  }

  ntot = ncases - comparisons + 1; // total number of cases in each test
  max_crit = -1.e40;

  for (irecent = min_recent; irecent <= max_recent; irecent++) {
    n1 = ntot - irecent;
    n2 = irecent;
    for (icomp = comparisons - 1; icomp >= 0; icomp--) {
      istart1 = icomp;
      istart2 = istart1 + n1;
      U_test(n1, xwork + istart1, n2, xwork + istart2, iwork, dwork, &crit);
      if (fabs(crit) > max_crit) {
        max_crit = fabs(crit);
        if (shuffle_index ==
            NULL) // Return boundary location for only unshuffled rep
          *ibreak = irecent;
      }
      if (shuffle_index == NULL) // If original data
        break;                   // Just do most recent set
    } // For all comparisons
  } // For all trial boundaries

  return max_crit;
}

/*
--------------------------------------------------------------------------------

   Routine to compute break in means statistic - serial correlation version
   This is a complete routine that should compile correctly standing alone.

--------------------------------------------------------------------------------
*/

double compute_break_mean_corr(
    int ncases,       // Number of rows in database
    int n_vars,       // Number of columns in database
    int varnum,       // Column index of variable to be tested
    int min_recent,   // Minimum size of recent history
    int max_recent,   // Maximum size of recent history
    int corr_lag,     // Maximum extent of serial correlation
    int irep,         // Replication number, for shuffling or not, and seed
    double *database, // Full database
    int *ibreak,      // Returns location of break
    int *iwork,       // Work vector ncases long
    double *dwork     // Work vector 2 * ncases long
) {
  int i, j, k, n, iseed, offset, nrecent, n1, n2;
  double *xwork, dtemp, crit, max_crit;

  ++corr_lag; // Make it inclusive
  max_crit = -1.e40;

  // Outermost loop processes all offsets from start
  for (offset = 0; offset < corr_lag; offset++) {

    // Get the data
    xwork = dwork + ncases;
    n = 0;
    for (i = offset; i < ncases; i += corr_lag) {
      k = ncases - 1 -
          i; // Data is in chronological order; make xwork[0] the most recent
      xwork[n++] = database[k * n_vars + varnum];
    }

    // If we are in a shuffle rep, shuffle the data
    if (irep) {
      iseed = irep;
      fast_unif(&iseed); // Warm up the random generator
      fast_unif(&iseed); // Ditto
      i = n;             // Number remaining to be shuffled
      while (i > 1) {    // While at least 2 left to shuffle
        j = (int)(fast_unif(&iseed) * i);
        if (j >= i)
          j = i - 1;
        dtemp = xwork[--i];
        xwork[i] = xwork[j];
        xwork[j] = dtemp;
      }
    }

    // This inner loop tries all boundaries in user-specified range
    for (nrecent = min_recent; nrecent <= max_recent;
         nrecent++) { // Boundary is when we have nrecent most recent
      if (nrecent < offset + 1)
        continue;
      n1 = (nrecent - offset - 1) / corr_lag + 1;
      n2 = n - n1;
      if (n2 < 1) // Can happen only in extreme situations
        continue;
      U_test(n1, xwork, n2, xwork + n1, iwork, dwork, &crit);
      if (fabs(crit) > max_crit) {
        max_crit = fabs(crit);
        if (irep == 0)
          *ibreak = nrecent;
      }
    }
  } // For offset

  return max_crit;
}

/*
*******************************************************************************

   All code from here on is intended to integrate into the VarScreen program.
   As a result, this code will not compile in a different environment.
   However, this code should provide a solid template for implementing your
   own interface to perform mean-break analysis.

*******************************************************************************
*/

/*
--------------------------------------------------------------------------------

   Thread stuff...
      Structure for passing information to/from threaded code
      Threaded code called by the main subroutine

--------------------------------------------------------------------------------
*/

typedef struct {
  int ivar;             // Index of predictor (in preds, not database)
  int varnum;           // Index of predictor (in database, not preds)
  int ncases;           // Number of cases
  int n_vars;           // Number of columns in database
  int min_recent;       // Minimum size of recent history
  int max_recent;       // Maximum size of recent history
  int comparisons_corr; // Number of multiple comparisons or correlation lag
  int mult_vs_dep;      // 1 if multiple comparisons, 0 if correlation lag
  int *shuffle_index;   // Index into rows of database for shuffling; ignored if
                        // mult_vs_dep=0
  int irep;             // Replication number; ignored if mult_vs_dep=1
  double *database;     // Full database
  int ibreak;           // Returns location of break
  int *iwork;           // Work area ncases long
  double *dwork;        // Work area 2*ncases long
  double crit;          // Criterion is returned here
} BREAK_MEAN_PARAMS;

static unsigned int __stdcall break_mean_threaded(LPVOID dp) {
  int ibreak;
  double crit;

  if (((BREAK_MEAN_PARAMS *)dp)->mult_vs_dep)
    crit = compute_break_mean(
        ((BREAK_MEAN_PARAMS *)dp)->ncases, ((BREAK_MEAN_PARAMS *)dp)->n_vars,
        ((BREAK_MEAN_PARAMS *)dp)->varnum,
        ((BREAK_MEAN_PARAMS *)dp)->min_recent,
        ((BREAK_MEAN_PARAMS *)dp)->max_recent,
        ((BREAK_MEAN_PARAMS *)dp)->comparisons_corr,
        ((BREAK_MEAN_PARAMS *)dp)->shuffle_index,
        ((BREAK_MEAN_PARAMS *)dp)->database, &ibreak,
        ((BREAK_MEAN_PARAMS *)dp)->iwork, ((BREAK_MEAN_PARAMS *)dp)->dwork);
  else
    crit = compute_break_mean_corr(
        ((BREAK_MEAN_PARAMS *)dp)->ncases, ((BREAK_MEAN_PARAMS *)dp)->n_vars,
        ((BREAK_MEAN_PARAMS *)dp)->varnum,
        ((BREAK_MEAN_PARAMS *)dp)->min_recent,
        ((BREAK_MEAN_PARAMS *)dp)->max_recent,
        ((BREAK_MEAN_PARAMS *)dp)->comparisons_corr,
        ((BREAK_MEAN_PARAMS *)dp)->irep, ((BREAK_MEAN_PARAMS *)dp)->database,
        &ibreak, ((BREAK_MEAN_PARAMS *)dp)->iwork,
        ((BREAK_MEAN_PARAMS *)dp)->dwork);

  ((BREAK_MEAN_PARAMS *)dp)->crit = crit;
  ((BREAK_MEAN_PARAMS *)dp)->ibreak = ibreak;
  return 0;
}

/*
--------------------------------------------------------------------------------

   Main subroutine to compute and print stationarity break in means

--------------------------------------------------------------------------------
*/

int break_mean(
    int npred,            // Number of predictors
    int *preds,           // Their indices are here
    int min_recent,       // Minimum size of recent history
    int max_recent,       // Maximum size of recent history
    int comparisons_corr, // Number of multiple comparisons, or correlation lag
    int mult_vs_dep,      // 1=multiple comparisons; 0=dependency
    int mcpt_reps         // Number of MCPT replications
) {
  int i, j, k, ret_val, ivar, irep, *index, ithread, n_threads, max_threads,
      empty_slot;
  int *mcpt_solo, *mcpt_bestof, *shuffle_index, *breaks, *work_shuffle_index,
      *iwork;
  char msg[256], msg2[256];
  double *crit, *original_crits, *sorted_crits, best_crit, *dwork;
  BREAK_MEAN_PARAMS break_mean_params[MAX_THREADS];
  HANDLE threads[MAX_THREADS];

  MEMTEXT("BREAK_MEAN: break_mean()");

  crit = NULL;
  index = NULL;
  iwork = NULL;
  dwork = NULL;

  ret_val = 0;

  max_threads = max_threads_limit;
  if (max_threads > npred)
    max_threads = npred;

  audit("");
  audit("");
  audit("******************************************************************");
  audit("*                                                                *");
  audit("* Computing stationarity test for break in means                 *");
  sprintf(msg,
          "* %5d predictor candidates                                     *",
          npred);
  audit(msg);
  sprintf(msg,
          "* %5d Minimum recent history cases                             *",
          min_recent);
  audit(msg);
  sprintf(msg,
          "* %5d Maximum recent history cases                             *",
          max_recent);
  audit(msg);
  if (mult_vs_dep)
    sprintf(msg,
            "* %5d Multiple comparisons                                     *",
            comparisons_corr);
  else
    sprintf(msg,
            "* %5d Maximum serial correlation lag                           *",
            comparisons_corr);
  audit(msg);
  if (mcpt_reps > 1) {
    sprintf(msg,
            "* %5d replications of complete Monte-Carlo Permutation Test    *",
            mcpt_reps);
    audit(msg);
  } else {
    sprintf(
        msg,
        "*       No Monte-Carlo Permutation Test                          *");
    audit(msg);
  }
  audit("*                                                                *");
  audit("******************************************************************");

  /*
     Allocate memory
  */

  crit = (double *)MALLOC(3 * npred * sizeof(double));
  original_crits = crit + npred;
  sorted_crits = original_crits + npred;

  index = (int *)MALLOC((3 * npred + 2 * n_cases) * sizeof(int));
  mcpt_solo = index + npred;
  mcpt_bestof = mcpt_solo + npred;
  shuffle_index = mcpt_bestof + npred;
  work_shuffle_index = shuffle_index + n_cases;
  breaks = (int *)MALLOC(npred * sizeof(int));

  iwork = (int *)MALLOC(max_threads * n_cases * sizeof(int));
  dwork = (double *)MALLOC(2 * max_threads * n_cases * sizeof(double));

  if (crit == NULL || index == NULL || breaks == NULL || iwork == NULL ||
      dwork == NULL) {
    ret_val = ERROR_INSUFFICIENT_MEMORY;
    goto FINISH;
  }

  /*
  --------------------------------------------------------------------------------

     Outer-most loop does MCPT replications

  --------------------------------------------------------------------------------
  */

  for (i = 0; i < n_cases; i++)
    shuffle_index[i] = i;

  if (mcpt_reps < 1)
    mcpt_reps = 1;

  for (irep = 0; irep < mcpt_reps; irep++) {

    /*
       Shuffle if in permutation run (irep>0) and not doing serial correlation
       version The serial correlation version does its own shuffling
    */

    if (irep && mult_vs_dep) { // If doing permuted runs in mult dep, shuffle
      i = n_cases;             // Number remaining to be shuffled
      while (i > 1) {          // While at least 2 left to shuffle
        j = (int)(unifrand_fast() * i);
        if (j >= i)
          j = i - 1;
        k = shuffle_index[--i];
        shuffle_index[i] = shuffle_index[j];
        shuffle_index[j] = k;
      }
    } // If in permutation run (irep > 0)

    /*
    -----------------------------------------------------------------------------------

       This next major block of code handles threading
       Compute and save criterion.  Keep track of maximum.

       NOTE ON THREADS... If the thread workload is low, overhead will keep CPU
    time low.  If debug printing, one will see that only the first thread keeps
    ending and getting replaced.  Threading behaves as expected only when each
    thread uses a lot of CPU time.

    -----------------------------------------------------------------------------------
    */

    /*
       Initialize those thread parameters which are constant for all threads.
    */

    for (ithread = 0; ithread < max_threads; ithread++) {
      break_mean_params[ithread].ncases = n_cases;
      break_mean_params[ithread].n_vars = n_vars;
      break_mean_params[ithread].min_recent = min_recent;
      break_mean_params[ithread].max_recent = max_recent;
      break_mean_params[ithread].comparisons_corr = comparisons_corr;
      break_mean_params[ithread].mult_vs_dep = mult_vs_dep;
      break_mean_params[ithread].database = database;
      break_mean_params[ithread].iwork = iwork + ithread * n_cases;
      break_mean_params[ithread].dwork = dwork + 2 * ithread * n_cases;
    } // For all threads, initializing constant stuff

    /*
       Do it
    */

    n_threads = 0; // Counts threads that are active
    for (i = 0; i < max_threads; i++)
      threads[i] = NULL;

    ivar = 0;        // Index of predictor in preds
    empty_slot = -1; // After full, will identify the thread that just completed
    for (;;) {       // Main thread loop processes all predictors

      /*
         Handle user ESCape
      */

      if (escape_key_pressed || user_pressed_escape()) {
        for (i = 0, k = 0; i < max_threads; i++) {
          if (threads[i] != NULL)
            threads[k++] = threads[i];
        }
        ret_val = WaitForMultipleObjects(k, threads, TRUE, 50000);
        sprintf(
            msg,
            "BREAK_MEAN.CPP: User abort; n_threads=%d  k=%d  Wait retval=%d",
            n_threads, k, ret_val);
        MEMTEXT(msg);
        for (i = 0; i < k; i++) {
          CloseHandle(threads[i]);
          threads[i] = NULL;
        }
        audit("ERROR: User pressed ESCape during STATIONARITY MEAN BREAK");
        ret_val = ERROR_ESCAPE;
        goto FINISH;
      }

      /*
         Start a new thread if we still have work to do
      */

      if (ivar < npred) {   // If there are still some to do
        if (empty_slot < 0) // Negative while we are initially filling the queue
          k = n_threads;
        else
          k = empty_slot;
        break_mean_params[k].ivar = ivar; // Needed for placing final result
        break_mean_params[k].varnum =
            preds[ivar]; // Needed for finding data column in database
        if (irep)
          break_mean_params[k].shuffle_index = shuffle_index;
        else
          break_mean_params[k].shuffle_index = NULL;
        break_mean_params[k].irep = irep;
        threads[k] = (HANDLE)_beginthreadex(NULL, 0, break_mean_threaded,
                                            &break_mean_params[k], 0, NULL);
        if (threads[k] == NULL) {
          for (i = 0; i < n_threads; i++) {
            if (threads[i] != NULL) {
              CloseHandle(threads[i]);
              threads[i] = NULL;
            }
          }
          audit("Internal ERROR: bad thread creation in BREAK_MEAN");
          return ERROR_INSUFFICIENT_MEMORY;
        }
#if DEBUG_THREADS
        sprintf(msg, "ivar=%3d  Thread %8d (k=%2d n_threads=%2d)", ivar,
                (int)threads[k], k, n_threads);
        MEMTEXT(msg);
#endif
        ++n_threads;
        ++ivar;
      } // if (ivar < npred)

      if (n_threads == 0) // Are we done?
        break;

      /*
         Handle full suite of threads running and more threads to add as soon as
         some are done. Wait for just one thread to finish.
      */

      if (n_threads == max_threads && ivar < npred) {
        ret_val = WaitForMultipleObjects(n_threads, threads, FALSE, 500000);
        if (ret_val == WAIT_TIMEOUT || ret_val == WAIT_FAILED || ret_val < 0 ||
            ret_val >= n_threads) {
          for (i = 0; i < n_threads; i++) {
            if (threads[i] != NULL) {
              CloseHandle(threads[i]);
              threads[i] = NULL;
            }
          }
          sprintf(msg,
                  "INTERNAL ERROR!!!  Thread wait failed (%d) in BREAK_MEAN",
                  ret_val);
          audit(msg);
          MEMTEXT(msg);
          return ERROR_INSUFFICIENT_MEMORY;
        }

#if DEBUG_THREADS
        sprintf(msg, "End Thread %8d (%2d)  ivar=%d  f=%.3lf  ibreak=%d",
                (int)threads[ret_val], ret_val, break_mean_params[ret_val].ivar,
                break_mean_params[ret_val].crit,
                break_mean_params[ret_val].ibreak);
        MEMTEXT(msg);
#endif

        crit[break_mean_params[ret_val].ivar] = break_mean_params[ret_val].crit;
        if (irep == 0)
          breaks[break_mean_params[ret_val].ivar] =
              break_mean_params[ret_val].ibreak;

        empty_slot = ret_val;
        CloseHandle(threads[empty_slot]);
        threads[empty_slot] = NULL;
        --n_threads;
      }

      /*
         Handle all work has been started and now we are just waiting for
         threads to finish
      */

      else if (ivar == npred) {
        ret_val = WaitForMultipleObjects(n_threads, threads, TRUE, 500000);
        if (ret_val == WAIT_TIMEOUT || ret_val == WAIT_FAILED || ret_val < 0 ||
            ret_val >= n_threads) {
          for (i = 0; i < n_threads; i++) {
            if (threads[i] != NULL) {
              CloseHandle(threads[i]);
              threads[i] = NULL;
            }
          }
          sprintf(
              msg,
              "INTERNAL ERROR!!!  Thread wait failed (%d) in BREAK_MEAN.CPP",
              ret_val);
          audit(msg);
          MEMTEXT(msg);
          return ERROR_INSUFFICIENT_MEMORY;
        }

        for (i = 0; i < n_threads; i++) {
#if DEBUG_THREADS
          sprintf(msg, "Final Thread %8d  ivar=%d  f=%.3lf  ibreak=%d",
                  (int)threads[i], break_mean_params[i].ivar,
                  break_mean_params[i].crit, break_mean_params[i].ibreak);
          MEMTEXT(msg);
#endif

          crit[break_mean_params[i].ivar] = break_mean_params[i].crit;
          if (irep == 0)
            breaks[break_mean_params[i].ivar] = break_mean_params[i].ibreak;

          CloseHandle(threads[i]);
        }
        break;
      }
    } // Endless loop which threads computation of criterion for all predictors

    /*
       The criterion for each predictor has been computed and saved in crit.
       Update the MCPT.
    */

    for (ivar = 0; ivar < npred; ivar++) {

      if (ivar == 0 || crit[ivar] > best_crit)
        best_crit = crit[ivar];

      if (irep == 0) { // Original, unpermuted data
        sorted_crits[ivar] = original_crits[ivar] = crit[ivar];
        index[ivar] = ivar;
        mcpt_bestof[ivar] = mcpt_solo[ivar] = 1;
      }

      else if (crit[ivar] >= original_crits[ivar])
        ++mcpt_solo[ivar];

    } // For all predictor candidates

    if (irep == 0) // Get the indices that sort the predictors' criteria
      qsortdsi(0, npred - 1, sorted_crits, index);

    else {
      for (ivar = 0; ivar < npred; ivar++) {
        if (best_crit >= original_crits[ivar]) // Valid only for largest
          ++mcpt_bestof[ivar];
      }
    }

  } // For all MCPT replications

  /*
  --------------------------------------------------------------------------------

     All main computation is finished.  Print results.

  --------------------------------------------------------------------------------
  */

  audit("");
  audit("");
  sprintf(msg, "------------------------------> Mean break test "
               "<------------------------------");
  audit(msg);
  audit("");
  audit("Z(U) refers to the maximum break across the user's specified range.");
  audit("Nrecent is the number of most recent cases this side of the break.");
  if (mcpt_reps > 1) {
    audit("Solo pval takes into account all tries across the specified range.");
    if (npred > 1)
      audit("Unbiased pval takes into account all variables tested.");
  }
  audit("");
  strcpy(msg, "       Variable        Z(U)      Nrecent");
  if (mcpt_reps > 1) {
    strcat(msg, "      Solo pval");
    if (npred > 1)
      strcat(msg, "  Unbiased pval");
  }
  audit(msg);
  audit("");

  for (i = npred - 1; i >= 0; i--) {
    k = index[i];
    sprintf(msg, "%15s %12.4lf %9d", var_names[preds[k]], original_crits[k],
            breaks[k]);
    if (mcpt_reps > 1) {
      sprintf(msg2, "   %12.4lf", (double)mcpt_solo[k] / (double)mcpt_reps);
      strcat(msg, msg2);
      if (npred > 1) {
        sprintf(msg2, " %12.4lf", (double)mcpt_bestof[k] / (double)mcpt_reps);
        strcat(msg, msg2);
      }
    }
    audit(msg);
  }

  /*
     Finished.  Clean up and exit.
  */

FINISH:

  if (crit != NULL)
    FREE(crit);
  if (index != NULL)
    FREE(index);
  if (breaks != NULL)
    FREE(breaks);
  if (iwork != NULL)
    FREE(iwork);
  if (dwork != NULL)
    FREE(dwork);
  return ret_val;
}
