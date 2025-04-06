/******************************************************************************/
/*                                                                            */
/*  PURIFY - All code for Purify class                                        */
/*                                                                            */
/******************************************************************************/

#include <assert.h>
#include <conio.h>
#include <float.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>


#include "classes.h"
#include "const.h"
#include "funcdefs.h"


#define DEBUG_PURIFY 0

/*
--------------------------------------------------------------------------------

   Constructor and destructor

--------------------------------------------------------------------------------
*/

Purify::Purify(
    int lb, // Primary lookback; number of cases in training set for each bar
    int trn_length, // Lookback for trend predictor
    int acc_length, // Lookback for acceleration predictor
    int v_length    // Lookback for volatility predictor
) {
  MEMTEXT("Purify: Purify constructor");
  ok = 1;

  lookback = lb;
  trend_length = trn_length;
  accel_length = acc_length;
  vol_length = v_length;

  npred =
      (trend_length ? 1 : 0) + (accel_length ? 1 : 0) + (vol_length ? 1 : 0);

  Legendre1 = Legendre2 = NULL;
  svd = new SingularValueDecomp(lookback, npred + 1, 1);
  if (svd == NULL || !svd->ok) {
    if (svd != NULL) {
      delete svd;
      svd = NULL;
    }
    ok = 0;
    return;
  }

  // These are small.  Allocate even if not needed.
  Legendre1 = (double *)MALLOC(lookback * sizeof(double));
  if (Legendre1 == NULL) {
    delete svd;
    ok = 0;
    return;
  }

  Legendre2 = (double *)MALLOC(lookback * sizeof(double));
  if (Legendre2 == NULL) {
    FREE(Legendre1);
    delete svd;
    ok = 0;
    return;
  }

  legendre_2(lookback, Legendre1, Legendre2);
}

Purify::~Purify() {
  MEMTEXT("Purify: Purify destructor");

  if (svd != NULL)
    delete svd;

  if (Legendre1 != NULL)
    FREE(Legendre1);

  if (Legendre2 != NULL)
    FREE(Legendre2);
}

/*
--------------------------------------------------------------------------------

   compute() - Do the work;

   The two input arrays are in chronological order and point to the current bar,
   meaning that the lookback window is prior to the pointers.

--------------------------------------------------------------------------------
*/

double Purify::compute(
    int use_log,       // Take log of predicted series?
    double *predicted, // Typically an indicator or 'special market' like VIX
    double *predictor  // Typically a market, so we always take logs here
) {
  int i, icase;
  double coefs[4], sum, diff, mse, *aptr, *bptr, *dptr;
  double *predicted_ptr, *predictor_ptr;
#if DEBUG_PURIFY
  char msg[256], msg2[256];
  sprintf_s(msg, "Predicted=%9.5lf  Predictor=%9.5lf", log(*predicted),
            log(*predictor));
  MEMTEXT("");
  MEMTEXT(msg);
#endif

  /*
     Fill in the design matrix and compute its svd.
     Then call backsub() to find the coefficients.
  */

  aptr = svd->a;
  bptr = svd->b;
  predicted_ptr = predicted; // Begin pointing at the current case
  predictor_ptr = predictor;

  for (icase = 0; icase < lookback; icase++) {

    if (trend_length) {
      sum = 0.0;
      dptr = predictor_ptr - trend_length + 1; // Start of inner window
      for (i = 0; i < trend_length; i++)
        sum += Legendre1[i] * log(dptr[i]); // Cumulate dot product
      *aptr++ = sum;                        // This is the trend predictor
    }

    if (accel_length) {
      sum = 0.0;
      dptr = predictor_ptr - accel_length + 1; // Start of inner window
      for (i = 0; i < accel_length; i++)
        sum += Legendre2[i] * log(dptr[i]); // Cumulate dot product
      *aptr++ = sum; // This is the acceleration predictor
    }

    if (vol_length) {
      sum = 0.0;
      dptr = predictor_ptr - vol_length + 1; // Start of inner window
      for (i = 0; i < vol_length - 1; i++)   // We are working with differences
        sum += fabs(log(dptr[i] / dptr[i + 1]));
      *aptr++ = sum / (vol_length - 1); // This is the volatility predictor
    }

    *aptr++ = 1.0; // Constant term
    if (use_log)
      *bptr++ = log(*predicted_ptr);
    else
      *bptr++ = *predicted_ptr;

#if DEBUG_PURIFY
    sprintf_s(msg, "%3d", icase);
    for (i = 0; i <= npred; i++) {
      sprintf_s(msg2, " %9.5lf", *(aptr - npred - 1 + i));
      strcat(msg, msg2);
    }
    sprintf_s(msg2, " : %9.5lf", *(bptr - 1));
    strcat(msg, msg2);
    MEMTEXT(msg);
#endif

    --predicted_ptr; // Move backward through grand lookback window
    --predictor_ptr;
  }

  svd->svdcmp();
  svd->backsub(1.e-7, coefs); // Computes optimal weights

#if DEBUG_PURIFY
  sprintf_s(msg, "Coefs:");
  for (i = 0; i <= npred; i++) {
    sprintf_s(msg2, " %9.5lf", coefs[i]);
    strcat(msg, msg2);
  }
  MEMTEXT(msg);
#endif

  /*
     We now have the linear regression coefficients.
     Make another pass through the data (which is still saved in svd)
     and compute the mean squared error.
  */

  aptr = svd->a;
  bptr = svd->b;
  mse = 0.0;
  for (icase = 0; icase < lookback; icase++) {
    sum = coefs[npred]; // Constant
    for (i = 0; i < npred; i++)
      sum += coefs[i] * *aptr++;
    ++aptr; // Skip the 1.0 that is for the constant
    // The prediction is now in sum.
    diff = *bptr++ - sum; // True minus predicted
    mse += diff * diff;

#if DEBUG_PURIFY
    sprintf_s(msg, "Actual=%9.5lf  Pred=%9.5lf  Diff=%9.5lf", *(bptr - 1), sum,
              diff);
    MEMTEXT(msg);
#endif
  }

  mse = sqrt(mse / lookback); // RMS error

#if DEBUG_PURIFY
  sprintf_s(msg, "MSE=%9.5lf", mse);
  MEMTEXT(msg);
#endif

  /*
     Recall that we built the design matrix starting at the current case
     and worked backwards.  Thus, when icase=0 we have the first case.
     Save its difference.
  */

  aptr = svd->a;
  bptr = svd->b;
  sum = coefs[npred]; // Constant
  for (i = 0; i < npred; i++)
    sum += coefs[i] * *aptr++;
  // The prediction is now in sum.
  diff = *bptr - sum; // True minus predicted

#if DEBUG_PURIFY
  sprintf_s(msg, "Final diff=%9.5lf  Output==%9.5lf", diff,
            diff / (mse + 1.e-6));
  MEMTEXT(msg);
#endif

  return diff / (mse + 1.e-6);
}