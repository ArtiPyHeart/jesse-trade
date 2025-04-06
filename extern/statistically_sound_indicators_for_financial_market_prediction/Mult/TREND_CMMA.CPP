/******************************************************************************/
/*                                                                            */
/*  TREND_CMMA - Compute linear trend and CMMA for multiple-market rank       */
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

/*
-------------------------------------------------------------------------------

   Trend

-------------------------------------------------------------------------------
*/

void trend(int n, int lookback, int atr_length, double *open, double *high,
           double *low, double *close, double *work, double *output) {
  int i, icase, k, front_bad;
  double sum, denom, *dptr, price, mean, dot_prod, diff, rsq;
  double yss, pred;

  /*
     Set the invalid values to zero
  */

  front_bad = ((lookback - 1) > atr_length) ? (lookback - 1) : atr_length;

  for (icase = 0; icase < front_bad; icase++)
    output[icase] = 0.0; // Set undefined values to neutral value

  /*
     Compute first-order Lengendre coefficients
  */

  sum = 0.0;
  for (i = 0; i < lookback; i++) {
    work[i] = 2.0 * i / (lookback - 1.0) - 1.0;
    sum += work[i] * work[i];
  }

  sum = sqrt(sum);
  for (i = 0; i < lookback; i++)
    work[i] /= sum;

  /*
     Compute trend for all remaining bars
  */

  for (icase = front_bad; icase < n; icase++) {
    dptr = work;

    // The regression coefficient (in dot_prod) is the dot product of the
    // log prices with the Legendre polynomial coefficients.

    dot_prod = 0.0;
    mean = 0.0; // We need this for rsq
    for (k = icase - lookback + 1; k <= icase;
         k++) { // The trend lookback window
      price = log(close[k]);
      mean += price;
      dot_prod += price * *dptr++; // Cumulate dot product
    }
    mean /= lookback;
    dptr -= lookback; // Reset coefs pointer to start for finding rsq

    // Dot_prod is regression coef (log price change per unit X change)
    // Total X change over window is 1 - (-1) = 2 (domain of Legendre
    // polynomial) So dot_prod * 2 is the fitted change over the window Denom is
    // change over window based on ATR if all changes exactly match Legendre
    // polynomial Thus, the basic indicator (prior to rsq and compression) is
    // the ratio of the achieved fitted change to the theoretical change based
    // on ATR.

    k = lookback - 1;
    if (lookback == 2)
      k = 2;
    denom = atr(1, icase, atr_length, open, high, low, close) * k;
    output[icase] = dot_prod * 2.0 /
                    (denom + 1.e-60); // Change over window / window ATR path

    // At this point, output[icase] is the ratio of the observed change to the
    // theoretical change implied by ATR that follows the tested path.
    // Compute R-square for degrading the indicator if it is a poor fit

    yss = rsq = 0.0;
    for (k = icase - lookback + 1; k <= icase;
         k++) { // The trend lookback window
      price = log(close[k]);
      diff = price - mean; // Y offset from its mean
      yss += diff * diff;  // Cumulate Y sum of squares
      pred = dot_prod *
             *dptr++; // Regression coefficient times X is predicted Y offset
      diff = diff - pred; // Y offset from mean minus predicted offset
      rsq += diff * diff; // Sum the squared error
    }
    rsq = 1.0 - rsq / (yss + 1.e-60); // Definition of R-square
    if (rsq < 0.0)                    // Should never happen
      rsq = 0.0;
    output[icase] *= rsq; // Degrade the indicator if it is a poor fit

    output[icase] =
        100.0 * normal_cdf(output[icase]) - 50.0; // Weakly compress outliers
  } // For all cases being computed
}

/*
-------------------------------------------------------------------------------

   Close minus MA

-------------------------------------------------------------------------------
*/

void cmma(int n, int lookback, int atr_length, double *open, double *high,
          double *low, double *close, double *work, double *output) {
  int icase, k, front_bad;
  double sum, denom;

  /*
     Set the invalid values to zero
  */

  front_bad = (lookback > atr_length)
                  ? lookback
                  : atr_length; // Number of undefined values at start

  for (icase = 0; icase < front_bad; icase++)
    output[icase] = 0.0; // Set undefined values to neutral value

  /*
     Compute CMMA for all remaining bars
  */

  for (icase = front_bad; icase < n; icase++) {

    sum = 0.0;
    for (k = icase - lookback; k < icase; k++)
      sum += log(close[k]);
    sum /= lookback;
    denom = atr(1, icase, atr_length, open, high, low, close);
    if (denom > 0.0) {
      denom *= sqrt(lookback + 1.0);
      output[icase] = (log(close[icase]) - sum) / denom;
      output[icase] =
          100.0 * normal_cdf(1.0 * output[icase]) -
          50.0; // Increase 1.0 for more compression, decrease for less
    } else
      output[icase] = 0.0;
  }
}