/******************************************************************************/
/*                                                                            */
/*  COMP_VAR - Compute paired-market variables                                */
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
--------------------------------------------------------------------------------

   atr() - Compute historical average true range

--------------------------------------------------------------------------------
*/

double atr(int use_log, int icase, int length, double *open, double *high,
           double *low, double *close) {
  int i;
  double term, sum;

  assert(icase >= length);

  // This is just a kludge to handle length=0
  if (length == 0) {
    if (use_log)
      return log(high[icase] / low[icase]);
    else
      return high[icase] - low[icase];
  }

  sum = 0.0;
  for (i = icase - length + 1; i <= icase; i++) {
    if (use_log) {
      term = high[i] / low[i];
      if (high[i] / close[i - 1] > term)
        term = high[i] / close[i - 1];
      if (close[i - 1] / low[i] > term)
        term = close[i - 1] / low[i];
      sum += log(term);
    } else {
      term = high[i] - low[i];
      if (high[i] - close[i - 1] > term)
        term = high[i] - close[i - 1];
      if (close[i - 1] - low[i] > term)
        term = close[i - 1] - low[i];
      sum += term;
    }
  }

  return sum / length;
}

/*
--------------------------------------------------------------------------------

   variance() - Compute historical variance of prices or change

--------------------------------------------------------------------------------
*/

double variance(int use_change, int icase, int length, double *prices) {
  int i;
  double term, sum, mean;

  if (use_change)
    assert(icase >= length);
  else
    assert(icase >= length - 1);

  sum = 0.0;
  for (i = icase - length + 1; i <= icase; i++) {
    if (use_change)
      term = log(prices[i] / prices[i - 1]);
    else
      term = log(prices[i]);
    sum += term;
  }

  mean = sum / length;

  sum = 0.0;
  for (i = icase - length + 1; i <= icase; i++) {
    if (use_change)
      term = log(prices[i] / prices[i - 1]) - mean;
    else
      term = log(prices[i]) - mean;
    sum += term * term;
  }

  return sum / length;
}

/*
--------------------------------------------------------------------------------

   comp_var - Routine computes paired-market variables

--------------------------------------------------------------------------------
*/

int comp_var(int n, int var_num, double param1, double param2, double param3,
             double param4, double *open1, double *high1, double *low1,
             double *close1, double *volume1, double *open2, double *high2,
             double *low2, double *close2, double *volume2, int *n_done,
             int *first_date, int *last_date, double *output, double *work1,
             double *work2, double *work3) {
  int i, k, lookback, icase, front_bad, ret_val;
  int length, atr_length;
  int trend_length, accel_length, vol_length, max_length;
  double sum, diff, xss, denom, xmean, ymean, xdiff, ydiff, xy, coef;
  double factor, alpha, smoothed, *xptr, *yptr;
  Purify *purify_ptr;

  ret_val = 0; // Be optimistic that there is no error

  if (var_num == VAR_CORRELATION || var_num == VAR_DELTA_CORRELATION) {
    lookback = (int)(param1 + 0.5);
    if (lookback < 2)
      lookback = 2;
    front_bad = lookback - 1;
    if (front_bad > n)
      front_bad = n;

    for (i = 0; i < front_bad; i++) // Set undefined bars to neutral value
      output[i] = 0.0;

    // Compute indicator for all remaining bars
    // There is no point in taking logs because Spearman Rho is based on rank
    // order
    for (icase = front_bad; icase < n; icase++) {
      xptr = close2 + icase - lookback + 1;
      yptr = close1 + icase - lookback + 1;
      output[icase] = 50.0 * spearman(lookback, xptr, yptr, work1, work2);
    } // For all cases

    if (var_num == VAR_DELTA_CORRELATION) {
      length = (int)(param2 + 0.5);
      if (length < 1)
        length = 1;
      front_bad += length;
      if (front_bad > n)
        front_bad = n;
      for (icase = n - 1; icase >= front_bad; icase--)
        output[icase] -= output[icase - length];
      for (i = 1; i <= length; i++) // Set undefined bars to neutral value
        output[front_bad - i] = 0.0;
    } // if DELTA

  } // VAR_(DELTA)CORRELATION

  else if (var_num == VAR_DEVIATION) {
    lookback = (int)(param1 + 0.5); // Lookback window
    length = (int)(param2 + 0.5);   // Smoothing lookback
    if (lookback < 2)
      lookback = 2;
    front_bad = lookback - 1;
    if (front_bad > n)
      front_bad = n;

    for (i = 0; i < front_bad; i++) // Set undefined bars to neutral value
      output[i] = 0.0;

    // Compute indicator for all remaining bars
    for (icase = front_bad; icase < n; icase++) {
      // Get the data and cumulate their means
      xmean = ymean = 0.0;
      for (i = 0; i < lookback; i++) {
        k = icase - i;
        xmean += log(close2[k]);
        ymean += log(close1[k]);
      } // For lookback, cumulating means
      xmean /= lookback;
      ymean /= lookback;

      // Cumulate sum squares and cross product; divide to get coef
      xss = xy = 0.0;
      for (i = 0; i < lookback; i++) {
        k = icase - i;
        xdiff = log(close2[k]) - xmean;
        ydiff = log(close1[k]) - ymean;
        xss += xdiff * xdiff;
        xy += xdiff * ydiff;
      }
      if (xss > 0.0)
        coef = xy / xss; // Linear regression slope
      else
        coef = 1.0; // Price may be constant; don't divide by 0 for coef

      // For all cases in this history batch cumulate the prediction and error.
      // By predicting the deviation from the mean we avoid the need for a
      // constant. Note that we include the current case in the fit. This helps
      // reduce the impact of outliers.

      sum = 0.0;
      for (i = lookback - 1; i >= 0;
           i--) { // Counting backwards leaves last diff at current point
        k = icase - i;
        xdiff = log(close2[k]) - xmean; // Index deviation from its mean
        ydiff =
            log(close1[k]) - ymean;  // Member equity deviation from its means
        diff = ydiff - coef * xdiff; // Actual minus predicted
        sum += diff * diff;          // Cumulate for error of fit
      }
      denom = sqrt(sum / lookback); // RMS error

      if (denom > 0.0) { // Normally the fit will be imperfect
        output[icase] =
            diff / denom; // Devalue the deviation if the fit is poor
        factor = 1.0 / exp(log((double)lookback) / 6.0);
        output[icase] = 100.0 * normal_cdf(factor * output[icase]) - 50.0;
      } else // We may rarely have a perfect fit, in which case the deviation is
             // 0
        output[icase] = 0.0;
    } // For all cases

    // If we are to smooth, do it now
    if (length > 1) {
      alpha = 2.0 / (length + 1.0);
      smoothed = output[front_bad];
      for (icase = front_bad + 1; icase < n; icase++) {
        smoothed = alpha * output[icase] + (1.0 - alpha) * smoothed;
        output[icase] = smoothed;
      } // For all cases, smoothing
    }

  } // VAR_DEVIATION

  else if (var_num == VAR_PURIFY || var_num == VAR_LOG_PURIFY) {
    lookback = (int)(param1 + 0.5);     // Grand lookback window
    trend_length = (int)(param2 + 0.5); // Trend lookback
    accel_length = (int)(param3 + 0.5); // Acceleration lookback
    vol_length = (int)(param4 + 0.5);   // Volatility lookback
    max_length = trend_length;
    if (accel_length > max_length)
      max_length = accel_length;
    if (vol_length > max_length)
      max_length = vol_length;
    if (lookback < 2)
      lookback = 2;
    front_bad = lookback + max_length - 1;
    if (front_bad > n)
      front_bad = n;

    purify_ptr = new Purify(lookback, trend_length, accel_length, vol_length);
    if (purify_ptr == NULL || !purify_ptr->ok) {
      if (purify_ptr != NULL) {
        delete purify_ptr;
        purify_ptr = NULL;
      }
      front_bad = n;
    }

    for (i = 0; i < front_bad; i++) // Set undefined bars to neutral value
      output[i] = 0.0;

    if (purify_ptr != NULL) {
      k = (var_num == VAR_LOG_PURIFY);
      for (icase = front_bad; icase < n;
           icase++) { // Compute indicator for all remaining bars
        output[icase] = purify_ptr->compute(k, close1 + icase, close2 + icase);
        output[icase] = 100.0 * normal_cdf(0.5 * output[icase]) - 50.0;
      }
      delete purify_ptr;
    } // if (purify_ptr != NULL)

  } // VAR_(LOG)PURIFY

  else if (var_num == VAR_TREND_DIFF || var_num == VAR_CMMA_DIFF) {
    lookback = (int)(param1 + 0.5);
    atr_length = (int)(param2 + 0.5);

    if (var_num == VAR_TREND_DIFF) {
      front_bad = ((lookback - 1) > atr_length) ? (lookback - 1) : atr_length;
      trend(n, lookback, atr_length, open1, high1, low1, close1, work1, work2);
      trend(n, lookback, atr_length, open2, high2, low2, close2, work1, work3);
    }

    else if (var_num == VAR_CMMA_DIFF) {
      front_bad = (lookback > atr_length) ? lookback : atr_length;
      cmma(n, lookback, atr_length, open1, high1, low1, close1, work1, work2);
      cmma(n, lookback, atr_length, open2, high2, low2, close2, work1, work3);
    }

    for (icase = 0; icase < front_bad; icase++)
      output[icase] = 0.0;

    for (icase = front_bad; icase < n; icase++)
      output[icase] = work2[icase] - work3[icase];

  } // VAR_TREND_DIFF and VAR_CMMA_DIFF

  *n_done = n - front_bad;
  *first_date = front_bad;
  *last_date = n - 1;
  return ret_val;
}