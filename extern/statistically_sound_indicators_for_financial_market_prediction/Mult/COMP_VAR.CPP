/******************************************************************************/
/*                                                                            */
/*  COMP_VAR - Compute multiple-market variables                              */
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

   first_pctile - Compute percentile of the first case in an array
      Actually, this is slightly modified so that the smallest will
      return 0.0 and the largest 100.0.

   median - Compute median
   range - Compute range
   iqr - Compute interquartile range
   clump - Clumped 60

--------------------------------------------------------------------------------
*/

double first_pctile(int n, double *x) {
  int i, count;
  double first;

  count = 0;
  first = x[0];
  for (i = 1; i < n; i++) {
    if (x[i] <= first)
      ++count;
  }
  return 100.0 * count / (n - 1.0);
}

double median(int n, double *x) {
  qsortd(0, n - 1, x);
  if (n % 2)
    return x[n / 2];
  else
    return 0.5 * (x[n / 2 - 1] + x[n / 2]);
}

double range(int n, double *x) {
  int i;
  double xmin, xmax;

  xmin = xmax = x[0];
  for (i = 1; i < n; i++) {
    if (x[i] < xmin)
      xmin = x[i];
    if (x[i] > xmax)
      xmax = x[i];
  }
  return xmax - xmin;
}

double iqr(int n, double *x) {
  qsortd(0, n - 1, x);
  return x[3 * n / 4] - x[n / 4];
}

double clump(int n, double *x) {
  int k, m;

  qsortd(0, n - 1, x); // Sort ascending

  k = (int)(0.4 * (n + 1)) - 1; // Crudely unbiased index of .4 fractile
  if (k < 0)                    // Never happens in normal situations
    k = 0;
  m = n - k - 1; // Index of 0.6 fractile

  // Basic rule
  if (x[k] > 0.0) // 0.4 fractile
    return x[k];
  else if (x[m] < 0.0) // 0.6 fractile
    return x[m];
  else
    return 0.0;
}

/*
--------------------------------------------------------------------------------

   comp_var - Routine computes multiple-market variables

--------------------------------------------------------------------------------
*/

int comp_var(int n, int n_markets, int var_num, double param1, double param2,
             double param3, double param4, double **open, double **high,
             double **low, double **close, double **volume, int *n_done,
             int *first_date, int *last_date, double *output,
             double *work1,     // MAX(n, n_markets)
             double *work2,     // MAX(n, n_markets)
             double *work3,     // MAX(n, n_markets)
             double *big_work,  // n_markets * MAX(n, n_markets)
             double *big_work2, // n_markets * n_markets
             double *big_work3, // n_markets * n_markets + 2 * n_markets
             int *iwork         // n_markets
) {
  int i, k, lookback, icase, imarket, jmarket, front_bad, ret_val;
  int atr_length, n_to_smooth, long_lookback, short_lookback;
  double sum, value, diff_i, diff_j, factor, *evals, alpha;
  double long_sum, short_sum, sumsq, variance, fraction, smoothed_numer,
      smoothed_denom;
  JANUS *janus;

  ret_val = 0; // Be optimistic that there is no error

  if (var_num == VAR_TREND_RANK || var_num == VAR_TREND_MEDIAN ||
      var_num == VAR_TREND_RANGE || var_num == VAR_TREND_IQR ||
      var_num == VAR_TREND_CLUMP || var_num == VAR_CMMA_RANK ||
      var_num == VAR_CMMA_MEDIAN || var_num == VAR_CMMA_RANGE ||
      var_num == VAR_CMMA_IQR || var_num == VAR_CMMA_CLUMP) {
    lookback = (int)(param1 + 0.5);
    atr_length = (int)(param2 + 0.5);

    // TREND basic indicator
    if (var_num == VAR_TREND_RANK || var_num == VAR_TREND_MEDIAN ||
        var_num == VAR_TREND_RANGE || var_num == VAR_TREND_IQR ||
        var_num == VAR_TREND_CLUMP) {
      front_bad = ((lookback - 1) > atr_length) ? (lookback - 1) : atr_length;
      for (imarket = 0; imarket < n_markets; imarket++) {
        trend(n, lookback, atr_length, open[imarket], high[imarket],
              low[imarket], close[imarket], work1, work2);
        for (icase = 0; icase < n; icase++)
          big_work[icase * n_markets + imarket] = work2[icase];
      }
    }

    // CMMA basic indicator
    else if (var_num == VAR_CMMA_RANK || var_num == VAR_CMMA_MEDIAN ||
             var_num == VAR_CMMA_RANGE || var_num == VAR_CMMA_IQR ||
             var_num == VAR_CMMA_CLUMP) {
      front_bad = (lookback > atr_length) ? lookback : atr_length;
      for (imarket = 0; imarket < n_markets; imarket++) {
        cmma(n, lookback, atr_length, open[imarket], high[imarket],
             low[imarket], close[imarket], work1, work2);
        for (icase = 0; icase < n; icase++)
          big_work[icase * n_markets + imarket] = work2[icase];
      }
    }

    for (icase = 0; icase < front_bad; icase++)
      output[icase] = 0.0;

    if (var_num == VAR_TREND_RANK || var_num == VAR_CMMA_RANK) {
      for (icase = front_bad; icase < n; icase++)
        output[icase] =
            first_pctile(n_markets, big_work + icase * n_markets) - 50.0;
    }

    else if (var_num == VAR_TREND_MEDIAN || var_num == VAR_CMMA_MEDIAN) {
      for (icase = front_bad; icase < n; icase++)
        output[icase] = median(n_markets, big_work + icase * n_markets);
    }

    else if (var_num == VAR_TREND_RANGE || var_num == VAR_CMMA_RANGE) {
      for (icase = front_bad; icase < n; icase++)
        output[icase] = range(n_markets, big_work + icase * n_markets);
    }

    else if (var_num == VAR_TREND_IQR || var_num == VAR_CMMA_IQR) {
      for (icase = front_bad; icase < n; icase++)
        output[icase] = iqr(n_markets, big_work + icase * n_markets);
    }

    else if (var_num == VAR_TREND_CLUMP || var_num == VAR_CMMA_CLUMP) {
      for (icase = front_bad; icase < n; icase++)
        output[icase] = clump(n_markets, big_work + icase * n_markets);
    }

  } // VAR_TREND_?, VAR_CMMA_?

  else if (var_num == VAR_MAHAL) {
    lookback = (int)(param1 + 0.5);
    n_to_smooth = (int)(param2 + 0.5);
    front_bad = lookback; // Current bar is not included in lookback window

    for (icase = 0; icase < front_bad; icase++)
      output[icase] = 0.0;

    for (icase = front_bad; icase < n; icase++) {
      // The lookback window is from icase-lookback through icase-1

      // Put mean of log market changes in work1.  There are lookback-1 changes.
      for (imarket = 0; imarket < n_markets; imarket++)
        work1[imarket] =
            log(close[imarket][icase - 1] / close[imarket][icase - lookback]) /
            (lookback - 1);

      // Compute covariance matrix; put it in big_work
      for (imarket = 0; imarket < n_markets; imarket++) {
        for (jmarket = 0; jmarket <= imarket; jmarket++)
          big_work[imarket * n_markets + jmarket] = 0.0;
      }

      for (imarket = 0; imarket < n_markets; imarket++) {
        for (jmarket = 0; jmarket <= imarket; jmarket++) {
          for (k = 1; k < lookback; k++) {
            diff_i =
                log(close[imarket][icase - k] / close[imarket][icase - k - 1]) -
                work1[imarket];
            diff_j =
                log(close[jmarket][icase - k] / close[jmarket][icase - k - 1]) -
                work1[jmarket];
            big_work[imarket * n_markets + jmarket] += diff_i * diff_j;
          }
          big_work[imarket * n_markets + jmarket] /= lookback - 1;
        }
      }

      for (imarket = 0; imarket < n_markets;
           imarket++) { // Fill in other half of symmetric matrix
        for (jmarket = 0; jmarket < imarket; jmarket++)
          big_work[jmarket * n_markets + imarket] =
              big_work[imarket * n_markets + jmarket];
      }

      // Invert the covariance, put inverse in big_work2.  If singular (rare!)
      // output zero.
      if (invert(n_markets, big_work, big_work2, &value, big_work3, iwork)) {
        output[icase] = 0.0;
        continue;
      }

      // Compute Mahalanobis distance
      sum = 0.0;
      for (imarket = 0; imarket < n_markets; imarket++) {
        diff_i = log(close[imarket][icase] / close[imarket][icase - 1]) -
                 work1[imarket];
        for (jmarket = 0; jmarket <= imarket; jmarket++) {
          diff_j = log(close[jmarket][icase] / close[jmarket][icase - 1]) -
                   work1[jmarket];
          value = diff_i * diff_j * big_work2[imarket * n_markets + jmarket];
          if (imarket == jmarket) // Diagonal
            sum += value;
          else // Inverse is symmetric
            sum += 2.0 * value;
        }
      } // Cumulating distance

      // Final transformation
      k = lookback - 1 - n_markets;
      sum *= (lookback - 1.0) * (double)k;
      sum /= n_markets * (lookback - 2.0) * lookback;
      sum = F_CDF(n_markets, k, sum);
#if 1
      if (sum > 0.99999)
        sum = 0.99999;
      if (sum < 0.5)
        sum = 0.5;
      output[icase] = log(sum / (1.0 - sum));
#else
      output[icase] = sum; // For diagnostics only
#endif
    } // For icase

    // Smooth if requested
    if (n_to_smooth > 1) {
      alpha = 2.0 / (n_to_smooth + 1.0);
      for (icase = front_bad + 1; icase < n; icase++)
        output[icase] =
            alpha * output[icase] + (1.0 - alpha) * output[icase - 1];
    }

  } // VAR_MAHAL

  else if (var_num == VAR_ABS_RATIO || var_num == VAR_ABS_SHIFT ||
           var_num == VAR_COHERENCE || var_num == VAR_DELTA_COHERENCE) {
    lookback = (int)(param1 + 0.5);

    front_bad = lookback - 1; // Current bar is included in lookback window

    for (icase = 0; icase < front_bad; icase++)
      output[icase] = 0.0;

    alpha = 2.0 / (lookback / 2.0 + 1.0); // For smoothing terms
    for (icase = front_bad; icase < n; icase++) {
      // The lookback window is from icase-lookback+1 through icase

      // Put mean of log market changes in work1.  There are lookback-1 changes.
      for (imarket = 0; imarket < n_markets; imarket++)
        work1[imarket] =
            log(close[imarket][icase] / close[imarket][icase - lookback + 1]) /
            (lookback - 1);

      // Compute covariance matrix; put it in big_work
      for (imarket = 0; imarket < n_markets; imarket++) {
        for (jmarket = 0; jmarket < imarket; jmarket++)
          big_work[imarket * n_markets + jmarket] = 0.0;
        big_work[imarket * n_markets + imarket] =
            1.e-60; // Prevent division by zero later
      }

      for (imarket = 0; imarket < n_markets; imarket++) {
        for (jmarket = 0; jmarket <= imarket; jmarket++) {
          for (k = 0; k < lookback - 1; k++) {
            diff_i =
                log(close[imarket][icase - k] / close[imarket][icase - k - 1]) -
                work1[imarket];
            diff_j =
                log(close[jmarket][icase - k] / close[jmarket][icase - k - 1]) -
                work1[jmarket];
            big_work[imarket * n_markets + jmarket] += diff_i * diff_j;
          }
          big_work[imarket * n_markets + jmarket] /= lookback - 1;
        }
      }

      // Convert covariance to correlation if (DELTA)COHERENCE
      if (var_num == VAR_COHERENCE || var_num == VAR_DELTA_COHERENCE) {
        for (imarket = 1; imarket < n_markets; imarket++) {
          for (jmarket = 0; jmarket < imarket; jmarket++)
            big_work[imarket * n_markets + jmarket] /=
                sqrt(big_work[imarket * n_markets + imarket] *
                     big_work[jmarket * n_markets + jmarket]);
        }
        for (imarket = 0; imarket < n_markets; imarket++)
          big_work[imarket * n_markets + imarket] = 1.0;
      }

      evals = big_work3;
      evec_rs(big_work, n_markets, 0, big_work2, evals, big_work3 + n_markets);

      if (var_num == VAR_ABS_RATIO || var_num == VAR_ABS_SHIFT) {
        fraction = param2;
        k = (int)(fraction * n_markets + 0.5);
        if (k < 1)
          k = 1;
        value = 0.0;
        for (i = 0; i < k; i++)
          value += evals[i];
        sum = value;
        for (i = k; i < n_markets; i++)
          sum += evals[i];
        if (icase == front_bad) {
          smoothed_numer = value;
          smoothed_denom = sum;
        } else {
          smoothed_numer = alpha * value + (1.0 - alpha) * smoothed_numer;
          smoothed_denom = alpha * sum + (1.0 - alpha) * smoothed_denom;
        }
        output[icase] = 100.0 * smoothed_numer / (smoothed_denom + 1.e-30);
      } // VAR_ABS_RATIO/SHIFT

      else if (var_num == VAR_COHERENCE || var_num == VAR_DELTA_COHERENCE) {
        factor = 0.5 * (n_markets - 1);
        value = sum = 0.0;
        for (i = 0; i < n_markets; i++) {
          sum += evals[i];
          value += (factor - i) * evals[i] / factor;
        }
        output[icase] = 200.0 * (value / sum - 0.5);
      }
    } // For icase

    // All cases are computed.  Compute SHIFT if requested.
    if (var_num == VAR_ABS_SHIFT) {
      long_lookback = (int)(param3 + 0.5);
      short_lookback = (int)(param4 + 0.5);
      if (long_lookback < short_lookback + 1)
        long_lookback = short_lookback + 1;
      for (icase = n - 1; icase >= front_bad + long_lookback - 1;
           icase--) { // Work backwards through bars
        short_sum = sumsq = 0.0;
        for (k = 0; k < short_lookback; k++) {
          value = output[icase - k];
          short_sum += value;
          sumsq += value * value;
        }
        long_sum = short_sum;
        for (; k < long_lookback; k++) {
          value = output[icase - k];
          long_sum += value;
          sumsq += value * value;
        }
        short_sum /= short_lookback;
        long_sum /= long_lookback;
        sumsq /= long_lookback;
        variance = sumsq - long_sum * long_sum; // Less accurate but acceptable
        if (variance <= 0.0) // Only happens very, very rarely
          output[icase] = 0.0;
        else
          output[icase] = (short_sum - long_sum) / sqrt(variance);
      } // For all cases, doing SHIFT

      front_bad += long_lookback - 1;
      while (icase >= 0)
        output[icase--] = 0.0;
    } // Computing standardized shift ABS SHIFT

    // All cases are computed.  Compute DELTA if requested.
    else if (var_num == VAR_DELTA_COHERENCE) {
      long_lookback = (int)(param2 + 0.5);
      for (icase = n - 1; icase >= front_bad + long_lookback;
           icase--) // Work backwards through bars
        output[icase] -= output[icase - long_lookback];
      front_bad += long_lookback;
      while (icase >= 0)
        output[icase--] = 0.0;
    } // Computing DELTA COHERENCE
  } // VAR_ABS_? and (DELTA)COHERENCE

  else if (var_num == VAR_JANUS_INDEX_MARKET || VAR_JANUS_INDEX_DOM ||
           var_num == VAR_JANUS_RAW_RS || var_num == VAR_JANUS_FRACTILE_RS ||
           var_num == VAR_JANUS_DELTA_FRACTILE_RS ||
           var_num == VAR_JANUS_DELTA_FRACTILE_RM || var_num == VAR_JANUS_RSS ||
           var_num == VAR_JANUS_DELTA_RSS || var_num == VAR_JANUS_DOM ||
           var_num == VAR_JANUS_DOE || var_num == VAR_JANUS_RAW_RM ||
           var_num == VAR_JANUS_FRACTILE_RM ||
           var_num == VAR_JANUS_RS_LEADER_EQUITY ||
           var_num == VAR_JANUS_RS_LAGGARD_EQUITY ||
           var_num == VAR_JANUS_RS_LEADER_ADVANTAGE ||
           var_num == VAR_JANUS_RS_LAGGARD_ADVANTAGE ||
           var_num == VAR_JANUS_RM_LEADER_EQUITY ||
           var_num == VAR_JANUS_RM_LAGGARD_EQUITY ||
           var_num == VAR_JANUS_RM_LEADER_ADVANTAGE ||
           var_num == VAR_JANUS_RM_LAGGARD_ADVANTAGE ||
           var_num == VAR_JANUS_RS_PS || var_num == VAR_JANUS_RM_PS ||
           var_num == VAR_JANUS_CMA_OOS || var_num == VAR_JANUS_OOS_AVG ||
           var_num == VAR_JANUS_LEADER_CMA_OOS) {
    lookback = (int)(param1 + 0.5);
    front_bad =
        lookback; // Current bar included in lookback, but must difference

    janus = new JANUS(n, n_markets, lookback, 0.1, 20, 60);
    if (janus == NULL || !janus->ok) {
      if (janus != NULL) {
        delete janus;
        janus = NULL;
      }
      for (i = 0; i < n; i++)
        output[i] = 0.0;
      front_bad = n;
      printf("\n\nERROR... Insufficient memory computing JANUS");
      ret_val = ERROR_INSUFFICIENT_MEMORY;
    }
    janus->prepare(close);
    janus->compute_rs(0);
    janus->compute_rs(1);
    janus->compute_rss();
    janus->compute_dom_doe();
    janus->compute_rm(0);
    janus->compute_rm(1);
    janus->compute_rs_ps();
    janus->compute_rm_ps();
    janus->compute_CMA();
    if (var_num == VAR_JANUS_INDEX_MARKET)
      janus->get_market_index(output);
    else if (var_num == VAR_JANUS_INDEX_DOM)
      janus->get_dom_index(output);
    else if (var_num == VAR_JANUS_RAW_RS) {
      k = (int)(param2 + 0.5);
      janus->get_rs(output, k);
    } else if (var_num == VAR_JANUS_FRACTILE_RS) {
      k = (int)(param2 + 0.5);
      janus->get_rs_fractile(output, k);
    } else if (var_num == VAR_JANUS_DELTA_FRACTILE_RS) {
      k = (int)(param2 + 0.5);
      janus->get_rs_fractile(output, k);
      k = (int)(param3 + 0.5);
      for (icase = n - 1; icase >= front_bad + k;
           icase--) // Work backwards through bars
        output[icase] -= output[icase - k];
      front_bad += k;
      while (icase >= 0)
        output[icase--] = 0.0;
    } else if (var_num == VAR_JANUS_RSS) {
      janus->get_rss(output);
      n_to_smooth = (int)(param2 + 0.5);
      if (n_to_smooth > 1) {
        alpha = 2.0 / (n_to_smooth + 1.0);
        for (icase = front_bad + 1; icase < n; icase++)
          output[icase] =
              alpha * output[icase] + (1.0 - alpha) * output[icase - 1];
      }
    } else if (var_num == VAR_JANUS_DELTA_RSS) {
      janus->get_rss_change(output);
      n_to_smooth = (int)(param2 + 0.5);
      if (n_to_smooth > 1) {
        alpha = 2.0 / (n_to_smooth + 1.0);
        for (icase = front_bad + 1; icase < n; icase++)
          output[icase] =
              alpha * output[icase] + (1.0 - alpha) * output[icase - 1];
      }
    } else if (var_num == VAR_JANUS_DOM) {
      k = (int)(param2 + 0.5);
      janus->get_dom(output, k);
    } else if (var_num == VAR_JANUS_DOE) {
      k = (int)(param2 + 0.5);
      janus->get_doe(output, k);
    } else if (var_num == VAR_JANUS_RAW_RM) {
      k = (int)(param2 + 0.5);
      janus->get_rm(output, k);
    } else if (var_num == VAR_JANUS_FRACTILE_RM) {
      k = (int)(param2 + 0.5);
      janus->get_rm_fractile(output, k);
    } else if (var_num == VAR_JANUS_DELTA_FRACTILE_RM) {
      k = (int)(param2 + 0.5);
      janus->get_rm_fractile(output, k);
      k = (int)(param3 + 0.5);
      for (icase = n - 1; icase >= front_bad + k;
           icase--) // Work backwards through bars
        output[icase] -= output[icase - k];
      front_bad += k;
      while (icase >= 0)
        output[icase--] = 0.0;
    } else if (var_num == VAR_JANUS_RS_LEADER_EQUITY)
      janus->get_rs_leader_equity(output);
    else if (var_num == VAR_JANUS_RS_LAGGARD_EQUITY)
      janus->get_rs_laggard_equity(output);
    else if (var_num == VAR_JANUS_RS_LEADER_ADVANTAGE)
      janus->get_rs_leader_advantage(output);
    else if (var_num == VAR_JANUS_RS_LAGGARD_ADVANTAGE)
      janus->get_rs_laggard_advantage(output);
    else if (var_num == VAR_JANUS_RS_PS)
      janus->get_rs_ps(output);
    else if (var_num == VAR_JANUS_RM_LEADER_EQUITY)
      janus->get_rm_leader_equity(output);
    else if (var_num == VAR_JANUS_RM_LAGGARD_EQUITY)
      janus->get_rm_laggard_equity(output);
    else if (var_num == VAR_JANUS_RM_LEADER_ADVANTAGE)
      janus->get_rm_leader_advantage(output);
    else if (var_num == VAR_JANUS_RM_LAGGARD_ADVANTAGE)
      janus->get_rm_laggard_advantage(output);
    else if (var_num == VAR_JANUS_RM_PS)
      janus->get_rm_ps(output);
    else if (var_num == VAR_JANUS_OOS_AVG)
      janus->get_oos_avg(output);
    else if (var_num == VAR_JANUS_CMA_OOS)
      janus->get_CMA_OOS(output);
    else if (var_num == VAR_JANUS_LEADER_CMA_OOS)
      janus->get_leader_CMA_OOS(output);

    for (i = 0; i < front_bad; i++)
      output[i] = 0.0;

    if (janus != NULL)
      delete janus;
  } // VAR_JANUS_?

  *n_done = n - front_bad;
  *first_date = front_bad;
  *last_date = n - 1;
  return ret_val;
}