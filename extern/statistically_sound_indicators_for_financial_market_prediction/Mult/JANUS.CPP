/******************************************************************************/
/*                                                                            */
/*  JANUS - Implement JANUS class for Gary Anderson's JANUS indicator family  */
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


#define DEBUG_JANUS 0

/*
--------------------------------------------------------------------------------

   Constructor and destructor

--------------------------------------------------------------------------------
*/

JANUS::JANUS(int p_nbars,          // Number of bars in market history
             int p_n_markets,      // Number of markets
             int p_lookback,       // Number of bars in lookback window
             double p_spread_tail, // Fraction (typically 0.1) of markets in
                                   // each tail for spread
             int p_min_CMA,        // Minimum lookback for CMA
             int p_max_CMA         // And maximum
) {
  int k;
  MEMTEXT("JANUS.CPP: JANUS constructor beginning");

  ok = 1; // Start out optimistic on memory allocation

  nbars = p_nbars;
  n_returns = nbars - 1;
  n_markets = p_n_markets;
  lookback = p_lookback;
  spread_tail = p_spread_tail;
  min_CMA = p_min_CMA;
  max_CMA = p_max_CMA;

  /*
     Allocate memory
  */

  k = (lookback > n_markets) ? lookback : n_markets;
  index = (double *)MALLOC(lookback * sizeof(double));
  sorted = (double *)MALLOC(k * sizeof(double));
  iwork = (int *)MALLOC(n_markets * sizeof(int));
  returns = (double *)MALLOC(n_returns * n_markets * sizeof(double));
  mkt_index_returns = (double *)MALLOC(n_returns * sizeof(double));
  dom_index_returns = (double *)MALLOC(n_returns * sizeof(double));
  CMA_alpha = (double *)MALLOC((max_CMA - min_CMA + 1) * sizeof(double));
  CMA_smoothed = (double *)MALLOC((max_CMA - min_CMA + 1) * sizeof(double));
  CMA_equity = (double *)MALLOC((max_CMA - min_CMA + 1) * sizeof(double));

  rs = (double *)MALLOC(n_returns * n_markets * sizeof(double));
  rs_fractile = (double *)MALLOC(n_returns * n_markets * sizeof(double));
  rs_lagged = (double *)MALLOC(n_returns * n_markets * sizeof(double));
  rs_leader = (double *)MALLOC(n_returns * sizeof(double));
  rs_laggard = (double *)MALLOC(n_returns * sizeof(double));
  oos_avg = (double *)MALLOC(n_returns * sizeof(double));
  rm_leader = (double *)MALLOC(n_returns * sizeof(double));
  rm_laggard = (double *)MALLOC(n_returns * sizeof(double));
  rss = (double *)MALLOC(n_returns * sizeof(double));
  rss_change = (double *)MALLOC(n_returns * sizeof(double));
  dom = (double *)MALLOC(n_returns * n_markets * sizeof(double));
  doe = (double *)MALLOC(n_returns * n_markets * sizeof(double));
  dom_index = (double *)MALLOC(n_returns * sizeof(double));
  doe_index = (double *)MALLOC(n_returns * sizeof(double));
  dom_sum = (double *)MALLOC(n_markets * sizeof(double));
  doe_sum = (double *)MALLOC(n_markets * sizeof(double));
  rm = (double *)MALLOC(n_returns * n_markets * sizeof(double));
  rm_fractile = (double *)MALLOC(n_returns * n_markets * sizeof(double));
  rm_lagged = (double *)MALLOC(n_returns * n_markets * sizeof(double));
  CMA_OOS = (double *)MALLOC(n_returns * sizeof(double));
  CMA_leader_OOS = (double *)MALLOC(n_returns * sizeof(double));

  if (index == NULL || sorted == NULL || iwork == NULL || returns == NULL ||
      mkt_index_returns == NULL || dom_index_returns == NULL ||
      CMA_alpha == NULL || CMA_smoothed == NULL || CMA_equity == NULL ||
      rs == NULL || rs_fractile == NULL || rs_lagged == NULL ||
      rs_leader == NULL || rs_laggard == NULL || oos_avg == NULL ||
      rm_leader == NULL || rm_laggard == NULL || rss == NULL ||
      rss_change == NULL || dom == NULL || doe == NULL || dom_index == NULL ||
      doe_index == NULL || dom_sum == NULL || doe_sum == NULL || rm == NULL ||
      rm_fractile == NULL || rm_lagged == NULL || CMA_OOS == NULL ||
      CMA_leader_OOS == NULL) {

    if (index != NULL) {
      FREE(index);
      index = NULL;
    }
    if (sorted != NULL) {
      FREE(sorted);
      sorted = NULL;
    }
    if (iwork != NULL) {
      FREE(iwork);
      iwork = NULL;
    }
    if (returns != NULL) {
      FREE(returns);
      returns = NULL;
    }
    if (mkt_index_returns != NULL) {
      FREE(mkt_index_returns);
      mkt_index_returns = NULL;
    }
    if (dom_index_returns != NULL) {
      FREE(dom_index_returns);
      dom_index_returns = NULL;
    }
    if (CMA_alpha != NULL) {
      FREE(CMA_alpha);
      CMA_alpha = NULL;
    }
    if (CMA_smoothed != NULL) {
      FREE(CMA_smoothed);
      CMA_smoothed = NULL;
    }
    if (CMA_equity != NULL) {
      FREE(CMA_equity);
      CMA_equity = NULL;
    }
    if (rs != NULL) {
      FREE(rs);
      rs = NULL;
    }
    if (rs_fractile != NULL) {
      FREE(rs_fractile);
      rs_fractile = NULL;
    }
    if (rs_lagged != NULL) {
      FREE(rs_lagged);
      rs_lagged = NULL;
    }
    if (rs_leader != NULL) {
      FREE(rs_leader);
      rs_leader = NULL;
    }
    if (rs_laggard != NULL) {
      FREE(rs_laggard);
      rs_laggard = NULL;
    }
    if (oos_avg != NULL) {
      FREE(oos_avg);
      oos_avg = NULL;
    }
    if (rm_leader != NULL) {
      FREE(rm_leader);
      rm_leader = NULL;
    }
    if (rm_laggard != NULL) {
      FREE(rm_laggard);
      rm_laggard = NULL;
    }
    if (rss != NULL) {
      FREE(rss);
      rss = NULL;
    }
    if (rss_change != NULL) {
      FREE(rss_change);
      rss_change = NULL;
    }
    if (dom != NULL) {
      FREE(dom);
      dom = NULL;
    }
    if (doe != NULL) {
      FREE(doe);
      doe = NULL;
    }
    if (dom_index != NULL) {
      FREE(dom_index);
      dom_index = NULL;
    }
    if (doe_index != NULL) {
      FREE(doe_index);
      doe_index = NULL;
    }
    if (rm != NULL) {
      FREE(rm);
      rm = NULL;
    }
    if (rm_fractile != NULL) {
      FREE(rm_fractile);
      rm_fractile = NULL;
    }
    if (rm_lagged != NULL) {
      FREE(rm_lagged);
      rm_lagged = NULL;
    }
    if (dom_sum != NULL) {
      FREE(dom_sum);
      dom_sum = NULL;
    }
    if (doe_sum != NULL) {
      FREE(doe_sum);
      doe_sum = NULL;
    }
    if (CMA_OOS != NULL) {
      FREE(CMA_OOS);
      CMA_OOS = NULL;
    }
    if (CMA_leader_OOS != NULL) {
      FREE(CMA_leader_OOS);
      CMA_leader_OOS = NULL;
    }
    ok = 0;
    return;
  }

  MEMTEXT("JANUS.CPP: JANUS constructor ended");
}

JANUS::~JANUS() {
  MEMTEXT("JANUS: JANUS destructor");

  if (index != NULL)
    FREE(index);
  if (sorted != NULL)
    FREE(sorted);
  if (iwork != NULL)
    FREE(iwork);
  if (returns != NULL)
    FREE(returns);
  if (mkt_index_returns != NULL)
    FREE(mkt_index_returns);
  if (dom_index_returns != NULL)
    FREE(dom_index_returns);
  if (rs != NULL)
    FREE(rs);
  if (rs_fractile != NULL)
    FREE(rs_fractile);
  if (CMA_alpha != NULL)
    FREE(CMA_alpha);
  if (CMA_smoothed != NULL)
    FREE(CMA_smoothed);
  if (CMA_equity != NULL)
    FREE(CMA_equity);
  if (rs_lagged != NULL)
    FREE(rs_lagged);
  if (rs_leader != NULL)
    FREE(rs_leader);
  if (rs_laggard != NULL)
    FREE(rs_laggard);
  if (oos_avg != NULL)
    FREE(oos_avg);
  if (rm_leader != NULL)
    FREE(rm_leader);
  if (rm_laggard != NULL)
    FREE(rm_laggard);
  if (rss != NULL)
    FREE(rss);
  if (rss_change != NULL)
    FREE(rss_change);
  if (dom != NULL)
    FREE(dom);
  if (doe != NULL)
    FREE(doe);
  if (dom_index != NULL)
    FREE(dom_index);
  if (doe_index != NULL)
    FREE(doe_index);
  if (rm != NULL)
    FREE(rm);
  if (rm_fractile != NULL)
    FREE(rm_fractile);
  if (rm_lagged != NULL)
    FREE(rm_lagged);
  if (dom_sum != NULL)
    FREE(dom_sum);
  if (doe_sum != NULL)
    FREE(doe_sum);
  if (CMA_OOS != NULL)
    FREE(CMA_OOS);
  if (CMA_leader_OOS != NULL)
    FREE(CMA_leader_OOS);
}

/*
----------------------------------------------------------------------------------

   prepare() - Gets market histories and compute returns and index returns

   This must be the first routine called to process a price history.
   It does not compute any JANUS variables, but it prepares for their
computation.

----------------------------------------------------------------------------------
*/

void JANUS::prepare(
    double **prices // prices[mkt] is pointer to array of nbars prices for mkt
) {
  int ibar, imarket;
  double *pptr, *rptr;

  /*
     Compute n_markets by n_returns matrix of returns from prices
     Anderson uses percent return but I prefer difference of logs.
     Actually, both methods give practically identical results,
     but I'm kind of fanatic about my method for ALL indicators.
  */

  rptr = returns;
  for (imarket = 0; imarket < n_markets; imarket++) {
    pptr = prices[imarket]; // Point to prices for this market
    for (ibar = 1; ibar < nbars; ibar++)
      *rptr++ = log(pptr[ibar] / pptr[ibar - 1]);
  }

  /*
     For each bar, compute the median across all markets of the returns.
     This serves as our baseline, Anderson's 'index'
     By precomputing it now we don't have to repeat this expensive operation
     every time the lookback window advances.  We just fetch for the window.
  */

  for (ibar = 0; ibar < n_returns; ibar++) {
    for (imarket = 0; imarket < n_markets; imarket++)
      sorted[imarket] = returns[imarket * n_returns + ibar];
    qsortd(0, n_markets - 1, sorted);
    if (n_markets % 2)
      mkt_index_returns[ibar] = sorted[n_markets / 2];
    else
      mkt_index_returns[ibar] =
          0.5 * (sorted[n_markets / 2 - 1] + sorted[n_markets / 2]);
  }
}

/*
----------------------------------------------------------------------------------

   compute_rs() - Compute relative strength and its rank transform

   This must be called immediately after prepare(), before other computation.

   For ordinary RS it would be called with lag=0, and results would be stored
   in rs_raw and rs_fractile.

   But when used for performance spread it would be called with lag=1,
   and results would be stored in rs_lagged.

   Currently there is no good reason for ever letting lag exceed 1,
   but preparing for this possibility is easy enough.

----------------------------------------------------------------------------------
*/

void JANUS::compute_rs(int lag) {
  int i, imarket, ibar;
  double *rptr, this_rs;
  double median, index_offensive, index_defensive, market_offensive,
      market_defensive;

  rs_lookahead = lag; // Save this for compute_rs_ps()

  for (ibar = lookback - 1; ibar < n_returns;
       ibar++) { // Main loop processes every bar

    // Fetch the precomputed index for this window; in reverse chronological
    // order
    for (i = 0; i < lookback; i++)
      index[i] = mkt_index_returns[ibar - i]; // This was computed in prepare()

    /*
       Compute the median index in the lookback window (with any lag bars
       ignored)
    */

    for (i = lag; i < lookback; i++)
      sorted[i - lag] = index[i];

    qsortd(0, lookback - lag - 1, sorted);

    if ((lookback - lag) % 2)
      median = sorted[(lookback - lag) / 2];
    else
      median = 0.5 * (sorted[(lookback - lag) / 2 - 1] +
                      sorted[(lookback - lag) / 2]);

    /*
       Compute total offensive and defensive returns for the index
    */

    index_offensive = 1.e-30; // Will soon divide by these so must not be 0
    index_defensive = -1.e-30;

    for (i = lag; i < lookback;
         i++) { // Recall that index is reverse chronological
      if (index[i] >= median)
        index_offensive += index[i] - median;
      else
        index_defensive += index[i] - median;
    }

    assert(index_offensive > 0.0);
    assert(index_defensive < 0.0);

    /*
       Compute offensive and defensive score for each market
       Recall that index is reverse chronological while markets are
       chronological This is an n_returns by n_markets matrix. (The returns
       matrix is n_markets by n_returns because that makes for slightly more
       efficient computation with it, while this order makes slightly more
       efficient computation later.)
    */

    for (imarket = 0; imarket < n_markets; imarket++) {
      rptr = returns + imarket * n_returns; // Point to returns for this market
      market_offensive = market_defensive = 0.0;
      for (i = lag; i < lookback; i++) {
        if (index[i] >= median)
          market_offensive += rptr[ibar - i] - median;
        else
          market_defensive += rptr[ibar - i] - median;
      }
      this_rs = 70.710678 * // This is 100 / sqrt(2)
                (market_offensive / index_offensive -
                 market_defensive / index_defensive);
      if (this_rs >
          200.0) // 200 is arbitrary, but some reasonable limit is needed
        this_rs = 200.0;
      if (this_rs < -200.0)
        this_rs = -200.0;
      if (lag == 0)
        rs[ibar * n_markets + imarket] = this_rs;
      else
        rs_lagged[ibar * n_markets + imarket] = this_rs;

      // Next two lines get ready for computing fractiles
      sorted[imarket] = this_rs;
      iwork[imarket] = imarket;
    } // For all markets, computing relative strength

    /*
       Compute fractiles of relative strength
    */

    qsortdsi(0, n_markets - 1, sorted, iwork); // Sort ascending, moving iwork
    for (imarket = 0; imarket < n_markets; imarket++) {
      if (lag == 0)
        rs_fractile[ibar * n_markets + iwork[imarket]] =
            (double)imarket / (n_markets - 1.0);
      // If you want to save lagged fractile somewhere else, do it here.
      // If you never want to save lagged fractiles you can easily revise this
      // code to do the computation only if lag=0.
    }

  } // For all bars
}

/*
----------------------------------------------------------------------------------

   compute_rss() - Compute relative strength spread

   This must be called immediately after compute_rs()

----------------------------------------------------------------------------------
*/

void JANUS::compute_rss() {
  int ibar, imarket, k, n;
  double this_width;

  for (ibar = lookback - 1; ibar < n_returns;
       ibar++) { // Main loop processes every bar
    for (imarket = 0; imarket < n_markets; imarket++)
      sorted[imarket] = rs[ibar * n_markets + imarket];
    qsortd(0, n_markets - 1, sorted);
    k = (int)(spread_tail * (n_markets + 1)) - 1;
    if (k < 0) // We will sum k+1 terms in the loop below
      k = 0;

    this_width = 0.0;
    n = k + 1; // This many summed
    while (k >= 0) {
      this_width += sorted[n_markets - 1 - k] - sorted[k];
      --k;
    }
    this_width /= n; // Make it an average
    rss[ibar] = this_width;
    if (ibar == lookback - 1)
      rss_change[ibar] = 0.0;
    else
      rss_change[ibar] = rss[ibar] - rss[ibar - 1];
  }
}

/*
----------------------------------------------------------------------------------

   compute_dom_doe() - Compute DOM and DOE

   This must be called after compute_rss()
   Note that dom and doe are undefined prior to lookback-1.
   In fact, because they depend on rss_change, and that is defined as 0
   until lookback, this really isn't knowd until lookback.

----------------------------------------------------------------------------------
*/

void JANUS::compute_dom_doe() {
  int ibar, imarket;
  double dom_index_sum, doe_index_sum;

  dom_index_sum = doe_index_sum = 0.0;
  for (imarket = 0; imarket < n_markets; imarket++)
    dom_sum[imarket] = doe_sum[imarket] = 0.0;

  for (ibar = lookback - 1; ibar < n_returns;
       ibar++) {                  // Main loop processes every bar
    if (rss_change[ibar] > 0.0) { // If the width is expanding
      dom_index_sum += mkt_index_returns[ibar];
      for (imarket = 0; imarket < n_markets; imarket++) {
        dom_sum[imarket] += returns[imarket * n_returns + ibar];
        dom[ibar * n_markets + imarket] = dom_sum[imarket];
        doe[ibar * n_markets + imarket] = doe_sum[imarket];
      }
    } // If width is expanding

    else if (rss_change[ibar] < 0.0) { // If the width is contracting
      doe_index_sum += mkt_index_returns[ibar];
      for (imarket = 0; imarket < n_markets; imarket++) {
        doe_sum[imarket] += returns[imarket * n_returns + ibar];
        dom[ibar * n_markets + imarket] = dom_sum[imarket];
        doe[ibar * n_markets + imarket] = doe_sum[imarket];
      }
    } // If width is contracting

    else { // Uncommon case that the width remained the same
      for (imarket = 0; imarket < n_markets; imarket++) {
        dom[ibar * n_markets + imarket] = dom_sum[imarket];
        doe[ibar * n_markets + imarket] = doe_sum[imarket];
      }
    }

    dom_index[ibar] = dom_index_sum;
    doe_index[ibar] = doe_index_sum;
  } // For ibar
}

/*
----------------------------------------------------------------------------------

   compute_rm() - Compute RM

   This must be called after compute_DOM_DOE()

----------------------------------------------------------------------------------
*/

void JANUS::compute_rm(int lag) {
  int i, ibar, imarket;
  double this_rs, ret;
  double median, index_offensive, index_defensive, market_offensive,
      market_defensive;

  rm_lookahead = lag; // Save this for compute_rm_ps()

  /*
     For each bar, compute the median across all markets of the DOM change.
     This serves as our baseline, Anderson's 'index'
     By precomputing it now we don't have to repeat this expensive operation
     every time the lookback window advances.  We just fetch for the window.
     Valid DOM exists only starting with lookback-1, so the first valid DOM
     return can be computed at lookback.  Prior to that we use market returns.
     We need it back to the beginning so we can compute DOM RS from lookback-1.
     If we did not do this we would have to start RM out further for new
     lookback. The implication is that the RMs do not start becoming more and
     more valid until lookback and need lookback-1 more bars to become fully
     valid.
  */

  for (ibar = 0; ibar < n_returns; ibar++) {
    if (ibar < lookback) {
      for (imarket = 0; imarket < n_markets; imarket++)
        sorted[imarket] = returns[imarket * n_returns + ibar];
    } else {
      for (imarket = 0; imarket < n_markets; imarket++)
        sorted[imarket] = dom[ibar * n_markets + imarket] -
                          dom[(ibar - 1) * n_markets + imarket];
    }
    qsortd(0, n_markets - 1, sorted);
    if (n_markets % 2)
      dom_index_returns[ibar] = sorted[n_markets / 2];
    else
      dom_index_returns[ibar] =
          0.5 * (sorted[n_markets / 2 - 1] + sorted[n_markets / 2]);
  }

  /*
     The precomputation of the median DOM return (the DOM index) is done.
     We now start the main loop.
  */

  for (ibar = lookback - 1; ibar < n_returns;
       ibar++) { // Main loop processes every bar

    // Fetch the precomputed index for this window; in reverse chronological
    // order
    for (i = 0; i < lookback; i++)
      index[i] = dom_index_returns[ibar - i]; // This was computed just above

    /*
       Compute the median index in the lookback window (with any lag bars
       ignored)
    */

    for (i = lag; i < lookback; i++)
      sorted[i - lag] = index[i];

    qsortd(0, lookback - lag - 1, sorted);

    if ((lookback - lag) % 2)
      median = sorted[(lookback - lag) / 2];
    else
      median = 0.5 * (sorted[(lookback - lag) / 2 - 1] +
                      sorted[(lookback - lag) / 2]);

    /*
       Compute total offensive and defensive returns for the index
    */

    index_offensive = 1.e-30; // Will soon divide by these so must not be 0
    index_defensive = -1.e-30;

    for (i = lag; i < lookback;
         i++) { // Recall that index is reverse chronological
      if (index[i] >= median)
        index_offensive += index[i] - median;
      else
        index_defensive += index[i] - median;
    }

    assert(index_offensive > 0.0);
    assert(index_defensive < 0.0);

    /*
       Compute offensive and defensive score for each market
       Recall that index is reverse chronological while markets are
       chronological
    */

    for (imarket = 0; imarket < n_markets; imarket++) {
      market_offensive = market_defensive = 0.0;
      for (i = lag; i < lookback; i++) {
        if (ibar - i < lookback) // DOM becomes valid at lookback-1
          ret = returns[imarket * n_returns + ibar - i];
        else
          ret = dom[(ibar - i) * n_markets + imarket] -
                dom[(ibar - i - 1) * n_markets + imarket];
        if (index[i] >= median)
          market_offensive += ret - median;
        else
          market_defensive += ret - median;
      }
      this_rs = 70.710678 * // This is 100 / sqrt(2)
                (market_offensive / index_offensive -
                 market_defensive / index_defensive);
      if (this_rs >
          300.0) // 300 is arbitrary, but some reasonable limit is needed
        this_rs = 300.0;
      if (this_rs < -300.0)
        this_rs = -300.0;
      if (lag == 0)
        rm[ibar * n_markets + imarket] = this_rs;
      else
        rm_lagged[ibar * n_markets + imarket] = this_rs;

      // Next two lines get ready for computing fractiles
      sorted[imarket] = this_rs;
      iwork[imarket] = imarket;
    } // For all markets, computing relative strength

    /*
       Compute fractiles of relative strength of DOM (RM)
    */

    qsortdsi(0, n_markets - 1, sorted, iwork); // Sort ascending, moving iwork
    for (imarket = 0; imarket < n_markets; imarket++) {
      if (lag == 0)
        rm_fractile[ibar * n_markets + iwork[imarket]] =
            (double)imarket / (n_markets - 1.0);
    }

  } // For all bars
}

/*
----------------------------------------------------------------------------------

   compute_rs_ps() - Compute rs_leader and rs_laggard (for performance spread)
                     This also compute oos_avg.

   This must be called after compute_rs() with a positive lag

----------------------------------------------------------------------------------
*/

void JANUS::compute_rs_ps() {
  int i, k, n, ibar, imarket, isub;

  for (ibar = lookback - 1; ibar < n_returns;
       ibar++) { // Main loop processes every bar
    for (imarket = 0; imarket < n_markets; imarket++) {
      sorted[imarket] = rs_lagged[ibar * n_markets + imarket];
      iwork[imarket] = imarket;
    }

    qsortdsi(0, n_markets - 1, sorted, iwork);

    k = (int)(spread_tail * (n_markets + 1)) - 1;
    if (k < 0)
      k = 0;
    n = k + 1;

    // This loop sums the leader and laggard for the relative strength extreme
    // markets Sums in the top k+1 and in the bottom k+1.

    rs_leader[ibar] = rs_laggard[ibar] = 0.0;
    while (k >= 0) {
      isub = iwork[k]; // Low relative strength market
      for (i = 0; i < rs_lookahead; i++)
        rs_laggard[ibar] +=
            returns[isub * n_returns + ibar -
                    i]; // Performance on dates past calculation set
      isub = iwork[n_markets - 1 - k]; // High relative strength market
      for (i = 0; i < rs_lookahead; i++)
        rs_leader[ibar] +=
            returns[isub * n_returns + ibar -
                    i]; // Performance on dates past calculation set
      --k;
    }

    rs_leader[ibar] /=
        n * rs_lookahead; // Make return a mean per market and bar
    rs_laggard[ibar] /= n * rs_lookahead;

    oos_avg[ibar] = 0.0;
    for (i = 0; i < n_markets; i++)
      oos_avg[ibar] += returns[i * n_returns + ibar];
    oos_avg[ibar] /= n_markets;
  } // For ibar
}

/*
----------------------------------------------------------------------------------

   compute_rm_ps() - Compute rm_leader and rm_laggard (for performance spread)

   This must be called after compute_rm() with a positive lag

----------------------------------------------------------------------------------
*/

void JANUS::compute_rm_ps() {
  int i, k, n, ibar, imarket, isub;

  for (ibar = lookback - 1; ibar < n_returns;
       ibar++) { // Main loop processes every bar
    for (imarket = 0; imarket < n_markets; imarket++) {
      sorted[imarket] = rm_lagged[ibar * n_markets + imarket];
      iwork[imarket] = imarket;
    }

    qsortdsi(0, n_markets - 1, sorted, iwork);

    k = (int)(spread_tail * (n_markets + 1)) - 1;
    if (k < 0)
      k = 0;
    n = k + 1;

    // This loop sums the leader and laggard for the relative momentum extreme
    // markets Sums in the top k+1 and in the bottom k+1.

    rm_leader[ibar] = rm_laggard[ibar] = 0.0;
    while (k >= 0) {
      isub = iwork[k];                   // Low RM (relative momentum) market
      for (i = 0; i < rm_lookahead; i++) // rm_lookahead is normally 1
        rm_laggard[ibar] +=
            returns[isub * n_returns + ibar -
                    i]; // Performance on dates past calculation set
      isub = iwork[n_markets - 1 - k]; // High
      for (i = 0; i < rm_lookahead; i++)
        rm_leader[ibar] +=
            returns[isub * n_returns + ibar -
                    i]; // Performance on dates past calculation set
      --k;
    }

    rm_leader[ibar] /=
        n * rm_lookahead; // Make return a mean per market and bar
    rm_laggard[ibar] /= n * rm_lookahead;
  } // For ibar
}

/*
----------------------------------------------------------------------------------

   compute_CMA() - Compute OOS equity using DOM relative to CMA smoothed
      This computes CMA_OOS (entire universe) and CMA_leader_OOS (leaders only)

   This must be called after compute_DOM_DOE()

   The first valid DOM is at lookback, though DOM at lookback-1 was set to 0.
   For simplicity we will act as if the first valid DOM is at lookback-1.
   So the first valid change in DOM is at lookback, the first valid
   in-sample equity is at lookback+1, and the first valid OOS equity
   is at lookback+2.  So we initialized 'smoothed' to 0, which is the
   implicit DOM at lookback-1.

----------------------------------------------------------------------------------
*/

void JANUS::compute_CMA() {
  int i, k, n, ibar, imarket, isub, ibest;
  double best_equity;

  /*
     Initialize various things
  */

  for (i = min_CMA; i <= max_CMA; i++) {
    CMA_alpha[i - min_CMA] = 2.0 / (i + 1.0);
    CMA_smoothed[i - min_CMA] = CMA_equity[i - min_CMA] = 0.0;
  }

  for (i = 0; i < lookback + 2; i++)
    CMA_OOS[i] = CMA_leader_OOS[i] = 0.0;

  /*
     Main loop
  */

  for (ibar = lookback + 2; ibar < n_returns; ibar++) {

    // This loop finds the lookback that maximizes universe gain.
    // I use this lookback for OOS universe and leader gain.
    // One might arge that leader gain should be used to find the best
    // lookback for OOS leader gain, but I suspect that this much smaller
    // number of markets would introduce instability into the optimization,
    // so I use the universe optimal lookback for both OOS gains.

    best_equity = -1.e60;
    for (i = min_CMA; i <= max_CMA; i++) {
      if (dom_index[ibar - 2] > CMA_smoothed[i - min_CMA]) // Up trend?
        CMA_equity[i - min_CMA] += oos_avg[ibar - 1];
      if (CMA_equity[i - min_CMA] > best_equity) {
        best_equity = CMA_equity[i - min_CMA];
        ibest = i;
      }
      // Update smoothed DOMs through ibar-2
      CMA_smoothed[i - min_CMA] =
          CMA_alpha[i - min_CMA] * dom_index[ibar - 2] +
          (1.0 - CMA_alpha[i - min_CMA]) * CMA_smoothed[i - min_CMA];
    }

    // We now have the lookback that maximizes gain in the complete universe.
    if (dom_index[ibar - 1] >
        CMA_smoothed[ibest - min_CMA]) { // Up trend?  Cumulate universe OOS.
      CMA_OOS[ibar] = oos_avg[ibar];

      // We just did the universe OOS.  Now do the leader OOS.
      // Find the rm leader markets known as of ibar-1
      for (imarket = 0; imarket < n_markets; imarket++) {
        sorted[imarket] = rm[(ibar - 1) * n_markets + imarket];
        iwork[imarket] = imarket;
      }
      qsortdsi(0, n_markets - 1, sorted, iwork);
      k = (int)(spread_tail * (n_markets + 1)) - 1;
      if (k < 0)
        k = 0;
      n = k + 1; // This many leader markets at end of sorted array

      CMA_leader_OOS[ibar] = 0.0;
      while (k >= 0) {
        isub = iwork[n_markets - 1 - k]; // Index of leader
        CMA_leader_OOS[ibar] += returns[isub * n_returns + ibar];
        --k;
      }
      CMA_leader_OOS[ibar] /= n;
    } // If up trend
  } // For ibar
}

/*
----------------------------------------------------------------------------------

   get_market_index() - Copy index to a destination array
   get_dom_index() - Copy DOM index to a destination array
   get_rs() - Copy relative strength to a destination array
   get_rs_fractile() - Copy relative strength fractile to a destination array
   get_rss() - Copy relative strength spread to a destination array
   get_rss_change() - Copy relative strength spread change to a destination
array get_dom ( double *dest , int ord_num ) get_doe ( double *dest , int
ord_num ) get_rm ( double *dest , int ord_num ) get_rm_fractile ( double *dest ,
int ord_num ) get_rs_leader_equity ( double *dest ) get_rs_laggard_equity (
double *dest ) get_rs_ps ( double *dest ) get_rs_leader_advantage ( double *dest
) get_rs_laggard_advantage ( double *dest ) get_oos_avg ( double *dest )
   get_CMA_OOS ( double *dest )
   get_CMA_leader_OOS ( double *dest )

   We have differenced, so the term at index i here corresponds to the price
   at index i+1.

   Functions that have an 'ord_num' parameter use that to specify the market,
   with the first market in the user's market list being 1.
   A value of 0 means return the item for the index.

----------------------------------------------------------------------------------
*/

void JANUS::get_market_index(double *dest) {
  int i;
  double sum;

  sum = 0.0;
  for (i = lookback; i < nbars; i++) {
    sum += mkt_index_returns[i - 1];
    dest[i] = sum;
  }
}

void JANUS::get_dom_index(double *dest) {
  int i;
  double sum;

  sum = 0.0;
  for (i = lookback; i < nbars; i++) {
    sum += dom_index_returns[i - 1];
    dest[i] = sum;
  }
}

void JANUS::get_rs(double *dest, int ord_num) {
  int i;

  for (i = lookback; i < nbars; i++)
    dest[i] = rs[(i - 1) * n_markets + ord_num - 1];
}

void JANUS::get_rs_fractile(double *dest, int ord_num) {
  int i;

  for (i = lookback; i < nbars; i++)
    dest[i] = rs_fractile[(i - 1) * n_markets + ord_num - 1];
}

void JANUS::get_rss(double *dest) {
  int i;

  for (i = lookback; i < nbars; i++)
    dest[i] = rss[i - 1];
}

void JANUS::get_rss_change(double *dest) {
  int i;

  for (i = lookback; i < nbars; i++)
    dest[i] = rss_change[i - 1];
}

void JANUS::get_dom(double *dest, int ord_num) {
  int i;

  if (ord_num == 0) {
    for (i = lookback; i < nbars; i++)
      dest[i] = dom_index[i - 1];
  }

  else {
    for (i = lookback; i < nbars; i++)
      dest[i] = dom[(i - 1) * n_markets + ord_num - 1];
  }
}

void JANUS::get_doe(double *dest, int ord_num) {
  int i;

  if (ord_num == 0) {
    for (i = lookback; i < nbars; i++)
      dest[i] = doe_index[i - 1];
  }

  else {
    for (i = lookback; i < nbars; i++)
      dest[i] = doe[(i - 1) * n_markets + ord_num - 1];
  }
}

void JANUS::get_rm(double *dest, int ord_num) {
  int i;

  for (i = lookback; i < nbars; i++)
    dest[i] = rm[(i - 1) * n_markets + ord_num - 1];
}

void JANUS::get_rm_fractile(double *dest, int ord_num) {
  int i;

  for (i = lookback; i < nbars; i++)
    dest[i] = rm_fractile[(i - 1) * n_markets + ord_num - 1];
}

void JANUS::get_oos_avg(double *dest) {
  int i;
  double sum;

  sum = 0.0;
  for (i = lookback; i < nbars; i++) {
    sum += oos_avg[i - 1];
    dest[i] = sum;
  }
}

void JANUS::get_rs_leader_equity(double *dest) {
  int i;
  double sum;

  sum = 0.0;
  for (i = lookback; i < nbars; i++) {
    sum += rs_leader[i - 1];
    dest[i] = sum;
  }
}

void JANUS::get_rs_laggard_equity(double *dest) {
  int i;
  double sum;

  sum = 0.0;
  for (i = lookback; i < nbars; i++) {
    sum += rs_laggard[i - 1];
    dest[i] = sum;
  }
}

void JANUS::get_rs_ps(double *dest) {
  int i;
  double sum;

  sum = 0.0;
  for (i = lookback; i < nbars; i++) {
    sum += rs_leader[i - 1] - rs_laggard[i - 1];
    dest[i] = sum;
  }
}

void JANUS::get_rs_leader_advantage(double *dest) {
  int i;
  double sum;

  sum = 0.0;
  for (i = lookback; i < nbars; i++) {
    sum += rs_leader[i - 1] - oos_avg[i - 1];
    dest[i] = sum;
  }
}

void JANUS::get_rs_laggard_advantage(double *dest) {
  int i;
  double sum;

  sum = 0.0;
  for (i = lookback; i < nbars; i++) {
    sum += rs_laggard[i - 1] - oos_avg[i - 1];
    dest[i] = sum;
  }
}

void JANUS::get_rm_leader_equity(double *dest) {
  int i;
  double sum;

  sum = 0.0;
  for (i = lookback; i < nbars; i++) {
    sum += rm_leader[i - 1];
    dest[i] = sum;
  }
}

void JANUS::get_rm_laggard_equity(double *dest) {
  int i;
  double sum;

  sum = 0.0;
  for (i = lookback; i < nbars; i++) {
    sum += rm_laggard[i - 1];
    dest[i] = sum;
  }
}

void JANUS::get_rm_ps(double *dest) {
  int i;
  double sum;

  sum = 0.0;
  for (i = lookback; i < nbars; i++) {
    sum += rm_leader[i - 1] - rm_laggard[i - 1];
    dest[i] = sum;
  }
}

void JANUS::get_rm_leader_advantage(double *dest) {
  int i;
  double sum;

  sum = 0.0;
  for (i = lookback; i < nbars; i++) {
    sum += rm_leader[i - 1] - oos_avg[i - 1];
    dest[i] = sum;
  }
}

void JANUS::get_rm_laggard_advantage(double *dest) {
  int i;
  double sum;

  sum = 0.0;
  for (i = lookback; i < nbars; i++) {
    sum += rm_laggard[i - 1] - oos_avg[i - 1];
    dest[i] = sum;
  }
}

void JANUS::get_CMA_OOS(double *dest) {
  int i;
  double sum;

  sum = 0.0;
  for (i = lookback; i < nbars; i++) {
    sum += CMA_OOS[i - 1];
    dest[i] = sum;
  }
}

void JANUS::get_leader_CMA_OOS(double *dest) {
  int i;
  double sum;

  sum = 0.0;
  for (i = lookback; i < nbars; i++) {
    sum += CMA_leader_OOS[i - 1];
    dest[i] = sum;
  }
}
