class JANUS {

public:
  JANUS(int p_nbars, int p_n_markets, int p_lookback, double p_spread_tail,
        int p_min_CMA, int p_max_CMA);
  ~JANUS();
  void prepare(double **prices);
  void compute_rs(int lag);
  void compute_rss();
  void compute_dom_doe();
  void compute_rm(int lag);
  void compute_rs_ps();
  void compute_rm_ps();
  void compute_CMA();
  void get_market_index(double *dest);
  void get_dom_index(double *dest);
  void get_rs(double *dest, int ord_num);
  void get_rs_fractile(double *dest, int ord_num);
  void get_rss(double *dest);
  void get_rss_change(double *dest);
  void get_dom(double *dest, int ord_num);
  void get_doe(double *dest, int ord_num);
  void get_rm(double *dest, int ord_num);
  void get_rm_fractile(double *dest, int ord_num);
  void get_rs_leader_equity(double *dest);
  void get_rs_laggard_equity(double *dest);
  void get_rs_leader_advantage(double *dest);
  void get_rs_laggard_advantage(double *dest);
  void get_rs_ps(double *dest);
  void get_rm_leader_equity(double *dest);
  void get_rm_laggard_equity(double *dest);
  void get_rm_leader_advantage(double *dest);
  void get_rm_laggard_advantage(double *dest);
  void get_rm_ps(double *dest);
  void get_oos_avg(double *dest);
  void get_CMA_OOS(double *dest);
  void get_leader_CMA_OOS(double *dest);

  int ok; // Did memory allocation go well?

private:
  int nbars;          // Number of bars in market history
  int n_returns;      // nbars-1
  int n_markets;      // Number of markets
  int lookback;       // Lookback window length
  int rs_lookahead;   // Lag used in call to compute_rs(); used for rs_ps(raw)
  int rm_lookahead;   // Lag used in call to compute_rm(); used for rm_ps(raw)
  double spread_tail; // Fraction (typically 0.1) of markets in each tail for
                      // spread
  int min_CMA;        // Minimum lookback for CMA
  int max_CMA;        // And maximum

  // These are computed for the user

  double *rs;          // Relative strength (each market, so this is a matrix)
  double *rs_fractile; // Fractile (0-1) of relative strength (each market, so
                       // this is a matrix)
  double
      *rs_lagged; // Lagged relative strength (each market, so this is a matrix)
  double *rss;    // Relative strength spread
  double *rss_change; // One-bar change in relative strength spread
  double *dom;       // Direction of momentum (each market, so this is a matrix)
  double *doe;       // Direction of entropy (each market, so this is a matrix)
  double *dom_index; // Direction of momentum of index
  double *doe_index; // Direction of entropy of index
  double *rm;        // Relative momentum (each market, so this is a matrix)
  double *rm_fractile; // Fractile (0-1) of relative momentum (each market, so
                       // this is a matrix)
  double
      *rm_lagged; // Lagged relative momentum (each market, so this is a matrix)
  double *rs_leader;  // OOS bar performance of RS leaders
  double *rs_laggard; // OOS bar performance of RS laggards
  double *oos_avg; // OOS bar performance of all markets, ignoring lag (just as
                   // if lag=1)
  double *rm_leader;      // OOS bar performance of RM leaders
  double *rm_laggard;     // OOS bar performance of RM laggards
  double *CMA_OOS;        // OOS bar performance of DOM CMA system
  double *CMA_leader_OOS; // OOS bar performance of DOM CMA system based on
                          // leaders only

  // These are work areas

  double *returns; // Changes computed from prices; save them so just done once
  double *mkt_index_returns; // Precomputed median across markets return for
                             // each bar
  double
      *dom_index_returns; // Precomputed median across markets DOM for each bar
  double *index;          // Work vector lookback long
  double *sorted;         // Work vector max ( n_markets , lookback ) long
  int *iwork;             // Work vector n_markets long
  double *dom_sum;        // Work vector n_markets long
  double *doe_sum;        // Work vector n_markets long
  double *CMA_alpha;      // Alpha for each CMA trial lookback
  double *CMA_smoothed;   // Current smoothed DOM for each CMA trial lookback
  double *CMA_equity; // Current in-sample equity for each CMA trial lookback
};
