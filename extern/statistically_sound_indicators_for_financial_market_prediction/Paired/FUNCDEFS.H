extern double atr(int use_log, int icase, int length, double *open,
                  double *high, double *low, double *close);
extern void basic_stats(int n, double *x, double *work, double *var_mean,
                        double *var_min, double *var_max, double *var_iqr);
extern void cmma(int n, int lookback, int atr_length, double *open,
                 double *high, double *low, double *close, double *work,
                 double *output);
extern int comp_var(int n, int var_num, double param1, double param2,
                    double param3, double param4, double *open1, double *high1,
                    double *low1, double *close1, double *volume1,
                    double *open2, double *high2, double *low2, double *close2,
                    double *volume2, int *n_done, int *first_date,
                    int *last_date, double *output, double *work1,
                    double *work2, double *work3);
extern double entropy(int n, double *x);
extern void legendre_2(int n, double *c1, double *c2);
extern void *memalloc(size_t n);
extern void *memallocX(size_t n);
extern void memclose();
extern void memcloseX();
extern void memfree(void *ptr);
extern void memfreeX(void *ptr);
extern void *memrealloc(void *ptr, size_t size);
extern void *memreallocX(void *ptr, size_t size);
extern void memtext(char *text);
extern double normal_cdf(double z);
extern void qsortd(int first, int last, double *data);
extern void qsortds(int first, int last, double *data, double *slave);
extern void qsortdsi(int first, int last, double *data, int *slave);
// extern void qsortisd ( int first , int last , int *data , double *slave ) ;
extern double spearman(int n, double *var1, double *var2, double *x, double *y);
extern void trend(int n, int lookback, int atr_length, double *open,
                  double *high, double *low, double *close, double *work,
                  double *output);
