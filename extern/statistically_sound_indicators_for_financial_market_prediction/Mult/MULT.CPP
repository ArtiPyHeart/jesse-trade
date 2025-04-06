/******************************************************************************/
/*                                                                            */
/*  MULT - Multiple-market indicators                                         */
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


#define MKTBUF 2048 /* Alloc for market info in chunks of this many records */
                    /* This is not critical and can be any reasonable value */

/*
   These are defined in MEM64.CPP
   This code is needed only if MALLOC maps to memalloc et cetera.
   If MALLOC maps to malloc (etc) then this is not needed.
   This is controlled by the MEMDEBUG flag in CONST.H.
*/

#if MEMDEBUG
extern int mem_keep_log;     // Keep a log file?
extern char mem_file_name[]; // Log file name
extern __int64 mem_max_used; // Maximum memory ever in use
#endif

/*
--------------------------------------------------------------------------------

   Local routines get parameters from user's line

--------------------------------------------------------------------------------
*/

void get_1_param(char *lptr, double lower1, double upper1, double *param1) {
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param1 = atof(lptr);
  if (*param1 < lower1)
    *param1 = lower1;
  if (*param1 > upper1)
    *param1 = upper1;
}

void get_2_params(char *lptr, double lower1, double upper1, double lower2,
                  double upper2, double *param1, double *param2) {
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param1 = atof(lptr);
  if (*param1 < lower1)
    *param1 = lower1;
  if (*param1 > upper1)
    *param1 = upper1;
  while (isgraph(*lptr))
    ++lptr;
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param2 = atof(lptr);
  if (*param2 < lower2)
    *param2 = lower2;
  if (*param2 > upper2)
    *param2 = upper2;
}

void get_3_params(char *lptr, double lower1, double upper1, double lower2,
                  double upper2, double lower3, double upper3, double *param1,
                  double *param2, double *param3) {
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param1 = atof(lptr);
  if (*param1 < lower1)
    *param1 = lower1;
  if (*param1 > upper1)
    *param1 = upper1;
  while (isgraph(*lptr))
    ++lptr;
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param2 = atof(lptr);
  if (*param2 < lower2)
    *param2 = lower2;
  if (*param2 > upper2)
    *param2 = upper2;
  while (isgraph(*lptr))
    ++lptr;
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param3 = atof(lptr);
  if (*param3 < lower3)
    *param3 = lower3;
  if (*param3 > upper3)
    *param3 = upper3;
}

void get_4_params(char *lptr, double lower1, double upper1, double lower2,
                  double upper2, double lower3, double upper3, double lower4,
                  double upper4, double *param1, double *param2, double *param3,
                  double *param4) {
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param1 = atof(lptr);
  if (*param1 < lower1)
    *param1 = lower1;
  if (*param1 > upper1)
    *param1 = upper1;
  while (isgraph(*lptr))
    ++lptr;
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param2 = atof(lptr);
  if (*param2 < lower2)
    *param2 = lower2;
  if (*param2 > upper2)
    *param2 = upper2;
  while (isgraph(*lptr))
    ++lptr;
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param3 = atof(lptr);
  if (*param3 < lower3)
    *param3 = lower3;
  if (*param3 > upper3)
    *param3 = upper3;
  while (isgraph(*lptr))
    ++lptr;
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param4 = atof(lptr);
  if (*param4 < lower4)
    *param4 = lower4;
  if (*param4 > upper4)
    *param4 = upper4;
}

/*
-------------------------------------------------------------------------------

   Local routine to read market prices and volume if present

-------------------------------------------------------------------------------
*/

static int read_market(char *MarketName, int **date, double **open,
                       double **high, double **low, double **close,
                       double **volume, int *nprices) {
  int i, bufcnt;
  char line[256], *cptr;
  FILE *fp;

  if (fopen_s(&fp, MarketName, "rt")) {
    printf("\n\nCannot open market history file %s", MarketName);
    return ERROR_FILE;
  }

  *date = (int *)MALLOC(MKTBUF * sizeof(int));
  *open = (double *)MALLOC(MKTBUF * sizeof(double));
  *high = (double *)MALLOC(MKTBUF * sizeof(double));
  *low = (double *)MALLOC(MKTBUF * sizeof(double));
  *close = (double *)MALLOC(MKTBUF * sizeof(double));
  *volume = (double *)MALLOC(MKTBUF * sizeof(double));
  if (*date == NULL || *open == NULL || *high == NULL || *low == NULL ||
      *close == NULL || *volume == NULL) {
    printf("\n\nInsufficient memory reading market history file %s",
           MarketName);
    fclose(fp);
    if (*date != NULL)
      FREE(*date);
    if (*open != NULL)
      FREE(*open);
    if (*high != NULL)
      FREE(*high);
    if (*low != NULL)
      FREE(*low);
    if (*close != NULL)
      FREE(*close);
    if (*volume != NULL)
      FREE(*volume);
    return ERROR_INSUFFICIENT_MEMORY;
  }

  bufcnt = MKTBUF; // There are this many record slots available now

  printf("\nReading market file %s", MarketName);

  *nprices = 0; // Counts lines (prices) read

  for (;;) {

    if (feof(fp)                          // If end of file
        || (fgets(line, 256, fp) == NULL) // Or unable to read line
        || (strlen(line) < 2))            // Or empty line
      break;                              // We are done reading price history

    if (ferror(fp)) { // If an error reading file
      fclose(fp);     // Quit immediately
      printf("\nError reading line %d of file %s", *nprices + 1, MarketName);
      return ERROR_FILE;
    }

    if (!bufcnt) { // Allocate a new memory block if needed
      *date = (int *)REALLOC(*date, (*nprices + MKTBUF) * sizeof(int));
      *open = (double *)REALLOC(*open, (*nprices + MKTBUF) * sizeof(double));
      *high = (double *)REALLOC(*high, (*nprices + MKTBUF) * sizeof(double));
      *low = (double *)REALLOC(*low, (*nprices + MKTBUF) * sizeof(double));
      *close = (double *)REALLOC(*close, (*nprices + MKTBUF) * sizeof(double));
      *volume =
          (double *)REALLOC(*volume, (*nprices + MKTBUF) * sizeof(double));
      if (*date == NULL || *open == NULL || *high == NULL || *low == NULL ||
          *close == NULL || *volume == NULL) {
        fclose(fp);
        printf("\n\nInsufficient memory reading market history file %s  Press "
               "any key...",
               MarketName);
        if (*date != NULL)
          FREE(*date);
        if (*open != NULL)
          FREE(*open);
        if (*high != NULL)
          FREE(*high);
        if (*low != NULL)
          FREE(*low);
        if (*close != NULL)
          FREE(*close);
        if (*volume != NULL)
          FREE(*volume);
        return ERROR_INSUFFICIENT_MEMORY;
      } // If insufficient memory
      bufcnt = MKTBUF; // There are this many new record slots available now
    } // If allocating new block

    // Parse the date and do a crude sanity check

    for (i = 0; i < 8; i++) {
      if ((line[i] < '0') || (line[i] > '9')) {
        fclose(fp);
        printf("\nInvalid date reading line %d of file %s", *nprices + 1,
               MarketName);
        FREE(*date);
        FREE(*open);
        FREE(*high);
        FREE(*low);
        FREE(*close);
        FREE(*volume);
        return ERROR_SYNTAX;
      }
    }
    (*date)[*nprices] = atoi(line);

    // Parse the open

    cptr = line + 9; // Price is in this column or beyond
                     // (Next loop allows price to start past this)

    while (*cptr == ' ' || *cptr == '\t' || *cptr == ',') // Delimiters
      ++cptr; // Move up to the price

    (*open)[*nprices] = atof(cptr);

    while (*cptr && *cptr != ' ' && *cptr != ',' && *cptr != '\t')
      ++cptr; // Pass the price; stop at delimiter

    // Parse the high

    while (*cptr == ' ' || *cptr == '\t' || *cptr == ',') // Delimiters
      ++cptr; // Move up to the price

    (*high)[*nprices] = atof(cptr);

    while (*cptr && *cptr != ' ' && *cptr != ',' && *cptr != '\t')
      ++cptr; // Pass the price; stop at delimiter

    // Parse the low

    while (*cptr == ' ' || *cptr == '\t' || *cptr == ',') // Delimiters
      ++cptr; // Move up to the price

    (*low)[*nprices] = atof(cptr);

    while (*cptr && *cptr != ' ' && *cptr != ',' && *cptr != '\t')
      ++cptr; // Pass the price; stop at delimiter

    // Parse the close

    while (*cptr == ' ' || *cptr == '\t' || *cptr == ',') // Delimiters
      ++cptr; // Move up to the price

    (*close)[*nprices] = atof(cptr);

    while (*cptr && *cptr != ' ' && *cptr != ',' && *cptr != '\t')
      ++cptr; // Pass the price; stop at delimiter

    // Parse the volume

    while (*cptr == ' ' || *cptr == '\t' || *cptr == ',') // Delimiters
      ++cptr;                                             // Move up

    (*volume)[*nprices] = atof(cptr);

    if ((*low)[*nprices] > (*open)[*nprices] ||
        (*low)[*nprices] > (*close)[*nprices] ||
        (*high)[*nprices] < (*open)[*nprices] ||
        (*high)[*nprices] < (*close)[*nprices]) {
      fclose(fp);
      printf("\nInvalid open/high/low/close reading line %d of file %s",
             *nprices + 1, MarketName);
      FREE(*date);
      FREE(*open);
      FREE(*high);
      FREE(*low);
      FREE(*close);
      FREE(*volume);
      return ERROR_SYNTAX;
    }

    ++*nprices;
    --bufcnt; // One less slot remains

  } // For all lines

  fclose(fp);

  printf("\nMarket price history read; %d cases from %d to %d", *nprices,
         (*date)[0], (*date)[*nprices - 1]);
  return 0;
}

/*
--------------------------------------------------------------------------------

   Main routine

--------------------------------------------------------------------------------
*/

int main(int argc,    // Number of command line arguments (includes prog name)
         char *argv[] // Arguments (prog name is argv[0])
) {
  int i, k, icase, nvars, var_num, n_done, ret_val, n_markets, front_bad;
  int line_number, date, first_date, last_date, max_date, all_same_date,
      n_cases;
  int **market_date, *market_index, *market_n, grand_index, *iwork;
  double param1, param2, param3, param4;
  double **market_open, **market_high, **market_low, **market_close,
      **market_volume;
  double *var_work, *vptr, *vars[MAX_VARS];
  double *work1, *work2, *work3, *big_work, *big_work2, *big_work3, var_min,
      var_max, var_mean, var_iqr, var_ent;
  char user_name[MAX_NAME_LENGTH + 1], *market_names, *mptr;
  char var_names[MAX_VARS][MAX_NAME_LENGTH + 1];
  char MarketListName[1024], MarketFileName[1024], ScriptName[1024], line[512],
      msg[256], *lptr;
  FILE *fp;
  SYSTEMTIME systime;

  var_work = work1 = work2 = work3 = big_work = big_work2 = big_work3 = NULL;
  iwork = NULL;
  for (i = 0; i < MAX_VARS; i++)
    vars[i] = NULL;
  nvars = 0;

  /*
     Process command line parameters
  */

#if 1
  if (argc != 3) {
    printf("\nUsage: MULT  MarketList  ScriptName");
    printf("\n  MarketList - List of all markets (complete file names)");
    printf("\n  ScriptName - name of variable script file");
    exit(1);
  }

  strcpy_s(MarketListName, argv[1]);
  strcpy_s(ScriptName, argv[2]);
#else
  strcpy_s(MarketListName, "MULT_MKTS.TXT"); // For diagnostics only
  strcpy_s(ScriptName, "VM.TXT");
#endif

  /*
     Memory checking stuff for MEM64.CPP safe memory allocation.
     This code is needed only if MALLOC maps to memalloc et cetera.
     If MALLOC maps to malloc (etc) then this is not needed.
     This is controlled by the MEMDEBUG flag in CONST.H.
  */

#if MEMDEBUG
  mem_max_used = 0;
  mem_keep_log = 1;
  _fullpath(mem_file_name, "MEM.LOG", 256);
  if (fopen_s(&fp, mem_file_name, "wt")) {
    mem_file_name[0] = 0;
    mem_keep_log = 0;
  } else {
    GetLocalTime(&systime);
    sprintf_s(line, "PAIRED memory log %s  %d/%d/%02d  %d:%02d:%02d",
              mem_file_name, systime.wMonth, systime.wDay, systime.wYear % 100,
              systime.wHour, systime.wMinute, systime.wSecond);
    fprintf(fp, "%s", line);
    fclose(fp);
  }
#endif

  market_names = NULL;
  market_date = NULL;
  market_index = NULL;
  market_n = NULL;
  market_open = NULL;
  market_high = NULL;
  market_low = NULL;
  market_close = NULL;
  market_volume = NULL;
  fp = NULL;

  /*
  -------------------------------------------------------------------------------

     Allocate memory that we can allocate now

  -------------------------------------------------------------------------------
  */

  if (fopen_s(&fp, MarketListName, "rt")) {
    printf("\nCannot open file list %s... aborting", MarketListName);
    goto FINISH;
  }

  market_names = (char *)MALLOC(MAX_MARKETS * MAX_NAME_LENGTH * sizeof(char));
  assert(market_names != NULL);

  market_date = (int **)MALLOC(MAX_MARKETS * sizeof(int *));
  assert(market_date != NULL);

  market_index = (int *)MALLOC(MAX_MARKETS * sizeof(int));
  assert(market_index != NULL);

  market_n = (int *)MALLOC(MAX_MARKETS * sizeof(int));
  assert(market_n != NULL);

  market_open = (double **)MALLOC(MAX_MARKETS * sizeof(double *));
  assert(market_open != NULL);

  market_high = (double **)MALLOC(MAX_MARKETS * sizeof(double *));
  assert(market_high != NULL);

  market_low = (double **)MALLOC(MAX_MARKETS * sizeof(double *));
  assert(market_low != NULL);

  market_close = (double **)MALLOC(MAX_MARKETS * sizeof(double *));
  assert(market_close != NULL);

  market_volume = (double **)MALLOC(MAX_MARKETS * sizeof(double *));
  assert(market_volume != NULL);

  for (i = 0; i < MAX_MARKETS; i++) {
    market_date[i] = NULL;
    market_open[i] = NULL;
    market_high[i] = NULL;
    market_low[i] = NULL;
    market_close[i] = NULL;
    market_volume[i] = NULL;
  }

  /*
  -------------------------------------------------------------------------------

     Read the market list file and market files

  -------------------------------------------------------------------------------
  */

  n_markets = 0;
  for (;;) {

    // Get the name of a market file
    if ((fgets(line, 256, fp) == NULL) || (strlen(line) < 2)) {
      if (ferror(fp) || !n_markets) {
        printf("\nERROR... Cannot read market list file %s", MarketListName);
        goto FINISH;
      } else
        break; // Normal end of list file
    }

    // Copy this market file name

    lptr = &line[0];
    k = 0;
    while (isalnum(*lptr) || *lptr == '_' || *lptr == '\\' || *lptr == ':' ||
           *lptr == '.')
      MarketFileName[k++] = *lptr++;
    MarketFileName[k] = 0; // This is now the exact file name

    // Get and save the name of the market from the file name
    // We assume it is just before the last period.

    strcpy_s(msg, MarketFileName);
    lptr = &msg[k - 1]; // Last character in file name
    while (lptr > &msg[0] && *lptr != '.')
      --lptr;
    if (*lptr != '.') { // We require an extension, not unreasonable
      printf("\nERROR... Market file name (%s) is not legal", MarketFileName);
      goto FINISH;
    }
    *lptr = 0; // This removes extension
    while (lptr > &msg[0] && *lptr != '.' && *lptr != '\\' && *lptr != ':')
      --lptr; // Back up until we get path stuff
    if (*lptr == '.' || *lptr == '\\' ||
        *lptr == ':') // If a path character caused loop exit, pass it
      ++lptr;
    if (strlen(lptr) > MAX_NAME_LENGTH - 1) {
      printf("\nERROR... Market name (%s) is too long", lptr);
      goto FINISH;
    }
    strcpy_s(market_names + n_markets * MAX_NAME_LENGTH, MAX_NAME_LENGTH, lptr);

    // We now have the name of a market history file.  Read this file.

    if (read_market(MarketFileName, market_date + n_markets,
                    market_open + n_markets, market_high + n_markets,
                    market_low + n_markets, market_close + n_markets,
                    market_volume + n_markets, market_n + n_markets)) {
      printf("\nFile error reading market %s... aborting", lptr);
      goto FINISH;
    }
    ++n_markets;
  } // Read all markets

  fclose(fp);
  fp = NULL;

  /*
  -----------------------------------------------------------------------------------------

     We have completely finished reading all markets, and their data is stored.
     However, our indicators require that all data be date aligned, and
     there is no guaranty that we have this.
     So now we remove all records that do not have data for all markets.
     The algorithm given here may not be the most efficient way to do this,
     but compared with what's coming soon this is a drop in the bucket of CPU
  time.

     The algorithm is as follows:
        Initialize each market's current index to 0
        Initialize the grand (compressed) index to 0
        Loop
           Find the latest (largest) date at each market's current index across
  all markets Advance all markets' current index until the date reaches or
  passes this date If all markets have the same current date: keep this date by
  copying this market record to the grand index spot Advance each market's
  current index as well as the grand index

  -----------------------------------------------------------------------------------------
  */

  for (i = 0; i < n_markets; i++)
    market_index[i] = 0;
  grand_index = 0;

  for (;;) {

    // Find max date at current index of each market

    max_date = 0;
    for (i = 0; i < n_markets; i++) {
      date = market_date[i][market_index[i]];
      if (date > max_date)
        max_date = date;
    }

    // Advance all markets until they reach or pass max date
    // Keep track of whether they all equal max_date

    all_same_date = 1;
    for (i = 0; i < n_markets; i++) {
      while (market_index[i] < market_n[i]) { // Must not over-run a market!
        date = market_date[i][market_index[i]];
        if (date >= max_date)
          break;
        ++market_index[i];
      }
      if (date != max_date) // Did some market jump over max?
        all_same_date = 0;

      if (market_index[i] >= market_n[i]) // If even one market runs out
        break;                            // We are done
    }

    if (i < n_markets) // If even one market runs out
      break;           // We are done

    // If we have a complete set for this date, grab it

    if (all_same_date) {
      for (i = 0; i < n_markets; i++) {
        assert(market_date[i][market_index[i]] = max_date);
        market_date[i][grand_index] = max_date; // Redundant, but clear
        market_open[i][grand_index] = market_open[i][market_index[i]];
        market_high[i][grand_index] = market_high[i][market_index[i]];
        market_low[i][grand_index] = market_low[i][market_index[i]];
        market_close[i][grand_index] = market_close[i][market_index[i]];
        market_volume[i][grand_index] = market_volume[i][market_index[i]];
        ++market_index[i];
      }
      ++grand_index;
    }
  } // Endless compression loop

  n_cases = grand_index;

  printf("\n\nMerged database has %d records from date %d to %d", n_cases,
         market_date[0][0], market_date[0][n_cases - 1]);

  /*
     Free memory that we no longer need
  */

  FREE(market_n);
  market_n = NULL;
  FREE(market_index);
  market_index = NULL;

  if (n_cases == 0) {
    printf("\nAborting because there are no common dates");
    goto FINISH;
  }

  /*
  -------------------------------------------------------------------------------

     Read and process variable script file

  -------------------------------------------------------------------------------
  */

  if (fopen_s(&fp, ScriptName, "rt")) {
    printf("\n\nCannot open variable script file %s", ScriptName);
    goto FINISH;
  }

  k = (n_cases > n_markets) ? n_cases : n_markets;
  var_work = (double *)MALLOC(n_cases * sizeof(double));
  work1 = (double *)MALLOC(k * sizeof(double));
  work2 = (double *)MALLOC(k * sizeof(double));
  work3 = (double *)MALLOC(k * sizeof(double));
  big_work = (double *)MALLOC(n_markets * k * sizeof(double));
  big_work2 = (double *)MALLOC(n_markets * n_markets * sizeof(double));
  big_work3 = (double *)MALLOC((n_markets * n_markets + 2 * n_markets) *
                               sizeof(double));
  iwork = (int *)MALLOC(n_markets * sizeof(int));
  if (var_work == NULL || work1 == NULL || work2 == NULL || work3 == NULL ||
      big_work == NULL || big_work2 == NULL || big_work3 == NULL ||
      iwork == NULL) {
    printf("\n\nInsufficient memory processing script file");
    goto FINISH;
  }

  param1 = param2 = param3 = param4 = 0.0; // Not needed, shuts up LINT
  nvars = 0;                               // Counts variables
  front_bad = 0;

  printf(
      "\n\nVarNum      Variable  N cases    First date   Last date          "
      "Mean       Minimum       Maximum    IQ Range     Rng/IQR  Rel Entropy");

  for (line_number = 1;; line_number++) {

    if (feof(fp)                          // If end of file
        || (fgets(line, 256, fp) == NULL) // Or unable to read line
        || (strlen(line) < 2))            // Or empty line
      break;                              // We are done reading price history

    if (ferror(fp)) { // If an error reading file
      printf("\nError reading line %d of file %s", line_number, ScriptName);
      goto FINISH;
    }

    /*
       Change line to all upper case for uniformity, then delete any comments
    */

    _strupr_s(line);

    mptr = strchr(line, ';');
    if (mptr != NULL)
      *mptr = NULL;

    if (strlen(line) < 2) // Ignore blank lines
      continue;

    /*
       Parse the variable's user name
    */

    // Copy the user's name for this variable
    lptr = line;
    mptr = &user_name[0];
    while (*lptr == ' ' || *lptr == '\t') // Bypass leading blanks
      ++lptr;
    k = 0; // Will count name length
    while (*lptr && *lptr != ' ' && *lptr != '\t' && *lptr != ':') {
      ++k;
      if (k > MAX_NAME_LENGTH) {
        printf("\nUser name longer than %d characters in line %d",
               MAX_NAME_LENGTH, line_number);
        goto FINISH;
      }
      *mptr++ = *lptr++;
    }
    *mptr = 0;

    for (i = 0; i < nvars; i++) {
      if (!strcmp(user_name, var_names[i])) {
        printf("\nUser name %s duplicated in line %d", user_name, line_number);
        goto FINISH;
      }
    }
    strcpy_s(var_names[nvars], user_name);

    // Bypass colon and blanks to get to parser name of variable
    while (*lptr == ' ' || *lptr == '\t' || *lptr == ':')
      ++lptr;

    /*
       Determine which variable it is and get necessary parameters
    */

    if (!strncmp(lptr, "TREND RANK", 10)) {
      var_num = VAR_TREND_RANK;
      lptr += 10;
      get_2_params(lptr, 2.0, n_cases / 2, 1.0, n_cases / 2, &param1, &param2);
    }

    else if (!strncmp(lptr, "CMMA RANK", 9)) {
      var_num = VAR_CMMA_RANK;
      lptr += 9;
      get_2_params(lptr, 2.0, n_cases / 2, 1.0, n_cases / 2, &param1, &param2);
    }

    else if (!strncmp(lptr, "TREND MEDIAN", 12)) {
      var_num = VAR_TREND_MEDIAN;
      lptr += 12;
      get_2_params(lptr, 2.0, n_cases / 2, 1.0, n_cases / 2, &param1, &param2);
    }

    else if (!strncmp(lptr, "CMMA MEDIAN", 11)) {
      var_num = VAR_CMMA_MEDIAN;
      lptr += 11;
      get_2_params(lptr, 2.0, n_cases / 2, 1.0, n_cases / 2, &param1, &param2);
    }

    else if (!strncmp(lptr, "TREND RANGE", 11)) {
      var_num = VAR_TREND_RANGE;
      lptr += 11;
      get_2_params(lptr, 2.0, n_cases / 2, 1.0, n_cases / 2, &param1, &param2);
    }

    else if (!strncmp(lptr, "CMMA RANGE", 10)) {
      var_num = VAR_CMMA_RANGE;
      lptr += 10;
      get_2_params(lptr, 2.0, n_cases / 2, 1.0, n_cases / 2, &param1, &param2);
    }

    else if (!strncmp(lptr, "TREND IQR", 9)) {
      var_num = VAR_TREND_IQR;
      lptr += 9;
      get_2_params(lptr, 2.0, n_cases / 2, 1.0, n_cases / 2, &param1, &param2);
    }

    else if (!strncmp(lptr, "CMMA IQR", 8)) {
      var_num = VAR_CMMA_IQR;
      lptr += 8;
      get_2_params(lptr, 2.0, n_cases / 2, 1.0, n_cases / 2, &param1, &param2);
    }

    else if (!strncmp(lptr, "TREND CLUMP", 11)) {
      var_num = VAR_TREND_CLUMP;
      lptr += 11;
      get_2_params(lptr, 2.0, n_cases / 2, 1.0, n_cases / 2, &param1, &param2);
    }

    else if (!strncmp(lptr, "CMMA CLUMP", 10)) {
      var_num = VAR_CMMA_CLUMP;
      lptr += 10;
      get_2_params(lptr, 2.0, n_cases / 2, 1.0, n_cases / 2, &param1, &param2);
    }

    else if (!strncmp(lptr, "MAHAL", 5)) {
      var_num = VAR_MAHAL;
      lptr += 5;
      get_2_params(lptr, 2.0, n_cases / 2, 0.0, n_cases / 2, &param1, &param2);
      if (param1 < n_markets + 10) {
        printf("\n\nMAHAL requires lookback at least 10 greater than number of "
               "markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "ABS RATIO", 9)) {
      var_num = VAR_ABS_RATIO;
      lptr += 9;
      get_2_params(lptr, 2.0, n_cases / 2, 0.0, 1.0, &param1, &param2);
      if (param1 < n_markets) {
        printf(
            "\n\nABS RATIO requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "ABS SHIFT", 9)) {
      var_num = VAR_ABS_SHIFT;
      lptr += 9;
      get_4_params(lptr, 2.0, n_cases / 2, 0.0, 1.0, 0.0, n_cases / 2, 0.0,
                   n_cases / 2, &param1, &param2, &param3, &param4);
      if (param1 < n_markets) {
        printf(
            "\n\nABS SHIFT requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "COHERENCE", 9)) {
      var_num = VAR_COHERENCE;
      lptr += 9;
      get_1_param(lptr, 2.0, n_cases / 2, &param1);
      if (param1 < n_markets) {
        printf(
            "\n\nCOHERENCE requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "DELTA COHERENCE", 15)) {
      var_num = VAR_DELTA_COHERENCE;
      lptr += 15;
      get_2_params(lptr, 2.0, n_cases / 2, 0.0, n_cases / 2, &param1, &param2);
      if (param1 < n_markets) {
        printf("\n\nDELTA COHERENCE requires lookback at least the number of "
               "markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS INDEX MARKET",
                      18)) { // Median of market returns
      var_num = VAR_JANUS_INDEX_MARKET;
      lptr += 18;
      get_1_param(lptr, 2.0, n_cases / 2, &param1);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS INDEX DOM",
                      15)) { // Median of DOM changes ('returns')
      var_num = VAR_JANUS_INDEX_DOM;
      lptr += 15;
      get_1_param(lptr, 2.0, n_cases / 2, &param1);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS RAW RS",
                      12)) { // Relative strength of each market's returns
      var_num = VAR_JANUS_RAW_RS;
      lptr += 12;
      get_2_params(lptr, 2.0, n_cases / 2, 1.0, n_markets, &param1, &param2);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS FRACTILE RS",
                      17)) { // RAW RS expressed as 0-1 fractile
      var_num = VAR_JANUS_FRACTILE_RS;
      lptr += 17;
      get_2_params(lptr, 2.0, n_cases / 2, 1.0, n_markets, &param1, &param2);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(
                 lptr, "JANUS DELTA FRACTILE RS",
                 23)) { // Change in relative strength of each market's returns
      var_num = VAR_JANUS_DELTA_FRACTILE_RS;
      lptr += 23;
      get_3_params(lptr, 2.0, n_cases / 2, 0.0, n_cases / 2, 1.0, n_cases / 2,
                   &param1, &param2, &param3);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS RSS",
                      9)) { // Spread of RS; a single array of bars; second
                            // param is smoothing
      var_num = VAR_JANUS_RSS;
      lptr += 9;
      get_2_params(lptr, 2.0, n_cases / 2, 0.0, n_cases / 2, &param1, &param2);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS DELTA RSS",
                      15)) { // Change in RSS; a single array of bars; second
                             // param is smoothing
      var_num = VAR_JANUS_DELTA_RSS;
      lptr += 15;
      get_2_params(lptr, 2.0, n_cases / 2, 0.0, n_cases / 2, &param1, &param2);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS DOM",
                      9)) { // Each market as well as index (Market 0)
      var_num = VAR_JANUS_DOM;
      lptr += 9;
      get_2_params(lptr, 2.0, n_cases / 2, 0.0, n_markets, &param1, &param2);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS DOE",
                      9)) { // Each market as well as index (Market 0)
      var_num = VAR_JANUS_DOE;
      lptr += 9;
      get_2_params(lptr, 2.0, n_cases / 2, 0.0, n_markets, &param1, &param2);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS RAW RM",
                      12)) { // Relative strength of DOM for each market
      var_num = VAR_JANUS_RAW_RM;
      lptr += 12;
      get_2_params(lptr, 2.0, n_cases / 2, 1.0, n_markets, &param1, &param2);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS FRACTILE RM",
                      17)) { // RM expressed as 0-1 fractile
      var_num = VAR_JANUS_FRACTILE_RM;
      lptr += 17;
      get_2_params(lptr, 2.0, n_cases / 2, 1.0, n_markets, &param1, &param2);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS DELTA FRACTILE RM",
                      23)) { // Change in fractile relative strength of each
                             // market's returns
      var_num = VAR_JANUS_DELTA_FRACTILE_RM;
      lptr += 23;
      get_3_params(lptr, 2.0, n_cases / 2, 0.0, n_cases / 2, 1.0, n_cases / 2,
                   &param1, &param2, &param3);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS RS LEADER EQUITY",
                      22)) { // Equity holding RS leaders
      var_num = VAR_JANUS_RS_LEADER_EQUITY;
      lptr += 22;
      get_1_param(lptr, 2.0, n_cases / 2, &param1);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS RS LAGGARD EQUITY",
                      23)) { // Equity holding RS laggards
      var_num = VAR_JANUS_RS_LAGGARD_EQUITY;
      lptr += 23;
      get_1_param(lptr, 2.0, n_cases / 2, &param1);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS RS LEADER ADVANTAGE",
                      25)) { // Equity of RS leaders minus universe equity
      var_num = VAR_JANUS_RS_LEADER_ADVANTAGE;
      lptr += 25;
      get_1_param(lptr, 2.0, n_cases / 2, &param1);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS RS LAGGARD ADVANTAGE",
                      26)) { // Equity of RS laggards minus universe equity
      var_num = VAR_JANUS_RS_LAGGARD_ADVANTAGE;
      lptr += 26;
      get_1_param(lptr, 2.0, n_cases / 2, &param1);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS RS PS",
                      11)) { // RS performance spread (leaders minus laggards)
      var_num = VAR_JANUS_RS_PS;
      lptr += 11;
      get_1_param(lptr, 2.0, n_cases / 2, &param1);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS RM LEADER EQUITY",
                      22)) { // Equity holding RM leaders
      var_num = VAR_JANUS_RM_LEADER_EQUITY;
      lptr += 22;
      get_1_param(lptr, 2.0, n_cases / 2, &param1);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS RM LAGGARD EQUITY",
                      23)) { // Equity holding RM laggards
      var_num = VAR_JANUS_RM_LAGGARD_EQUITY;
      lptr += 23;
      get_1_param(lptr, 2.0, n_cases / 2, &param1);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS RM LEADER ADVANTAGE",
                      25)) { // Equity of RM leaders minus universe equity
      var_num = VAR_JANUS_RM_LEADER_ADVANTAGE;
      lptr += 25;
      get_1_param(lptr, 2.0, n_cases / 2, &param1);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS RM LAGGARD ADVANTAGE",
                      26)) { // Equity of RM laggards minus universe equity
      var_num = VAR_JANUS_RM_LAGGARD_ADVANTAGE;
      lptr += 26;
      get_1_param(lptr, 2.0, n_cases / 2, &param1);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS RM PS", 11)) { // RM performance spread
      var_num = VAR_JANUS_RM_PS;
      lptr += 11;
      get_1_param(lptr, 2.0, n_cases / 2, &param1);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS CMA OOS",
                      13)) { // OOS equity of DOM-trigger universe system
      var_num = VAR_JANUS_CMA_OOS;
      lptr += 13;
      get_1_param(lptr, 2.0, n_cases / 2, &param1);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS LEADER CMA OOS",
                      20)) { // OOS equity of DOM-trigger leader system
      var_num = VAR_JANUS_LEADER_CMA_OOS;
      lptr += 20;
      get_1_param(lptr, 2.0, n_cases / 2, &param1);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "JANUS OOS AVG", 13)) { // Average of market returns
      var_num = VAR_JANUS_OOS_AVG;
      lptr += 13;
      get_1_param(lptr, 2.0, n_cases / 2, &param1);
      if (param1 < n_markets) {
        printf("\n\nJANUS requires lookback at least the number of markets");
        printf(
            "\n      and preferably a lot more.  Aborting.  Press any key...");
        goto FINISH;
      }
    }

    else {
      printf("\n\nInvalid variable (%s)  Press any key...", lptr);
      goto FINISH;
    }

    /*
       Compute, analyze, and save the variable
    */

    vptr = vars[nvars] = (double *)MALLOC(n_cases * sizeof(double));
    if (vars[nvars] == NULL) {
      printf("\n\nInsufficient memory reading market script file %s  Press any "
             "key...",
             ScriptName);
      goto FINISH;
    } // If insufficient memory

    ret_val =
        comp_var(n_cases, n_markets, var_num, param1, param2, param3, param4,
                 market_open, market_high, market_low, market_close,
                 market_volume, &n_done, &first_date, &last_date, vptr, work1,
                 work2, work3, big_work, big_work2, big_work3, iwork);
    if (ret_val) {
      printf("\n\nERROR processing variable script file... Aborting");
      goto FINISH;
    }

    if (n_cases - n_done >
        front_bad) // Keep track of max invalid at start of series
      front_bad = n_cases - n_done;

    var_ent = entropy(last_date - first_date + 1, vptr + first_date);
    basic_stats(last_date - first_date + 1, vptr + first_date, var_work,
                &var_mean, &var_min, &var_max, &var_iqr);
    printf("\n%4d %15s %7d      %8d     %8d     %9.3lf     %9.3lf     %9.3lf  "
           "%9.3lf    %8.3lf     %.3lf",
           nvars + 1, user_name, n_done, market_date[0][first_date],
           market_date[0][last_date], var_mean, var_min, var_max, var_iqr,
           (var_max - var_min) / (var_iqr + 1.e-60), var_ent);

    ++nvars;
  } // For all script lines

  fclose(fp);
  fp = NULL;

  printf("\nFinished processing variable script file.  Writing output file...");

  /*
  -------------------------------------------------------------------------------

     All variables are computed.  Write the output file.

  -------------------------------------------------------------------------------
  */

  if (fopen_s(&fp, "OUTVARS.TXT", "wt")) {
    printf("\n\nCannot open variable output file OUTVARS.TXT");
    goto FINISH;
  }

  // Write the header of variable names

  fprintf(fp, "Date");
  for (i = 0; i < nvars; i++) {
    fprintf(fp, " %s", var_names[i]);
    if (ferror(fp)) {
      printf("\nError writing variable output file OUTVARS.TXT");
      goto FINISH;
    }
  }

  // Write the cases

  for (icase = front_bad; icase < n_cases; icase++) {
    fprintf(fp, "\n%8d", market_date[0][icase]);
    for (i = 0; i < nvars; i++) {
      fprintf(fp, " %10.5lf", vars[i][icase]);
      if (ferror(fp)) {
        printf("\nError writing variable output file OUTVARS.TXT");
        goto FINISH;
      }
    } // For variables
  } // For cases

  fclose(fp);
  fp = NULL;
  printf("\nSuccess");

FINISH:

  if (fp != NULL)
    fclose(fp);

  printf("\n\nPress any key...");
  _getch(); // Wait for user to press a key

  for (i = 0; i < n_markets; i++) {
    if (market_date != NULL && market_date[i] != NULL)
      FREE(market_date[i]);
    if (market_open != NULL && market_close[i] != NULL)
      FREE(market_open[i]);
    if (market_high != NULL && market_close[i] != NULL)
      FREE(market_high[i]);
    if (market_low != NULL && market_close[i] != NULL)
      FREE(market_low[i]);
    if (market_close != NULL && market_close[i] != NULL)
      FREE(market_close[i]);
    if (market_volume != NULL && market_close[i] != NULL)
      FREE(market_volume[i]);
  }

  if (market_names != NULL)
    FREE(market_names);
  if (market_date != NULL)
    FREE(market_date);
  if (market_n != NULL)
    FREE(market_n);
  if (market_index != NULL)
    FREE(market_index);
  if (market_open != NULL)
    FREE(market_open);
  if (market_high != NULL)
    FREE(market_high);
  if (market_low != NULL)
    FREE(market_low);
  if (market_close != NULL)
    FREE(market_close);
  if (market_volume != NULL)
    FREE(market_volume);
  if (var_work != NULL)
    FREE(var_work);
  if (work1 != NULL)
    FREE(work1);
  if (work2 != NULL)
    FREE(work2);
  if (work3 != NULL)
    FREE(work3);
  if (big_work != NULL)
    FREE(big_work);
  if (big_work2 != NULL)
    FREE(big_work2);
  if (big_work3 != NULL)
    FREE(big_work3);
  if (iwork != NULL)
    FREE(iwork);
  for (i = 0; i < nvars; i++) {
    if (vars[i] != NULL)
      FREE(vars[i]);
  }

#if MEMDEBUG
  memclose();
#endif

  exit(0);
}
