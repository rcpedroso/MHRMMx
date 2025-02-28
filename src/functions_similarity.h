#ifndef SIM_FUNCTION_H
#define SIM_FUNCTION_H

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;


const double log2pi = log(2*arma::datum::pi);


/* similarity function for continuous variables */
double gsim_con(int nj, int v0, double sumxj, double sumx2j) {
  double vj = 1/((double)nj + 1/(double)v0);
  double log_con = 0.5*log(vj) - log(v0) -
    0.5*( (double)nj*log2pi - sumx2j + vj*sumxj*sumxj );
  return log_con;
}


/* similarity function ratio for continuous variables */
double gsim_con_gibbs(int nj, int v0, double xi, double xi2, double sumxj) {
  double vj = 1/((double)nj + 1 + 1/(double)v0);
  double vj_i = 1/((double)nj + 1/(double)v0);
  double log_con = 0.5*( log(vj) - log(vj_i) - log2pi ) - 
    0.5*( xi2 - vj*(sumxj+xi)*(sumxj+xi) + vj_i*sumxj*sumxj );
  return log_con;
}


/* similarity function for categorical variables */
double gsim_cat(int nj, double alpha_cat, int ncat, arma::Col<int> ncatj) {
  
  double log_cat = 0;
  
  for (int l = 0; l < ncat; l++) {
    for (int k = 1; k <= ncatj(l); k++) {
      log_cat += log( (double)ncatj(l) + alpha_cat - double(k) );
    }
  }
  
  for (int k = 1; k <= nj; k++) {
    log_cat -= log( (double)nj + (double)ncat*alpha_cat - double(k) );
  }

  return log_cat;
}


/* similarity function ratio for categorical variables */
double gsim_cat_gibbs(int nj, double alpha_cat, int ncat, int ncatj) {
  double log_cat = log((double)ncatj + alpha_cat) - log((double)nj + (double)ncat*alpha_cat);
  return log_cat;
}


#endif