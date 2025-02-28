#ifndef RCPPDIST_RMVNORM_ARMA_H
#define RCPPDIST_RMVNORM_ARMA_H

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

arma::rowvec rmvnorm_arma(const arma::vec m, 
                          const arma::mat V,
                          const double a,
                          const int P) {
  
  arma::mat S = V/a;
  arma::rowvec result(P);

  for (int l = 0; l < P; ++l) result(l) = R::rnorm(0.0, 1.0);
  result *= arma::chol(S)*sqrt(a);
  result += m.t();
  
  return result;
}



#endif