#ifndef RCPPDIST_RMVNORM_EIGEN_H
#define RCPPDIST_RMVNORM_EIGEN_H

#include <RcppArmadillo.h>
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppArmadillo)]]

arma::rowvec rmvnorm_eigen(const arma::vec m,
                           arma::mat V,
                           const double a,
                           const int P) {
  
  Eigen::MatrixXd S(P,P);
  Eigen::MatrixXd eigen_L(P,P);
  arma::rowvec result(P);
  
  S = Eigen::Map<Eigen::MatrixXd>(V.memptr(),P,P);
  S = S/a;

  Eigen::LLT<Eigen::MatrixXd> eigen_LLT(S); 
  eigen_L = eigen_LLT.matrixL();
  
  arma::mat arma_L = arma::mat(eigen_L.data(),
                               eigen_L.rows(),
                               eigen_L.cols(),
                               false, false);

  for (int l = 0; l < P; l++) result(l) = R::rnorm(0.0, 1.0);
  result *= arma_L*sqrt(a);
  result += m.t();
  
  return result;
  
}



#endif