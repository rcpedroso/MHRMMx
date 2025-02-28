#ifndef SM_AUXILIARY_H
#define SM_AUXILIARY_H

#include <RcppArmadillo.h>



/* b_tau2_SM */
double b_tau2_SM(const arma::mat theta, const arma::mat& mu,
                 const arma::vec btD, const arma::mat& K, const int d) {
  
  arma::vec theta_mu = theta.col(d) - mu.col(d);
  return btD(d) + 0.5*arma::as_scalar( theta_mu.t()*K*theta_mu );
  
}  



/* beta_theta_SM_s */
arma::vec beta_theta_SM_s(const arma::cube theta, const arma::cube& beta,
                      const arma::Row<int> z_split, arma::uvec zij,
                      const int i, const int j, arma::Col<int> S, const int nS,
                      const int d, const int zi, const int zj) {
  
  arma::vec beta_theta_s(2);
  arma::vec bi0 = beta.slice(d).col(i) - theta.slice(d).col(zi);
  beta_theta_s(0) = arma::as_scalar( bi0.t()*bi0 );
  arma::vec bj1 = beta.slice(d).col(j) - theta.slice(d).col(zj);
  beta_theta_s(1) = arma::as_scalar( bj1.t()*bj1 );
  for (int k = 0; k < nS; k++) {
    if (z_split(S(k)) == zij(0)) {
      arma::vec bk = beta.slice(d).col(S(k)) - theta.slice(d).col(zi);
      beta_theta_s(0) += arma::as_scalar( bk.t()*bk );
    } else {
      arma::vec bk = beta.slice(d).col(S(k)) - theta.slice(d).col(zj);
      beta_theta_s(1) += arma::as_scalar( bk.t()*bk );
    }
  }
  
  return beta_theta_s;
}  



/* beta_theta_SM_m */
double beta_theta_SM_m(const arma::mat theta, const arma::cube& beta,
                   const int i, const int j, arma::Col<int> S, const int nS,
                   const int d) {

  double beta_theta_m = 0;
  arma::vec bi = beta.slice(d).col(i) - theta.col(d);
  beta_theta_m += arma::as_scalar( bi.t()*bi );
  arma::vec bj = beta.slice(d).col(j) - theta.col(d);
  beta_theta_m += arma::as_scalar( bj.t()*bj );
  for (int k = 0; k < nS; k++) {
    arma::vec bk = beta.slice(d).col(S(k)) - theta.col(d);
    beta_theta_m += arma::as_scalar( bk.t()*bk );
  }
  
  return beta_theta_m;
}



/* beta_theta_SM_s */
arma::mat beta_SM_s(const arma::cube& beta, const arma::Row<int> z_split, arma::uvec zij,
                    const int i, const int j, arma::Col<int> S, const int nS,
                    const int d, const int P) {
  
  arma::mat beta_s(P,2);
  beta_s.col(0) = beta.slice(d).col(i);
  beta_s.col(1) = beta.slice(d).col(j);
  for (int k = 0; k < nS; k++) {
    if (z_split(S(k)) == zij(0)) {
      beta_s.col(0) += beta.slice(d).col(S(k));
    } else {
      beta_s.col(1) += beta.slice(d).col(S(k));
    }
  }
  
  return beta_s;
}  



arma::vec beta_SM_m(const arma::cube& beta, const int i, const int j,
                    arma::Col<int> S, const int nS, const int d) {
  
  // arma::vec beta_m(P);
  arma::vec beta_m = beta.slice(d).col(i) + beta.slice(d).col(j);
  for (int k = 0; k < nS; k++) beta_m += beta.slice(d).col(S(k));
  
  return beta_m;
}  




#endif
