#ifndef UPDATE_THETA_H
#define UPDATE_THETA_H

#include <RcppArmadillo.h>
#include "functions_repulsion.h"

#include <mvnorm.h>
// [[Rcpp::depends(RcppDist)]]



/* update_theta */
void update_theta(arma::cube& theta,
                  arma::cube& Vadapt,
                  arma::Mat<int>& accept_mat,
                  arma::mat& accept_logratio_mat,
                  const arma::Col<int> nj,
                  const arma::mat lam2,
                  const arma::mat tau2,
                  const arma::cube beta,
                  const arma::mat mu,
                  const arma::Row<int> z,
                  const arma::mat K,
                  const arma::mat Ki,
                  const int m,
                  const int d,
                  const int J,
                  const int P,
                  const arma::mat Ip,
                  const int n_theta,
                  const arma::mat Ht0,
                  const int theta_sampler,
                  const int repulsive,
                  const double phi,
                  const double nu,
                  const double r,
                  const double nr,
                  const double eps,
                  const double eps2,
                  const int b,
                  const int burn) {

  arma::mat beta_clr(P,J);
  beta_clr.zeros();
  
  // Rprintf("\n");
  // Rprintf("theta_sampler\n");
  // Rprintf("max(z) = %i\n", arma::max(z));
  // Rprintf("min(nj=0) = %i\n", arma::min( arma::find(nj==0) ));
  
  for (int i = 0; i < m; i++) {
    // Rprintf("d = %i , i = %i\n", d,i);
    // Rprintf("z(%i) = %i , nj(%i) = %i\n", i,z(i),z(i),nj(z(i)));
    beta_clr.col(z(i)) += beta.slice(d).col(i);
  }
  

  if (repulsive != 0) accept_mat.col(d).zeros();
  

  RNGScope scope;
  
  
  for (int j = 0; j < J; j++) {
    
    arma::vec m_j(P);
    arma::mat V_j(P,P);
    arma::mat Vi_j(P,P);
    
    if (nj(j) > 0) {
      
      Vi_j = K/tau2(j,d) + (double)nj(j)*Ip/lam2(j,d);
      V_j = arma::inv( Vi_j );
      m_j = V_j * ( K*mu.col(d)/tau2(j,d) + beta_clr.col(j)/lam2(j,d) );
      
    } else {
      
      Vi_j = K/tau2(j,d);
      V_j = tau2(j,d)*Ki;
      m_j = mu.col(d);

    }
    

    if (repulsive == 0) {
      

      theta.slice(d).col(j) = rmvnorm(1, m_j , V_j).t();


    } else {
      
      
      arma::vec theta_prop(P);
      double log_ratio;
      
      // column j of theta matrix
      arma::vec theta_j = theta.slice(d).col(j);
      
      // theta matrix without column j
      arma::mat theta_ = theta.slice(d);
      theta_.shed_col(j);
      
      // difference between current curve f(theta_j) and the other curves
      arma::mat df(n_theta,J-1);
      for (int h = 0; h < J-1; h++) df.col(h) = Ht0*(theta_j - theta_.col(h));
      
      
      if (theta_sampler == 1) { // ARWMH (Quinlan etal 2021)
        
        // proposal sample
        if (b <= burn) {
          // theta_prop = rmvnorm(1, theta_j , V_j).t();
          theta_prop = rmvnorm(1, theta_j , eps*V_j).t();
          Vadapt.slice(j) +=  V_j/burn;
        } else {
          theta_prop = rmvnorm(1, theta_j , eps*Vadapt.slice(j)).t();
        }
        
        // difference between curve of theta_prop and other curves
        arma::mat df_prop(n_theta,J-1);
        for (int h = 0; h < J-1; h++) df_prop.col(h) = Ht0*(theta_prop - theta_.col(h));
        
        // log(FRc) value for the proposed theta_prop value
        double log_FRc_prop = logFRc(repulsive, J, phi, nu, r, n_theta, df_prop);
        
        // log(FRc) value of the current theta_j value
        double log_FRc = logFRc(repulsive, J, phi, nu, r, n_theta, df);
        
        // log-ratio of the Normal posterior densities in the RWMH ratio
        double log_ind = -0.5 *
          as_scalar( theta_prop.t()*Vi_j*(theta_prop - 2*m_j) - theta_j.t()*Vi_j*(theta_j - 2*m_j) );
        
        // acceptance log-ratio
        log_ratio = log_ind + log_FRc_prop - log_FRc;
        
        
        
      } else if (theta_sampler == 2) { // MALA
        
        // gradient objects
        arma::vec D_ind(P);
        arma::vec D_rep(P);
        arma::vec D_ind_prop(P);
        arma::vec D_rep_prop(P);
        
        // gradient of the log-posterior density of theta_j (independent factor)
        D_ind = -((theta_j-m_j).t()*Vi_j).t();
        
        // gradient of the log-posterior density of theta_j (repulsive factor)
        D_rep = D_logFRc(repulsive, J, P, nr, phi, nu, r, Ht0, n_theta, df);
        
        // proposal value to replace theta_j
        arma::vec m_mala = theta_j + eps2*(D_ind+D_rep)/2;
        theta_prop = rmvnorm(1, m_mala , eps2*Ip).t();
        
        // difference between current curve of theta_prop and other curves
        arma::mat df_prop(n_theta,J-1);
        for (int h = 0; h < J-1; h++) df_prop.col(h) = Ht0*(theta_prop - theta_.col(h));
        
        // log-posterior distribution of theta_j
        double log_f = as_scalar( -0.5 * (theta_j-m_j).t() * Vi_j * (theta_j-m_j) );
        double log_FRc = logFRc(repulsive, J, phi, nu, r, n_theta, df);
        
        // log-posterior distribution of theta_prop
        double log_f_prop = as_scalar( -0.5 * (theta_prop-m_j).t() * Vi_j * (theta_prop-m_j) );
        double log_FRc_prop = logFRc(repulsive, J, phi, nu, r, n_theta, df_prop);
        
        // log-proposal distribution of theta_prop
        double log_q_prop = as_scalar( -0.5 * (theta_prop-m_mala).t() * (theta_prop-m_mala) / eps2);
        
        // log-proposal distribution of theta_j
        D_ind_prop = -((theta_prop-m_j).t()*Vi_j).t();
        D_rep_prop = D_logFRc(repulsive, J, P, nr, phi, nu, r, Ht0, n_theta, df_prop);
        arma::vec m_mala_prop = theta_prop + eps2*(D_ind_prop+D_rep_prop)/2;
        double log_q = as_scalar( -0.5 * (theta_j-m_mala_prop).t() * (theta_j-m_mala_prop) / eps2);
        
        // acceptance log-ratio
        log_ratio = (log_f_prop + log_FRc_prop) - (log_f + log_FRc) + (log_q - log_q_prop);

      }
      
      // Rprintf("b = %i , log_ind(%i) = %f\n", b, j, log_ind);
      // Rprintf("b = %i , logFRc_prop(%i) = %f\n", b, j, logFRc_prop);
      // Rprintf("b = %i , logFRc_j(%i) = %f\n", b, j, logFRc_j);
      // Rprintf("b = %i , log_ratio(%i) = %f\n", b, j, log_ratio);
      
      /* acceptance or rejection procedure*/
      accept_logratio_mat(j,d) = log_ratio;
      double u = Rf_runif(0.0,1.0);
      
      if (log(u) < log_ratio) {
        theta.slice(d).col(j) = theta_prop;
        accept_mat(j,d) = 1;
      }
      
    } // endif repulsive
    

  } // end for j


}  



#endif
