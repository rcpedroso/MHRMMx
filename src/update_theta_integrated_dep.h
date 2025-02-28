#ifndef UPDATE_THETA_INTEGRATED_DEP_H
#define UPDATE_THETA_INTEGRATED_DEP_H

#include <RcppArmadillo.h>
#include "functions_repulsion.h"

#include <mvnorm.h>
// [[Rcpp::depends(RcppDist)]]



/* update_theta */
void update_theta_dep(arma::cube& theta,
                      arma::cube& Winv,
                      arma::field<arma::mat>& WH,
                      arma::cube& WHH,
                      arma::Col<int>& accept_vec,
                      arma::vec& accept_logratio_vec,
                      const arma::Col<int> nj,
                      const arma::mat lam2,
                      const arma::mat tau2,
                      const arma::mat beta0,
                      const arma::mat sig2,
                      const arma::mat mu,
                      const arma::Row<int> z,
                      const arma::mat K,
                      const arma::mat Ki,
                      const int m,
                      const int D,
                      const int J,
                      const int P,
                      const int PD,
                      const arma::mat Ip,
                      const Rcpp::List Y,
                      const int n_theta,
                      const arma::mat Ht0,
                      const int theta_sampler,
                      const int repulsive,
                      const double phi,
                      const double nu,
                      const double r,
                      const double nr,
                      const double eps,
                      const double eps0) {
  
  arma::cube sumV(PD,PD,J);
  arma::mat sumVH(PD,J);
  sumV.zeros();
  sumVH.zeros();
  
  // Rprintf("\n");
  // Rprintf("theta_sampler\n");
  // Rprintf("max(z) = %i\n", arma::max(z));
  // Rprintf("min(nj=0) = %i\n", arma::min( arma::find(nj==0) ));
  
  for (int i = 0; i < m; i++) {
    // Rprintf("i = %i (theta update)\n", i);
    // Rprintf("z(%i) = %i , nj(%i) = %i\n", i,z(i),z(i),nj(z(i)));
    
    // arma::vec yi = Yi.col(d) - beta0(i,d);
    // arma::mat V = arma::inv( HHi/sig2(i,d) + Ip/lam2(z(i),d) );
    // sumV.slice(z(i)) += V;
    // sumVH.col(z(i)) += V*Hi.t()*yi/sig2(i,d);
    
    // individual-specific objects
    // arma::mat Hi = H[i];
    // arma::mat HHi = HH.slice(i);
    arma::mat Yi = Y[i];
    arma::vec yi = (Yi - arma::ones(Yi.n_rows)*beta0.row(i)).as_col();
    arma::mat Wi_inv = Winv.slice(i);
    // arma::mat WH = kron(Wi_inv,Hi.t());
    // arma::mat WHH = kron(Wi_inv,HHi);
    
    // Rprintf("1 (theta update)\n");

    // cluster-specific objects
    arma::mat lj_inv(D,D, arma::fill::zeros);
    arma::mat tj_inv(D,D, arma::fill::zeros);
    for (int d = 0; d < D; d++) {
      lj_inv(d,d) = 1/lam2(z(i),d);
      tj_inv(d,d) = 1/tau2(z(i),d);
    }
    arma::mat Lj_inv = kron(lj_inv,Ip);
    arma::mat Tj_inv = kron(tj_inv,K);
    
    // Rprintf("2 (theta update) (i=%i)\n", i);

    // Vi sum
    // arma::mat Vi = arma::inv( WHH.slice(i) + Lj_inv );
    arma::mat Vi = inv_sympd( WHH.slice(i) + Lj_inv );
    sumV.slice(z(i)) += Vi;
    sumVH.col(z(i)) += Vi*WH[i]*yi;
    
    // Rprintf("3 (theta update)\n");
    
  }
  // Rprintf("theta update (for i)\n");
  

  if (repulsive != 0) accept_vec.zeros();
  

  RNGScope scope;

  
  /////* theta update */////
  for (int j = 0; j < J; j++) {
    
    arma::vec m_j(P);
    arma::mat V_j(P,P);
    arma::mat Vi_j(P,P);

    if (nj(j) > 0) {
      

      // // Vi_j = K/tau2(j,d) + (double)nj(j)*Ip/lam2(j,d) - sumV.slice(j)/lam2(j,d)/lam2(j,d);
      // // V_j = arma::inv( Vi_j );
      // // m_j = V_j * ( K*mu.col(d)/tau2(j,d) + sumVH.col(j)/lam2(j,d) );
      // 
      // Vi_j = lam2(j,d)*K/tau2(j,d) + (double)nj(j)*Ip - sumV.slice(j); // Vi_j * lam2(j,d)
      // V_j = arma::inv( Vi_j ) * lam2(j,d);
      // m_j = V_j * ( K*mu.col(d)/tau2(j,d) + sumVH.col(j) );
      // 
      // Vi_j = Vi_j/lam2(j,d); // correct Vi_j

      arma::mat lj_inv(D,D, arma::fill::zeros);
      arma::mat tj_inv(D,D, arma::fill::zeros);
      for (int d = 0; d < D; d++) {
        lj_inv(d,d) = 1/lam2(j,d);
        tj_inv(d,d) = 1/tau2(j,d);
      }
      arma::mat Lj_inv = kron(lj_inv,Ip);
      arma::mat Tj_inv = kron(tj_inv,K);
      
      Vi_j = Tj_inv + (double)nj(j)*Lj_inv - Lj_inv*sumV.slice(j)*Lj_inv;
      V_j = arma::inv( Vi_j );
      m_j = V_j * ( Tj_inv*mu.as_col() + Lj_inv*sumVH.col(j) );


    } else {
      

      // Vi_j = K/tau2(j,d);
      // V_j = tau2(j,d)*Ki;
      // m_j = mu.col(d);
      
      arma::mat tj(D,D, arma::fill::zeros);
      arma::mat tj_inv(D,D, arma::fill::zeros);
      for (int d = 0; d < D; d++) {
        tj(d,d) = tau2(j,d);
        tj_inv(d,d) = 1/tau2(j,d);
      }
      arma::mat Tj = kron(tj,Ki);
      arma::mat Tj_inv = kron(tj_inv,K);
      
      Vi_j = Tj_inv;
      V_j = Tj;
      m_j = mu.as_col();
      
      
    }
    

    if (repulsive == 0) {
      
      
      arma::vec theta_vec = rmvnorm(1, m_j , V_j).t();
      int pos = -1;
      for (int d = 0; d < D; d++) {
        for (int p = 0; p < P; p++) {
          pos += 1;
          theta(p,j,d) = theta_vec(pos);
        }
      }
      

    } else {
      

      // arma::vec theta_prop(P);
      arma::mat theta_prop(P,D);
      double log_ratio = 0;
      
      // // column j of theta matrix
      // arma::vec theta_j = theta.slice(d).col(j);
      // matrix j of theta cube
      arma::mat theta_j = theta.col_as_mat(j);
      
      // // theta matrix without column j
      // arma::mat theta_ = theta.slice(d);
      // theta_.shed_col(j);
      // theta cube without column j
      arma::cube theta_ = theta;
      theta_.shed_col(j);
      
      // // difference between current curve f(theta_j) and the other curves
      // arma::mat df(n_theta,J-1);
      // for (int h = 0; h < J-1; h++) df.col(h) = Ht0*(theta_j - theta_.col(h));
      // difference between current curves f(theta_j) and the other curves
      arma::cube df(n_theta,J-1,D);
      for (int h = 0; h < J-1; h++) {
        for (int d = 0; d < D; d++) {
          df.slice(d).col(h) = Ht0*(theta_j.col(d) - theta_.slice(d).col(h)); 
        }
      }
      
      
      if (theta_sampler == 1) { // ARWMH (Quinlan etal 2021)

        if (nj(j) > 0) {
          for (int d = 0; d < D; d++) {
            theta_prop.col(d) = rmvnorm(1, theta_j.col(d) , eps*Ki).t();  
          }
        } else {
          for (int d = 0; d < D; d++) {
            theta_prop.col(d) = rmvnorm(1, theta_j.col(d) , eps0*Ki).t();  
          }
        }


        // // difference between curve of theta_prop and other curves
        // arma::mat df_prop(n_theta,J-1);
        // for (int h = 0; h < J-1; h++) df_prop.col(h) = Ht0*(theta_prop - theta_.col(h));
        // difference between current curves f(theta_prop) and the other curves
        arma::cube df_prop(n_theta,J-1,D);
        for (int h = 0; h < J-1; h++) {
          for (int d = 0; d < D; d++) {
            df_prop.slice(d).col(h) = Ht0*(theta_prop.col(d) - theta_.slice(d).col(h)); 
          }
        }
        
        // log(FRc) value for the proposed theta_prop value
        // double log_FRc_prop = logFRc(repulsive, J, phi, nu, r, n_theta, df_prop);
        arma::vec log_FRc_prop(D);
        for (int d = 0; d < D; d++) log_FRc_prop(d) = logFRc(repulsive, J, phi, nu, r, n_theta, df_prop.slice(d));
        
        // log(FRc) value of the current theta_j value
        // double log_FRc = logFRc(repulsive, J, phi, nu, r, n_theta, df);
        arma::vec log_FRc(D);
        for (int d = 0; d < D; d++) log_FRc(d) = logFRc(repulsive, J, phi, nu, r, n_theta, df.slice(d));
        
        // log-ratio of the Normal posterior densities in the RWMH ratio
        // double log_ind = -0.5 *
        //   as_scalar( theta_prop.t()*Vi_j*(theta_prop - 2*m_j) - theta_j.t()*Vi_j*(theta_j - 2*m_j) );
        double log_ind = -0.5 *
          as_scalar( theta_prop.as_col().t()*Vi_j*(theta_prop.as_col() - 2*m_j) - theta_j.as_col().t()*Vi_j*(theta_j.as_col() - 2*m_j) );
        
        // acceptance log-ratio
        log_ratio = log_ind;
        for (int d = 0; d < D; d++) log_ratio +=  log_FRc_prop(d) - log_FRc(d);
        
        
        
      // } else if (theta_sampler == 2) { // MALA
      //   
      //   // gradient objects
      //   arma::vec D_ind(P);
      //   arma::vec D_rep(P);
      //   arma::vec D_ind_prop(P);
      //   arma::vec D_rep_prop(P);
      //   
      //   // gradient of the log-posterior density of theta_j (independent factor)
      //   D_ind = -((theta_j-m_j).t()*Vi_j).t();
      //   
      //   // gradient of the log-posterior density of theta_j (repulsive factor)
      //   D_rep = D_logFRc(repulsive, J, P, nr, phi, nu, r, Ht0, n_theta, df);
      //   
      //   
      //   // proposal value to replace theta_j
      //   double eps2;
      //   if (nj(j) > 0) {
      //     eps2 = eps*eps;
      //   } else {
      //     eps2 = eps0*eps0;
      //   }
      //   arma::vec m_mala = theta_j + eps2*(D_ind+D_rep)/2;
      //   theta_prop = rmvnorm(1, m_mala , eps2*Ip).t();
      // 
      //   
      //   // difference between current curve of theta_prop and other curves
      //   arma::mat df_prop(n_theta,J-1);
      //   for (int h = 0; h < J-1; h++) df_prop.col(h) = Ht0*(theta_prop - theta_.col(h));
      //   
      //   // log-posterior distribution of theta_j
      //   double log_f = as_scalar( -0.5 * (theta_j-m_j).t() * Vi_j * (theta_j-m_j) );
      //   double log_FRc = logFRc(repulsive, J, phi, nu, r, n_theta, df);
      //   
      //   // log-posterior distribution of theta_prop
      //   double log_f_prop = as_scalar( -0.5 * (theta_prop-m_j).t() * Vi_j * (theta_prop-m_j) );
      //   double log_FRc_prop = logFRc(repulsive, J, phi, nu, r, n_theta, df_prop);
      //   
      //   // log-proposal distribution of theta_prop
      //   double log_q_prop = as_scalar( -0.5 * (theta_prop-m_mala).t() * (theta_prop-m_mala) / eps2);
      //   
      //   // log-proposal distribution of theta_j
      //   D_ind_prop = -((theta_prop-m_j).t()*Vi_j).t();
      //   D_rep_prop = D_logFRc(repulsive, J, P, nr, phi, nu, r, Ht0, n_theta, df_prop);
      //   arma::vec m_mala_prop = theta_prop + eps2*(D_ind_prop+D_rep_prop)/2;
      //   double log_q = as_scalar( -0.5 * (theta_j-m_mala_prop).t() * (theta_j-m_mala_prop) / eps2);
      //   
      //   // acceptance log-ratio
      //   log_ratio = (log_f_prop + log_FRc_prop) - (log_f + log_FRc) + (log_q - log_q_prop);

      }
      
      // Rprintf("b = %i , log_ind(%i) = %f\n", b, j, log_ind);
      // Rprintf("b = %i , logFRc_prop(%i) = %f\n", b, j, logFRc_prop);
      // Rprintf("b = %i , logFRc_j(%i) = %f\n", b, j, logFRc_j);
      // Rprintf("b = %i , log_ratio(%i) = %f\n", b, j, log_ratio);
      
      /* acceptance or rejection procedure*/
      accept_logratio_vec(j) = log_ratio;
      double u = Rf_runif(0.0,1.0);
      
      if (log(u) < log_ratio) {
        for (int d = 0; d < D; d++) theta.slice(d).col(j) = theta_prop.col(d);
        accept_vec(j) = 1;
      }
      
    } // endif repulsive
    

  } // end for j


}  



#endif
