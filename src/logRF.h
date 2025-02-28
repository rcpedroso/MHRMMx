#ifndef LOGRF_H
#define LOGRF_H

#include <RcppArmadillo.h>
#include "functions_repulsion.h"


/* logRF */
double logRF(int repulsive, double phi, double nu, double r, int n, arma::mat df) {
  
  double sum_log_g = 0;

  // absolute values of the differences
  arma::mat df_abs = arma::abs(df);
  
  if (repulsive == 1) { // repulsive function based on Petralia etal (2012)
    
    for (int h = 0; h < df.n_cols; h++) {
      double dist = arma::as_scalar( arma::pow( sum(arma::pow(df_abs.col(h),r))/(double)n , 1/r) );
      sum_log_g += g1_log(dist, phi, nu);
    }
    
  } else if (repulsive == 2) { // repulsive function based on Quinlan etal (2021)
    
    for (int h = 0; h < df.n_cols; h++) {
      double dist = arma::as_scalar( arma::pow( sum(arma::pow(df_abs.col(h),r))/(double)n , 1/r) );
      sum_log_g += g2_log(dist, phi, nu);
    }
    
  }
  
  return sum_log_g;
}




// /* 1st derivative of log(FRc) */
// arma::vec D_logFRc(int repulsive, int J, int P,
//                    double nr, double phi, double nu, double r,
//                    arma::mat Ht0, int n, arma::mat df) {
//   
//   arma::vec D(P);
//   D.zeros();
//   
//   // absolute values of the differences between the
//   // current or proposal curves and the other curves
//   const arma::mat df_abs = arma::abs(df);
//   const arma::mat df_abs_r = arma::pow(df_abs,r);
//   
//   
//   if (repulsive == 1) { // Petralia etal (2012)
//     
//     for (int h = 0; h < J-1; h++) {
//       
//       double d = pow( sum(df_abs_r.col(h))/(double)n , 1/r );
//       arma::vec Sh_ = arma::pow( df_abs.col(h) , r-1 ) % arma::sign(df.col(h));
//       
//       for (int l = 0; l < P; l++) {
//         arma::vec Sh = Sh_ % Ht0.col(l);
//         D(l) = pow( sum(df_abs_r.col(h)) , 1/r-1 ) * sum(Sh) / pow(d,nu+1);
//       }
//       
//     }
//     
//     D *= phi*nu/nr;
//     
//     
//   } else if (repulsive == 2) { // Quinlan etal (2021)
//     
//     for (int h = 0; h < J-1; h++) {
//       
//       double d = pow( sum(df_abs_r.col(h))/(double)n , 1/r );
//       double d_nu = pow(d,nu);
//       arma::vec Sh_ = arma::pow( df_abs.col(h) , r-1 ) % arma::sign(df.col(h));
//       
//       for (int l = 0; l < P; l++) {
//         arma::vec Sh = Sh_ % Ht0.col(l);
//         D(l) += (d_nu/d) * pow( sum(df_abs_r.col(h)) , 1/r-1 ) * sum(Sh) / (exp(d_nu/phi)-1);
//       }
//       
//     }
//     
//     D *= nu/phi/nr;
//     
//   }
//   
//   return D;
// }
// 
// 
// 
// 
// /* 2nd derivative of log(FRc) */
// arma::mat D2_logFRc(const int repulsive, const int J, const int P, double nr,
//                     const double phi, const double nu, const double r, arma::mat Ht0,
//                     const int n, const arma::mat df) {
//   
//   
//   arma::mat D2(P,P);
//   D2.zeros();
//   
//   // absolute values of the differences between current or proposal curves and the other curves
//   const arma::mat df_abs = arma::abs(df);
//   const arma::mat df_abs_r = arma::pow(df_abs,r);
//   
//   for (int h = 0; h < J-1; h++) {
//     
//     double sum_df_abs_r = sum(df_abs_r.col(h));
//     double d = pow( sum_df_abs_r/(double)n , 1/r );
//     double d_nu = pow( d , nu );
//     double exp_d_nu = exp(d_nu/phi);
//     double F0 = d_nu / (d*d*(exp_d_nu-1)) * (nu - 1 - nu/phi*d_nu*exp_d_nu/(exp_d_nu-1));
//     
//     double F1 = (d_nu/d) / (exp_d_nu-1);
//     // Rprintf("h=%i , F1=%f\n", h, F1);
//     
//     arma::vec F2(P);
//     F2.zeros();
//     arma::vec Sh = arma::pow( df_abs.col(h) , r-1 ) % arma::sign(df.col(h));
//     
//     for (int l = 0; l < P; l++) {
//       
//       // F2(l) = pow( sum_df_abs_r , 1/r-1 ) / nr * sum( Sh % Ht0.col(l) );
//       
//       for (int c = l; c < P; c++) {
//         
//         F2(c) = pow( sum_df_abs_r , 1/r-1 ) / nr * sum( Sh % Ht0.col(c) );
//         
//         D2(l,c) += (F0 * F2(c) * F2(l)) +
//           F1 / nr *  pow( sum_df_abs_r , 1/r-2 ) * (1-r) *
//           ( sum( Sh % Ht0.col(c) ) *  sum( Sh % Ht0.col(l) ) -
//           sum_df_abs_r * sum( Ht0.col(l) % Ht0.col(c) % arma::pow( df_abs.col(h) , r-2 ) )
//           );
//         
//         D2(c,l) = D2(l,c);
//         // Rprintf("h=%i , D2(%i,%i)=%f\n", h, l, c, D2(l,c));
//       }
//       Rprintf("h=%i , D2(%i,%i)=%f\n", h, l, l, D2(l,l));
//       
//     }
//     
//   }
//   
//   return nu/phi * D2;
// }


#endif
