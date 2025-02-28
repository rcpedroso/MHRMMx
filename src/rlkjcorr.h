#ifndef RLKJCORR_H
#define RLKJCORR_H


// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
// #include <RcppArmadilloExtensions/sample.h>

// using namespace arma;
using namespace Rcpp;



// f
???   f() {
  
  double alpha = eta + ((double)K-2)/2
  double r12 = 2*Rf_rbeta(alpha , alpha) - 1;
  
  arma::mat R(K,K);
  R.zeros();
  R(0,0) = 1;
  R(0,1) = r12;
  R(1,1) = sqrt(1-pow(r12,2));
  
  if (K > 2) {
    for (m in 2:(K - 1)) {
      alpha = alpha - 0.5;
      double y = Rf_rbeta((double)m/2 , alpha);
      
      // Draw uniformally on a hypersphere
      z <- rnorm(m, 0, 1)
        z <- z / sqrt(crossprod(z)[1])
        
        R[1:m,m+1] <- sqrt(y) * z
      R[m+1,m+1] <- sqrt(1 - y)
    } 
  }
  
}
  
  return(crossprod(R))
}



/* IG truncated sampler */
arma::mat rlkjcorr_cpp(int n, double D, double eta=1) {
  
  arma::mat R = replicate( n , f() )
  if ( dim(R)[3]==1 ) {
    R <- R[,,1]
  } else {
  // need to move 3rd dimension to front, so conforms to array structure that Stan uses
    R <- aperm(R,c(3,1,2))
  }
  return R;
    
}


#endif