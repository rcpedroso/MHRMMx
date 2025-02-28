#ifndef RSIG_ARS_H
#define RSIG_ARS_H


// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>

using namespace Rcpp;


// zj function bellow is already defined in the gamma ars header
// /* tangent intersect points */
// double zj(double xj, double xj1) {
//   return (log(xj1)-log(xj)) / (1/xj-1/xj1);
// } 



/* ARS for individual-specific sigma sampling */
// [[Rcpp::export]]
double rsig_ars(double L, double Lu, int n, double a, double b) {
  
  double w;
  
  /* basic settings */
  double M = arma::datum::log_max;
  int k = 20;
  
  if (Lu <= 1/L) stop("\nError: Lu value less or equal to the lower bound\n");
  
  /* x (envelope tangent points) */
  arma::vec xinit = arma::linspace(1/L, Lu, k+2);
  arma::vec x = xinit.subvec(1,k); // remove xinit[0]=1/L and xinit[k+1]=Lu
  
  
  /* z (envelope partition points) */
  arma::vec z(k-1);
  for (int j = 0; j < k-1; j++) z(j) = zj(x(j),x(j+1));
  
  
  
  /* ARS */
  RNGScope scope;
  
  int accept = 0;
  while (accept == 0) {
    
    /* pj */
    arma::vec logA(k);
    double logAmax = -arma::datum::inf;
    
    for (int j = 0; j < k; j++) {
      double xj = x(j);
      
      double zl;
      if (j == 0) {
        zl = 1/L;
      } else {
        zl = z(j-1);
      }
      
      double zu;
      if (j == k-1) {
        zu = Lu;
      } else {
        zu = z(j);
      }
      
      // double dhj = (a-1)/xj - b;
      double dhj = (n-2)/xj - a*xj - b;
      
      if (dhj > 0) {
        // logA(j) = (a-1)*(log(xj)-1) + zu*dhj - log(dhj) + log(1-exp(-(zu-zl)*dhj));
        logA(j) = (n-2)*(log(xj)-1) + 0.5*a*xj*xj + zu*dhj + log(1-exp(-(zu-zl)*dhj)) - log(dhj);
      } else {
        // logA(j) = (a-1)*(log(xj)-1) + zl*dhj + log(1-exp((zu-zl)*dhj)) - log(-dhj);
        logA(j) = (n-2)*(log(xj)-1) + 0.5*a*xj*xj + zl*dhj + log(1-exp((zu-zl)*dhj)) - log(-dhj);
      }
      
      if (logA(j) > logAmax) logAmax = logA(j);
      
      // Rprintf("j = %i\n", j);
      // Rprintf("xj = %f\n", xj);
      // Rprintf("zl = %f\n", zl);
      // Rprintf("zu = %f\n", zu);
      // Rprintf("dhj(%i) = %f\n", j,dhj);
      // Rprintf("logA(%i) = %f\n\n", j,logA(j));
      
    }
    
    // double maxlog = M - logAmax - log(k);
    // arma::vec A = exp(logA + maxlog);
    // arma::vec pj = A/sum(A);
    arma::vec A = exp(logA - logAmax + M - log(k));
    arma::vec pj = A/sum(A);
    // Rprintf("maxlog = %f\n", maxlog);
    // for (int j = 0; j < k; j++) Rprintf("logA(%i) = %f\n", j,logA(j));
    // for (int j = 0; j < k; j++) Rprintf("A(%i) = %f\n", j,A(j));
    
    
    
    /* A(j) sample */
    arma::uvec kvec = arma::linspace<arma::uvec>(0, k-1, k);
    int j = RcppArmadillo::sample(kvec, 1, true, pj).at(0);
    
    
    
    /* xstar sample */    
    double xj = x(j);
    
    double zl;
    if (j == 0) {
      zl = 1/L;
    } else {
      zl = z(j-1); 
    }
    
    double zu;
    if (j == k-1) {
      zu = Lu;
    } else {
      zu = z(j);
    }
    
    double dhj = (n-2)/xj - a*xj - b;
    
    double xstar = zl;
    while (xstar == zl) {
      
      double u = Rf_runif(0.0,1.0);
      
      if (dhj > 0) {
        double ld = (zu-zl)*dhj;
        xstar = zl + (ld + log(u*(1-exp(-ld))+exp(-ld)))/dhj;
      } else {
        double ld = (zu-zl)*dhj;
        xstar = zl + log(u*(exp(ld)-1)+1)/dhj;
      }
      
      // if (xstar == arma::datum::inf) {
      //   Rprintf("j = %i\n", j);
      //   Rprintf("xj = %f\n", xj);
      //   Rprintf("zl = %f\n", zl);
      //   Rprintf("zu = %f\n", zu);
      //   Rprintf("dhj = %f\n", dhj);
      //   Rprintf("u = %f\n\n", u);
      //   stop("xstar = inf");
      // }
    }
    
    
    if (xstar > zu) stop("\nError: ARS proposal greater than upper bound (xstar > zu)\n");
    // double r = (a-1) * ( log(xstar)-log(xj) - (xstar-xj)/xj );
    double r = (n-2)*( log(xstar)-log(xj) - (xstar-xj)/xj ) +
      0.5*a*( xj*xj - xstar*xstar ) + a*(xstar-xj)*xj;
    
    // double r = -arma::datum::inf;
    // if (xstar <= zu) r = (a-1)*(log(xstar)-log(xj)) + (xj-xstar)*(a-1)/xj;
    
    double u = Rf_runif(0.0,1.0);
    
    if (log(u) <= r) {
      
      w = xstar;
      accept = 1;
      
    } else {
      
      if (xstar != x(j)) { // possibly unnecessary
        
        arma::vec xnew(k+1);
        
        if (xstar > x(j)) {
          
          xnew(j) = x(j);
          xnew(j+1) = xstar;
          
        } else {
          
          xnew(j) = xstar;
          xnew(j+1) = x(j);
          
        }
        
        if (j == 0) {
          
          for (int i = 1; i < k; i++) xnew(i+1) = x(i);
          
        } else if (j < k) {
          
          for (int i = 0; i < j; i++) xnew(i) = x(i);
          for (int i = j+1; i < k; i++) xnew(i+1) = x(i);
          
        } else { // j=k
          
          for (int i = 0; i < k-1; i++) xnew(i) = x(i);
          
        }
        
        // update x, k and z
        x = xnew;
        k = k + 1;
        z = arma::vec(k-1);
        for (int j = 0; j < k-1; j++) z(j) = zj(x(j), x(j+1));
        
      } // end if possibly unnecessary
      
    } // end if
    
  } // end while
  
  return w;
  
}


#endif