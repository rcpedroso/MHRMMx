#ifndef H_MAT_H
#define H_MAT_H

#include <RcppArmadillo.h>


/* B-spline basis functions */
double B(int j, double x, int p, int q, arma::vec k) {
  
  double b = 0;
  
  if (q == 0) {
    if (k(j) <= x && x < k(j+1)) b = 1;
  } else {
    b = (x-k(j))/(k(j+q)-k(j))*B(j,x,p,q-1,k) + (k(j+q+1)-x)/(k(j+q+1)-k(j+1))*B(j+1,x,p,q-1,k);
  }
  
  return b;
}




/* Hmat */
arma::mat Hmat(int p, int q, int n) {
  
  // vector of values at which the
  // B-spline basis functions are to be evaluated
  // seq(0,1,length.out=n)
  arma::vec x(n);
  for (int t = 0; t < n; t++) x(t) = (double)t/((double)n-1);
  
  // knots
  int nk = p+1+2*q;
  arma::vec knots(nk);
  double xmin = min(x);
  double xmax = max(x);
  double dx = (xmax-xmin)/(double)p;
  for (int j = 0; j < nk; j++) knots(j) = xmin-((double)q-(double)j)*dx;
  
  // design matrix H
  int P = p + q;
  arma:: mat H(n,P);
  for (int t = 0; t < n; t++) {
    for (int j = 0; j < P; j++) {
      H(t,j) = B(j,x(t),p,q,knots);
    }
  }
  
  return H;
}



/* Kmat */
// [[Rcpp::export]]
arma::mat Kmat(int P, double v) {
  
  arma::mat D = arma::zeros(P-1,P);
  for (int l = 0; l < P-1; l++) { D.submat(l,l,l,l+1) = {-1,1}; }
  arma::mat K = D.t()*D;
  K(0,0) = 1 + v*v;
  
  return K;
}


#endif



