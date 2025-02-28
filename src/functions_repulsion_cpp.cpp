
#include <Rcpp.h>

/* repulsive function based on Petralia (2012) */
// [[Rcpp::export]]
double g1_log(double dist, double phi, double nu) {
  return -phi/pow(dist,nu);
}

/* repulsive function based on Quinlan (2021) */
// [[Rcpp::export]]
double g2_log(double dist, double phi, double nu) {
  return log(1-exp(-pow(dist,nu)/phi));
}

