#ifndef SM_AUXILIARY_THETA_H
#define SM_AUXILIARY_THETA_H

#include <RcppArmadillo.h>


/* sumV_SM_s */
Rcpp::List sumV_SM_s(const arma::mat lam2_s,
                     const arma::Row<int> z_split,
                     arma::uvec zij,
                     const int i,
                     const int j,
                     arma::Col<int> S,
                     const int nS,
                     const arma::mat beta0,
                     const arma::mat sig2,
                     const arma::mat Ip,
                     const int d,
                     const int P,
                     const Rcpp::List Y,
                     const Rcpp::List H,
                     const arma::cube HH) {

  arma::cube sumV(P,P,2);
  arma::mat sumVH(P,2);
  sumV.zeros();
  sumVH.zeros();
  
  // individual i
  arma::mat Hi = H[i];
  arma::mat HHi = HH.slice(i);
  arma::mat Yi = Y[i];
  arma::vec yi = Yi.col(d) - beta0(i,d);
  arma::mat Vi = arma::inv( lam2_s(0,d)*HHi/sig2(i,d) + Ip );
  sumV.slice(0) += Vi;
  sumVH.col(0) += Vi*Hi.t()*yi/sig2(i,d);
  
  // individual j
  arma::mat Hj = H[j];
  arma::mat HHj = HH.slice(j);
  arma::mat Yj = Y[j];
  arma::vec yj = Yj.col(d) - beta0(j,d);
  arma::mat Vj = arma::inv( lam2_s(1,d)*HHj/sig2(j,d) + Ip );
  sumV.slice(1) += Vj;
  sumVH.col(1) += Vj*Hj.t()*yj/sig2(j,d);

  
  // individual k in S
  for (int k = 0; k < nS; k++) {

    int Sk = S(k);
    arma::mat Hk = H[Sk];
    arma::mat HHk = HH.slice(Sk);
    arma::mat Yk = Y[Sk];
    arma::vec yk = Yk.col(d) - beta0(Sk,d);
    
    if (z_split(Sk) == zij(0)) {
      arma::mat Vk = arma::inv( lam2_s(0,d)*HHk/sig2(Sk,d) + Ip );
      sumV.slice(0) += Vk;
      sumVH.col(0) += Vk*Hk.t()*yk/sig2(Sk,d);
    } else {
      arma::mat Vk = arma::inv( lam2_s(1,d)*HHk/sig2(Sk,d) + Ip );
      sumV.slice(1) += Vk;
      sumVH.col(1) += Vk*Hk.t()*yk/sig2(Sk,d);
    }
    
  }
   
  List out;
  out["sumV"] = sumV;
  out["sumVH"] = sumVH;

  return out;

}  




/* sumV_SM_m */
Rcpp::List sumV_SM_m(const arma::vec lam2_m,
                     const int i,
                     const int j,
                     arma::Col<int> S,
                     const int nS,
                     const arma::mat beta0,
                     const arma::mat sig2,
                     const arma::mat Ip,
                     const int d,
                     const int P,
                     const Rcpp::List Y,
                     const Rcpp::List H,
                     const arma::cube HH) {
  
  arma::mat sumV(P,P);
  arma::vec sumVH(P);
  sumV.zeros();
  sumVH.zeros();
  
  // individual i
  arma::mat Hi = H[i];
  arma::mat HHi = HH.slice(i);
  arma::mat Yi = Y[i];
  arma::vec yi = Yi.col(d) - beta0(i,d);
  arma::mat Vi = arma::inv( lam2_m(d)*HHi/sig2(i,d) + Ip );
  sumV += Vi;
  sumVH += Vi*Hi.t()*yi/sig2(i,d);
  
  // individual j
  arma::mat Hj = H[j];
  arma::mat HHj = HH.slice(j);
  arma::mat Yj = Y[j];
  arma::vec yj = Yj.col(d) - beta0(j,d);
  arma::mat Vj = arma::inv( lam2_m(d)*HHj/sig2(j,d) + Ip );
  sumV += Vj;
  sumVH += Vj*Hj.t()*yj/sig2(j,d);
  
  // individual k in S
  for (int k = 0; k < nS; k++) {
    
    int Sk = S(k);
    arma::mat Hk = H[Sk];
    arma::mat HHk = HH.slice(Sk);
    arma::mat Yk = Y[Sk];
    arma::vec yk = Yk.col(d) - beta0(Sk,d);
    arma::mat Vk = arma::inv( lam2_m(d)*HHk/sig2(Sk,d) + Ip );
    sumV += Vk;
    sumVH += Vk*Hk.t()*yk/sig2(Sk,d);

  }
  
  List out;
  out["sumV"] = sumV;
  out["sumVH"] = sumVH;
  
  return out;
 
}  


#endif
