// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>

// using namespace arma;
using namespace Rcpp;

// PG sampler
extern "C" {
  #include "pgdraw/src/rcpp_pgdraw.cpp"
}

// multivariate Normal sampler
#include <mvnorm.h>
// [[Rcpp::depends(RcppDist)]]
#include "rmvnorm_arma.h"
#include "rmvnorm_eigen.h"

// header files of repulsive functions
#include "logFRc.h"
#include "logRF.h"

// header file of similarity functions
#include "functions_similarity.h"

// header file of design matrices H and K
#include "Hmat.h"

// header file of left truncated gamma sampler
// #include "rgamma_ars.h"
#include "rgamma_ars_add.h"

// header file of theta sampler
#include "update_theta_integrated.h"







// dmvnorm_log
double dmvnorm_log(arma::vec x, arma::vec m, double var, int P){
  arma::vec x_m = x - m;
  double out = -0.5 * ((double)P*log(var) + arma::as_scalar(x_m.t()*x_m)/var);
  return out;
}


// fbeta_log_int
double fbeta_log_int(const arma::mat& Hi, const arma::mat& HHi, const arma::mat& Yi,
                     const arma::mat beta0, const arma::mat sig2,
                     const arma::vec theta, const double lam2,
                     const int i, const int d, const int P, const arma::mat& Ip) {

  arma::vec yi = Yi.col(d) - beta0(i,d);
  arma::mat Vi = arma::inv( HHi/sig2(i,d) + Ip/lam2 );
  arma::vec mi = Hi.t()*yi/sig2(i,d) + theta/lam2;
  double out = -(double)P*log(lam2) + log(arma::det(Vi));
  out -= arma::as_scalar( theta.t()*theta )/lam2;
  out += arma::as_scalar( mi.t()*Vi*mi );
  return 0.5*out;
  
}


// dinvgamma_log
double dinvgamma_log(double x, double a, double b){
  return a*log(b) - lgamma(a) - (a+1)*log(x) - b/x;
}



////* hrcmx *////
// [[Rcpp::export]]
Rcpp::List mhrcmx_ind_rep_int(const Rcpp::List Y,        // observations
                  const int D,                          // number of curves of each individual 
                  const arma::mat X,                    // LSBP - covariates matrix
                  const arma::vec mu_alpha,             // LSBP - mean of alpha prior
                  const arma::mat S_alpha,              // LSBP - variance of alpha prior
                  const arma::Mat<double> Xcon,               // PPMx - continuous covariates matrix
                  const arma::Mat<int> Xcat,            // PPMx - categorical covariates matrix
                  const int p,                          // number of knots
                  const int n_theta,                    // dimension of cluster-specific curves
                  const double v,                       // theta prior
                  const double a0, const double b0,     // sig02 prior
                  const double as, const double bs,     // sig2 prior
                  const double at,                      // tau2 prior
                  const double bt=0,                    // tau2 prior
                  const double at0=0.01,                // bt prior
                  const double bt0=1,                   // bt prior
                  const double A=1,                     // lam2 prior
                  const double lam2_min = 1e-12,        // lam2 prior
                  const double Asig=1,                  // sig2 prior (truncated)
                  const double s2mu=1e4,                // mu prior
                  const double s02=1e4,                 // mu0 prior
                  const double M=1,                     // cohesion prior
                  const int ns=1e3, const int thin=10,  // MCMC settings
                  const int burn=1e4,                 // MCMC settings
                  const int H0=0,           // if H0=1 the model restrict sum(H*coefficients)=0
                  int J=0,                  // number of components in the FMMx
                  const double alpha_dir=1, // dirichlet prior
                  int repulsive=0,          // repulsive choice
                  double eps=1,             // initial factor of proposal variance
                  double eps0=1,            // initial factor of proposal variance
                  const double phi=1, const double nu=1, const double r=2, // repulsive components
                  const int theta_sampler=1,
                  const int nadapt=1e3,
                  const double alpha_cat=0.1, // dirichlet parameter similarity function
                  const double m0=0,          // normal mean similarity function
                  const double v0=10,         // normal variance similarity function     
                  const int only_nclr=0,
                  const int H_out=1,          // include design matrices in output list
                  const int SM=1,             // if SM=1 execute SM step
                  const int nGS=10) {         // number of iterations of Gibbs scan

  int mu_par = 1;
  if (H0 == 1) mu_par = 0;

  Rprintf("burn = %i\n", burn);
  Rprintf("thin = %i\n", thin);
  Rprintf("ns = %i\n", ns);
  Rprintf("\n");

  Rprintf("Asig = %f\n", Asig);
  Rprintf("A = %f\n", A);
  Rprintf("J = %i\n", J);
  Rprintf("p = %i\n", p);
  Rprintf("bt = %f\n", bt);
  Rprintf("\n");
  
  if (repulsive > 0) Rprintf("repulsive = %i\n", repulsive);
  if (repulsive > 0) Rprintf("phi = %f\n", phi);
  if (repulsive > 0) Rprintf("nu = %f\n", nu);
  if (repulsive > 0) Rprintf("r = %f\n", r);
  if (repulsive > 0) Rprintf("eps MH = %.1e\n", eps);
  if (repulsive > 0) Rprintf("eps0 MH = %.1e\n", eps0);
  Rprintf("\n");
  
  
  
  int ppmx = 1;
  int m_neal = 1;
  int fmmx = 0;
  
  if (J > 0) {
    fmmx = 1;
    ppmx = 0;
    m_neal = 0;
  }
  
  if (J == 0) repulsive = 0;
  
  const int m = Y.size();
  if (ppmx == 1) { J = m; }
  
  const int q = 3;
  const int P = p + q;
  const arma::mat Ip = arma::eye(P,P);

  
  // individual design matrices
  Rcpp::List H(m);
  for (int i = 0; i < m; i++) {
    arma::mat yi = Y[i];
    int ni = yi.n_rows;
    arma::mat Hi = Hmat(p,q,ni);
    // if H0=1 transform H
    if (H0 == 1) {
      // arma::rowvec Hmean = arma::mean(Hi,0);
      // arma::vec col1ni = arma::ones(ni);
      // H[i] = Hi - col1ni*Hmean;
      H[i] = Hi - arma::ones(ni)*arma::mean(Hi,0);
    } else {
      H[i] = Hi;
    }
  }

  // cluster-specific design matrix
  arma::mat Ht = Hmat(p,q,n_theta);
  // if H0=1 transform Ht
  if (H0 == 1) Ht = Ht - arma::ones(n_theta)*arma::mean(Ht,0); 

  
  // cluster-specific design matrix to compute shape distance
  arma::mat Ht0 = Hmat(p,q,n_theta);
  // transform Ht0 to restrict sum(Ht0*theta)=0
  Ht0 = Ht0 - arma::ones(n_theta)*arma::mean(Ht0,0);
  

  // // matrix product Hi.t()*Hi
  // Rcpp::List HH(m);
  // for (int i = 0; i < m; i++) {
  //   arma::mat Hi = H[i];
  //   HH[i] = Hi.t()*Hi;
  // }
  arma::cube HH(P,P,m);
  for (int i = 0; i < m; i++) {
    arma::mat Hi = H[i];
    HH.slice(i) = Hi.t()*Hi;
  }
  
  
  // K (penalty matrix)
  const arma::mat K = Kmat(P,v);
  const arma::mat Ki = arma::inv(K);
  
  
  // likelihood
  arma::cube beta(P,m,D);
  arma::mat beta0(m,D);
  arma::mat sig2(m,D);

  
  // prior
  arma::mat mu(P,D);
  arma::vec mu0(D);
  arma::vec sig02(D);

  
  // cluster-specific
  int Jmax = 0;
  if (J > 0) Jmax = J;
  int nclr;
  arma::Col<int> nj(J);
  arma::cube theta(P,J,D);
  arma::mat lam2(J,D);
  arma::mat tau2(J,D);
  arma::vec btD(D);

  
  // z (cluster indicator)
  arma::Row<int> z(m);
  const double max_log = arma::datum::log_max;

  
  // neal8 objects
  const arma::uvec jvec = arma::linspace<arma::uvec>(0, m+m_neal-1, m+m_neal);
  const double log_M = log(M);
  const double log_m = log(m_neal);
  
  
  // jain8 objects
  const arma::uvec nvec = arma::linspace<arma::uvec>(0, m-1, m);
  // arma::vec pn(n, arma::fill::value(1/(double)n));
  // arma::vec p2(2, arma::fill::value(0.5));
  int t;
  double log_FRc_GS_split;
  double log_FRc_GS_merge;
  

  
  // FMM objects
  const arma::uvec hvec_J = arma::linspace<arma::uvec>(0, J-1, J);
  arma::vec pj(J, arma::fill::value(1/(double)J));

  
  // sig2 and lam2 truncation
  const double A2 = A*A;
  const double A2sig = Asig*Asig;
  const double Lu = 1/lam2_min;
  const double lam_min = sqrt(lam2_min);

  
  // theta sampling
  double nr = pow((double)n_theta,1/r);
  // arma::cube VJ(P,P,D, arma::fill::zeros);


  // acceptance rate objects
  arma::cube accept_logratio(J,D,ns, arma::fill::zeros);
  arma::mat accept_logratio_mat(J,D, arma::fill::zeros);
  arma::Cube<int> accept(J,D,ns, arma::fill::zeros);
  arma::Mat<int> accept_mat(J,D, arma::fill::zeros);
  arma::Col<int> accept_GS(D, arma::fill::zeros);
  
  
  // eps adaptation objects
  int beps = 0;
  // const int nadapt = 1e3;
  arma::Cube<int> accept_cube(J,D,nadapt, arma::fill::zeros);
  arma::vec epsD(D, arma::fill::value(eps));
  arma::vec eps0D(D, arma::fill::value(eps0));

  
  // logit model with covariate-dependent component weights (Rigon, Durante (2021))
  int covariates_fmm = 0;
  if (fmmx == 1 && X.n_rows == m) covariates_fmm = 1;
  const int ncov = X.n_cols;
  arma::mat alpha(ncov, J-1, arma::fill::value(1));
  arma::mat Si_alpha = S_alpha.i();
  arma::vec Si_mu_alpha = Si_alpha * mu_alpha;
  arma::cube xTx(ncov,ncov,m);
  if (covariates_fmm == 1) {
    for (int i = 0; i < m; i++) { xTx.slice(i) = X.row(i).t() * X.row(i); } 
  }
  double wi;
  

  
    
  // ppmx (Muller, Quintana (2011))
  int covariates_con = 0;
  int covariates_cat = 0;
  if (ppmx == 1 && Xcon.n_rows == m) covariates_con = 1;
  if (ppmx == 1 && Xcat.n_rows == m) covariates_cat = 1;
  // Xcon = round(Xcon, 12);
  // const arma::vec X2con = round(Xcon % Xcon, 24);
  const arma::vec X2con = Xcon % Xcon;
  arma::vec sumxj(J);
  arma::vec sumx2j(J);
  const int Ncat = Xcat.n_cols; // number of categorical covariates
  arma::Col<int> ncat(Ncat); // number of distinct categories in each categorical covariate
  
  if (covariates_cat == 1) Rprintf("Ncat=%i\n", Ncat);
  for (int t = 0; t < Ncat; t++) {
    arma::Col<int> Xcat_t = arma::unique(Xcat.col(t));
    ncat(t) = Xcat_t.size();
    if (covariates_cat == 1) Rprintf("ncat(%i)=%i\n", t,ncat(t));
  }
  arma::Cube<int> ncatJ(arma::max(ncat),J,Ncat);
  arma::vec gsim_cat_new(Ncat);
  for (int t = 0; t < Ncat; t++) gsim_cat_new(t) = - log((double)ncat(t));
  

    
  
  // print some data and model summaries
  Rprintf("m = %i\n", m);
  Rprintf("fmmx = %i\n", fmmx);
  Rprintf("ppmx = %i\n", ppmx);
  if (covariates_con == 1) Rprintf("covariates_con\n");
  if (covariates_cat == 1) Rprintf("covariates_cat\n");
  if (covariates_fmm == 1) Rprintf("covariates_fmm\n");
  Rprintf("\n");
  
  // if (covariates_cat == 1) for (int t = 0; t < Ncat; t++) for (int i = 0; i < m; i++) Rprintf("Xcat(%i,%i)=%i\n", i,t,Xcat(i,t));  
  // Rprintf("\n");
    
  
  
  
  
  // parameter chains
  arma::Col<int> nclr_chain(ns);
  int ns1 = ns - only_nclr*(ns-1);
  arma::field<arma::cube> beta_chain(ns1);
  arma::field<arma::cube> theta_chain(ns1);
  arma::cube beta0_chain(m,D,ns1);
  arma::cube sig2_chain(m,D,ns1);
  arma::mat mu0_chain(D,ns1);
  arma::mat sig02_chain(D,ns1);
  arma::cube lam2_chain(J,D,ns1);
  arma::cube tau2_chain(J,D,ns1);
  arma::Mat<int> z_chain(ns1,m);
  arma::cube alpha_chain(ncov,J-1,ns1);
  arma::Mat<int> nj_chain(J,ns1);
  
  int ns1mu = ns - (1-mu_par)*(ns-1);
  arma::cube mu_chain(P,D,ns1mu);
  
  int ns1bt = 1;
  if (bt == 0) ns1bt = ns1;
  arma::mat bt_chain(D,ns1bt);
  
  
  RNGScope scope;

    
  // initial individual-specific values
  beta.ones();
  mu0.ones();
  sig02.ones();
  mu.zeros();
  if (Asig == 0) {
    sig2.ones();
  } else {
    double sig2init = Asig*Asig/3;
    sig2.fill(sig2init);
  }

  
  // initial cluster-specific stats
  nclr = 1;
  z.zeros();
  nj.zeros();
  nj(0) = m;
  
  sumxj.zeros();
  sumx2j.zeros();
  sumxj(0) = sum(Xcon.col(0));
  sumx2j(0) = sum(X2con.col(0));
  
  ncatJ.zeros();
  if (covariates_cat == 1) {
    for (int t = 0; t < Ncat; t++) {
      for (int i = 0; i < m; i++) {
        ncatJ.slice(t).col(0).row(Xcat(i,t)) += 1;
      }
    }
  }
  
  
  // if (covariates_cat == 1) for (int n = 0; n < ncatJ.n_rows; n++) {
  //   Rprintf("ncatJ(%i,0,0)=%i and ncatJ(%i,1,0)=%i\n", n,ncatJ(n,0,0),n,ncatJ(n,1,0));   
  // }
  // Rprintf("\n");
  // 
  // if (covariates_cat == 1) for (int n = 0; n < ncatJ.n_rows; n++) {
  //   Rprintf("ncatJ(%i,0,1)=%i and ncatJ(%i,1,1)=%i\n", n,ncatJ(n,0,1),n,ncatJ(n,1,1));
  // }
  // Rprintf("\n");
  

  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////
  ///// review the code to consider matrix Xcon /////
  ///// with possibly more than one column      /////
  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////


  // initial cluster-specific parameters
  if (bt != 0) {
    btD.fill(bt);
  } else{
    btD.fill(at0/bt0);
  }
  
  if (ppmx == 1) {
    lam2.row(0).fill(A*A/3);
    tau2.row(0).fill(1);
  } else {
    lam2.fill(A*A/3);
    tau2.ones();
  }
  

  
  // beta and theta must be initialized different
  // and theta vectors must be initialized different
  if (ppmx == 1) {
    for (int d = 0; d < D; d++) theta.slice(d).col(0) = rmvnorm(1, mu.col(d) , tau2(0,d)*Ki).t();
  } else {
    for (int d = 0; d < D; d++) {
      for (int j = 0; j < J; j++) {
        theta.slice(d).col(j) = rmvnorm(1, mu.col(d) , tau2(j,d)*Ki).t(); 
      }
    }
  }
  

  
  // mcmc
  int b = 0;
  int s = 0;
  while (s < ns) {

    b += 1;
    // Rprintf("b = %i \n", b);
    // Rprintf("s = %i \n", s);

    

    //////////* z sample (sm8) *//////////
    if (SM == 1) {
      
      /* (1) select randomly {i,j} */
      arma::uvec ij(2);
      int go = 0;
      while(go == 0) {
        // arma::uvec ij = RcppArmadillo::sample(nvec, 2, false, pn);
        ij = RcppArmadillo::sample(nvec, 2, false);
        if (ppmx == 1) {
          go = 1;
        } else if (nclr < Jmax) {
          go = 1;
        } else if (nclr > Jmax) {
          stop("\nError: nclr > Jmax");
        } else if (z(ij(0)) != z(ij(1))) {
          go = 1;
        }
      }
      int i = ij(0);
      int j = ij(1);
      int zi = z(i);
      int zj = z(j);
      // Rprintf("jain8: i=%i , j=%i\n",i,j);
      
      
      
      /* (2) create set S */
      arma::Col<int> S_1(m, arma::fill::value(-1));
      for (int k = 0; k < m; k++) {
        if ( k!=i && k!=j && (z(k)==zi || z(k)==zj) ) S_1(k) = k;
      }
      arma::uvec kS = arma::find(S_1 > -1);
      arma::Col<int> S = S_1.elem(kS);
      int nS = S.size();
      int nS2 = nS + 2; // nS + elements i and j
      
      
      
      /* (3) launch states */
      
      
      /* (3.1) split launch state */
      arma::mat lam2_s(2,D);
      arma::mat tau2_s(2,D);
      arma::cube theta_s(P,2,D);
      arma::Col<int> nj_s(2);
      arma::vec sumxj_s(2);
      arma::vec sumx2j_s(2);
      
      // arma::Mat<int> ncatj_s(ncat,2);
      arma::Cube<int> ncatJ_s(ncatJ.n_rows,2,Ncat);
      
      arma::Row<int> z_split = z;
      arma::uvec zij(2);
      // arma::vec pj_s(2)
      
      
      // jnew is used bellow as the new cluster index in the split proposal
      int jnew; // jnew is used only if zi == zj
      
      // for (int j=0; j<Jmax; j++) Rprintf("nj(%i)=%i\n", j,nj(j));
      // Rprintf("nclr=%i\n", nclr);
      // Rprintf("zi=%i\n", zi);
      // Rprintf("zj=%i\n", zj);
      
      if (zi == zj) {
        if (fmmx==1) jnew = arma::min( arma::find(nj==0) );
        if (ppmx==1) jnew = nclr; // nclr is used here as the minimum empty clusters index
        z_split(i) = jnew;
      }
      
      // Rprintf("jnew=%i\n",jnew);
      
      /* one uniform sample of z_split(k) for k \in S */
      zij = arma::conv_to<arma::uvec>::from(arma::Col<int>{z_split(i),zj});
      for (int k = 0; k < nS; k++) {
        z_split(S(k)) = RcppArmadillo::sample(zij, 1, false).at(0);
      }
      
      
      /* initialize model split parameters from their prior distributions */
      /* (it would be possible to initialize with the current chain value */ 
      /* in the case of the fmmx model)                                   */ 
      if (fmmx==1) {
        tau2_s.row(0) = tau2.row(z_split(i));
        tau2_s.row(1) = tau2.row(z_split(j));
      }
      for (int l = 0; l < 2; l++) {
        for (int d = 0; d < D; d++) {
          if (ppmx==1) tau2_s(l,d) = 1/Rf_rgamma(at,1/btD(d));
          theta_s.slice(d).col(l) = rmvnorm(1, mu.col(d) , tau2_s(l,d)*Ki).t();
          double lam = 0.0;
          while (lam < lam_min) lam = Rf_runif(0.0,A);
          lam2_s(l,d) = lam*lam;
        }
      }
      
      
      
      /* Gibbs sampling scans */
      
      // update sufficient stats (elements i and j, that are not in S)
      nj_s.ones();
      if (ppmx == 1) {
        
        if (covariates_con == 1) {
          sumxj_s(0) = Xcon(ij(0));
          sumxj_s(1) = Xcon(ij(1));
          sumx2j_s(0) = X2con(ij(0));
          sumx2j_s(1) = X2con(ij(1));
        }
        
        if (covariates_cat == 1) {
          // ncatj_s.zeros();
          // ncatj_s(Xcat(ij(0)),0) = 1;
          // ncatj_s(Xcat(ij(1)),1) = 1;
          ncatJ_s.zeros();
          for (int t = 0; t < Ncat; t++) {
            ncatJ_s(Xcat(ij(0),t),0,t) = 1;
            ncatJ_s(Xcat(ij(1),t),1,t) = 1;
          }
        }
        
      }
      
      // update stats (elements in S)
      for (int k = 0; k < nS; k++) {
        if (z_split(S(k)) == zij(0)) {
          nj_s(0) += 1;
          
          if (ppmx == 1) {
            if (covariates_con == 1) {
              sumxj_s(0) += Xcon(S(k));
              sumx2j_s(0) += X2con(S(k));
            }
            // if (covariates_cat == 1) ncatj_s(Xcat(S(k)),0) += 1;
            if (covariates_cat == 1) {
              for (int t = 0; t < Ncat; t++) {
                ncatJ_s(Xcat(S(k),t),0,t) += 1;
              }
            }
          }
          
        } else {
          nj_s(1) += 1;
          
          if (ppmx == 1) {
            
            if (covariates_con == 1) {
              sumxj_s(1) += Xcon(S(k));
              sumx2j_s(1) += X2con(S(k));
            }
            
            // if (covariates_cat == 1) ncatj_s(Xcat(S(k)),1) += 1;
            if (covariates_cat == 1) {
              for (int t = 0; t < Ncat; t++) {
                ncatJ_s(Xcat(S(k),t),1,t) += 1;
              }
            }
            
          }
        }
      }
      
      t = 0;
      while (t < nGS) {
        
        for (int d = 0; d < D; d++) {
          
          /* theta_s sample */
          arma::mat beta_clr_s(P,2);
          beta_clr_s.col(0) = beta.slice(d).col(i);
          beta_clr_s.col(1) = beta.slice(d).col(j);
          for (int k = 0; k < nS; k++) {
            if (z_split(S(k)) == zij(0)) {
              beta_clr_s.col(0) += beta.slice(d).col(S(k));
            } else {
              beta_clr_s.col(1) += beta.slice(d).col(S(k));
            }
          }
          for (int l = 0; l < 2; l++) {
            arma::vec m_s(P);
            arma::mat V_s(P,P);
            arma::mat Vi_s(P,P);
            Vi_s = K/tau2_s(l,d) + (double)nj_s(l)*Ip/lam2_s(l,d);
            V_s = arma::inv( Vi_s );
            m_s = V_s * ( K*mu.col(d)/tau2_s(l,d) + beta_clr_s.col(l)/lam2_s(l,d) );
            theta_s.slice(d).col(l) = rmvnorm(1, m_s , V_s).t();
          }
          
          /* tau2_s sample */
          if (ppmx==1) {
            double a_tau2_s = at + 0.5*(double)P;
            for (int l = 0; l < 2; l++) {
              arma::vec theta_mu_s = theta_s.slice(d).col(l)-mu.col(d);
              double b_tau2_s = btD(d) + 0.5*arma::as_scalar( theta_mu_s.t()*K*theta_mu_s );
              tau2_s(l,d) = 1/Rf_rgamma( a_tau2_s , 1/b_tau2_s );
            }
          }
          
          /* lam2_s sample */
          arma::vec beta_theta_clr_s(2);
          // beta_theta_clr_s.zeros();
          arma::vec b_i0 = beta.slice(d).col(i)-theta_s.slice(d).col(0);
          beta_theta_clr_s(0) = arma::as_scalar( b_i0.t()*b_i0 );
          arma::vec b_j1 = beta.slice(d).col(j)-theta_s.slice(d).col(1);
          beta_theta_clr_s(1) = arma::as_scalar( b_j1.t()*b_j1 );
          for (int k = 0; k < nS; k++) {
            if (z_split(S(k)) == zij(0)) {
              arma::vec b_k = beta.slice(d).col(S(k))-theta_s.slice(d).col(0);
              beta_theta_clr_s(0) += arma::as_scalar( b_k.t()*b_k );
            } else {
              arma::vec b_k = beta.slice(d).col(S(k))-theta_s.slice(d).col(1);
              beta_theta_clr_s(1) += arma::as_scalar( b_k.t()*b_k );
            }
          }
          // ARS to sample from w = 1/x
          for (int l = 0; l < 2; l++) {
            int njP = nj_s(l)*P;
            double a_lam2_s = 0.5*((double)njP-1);
            double b_lam2_s = 0.5*beta_theta_clr_s(l);
            lam2_s(l,d) = 1/rgamma_ars(A2, Lu, a_lam2_s, b_lam2_s);
          }
          
        } // end for(d)
        
        
        /* z_split sample */
        arma::vec logPij(2);
        arma::mat Pij(2,nS);
        for (int k = 0; k < nS; k++) {
          
          // remove element S(k) from stats
          if (z_split(S(k)) == zij(0)) {
            nj_s(0) -= 1;
            if (ppmx == 1) {
              
              if (covariates_con == 1) {
                sumxj_s(0) -= Xcon(S(k));
                sumx2j_s(0) -= X2con(S(k));
              }
              
              // if (covariates_cat == 1) ncatj_s(Xcat(S(k)),0) -= 1;
              if (covariates_cat == 1) {
                for (int t = 0; t < Ncat; t++) {
                  ncatJ_s(Xcat(S(k),t),0,t) -= 1;
                }
              }
              
            }
          } else {
            nj_s(1) -= 1;
            if (ppmx == 1) {
              if (covariates_con == 1) {
                sumxj_s(1) -= Xcon(S(k));
                sumx2j_s(1) -= X2con(S(k));
              }
              // if (covariates_cat == 1) ncatj_s(Xcat(S(k)),1) -= 1;
              if (covariates_cat == 1) {
                for (int t = 0; t < Ncat; t++) {
                  ncatJ_s(Xcat(S(k),t),1,t) -= 1;
                }
              }
            }
          }
          
          /* weights of components */
          if (fmmx == 1) {
            logPij(0) = log(pj(zij(0)));
            logPij(1) = log(pj(zj));
          }
          if (ppmx == 1) {
            logPij(0) = log((double)nj_s(0));
            logPij(1) = log((double)nj_s(1));
            
            // similarity functions of continuous covariates
            if (covariates_con == 1) {
              logPij(0) += gsim_con_gibbs( nj_s(0), v0, Xcon(S(k)), X2con(S(k)), sumxj_s(0) );
              logPij(1) += gsim_con_gibbs( nj_s(1), v0, Xcon(S(k)), X2con(S(k)), sumxj_s(1) );
            }
            
            // similarity functions of categorical covariates
            if (covariates_cat == 1) {
              // logPij(0) += gsim_cat_gibbs( nj_s(0), alpha_cat, ncat, ncatj_s(Xcat(S(k)),0) );
              // logPij(1) += gsim_cat_gibbs( nj_s(1), alpha_cat, ncat, ncatj_s(Xcat(S(k)),1) );
              for (int t = 0; t < Ncat; t++) {
                logPij(0) += gsim_cat_gibbs( nj_s(0), alpha_cat, ncat(t), ncatJ_s(Xcat(S(k),t),0,t) );
                logPij(1) += gsim_cat_gibbs( nj_s(1), alpha_cat, ncat(t), ncatJ_s(Xcat(S(k),t),1,t) );
              }
            }
            
          }
          for (int l = 0; l < 2; l++) {
            // beta density
            for (int d = 0; d < D; d++) {
              logPij(l) += dmvnorm_log(beta.slice(d).col(S(k)), theta_s.slice(d).col(l), lam2_s(l,d), P);
            }
          }
          
          // z_split sample
          double max_log_k = max_log - max(logPij) - log(2.0);
          arma::vec Pijk = exp(logPij + max_log_k);
          Pij.col(k) = Pijk/sum(Pijk);
          z_split(S(k)) = RcppArmadillo::sample(zij, 1, false, Pij.col(k)).at(0);
          
          // put back element S(k) in the sufficient stats
          if (z_split(S(k)) == zij(0)) {
            nj_s(0) += 1;
            if (ppmx == 1) {
              
              if (covariates_con == 1) {
                sumxj_s(0) += Xcon(S(k));
                sumx2j_s(0) += X2con(S(k));
              }
              
              // if (covariates_cat == 1) ncatj_s(Xcat(S(k)),0) += 1;
              if (covariates_cat == 1) {
                for (int t = 0; t < Ncat; t++) {
                  ncatJ_s(Xcat(S(k),t),0,t) += 1;
                }
              }
              
            }
          } else {
            nj_s(1) += 1;
            if (ppmx == 1) {
              
              if (covariates_con == 1) {
                sumxj_s(1) += Xcon(S(k));
                sumx2j_s(1) += X2con(S(k));
              }
              
              // if (covariates_cat == 1) ncatj_s(Xcat(S(k)),1) += 1;
              if (covariates_cat == 1) {
                for (int t = 0; t < Ncat; t++) {
                  ncatJ_s(Xcat(S(k),t),1,t) += 1;
                }
              }
              
            }
          }
          
        } // end z_split sample
        
        t += 1;
      } // end while
      
      
      
      
      /* (3.2) merge launch state */
      arma::vec lam2_m(D);
      arma::vec tau2_m(D);
      arma::mat theta_m(P,D);
      int nj_m = nS2;
      // double pj_m;
      arma::Row<int> z_merge = z;
      z_merge(i) = zj;
      for (int k = 0; k < nS; k++) z_merge(S(k)) = zj;
      
      
      
      /* initialize model parameters from their prior distributions */
      if (fmmx==1) tau2_m = tau2.row(zj).t();
      for (int d = 0; d < D; d++) {
        if (ppmx==1) tau2_m(d) = 1/Rf_rgamma(at,1/btD(d));
        theta_m.col(d) = rmvnorm(1, mu.col(d) , tau2_m(d)*Ki).t();
        double lam = 0.0;
        while (lam < lam_min) lam = Rf_runif(0.0,A);
        lam2_m(d) = lam*lam;
      }
      
      /* Gibbs sampling scans */
      t = 0;
      while (t < nGS) {
        
        for (int d = 0; d < D; d++) {
          
          /* tau2_m sample */
          if (ppmx==1) {
            double a_tau2_m = at + 0.5*(double)P;
            arma::vec theta_mu_m = theta_m.col(d)-mu.col(d);
            double b_tau2_m = btD(d) + 0.5*arma::as_scalar( theta_mu_m.t()*K*theta_mu_m );
            tau2_m(d) = 1/Rf_rgamma( a_tau2_m , 1/b_tau2_m );
          }
          
          
          /* theta_m sample */
          arma::vec beta_clr_m(P);
          beta_clr_m = beta.slice(d).col(i) + beta.slice(d).col(j);
          for (int k = 0; k < nS; k++) beta_clr_m += beta.slice(d).col(S(k));
          arma::vec m_m(P);
          arma::mat V_m(P,P);
          arma::mat Vi_m(P,P);
          Vi_m = K/tau2_m(d) + (double)nj_m*Ip/lam2_m(d);
          V_m = arma::inv( Vi_m );
          m_m = V_m * ( K*mu.col(d)/tau2_m(d) + beta_clr_m/lam2_m(d) );
          theta_m.col(d) = rmvnorm(1, m_m , V_m).t();
          
          
          /* lam2_m sample */
          double beta_theta_clr_m = 0;
          arma::vec b_i = beta.slice(d).col(i) - theta_m.col(d);
          beta_theta_clr_m += arma::as_scalar( b_i.t()*b_i );
          arma::vec b_j = beta.slice(d).col(j) - theta_m.col(d);
          beta_theta_clr_m += arma::as_scalar( b_j.t()*b_j );
          for (int k = 0; k < nS; k++) {
            arma::vec b_k = beta.slice(d).col(S(k)) - theta_m.col(d);
            beta_theta_clr_m += arma::as_scalar( b_k.t()*b_k );
          }
          // ARS to sample from w = 1/x
          int njP = nj_m*P;
          double a_lam2 = 0.5*((double)njP-1);
          double b_lam2 = 0.5*beta_theta_clr_m;
          lam2_m(d) = 1/rgamma_ars(A2, Lu, a_lam2, b_lam2);
          
        } // end for(d)
        
        t += 1;
      } // end while
      
      
      
      
      /* (4) split proposal */
      if (zi == zj) {
        
        // z_split was already built in step (3.1) above
        
        // jnew may be used bellow as the new proposed cluster index
        // because when zi == zj we have zij(0) = jnew
        // (we do z_split(i) = jnew in step (3.1))
        
        
        /* transition kernel objects */
        double log_phi_s4 = 0;
        double log_rho_s4 = 0;
        double log_phi_m4 = 0;
        
        
        /* ONE Gibbs sampling scan */
        for (int d = 0; d < D; d++) {
          
          /* tau2_s sample */
          if (ppmx==1) {
            double a_tau2_s = at + 0.5*(double)P;
            for (int l = 0; l < 2; l++) {
              arma::vec theta_mu_s = theta_s.slice(d).col(l)-mu.col(d);
              double b_tau2_s = btD(d) + 0.5*arma::as_scalar( theta_mu_s.t()*K*theta_mu_s );
              tau2_s(l,d) = 1/Rf_rgamma( a_tau2_s , 1/b_tau2_s );
              log_phi_s4 += dinvgamma_log(tau2_s(l,d), a_tau2_s, b_tau2_s);
            }
          }
          
          
          /* theta_s sample */
          arma::mat beta_clr_s(P,2);
          beta_clr_s.col(0) = beta.slice(d).col(i);
          beta_clr_s.col(1) = beta.slice(d).col(j);
          // for (int i = 0; i < m; i++) { beta_clr.col(z(i)) += beta.slice(d).col(i); }
          for (int k = 0; k < nS; k++) {
            if (z_split(S(k)) == zij(0)) {
              beta_clr_s.col(0) += beta.slice(d).col(S(k));
            } else {
              beta_clr_s.col(1) += beta.slice(d).col(S(k));
            }
          }
          for (int l = 0; l < 2; l++) {
            arma::vec m_s(P);
            arma::mat V_s(P,P);
            arma::mat Vi_s(P,P);
            Vi_s = K/tau2_s(l,d) + (double)nj_s(l)*Ip/lam2_s(l,d);
            V_s = arma::inv( Vi_s );
            m_s = V_s * ( K*mu.col(d)/tau2_s(l,d) + beta_clr_s.col(l)/lam2_s(l,d) );
            theta_s.slice(d).col(l) = rmvnorm(1, m_s , V_s).t();
            log_phi_s4 += dmvnorm(theta_s.slice(d).col(l).t(), m_s, V_s, true).at(0);
          }
          
          
          /* lam2_s sample */
          arma::vec beta_theta_clr_s(2);
          arma::vec b_i0 = beta.slice(d).col(i) - theta_s.slice(d).col(0);
          beta_theta_clr_s(0) = arma::as_scalar( b_i0.t()*b_i0 );
          arma::vec b_j1 = beta.slice(d).col(j) - theta_s.slice(d).col(1);
          beta_theta_clr_s(1) = arma::as_scalar( b_j1.t()*b_j1 );
          for (int k = 0; k < nS; k++) {
            if (z_split(S(k)) == zij(0)) {
              arma::vec b_k = beta.slice(d).col(S(k)) - theta_s.slice(d).col(0);
              beta_theta_clr_s(0) += arma::as_scalar( b_k.t()*b_k );
            } else {
              arma::vec b_k = beta.slice(d).col(S(k)) - theta_s.slice(d).col(1);
              beta_theta_clr_s(1) += arma::as_scalar( b_k.t()*b_k );
            }
          }
          // ARS to sample from w = 1/x
          for (int l = 0; l < 2; l++) {
            int njP = nj_s(l)*P;
            double a_lam2 = 0.5*((double)njP-1);
            double b_lam2 = 0.5*beta_theta_clr_s(l);
            lam2_s(l,d) = 1/rgamma_ars(A2, Lu, a_lam2, b_lam2);
            log_phi_s4 += dinvgamma_log(lam2_s(l,d), a_lam2, b_lam2);
          }
          
        } // end for(d)
        
        
        
        /* z_split sample */
        arma::vec logPij(2);
        arma::mat Pij(2,nS);
        for (int k = 0; k < nS; k++) {
          
          // remove element S(k) from sufficient stats
          if (z_split(S(k)) == zij(0)) {
            nj_s(0) -= 1;
            if (ppmx == 1) {
              if (covariates_con == 1) {
                sumxj_s(0) -= Xcon(S(k));
                sumx2j_s(0) -= X2con(S(k));
              }
              // if (covariates_cat == 1) ncatj_s(Xcat(S(k)),0) -= 1;
              if (covariates_cat == 1) {
                for (int t = 0; t < Ncat; t++) {
                  ncatJ_s(Xcat(S(k),t),0,t) -= 1;
                }
              }
            }
          } else {
            nj_s(1) -= 1;
            if (ppmx == 1) {
              if (covariates_con == 1) {
                sumxj_s(1) -= Xcon(S(k));
                sumx2j_s(1) -= X2con(S(k));
              }
              // if (covariates_cat == 1) ncatj_s(Xcat(S(k)),1) -= 1;
              if (covariates_cat == 1) {
                for (int t = 0; t < Ncat; t++) {
                  ncatJ_s(Xcat(S(k),t),1,t) -= 1;
                }
              }
            }
          }
          
          // weights of components
          if (fmmx == 1) {
            logPij(0) = log(pj(zij(0)));
            logPij(1) = log(pj(zj));
          }
          if (ppmx == 1) {
            logPij(0) = log((double)nj_s(0));
            logPij(1) = log((double)nj_s(1));
            
            // similarity functions of continuous covariates
            if (covariates_con == 1) {
              logPij(0) += gsim_con_gibbs( nj_s(0), v0, Xcon(S(k)), X2con(S(k)), sumxj_s(0) );
              logPij(1) += gsim_con_gibbs( nj_s(1), v0, Xcon(S(k)), X2con(S(k)), sumxj_s(1) );
            }
            
            // similarity functions of categorical covariates
            if (covariates_cat == 1) {
              // logPij(0) += gsim_cat_gibbs( nj_s(0), alpha_cat, ncat, ncatj_s(Xcat(S(k)),0) );
              // logPij(1) += gsim_cat_gibbs( nj_s(1), alpha_cat, ncat, ncatj_s(Xcat(S(k)),1) );
              for (int t = 0; t < Ncat; t++) {
                logPij(0) += gsim_cat_gibbs( nj_s(0), alpha_cat, ncat(t), ncatJ_s(Xcat(S(k),t),0,t) );
                logPij(1) += gsim_cat_gibbs( nj_s(1), alpha_cat, ncat(t), ncatJ_s(Xcat(S(k),t),1,t) );
              }
            }
          }
          
          // beta density
          for (int l = 0; l < 2; l++) {
            for (int d = 0; d < D; d++) {
              logPij(l) += dmvnorm_log(beta.slice(d).col(S(k)), theta_s.slice(d).col(l), lam2_s(l,d), P);
            }
          }
          
          // z_split sample
          double max_log_k = max_log - max(logPij) - log(2.0);
          arma::vec Pijk = exp(logPij + max_log_k);
          Pij.col(k) = Pijk/sum(Pijk);
          z_split(S(k)) = RcppArmadillo::sample(zij, 1, false, Pij.col(k)).at(0);
          
          // put back element S(k) in the sufficient stats
          if (z_split(S(k)) == zij(0)) {
            nj_s(0) += 1;
            if (ppmx == 1) {
              
              if (covariates_con == 1) {
                sumxj_s(0) += Xcon(S(k));
                sumx2j_s(0) += X2con(S(k));
              }
              
              // if (covariates_cat == 1) ncatj_s(Xcat(S(k)),0) += 1;
              if (covariates_cat == 1) {
                for (int t = 0; t < Ncat; t++) {
                  ncatJ_s(Xcat(S(k),t),0,t) += 1;
                }
              }
              
            }
            log_rho_s4 += log(Pij(0,k));
          } else {
            nj_s(1) += 1;
            if (ppmx == 1) {
              
              if (covariates_con == 1) {
                sumxj_s(1) += Xcon(S(k));
                sumx2j_s(1) += X2con(S(k));
              }
              
              // if (covariates_cat == 1) ncatj_s(Xcat(S(k)),1) += 1;
              if (covariates_cat == 1) {
                for (int t = 0; t < Ncat; t++) {
                  ncatJ_s(Xcat(S(k),t),1,t) += 1;
                }
              }
              
            }
            log_rho_s4 += log(Pij(1,k));
          }
          
        } // end z_split sample
        
        
        
        
        /* HYPOTETICAL Gibbs sampling scan to compute numerator of Eq. (15) */
        for (int d = 0; d < D; d++) {
          
          if (ppmx==1) {
            /* tau2(zi,d) density conditional to theta_m.col(d) (from launch merge state) */
            double a_tau2_m = at + 0.5*(double)P;
            arma::vec theta_mu_m = theta_m.col(d)-mu.col(d);
            double b_tau2_m = btD(d) + 0.5*arma::as_scalar( theta_mu_m.t()*K*theta_mu_m );
            // tau2_m(d) = 1/Rf_rgamma( a_tau2_m , 1/b_tau2_m ); // HYPOTETICAL Gibbs sampling
            log_phi_m4 += dinvgamma_log(tau2(zi,d), a_tau2_m, b_tau2_m);
          }
          
          
          /* theta.slice(d).col(zi) density conditional to lam2_m(d) and tau2(zi,d) (from launch merge state) */
          arma::vec beta_clr_merged(P);
          beta_clr_merged = beta.slice(d).col(i) + beta.slice(d).col(j);
          for (int k = 0; k < nS; k++) beta_clr_merged += beta.slice(d).col(S(k));
          arma::vec m_m(P);
          arma::mat V_m(P,P);
          arma::mat Vi_m(P,P);
          Vi_m = K/tau2(zi,d) + (double)nj_m*Ip/lam2_m(d);
          V_m = arma::inv( Vi_m );
          m_m = V_m * ( K*mu.col(d)/tau2(zi,d) + beta_clr_merged/lam2_m(d) );
          log_phi_m4 += dmvnorm(theta.slice(d).col(zi).t(), m_m, V_m, true).at(0);
          
          
          /* lam(zi,d) density conditional to original theta.slice(d).col(zi) */
          double beta_theta_clr_merged = 0;
          arma::vec b_i = beta.slice(d).col(i) - theta.slice(d).col(zi);
          beta_theta_clr_merged += arma::as_scalar( b_i.t()*b_i );
          arma::vec b_j = beta.slice(d).col(j) - theta.slice(d).col(zi);
          beta_theta_clr_merged += arma::as_scalar( b_j.t()*b_j );
          for (int k = 0; k < nS; k++) {
            arma::vec b_k = beta.slice(d).col(S(k)) - theta.slice(d).col(zi);
            beta_theta_clr_merged += arma::as_scalar( b_k.t()*b_k );
          }
          // ARS to sample from w = 1/x
          int njP = nj_m*P;
          double a_lam2 = 0.5*((double)njP-1);
          double b_lam2 = 0.5*beta_theta_clr_merged;
          // lam2_m(d) = 1/rgamma_ars(A2, Lu, a_lam2, b_lam2); // HYPOTETICAL Gibbs sampling
          log_phi_m4 += dinvgamma_log(lam2(zi,d), a_lam2, b_lam2);
          
        }
        
        
        
        /* acceptance log ratio */
        
        
        /* proposal log ratio (Eq. (15)) */
        double log_q = 0;
        log_q += log_phi_m4;
        log_q -= log_phi_s4 + log_rho_s4;
        
        
        /* likelihood log ratio (Eq. (11)) */
        double log_L = 0;
        // elements i and j
        for (int d = 0; d < D; d++) {
          log_L += dmvnorm(beta.slice(d).col(i).t(), theta_s.slice(d).col(0), lam2_s(0,d)*Ip, true).at(0);
          log_L += dmvnorm(beta.slice(d).col(j).t(), theta_s.slice(d).col(1), lam2_s(1,d)*Ip, true).at(0);
          log_L -= dmvnorm(beta.slice(d).col(i).t(), theta.slice(d).col(zi), lam2(zi,d)*Ip, true).at(0); // zi == zj
          log_L -= dmvnorm(beta.slice(d).col(j).t(), theta.slice(d).col(zj), lam2(zj,d)*Ip, true).at(0); // zi == zj
          // elements in S
          for (int k = 0; k < nS; k++) {
            if (z_split(S(k))== zij(0)) {
              log_L += dmvnorm(beta.slice(d).col(S(k)).t(), theta_s.slice(d).col(0), lam2_s(0,d)*Ip, true).at(0);
              log_L -= dmvnorm(beta.slice(d).col(S(k)).t(), theta.slice(d).col(zi), lam2(zi,d)*Ip, true).at(0); // zi == zj
            } else {
              log_L += dmvnorm(beta.slice(d).col(S(k)).t(), theta_s.slice(d).col(1), lam2_s(1,d)*Ip, true).at(0);
              log_L -= dmvnorm(beta.slice(d).col(S(k)).t(), theta.slice(d).col(zj), lam2(zj,d)*Ip, true).at(0); // zi == zj
            }
          }
        }
        
        
        /* prior log ratio (Eq. (7)) */
        double log_p = 0;
        
        if (fmmx == 1) {
          // log_p += log(pj(zij(0))) + log(pj(zj));
          // for (int k = 0; k < nS; k++) log_p += log(pj(z_split(S(k))));
          log_p += (double)nj_s(0)*log(pj(zij(0))) + (double)nj_s(1)*log(pj(zj));
          log_p -= ((double)nS2)*log(pj(zj));
        }
        
        if (ppmx == 1) {
          
          // cohesions
          log_p += log_M;
          for (int k = 1; k < nj_s(0); k++) { log_p += log((double)k); }
          for (int k = 1; k < nj_s(1); k++) { log_p += log((double)k); }
          for (int k = 1; k < nS2; k++) { log_p -= log((double)k); }
          
          // similarity functions (continuous covariates)
          if (covariates_con == 1) {
            log_p += gsim_con(nj_s(0), v0, sumxj_s(0), sumx2j_s(0));
            log_p += gsim_con(nj_s(1), v0, sumxj_s(1), sumx2j_s(1));
            // double sumxj_s01 = sumxj_s(0) + sumxj_s(1);
            // double sumx2j_s01 = sumx2j_s(0) + sumx2j_s(1);
            // if (sumxj(zj) != sumxj_s01) stop("sumxj(zj) != sumxj_s01\n");
            // if (sumx2j(zj) != sumx2j_s01) stop("sumx2j(zj) != sumx2j_s01\n");
            // log_p -= gsim_con(nS2, v0, sumxj_s01, sumx2j_s01);
            log_p -= gsim_con(nS2, v0, sumxj(zj), sumx2j(zj));
          }
          
          // similarity functions (categorical covariates)
          if (covariates_cat == 1) {
            
            int nj_s01 = nj_s(0) + nj_s(1);
            
            // log_p += gsim_cat(nj_s(0), alpha_cat, ncat, ncatj_s.col(0));
            // log_p += gsim_cat(nj_s(1), alpha_cat, ncat, ncatj_s.col(1));
            // arma::Col<int> ncatj_s01 = ncatj_s.col(0) + ncatj_s.col(1);
            // log_p -= gsim_cat(nj_s01, alpha_cat, ncat, ncatj_s01);
            // 
            // if (nS2 != nj_s01 || nj(zj) != nj_s01) stop("nS2 != nj_s01 || nj(zj) != nj_s01\n");
            // for (int icat = 0; icat < ncat; icat++) {
            //   if (ncatj(icat,zj) != ncatj_s01(icat)) stop("ncatj.col(zj) != ncatj_s01\n");
            // }
            
            for (int t = 0; t < Ncat; t++) {
              log_p += gsim_cat(nj_s(0), alpha_cat, ncat(t), ncatJ_s.slice(t).col(0));
              log_p += gsim_cat(nj_s(1), alpha_cat, ncat(t), ncatJ_s.slice(t).col(1));
              arma::Col<int> ncatj_s01 = ncatJ_s.slice(t).col(0) + ncatJ_s.slice(t).col(1);
              log_p -= gsim_cat(nj_s01, alpha_cat, ncat(t), ncatj_s01);
            }
            
            if (nS2 != nj_s01 || nj(zj) != nj_s01) stop("nS2 != nj_s01 || nj(zj) != nj_s01\n");
            for (int t = 0; t < Ncat; t++) {
              arma::Col<int> ncatj_s01 = ncatJ_s.slice(t).col(0) + ncatJ_s.slice(t).col(1);
              for (int icat = 0; icat < ncat(t); icat++) {
                if (ncatJ(icat,zj,t) != ncatj_s01(icat)) stop("ncatj.col(zj) != ncatj_s01\n");
              }
            }
            
          }
        }
        
        for (int d = 0; d < D; d++) {
          log_p +=
            dmvnorm(theta_s.slice(d).col(0).t(), mu.col(d), tau2_s(0,d)*Ki, true).at(0) +
            dmvnorm(theta_s.slice(d).col(1).t(), mu.col(d), tau2_s(1,d)*Ki, true).at(0) +
            -0.5*log(lam2_s(0,d)) - log(2.0*A) +
            -0.5*log(lam2_s(1,d)) - log(2.0*A);
            log_p -=
            dmvnorm(theta.slice(d).col(zj).t(), mu.col(d), tau2(zj,d)*Ki, true).at(0) +
            -0.5*log(lam2(zj,d)) - log(2.0*A);
            // if PPMx include tau2 prior densities
            if (ppmx==1) {
              log_p += dinvgamma_log(tau2_s(0,d), at, btD(d));
              log_p += dinvgamma_log(tau2_s(1,d), at, btD(d));
              log_p -= dinvgamma_log(tau2(zj,d) , at, btD(d));
            }
        }
        
        
        /* prior log ratio (Eq. (7)) - Repulsive factor */
        if (fmmx==1) {
          if (repulsive != 0) {
            
            double log_RF = 0;
            
            for (int d = 0; d < D; d++) {
              
              // theta matrix without columns jnew and zj
              arma::mat theta_ = theta.slice(d);
              // arma::uvec zij_s(2);
              // zij_s(0) = jnew;
              // zij_s(1) = zj;
              // theta_.shed_cols(zij_s);
              theta_.shed_cols(zij);
              
              // difference between curves jnew and zj
              arma::mat df_01s(n_theta,1);
              df_01s.col(0) = Ht0*(theta_s.slice(d).col(0) - theta_s.slice(d).col(1));
              arma::mat df_01(n_theta,1);
              df_01.col(0) = Ht0*(theta.slice(d).col(jnew) - theta.slice(d).col(zj));
              
              // difference between curve jnew and other curves
              arma::mat df_0s(n_theta,J-2);
              for (int h = 0; h < J-2; h++) df_0s.col(h) = Ht0*(theta_s.slice(d).col(0) - theta_.col(h));
              arma::mat df_0(n_theta,J-2);
              for (int h = 0; h < J-2; h++) df_0.col(h) = Ht0*(theta.slice(d).col(jnew) - theta_.col(h));
              
              // difference between curve zj and other curves
              arma::mat df_1s(n_theta,J-2);
              for (int h = 0; h < J-2; h++) df_1s.col(h) = Ht0*(theta_s.slice(d).col(1) - theta_.col(h));
              arma::mat df_1(n_theta,J-2);
              for (int h = 0; h < J-2; h++) df_1.col(h) = Ht0*(theta.slice(d).col(zj) - theta_.col(h));
              
              // repulsive factor (log)
              log_RF += logRF(repulsive, phi, nu, r, n_theta, df_01s);
              log_RF -= logRF(repulsive, phi, nu, r, n_theta, df_01);
              
              log_RF += logRF(repulsive, phi, nu, r, n_theta, df_0s);
              log_RF -= logRF(repulsive, phi, nu, r, n_theta, df_0);
              
              log_RF += logRF(repulsive, phi, nu, r, n_theta, df_1s);
              log_RF -= logRF(repulsive, phi, nu, r, n_theta, df_1);
              
            } // end for(d)
            
            log_p += log_RF;
            
          } // end if repulsive
        } // end if fmmx
        
        
        
        /* M-H */
        double log_alpha = log_q + log_p + log_L;
        double u = Rf_runif(0.0,1.0);
        if (log(u) < log_alpha) {
          z = z_split;
          // parameters update
          for (int d = 0; d < D; d++) {
            if (ppmx==1) {
              tau2(zij(0),d) = tau2_s(0,d);
              tau2(zj,d) = tau2_s(1,d);
            }
            theta.slice(d).col(zij(0)) = theta_s.slice(d).col(0);
            theta.slice(d).col(zj) = theta_s.slice(d).col(1);
            lam2(zij(0),d) = lam2_s(0,d);
            lam2(zj,d) = lam2_s(1,d);
          }
          // sufficient stats update
          nj(zij(0)) = nj_s(0);
          nj(zj) = nj_s(1);
          
          if (ppmx == 1) {
            nclr += 1;
            J = nclr;
            
            if (covariates_con == 1) {
              sumxj(zij(0)) = sumxj_s(0);
              sumxj(zj) = sumxj_s(1);
              sumx2j(zij(0)) = sumx2j_s(0);
              sumx2j(zj) = sumx2j_s(1);
            }
            
            if (covariates_cat == 1) {
              for (int t = 0; t < Ncat; t++) {
                ncatJ.slice(t).col(zij(0)) = ncatJ_s.slice(t).col(0);
                ncatJ.slice(t).col(zj) = ncatJ_s.slice(t).col(1);
              }
            }
            
          }
          
          // Rprintf("sumxj(%i) = %f (MH split)\n", zj, sumxj(zj));
          // Rprintf("nclr (M-H split) = %i (b=%i)\n",nclr,b);
        }
        
        if (sum(nj) != m) stop("sum(nj) != m (split proposal)\n");
        
        if (ppmx==1 && arma::max(z) != arma::min( arma::find(nj==0) )-1) {
          Rprintf("b = %i\n", b);
          Rprintf("arma::max(z) = %i\n", arma::max(z));
          Rprintf("arma::min( arma::find(nj==0) ) = %i\n", arma::min( arma::find(nj==0) ));
          stop("arma::max(z) != arma::min( arma::find(nj==0) )-1 (split proposal)\n");
        }
        
      } // end of (4) split proposal
      
      
      
      if (ppmx==1 && arma::max(z) != arma::min( arma::find(nj==0) )-1) {
        Rprintf("arma::max(z) = %i\n", arma::max(z));
        Rprintf("arma::min( arma::find(nj==0) ) = %i\n", arma::min( arma::find(nj==0) ));
        stop("arma::max(z) != arma::min( arma::find(nj==0) )-1 (4-5)\n");
      }
      
      
      
      
      /* (5) merge proposal */
      if (zi != zj) {
        
        // z_merge was already created in step (3.2) above
        
        /* transition kernel object */
        double log_phi_m5 = 0;
        double log_phi_s5 = 0;
        double log_rho_s5 = 0;
        
        
        /* ONE Gibbs sampling scan */
        // nj_m = nS2; // already done above
        
        for (int d = 0; d < D; d++) {
          
          /* tau2_m sample */
          if (ppmx==1) {
            double a_tau2_m = at + 0.5*(double)P;
            arma::vec theta_mu_m = theta_m.col(d)-mu.col(d);
            double b_tau2_m = btD(d) + 0.5*arma::as_scalar( theta_mu_m.t()*K*theta_mu_m );
            tau2_m(d) = 1/Rf_rgamma( a_tau2_m , 1/b_tau2_m );
            log_phi_m5 += dinvgamma_log(tau2_m(d), a_tau2_m, b_tau2_m);
          }
          
          
          /* theta_m sample */
          arma::vec beta_clr_m(P);
          beta_clr_m = beta.slice(d).col(i) + beta.slice(d).col(j);
          for (int k = 0; k < nS; k++) beta_clr_m += beta.slice(d).col(S(k));
          arma::vec m_m(P);
          arma::mat V_m(P,P);
          arma::mat Vi_m(P,P);
          Vi_m = K/tau2_m(d) + (double)nj_m*Ip/lam2_m(d);
          V_m = arma::inv( Vi_m );
          m_m = V_m * ( K*mu.col(d)/tau2_m(d) + beta_clr_m/lam2_m(d) );
          theta_m.col(d) = rmvnorm(1, m_m , V_m).t();
          log_phi_m5 += dmvnorm(theta_m.col(d).t(), m_m, V_m, true).at(0);
          
          /* lam2_m sample */
          double beta_theta_clr_m = 0;
          arma::vec b_i = beta.slice(d).col(i) - theta_m.col(d);
          beta_theta_clr_m += arma::as_scalar( b_i.t()*b_i );
          arma::vec b_j = beta.slice(d).col(j) - theta_m.col(d);
          beta_theta_clr_m += arma::as_scalar( b_j.t()*b_j );
          for (int k = 0; k < nS; k++) {
            arma::vec b_k = beta.slice(d).col(S(k)) - theta_m.col(d);
            beta_theta_clr_m += arma::as_scalar( b_k.t()*b_k );
          }
          // ARS to sample from w = 1/x
          int njP = nj_m*P;
          double a_lam2 = 0.5*((double)njP-1);
          double b_lam2 = 0.5*beta_theta_clr_m;
          lam2_m(d) = 1/rgamma_ars(A2, Lu, a_lam2, b_lam2);
          log_phi_m5 += dinvgamma_log(lam2_m(d), a_lam2, b_lam2);
          
        } // end for(d)
        
        
        
        /* HYPOTETICAL Gibbs sampling scan to compute numerator of Eq. (16) */
        for (int d = 0; d < D; d++) {
          
          /* tau2(zi,d),tau2(zj,d) densities conditional to theta_s (from launch split state) */
          if (ppmx==1) {
            double a_tau2_s = at + 0.5*(double)P;
            for (int l = 0; l < 2; l++) {
              arma::vec theta_mu_s = theta_s.slice(d).col(l)-mu.col(d);
              double b_tau2_s = btD(d) + 0.5*arma::as_scalar( theta_mu_s.t()*K*theta_mu_s );
              if (l == 0) {
                // tau2_s(l,d) = 1/Rf_rgamma( a_tau2_s , 1/b_tau2_s ); // HYPOTETICAL Gibbs sampling
                log_phi_s5 += dinvgamma_log(tau2(zi,d), a_tau2_s, b_tau2_s);
              } else {
                // tau2_s(l,d) = 1/Rf_rgamma( a_tau2_s , 1/b_tau2_s ); // HYPOTETICAL Gibbs sampling
                log_phi_s5 += dinvgamma_log(tau2(zj,d), a_tau2_s, b_tau2_s);
              }
            }
          }
          
          
          /* theta.slice(d).col(zi),theta.slice(d).col(zj) densities         */
          /* conditional to original tau2(zi,d),tau2(zj,d)                   */
          /* conditional to lam2_s(0,d),lam2_s(1,d)(from launch split state) */
          arma::mat beta_clr_s(P,2);
          beta_clr_s.col(0) = beta.slice(d).col(i);
          beta_clr_s.col(1) = beta.slice(d).col(j);
          for (int k = 0; k < nS; k++) {
            if (z_split(S(k)) == zij(0)) {
              beta_clr_s.col(0) += beta.slice(d).col(S(k));
            } else {
              beta_clr_s.col(1) += beta.slice(d).col(S(k));
            }
          }
          for (int l = 0; l < 2; l++) {
            arma::vec m_s(P);
            arma::mat V_s(P,P);
            arma::mat Vi_s(P,P);
            if (l == 0) {
              Vi_s = K/tau2(zi,d) + (double)nj_s(l)*Ip/lam2_s(l,d);
              V_s = arma::inv( Vi_s );
              m_s = V_s * ( K*mu.col(d)/tau2(zi,d) + beta_clr_s.col(l)/lam2_s(l,d) );
              log_phi_s5 += dmvnorm(theta.slice(d).col(zi).t(), m_s, V_s, true).at(0);
            } else {
              Vi_s = K/tau2(zj,d) + (double)nj_s(l)*Ip/lam2_s(l,d);
              V_s = arma::inv( Vi_s );
              m_s = V_s * ( K*mu.col(d)/tau2(zj,d) + beta_clr_s.col(l)/lam2_s(l,d) );
              log_phi_s5 += dmvnorm(theta.slice(d).col(zj).t(), m_s, V_s, true).at(0);
            }
          }
          
          
          /* lam2(zi,d),lam2(zj,d) densities                                       */
          /* conditional to original theta.slice(d).col(zi),theta.slice(d).col(zj) */
          arma::vec beta_theta_clr_s(2);
          arma::vec b_zi = beta.slice(d).col(i) - theta.slice(d).col(zi);
          beta_theta_clr_s(0) = arma::as_scalar( b_zi.t()*b_zi );
          arma::vec b_zj = beta.slice(d).col(j) - theta.slice(d).col(zj);
          beta_theta_clr_s(1) = arma::as_scalar( b_zj.t()*b_zj );
          for (int k = 0; k < nS; k++) {
            if (z_split(S(k)) == zij(0)) {
              arma::vec b_k = beta.slice(d).col(S(k)) - theta.slice(d).col(zi);
              beta_theta_clr_s(0) += arma::as_scalar( b_k.t()*b_k );
            } else {
              arma::vec b_k = beta.slice(d).col(S(k)) - theta.slice(d).col(zj);
              beta_theta_clr_s(1) += arma::as_scalar( b_k.t()*b_k );
            }
          }
          // ARS to sample from w = 1/x
          for (int l = 0; l < 2; l++) {
            int njP = nj_s(l)*P;
            double a_lam2 = 0.5*((double)njP-1);
            double b_lam2 = 0.5*beta_theta_clr_s(l);
            if (l == 0) {
              // lam2_s(l,d) = 1/rgamma_ars(A2, Lu, a_lam2, b_lam2); // HYPOTETICAL Gibbs sampling
              log_phi_s5 += dinvgamma_log(lam2(zi,d), a_lam2, b_lam2);
            } else {
              // lam2_s(l,d) = 1/rgamma_ars(A2, Lu, a_lam2, b_lam2); // HYPOTETICAL Gibbs sampling
              log_phi_s5 += dinvgamma_log(lam2(zj,d), a_lam2, b_lam2);
            }
          }
          
        } // end for(d)
        
        
        
        /* z_split sample */
        arma::vec logPij(2);
        arma::mat Pij(2,nS);
        for (int k = 0; k < nS; k++) {
          
          if (z_split(S(k)) == zij(0)) {
            nj_s(0) -= 1;
            if (ppmx == 1) {
              
              if (covariates_con == 1) {
                sumxj_s(0) -= Xcon(S(k));
                sumx2j_s(0) -= X2con(S(k));
              }
              
              // if (covariates_cat == 1) ncatj_s(Xcat(S(k)),0) -= 1;
              if (covariates_cat == 1) {
                for (int t = 0; t < Ncat; t++) {
                  ncatJ_s(Xcat(S(k),t),0,t) -= 1;
                }
              }
              
              
            }
          } else {
            nj_s(1) -= 1;
            if (ppmx == 1) {
              
              if (covariates_con == 1) {
                sumxj_s(1) -= Xcon(S(k));
                sumx2j_s(1) -= X2con(S(k));
              }
              
              // if (covariates_cat == 1) ncatj_s(Xcat(S(k)),1) -= 1;
              if (covariates_cat == 1) {
                for (int t = 0; t < Ncat; t++) {
                  ncatJ_s(Xcat(S(k),t),1,t) -= 1;
                }
              }
              
            }
          }
          
          /* weights of components */
          if (fmmx == 1) {
            logPij(0) = log(pj(zij(0)));
            logPij(1) = log(pj(zj));
          }
          if (ppmx == 1) {
            logPij(0) = log((double)nj_s(0));
            logPij(1) = log((double)nj_s(1));
            // similarity functions of continuous covariates
            if (covariates_con == 1) {
              logPij(0) += gsim_con_gibbs( nj_s(0), v0, Xcon(S(k)), X2con(S(k)), sumxj_s(0) );
              logPij(1) += gsim_con_gibbs( nj_s(1), v0, Xcon(S(k)), X2con(S(k)), sumxj_s(1) );
            }
            // similarity functions of categorical covariates
            if (covariates_cat == 1) {
              // logPij(0) += gsim_cat_gibbs( nj_s(0), alpha_cat, ncat, ncatj_s(Xcat(S(k)),0) );
              // logPij(1) += gsim_cat_gibbs( nj_s(1), alpha_cat, ncat, ncatj_s(Xcat(S(k)),1) );
              for (int t = 0; t < Ncat; t++) {
                logPij(0) += gsim_cat_gibbs( nj_s(0), alpha_cat, ncat(t), ncatJ_s(Xcat(S(k),t),0,t) );
                logPij(1) += gsim_cat_gibbs( nj_s(1), alpha_cat, ncat(t), ncatJ_s(Xcat(S(k),t),1,t) );
              }
            }
          }
          for (int l = 0; l < 2; l++) {
            // beta density
            for (int d = 0; d < D; d++) {
              logPij(l) += dmvnorm_log(beta.slice(d).col(S(k)), theta_s.slice(d).col(l), lam2_s(l,d), P);
            }
          }
          
          double max_log_k = max_log - max(logPij) - log(2.0);
          arma::vec Pijk = exp(logPij + max_log_k);
          Pij.col(k) = Pijk/sum(Pijk);
          // z_split(S(k)) = RcppArmadillo::sample(zij, 1, false, Pij.col(k)).at(0);
          z_split(S(k)) = z(S(k)); // HYPOTETICAL update of z_split to original z
          // z_split is updated here only to be used in the "if" next
          
          if (z_split(S(k)) == zi) {
            // // the updates of the sufficient statistics bellow
            // // are not necessary because they are not used anymore
            // nj_s(0) += 1;
            // if (covariates_con == 1) {
            //   sumxj_s(0) += Xcon(S(k));
            //   sumx2j_s(0) += X2con(S(k));
            // }
            // // if (covariates_cat == 1) ncatj_s(Xcat(S(k)),0) += 1;
            // if (covariates_cat == 1) {
            //   for (int t = 0; t < Ncat; t++) {
            //     ncatJ_s(Xcat(S(k),t),0,t) += 1;
            //   }
            // }
            log_rho_s5 += log(Pij(0,k));
          } else {
            // // the updates of the sufficient statistics bellow
            // // are not necessary because they are not used anymore
            // nj_s(1) += 1;
            // if (covariates_con == 1) {
            //   sumxj_s(1) += Xcon(S(k));
            //   sumx2j_s(1) += X2con(S(k));
            // }
            // // if (covariates_cat == 1) ncatj_s(Xcat(S(k)),1) += 1;
            // if (covariates_cat == 1) {
            //   for (int t = 0; t < Ncat; t++) {
            //     ncatJ_s(Xcat(S(k),t),1,t) += 1;
            //   }
            // }
            log_rho_s5 += log(Pij(1,k));
          }
          
        }
        
        
        
        
        /* acceptance log ratio */
        
        
        /* proposal log ratio (Eq. (16)) */
        double log_q = 0;
        log_q += log_phi_s5 + log_rho_s5;
        log_q -= log_phi_m5;
        
        
        /* likelihood log ratio (Eq. (12)) */
        double log_L = 0;
        for (int d = 0; d < D; d++) {
          // elements i and j
          log_L += dmvnorm(beta.slice(d).col(i).t(), theta_m.col(d), lam2_m(d)*Ip, true).at(0);
          log_L += dmvnorm(beta.slice(d).col(j).t(), theta_m.col(d), lam2_m(d)*Ip, true).at(0);
          log_L -= dmvnorm(beta.slice(d).col(i).t(), theta.slice(d).col(zi), lam2(zi,d)*Ip, true).at(0);
          log_L -= dmvnorm(beta.slice(d).col(j).t(), theta.slice(d).col(zj), lam2(zj,d)*Ip, true).at(0);
          // elements in S
          for (int k = 0; k < nS; k++) {
            log_L += dmvnorm(beta.slice(d).col(S(k)).t(), theta_m.col(d), lam2_m(d)*Ip, true).at(0);
            if (z(S(k))== zi) {
              log_L -= dmvnorm(beta.slice(d).col(S(k)).t(), theta.slice(d).col(zi), lam2(zi,d)*Ip, true).at(0);
            } else {
              log_L -= dmvnorm(beta.slice(d).col(S(k)).t(), theta.slice(d).col(zj), lam2(zj,d)*Ip, true).at(0);
            }
          }
        }
        
        
        
        /* prior log ratio (Eq. (8)) */
        double log_p = 0;
        
        if (fmmx == 1) {
          // log_p -= log(pj(zi)) + log(pj(zj));
          // for (int k = 0; k < nS; k++) log_p -= log(pj(z(S(k))));
          log_p += (double)nS2*log(pj(zj));
          log_p -= (double)nj(zi)*log(pj(zi)) + (double)nj(zj)*log(pj(zj));
        }
        
        if (ppmx == 1) {
          
          // cohesions
          log_p += log_M;
          for (int k = 1; k < nS2; k++) { log_p += log((double)k); }
          for (int k = 1; k < nj(zi); k++) { log_p -= log((double)k); }
          for (int k = 1; k < nj(zj); k++) { log_p -= log((double)k); }
          
          // similarity functions (continuous covariates)
          if (covariates_con == 1) {
            double sumxj_m = sumxj(zi) + sumxj(zj);
            double sumx2j_m = sumx2j(zi) + sumx2j(zj);
            log_p += gsim_con(nj_m, v0, sumxj_m, sumx2j_m);
            log_p -= gsim_con(nj(zi), v0, sumxj(zi), sumx2j(zi));
            log_p -= gsim_con(nj(zj), v0, sumxj(zj), sumx2j(zj));
          }
          
          // similarity functions (categorical covariates)
          if (covariates_cat == 1) {
            // arma::Col<int> ncatj_m = ncatj.col(zi) + ncatj.col(zj);
            // log_p += gsim_cat(nj_m, alpha_cat, ncat, ncatj_m);
            // log_p -= gsim_cat(nj(zi), alpha_cat, ncat, ncatj.col(zi));
            // log_p -= gsim_cat(nj(zj), alpha_cat, ncat, ncatj.col(zj));
            for (int t = 0; t < Ncat; t++) {
              arma::Col<int> ncatj_m = ncatJ.slice(t).col(zi) + ncatJ.slice(t).col(zj);
              log_p += gsim_cat(nj_m, alpha_cat, ncat(t), ncatj_m);
              log_p -= gsim_cat(nj(zi), alpha_cat, ncat(t), ncatJ.slice(t).col(zi));
              log_p -= gsim_cat(nj(zj), alpha_cat, ncat(t), ncatJ.slice(t).col(zj));
            }
          }
          
        }
        
        
        for (int d = 0; d < D; d++) {
          log_p +=
            dmvnorm(theta_m.col(d).t(), mu.col(d), tau2_m(d)*Ki, true).at(0) +
            -0.5*log(lam2_m(d)) - log(2.0*A);
            log_p -=
            dmvnorm(theta.slice(d).col(zi).t(), mu.col(d), tau2(zi,d)*Ki, true).at(0) +
            dmvnorm(theta.slice(d).col(zj).t(), mu.col(d), tau2(zj,d)*Ki, true).at(0) +
            -0.5*log(lam2(zi,d)) - log(2.0*A) +
            -0.5*log(lam2(zj,d)) - log(2.0*A);
            // if PPMx include tau2 prior densities
            if (ppmx==1) {
              log_p += dinvgamma_log(tau2_m(d) , at, btD(d));
              log_p -= dinvgamma_log(tau2(zi,d), at, btD(d));
              log_p -= dinvgamma_log(tau2(zj,d), at, btD(d));
            }
        }
        
        
        
        /* prior log ratio (Eq. (8)) - Repulsive factor */
        if (fmmx==1) {
          if (repulsive != 0) {
            
            double log_RF = 0;
            
            for (int d = 0; d < D; d++) {
              
              // theta matrix without columns zi and zj
              arma::mat theta_ = theta.slice(d);
              theta_.shed_col(zj);
              
              // difference between curve zj and other curves
              arma::mat df_m(n_theta,J-1);
              for (int h = 0; h < J-1; h++) df_m.col(h) = Ht0*(theta_m.col(d) - theta_.col(h));
              arma::mat df(n_theta,J-1);
              for (int h = 0; h < J-1; h++) df.col(h) = Ht0*(theta.slice(d).col(zj) - theta_.col(h));
              
              // repulsive factor (log)
              log_RF += logRF(repulsive, phi, nu, r, n_theta, df_m);
              log_RF -= logRF(repulsive, phi, nu, r, n_theta, df);
              
            }
            
            log_p += log_RF;
            
          }
        }
        
        
        // Rprintf("zi = %i , nj(%i) = %i (antes do MH)\n", zi,zi,nj(zi));
        // Rprintf("zj = %i , nj(%i) = %i (antes do MH)\n", zj,zj,nj(zj));
        
        /* M-H */
        double log_alpha = log_q + log_p + log_L;
        double u = Rf_runif(0.0,1.0);
        if (log(u) < log_alpha) {
          z = z_merge;
          
          // parameters update
          for (int d = 0; d < D; d++) {
            if (ppmx==1) {
              //  merged cluster zj
              tau2(zj,d) = tau2_m(d);
              theta.slice(d).col(zj) = theta_m.col(d);
              lam2(zj,d) = lam2_m(d);
            }
            if (fmmx==1) {
              // merged cluster zj
              theta.slice(d).col(zj) = theta_m.col(d);
              lam2(zj,d) = lam2_m(d);
              // empty cluster zi
              arma::vec m_zi = mu.col(d);
              arma::mat V_zi = tau2(zi,d)*Ki;
              theta.slice(d).col(zi) = rmvnorm(1, mu.col(d) , tau2(zi,d)*Ki).t();
              double lam = 0.0;
              while (lam < lam_min) lam = Rf_runif(0.0,A);
              lam2(zi,d) = lam*lam;
            }
          }
          
          // stats update
          nj(zj) = nj_m;
          nj(zi) = 0;
          
          if (ppmx == 1) {
            
            if (covariates_con == 1) {
              sumxj(zj) += sumxj(zi);
              sumx2j(zj) += sumx2j(zi);
              sumxj(zi) = 0;
              sumx2j(zi) = 0;
            }
            
            if (covariates_cat == 1) {
              // ncatj.col(zj) += ncatj.col(zi);
              // ncatj.col(zi).zeros();
              for (int t = 0; t < Ncat; t++) {
                
                ncatJ.slice(t).col(zj) += ncatJ.slice(t).col(zi);
                
                // Rprintf("cat%i ANTES\n", t);
                // for (int i = 0; i < ncat(t); i++) Rprintf("ncatJ(%i,%i,%i)=%i\n", i,zi,t,ncatJ(i,zi,t));
                
                ncatJ.slice(t).col(zi).zeros();
                
                // Rprintf("cat%i DEPOIS\n", t);
                // for (int i = 0; i < ncat(t); i++) Rprintf("ncatJ(%i,%i,%i)=%i\n", i,zi,t,ncatJ(i,zi,t));
                // Rprintf("\n");
                
              }
            }
            
            // -> move the max cluster index to the (now) empty cluster zi
            // (only required if the (now) empty cluster zi is not the max cluster index)
            int max_j = nclr - 1;
            if (zi != max_j) {
              
              // -> relabel z
              for (int i = 0; i < m; i++) { if (z(i) == max_j) z(i) = zi; }
              // -> move cluster-specific parameters from cluster max_j to zi
              for (int d = 0; d < D; d++) {
                tau2(zi,d) = tau2(max_j,d);
                lam2(zi,d) = lam2(max_j,d);
                theta.slice(d).col(zi) = theta.slice(d).col(max_j);
              }
              // -> move cluster stats from cluster max_j to zi
              nj(zi) = nj(max_j);
              if (covariates_con == 1) {
                sumxj(zi) = sumxj(max_j);
                sumx2j(zi) = sumx2j(max_j);
              }
              // if (covariates_cat == 1) ncatj.col(zi) = ncatj.col(max_j);
              if (covariates_cat == 1) {
                for (int t = 0; t < Ncat; t++) {
                  ncatJ.slice(t).col(zi) = ncatJ.slice(t).col(max_j);
                }
              }
              // -> set stats of the max index cluster to empty:
              // (max index cluster does not exist anymore)
              nj(max_j) = 0;
              if (covariates_con == 1) {
                sumxj(max_j) = 0;
                sumx2j(max_j) = 0;
              }
              // if (covariates_cat == 1) ncatj.col(max_j).zeros();
              if (covariates_cat == 1) {
                for (int t = 0; t < Ncat; t++) {
                  ncatJ.slice(t).col(max_j).zeros();
                }
              }
              
              
            }
            
            // -> reduce the number of clusters by one
            nclr -= 1;
            J = nclr;
            
          }
          
          // Rprintf("sumxj(%i) = %f (MH merge)\n", zj, sumxj(zj));
          // Rprintf("nclr (M-H merge) = %i (b=%i)\n",nclr,b);
          // Rprintf("MH merge step\n");
          if (ppmx==1 && arma::max(z) != arma::min( arma::find(nj==0) )-1) {
            Rprintf("zi = %i , nj(%i) = %i\n", zi,zi,nj(zi));
            Rprintf("zj = %i , nj(%i) = %i\n", zj,zj,nj(zj));
            Rprintf("arma::max(z) = %i\n", arma::max(z));
            Rprintf("arma::min( arma::find(nj==0) ) = %i\n", arma::min( arma::find(nj==0) ));
            stop("arma::max(z) != arma::min( arma::find(nj==0) )-1 (merge proposal)\n");
          }
        }
        
        // Rprintf("z (merged):\n");
        // for (int k = 0; k < n; k++) Rprintf("nj(%i) = %i\n", k,nj(k));
        // Rprintf("\n");
        
        if (sum(nj) != m) stop("sum(nj) != n (merge proposal)\n");
        
        // Rprintf("merge step\n");
        // Rprintf("arma::max(z) = %i\n", arma::max(z));
        // Rprintf("arma::min( arma::find(nj==0) ) = %i\n", arma::min( arma::find(nj==0) ));
        // if (arma::max(z) != arma::min( arma::find(nj==0) )-1) {
        //   stop("arma::max(z) != arma::min( arma::find(nj==0) )-1 (merge proposal)\n");
        // }
        
        
        
      } // end of (5) merge proposal
      
      // Rprintf("SM8\n");
      // Rprintf("SM8: nclr=%i\n", nclr);
      
      // if (b > burn) {
      //   if ((b-burn) % thin == 0) {
      //     Rprintf("nclr = %i (after s-m)\n", nclr);
      //   }
      // }
      
      
    } // end if SM


    
    
    
    //////////* z sample (neal8) *//////////

    // PPMx (Neal 8)
    if (ppmx == 1) {
      
      // Rprintf("nclr = %i (before neal8)\n", nclr);
      
      for (int i = 0; i < m; i++) {
        
        arma::mat Hi = H[i];
        arma::mat HHi = HH.slice(i);
        arma::mat Yi = Y[i];

        int nc = nclr;
        int zi = z(i);
        nj(zi) -= 1;

        if (covariates_con == 1) {
          sumxj(zi) -= Xcon(i);
          sumx2j(zi) -= X2con(i);
        }
        
        // if (covariates_cat == 1) ncatj(Xcat(i),zi) -= 1;
        if (covariates_cat == 1) {
          for (int t = 0; t < Ncat; t++) ncatJ(Xcat(i,t),zi,t) -= 1;
        }
        
        
        // sample of the parameters for the new proposed clusters
        arma::mat lam2_new(m_neal,D);
        arma::mat tau2_new(m_neal,D);
        arma::cube theta_new(P,m_neal,D);
        
        for (int j = 0; j < m_neal; j++) {
          for (int d = 0; d < D; d++) {
            tau2_new(j,d) = 1/Rf_rgamma(at,1/btD(d));
            theta_new.slice(d).col(j) = rmvnorm(1, mu.col(d) , tau2_new(j,d)*Ki).t();
            double lam = 0.0;
            while (lam < lam_min) lam = Rf_runif(0.0,A);
            lam2_new(j,d) = lam*lam;
          }
        }
        

        // vector of cluster indicators
        arma::uvec hvec = jvec.subvec(0,nc+m_neal-1);
        // Rprintf("nc = %i\n", nc);
        // Rprintf("nclr = %i\n", nclr);
        // Rprintf("hvec.size = %i\n", hvec.size());
        // Rprintf("\n");

        // if not singleton //////////////////////////////////////////
        if (nj(zi) > 0) {

          // density objects
          arma::vec fbeta_log(nc+m_neal);
          arma::vec fbeta(nc+m_neal);
          fbeta_log.zeros();

          // weights of the new proposed clusters
          for (int j = 0; j < m_neal; j++) {
            // beta density
            for (int d = 0; d < D; d++) {
              // fbeta_log(nc+j) += dmvnorm_log(beta.slice(d).col(i), theta_new.slice(d).col(j), lam2_new(j,d), P);
              // fbeta_log(nc+j) += fbeta_log_int(H, HH, Y, beta0, sig2, theta_new.slice(d).col(j), lam2_new(j,d), i, d, P, Ip);
              fbeta_log(nc+j) += fbeta_log_int(Hi, HHi, Yi, beta0, sig2, theta_new.slice(d).col(j), lam2_new(j,d), i, d, P, Ip);
            }
            // cohesion
            fbeta_log(nc+j) += log_M - log_m;
            // similarity functions of continuous covariates
            if (covariates_con == 1) {
              double gsim_con_new = -0.5 * ( log(v0+1) + X2con(i)/(v0+1) );
              fbeta_log(nc+j) += gsim_con_new;
            }
            // similarity functions of categorical covariates
            if (covariates_cat == 1) {
              // fbeta_log(nc+j) += gsim_cat_new;
              for (int t = 0; t < Ncat; t++) {
                fbeta_log(nc+j) += gsim_cat_new(t);
              }
            }
            
          }
          

          // weights of the existing clusters
          for (int j = 0; j < nc; j++) {
            // beta density
            for (int d = 0; d < D; d++) {
              // fbeta_log(j) += dmvnorm_log(beta.slice(d).col(i), theta.slice(d).col(j), lam2(j,d), P);
              // fbeta_log(j) += fbeta_log_int(H, HH, Y, beta0, sig2, theta.slice(d).col(j), lam2(j,d), i, d, P, Ip);
              fbeta_log(j) += fbeta_log_int(Hi, HHi, Yi, beta0, sig2, theta.slice(d).col(j), lam2(j,d), i, d, P, Ip);
            }
            // Rprintf("fbeta_log1(%i) = %f\n", j,fbeta_log(j));
            // cohesion
            fbeta_log(j) += log((double)nj(j));
            // Rprintf("fbeta_log2(%i) = %f\n", j,fbeta_log(j));
            // similarity functions of continuous covariates
            if (covariates_con == 1) {
              fbeta_log(j) += gsim_con_gibbs( nj(j), v0, Xcon(i), X2con(i), sumxj(j) );
              // Rprintf("fbeta_log3(%i) = %f\n", j,fbeta_log(j));
            }
            // similarity functions of categorical covariates
            if (covariates_cat == 1) {
              // fbeta_log(j) += gsim_cat_gibbs( nj(j), alpha_cat, ncat, ncatj(Xcat(i),j) );
              for (int t = 0; t < Ncat; t++) {
                fbeta_log(j) += gsim_cat_gibbs( nj(j), alpha_cat, ncat(t), ncatJ(Xcat(i,t),j,t) );
              }
              // Rprintf("fbeta_log4(%i) = %f\n", j,fbeta_log(j));
            }
          }


          // standardized weights
          double max_log_i = max_log - max(fbeta_log) - log((double)nc+(double)m_neal);
          fbeta = exp( fbeta_log + max_log_i );
          fbeta = fbeta/sum(fbeta);


          // cluster sample
          int zi_sample = RcppArmadillo::sample(hvec, 1, true, fbeta).at(0);

          // "nc" used bellow is equal to the smaller index of the new proposed clusters

          // if a new cluster is sampled:
          if (zi_sample >= nc) {
            // -> set the new cluster id equal to nclr (that is the current smaller id for a new cluster):
            z(i) = nc;
            // -> set the number of elements of the new cluster equal to one:
            nj(nc) = 1;
            
            // -> update covariate stats
            if (covariates_con == 1) {
              
              // // if (sumxj(nc)!=0 || sumx2j(nc)!=0) stop("sumxj(nc)!=0 || sumx2j(nc)!=0\n");
              // NumericVector snc(1);
              // snc(0) = sumxj(nc);
              // arma::vec rnd = round(snc,10);
              // NumericVector s2nc(1);
              // s2nc(0) = sumx2j(nc);
              // arma::vec rnd2 = round(s2nc,10);
              // if (rnd(0)!=0.0 || rnd2(0)!=0.0) {
              //   Rprintf("sumxj(nc)=%.20f\n", sumxj(nc));
              //   Rprintf("sumx2j(nc)=%.20f\n", sumx2j(nc));
              //   Rprintf("snc(0)=%.20f\n", snc(0));
              //   Rprintf("s2nc(0)=%.20f\n", s2nc(0));
              //   stop("sumxj(nc)!=0 || sumx2j(nc)!=0\n");
              // }
              
              sumxj(nc) = Xcon(i);
              sumx2j(nc) = X2con(i);
              
            }
            if (covariates_cat == 1) {
              
              // ncatj(Xcat(i),nc) = 1;
              
              for (int t = 0; t < Ncat; t++) {
                
                if (ncatJ(Xcat(i,t),nc,t) != 0) {
                  Rprintf("antes tem que ser 0:\n");
                  Rprintf("ncatJ(%i,%i,%i) = %i (antes)\n", Xcat(i,t),nc,t,ncatJ(Xcat(i,t),nc,t));
                }
                
                ncatJ(Xcat(i,t),nc,t) = 1; 
                
                if (ncatJ(Xcat(i,t),nc,t) != 1) {
                  Rprintf("depois tem que ser 1:\n");
                  Rprintf("ncatJ(%i,%i,%i) = %i (depois)\n", Xcat(i,t),nc,t,ncatJ(Xcat(i,t),nc,t));
                }
                
              }
            }
            
            // -> save the new cluster-specific parameter values:
            int j_new = zi_sample - nc;
            for (int d = 0; d < D; d++) {
              tau2(nc,d) = tau2_new(j_new,d);
              lam2(nc,d) = lam2_new(j_new,d);
              theta.slice(d).col(nc) = theta_new.slice(d).col(j_new);
            }
            // -> increase the number of clusters by one:
            nclr += 1;
            J = nclr;
          }

          // if some existing cluster is sampled:
          if (zi_sample < nc) {
            // -> set z(i) equal to the sampled cluster zi
            z(i) = zi_sample;
            // -> increase the number of elements of the sampled cluster by one
            nj(zi_sample) += 1;

            // -> update covariate stats
            if (covariates_con == 1) {
              sumxj(zi_sample) += Xcon(i);
              sumx2j(zi_sample) += X2con(i);
            }
            
            // if (covariates_cat == 1) ncatj(Xcat(i),zi_sample) += 1; 
            if (covariates_cat == 1) {
              for (int t = 0; t < Ncat; t++) {
                ncatJ(Xcat(i,t),zi_sample,t) += 1;
              }
            }
            
            // -> do not increase the number of clusters by one:
            J = nclr;

          }
          
          // Rprintf("nclr = %i (after neal8 not singleton)\n", nclr);

        } // if not singleton
        
        // if singleton //////////////////////////////////////////////
        if (nj(zi) == 0) {
          
          // Rprintf("nclr = %i (before)\n", nclr);
          // Rprintf("hvec.size = %i\n", hvec.size());
          // Rprintf("zi = %i\n", zi);
          

          // remove zi
          hvec.shed_row(zi);
          nc -= 1;


          // density objects
          arma::vec fbeta_log(nc+m_neal);
          arma::vec fbeta(nc+m_neal);
          fbeta_log.zeros();


          // weights of the new proposed clusters
          for (int j = 0; j < m_neal; j++) {
            // beta density
            for (int d = 0; d < D; d++) {
              // fbeta_log(nc+j) += dmvnorm_log(beta.slice(d).col(i), theta_new.slice(d).col(j), lam2_new(j,d), P);
              // fbeta_log(nc+j) += fbeta_log_int(H, HH, Y, beta0, sig2, theta_new.slice(d).col(j), lam2_new(j,d), i, d, P, Ip);
              fbeta_log(nc+j) += fbeta_log_int(Hi, HHi, Yi, beta0, sig2, theta_new.slice(d).col(j), lam2_new(j,d), i, d, P, Ip);
            }
            // cohesion
            fbeta_log(nc+j) += log_M - log_m;
            // similarity functions of continuous covariates
            if (covariates_con == 1) {
              double gsim_con_new = -0.5 * ( log(v0+1) + X2con(i)/(v0+1) );
              fbeta_log(nc+j) += gsim_con_new;
            }
            // similarity functions of categorical covariates
            if (covariates_cat == 1) {
              // fbeta_log(nc+j) += gsim_cat_new;
              for (int t = 0; t < Ncat; t++) {
                fbeta_log(nc+j) += gsim_cat_new(t);
              }
            }
          }


          // weights of the existing clusters
          for (int j = 0; j < nc; j++) {

            int h = hvec(j);

            // beta density
            for (int d = 0; d < D; d++) {
              // fbeta_log(j) += dmvnorm_log(beta.slice(d).col(i), theta.slice(d).col(h), lam2(h,d), P);
              // fbeta_log(j) += fbeta_log_int(H, HH, Y, beta0, sig2, theta.slice(d).col(h), lam2(h,d), i, d, P, Ip);
              fbeta_log(j) += fbeta_log_int(Hi, HHi, Yi, beta0, sig2, theta.slice(d).col(h), lam2(h,d), i, d, P, Ip);
            }
            // cohesion
            fbeta_log(j) += log((double)nj(h));
            // similarity functions of continuous covariates
            if (covariates_con == 1) {
              fbeta_log(j) += gsim_con_gibbs( nj(h), v0, Xcon(i), X2con(i), sumxj(h) );
            }
            // similarity functions of categorical covariates
            if (covariates_cat == 1) {
              // fbeta_log(j) += gsim_cat_gibbs( nj(h), alpha_cat, ncat, ncatj(Xcat(i),h) );
              for (int t = 0; t < Ncat; t++) {
                fbeta_log(j) += gsim_cat_gibbs( nj(h), alpha_cat, ncat(t), ncatJ(Xcat(i,t),h,t) );
              }
            }
          }

          // standardized weights
          double max_log_i = max_log - max(fbeta_log) - log((double)nc+(double)m_neal);
          fbeta = exp( fbeta_log + max_log_i );
          fbeta = fbeta/sum(fbeta);

          // cluster sample
          int zi_sample = RcppArmadillo::sample(hvec, 1, true, fbeta).at(0);

          nc += 1; // "nc" used bellow is equal to the smaller index of the new proposed clusters

          // if a new cluster is sampled:
          if (zi_sample >= nc ) {
            // -> set the new cluster index to be the removed cluster index:
            // z(i) = zi; // (not necessary because z(i) is already equal to zi)
            // -> set the number of elements of the new cluster equal to one:
            nj(zi) = 1;
            
            // -> update covariate stats
            if (covariates_con == 1) {
              
              // // if (sumxj(zi)!=0 || sumx2j(zi)!=0) stop("sumxj(zi)!=0 || sumx2j(zi)!=0\n");
              // NumericVector szi(1);
              // szi(0) = sumxj(zi);
              // arma::vec rnd = round(szi,10);
              // NumericVector s2zi(1);
              // s2zi(0) = sumx2j(zi);
              // arma::vec rnd2 = round(s2zi,10);
              // 
              // Rprintf("\n");
              // Rprintf("b=%i\n", b);
              // Rprintf("i=%i\n", i);
              // Rprintf("zi=%i\n", zi);
              // Rprintf("J=%i\n", J);
              // Rprintf("nclr=%i\n", nclr);
              // Rprintf("\n");
              // for (int j = 0; j < J; j++) Rprintf("nj(%i)=%i\n", j,nj(j));
              // Rprintf("\n");
              // for (int j = 0; j < J; j++) Rprintf("sumxj(%i)=%.60f\n", j,sumxj(j));
              // Rprintf("\n");
              // for (int j = 0; j < J; j++) Rprintf("sumx2j(%i)=%.60f\n", j,sumx2j(j));
              // Rprintf("\n");
              // 
              // if (rnd(0)!=0.0 || rnd2(0)!=0.0) {
              //   Rprintf("rnd(0)=%.60f\n", rnd(0));
              //   Rprintf("rnd2(0)=%.60f\n", rnd2(0));
              //   Rprintf("sumxj(zi)=%.60f\n", sumxj(zi));
              //   Rprintf("sumx2j(zi)=%.60f\n", sumx2j(zi));
              //   Rprintf("szi(0)=%.60f\n", szi(0));
              //   Rprintf("s2zi(0)=%.60f\n", s2zi(0));
              //   stop("sumxj(zi)!=0 || sumx2j(zi)!=0\n");
              // }

              sumxj(zi) = Xcon(i);
              sumx2j(zi) = X2con(i);
              
              // Rprintf("sumxj(%i)=%.60f\n", zi,sumxj(zi));
              // Rprintf("sumx2j(%i)=%.60f\n", zi,sumx2j(zi));
              // Rprintf("\n");
              
            }
            if (covariates_cat == 1) {

              for (int t = 0; t < Ncat; t++) {
                
                if (ncatJ(Xcat(i,t),zi,t) != 0) {
                  Rprintf("antes tem que ser 0:\n");
                  Rprintf("ncatJ(%i,%i,%i) = %i (antes)\n", Xcat(i,t),zi,t,ncatJ(Xcat(i,t),zi,t));
                }
                
                ncatJ(Xcat(i,t),zi,t) = 1; 
                
                if (ncatJ(Xcat(i,t),zi,t) != 1) {
                  Rprintf("depois tem que ser 1:\n");
                  Rprintf("ncatJ(%i,%i,%i) = %i (depois)\n", Xcat(i,t),zi,t,ncatJ(Xcat(i,t),zi,t));
                }

              }
              

            }
            // -> save the new cluster-specific parameter values:
            int j_new = zi_sample - nc;
            for (int d = 0; d < D; d++) {
              tau2(zi,d) = tau2_new(j_new,d);
              lam2(zi,d) = lam2_new(j_new,d);
              theta.slice(d).col(zi) = theta_new.slice(d).col(j_new);
            }
            
            // Rprintf("another singleton sampled\n");
            
          }

          // if some existing cluster is sampled:
          if (zi_sample < nc) {

            // -> set z(i) equal to the sampled cluster:
            z(i) = zi_sample;
            // -> increase the number of elements of the sampled cluster by one:
            nj(zi_sample) += 1;
            
            
            // -> update covariate stats
            if (covariates_con == 1) {
              sumxj(zi_sample) += Xcon(i);
              sumx2j(zi_sample) += X2con(i);
            }
            // if (covariates_cat == 1) ncatj(Xcat(i),zi_sample) += 1;
            if (covariates_cat == 1) {
              for (int t = 0; t < Ncat; t++) {
                ncatJ(Xcat(i,t),zi_sample,t) += 1;
              }
            }
            


            // -> relabel the max cluster index to be the removed cluster index zi
            // and set the number of elements of the max index cluster equal to zero:
            // (it is only required if the removed cluster was not the one with the max cluster index)
            int max_j = nc - 1;
            
            // Rprintf("nclr = %i (antes)\n",nclr);
            // Rprintf("max_j = %i\n",max_j);
            
            if (zi != max_j) {
              
              // relabel z
              for (int i = 0; i < m; i++) { if (z(i) == max_j) { z(i) = zi; }
              }
              // -> move cluster-specific parameters from the max index cluster max_j
              // to the removed cluster index zi:
              for (int d = 0; d < D; d++) {
                tau2(zi,d) = tau2(max_j,d);
                lam2(zi,d) = lam2(max_j,d);
                theta.slice(d).col(zi) = theta.slice(d).col(max_j);
              }
              nj(zi) = nj(max_j);
              if (covariates_con == 1) {
                sumxj(zi) = sumxj(max_j);
                sumx2j(zi) = sumx2j(max_j);
              }
              // if (covariates_cat == 1) ncatj.col(zi) = ncatj.col(max_j);
              if (covariates_cat == 1) {
                for (int t = 0; t < Ncat; t++) {
                  ncatJ.slice(t).col(zi) = ncatJ.slice(t).col(max_j);
                }
              }
              // -> set stats of the max index cluster to empty:
              // (max index cluster does not exist anymore)
              nj(max_j) = 0;
              if (covariates_con == 1) {
                sumxj(max_j) = 0;
                sumx2j(max_j) = 0;
              }
              // if (covariates_cat == 1) ncatj.col(max_j).zeros();
              if (covariates_cat == 1) {
                for (int t = 0; t < Ncat; t++) {
                  ncatJ.slice(t).col(max_j).zeros();
                }
              }
              
            }
            // -> reduce the number of clusters by one
            nclr -= 1;
            J = nclr;
            // Rprintf("nclr = %i (depois)\n",nclr);
          }
          
          // Rprintf("nclr = %i (after neal8 singleton)\n", nclr);
          
          // Rprintf("nclr = %i (after)\n", nclr);
          // Rprintf("\n");

        } // if singleton

      } // for i = 1,...,m

      // Rprintf("J = %i\n", J);
      

      if (arma::max(z) != arma::min( arma::find(nj==0) )-1) {
        Rprintf("\n");
        Rprintf("neal8\n");
        Rprintf("arma::max(z) = %i\n", arma::max(z));
        Rprintf("arma::min( arma::find(nj==0) ) = %i\n", arma::min( arma::find(nj==0) ));
        stop("arma::max(z) != arma::min( arma::find(nj==0) )-1 (merge proposal)\n");
      }
      
      // Rprintf("nclr = %i (after neal8)\n", nclr);

    } // if ppmx
    // Rprintf("nclr_neal8=%i\n",nclr);
    
    // if (b > burn) {
    //   if ((b-burn) % thin == 0) {
    //     Rprintf("nclr = %i (after neal8)\n", nclr);
    //   }
    // }
    

    // FMMx
    if (fmmx == 1) {
      
      // Rprintf("nclr = %i (before LSBP)\n", nclr);

      ////* z sample *////
      if (covariates_fmm == 1) { // LSBP strategy (Rigon etal 2021)

        for (int j = 0; j < J-1; j++) {

          arma::vec sum_x(ncov);
          arma::mat sum_wxTx(ncov,ncov);
          sum_wxTx.zeros();
          sum_x.zeros();

          // w sample
          for (int i = 0; i < m; i++) {

            if (z(i) >= j) {
              double xi_alpha = arma::as_scalar( X.row(i)*alpha.col(j) );
              wi = samplepg(xi_alpha);
              // sums for the posterior distribution of alpha(j)
              sum_wxTx += (wi * xTx.slice(i));
              if (z(i) == j) {
                sum_x += X.row(i).t();
              } else {
                sum_x -= X.row(i).t();
              }
            } // end if
          } // end for i

          // alpha sample
          arma::mat V_alpha = (Si_alpha + sum_wxTx).i();
          arma::vec m_alpha = V_alpha * (Si_mu_alpha + sum_x/2);
          alpha.col(j) = rmvnorm(1, m_alpha, V_alpha).t();

        } // end for j


        // z sample
        nj.zeros();
        for (int i = 0; i < m; i++) {
          
          arma::mat Hi = H[i];
          arma::mat HHi = HH.slice(i);
          arma::mat Yi = Y[i];

          // cluster probabilities for individual i
          arma::vec fbeta_log(J);
          arma::vec fbeta(J);
          fbeta_log.zeros();
          double xi_alpha;
          pj.zeros();

          // pj for j = 0
          xi_alpha = arma::as_scalar( X.row(i) * alpha.col(0) );
          pj(0) = 1 / (1 + exp(-xi_alpha));

          // pj for j = 1,...,J-2
          for (int j = 1; j < J-1; j++) {

            xi_alpha = arma::as_scalar( X.row(i) * alpha.col(j) );
            pj(j) = 1 / (1 + exp(-xi_alpha));

            for (int l = 0; l < j; l++) {
              xi_alpha = arma::as_scalar( X.row(i) * alpha.col(l) );
              double vl = 1 / (1 + exp(-xi_alpha));
              pj(j) *= 1 - vl;
            }

          }

          // pj for j = J-1
          pj(J-1) = 1;
          for (int l = 0; l < J-1; l++) {
            xi_alpha = arma::as_scalar( X.row(i) * alpha.col(l) );
            double vl = 1 / (1 + exp(-xi_alpha));
            pj(J-1) *= 1 - vl;
          }

          // weights of components
          for (int j = 0; j < J; j++) {
            // beta density
            for (int d = 0; d < D; d++) {
              // fbeta_log(j) += dmvnorm_log(beta.slice(d).col(i), theta.slice(d).col(j), lam2(j,d), P);
              // fbeta_log(j) += fbeta_log_int(H, HH, Y, beta0, sig2, theta.slice(d).col(j), lam2(j,d), i, d, P, Ip);
              fbeta_log(j) += fbeta_log_int(Hi, HHi, Yi, beta0, sig2, theta.slice(d).col(j), lam2(j,d), i, d, P, Ip);
            }
            // weights pj
            fbeta_log(j) += log(pj(j));
          }

          // standardized weights
          // double max_log_i = max_log_J - max(fbeta_log);
          double max_log_i = max_log - max(fbeta_log) - log(J);
          fbeta = exp( fbeta_log + max_log_i );
          fbeta = fbeta/sum(fbeta);

          // component sample
          z(i) = RcppArmadillo::sample(hvec_J, 1, true, fbeta).at(0);
          nj(z(i)) += 1;
        }


      } else { // no covariates

        /* pj sample (BDA3, page 585) (no covariates) */
        for (int j = 0; j < J; j++) { pj(j) = Rf_rgamma( alpha_dir+(double)nj(j) , 1 ); }
        pj = pj/sum(pj);


        /* z sample (no covariates) */
        nj.zeros();
        for (int i = 0; i < m; i++) {
          
          arma::mat Hi = H[i];
          arma::mat HHi = HH.slice(i);
          arma::mat Yi = Y[i];

          arma::vec fbeta_log(J);
          arma::vec fbeta(J);
          fbeta_log.zeros();
          
          // weights of components
          for (int j = 0; j < J; j++) {
            
            // weights pj
            fbeta_log(j) += log(pj(j));
            
            for (int d = 0; d < D; d++) {
              
              // // beta density
              // // fbeta_log(j) += arma::as_scalar(dmvnorm(beta.slice(d).col(i).t(), theta.slice(d).col(j), lam2(j,d)*Ip, true));
              // fbeta_log(j) += dmvnorm_log(beta.slice(d).col(i), theta.slice(d).col(j), lam2(j,d), P);
              
              // beta integrated out
              // fbeta_log(j) += fbeta_log_int(H, HH, Y, beta0, sig2, theta.slice(d).col(j), lam2(j,d), i, d, P, Ip);
              fbeta_log(j) += fbeta_log_int(Hi, HHi, Yi, beta0, sig2, theta.slice(d).col(j), lam2(j,d), i, d, P, Ip);

            }
            
          }

          // standardized weights of components
          double max_log_i = max_log - max(fbeta_log) - log(J);
          fbeta = exp( fbeta_log + max_log_i );
          fbeta = fbeta/sum(fbeta);
          
          // z sample
          z(i) = RcppArmadillo::sample(hvec_J, 1, true, fbeta).at(0);
          nj(z(i)) += 1;
          
        }
        
      }
      
      nclr = 0;
      for (int j = 0; j < J; j++) { if (nj(j) != 0) nclr += 1; }
      
      // Rprintf("nclr = %i (after LSBP)\n", nclr);

    } // if fmmx
    
    
    
    
    
    //////////* theta sample *//////////
    for (int d = 0; d < D; d++) {

      // if (d == 0) {
      //   Rprintf("antes:\n");
      //   for (int l = 0; l < P; l++) {
      //     Rprintf("theta(%i,0,d) = %f\n", l,theta(l,0,d));
      //   }
      //   Rprintf("\n");
      // }

      eps = epsD(d);
      eps0 = eps0D(d);
      update_theta(theta, accept_mat, accept_logratio_mat,
                   nj, lam2, tau2, beta0, sig2, mu, z, K, Ki,
                   m, d, J, P, Ip, Y, H, HH, n_theta, Ht0, theta_sampler,
                   repulsive, phi, nu, r, nr, eps, eps0);

      // if (d == 0) {
      //   Rprintf("depois:\n");
      //   for (int l = 0; l < P; l++) {
      //     Rprintf("theta(%i,0,d) = %f\n", l,theta(l,0,d));
      //   }
      //   Rprintf("\n");
      // }

    } // end for d
    // Rprintf("theta %i\n", b);
    // if (b % 10000 == 0) Rprintf("theta %i\n", b);
    
    
    // for (int j = 0; j < J; j++) {
    //   for (int l = 0; l < P; l++) Rprintf("theta(%i,%i,0)=%.10f\n", l,j,theta(l,j,0)); 
    // }
    

    
    //////////* eps update *//////////
    if (repulsive != 0) {
      if (b <= burn) {
        
        if (b % nadapt == 0) {
          
          arma::Mat<int> acc(J,D, arma::fill::zeros);
          for (int n = 0; n < nadapt; n++) acc += accept_cube.slice(n);
          
          arma::uvec jn = arma::find(nj>0);
          arma::uvec j0 = arma::find(nj==0);
          
          // for (int l = 0; l < nj.size(); l++) Rprintf("nj(%i)=%i\n",l,nj(l));
          // for (int l = 0; l < jn.size(); l++) Rprintf("jn(%i)=%i\n",l,jn(l));
          // for (int l = 0; l < j0.size(); l++) Rprintf("j0(%i)=%i\n",l,j0(l));
          // Rprintf("\n\n");
          
          for (int d = 0; d < D; d++) {
            
            arma::Col<int> accd = acc.col(d);
            
            // eps update
            double rmin = arma::min( accd.elem(jn) )/(double)nadapt;
            double rmax = arma::max( accd.elem(jn) )/(double)nadapt;
            if (rmin > 0.50) {
              epsD(d) *= 1.20;
            } else if (rmin < 0.05) {
              epsD(d) *= 0.95;
            } else if (rmax > 0.25) {
              epsD(d) *= 1.05;
            }
            
            // eps0 update
            if (j0.size() > 0) {
              
              double rmin0 = arma::min( accd.elem(j0) )/(double)nadapt;
              double rmax0 = arma::max( accd.elem(j0) )/(double)nadapt;
              if (rmin0 > 0.50) {
                eps0D(d) *= 1.20;
              } else if (rmin0 < 0.01) {
                eps0D(d) *= 0.95;
              } else if (rmax0 > 0.30) {
                eps0D(d) *= 1.05;
              }
              
              // Rprintf("b=%i\n",b);
              // Rprintf("rmin0=%.50f\n",rmin0);
              // Rprintf("rmax0=%.50f\n",rmax0);
              // Rprintf("eps0D(%i)=%.10f\n", d,eps0D(d));
              
            }
            
          } // end for d
          
          beps = 0;
          accept_cube.zeros();
          
        } else {
          
          accept_cube.slice(beps) = accept_mat;
          beps += 1;
          
        }
        
      }
    }
    // Rprintf("eps update %i\n", b);

    
    

    //////////* beta sample *//////////
    for (int i = 0; i < m; i++) {
      
      int zi = z(i);
      arma::mat Yi = Y[i];
      arma::mat Hi = H[i];
      // arma::mat HHi = HH[i];
      arma::mat HHi = HH.slice(i);
      
      // for (int d = 0; d < D; d++) {
      //   arma::vec yi = Yi.col(d) - beta0(i,d);
      //   arma::mat V_beta = arma::inv( HHi/sig2(i,d) + Ip/lam2(zi,d) );
      //   arma::vec m_beta = V_beta * ( Hi.t()*yi/sig2(i,d) + theta.slice(d).col(zi)/lam2(zi,d) );
      //   beta.slice(d).col(i) = rmvnorm(1, m_beta , V_beta).t();
      // }
      
      for (int d = 0; d < D; d++) {
        arma::vec yi = Yi.col(d) - beta0(i,d);
        arma::mat V_beta = arma::inv( lam2(zi,d)*HHi/sig2(i,d) + Ip );
        arma::vec m_beta = V_beta * ( lam2(zi,d)*Hi.t()*yi/sig2(i,d) + theta.slice(d).col(zi) );
        beta.slice(d).col(i) = rmvnorm_eigen( m_beta , V_beta*lam2(zi,d) , lam2(zi,d) , P).t();
      }
      
    }
    // Rprintf("beta %i\n", b);


    
    //////////* tau2 sample *//////////
    double a_tau2 = at + 0.5*(double)P;
    for (int d = 0; d < D; d++) {
      for (int j = 0; j < J; j++) {
        arma::vec theta_mu = theta.slice(d).col(j)-mu.col(d);
        double b_tau2 = btD(d) + 0.5*arma::as_scalar( theta_mu.t()*K*theta_mu );
        tau2(j,d) = 1/Rf_rgamma( a_tau2 , 1/b_tau2 );
      }
    }
    // Rprintf("tau2 %i\n", b);


    
    //////////* btD sample *//////////
    if (bt == 0) {
      double alp_bt = (double)J*at + at0;
      for (int d = 0; d < D; d++) {
        double sum_tau2 = 0.0;
        for (int j = 0; j < J; j++) sum_tau2 += 1/tau2(j,d);
        double scl_bt = 1.0 / (sum_tau2 + bt0);
        btD(d) = Rf_rgamma( alp_bt , scl_bt );
        // Rprintf("d = %i\n",J);
        // Rprintf("alp_bt = %f\n", alp_bt);
        // Rprintf("scl_bt = %f\n", scl_bt);
        // Rprintf("btD = %f\n", btD(d));
        // Rprintf("\n", btD(d));
      }
    }
    // Rprintf("bt %i\n", b);

    
    
    
    
    //////////* lam2 sample *//////////
    for (int d = 0; d < D; d++) {
      
      arma::vec beta_theta_clr(J);
      beta_theta_clr.zeros();
      
      for (int i = 0; i < m; i++) {
        int zi = z(i);
        arma::vec b_i = beta.slice(d).col(i)-theta.slice(d).col(zi);
        beta_theta_clr(zi) += arma::as_scalar( b_i.t()*b_i );
      }
      
      for (int j = 0; j < J; j++) {
        
        if (nj(j)>0) {
          
          int njP = nj(j)*P;
          double a_lam2 = 0.5*((double)njP-1);
          double b_lam2 = 0.5*beta_theta_clr(j);
          
          // // ARS to sample from w = log(p*x+1)
          // lam2(j,d) = rinvgamma_ars(njP, A2, b_lam2);
          
          // ARS to sample from w = 1/x
          lam2(j,d) = 1/rgamma_ars(A2, Lu, a_lam2, b_lam2);
          if (lam2(j,d) < lam2_min) stop("w > Lu");
          
          // Rprintf("lam2(%i,%i) = %.20f\n", j,d,lam2(j,d));
          // Rprintf("a_lam2 = %.20f\n", a_lam2);
          // Rprintf("b_lam2 = %.20f\n", b_lam2);
          // Rprintf("\n");
          
        } else {
          
          // double lam = Rf_runif(0.0,A);
          // lam2(j,d) = lam*lam;
          
          double lam = 0.0;
          while (lam < lam_min) lam = Rf_runif(0.0,A);
          lam2(j,d) = lam*lam;
          
        }
        
      }
      
    }
    // Rprintf("lam2 %i\n", b);
    
    

    
    //////////* mu sample *//////////
    if (mu_par == 1) {
      for (int d = 0; d < D; d++) {
        arma::vec sum_theta(P);
        sum_theta.zeros();
        double sum_tau2 = 0;
        for (int j = 0; j < J; j++) {
          sum_theta += theta.slice(d).col(j)/tau2(j,d);
          sum_tau2 += 1/tau2(j,d);
        }
        
        // arma::mat V_mu = arma::inv( K*sum_tau2 + Ip/s2mu );
        // arma::vec m_mu = V_mu * K * sum_theta;
        // mu.col(d) = rmvnorm(1, m_mu , V_mu).t();
        
        arma::mat V_mu = arma::inv( K*sum_tau2 + Ip/s2mu );
        arma::vec m_mu = V_mu * K * sum_theta;
        mu.col(d) = rmvnorm_eigen( m_mu , V_mu , 1.0 , P).t();
        
      }
    }
    // Rprintf("mu %i\n", b);

    
      
    
    
    //////////* beta0 and sig2 sample *//////////
    for (int i = 0; i < m; i++) {
      
      arma::mat Yi = Y[i];
      arma::mat Hi = H[i];
      int ni = Yi.n_rows;
      
      for (int d = 0; d < D; d++) {
        
        arma::vec yi = Yi.col(d);
        arma::vec HB = Hi*beta.slice(d).col(i);
        
        // beta0
        double V_beta0 = (sig02(d)*sig2(i,d))/((double)ni*sig02(d)+sig2(i,d));
        double m_beta0 = V_beta0 * (  sum(yi-HB)/sig2(i,d) + mu0(d)/sig02(d) );
        beta0(i,d) = Rf_rnorm( m_beta0 , sqrt(V_beta0) );
        
        // sig2
        if (Asig == 0) {
          
          double a_sig2 = as + 0.5*(double)ni;
          arma::vec fi = yi - beta0(i,d) - HB;
          double b_sig2 = bs + 0.5*arma::as_scalar( fi.t()*fi );
          sig2(i,d) = 1/Rf_rgamma( a_sig2 , 1/b_sig2 );
          
        } else {
          
          arma::vec fi = yi - beta0(i,d) - HB;
          double a_sig2 = as + 0.5*(double)ni;
          double b_sig2 = 0.5*arma::as_scalar( fi.t()*fi );
          
          // // ARS to sample from w = log(p*x+1)
          // sig2(i,d) = rinvgamma_ars(ni, A2sig, b_sig2);
          
          // ARS to sample from w = 1/x
          sig2(i,d) = 1/rgamma_ars(A2sig, Lu, a_sig2, b_sig2);
          
        }
        
      }
      
    }
    
    
    
    
    
    //////////* sig02 sample *//////////
    double a_sig02 = a0 + 0.5*(double)m;
    for (int d = 0; d < D; d++) {
      double b_sig02 = b0 + 0.5*arma::as_scalar((beta0.col(d)-mu0(d)).t()*(beta0.col(d)-mu0(d)));
      sig02(d) = 1/Rf_rgamma( a_sig02 , 1/b_sig02 );
    }
    
    
    
    //////////* mu0 sample *//////////
    for (int d = 0; d < D; d++) {
      double V_mu0 = (s02*sig02(d)) / ((double)m*s02 + sig02(d));
      double m_mu0 = V_mu0 * sum(beta0.col(d))/sig02(d);
      mu0(d) = Rf_rnorm( m_mu0 , sqrt(V_mu0) );
    }
    
    
    
    
    
    //////////* save sample *////////// 
    if (b > burn) {
      if ((b-burn) % thin == 0) {
        
        // // number of clusters (fmmx)
        // if (fmmx == 1) {
        //   nclr = 0;
        //   for (int j = 0; j < J; j++) { if (nj(j) != 0) nclr += 1; }
        // }
        // nclr when fmmx model needs now to be computed immediately
        // after z sample because it is used in the begining of JN (3)

        // parameter chains
        if (only_nclr == 0) {
          
          // individual-specific parameters
          beta_chain(s) = beta;
          mu0_chain.col(s) = mu0;
          sig02_chain.col(s) = sig02;
          beta0_chain.slice(s) = beta0;
          sig2_chain.slice(s) = sig2;
          alpha_chain.slice(s) = alpha;
          
          // cluster-specific parameters
          theta_chain(s) = theta;
          z_chain.row(s) = z;
          lam2_chain.slice(s) = lam2;
          tau2_chain.slice(s) = tau2;
          nj_chain.col(s) = nj;
          if (mu_par == 1) mu_chain.slice(s) = mu;
          if (bt == 0) bt_chain.col(s) = btD;
          
        }
        
        // number of clusters
        nclr_chain(s) = nclr;

        // acceptance
        if (repulsive != 0) {
          accept_logratio.slice(s) = accept_logratio_mat;
          accept.slice(s) = accept_mat;
        }
        
        // print s
        if (s % 100 == 0) Rprintf("nclr(%i) = %i\n", s, nclr);  
        s += 1;

      }
    }
    
    if (b == burn) Rprintf("burn finished :)\n\n");

  } // end while mcmc


  List out;
  
  if (only_nclr == 0) {
    
    // individual-specific parameters
    out["mu0"] = mu0_chain;
    out["sig02"] = sig02_chain;
    out["beta0"] = beta0_chain;
    out["sig2"] = sig2_chain;
    out["beta"] = beta_chain;
    out["alpha"] = alpha_chain;
    
    // cluster-specific parameters
    out["z"] = z_chain;
    out["theta"] = theta_chain;
    out["lam2"] = lam2_chain;
    out["tau2"] = tau2_chain;
    out["nj"] = nj_chain;
    out["K"] = K;
    out["Ki"] = Ki;
    if (mu_par == 1) out["mu"] = mu_chain;
    if (bt == 0) out["bt"] = bt_chain;

  }
  
  // number of clusters
  out["nclr"] = nclr_chain;
  out["only_nclr"] = only_nclr;

  // hyperparameters
  out["A"] = A;
  out["Asig"] = Asig;
  out["M"] = M;
  out["p"] = p;
  out["q"] = q;
  out["P"] = P;
  out["v"] = v;
  out["ns"] = ns;
  out["burn"] = burn;
  out["thin"] = thin;
  if (fmmx == 1) out["J"] = J;
  out["at"] = at;
  if (bt > 0) out["bt"] = bt;
  out["as"] = as;
  out["bs"] = bs;
  out["alpha_dir"] = alpha_dir;
  out["D"] = D;

  // data
  out["Y"] = Y;
  out["X"] = X;
  out["Xcon"] = Xcon;
  out["Xcat"] = Xcat;
  out["H0"] = H0;
  out["n_theta"] = n_theta;
  if (H_out == 1) {
    out["H"] = H;
    out["Ht"] = Ht;
    // out["HH"] = HH;
    out["Ht0"] = Ht0;
  }
  
  // repulsive
  out["repulsive"] = repulsive;
  if (repulsive != 0) {
    out["accept"] = accept;
    out["accept_logratio"] = accept_logratio;
    out["phi"] = phi;
    out["nu"] = nu;
    out["r"] = r;
    out["theta_sampler"] = theta_sampler;
    out["eps"] = epsD;
    out["eps0"] = eps0D;
  }
  
  // SM
  out["SM"] = SM;
  if (SM == 1) out["nGS"] = nGS;
  
  // ARWMH
  // if (theta_sampler == 1) out["VJ"] = VJ;
  if (theta_sampler >= 1) out["nadapt"] = nadapt;
  

  // model
  if (ppmx == 1) {
    if (covariates_con == 1 && covariates_cat == 1) {
      out["model"] = "ppmx";
    } else if (covariates_con == 1 && covariates_cat == 0) {
      out["model"] = "ppmxcon";
    } else if (covariates_con == 0 && covariates_cat == 1) {
      out["model"] = "ppmxcat";
    } else {
      out["model"] = "ppm";
    }
  } else if (fmmx == 1) {
    if (covariates_fmm == 1) {
      out["model"] = "fmmx";
      out["s2a"] = S_alpha(0,0);
    } else {
      out["model"] = "fmm";
    }
  }
    

  return out;
}







