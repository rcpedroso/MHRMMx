#' Generate random samples from the Polya-Gamma distribution
#'
#' @title Generate random samples from the Polya-Gamma distribution, PG(b,c)
#' @param b Either a single integer scalar, or a vector of integers, corresponding to the
#' 'b' parameter for the PG(b,c) distribution. If \code{b} is a scalar, then
#' the same value is paired with every value in \code{c}; if \code{b} is a vector
#' then it must be of the same length as the \code{c} parameter.
#' @param c A vector of real numbers corresponding to the 'c' parameter for the PG(b,c) distribution.
#' @section Details:
#' This code generates random variates from the Polya-Gamma distribution with desired 'b' and 'c' parameters.
#' The underlying code is written in C and is an implementation of the algorithm described in J. Windle's PhD thesis.
#'  
#' The main application of the Polya-Gamma distribution is in Bayesian analysis as it 
#' allows for a data augmentation (via a scale mixture of normals) approach for representation
#' of the logistic regression likelihood (see Example 2 below).
#' @return A vector of samples from the Polya-Gamma distribution, one for each entry of \code{c}
#' 
#' @note     To cite this package please reference: 
#'
#' Makalic, E. & Schmidt, D. F.
#' High-Dimensional Bayesian Regularised Regression with the BayesReg Package
#' arXiv:1611.06649 [stat.CO], 2016 \url{https://arxiv.org/pdf/1611.06649.pdf}
#' 
#' A MATLAB-compatible implementation of the sampler in this package can be obtained from:
#' 
#' \url{http://dschmidt.org/?page_id=189}
#' 
#' @references 
#' 
#'  Jesse Bennett Windle
#'  Forecasting High-Dimensional, Time-Varying Variance-Covariance Matrices
#'  with High-Frequency Data and Sampling Polya-Gamma Random Variates for
#'  Posterior Distributions Derived from Logistic Likelihoods,  
#'  PhD Thesis, 2013 
#'  
#'  Bayesian Inference for Logistic Models Using Polya-Gamma Latent Variables
#'  Nicholas G. Polson, James G. Scott and Jesse Windle,
#'  Journal of the American Statistical Association
#'  Vol. 108, No. 504, pp. 1339--1349, 2013
#'  
#'  Chung, Y.: Simulation of truncated gamma variables,
#'  Korean Journal of Computational & Applied Mathematics, 1998, 5, 601-610
#'   
#' @examples 
#' # -----------------------------------------------------------------
#' # Example 1: Simulated vs exact moments
#' u = matrix(1,1e6,1)
#' x = pgdraw(1,0.5*u)
#' mean(x)
#' var(x)
#' pgdraw.moments(1,0.5)
#' 
#' x = pgdraw(2,2*u)
#' mean(x)
#' var(x)
#' pgdraw.moments(2,2)
#' 
#' 
#' # -----------------------------------------------------------------
#' # Example 2: Simple logistic regression
#' #   Sample from the following Bayesian hierarchy:
#' #    y_i ~ Be(1/(1+exp(-b)))
#' #    b   ~ uniform on R (improper)
#' #
#' #   which is equivalent to
#' #    y_i - 1/2 ~ N(b, 1/omega2_i)
#' #    omega2_i  ~ PG(1,0)
#' #    b         ~ uniform on R
#' #
#' sample_simple_logreg <- function(y, nsamples)
#' {
#'   n = length(y)
#'   omega2 = matrix(1,n,1)   # Polya-Gamma latent variables
#'   beta   = matrix(0,nsamples,1)
#'   
#'   for (i in 1:nsamples)
#'   {
#'     # Sample 'beta'
#'     s = sum(omega2)
#'     m = sum(y-1/2)/s
#'     beta[i] = rnorm(1, m, sqrt(1/s))
#    
#'     # Sample P-G L.Vs
#'     omega2 = pgdraw(1, matrix(1,n,1)*beta[i])
#'   }
#'   return(beta)
#' }
#'
#' # 3 heads, 7 tails; ML estimate of p = 3/10 = 0.3
#' y = c(1,1,1,0,0,0,0,0,0,0)  
#' 
#' # Sample
#' b = sample_simple_logreg(y, 1e4)
#' hist(x=b)
#' 
#' # one way of estimating of 'p' from posterior samples
#' 1/(1+exp(-mean(b)))         
#' 
#'
#' @seealso \code{\link{pgdraw.moments}}
#' @export
pgdraw <- function(b,c)
{
  # Check inputs
  if (length(b)>1 && length(b) != length(c))
  {
    stop("b parameter must either be of length one, or the same length as the c parameter.")
  }
  if (any(b<=0) || !all(floor(b) == b,na.rm=T))
  {
    stop("b parameter must contain only postive integers.")
  }
  
  rcpp_pgdraw(b,c)
}



#' Compute exact first and second moments for the Polya-Gamma distribution
#'
#' @title Compute exact first and second moments for the Polya-Gamma distribution, PG(b, c)
#' @param b The 'b' parameter of the Polya-Gamma distribution.
#' @param c The 'c' parameter of the Polya-Gamma distribution.
#' @section Details:
#' This code computes the exact mean and variance of the Polya-Gamma distribution for the specified parameters.
#' @return A list containing the mean and variance.
#' 
#' @references 
#' 
#'  Jesse Bennett Windle
#'  Forecasting High-Dimensional, Time-Varying Variance-Covariance Matrices
#'  with High-Frequency Data and Sampling Polya-Gamma Random Variates for
#'  Posterior Distributions Derived from Logistic Likelihoods  
#'  PhD Thesis, 2013 
#'  
#'  Bayesian Inference for Logistic Models Using Polya-Gamma Latent Variables
#'  Nicholas G. Polson, James G. Scott and Jesse Windle
#'  Journal of the American Statistical Association
#'  Vol. 108, No. 504, pp. 1339--1349, 2013
#' 
#' @examples 
#' # -----------------------------------------------------------------
#' # Example: Simulated vs exact moments
#' 
#' u = matrix(1,1e6,1)
#' x = pgdraw(1,0.5*u)
#' mean(x)
#' var(x)
#' pgdraw.moments(1,0.5)
#' 
#' x = pgdraw(2,2*u)
#' mean(x)
#' var(x)
#' pgdraw.moments(2,2)
#' 
#'
#' @seealso \code{\link{pgdraw}}
#' @export
pgdraw.moments <- function(b, c)
{
  rv     = list()
  rv$mu  = b/2/c*tanh(c/2)
  rv$var = b/(4*c^3)*(sinh(c)-c)*(1/cosh(c/2)^2)
  return(rv)
}