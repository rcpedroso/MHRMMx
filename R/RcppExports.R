# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

Kmat <- function(P, v) {
    .Call(`_mhrcmxIntDep_Kmat`, P, v)
}

mhrcmx_dep_rep_int <- function(Y, D, X, mu_alpha, S_alpha, Xcon, Xcat, p, n_theta, v, a0, b0, as, bs, at, bt = 0, at0 = 0.01, bt0 = 1, A = 1, lam2_min = 1e-12, Asig = 0, dof = 1, eta = 1, eta0 = 1, s2mu = 1e4, s02 = 1e4, M = 1, ns = 1e3L, thin = 10L, burn = 1e4L, H0 = 0L, J = 0L, alpha_dir = 1, repulsive = 0L, eps = 1, eps0 = 1, phi = 1, nu = 1, r = 2, theta_sampler = 1L, nadapt = 1e3L, alpha_cat = 0.1, m0 = 0, v0 = 10, only_nclr = 0L, H_out = 1L, SM = 1L, nGS = 10L) {
    .Call(`_mhrcmxIntDep_mhrcmx_dep_rep_int`, Y, D, X, mu_alpha, S_alpha, Xcon, Xcat, p, n_theta, v, a0, b0, as, bs, at, bt, at0, bt0, A, lam2_min, Asig, dof, eta, eta0, s2mu, s02, M, ns, thin, burn, H0, J, alpha_dir, repulsive, eps, eps0, phi, nu, r, theta_sampler, nadapt, alpha_cat, m0, v0, only_nclr, H_out, SM, nGS)
}

rcpparma_hello_world <- function() {
    .Call(`_mhrcmxIntDep_rcpparma_hello_world`)
}

rcpparma_outerproduct <- function(x) {
    .Call(`_mhrcmxIntDep_rcpparma_outerproduct`, x)
}

rcpparma_innerproduct <- function(x) {
    .Call(`_mhrcmxIntDep_rcpparma_innerproduct`, x)
}

rcpparma_bothproducts <- function(x) {
    .Call(`_mhrcmxIntDep_rcpparma_bothproducts`, x)
}

rgamma_ars <- function(L, Lu, a, b) {
    .Call(`_mhrcmxIntDep_rgamma_ars`, L, Lu, a, b)
}

