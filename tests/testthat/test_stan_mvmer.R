# Part of the rstanarm package for estimating model parameters
# Copyright (C) 2015, 2016 Trustees of Columbia University
# Copyright (C) 2017 Sam Brilleman
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

# tests can be run using devtools::test() or manually by loading testthat 
# package and then running the code below possibly with options(mc.cores = 4).

library(rstanarm)
stopifnot(require(lme4))
ITER <- 1000
CHAINS <- 1
SEED <- 12345
REFRESH <- ITER
set.seed(SEED)
if (interactive()) 
  options(mc.cores = parallel::detectCores())

TOLSCALES <- list(
  lmer_fixef = 0.25,  # how many SEs can stan_jm fixefs be from lmer fixefs
  lmer_ranef = 0.05, # how many SDs can stan_jm ranefs be from lmer ranefs
  glmer_fixef = 0.3, # how many SEs can stan_jm fixefs be from glmer fixefs
  glmer_ranef = 0.1 # how many SDs can stan_jm ranefs be from glmer ranefs
)

source(file.path("helpers", "expect_matrix.R"))
source(file.path("helpers", "expect_stanreg.R"))
source(file.path("helpers", "expect_stanmvreg.R"))
source(file.path("helpers", "expect_survfit.R"))
source(file.path("helpers", "expect_ppd.R"))
source(file.path("helpers", "SW.R"))
source(file.path("helpers", "get_tols.R"))
source(file.path("helpers", "recover_pars.R"))

context("stan_mvmer")

#----  Data (for non-Gaussian families)

pbcLong$ybern <- as.integer(pbcLong$logBili >= mean(pbcLong$logBili))
pbcLong$ybino <- as.integer(rpois(nrow(pbcLong), 5))
pbcLong$ypois <- as.integer(pbcLong$albumin)
pbcLong$ygamm <- as.numeric(pbcLong$platelet / 10)
pbcLong$xbern <- as.numeric(pbcLong$platelet / 100)
pbcLong$xpois <- as.numeric(pbcLong$platelet / 100)
pbcLong$xgamm <- as.numeric(pbcLong$logBili)

#----  Models

# univariate GLM
fm1 <- logBili ~ year + (year | id)
m1 <- stan_mvmer(fm1, pbcLong, iter = 10, chains = 1, seed = SEED)

# multivariate GLM
fm2 <- list(logBili ~ year + (year | id), albumin ~ year + (year | id))
m2 <- stan_mvmer(fm2, pbcLong, iter = 10, chains = 1, seed = SEED)

#----  Tests for stan_jm arguments

test_that("formula argument works", {
  expect_identical(as.matrix(m1), as.matrix(update(m1, formula. = list(fm1)))) # fm as list
})

test_that("data argument works", {
  expect_identical(as.matrix(m1), as.matrix(update(m1, data = list(pbcLong)))) # data as list
  expect_identical(as.matrix(m2), as.matrix(update(m2, data = list(pbcLong, pbcLong))))
})

test_that("family argument works", {
  
  expect_output(ret <- update(m1, family = "gaussian"))
  expect_output(ret <- update(m1, family = gaussian))
  expect_output(ret <- update(m1, family = gaussian(link = identity)))
  
  expect_output(ret <- update(m1, formula. = ybern ~ ., family = binomial))
  expect_output(ret <- update(m1, formula. = ypois ~ ., family = poisson))
  expect_output(ret <- update(m1, formula. = ypois ~ ., family = neg_binomial_2))
  expect_output(ret <- update(m1, formula. = ygamm ~ ., family = Gamma))
  #expect_output(ret <- update(m1, formula. = ygamm ~ ., family = inverse.gaussian))
  
  expect_error(ret <- update(m1, formula. = ybino ~ ., family = binomial))
  
  # multivariate model with combinations of family
  expect_output(ret <- update(m2, formula. = list(~ ., ybern ~ .), 
                              family = list(gaussian, binomial)))
})

test_that("prior_PD argument works", {
  expect_output(update(m1, prior_PD = TRUE))
})

test_that("adapt_delta argument works", {
  expect_output(update(m1, adapt_delta = NULL))
  expect_output(update(m1, adapt_delta = 0.8))
})

test_that("error message occurs for arguments not implemented", {
  expect_error(update(m1, weights = 1:10), "not yet implemented")
  expect_error(update(m1, offset = 1:10), "not yet implemented")
  expect_error(update(m1, QR = TRUE), "not yet implemented")
  expect_error(update(m1, sparse = TRUE), "not yet implemented")
})

#----  Check models with multiple groupiing factors

test_that("multiple grouping factors are ok", {
  
  tmpdat <- pbcLong
  tmpdat$practice <- cut(pbcLong$id, c(0,10,20,30,40))
  
  tmpfm1 <- logBili ~ year + (year | id) + (1 | practice)
  ok_mod1 <- update(m1, formula. = tmpfm1, data = tmpdat, init = 0)
  expect_stanmvreg(ok_mod1)
  
  tmpfm2 <- list(
    logBili ~ year + (year | id) + (1 | practice),
    albumin ~ year + (year | id))
  ok_mod2 <- update(m2, formula. = tmpfm2, data = tmpdat)
  expect_stanmvreg(ok_mod2)
  
  tmpfm3 <- list(
    logBili ~ year + (year | id) + (1 | practice),
    albumin ~ year + (year | id) + (1 | practice))
  ok_mod3 <- update(m2, formula. = tmpfm3, data = tmpdat)
  expect_stanmvreg(ok_mod3)
  
  # check reordering grouping factors is ok
  tmpfm4 <- list(
    logBili ~ year + (1 | practice) + (year | id) ,
    albumin ~ year + (year | id))
  expect_identical(as.matrix(ok_mod2), as.matrix(update(ok_mod2, formula. = tmpfm4)))
  
  tmpfm5 <- list(
    logBili ~ year + (1 | practice) + (year | id) ,
    albumin ~ year + (year | id) + (1 | practice))
  expect_identical(as.matrix(ok_mod3), as.matrix(update(ok_mod3, formula. = tmpfm5)))
  
  tmpfm6 <- list(
    logBili ~ year + (1 | practice) + (year | id) ,
    albumin ~ year + (1 | practice) + (year | id))
  expect_identical(as.matrix(ok_mod3), as.matrix(update(ok_mod3, formula. = tmpfm6)))
})

#----  Compare parameter estimates: univariate stan_mvmer vs stan_glmer

if (interactive()) {
  compare_glmer <- function(fmLong, fam = gaussian, ...) {
    y1 <- stan_glmer(fmLong, pbcLong, fam, iter = 1000, chains = CHAINS, seed = SEED)
    y2 <- stan_mvmer(fmLong, pbcLong, fam, iter = 1000, chains = CHAINS, seed = SEED, ...) 
    tols <- get_tols(y1, tolscales = TOLSCALES)
    pars <- recover_pars(y1)
    pars2 <- recover_pars(y2)
    for (i in names(tols$fixef))
      expect_equal(pars$fixef[[i]], pars2$fixef[[i]], tol = tols$fixef[[i]])     
    for (i in names(tols$ranef))
      expect_equal(pars$ranef[[i]], pars2$ranef[[i]], tol = tols$ranef[[i]])
  }
  test_that("coefs same for stan_jm and stan_lmer/coxph", {
    compare_glmer(logBili ~ year + (1 | id), gaussian)})
  test_that("coefs same for stan_jm and stan_glmer, bernoulli", {
    compare_glmer(ybern ~ year + xbern + (1 | id), binomial)})
  test_that("coefs same for stan_jm and stan_glmer, poisson", {
    compare_glmer(ypois ~ year + xpois + (1 | id), poisson, init = 0)})
  test_that("coefs same for stan_jm and stan_glmer, negative binomial", {
    compare_glmer(ypois ~ year + xpois + (1 | id), neg_binomial_2)})
  test_that("coefs same for stan_jm and stan_glmer, Gamma", {
    compare_glmer(ygamm ~ year + xgamm + (1 | id), Gamma(log))})
#  test_that("coefs same for stan_jm and stan_glmer, inverse gaussian", {
#    compare_glmer(ygamm ~ year + xgamm + (1 | id), inverse.gaussian)})  
}

#--------  Check post-estimation functions

f1 <- stan_mvmer(logBili ~ year + (year | id), pbcLong,
                 chains = 1, cores = 1, seed = 12345, iter = 10)
  
for (j in 1) {
  tryCatch({
    mod <- get(paste0("f", j))
    cat("Checking model:", paste0("f", j), "\n")
 
    expect_error(log_lik(mod), "not yet implemented")
    expect_error(loo(mod), "not yet implemented")
    expect_error(waic(mod), "not yet implemented")
    expect_error(posterior_traj(mod), "can only be used")
    expect_error(posterior_survfit(mod), "can only be used")
       
    test_that("posterior_predict works with estimation data", {
      pp <- posterior_predict(mod, m = 1)
      expect_ppd(pp)
    }) 
    
    ndL <- pbcLong[pbcLong$id == 2,]
    ndE <- pbcSurv[pbcSurv$id == 2,]
    test_that("posterior_predict works with new data (one individual)", {
      pp <- posterior_predict(mod, m = 1, newdataLong = ndL, newdataEvent = ndE)
      expect_ppd(pp)
    })  
    
    ndL <- pbcLong[pbcLong$id %in% c(1,2),]
    ndE <- pbcSurv[pbcSurv$id %in% c(1,2),]
    test_that("posterior_predict works with new data (multiple individuals)", {
      pp <- posterior_predict(mod, m = 1, newdataLong = ndL, newdataEvent = ndE)
      expect_ppd(pp)
    })  
  }, error = function(e)
    cat(" Failed for model", paste0("f", j), " due to error:\n", paste(e)))
}



