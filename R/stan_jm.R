# Part of the rstanarm package for estimating model parameters
# Copyright (C) 2013, 2014, 2015, 2016 Trustees of Columbia University
# Copyright (C) 2016 Monash University
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

#' Bayesian joint longitudinal and time-to-event models via Stan
#' 
#' Fits a shared parameter joint model for longitudinal and time-to-event 
#' (e.g. survival) data under a Bayesian framework using Stan.
#' 
#' @export
#' @templateVar rareargs na.action,contrasts
#' @template args-same-as-rarely
#' @template args-dots
#' @template args-prior_covariance
#' @template args-prior_PD
#' @template args-algorithm
#' @template args-adapt_delta
#' @template args-QR
#' @template args-sparse
#' 
#' @param formulaLong A two-sided linear formula object describing both the 
#'   fixed-effects and random-effects parts of the longitudinal submodel  
#'   (see \code{\link[lme4]{glmer}} for details). For a multivariate joint 
#'   model (i.e. more than one longitudinal marker) this should 
#'   be a list of such formula objects, with each element
#'   of the list providing the formula for one of the longitudinal submodels.
#' @param dataLong A data frame containing the variables specified in
#'   \code{formulaLong}. If fitting a multivariate joint model, then this can
#'   be either a single data frame which contains the data/variables for all 
#'   the longitudinal submodels, or it can be a list of data frames where each
#'   element of the list provides the data for one of the longitudinal 
#'   submodels.
#' @param formulaEvent A two-sided formula object describing the event
#'   submodel. The left hand side of the formula should be a \code{Surv()} 
#'   object. See \code{\link[survival]{Surv}}.
#' @param dataEvent A data frame containing the variables specified in
#'   \code{formulaEvent}.
#' @param time_var A character string specifying the name of the variable 
#'   in \code{dataLong} which represents time.
#' @param id_var A character string specifying the name of the variable in
#'   \code{dataLong} which distinguishes between individuals. This can be
#'   left unspecified if there is only one grouping factor (which is assumed
#'   to be the individual). If there is more than one grouping factor (i.e.
#'   clustering beyond the level of the individual) then the \code{id_var}
#'   argument must be specified.
#' @param offset Not currently implemented. Same as \code{\link[stats]{glm}}.
#' @param family The family (and possibly also the link function) for the 
#'   longitudinal submodel(s). See \code{\link[lme4]{glmer}} for details. 
#'   If fitting a multivariate joint model, then this can optionally be a
#'   list of families, in which case each element of the list specifies the
#'   family for one of the longitudinal submodels.
#' @param assoc A character string or character vector specifying the joint
#'   model association structure. Possible association structures that can
#'   be used include: "etavalue" (the default); "etaslope"; "etalag"; "etaauc"; 
#'   "muvalue"; "muslope"; "mulag"; "muauc"; "shared_b"; "shared_coef"; or "null". 
#'   These are described in the \strong{Details} section below. For a multivariate 
#'   joint model, different association structures can optionally be used for 
#'   each longitudinal submodel by specifying a list of character
#'   vectors, with each element of the list specifying the desired association 
#'   structure for one of the longitudinal submodels. Specifying \code{assoc = NULL}
#'   will fit a joint model with no association structure (equivalent  
#'   to fitting separate longitudinal and time-to-event models). It is also 
#'   possible to include interaction terms between the association term 
#'   ("etavalue", "etaslope", "muvalue", "muslope") and observed data/covariates. 
#'   It is also possible, when fitting a multivariate joint model, to include 
#'   interaction terms between the association terms ("etavalue" or "muvalue") 
#'   corresponding to the different longitudinal outcomes. See the 
#'   \strong{Details} section as well as the \strong{Examples} below.
#' @param basehaz A character string indicating which baseline hazard to use
#'   for the event submodel. Options are a Weibull baseline hazard
#'   (\code{"weibull"}, the default), a B-splines approximation estimated 
#'   for the log baseline hazard (\code{"bs"}), or a piecewise
#'   constant baseline hazard (\code{"piecewise"}).
#' @param basehaz_ops A named list specifying options related to the baseline
#'   hazard. Currently this can include: \cr
#'   \describe{
#'     \item{\code{df}}{A positive integer specifying the degrees of freedom 
#'     for the B-splines if \code{basehaz = "bs"}, or the number of
#'     intervals used for the piecewise constant baseline hazard if 
#'     \code{basehaz = "piecewise"}. The default is 6.}
#'     \item{\code{knots}}{An optional numeric vector specifying the internal knot 
#'     locations for the B-splines if \code{basehaz = "bs"}, or the 
#'     internal cut-points for defining intervals of the piecewise constant 
#'     baseline hazard if \code{basehaz = "piecewise"}. Knots cannot be
#'     specified if \code{df} is specified. If not specified, then the 
#'     default is to use \code{df - 4} knots if \code{basehaz = "bs"},
#'     or \code{df - 1} knots if \code{basehaz = "piecewise"}, which are
#'     placed at equally spaced percentiles of the distribution of
#'     observed event times.}
#'   }
#' @param dataAssoc A data frame containing observed covariates used in the
#'   interactions between association terms and observed data. See the
#'   \strong{Details} and \strong{Examples} sections for details on how to 
#'   specify the formulas for the interactions as part of the \code{assoc}  
#'   argument. If interactions between association terms and observed data are
#'   specified as part of the \code{assoc} argument, but \code{dataAssoc} is 
#'   not specified, then the values for the covariates will be taken from the 
#'   data frame(s) provided in \code{dataLong}. (Specifying \code{dataAssoc}
#'   directly means that a different measurement time schedule could be used 
#'   for the covariates in the association term interactions when compared with
#'   the measurement time schedule used for the longitudinal outcomes and  
#'   covariates provided in \code{dataLong}.
#' @param quadnodes The number of nodes to use for the Gauss-Kronrod quadrature
#'   that is used to evaluate the cumulative hazard in the likelihood function. 
#'   Options are 15 (the default), 11 or 7.
#' @param subsetLong,subsetEvent Same as subset in \code{\link[stats]{glm}}.
#'   However, if fitting a multivariate joint model and a list of data frames 
#'   is provided in \code{dataLong} then a corresponding list of subsets 
#'   must be provided in \code{subsetLong}.
#' @param weights Experimental and should be used with caution. The 
#'   user can optionally supply a 2-column data frame containing a set of
#'   'prior weights' to be used in the estimation process. The data frame should
#'   contain two columns: the first containing the IDs for each individual, and 
#'   the second containing the corresponding weights. The data frame should only
#'   have one row for each individual; that is, weights should be constant 
#'   within individuals.
#' @param init The method for generating the initial values for the MCMC.
#'   The default is \code{"model_based"}, which uses those obtained from 
#'   fitting separate longitudinal and time-to-event models prior  
#'   to fitting the joint model. Parameters that cannot be obtained from 
#'   fitting separate longitudinal and time-to-event models are initialised 
#'   at 0. This provides reasonable initial values which should aid the MCMC
#'   sampler. However, it is recommended that any final analysis should be
#'   performed with several MCMC chains each initiated from a different
#'   set of initial values; this can be obtained by setting
#'   \code{init = "random"}. Other possibilities for specifying \code{init}
#'   are those described for \code{\link[rstan]{stan}}.  
#' @param priorLong,priorEvent,priorAssoc The prior distributions for the 
#'   regression coefficients in the longitudinal submodel(s), event submodel,
#'   and the association parameter(s). Can be a call to one of the various functions 
#'   provided by \pkg{rstanarm} for specifying priors. The subset of these functions 
#'   that can be used for the prior on the coefficients can be grouped into several 
#'   "families":
#'   
#'   \tabular{ll}{
#'     \strong{Family} \tab \strong{Functions} \cr 
#'     \emph{Student t family} \tab \code{normal}, \code{student_t}, \code{cauchy} \cr 
#'     \emph{Hierarchical shrinkage family} \tab \code{hs}, \code{hs_plus} \cr 
#'     \emph{Laplace family} \tab \code{laplace}, \code{lasso} \cr
#'   }
#'   
#'   See the \link[=priors]{priors help page} for details on the families and 
#'   how to specify the arguments for all of the functions in the table above.
#'   To omit a prior ---i.e., to use a flat (improper) uniform prior---
#'   \code{prior} can be set to \code{NULL}, although this is rarely a good
#'   idea.
#'   
#'   \strong{Note:} Unless \code{QR=TRUE}, if \code{prior} is from the Student t
#'   family or Laplace family, and if the \code{autoscale} argument to the 
#'   function used to specify the prior (e.g. \code{\link{normal}}) is left at 
#'   its default and recommended value of \code{TRUE}, then the default or 
#'   user-specified prior scale(s) may be adjusted internally based on the scales
#'   of the predictors. See the \link[=priors]{priors help page} for details on
#'   the rescaling and the \code{\link{prior_summary}} function for a summary of
#'   the priors used for a particular model.
#' @param priorLong_intercept,priorEvent_intercept The prior distributions  
#'   for the intercepts in the longitudinal submodel(s) and event submodel. 
#'   Can be a call to \code{normal}, \code{student_t} or 
#'   \code{cauchy}. See the \link[=priors]{priors help page} for details on 
#'   these functions. To omit a prior on the intercept ---i.e., to use a flat
#'   (improper) uniform prior--- \code{prior_intercept} can be set to
#'   \code{NULL}.
#'   
#'   \strong{Note:} If using a dense representation of the design matrix 
#'   ---i.e., if the \code{sparse} argument is left at its default value of
#'   \code{FALSE}--- then the prior distribution for the intercept is set so it
#'   applies to the value when all predictors are centered.
#' @param priorLong_aux The prior distribution for the "auxiliary" parameters
#'   in the longitudinal submodels (if applicable). 
#'   The "auxiliary" parameter refers to a different parameter 
#'   depending on the \code{family}. For Gaussian models \code{priorLong_aux} 
#'   controls \code{"sigma"}, the error 
#'   standard deviation. For negative binomial models \code{priorLong_aux} controls 
#'   \code{"reciprocal_dispersion"}, which is similar to the 
#'   \code{"size"} parameter of \code{\link[stats]{rnbinom}}:
#'   smaller values of \code{"reciprocal_dispersion"} correspond to 
#'   greater dispersion. For gamma models \code{priorLong_aux} sets the prior on 
#'   to the \code{"shape"} parameter (see e.g., 
#'   \code{\link[stats]{rgamma}}), and for inverse-Gaussian models it is the 
#'   so-called \code{"lambda"} parameter (which is essentially the reciprocal of
#'   a scale parameter). Binomial and Poisson models do not have auxiliary 
#'   parameters. 
#'   
#'   \code{priorLong_aux} can be a call to \code{exponential} to 
#'   use an exponential distribution, or \code{normal}, \code{student_t} or 
#'   \code{cauchy}, which results in a half-normal, half-t, or half-Cauchy 
#'   prior. See \code{\link{priors}} for details on these functions. To omit a 
#'   prior ---i.e., to use a flat (improper) uniform prior--- set 
#'   \code{priorLong_aux} to \code{NULL}.
#'   
#'   If fitting a multivariate joint model, you have the option to
#'   specify a list of prior distributions, however the elements of the list
#'   that correspond to any longitudinal submodel which does not have an 
#'   auxiliary parameter will be ignored. 
#' @param priorEvent_aux The prior distribution for the "auxiliary" parameters
#'   in the event submodel. The "auxiliary" parameters refers to different  
#'   parameters depending on the baseline hazard. For \code{basehaz = "weibull"}
#'   the auxiliary parameter is the Weibull shape parameter. For 
#'   \code{basehaz = "bs"} the auxiliary parameters are the coefficients for the
#'   B-spline approximation to the log baseline hazard.
#'   For \code{basehaz = "piecewise"} the auxiliary parameters are the piecewise
#'   estimates of the log baseline hazard.
#' @param long_lp A logical scalar (defaulting to TRUE) indicating whether to 
#'   conditioning on the longitudinal outcome(s).    
#' @param event_lp A logical scalar (defaulting to TRUE) indicating whether to 
#'   conditioning on the event outcome.
#'   
#' @details The \code{stan_jm} function can be used to fit a joint model (also 
#'   known as a shared parameter model) for longitudinal and time-to-event data 
#'   under a Bayesian framework. 
#'   The joint model may be univariate (with only one longitudinal submodel) or
#'   multivariate (with more than one longitudinal submodel). Multi-level 
#'   clustered data are allowed (e.g. patients within clinics), provided that the
#'   individual (e.g. patient) is the lowest level of clustering. The underlying
#'   estimation is carried out using the Bayesian C++ package Stan 
#'   (\url{http://mc-stan.org/}). \cr
#'   \cr 
#'   For the longitudinal submodel a generalised linear mixed model is assumed 
#'   with any of the \code{\link[stats]{family}} choices allowed by 
#'   \code{\link[lme4]{glmer}}. If a multivariate joint model is specified (by
#'   providing a list of formulas in the \code{formulaLong} argument), then
#'   the multivariate longitudinal submodel consists of a multivariate generalized  
#'   linear model (GLM) with group-specific terms that are assumed to be correlated
#'   across the different GLM submodels. That is, within
#'   a grouping factor (for example, patient ID) the group-specific terms are
#'   assumed to be correlated across the different GLM submodels. It is 
#'   possible to specify a different outcome type (for example a different
#'   family and/or link function) for each of the GLM submodels, by providing
#'   a list of \code{\link[stats]{family}} objects in the \code{family} 
#'   argument. \cr
#'   \cr
#'   For the event submodel a parametric
#'   proportional hazards model is assumed. The baseline hazard can be estimated 
#'   using either a Weibull distribution (\code{basehaz = "weibull"}) or a
#'   piecewise constant baseline hazard (\code{basehaz = "piecewise"}), or 
#'   approximated using cubic B-splines (\code{basehaz = "bs"}). 
#'   If either of the latter two are used then the degrees of freedom, 
#'   or the internal knot locations, can be optionally specified. If
#'   the degrees of freedom are specified (through the \code{df} argument) then
#'   the knot locations are automatically generated based on the 
#'   distribution of the observed event times (not including censoring times). 
#'   Otherwise internal knot locations can be specified 
#'   directly through the \code{knots} argument. If neither \code{df} or
#'   \code{knots} is specified, then the default is to set \code{df} equal to 6.
#'   It is not possible to specify both \code{df} and \code{knots}. \cr
#'   \cr
#'   Time-varying covariates are allowed in both the 
#'   longitudinal and event submodels. These should be specified in the data 
#'   in the same way as they normally would when fitting a separate 
#'   longitudinal model using \code{\link[lme4]{lmer}} or a separate 
#'   time-to-event model using \code{\link[survival]{coxph}}. These time-varying
#'   covariates should be exogenous in nature, otherwise they would perhaps 
#'   be better specified as an additional outcome (i.e. by including them as an 
#'   additional longitudinal outcome/submodel in the joint model). \cr
#'   \cr
#'   Bayesian estimation of the joint model is performed via MCMC. The Bayesian  
#'   model includes independent priors on the 
#'   regression coefficients for both the longitudinal and event submodels, 
#'   including the association parameter(s) (in much the same way as the
#'   regression parameters in \code{\link{stan_glm}}) and
#'   priors on the terms of a decomposition of the covariance matrices of the
#'   group-specific parameters (in the same way as \code{\link{stan_glmer}}). 
#'   See \code{\link{priors}} for more information about the priors distributions
#'   that are available. \cr
#'   \cr
#'   Gauss-Kronrod quadrature is used to numerically evaluate the integral  
#'   over the cumulative hazard in the likelihood function for the event submodel.
#'   The accuracy of the numerical approximation can be controlled using the
#'   number of quadrature nodes, specified through the \code{quadnodes} 
#'   argument. Using a higher number of quadrature nodes will result in a more 
#'   accurate approximation.
#'   
#'   \subsection{Association structures}{
#'   The association structure for the joint model can be based on any of the 
#'   following parameterisations: 
#'     \itemize{
#'       \item current value of the linear predictor in the 
#'         longitudinal submodel (\code{"etavalue"}) 
#'       \item first derivative (slope) of the linear predictor in the 
#'         longitudinal submodel (\code{"etaslope"}) 
#'       \item lagged value of the linear predictor in the longitudinal 
#'         submodel (\code{"etalag(#)"}, replacing \code{#} with the desired 
#'         lag in units of the time variable);
#'       \item the area under the curve of the linear predictor in the 
#'         longitudinal submodel (\code{"etaauc"}) 
#'       \item current expected value of the longitudinal submodel 
#'         (\code{"muvalue"})
#'       \item lagged expected value of the longitudinal submodel 
#'         (\code{"mulag(#)"}, replacing \code{#} with the desired lag in 
#'         units of the time variable) 
#'       \item the area under the curve of the expected value from the 
#'         longitudinal submodel (\code{"muauc"})
#'       \item shared individual-level random effects (\code{"shared_b"}) 
#'       \item shared individual-level random effects which also incorporate 
#'         the corresponding fixed effect as well as any corresponding 
#'         random effects for clustering levels higher than the individual)
#'         (\code{"shared_coef"})
#'       \item interactions between association terms and observed data/covariates
#'         (\code{"etavalue_data"}, \code{"etaslope_data"}, \code{"muvalue_data"}, 
#'         \code{"muslope_data"}). These are described further below.
#'       \item interactions between association terms corresponding to different 
#'         longitudinal outcomes in a multivariate joint model 
#'         (\code{"etavalue_etavalue(#)"}, \code{"etavalue_muvalue(#)"},
#'         \code{"muvalue_etavalue(#)"}, \code{"muvalue_muvalue(#)"}). These
#'         are described further below.      
#'       \item no association structure (equivalent to fitting separate 
#'         longitudinal and event models) (\code{"null"} or \code{NULL}) 
#'     }
#'   More than one association structure can be specified, however,
#'   not all possible combinations are allowed.   
#'   Note that for the lagged association structures (\code{"etalag(#)"} and 
#'   \code{"mulag(#)"}) baseline values (time = 0) are used for the instances 
#'   where the time lag results in a time prior to baseline. When using the 
#'   \code{"etaauc"} or \code{"muauc"} association structures, the area under
#'   the curve is evaluated using Gauss-Kronrod quadrature with 15 quadrature 
#'   nodes. By default, \code{"shared_b"} and \code{"shared_coef"} contribute 
#'   all random effects to the association structure; however, a subset of the 
#'   random effects can be chosen by specifying their indices between parentheses 
#'   as a suffix, for example, \code{"shared_b(1)"} or \code{"shared_b(1:3)"} or 
#'   \code{"shared_b(1,2,4)"}, and so on. \cr
#'   \cr 
#'   In addition, several association terms (\code{"etavalue"}, \code{"etaslope"},
#'   \code{"muvalue"}, \code{"muslope"}) can be interacted with observed 
#'   data/covariates. To do this, use the association term's main handle plus a
#'   suffix of \code{"_data"} then followed by the model matrix formula in 
#'   parentheses. For example if we had a variable in our dataset for gender 
#'   named \code{sex} then we might want to obtain different estimates for the 
#'   association between the current slope of the marker and the risk of the 
#'   event for each gender. To do this we would specify 
#'   \code{assoc = "etaslope_data(~ sex)"}. \cr
#'   \cr
#'   It is also possible, when fitting  a multivariate joint model, to include 
#'   interaction terms between the association terms themselves (this only
#'   applies for interacting \code{"etavalue"} or \code{"muvalue"}). For example, 
#'   if we had a joint model with two longitudinal markers, we could specify 
#'   \code{assoc = list(c("etavalue", "etavalue_etavalue(2)"), "etavalue")}.
#'   The first element of list says we want to use the value of the linear
#'   predictor for the first marker, as well as it's interaction with the
#'   value of the linear predictor for the second marker. The second element of 
#'   the list says we want to also include the expected value of the second marker 
#'   (i.e. as a "main effect"). Therefore, the linear predictor for the event 
#'   submodel would include the "main effects" for each marker as well as their
#'   interaction. \cr
#'   \cr
#'   There are additional examples in the \strong{Examples} section below.
#'   }
#' 
#' @return A \link[=stanjm-object]{stanjm} object is returned.
#' 
#' @seealso \code{\link{stanjm-object}}, \code{\link{stanjm-methods}}, 
#'   \code{\link{print.stanjm}}, \code{\link{summary.stanjm}},
#'   \code{\link{posterior_traj}}, \code{\link{posterior_survfit}}, 
#'   \code{\link{posterior_predict}}, \code{\link{posterior_interval}},
#'   \code{\link{pp_check}}, \code{\link{ps_check}}.
#' 
#' @examples
#' \donttest{
#' #####
#' # Univariate joint model, with association structure based on the 
#' # current value of the linear predictor
#' f1 <- stan_jm(formulaLong = logBili ~ year + (1 | id), 
#'               dataLong = pbcLong_subset,
#'               formulaEvent = Surv(futimeYears, death) ~ sex + trt, 
#'               dataEvent = pbcSurv_subset,
#'               time_var = "year")
#' summary(f1) 
#'         
#' #####
#' # Univariate joint model, with association structure based on the 
#' # current value of the linear predictor and shared random intercept
#' f2 <- stan_jm(formulaLong = logBili ~ year + (1 | id), 
#'               dataLong = pbcLong_subset,
#'               formulaEvent = Surv(futimeYears, death) ~ sex + trt, 
#'               dataEvent = pbcSurv_subset,
#'               assoc = c("etavalue", "shared_b"),
#'               time_var = "year")
#' summary(f2)          
#' 
#' ######
#' # Multivariate joint model, with association structure based 
#' # on the current value of the linear predictor in each longitudinal 
#' # submodel and shared random intercept from the second longitudinal 
#' # submodel only (which is the first random effect in that submodel
#' # and is therefore indexed the '(1)' suffix in the code below)
#' mv1 <- stan_jm(
#'         formulaLong = list(
#'           logBili ~ year + (1 | id), 
#'           albumin ~ sex + year + (1 + year | id)),
#'         dataLong = pbcLong_subset,
#'         formulaEvent = Surv(futimeYears, death) ~ sex + trt, 
#'         dataEvent = pbcSurv_subset,
#'         assoc = list("etavalue", c("etavalue", "shared_b(1)")), 
#'         time_var = "year")
#' summary(mv1)
#' 
#' # To include both the random intercept and random slope in the shared 
#' # random effects association structure for the second longitudinal 
#' # submodel, we could specify the following:
#' #   update(mv1, assoc = list("etavalue", c("etavalue", "shared_b"))
#' # which would be equivalent to:  
#' #   update(mv1, assoc = list("etavalue", c("etavalue", "shared_b(1,2)"))
#' # or:
#' #   update(mv1, assoc = list("etavalue", c("etavalue", "shared_b(1:2)"))     
#' 
#' ######
#' # Multivariate joint model, estimated using multiple MCMC chains 
#' # run in parallel across all available PC cores
#' mv2 <- stan_jm(
#'         formulaLong = list(
#'           logBili ~ year + (1 | id), 
#'           albumin ~ sex + year + (1 + year | id)),
#'         dataLong = pbcLong_subset,
#'         formulaEvent = Surv(futimeYears, death) ~ sex + trt, 
#'         dataEvent = pbcSurv_subset,
#'         assoc = list("etavalue", c("etavalue", "shared_b(1)")),
#'         time_var = "year",
#'         chains = 3, refresh = 25,
#'         cores = parallel::detectCores())
#' summary(mv2)  
#' 
#' #####
#' # Here we provide an example of specifying an association structure 
#' # based on the lagged value of the linear predictor, where the lag
#' # is 2 time units (i.e. 2 years in this example)
#' f3 <- stan_jm(formulaLong = logBili ~ year + (1 | id), 
#'               dataLong = pbcLong_subset,
#'               formulaEvent = Surv(futimeYears, death) ~ sex + trt, 
#'               dataEvent = pbcSurv_subset,
#'               time_var = "year",
#'               assoc = "etalag(2)")
#' summary(f3) 
#' 
#' #####
#' # Here we provide an example of specifying an association structure with 
#' # interaction terms. Here we specify that we want to use an association
#' # structure based on the current value of the linear predictor from
#' # the longitudinal submodel ("etavalue"), but we will also specify
#' # that we want to interact this with the treatment covariate (trt) from
#' # pbcLong_subset data frame so that we can estimate a different association 
#' # parameter (i.e. estimated effect of log serum bilirubin on the log hazard 
#' # of death) for each treatment group
#' f4 <- stan_jm(formulaLong = logBili ~ year + (1 | id), 
#'               dataLong = pbcLong_subset,
#'               formulaEvent = Surv(futimeYears, death) ~ sex + trt, 
#'               dataEvent = pbcSurv_subset,
#'               time_var = "year", chains = 1,
#'               assoc = c("etavalue", "etavalue_data(~ trt)"))
#' 
#' #####
#' # Here we provide an example of a multivariate joint model, where the
#' # association structure is formed by including the expected value of 
#' # each marker (logBili and albumin) in the linear predictor of the event
#' # submodel, as well as their interaction effect. (Noting that whether an  
#' # association structure based on a marker by marker interaction term makes 
#' # sense will depend on the context of your application -- here we just show
#' # it for demostration purposes).
#' mv3 <- stan_jm(
#'         formulaLong = list(
#'           logBili ~ year + (1 | id), 
#'           albumin ~ sex + year + (1 + year | id)),
#'         dataLong = pbcLong_subset,
#'         formulaEvent = Surv(futimeYears, death) ~ sex + trt, 
#'         dataEvent = pbcSurv_subset,
#'         time_var = "year", chains = 1,
#'         assoc = list(c("etavalue", "etavalue_etavalue(2)"), "etavalue"))
#' }
#' 
#' @import data.table
#' @importFrom lme4 lmerControl glmerControl
#' 
stan_jm <- function(formulaLong, dataLong, formulaEvent, dataEvent, time_var, 
                    id_var, family = gaussian, assoc = "etavalue", dataAssoc,
                    basehaz = c("weibull", "bs", "piecewise"), basehaz_ops, 
                    quadnodes = 15, subsetLong, subsetEvent, init = "model_based", 
                    na.action = getOption("na.action", "na.omit"), weights, 
                    offset, contrasts, ...,				          
                    priorLong = normal(), priorLong_intercept = normal(), 
                    priorLong_aux = cauchy(0, 5), priorEvent = normal(), 
                    priorEvent_intercept = normal(), priorEvent_aux = cauchy(0, 50),
                    priorAssoc = normal(), prior_covariance = decov(), prior_PD = FALSE, 
                    algorithm = c("sampling", "meanfield", "fullrank"), 
                    adapt_delta = NULL, max_treedepth = NULL, QR = FALSE, 
                    sparse = FALSE, long_lp = TRUE, event_lp = TRUE) {
  
  
  #-----------------------------
  # Pre-processing of arguments
  #-----------------------------  
  
  # Check for arguments not yet implemented
  if (!missing(dataAssoc))
    stop("'dataAssoc argument not yet implemented.")
  if (!missing(offset)) 
    stop("Offsets are not yet implemented for stan_jm")
  if (QR)               
    stop("QR decomposition not yet implemented for stan_jm")
  if (sparse)
    stop("'sparse' option is not yet implemented for stan_jm")
  #  if (algorithm %in% c("meanfield", "fullrank"))
  #    stop ("Meanfield and fullrank algorithms not yet implemented for stan_jm")
  if (missing(offset))      offset      <- NULL 
  if (missing(basehaz_ops)) basehaz_ops <- NULL
  if (missing(weights))     weights     <- NULL
  if (missing(id_var))      id_var      <- NULL
  if (missing(subsetLong))  subsetLong  <- NULL
  if (missing(dataAssoc))   dataAssoc   <- NULL 
  
  basehaz   <- match.arg(basehaz)
  algorithm <- match.arg(algorithm)
  
  # Validate arguments
  formulaLong <- validate_arg(formulaLong, "formula")
  M           <- length(formulaLong)
  dataLong    <- validate_arg(dataLong,   "data.frame", null_ok = TRUE, validate_length = M)
  subsetLong  <- validate_arg(subsetLong, "vector",     null_ok = TRUE, validate_length = M)
  assoc       <- validate_arg(assoc,      "character",  null_ok = TRUE, validate_length = M, broadcast = TRUE)

  # Validate family and link
  supported_families <- c("binomial", "gaussian", "Gamma", "inverse.gaussian",
                          "poisson", "neg_binomial_2")
  if (!is(family, "list")) {
    family <- rep(list(family), M) 
  } else if (!length(family) == M) {
    stop("family is a list of the incorrect length.")
  }
  family <- lapply(family, validate_family)
  fam <- lapply(family, function(x) 
    which(pmatch(supported_families, x$family, nomatch = 0L) == 1L))
  if (any(lapply(fam, length) == 0L)) 
    stop("'family' must be one of ", paste(supported_families, collapse = ", "))
  supported_links <- lapply(fam, function(x) supported_glm_links(supported_families[x]))
  link <- mapply(function(x, i) which(supported_links[[i]] == x$link),
                 family, seq_along(family), SIMPLIFY = TRUE)
  if (any(lapply(link, length) == 0L)) 
    stop("'link' must be one of ", paste(supported_links, collapse = ", "))

  # Matched call
  call <- match.call(expand.dots = TRUE)    
  mc   <- match.call(expand.dots = FALSE)
  mc$time_var <- mc$id_var <- mc$assoc <- 
    mc$basehaz <- mc$basehaz_ops <-
    mc$df <- mc$knots <- mc$quadnodes <- NULL
  mc$priorLong <- mc$priorLong_intercept <- mc$priorLong_aux <-
    mc$priorEvent <- mc$priorEvent_intercept <- mc$priorEvent_aux <-
    mc$priorAssoc <- mc$prior_covariance <-
    mc$prior_PD <- mc$algorithm <- mc$scale <- 
    mc$concentration <- mc$shape <- mc$init <- 
    mc$adapt_delta <- mc$max_treedepth <- 
    mc$... <- mc$QR <- NULL
  mc$weights <- NULL 
  mc$long_lp <- mc$event_lp <- NULL

  # Create call for longitudinal submodel  
  y_mc <- mc
  y_mc <- strip_nms(y_mc, "Long") 
  y_mc$formulaEvent <- y_mc$dataEvent <- y_mc$subsetEvent <- NULL

  # Create call for each longitudinal submodel separately
  m_mc <- lapply(1:M, function(m, old_call) {
    new_call <- old_call
    fm       <- eval(old_call$formula)
    data     <- eval(old_call$data)
    subset   <- eval(old_call$subset)
    family   <- eval(old_call$family)
    new_call$formula <- if (is(fm, "list"))     fm[[m]]     else old_call$formula
    new_call$data    <- if (is(data, "list"))   data[[m]]   else old_call$data
    new_call$subset  <- if (is(subset, "list")) subset[[m]] else old_call$subset
    new_call$family  <- if (is(family, "list")) family[[m]] else old_call$family
    new_call
  }, old_call = y_mc)

  # Create call for event submodel
  e_mc <- mc
  e_mc <- strip_nms(e_mc, "Event")
  e_mc$formulaLong <- e_mc$dataLong <- e_mc$family <- e_mc$subsetLong <- NULL
  
  # Is priorLong* already a list?
  priorLong           <- maybe_broadcast_priorarg(priorLong)
  priorLong_intercept <- maybe_broadcast_priorarg(priorLong_intercept)
  priorLong_aux       <- maybe_broadcast_priorarg(priorLong_aux)
    
  #--------------------------------
  # Data for longitudinal submodel
  #--------------------------------
  
  # Fit separate longitudinal submodels
  y_mod_stuff <- Map(handle_glmod, m_mc, family, sparse)

  # Construct single cnms list for all longitudinal submodels
  cnms <- get_common_cnms(fetch(y_mod_stuff, "cnms"))
  cnms_nms <- names(cnms)
  
  # Additional error checks
  id_var <- check_id_var (id_var, fetch(y_mod_stuff, "cnms"))
  unique_id_list <- check_id_list(id_var, fetch(y_mod_stuff, "flist"))
  
  # Construct prior weights
  has_weights <- (!is.null(weights))
  if (has_weights) check_arg_weights(weights, id_var)
  y_weights <- lapply(y_mod_stuff, handle_weights, weights, id_var)
  
  #-------------------------
  # Data for event submodel
  #-------------------------
  
  if (!id_var %in% colnames(dataEvent))
    stop(paste0("Variable '", id_var, "' must be appear in dataEvent"), call. = FALSE)
  
  # Fit separate event submodel
  e_mod_stuff <- handle_coxmod(e_mc, quadnodes = quadnodes, id_var = id_var, 
                                unique_id_list = unique_id_list, sparse = sparse)
  
  # Construct prior weights
  e_weights <- handle_weights(e_mod_stuff, weights, id_var)

  # Baseline hazard
  ok_basehaz <- nlist("weibull", "bs", "piecewise")
  basehaz <- handle_basehaz(basehaz, basehaz_ops, ok_basehaz = ok_basehaz, 
                            eventtime = e_mod_stuff$eventtime, d = e_mod_stuff$d)
  
  # Incorporate intercept term if Weibull baseline hazard
  e_mod_stuff$has_intercept <- (basehaz$type == 1L)

  #--------------------------------
  # Data for association structure
  #--------------------------------
  
  # Handle association structure
  # !!! NB if ordering is changed here, then must also change standata$has_assoc
  ok_assoc <- c("null", "etavalue","etaslope", "etalag", "etaauc", "muvalue", 
                "muslope", "mulag", "muauc", "shared_b", "shared_coef")
  ok_assoc_data         <- ok_assoc[c(2:3,6:7)]
  ok_assoc_interactions <- ok_assoc[c(2,6)]
  
  assoc <- mapply(validate_assoc, assoc, y_mod_stuff, 
                  MoreArgs = list(ok_assoc = ok_assoc, ok_assoc_data = ok_assoc_data,
                                  ok_assoc_interactions = ok_assoc_interactions, 
                                  id_var = id_var, M = M))
  assoc <- check_order_of_assoc_interactions(assoc, ok_assoc_interactions)
  colnames(assoc) <- paste0("Long", 1:M)

  # Time shift used for numerically calculating derivative of linear predictor 
  # or expected value of longitudinal outcome using one-sided difference
  eps <- 1E-5
  
  # Unstandardised quadrature nodes for AUC association structure
  auc_quadnodes <- 15
  auc_quadpoints <- get_quadpoints(auc_quadnodes)
  auc_quadweights <- unlist(
    lapply(e_mod_stuff$quadtimes, function(x) 
      lapply(x, function(y) 
        lapply(auc_quadpoints$weights, unstandardise_quadweights, 0, y))))

  # Return design matrices for evaluating longitudinal submodel quantities
  # at the quadrature points
  a_mod_stuff <- mapply(handle_assocmod, 1:M, m_mc, y_mod_stuff, SIMPLIFY = FALSE,
                        MoreArgs = list(e_mod_stuff = e_mod_stuff, assoc = assoc, 
                                        id_var = id_var, time_var = time_var, 
                                        eps = eps, dataAssoc = dataAssoc))
  a_mod_stuff <- structure(a_mod_stuff, auc_quadnodes = auc_quadnodes,
                           auc_quadweights = auc_quadweights, eps = eps,
                           K = get_num_assoc_pars(assoc, a_mod_stuff))
  
  #-----------
  # Fit model
  #-----------
  
  stanfit <- stan_jm.fit(y_mod_stuff = y_mod_stuff, e_mod_stuff = e_mod_stuff, 
                         a_mod_stuff = a_mod_stuff, assoc = assoc, 
                         time_var = time_var, id_var = id_var, family = family,
                         basehaz = basehaz, quadnodes = quadnodes,
                         init = init, offset = offset,
                         y_weights = y_weights, e_weights = e_weights, ...,				          
                         priorLong = priorLong, priorLong_intercept = priorLong_intercept, 
                         priorLong_aux = priorLong_aux, priorEvent = priorEvent, 
                         priorEvent_intercept = priorEvent_intercept, 
                         priorEvent_aux = priorEvent_aux, priorAssoc = priorAssoc, 
                         prior_covariance = prior_covariance, prior_PD = prior_PD, 
                         algorithm = algorithm, adapt_delta = adapt_delta, 
                         max_treedepth = max_treedepth, QR = QR, sparse = sparse, 
                         long_lp = long_lp, event_lp = event_lp)
  
  # Undo ordering of matrices if bernoulli
  y_mod_stuff <- lapply(y_mod_stuff, unorder_bernoulli)
  
  fit <- nlist(stanfit, family, formula = c(formulaLong, formulaEvent), 
               id_var, time_var, offset, weights, quadnodes, basehaz,
               y_mod_stuff, e_mod_stuff, a_mod_stuff, assoc, cnms, 
               dataLong, dataEvent, call, na.action, algorithm)
  out <- stanjm(fit)
  return(out)
}


