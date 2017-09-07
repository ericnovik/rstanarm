  // prior family: 0 = none, 1 = normal, 2 = student_t, 3 = hs, 4 = hs_plus, 
  //   5 = laplace, 6 = lasso
  int<lower=0,upper=6> e_prior_dist_for_assoc;

  // data for association structure
  int<lower=0> e_A;                     // num. of association parameters
  int<lower=0,upper=1> assoc;           // 0 = no assoc structure, 1 = any assoc structure
  int<lower=0,upper=1> assoc_uses[6];   // which components required to build association terms
  int<lower=0,upper=1> has_assoc[16,M]; // which association terms does each submodel use
  int<lower=0> sum_size_which_b;        // num. of shared random effects
  int<lower=0> size_which_b[M];         // num. of shared random effects for each long submodel
  int<lower=1> which_b_zindex[sum_size_which_b]; // which random effects are shared for each long submodel
  int<lower=0> sum_size_which_coef;     // num. of shared random effects incl fixed component
  int<lower=0> size_which_coef[M];      // num. of shared random effects incl fixed component for each long submodel
  int<lower=1> which_coef_zindex[sum_size_which_coef]; // which random effects are shared incl fixed component
  int<lower=1> which_coef_xindex[sum_size_which_coef]; // which fixed effects are shared
  int<lower=0,upper=e_A> sum_a_K_data;  // total num pars used in assoc*data interactions
  int<lower=0,upper=sum_a_K_data> a_K_data[M*4]; // num pars used in assoc*data interactions, by submodel and by ev/es/mv/ms interactions
  int<lower=0> sum_size_which_interactions; // total num pars used in assoc*assoc interactions
  int<lower=0,upper=sum_size_which_interactions> size_which_interactions[M*4]; // num pars used in assoc*assoc interactions, by submodel and by evev/evmv/mvev/mvmv interactions
  int<lower=1> which_interactions[sum_size_which_interactions];  // which terms to interact with

  // data for calculating eta in GK quadrature
  int<lower=0> nrow_y_Xq[M];      // num. rows in long. predictor matrix at quadpoints
  int<lower=0> idx_q[M,2];        // indices of first and last rows in eta at quadpoints
  matrix[sum(nrow_y_Xq)*(assoc_uses[1]>0),K] y_Xq_eta; // predictor matrix (long submodel) at quadpoints, centred     
  int<lower=0> nnz_Zq_eta;        // number of non-zero elements in the Z matrix (at quadpoints)
  vector[nnz_Zq_eta] w_Zq_eta;    // non-zero elements in the implicit Z matrix (at quadpoints)
  int<lower=0> v_Zq_eta[nnz_Zq_eta]; // column indices for w (at quadpoints)
  int<lower=0> u_Zq_eta[(sum(nrow_y_Xq)*(assoc_uses[1]>0) + 1)]; // where the non-zeros start in each row (at quadpoints)

  // data for calculating slope in GK quadrature
  real<lower=0> eps;  // time shift used for numerically calculating derivative
  matrix[sum(nrow_y_Xq)*(assoc_uses[2]>0),K] 
    y_Xq_eps; // predictor matrix (long submodel) at quadpoints plus time shift of epsilon              
  int<lower=0> nnz_Zq_eps;        // number of non-zero elements in the Zq_eps matrix (at quadpoints plus time shift of epsilon)
  vector[nnz_Zq_eps] w_Zq_eps;    // non-zero elements in the implicit Zq_eps matrix (at quadpoints plus time shift of epsilon)
  int<lower=0> v_Zq_eps[nnz_Zq_eps]; // column indices for w (at quadpoints plus time shift of epsilon)
  int<lower=0> u_Zq_eps[(sum(nrow_y_Xq)*(assoc_uses[2]>0) + 1)]; 
    // where the non-zeros start in each row (at quadpoints plus time shift of epsilon)

  // data for calculating auc in GK quadrature
  int<lower=0> nrow_y_Xq_auc[M];     // num. rows in long. predictor matrix at auc quadpoints
  int<lower=0> idx_qauc[M,2];        // indices of first and last row in eta at auc quadpoints corresponding to each submodel
  int<lower=0> auc_quadnodes;     // num. of nodes for Gauss-Kronrod quadrature for area under marker trajectory 
  vector[sum(nrow_y_Xq_auc)*(assoc_uses[3]>0)] auc_quadweights;
  matrix[sum(nrow_y_Xq_auc)*(assoc_uses[3]>0),K] 
    y_Xq_auc; // predictor matrix (long submodel) at auc quadpoints            
  int<lower=0> nnz_Zq_auc;        // number of non-zero elements in the Zq_lag matrix (at auc quadpoints)
  vector[nnz_Zq_auc] w_Zq_auc;    // non-zero elements in the implicit Zq_lag matrix (at auc quadpoints)
  int<lower=0> v_Zq_auc[nnz_Zq_auc]; // column indices for w (at auc quadpoints)
  int<lower=0> u_Zq_auc[(sum(nrow_y_Xq_auc)*(assoc_uses[3]>0) + 1)]; 
    // where the non-zeros start in each row (at auc quadpoints)

  // data for calculating assoc*data interactions in GK quadrature
  matrix[sum(nrow_y_Xq),sum_a_K_data] y_Xq_data; // design matrix for interacting with ev/es/mv/ms at quadpoints
  
  // data for combining lower level units clustered within patients
  int<lower=0,upper=1> has_clust[M]; // 1 = has clustering below patient level
  int<lower=0> clust_nnz; // info on sparse design matrix used for combining lower level units clustered within patients
  vector<lower=0,upper=1>[clust_nnz] clust_w; 
  int<lower=0> clust_v[clust_nnz];      
  int<lower=0> clust_u[sum(has_clust) > 0 ? nrow_e_Xq + 1 : 0]; 
