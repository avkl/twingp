/*
    Copyright 2023 Akhil Vakayil (akhilv@gatech.edu). All rights reserved.
    License: Apache-2.0
*/

// [[Rcpp::plugins("cpp11")]]
#include <memory>
#include <vector>
#include <Rcpp.h>
#include <cmath>
#include <limits>
#include <nloptrAPI.h>
#include <Eigen/Dense>
#include <nanoflann.hpp>


class DF
{
private:
	std::shared_ptr<Rcpp::NumericMatrix> df_;

public:
	void import_data(Rcpp::NumericMatrix& df) { df_ = std::make_shared<Rcpp::NumericMatrix>(Rcpp::transpose(df)); }

	const double* get_row(const std::size_t idx) const { return &(*df_)(0, idx); }

    double get_value(const std::size_t idx, const std::size_t dim) const { return (*df_)(dim, idx); }

    std::size_t kdtree_get_point_count() const { return df_->cols(); }

	double kdtree_get_pt(const std::size_t idx, const std::size_t dim) const { return (*df_)(dim, idx); }

    template <class BBOX>
	bool kdtree_get_bbox(BBOX&) const { return false; }
};

double mse(unsigned n, const double* sParams, double* grad, void* mse_data);

double nllg(unsigned n, const double* gParams, double* grad, void* nllg_data);

typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<nanoflann::L2_Adaptor<double, DF>, DF, -1, std::size_t> kdTree;


class GP
{
private:
    DF xy_;
    DF x_test_;
    std::size_t dim_;
    std::size_t gNum_;
    std::size_t lNum_;
    std::vector<std::size_t> gIndices_;
    std::vector<double> gParams_;
    bool nugget_;
    double lam_;
    double nug_;
    double l_;

    double theta_;
    std::vector<std::size_t> predIndices_;

    Eigen::MatrixXd Rg_;
    Eigen::MatrixXd Rl_;
    Eigen::MatrixXd Ainv_;
    Eigen::VectorXd yg_;

    Eigen::VectorXd oneVecG_;
    Eigen::VectorXd oneVecL_;
    Eigen::VectorXd oneVecGL_;

    kdTree* tree_;
    std::size_t leaf_size_;

public:
    GP(Rcpp::NumericMatrix& xy, Rcpp::NumericMatrix& x_test, std::vector<std::size_t>& gIndices, double theta, 
        std::vector<std::size_t>& predIndices, std::size_t lNum, std::size_t leaf_size, bool nugget=false)
    {
        xy_.import_data(xy);
        x_test_.import_data(x_test);

        dim_ = xy.cols() - 1;
        gIndices_ = gIndices;
        gNum_ = gIndices_.size();
        predIndices_ = predIndices;

        l_ = std::floor(dim_ / 2.0) + 2.0;
        nugget_ = nugget;
        gParams_.resize(dim_ + 2);

        Rg_.resize(gNum_, gNum_);
        Rl_.resize(gNum_, gNum_);
        Ainv_.resize(gNum_, gNum_);
        yg_.resize(gNum_);

        lNum_ = lNum;
        theta_ = theta;
        leaf_size_ = leaf_size;
        tree_ = new kdTree(dim_, xy_, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_size_));
        for(std::size_t i = 0; i < gNum_; i++)
        {
            yg_(i) = xy_.get_value(gIndices_[i], dim_);
            tree_->removePoint(gIndices_[i]);
        }

        oneVecG_ = Eigen::VectorXd::Constant(gNum_, 1.0);
        oneVecL_ = Eigen::VectorXd::Constant(lNum_, 1.0);
        oneVecGL_ = Eigen::VectorXd::Constant(gNum_ + lNum_, 1.0);
    }

    ~GP() { delete tree_; }

    void set_gParams(std::vector<double>& gParams) { gParams_ = gParams; }

    void estimate_gParams();

    void estimate_sParams();

    void find_RgRl();

    void find_Ainv(double lam, double nugget);

    double get_mse(double lam, double nugget);

    double get_nllg(const double* gParams);

    void predict(std::size_t index, double lam, double nugget, double* pred, void* sigma, bool test=false);

    Rcpp::List gp_predict();
};


void GP::find_RgRl()
{
    #pragma omp parallel for
    for(std::size_t i = 0; i < gNum_; i++)
        for(std::size_t j = i; j < gNum_; j++)
        {
            if(i == j)
            {
                Rg_(i, j) = 1.0;
                Rl_(i, j) = 1.0;
            }
            else
            {
                double dist = 0.0;
                double value = 0.0;
                const double* row_i = xy_.get_row(gIndices_[i]);
                const double* row_j = xy_.get_row(gIndices_[j]);
                for(std::size_t d = 0; d < dim_; d++)
                {
                    double temp = std::abs(*(row_i + d) - *(row_j + d));
                    value -= gParams_[d] * std::pow(temp, gParams_[dim_]);
                    dist += std::pow(temp, 2);
                }
                
                double r = std::sqrt(dist) / theta_;
                double w = std::pow(std::max(0.0, 1.0 - r), l_ + 1.0) * ((l_ + 1.0) * r + 1.0);
                
                Rl_(i, j) = w;
                Rl_(j, i) = Rl_(i, j);

                Rg_(i, j) = std::exp(value);
                Rg_(j, i) = Rg_(i, j);
            }
        }
}


void GP::find_Ainv(double lam, double nugget)
{
    Eigen::MatrixXd A = (1 - lam) * Rg_ + lam * Rl_;
    for(std::size_t i = 0; i < gNum_; i++)
        A(i, i) += nugget;
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigSolver;
    eigSolver.compute(A);
    Eigen::MatrixXd AEigVectors = eigSolver.eigenvectors();
    Eigen::VectorXd AEigValues = eigSolver.eigenvalues();

    double kappa = AEigValues.maxCoeff() / AEigValues.minCoeff();
    double delta = std::max(0.0, AEigValues.minCoeff() * (kappa - std::exp(20.0)) / (std::exp(20.0) - 1.0));
    Ainv_ = AEigVectors * (AEigValues + oneVecG_ * delta).cwiseInverse().asDiagonal() * AEigVectors.transpose();
}


void GP::estimate_sParams()
{
    std::vector<double> lb = {std::log(1e-7), std::log(1e-7)};
    std::vector<double> ub = {std::log(0.999), 0.0};

    int opt_dim = nugget_ ? 2 : 1;
    int max_eval = 20;
    
    nlopt_opt optimizer = nlopt_create(NLOPT_LN_SBPLX, opt_dim);
    nlopt_set_lower_bounds(optimizer, &lb[0]);
    nlopt_set_upper_bounds(optimizer, &ub[0]);
    nlopt_set_min_objective(optimizer, mse, this);
    nlopt_set_maxeval(optimizer, max_eval);

    std::vector<double> sParams = {std::log(1e-1), nugget_ ? std::log(1e-3) : std::log(1e-7)};
    double mse_min;
    nlopt_optimize(optimizer, &sParams[0], &mse_min);
    nlopt_destroy(optimizer);
    lam_ = std::exp(sParams[0]);
    nug_ = (1.0 - lam_) * gParams_[dim_ + 1] + lam_ * std::exp(sParams[1]);
}


void GP::estimate_gParams()
{
    std::vector<double> lb, ub;
    lb.resize(dim_ + 2);
    ub.resize(dim_ + 2);
    for(std::size_t i = 0; i < dim_; i++)
    {
        lb[i] = 1e-7;
        ub[i] = 1000.0;
    }

    lb[dim_] = 1.0;
    ub[dim_] = 2.0;
    lb[dim_ + 1] = std::log(1e-7);
    ub[dim_ + 1] = 0.0;

    int opt_dim = nugget_ ? dim_ + 2 : dim_ + 1;
    double nugget = nugget_ ? std::log(1e-3) : std::log(1e-7);

    Eigen::ArrayXd rho = Eigen::ArrayXd::Constant(dim_, std::sqrt(dim_)).inverse();
    Eigen::ArrayXd alpha = Eigen::ArrayXd::LinSpaced(11, 3.0, -3.0).exp();

    std::vector<double> nllg_values;
    nllg_values.resize(alpha.size());

    #pragma omp parallel for
    for(std::size_t i = 0; i < static_cast<std::size_t>(alpha.size()); i++)
    {
        std::vector<double> gParams;
        gParams.resize(dim_ + 2);
        gParams[dim_] = 1.95;
        gParams[dim_ + 1] = nugget;
        for(std::size_t j = 0; j < dim_; j++)
            gParams[j] = alpha(i) * rho(j);

        nllg_values[i] = nllg(opt_dim, &gParams[0], NULL, this);
    }

    int min_index = std::min_element(nllg_values.begin(), nllg_values.end()) - nllg_values.begin();
    std::size_t num_opt = std::min(11, std::max(3, Eigen::nbThreads()));
    Eigen::ArrayXd factor = Eigen::ArrayXd::LinSpaced(num_opt, -0.5, 0.5).exp();
    std::vector<nlopt_opt> optimizer;
    optimizer.resize(num_opt);
    int max_eval = std::min(500.0, 100.0 * std::log(1.0 + dim_));
    for(std::size_t i = 0; i < num_opt; i++)
    {
        optimizer[i] = nlopt_create(NLOPT_LN_SBPLX, opt_dim);
        nlopt_set_lower_bounds(optimizer[i], &lb[0]);
        nlopt_set_upper_bounds(optimizer[i], &ub[0]);
        nlopt_set_min_objective(optimizer[i], nllg, this);
        nlopt_set_maxeval(optimizer[i], max_eval);
    }
    
    std::vector<std::pair<double, std::vector<double>>> opt_results;
    opt_results.resize(num_opt);

    #pragma omp parallel for
    for(std::size_t i = 0; i < num_opt; i++)
    {
        std::vector<double> gParams;
        gParams.resize(dim_ + 2);
        gParams[dim_] = 1.95;
        gParams[dim_ + 1] = nugget;
        for(std::size_t j = 0; j < dim_; j++)
            gParams[j] = alpha(min_index) * rho(j) * factor(i);
        
        double nllg_min;
        nlopt_optimize(optimizer[i], &gParams[0], &nllg_min);
        opt_results[i].first = nllg_min;
        opt_results[i].second = gParams;
    }

    double nllg_min = std::numeric_limits<double>::max();
    std::vector<double> gParams;
    for(std::size_t i = 0; i < num_opt; i++)
    {
        if(opt_results[i].first < nllg_min)
        {
            nllg_min = opt_results[i].first;
            gParams = opt_results[i].second;
        }

        nlopt_destroy(optimizer[i]);
    }

    gParams[dim_ + 1] = std::exp(gParams[dim_ + 1]);
    set_gParams(gParams);
    find_RgRl();
}


Rcpp::List GP::gp_predict()
{
    find_Ainv(lam_, nug_);
    std::size_t test_num = x_test_.kdtree_get_point_count();
    std::vector<double> predictions, sigmas;
    predictions.resize(test_num);
    sigmas.resize(test_num);

    #pragma omp parallel for
    for(std::size_t i = 0; i < test_num; i++)
    {
        double pred, sigma;
        predict(i, lam_, nug_, &pred, &sigma, true);
        predictions[i] = pred;
        sigmas[i] = sigma;
    }

    Rcpp::List returns;
    returns["mu"] = predictions;
    returns["sigma"] = sigmas;
    return returns;
}


void GP::predict(std::size_t ind, double lam, double nugget, double* pred, void* sigma, bool test)
{
    std::size_t nn = test ? lNum_ : lNum_ + 1;
    nanoflann::KNNResultSet<double> resultSet(nn);
    std::size_t *index = new std::size_t[nn];
    double *distance = new double[nn];

    resultSet.init(index, distance);
    if(test)
        tree_->findNeighbors(resultSet, x_test_.get_row(ind));
    else
        tree_->findNeighbors(resultSet, xy_.get_row(ind));

    Eigen::VectorXd yl(lNum_);
    for(std::size_t j = 0; j < lNum_; j++)
        yl(j) = xy_.get_value(index[j + nn - lNum_], dim_);

    Eigen::VectorXd y(gNum_ + lNum_);
    y << yg_, yl;

    double value, dist, temp, r, w;
    Eigen::MatrixXd D(lNum_, lNum_);
    for(std::size_t u = 0; u < lNum_; u++)
        for(std::size_t v = u; v < lNum_; v++)
        {
            if(u == v)
                D(u, v) = 1.0 + nugget;
            else
            {
                value = 0.0;
                dist = 0.0;
                const double* row_u = xy_.get_row(index[u + nn - lNum_]);
                const double* row_v = xy_.get_row(index[v + nn - lNum_]);
                for(std::size_t d = 0; d < dim_; d++)
                {
                    temp = std::abs(*(row_u + d) - *(row_v + d));
                    value -= gParams_[d] * std::pow(temp, gParams_[dim_]);
                    dist += std::pow(temp, 2);
                }

                r = std::sqrt(dist) / theta_;
                w = std::pow(std::max(0.0, 1.0 - r), l_ + 1.0) * ((l_ + 1.0) * r + 1.0);

                D(u, v) = (1.0 - lam) * std::exp(value) + lam * w;
                D(v, u) = D(u, v);
            }
        }
    
    Eigen::MatrixXd B(gNum_, lNum_);
    for(std::size_t u = 0; u < gNum_; u++)
        for(std::size_t v = 0; v < lNum_; v++)
        {
            value = 0.0;
            dist = 0.0;
            const double* row_u = xy_.get_row(gIndices_[u]);
            const double* row_v = xy_.get_row(index[v + nn - lNum_]);
            for(std::size_t d = 0; d < dim_; d++)
            {
                temp = std::abs(*(row_u + d) - *(row_v + d));
                value -= gParams_[d] * std::pow(temp, gParams_[dim_]);
                dist += std::pow(temp, 2);
            }

            r = std::sqrt(dist) / theta_;
            w = std::pow(std::max(0.0, 1.0 - r), l_ + 1.0) * ((l_ + 1.0) * r + 1.0);

            B(u, v) = (1.0 - lam) * std::exp(value) + lam * w;
        }

    Eigen::MatrixXd CAinv =  B.transpose() * Ainv_; 
    Eigen::MatrixXd S = D - CAinv * B;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigSolver;
    eigSolver.compute(S);
    Eigen::MatrixXd SEigVectors = eigSolver.eigenvectors();
    Eigen::VectorXd SEigValues = eigSolver.eigenvalues();

    double kappa = SEigValues.maxCoeff() / SEigValues.minCoeff();
    double delta = std::max(0.0, SEigValues.minCoeff() * (kappa - std::exp(20.0)) / (std::exp(20.0) - 1.0));
    Eigen::MatrixXd Sinv = SEigVectors * (SEigValues + oneVecL_ * delta).cwiseInverse().asDiagonal() * SEigVectors.transpose();

    Eigen::MatrixXd CAinvTSinv = CAinv.transpose() * Sinv;
    Eigen::MatrixXd Rinv(gNum_ + lNum_, gNum_ + lNum_);
    Rinv.block(0, 0, gNum_, gNum_) = Ainv_ + CAinvTSinv * CAinv;
    Rinv.block(0, gNum_, gNum_, lNum_) = -1.0 * CAinvTSinv;
    Rinv.block(gNum_, 0, lNum_, gNum_) = -1.0 * CAinvTSinv.transpose();
    Rinv.block(gNum_, gNum_, lNum_, lNum_) = Sinv;
    
    Eigen::VectorXd rVec(gNum_ + lNum_);
    const double* row = test ? x_test_.get_row(ind) : xy_.get_row(ind);
    for(std::size_t u = 0; u < gNum_ + lNum_; u++)
    {
        value = 0.0;
        dist = 0.0;
        const double* row_u = (u < gNum_) ? xy_.get_row(gIndices_[u]) : xy_.get_row(index[u - gNum_ + nn - lNum_]);
        for(std::size_t d = 0; d < dim_; d++)
        {
            temp = std::abs(*(row_u + d) - *(row + d));
            value -= gParams_[d] * std::pow(temp, gParams_[dim_]);
            dist += std::pow(temp, 2);
        }

        r = std::sqrt(dist) / theta_;
        w = std::pow(std::max(0.0, 1.0 - r), l_ + 1.0) * ((l_ + 1.0) * r + 1.0);

        rVec(u) = (1 - lam) * std::exp(value) + lam * w;
    }     
    
    Eigen::MatrixXd mu = (Rinv.colwise().sum() * y) / Rinv.sum();
    Eigen::VectorXd y_mu = y - oneVecGL_ * mu;
    Eigen::MatrixXd prediction = mu + rVec.transpose() * Rinv * y_mu;
    *pred = prediction(0, 0);

    if(sigma)
    {
        Eigen::MatrixXd tau2 = (1.0 / (gNum_ + lNum_)) * y_mu.transpose() * Rinv * y_mu;
        double* sig = (double*) sigma;
        *sig = std::sqrt(tau2(0, 0) * std::max(1e-7, 1.0 + nugget - rVec.transpose() * Rinv * rVec));
    }
}


double mse(unsigned n, const double* sParams, double* grad, void* mse_data)
{
    GP* glgp = (GP*) mse_data;
    return glgp->get_mse(std::exp(sParams[0]), std::exp(sParams[1]));
}


double GP::get_mse(double lam, double nugget)
{
    nugget = (1.0 - lam) * gParams_[dim_ + 1] + lam * nugget;
    find_Ainv(lam, nugget);
    double mse = 0.0;

    #pragma omp parallel for reduction(+ : mse)
    for(std::size_t i = 0; i < static_cast<std::size_t>(predIndices_.size()); i++)
    {
        double pred;
        predict(predIndices_[i], lam, nugget, &pred, NULL);
        mse += std::pow(xy_.get_value(predIndices_[i], dim_) - pred, 2);
    }
    
    return mse;
}


double nllg(unsigned n, const double* gParams, double* grad, void* nllg_data)
{
    GP* glgp = (GP*) nllg_data;
    return glgp->get_nllg(gParams);
}


double GP::get_nllg(const double* gParams)
{
    double value;
    double nugget = std::exp(gParams[dim_ + 1]);
    Eigen::MatrixXd Rg(gNum_, gNum_);
    for(std::size_t i = 0; i < gNum_; i++)
        for(std::size_t j = i; j < gNum_; j++)
        {
            if(i == j)
                Rg(i, j) = 1.0 + nugget;
            else
            {
                value = 0.0;
                const double* row_i = xy_.get_row(gIndices_[i]);
                const double* row_j = xy_.get_row(gIndices_[j]);
                for(std::size_t d = 0; d < dim_; d++)
                    value -= gParams[d] * std::pow(std::abs(*(row_i + d) - *(row_j + d)), gParams[dim_]);

                Rg(i, j) = std::exp(value);
                Rg(j, i) = Rg(i, j);
            }
        }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigSolver;
    eigSolver.compute(Rg);
    Eigen::MatrixXd RgEigVectors = eigSolver.eigenvectors();
    Eigen::VectorXd RgEigValues = eigSolver.eigenvalues();

    double kappa = RgEigValues.maxCoeff() / RgEigValues.minCoeff();
    double delta = std::max(0.0, RgEigValues.minCoeff() * (kappa - std::exp(20.0)) / (std::exp(20.0) - 1.0));
    
    Eigen::MatrixXd RgInv = RgEigVectors * (RgEigValues + oneVecG_ * delta).cwiseInverse().asDiagonal() * RgEigVectors.transpose();
    double RgLogDet = (RgEigValues + oneVecG_ * delta).array().log().sum();

    Eigen::MatrixXd mu = (RgInv.colwise().sum() * yg_) / RgInv.sum();
    Eigen::VectorXd yg_mu = yg_ - oneVecG_ * mu;
    Eigen::MatrixXd tau2 = (1.0 / gNum_) * yg_mu.transpose() * RgInv * yg_mu;

    return gNum_ * std::log(tau2(0, 0)) + RgLogDet;
}


// [[Rcpp::export]]
Rcpp::List glgp_cpp(Rcpp::NumericMatrix& xy, Rcpp::NumericMatrix& x_test, std::vector<std::size_t>& gIndices, double theta, 
    std::vector<std::size_t>& predIndices, std::size_t lNum, bool nugget, std::size_t leaf_size)
{
    GP glgp(xy, x_test, gIndices, theta, predIndices, lNum, leaf_size, nugget);
    glgp.estimate_gParams();
    glgp.estimate_sParams();
    return glgp.gp_predict();
}


