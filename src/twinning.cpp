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
#include <algorithm>
#include <numeric>
#include <nanoflann.hpp>


class DF2
{
private:
    std::shared_ptr<Rcpp::NumericMatrix> df_;
    bool subset_ = false;
    std::vector<std::size_t>* indices_;

public:
    void import_data(Rcpp::NumericMatrix& df) { df_ = std::make_shared<Rcpp::NumericMatrix>(Rcpp::transpose(df)); }
    
    std::size_t kdtree_get_point_count() const { return subset_ ? indices_->size() : df_->cols(); }

    double kdtree_get_pt(const std::size_t idx, const std::size_t dim) const { return subset_ ? (*df_)(dim, indices_->at(idx)) : (*df_)(dim, idx); }
    
    const double* get_row(const std::size_t idx) const { return &(*df_)(0, idx); }
    
    void subset_on(std::vector<std::size_t>* indices) 
    { 
        subset_ = true; 
        indices_ = indices;
    }
    
    void subset_off() { subset_ = false; }
    
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};


typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<nanoflann::L2_Adaptor<double, DF2>, DF2, -1, std::size_t> kdTree;


class KDTree
{
private:
    const std::size_t dim_;
    const std::size_t N_;
    const std::size_t r_;
    const std::size_t rv_;
    const std::size_t runs_;
    const std::vector<std::size_t> u1_;
    const std::size_t leaf_size_;
    DF2 data_;
    Rcpp::List returns_;

public:
	KDTree(Rcpp::NumericMatrix& data, std::size_t r, std::size_t rv, std::size_t runs, std::vector<std::size_t>& u1, std::size_t leaf_size) : 
	dim_(data.cols()), N_(data.rows()), r_(r), rv_(rv), runs_(runs), u1_(u1), leaf_size_(leaf_size)
	{
		data_.import_data(data);
	}

	Rcpp::List twin()
	{
        std::vector<std::vector<std::size_t>> all_indices;
        std::vector<double> min_dist;
        all_indices.resize(runs_);
        min_dist.resize(runs_);

        #pragma omp parallel for
        for(std::size_t run = 0; run < runs_; run++)
        {
            kdTree tree(dim_, data_, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_size_));
            nanoflann::KNNResultSet<double> resultSet(r_);
            std::size_t *index = new std::size_t[r_];
            double *distance = new double[r_];

            nanoflann::KNNResultSet<double> resultSet_next_u(1);
            std::size_t index_next_u;
            double distance_next_u;

            std::vector<std::size_t> indices;
            indices.reserve(N_ / r_ + 1);
            std::size_t position = u1_[run];

            while(true)
            {
                resultSet.init(index, distance);
                tree.findNeighbors(resultSet, data_.get_row(position));
                indices.push_back(index[0]);

                for(std::size_t i = 0; i < r_; i++)
                    tree.removePoint(index[i]);

                resultSet_next_u.init(&index_next_u, &distance_next_u);
                tree.findNeighbors(resultSet_next_u, data_.get_row(index[r_ - 1]));	
                position = index_next_u;

                if(N_ - indices.size() * r_ <= r_)
                {
                    indices.push_back(position);
                    break;
                }
            }

            delete[] index;
            delete[] distance;

            all_indices[run] = indices;
            double dist;
            double min_dist_value = std::numeric_limits<double>::max();
            for(std::size_t i = 0; i < indices.size(); i++)
                for(std::size_t j = i + 1; j < indices.size(); j++)
                {
                    const double* u = data_.get_row(indices[i]);  
                    const double* v = data_.get_row(indices[j]);   
                    dist = 0.0;
                    for(std::size_t k = 0; k < dim_ - 1; k++)
                        dist += std::pow(*(u + k) - *(v + k), 2);
                    
                    if(dist < min_dist_value)
                        min_dist_value = dist;
                }

            min_dist[run] = min_dist_value;
        }

        int max_index = std::max_element(min_dist.begin(), min_dist.end()) - min_dist.begin();  
        returns_["gIndices"] = all_indices[max_index];
        returns_["theta_l"] = theta_l();
        returns_["vIndices"] = vIndices();
        return returns_;
	}

    double theta_l()
    {
        std::vector<std::size_t> twinIndices = returns_["gIndices"];
        data_.subset_on(&twinIndices);
        kdTree tree(dim_ - 1, data_, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_size_));
        data_.subset_off();

        std::vector<double> distances;
        distances.resize(N_);

        #pragma omp parallel for
        for(std::size_t i = 0; i < N_; i++)
        {
            nanoflann::KNNResultSet<double> resultSet(1);
            std::size_t index;
            double distance;

            resultSet.init(&index, &distance);
            tree.findNeighbors(resultSet, data_.get_row(i));
            distances[i] = distance;
        }

        std::sort(distances.begin(), distances.end());
        std::size_t q01 = (N_ - twinIndices.size()) * 0.01;
        return distances[N_ - q01 - 1];
    }

    std::vector<std::size_t> vIndices()
    {
        kdTree tree(dim_, data_, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_size_));
        std::vector<std::size_t> twinIndices = returns_["gIndices"];
        for(std::size_t i = 0; i < twinIndices.size(); i++)
            tree.removePoint(twinIndices[i]);

        nanoflann::KNNResultSet<double> resultSet(rv_);
        std::size_t *index = new std::size_t[rv_];
        double *distance = new double[rv_];

        nanoflann::KNNResultSet<double> resultSet_next_u(1);
        std::size_t index_next_u;
        double distance_next_u;

        std::size_t N = N_ - twinIndices.size();
        std::vector<std::size_t> indices;
        indices.reserve(N / rv_ + 1);
        
        std::size_t position = 0;
        std::sort(twinIndices.begin(), twinIndices.end());
        std::size_t marker = 0;
        while(true)
        {
            if(position < twinIndices[marker])
                break;
            else
            {
                position++;
                marker++;
            }
        }

        while(true)
        {
            resultSet.init(index, distance);
            tree.findNeighbors(resultSet, data_.get_row(position));
            indices.push_back(index[0]);

            for(std::size_t i = 0; i < rv_; i++)
                tree.removePoint(index[i]);

            resultSet_next_u.init(&index_next_u, &distance_next_u);
            tree.findNeighbors(resultSet_next_u, data_.get_row(index[rv_ - 1]));	
            position = index_next_u;

            if(N - indices.size() * rv_ <= rv_)
            {
                indices.push_back(position);
                break;
            }
        }

        delete[] index;
        delete[] distance;
        return indices;
    }
};


// [[Rcpp::export]]
Rcpp::List get_twinIndices(Rcpp::NumericMatrix& data, std::size_t r, std::size_t rv, std::size_t runs, std::vector<std::size_t>& u1, std::size_t leaf_size=8)
{
	KDTree tree(data, r, rv, runs, u1, leaf_size);
	return tree.twin();
}

