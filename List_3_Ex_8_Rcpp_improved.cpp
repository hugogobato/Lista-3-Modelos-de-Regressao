#include <iostream>
#include <cmath>
#include <Rcpp.h>
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <vector>
#include <algorithm>


using namespace std;
using namespace Eigen;


const int beta_size=2;
const double y_values[] = {2.3, 1.8, 1.5, 1.3, 1.2};  // Define values in a raw array
const int size_list = sizeof(y_values) / sizeof(y_values[0]);
const VectorXd y = Map<const VectorXd>(y_values, 5);
Matrix<double, 5, 2> x = (Matrix<double, 5, 2>() << 
                                1, 2,
                                1, 4,
                                1, 6,
                                1, 9,
                                1, 10).finished();

VectorXd mu(size_list);
VectorXd eta(size_list);  // Use VectorXd for eta to leverage Eigen’s vectorization
MatrixXd w = MatrixXd::Zero(size_list, size_list);  // Initialize a zero matrix 
VectorXd z(size_list); 
VectorXd beta_vector(beta_size); 

// Vectorized initialization of eta
void initiating_eta() {
    eta = (1.0 / y.array()).square().matrix();  // Compute (1/y)^2 for each element
}

// Vectorized computation of mu
void estimating_mu() {
    mu = (1.0 / eta.array()).sqrt().matrix();  // Compute sqrt(1/eta) for each element
}

// Vectorized computation of w’s diagonal elements
void estimating_w() {
    w.diagonal() = (1.0 / eta.array()).pow(1.5).matrix();  // Compute (1/eta)^1.5 for each element
}

// Vectorized computation of z
void estimating_z() {
    z = ((3 * mu.array() - 2 * y.array()) / mu.array().cube()).matrix();  // Compute (3*mu - 2*y) / mu^3 element-wise
}

void updating_beta(){
  beta_vector = (x.transpose() * w * x).inverse() * x.transpose() * w * z;

}

void updating_eta(){
    eta =(x*beta_vector);

}

double estimating_phi_inverse_gaussian() {
  double deviation_term = ((y - mu).array().square() / (y.array() * mu.array().square())).sum();
  double phi=deviation_term/size_list;
  return phi;
}


//const int number_iterations = 10;

// [[Rcpp::export]]
Rcpp::List beta_estimation(int number_iterations = 10) {
  for (int i = 0; i < number_iterations; i++) {
    if (i==0){
    initiating_eta();
    estimating_w();
    estimating_mu();
    estimating_z();
    updating_beta();
    updating_eta();

   /*
   // Print the values of eta, mu, w, z, and beta_vector
        cout << "Iteration " << i + 1 << ":\n";

        // Print eta
        cout << "Eta:\n";
        for (int j = 0; j < size_list; j++) {
            cout << eta[j] << " ";
        }
        cout << endl;

        // Print mu
        cout << "Mu:\n";
        for (int j = 0; j < size_list; j++) {
            cout << mu[j] << " ";
        }
        cout << endl;

        // Print w (as a matrix)
        cout << "W matrix:\n" << w << endl;

        // Print z
        cout << "Z:\n";
        for (int j = 0; j < size_list; j++) {
            cout << z[j] << " ";
        }
        cout << endl;

        // Print beta_vector
        cout << "Beta vector:\n" << beta_vector << endl;

        cout << "----------------------------------\n";
        */
    }
    else {
    estimating_w();
    estimating_mu();
    estimating_z();
    updating_beta();
    updating_eta();

    /*
   // Print the values of eta, mu, w, z, and beta_vector
        cout << "Iteration " << i + 1 << ":\n";

        // Print eta
        cout << "Eta:\n";
        for (int j = 0; j < size_list; j++) {
            cout << eta[j] << " ";
        }
        cout << endl;

        // Print mu
        cout << "Mu:\n";
        for (int j = 0; j < size_list; j++) {
            cout << mu[j] << " ";
        }
        cout << endl;

        // Print w (as a matrix)
        cout << "W matrix:\n" << w << endl;

        // Print z
        cout << "Z:\n";
        for (int j = 0; j < size_list; j++) {
            cout << z[j] << " ";
        }
        cout << endl;

        // Print beta_vector
        cout << "Beta vector:\n" << beta_vector << endl;

        cout << "----------------------------------\n";
        */
    }
    }

    double phi = estimating_phi_inverse_gaussian();

    // Print the result
    // cout << "Phi value: " << phi << endl;

    return Rcpp::List::create(
        Rcpp::Named("beta_vector") = Rcpp::wrap(beta_vector),
        Rcpp::Named("phi") = phi);

}

// [[Rcpp::export]]
Rcpp::NumericMatrix create_x_sequence(const Eigen::Map<Eigen::MatrixXd>& x, int length_out = 100) {
    int num_cols = x.cols();
    MatrixXd x_sequence(length_out, num_cols);

    // Generate a sequence for each column
    for (int col = 0; col < num_cols; ++col) {
        double min_x = x.col(col).minCoeff();
        double max_x = x.col(col).maxCoeff();
        double step = (max_x - min_x) / (length_out - 1);

        // Fill the column in x_sequence with the generated sequence
        for (int i = 0; i < length_out; ++i) {
            x_sequence(i, col) = min_x + i * step;
        }
    }

    return Rcpp::wrap(x_sequence);
}

// [[Rcpp::export]]
Rcpp::NumericVector predicted_y(const Eigen::Map<Eigen::MatrixXd>& x, const Eigen::Map<Eigen::VectorXd>& beta_vector) {
    // Ensure x has 2 columns as expected by create_x_sequence
    if (x.cols() != 2) {
        Rcpp::stop("Input matrix x must have exactly 2 columns.");
    }
    int pred_size=x.rows();
    VectorXd pred_eta(pred_size);
    pred_eta = x*beta_vector;
    VectorXd pred_mu(pred_size);
    pred_mu = ((1/pred_eta.array()).pow(0.5)).matrix();
    return Rcpp::wrap(pred_mu);
}