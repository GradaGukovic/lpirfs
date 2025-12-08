// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

using namespace Rcpp;
using namespace Eigen;

//' Fast Hodrick-Prescott Filter
//'
//' Decomposes a time series into trend and cyclical components using 
//' the Hodrick-Prescott (HP) filter. This implementation is highly optimized 
//' using sparse matrix algebra (via the Eigen library) and a fast Cholesky solver.
//' Was built with the help of Gemini 3.
//'
//' @param y_in A numeric vector containing the time series data to decompose.
//' @param lambda A numeric value determining the smoothness of the trend. 
//'   Common values are:
//'   \itemize{
//'     \item 1600 for quarterly data
//'     \item 14400 for monthly data
//'     \item 6.25 for annual data
//'   }
//'
//' @return A list with two components:
//' \describe{
//'   \item{trend}{A numeric vector containing the estimated trend component.}
//'   \item{cycle}{A numeric vector containing the estimated cyclical component (\code{y_in - trend}).}
//' }
//'
//' This function constructs the sparse matrix directly and uses a Simplicial LLT (Cholesky) solver.
//' This reduces the computational complexity from \eqn{O(N^3)} to linear time \eqn{O(N)}.
//'
//' @references
//' Hodrick, R. J., & Prescott, E. C. (1997). Postwar U.S. business cycles: an empirical investigation. 
//' \emph{Journal of Money, Credit, and Banking}, 1-16.
//'
//' @examples
//' \dontrun{
//' # Generate random data
//' set.seed(123)
//' y <- cumsum(rnorm(100))
//'
//' # Filter with lambda = 1600
//' res <- hp_filter_fast(y, 1600)
//'
//' # Plot
//' plot(y, type = "l", col = "gray", main = "HP Filter")
//' lines(res$trend, col = "red", lwd = 2)
//' }
//' @export
// [[Rcpp::export]]
List hp_filter_fast(NumericVector y_in, double lambda) {
    
    // Zero-copy map of the input vector
    Map<VectorXd> y(as<Map<VectorXd> >(y_in));
    int n = y.size();

    // A has exactly 5 diagonals. 
    // Total non-zeros approx 5*n. Reserve memory
    typedef Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(5 * n - 6);

    // --- 1. Fill the Diagonals Manually
    
    // The "Internal" values (bulk of the matrix)
    double diag_main = 1.0 + 6.0 * lambda;
    double diag_off1 = -4.0 * lambda;
    double diag_off2 = 1.0 * lambda;

    // Boundary values
    double diag_0 = 1.0 + lambda;
    double diag_1 = 1.0 + 5.0 * lambda;
    double off1_0 = -2.0 * lambda;

    // A. Main Diagonal (offset 0)
    tripletList.push_back(T(0, 0, diag_0));             // First
    tripletList.push_back(T(1, 1, diag_1));             // Second
    for(int i = 2; i < n - 2; ++i) {
        tripletList.push_back(T(i, i, diag_main));      // Middle
    }
    tripletList.push_back(T(n - 2, n - 2, diag_1));     // Second to last
    tripletList.push_back(T(n - 1, n - 1, diag_0));     // Last

    // B. First Off-Diagonal (offset +/- 1)
    tripletList.push_back(T(0, 1, off1_0));             // (0,1)
    tripletList.push_back(T(1, 0, off1_0));             // (1,0)
    
    for(int i = 1; i < n - 2; ++i) {
        tripletList.push_back(T(i, i + 1, diag_off1));  // Upper band
        tripletList.push_back(T(i + 1, i, diag_off1));  // Lower band
    }
    
    tripletList.push_back(T(n - 2, n - 1, off1_0));     // Last upper
    tripletList.push_back(T(n - 1, n - 2, off1_0));     // Last lower

    // C. Second Off-Diagonal (offset +/- 2)
    for(int i = 0; i < n - 2; ++i) {
        tripletList.push_back(T(i, i + 2, diag_off2));  // Upper band
        tripletList.push_back(T(i + 2, i, diag_off2));  // Lower band
    }

    // --- 2. Construct Matrix ---
    SparseMatrix<double> A(n, n);
    A.setFromTriplets(tripletList.begin(), tripletList.end());

    // --- 3. Solve (SimplicialLLT is fast for Symmetric Positive Definite) ---
    SimplicialLLT<SparseMatrix<double>> solver;
    solver.compute(A);
    
    VectorXd trend = solver.solve(y);
    VectorXd cycle = y - trend;

    return List::create(_["trend"] = trend, _["cycle"] = cycle);
}