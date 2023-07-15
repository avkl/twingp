#' A Fast Global-Local Gaussian Process Approximation
#'
#' @param x \code{n * d} numeric matrix representing the training features
#' @param y \code{n * 1} response vector corresponding to \code{x}
#' @param x_test \code{t * d} numeric matrix representing the \code{t} testing locations
#' @param nugget Boolean indicating if a nugget to model observation noise is included in the model, the default is \code{True}
#' @param twins Number of twinning samples computed to identify the best set of global points, the default is \code{5}
#' @param g_num Number of global points included in the model, the default is \code{min(50 * d, max(sqrt(n), 10 * d))}
#' @param l_num Number of local points included in the model, the default is \code{max(25, 3 * d)}
#' @param v_num Number of validation points, the default is \code{2 * g_num}
#'
#' @return A list of two \code{t * 1} vectors \code{mu}, and \code{sigma} representing the mean prediction and associated standard error corresponding to \code{x_test}
#'
#' @details We employ a combined global-local approach in building the Gaussian process approximation. Our framework uses a subset-of-data approach where the subset is a union of a set of global points designed to capture the global trend in the data, and a set of local points specific to a given testing location. We use \code{Twinning} (Vakayil and Joseph, 2022) to identify the set of global points. The local points are identified as the nearest neighbors to the testing location. The correlation function is also modeled as a combination of a global, and a local kernel. For further details on the methodology, please refer to Vakayil and Joseph (2023). 
#'
#' @export
#' @examples
#' \dontrun{
#' 
#' grlee12 = function(x) {
#'   term1 = sin(10 * pi * x) / (2 * x)
#'   term2 = (x - 1)^4
#'   y = term1 + term2
#'   return(y)
#' }
#' 
#' x = matrix(seq(0.5, 2.5, length=500), ncol=1)
#' y = apply(x, 1, grlee12) + rnorm(nrow(x)) * 0.1
#' x_test = matrix(seq(0.5, 2.5, length=2000), ncol=1)
#' y_test = apply(x_test, 1, grlee12)
#' 
#' result = twingp(x, y, x_test)
#' rmse = sqrt(mean((y_test - result$mu)^2))
#' }
#'
#' @references
#' Vakayil, A., & Joseph, V. R. (2023). A Global-Local Approximation Framework for Large-Scale Gaussian Process Modeling. ArXiv [Stat.ML]. http://arxiv.org/abs/2305.10158
#' 
#' Vakayil, A., & Joseph, V. R. (2022). Data Twinning. Statistical Analysis and Data Mining: The ASA Data Science Journal. https://doi.org/10.1002/sam.11574

twingp = function(x, y, x_test, nugget=TRUE, twins=5, g_num=NULL, l_num=NULL, v_num=NULL)
{
    d = ncol(x)
    N = nrow(x)
    
    if(is.null(l_num)) {
        l_num = max(25, 3 * d)
    }

    if(is.null(g_num)) {
        g_num = min(50 * d, max(sqrt(N), 10 * d))
    } 

    if(is.null(v_num)) {
        v_num = 2 * g_num
    }

    g = max(2, round(nrow(x) / g_num))
    v = max(2, round((nrow(x) - g_num) / v_num))

    m = apply(x, 2, min)
    M = apply(x, 2, max)

    x = sweep(x, 2, m, "-")
    x = sweep(x, 2, M - m, "/")
    x_test = sweep(x_test, 2, m, "-")
    x_test = sweep(x_test, 2, M - m, "/")

    twinning_result = get_twinIndices(cbind(x, (y - min(y)) / (max(y) - min(y))), r=g, rv=v, twins, sample(N, twins) - 1, 8)
    return(glgp_cpp(cbind(x, y), x_test, twinning_result$gIndices, twinning_result$theta_l, twinning_result$vIndices, l_num, nugget, 8))
}



