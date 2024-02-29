# Read the CSV file
data <- read.csv("C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/Only data.csv")


# Min-Max Scaling: This method scales the data to a fixed range, typically [0, 1]. 
# It's simple but sensitive to outliers. If your data is extremely skewed, 
# this might not be the best option, as outliers can distort the scale for the rest of the data.

min_max_scaling <- function(x) {
    (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}
scaled_data_min_max <- as.data.frame(lapply(data, min_max_scaling))


# Standard Scaling (Z-Score Normalization): This involves subtracting the mean and dividing
# by the standard deviation. It's effective if your data, despite being non-normal,
# is symmetric and doesn't have extreme outliers.

z_score_scaling <- function(x) {
    (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
}
scaled_data_z_score <- as.data.frame(lapply(data, z_score_scaling))


# Robust Scaling: This method uses the median and the interquartile range (IQR). 
# It is less sensitive to outliers than min-max scaling or z-score normalization. 
# Robust scaling is a good choice if your data contains many outliers.

robust_scaling <- function(x) {
    (x - median(x, na.rm = TRUE)) / IQR(x, na.rm = TRUE)
}
scaled_data_robust <- as.data.frame(lapply(data, robust_scaling))


# Log Transformation: A log transformation can help in stabilizing the variance and making 
# the distribution more normal-like, especially for positively skewed data. 
# However, it can't be applied to zero or negative values without adjustments.

# Adding 1 to avoid log(0) which is undefined
log_transform <- function(x) {
    log(x + 1)
}
transformed_data_log <- as.data.frame(lapply(data, log_transform))


# Log Transformation: A log transformation can help in stabilizing the variance and making 
# the distribution more normal-like, especially for positively skewed data. 
# However, it can't be applied to zero or negative values without adjustments.

# Install the 'caret' package if not already installed
if(!require(caret)) {
    install.packages("caret")
    library(caret)
}

# Box-Cox Transformation
transformed_data_boxcox <- preProcess(data, method = "BoxCox")
scaled_data_boxcox <- predict(transformed_data_boxcox, newdata = data)

# Yeo-Johnson Transformation
transformed_data_yeojohnson <- preProcess(data, method = "YeoJohnson")
scaled_data_yeojohnson <- predict(transformed_data_yeojohnson, newdata = data)


# Quantile Transformation: This transformation maps the data to a uniform or Gaussian distribution. 
# It's very useful for datasets with unusual distributions and works well with outliers.

# Install the 'bestNormalize' package if not already installed
if(!require(bestNormalize)) {
    install.packages("bestNormalize")
    library(bestNormalize)
}

quantile_transform <- function(x) {
    bn_fit <- bestNormalize(x)
    bn_fit$x.t
}
transformed_data_quantile <- as.data.frame(lapply(data, quantile_transform))


# Rank Transformation: Converting values to their rank order is another way to deal with highly
# skewed data. It's particularly useful when the exact differences between values are not
# as important as their relative order.

rank_transform <- function(x) {
    rank(x, na.last = "keep")
}
transformed_data_rank <- as.data.frame(lapply(data, rank_transform))

