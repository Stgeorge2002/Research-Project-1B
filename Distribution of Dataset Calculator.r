# Checks if the moments package is installed and loads it
if(!require(moments)) {
  install.packages("moments")
  library(moments)
}

# Loads necessary libraries
library(moments) # for skewness and kurtosis

# Reads the CSV file
data <- read.csv("C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/WholeData.csv")

# Initializes an empty data frame for summary
summary_table <- data.frame(Variable = character(),
                            Mean = numeric(),
                            Median = numeric(),
                            StdDev = numeric(),
                            Skewness = numeric(),
                            Kurtosis = numeric(),
                            stringsAsFactors = FALSE)

# Loops through each column in the dataset
for (var in names(data)) {
  # Calculates distribution coefficients
  mean_val <- mean(data[[var]], na.rm = TRUE)
  median_val <- median(data[[var]], na.rm = TRUE)
  sd_val <- sd(data[[var]], na.rm = TRUE)
  skew_val <- skewness(data[[var]], na.rm = TRUE)
  kurt_val <- kurtosis(data[[var]], na.rm = TRUE)
  
  # Appends to the summary table
  summary_table <- rbind(summary_table, c(var, mean_val, median_val, sd_val, skew_val, kurt_val))
}

# Sets column names
colnames(summary_table) <- c("Variable", "Mean", "Median", "StdDev", "Skewness", "Kurtosis")

# Outputs the summary table
print(summary_table)

# This ptionally, write the summary table to a new CSV file
write.csv(summary_table, "C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/OUTPUT/variable_distribution_summary.csv", row.names = FALSE)
