library(Rcpp)
library(RcppEigen)
sourceCpp("List_3_Ex_8_Rcpp_improved.cpp")

x <- matrix(c(1, 2, 1, 4, 1, 6, 1, 9, 1, 10), ncol = 2, byrow = TRUE)
y <- c(2.3, 1.8, 1.5, 1.3, 1.2)

result <- beta_estimation()
beta_vector <- result$beta_vector
phi <- result$phi

cat("Estimated Beta parameters:", beta_vector, "\n")
cat("Estimated dispersion parameter (phi):", phi, "\n")

x_seq <- create_x_sequence(x,100)

# print(head(x_seq))

y_pred <- predicted_y(x_seq, beta_vector)

# Plot the original data
plot(x[,2], y, main = "Inverse Gaussian GLM with Canonical Link", xlab = "x", ylab = "y")
lines(x_seq[,2], y_pred, col = "blue", lwd = 2)