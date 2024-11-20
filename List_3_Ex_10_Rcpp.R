library(Rcpp)
library(RcppEigen)
library(stats)
sourceCpp("List_3_Ex_10_Rcpp.cpp")
par(mfrow=c(1,1))

x_values <- c(22, 27, 34, 43, 51, 42, 49, 67, 81, 90)

x_values<- matrix(x_values, ncol = 1, byrow = FALSE)

rows <- 10
cols <- 1

y <- c(1,1,1,1,1,0,0,0,0,0)

number_iterations <- 10
x_original <- cbind(1, x_values)
x_test <- matrix(1, nrow = rows, ncol = cols)
y <- as.numeric(y)
start_time <- Sys.time()
beta_vector <- beta_estimation(y,x_original,number_iterations)
cat("\n")
cat("Estimated Beta parameters:", beta_vector, "\n")
cat("\n")
end_time <- Sys.time()
time_taken_beta_my_code <- end_time - start_time


start_time <- Sys.time()
Q_LR <- Q_LR(y,x_original,x_test,number_iterations)
Q_LR_p_value <- pchisq(Q_LR, df = 1, lower.tail = FALSE)
end_time <- Sys.time()
time_taken_LR_my_code <- end_time - start_time


Q_SR <- Q_SR(y,x_test,number_iterations)
Q_SR_p_value <- pchisq(Q_SR, df = 1, lower.tail = FALSE)
Q_W <- Q_W(y,x_original,ncol(x_original),number_iterations)
Q_W_p_value <- pchisq(Q_W, df = 1, lower.tail = FALSE)
Q_G <- Q_G(y,x_original,x_test,number_iterations)
Q_G_p_value <- pchisq(Q_G, df = 1, lower.tail = FALSE)

cat("Estimated Statistic and p-value for Q_LR:", Q_LR, "and", Q_LR_p_value, "\n")
cat("Estimated Statistic and p-value for Q_SR:", Q_SR, "and", Q_SR_p_value, "\n")
cat("Estimated Statistic and p-value for Q_W:", Q_W, "and", Q_W_p_value, "\n")
cat("Estimated Statistic and p-value for Q_G:", Q_G, "and", Q_G_p_value, "\n")
cat("\n")
start_time <- Sys.time()
# Full model (with beta1)
model_full <- glm(y ~ x_original, family = binomial(link = "logit"),control = glm.control(maxit =number_iterations))
# Print only the estimated beta coefficients
beta_estimates <- summary(model_full)$coefficients[, 1]  # Extract the first column of coefficients
print(beta_estimates)
cat("\n")
end_time <- Sys.time()
time_taken_beta_glm_package <- end_time - start_time

start_time <- Sys.time()
# Null model (without beta1, i.e., only intercept)
model_null <- glm(y ~ x_test, family = binomial(link = "logit"),control = glm.control(maxit =number_iterations))

# Likelihood Ratio Test
lrt_statistic <- 2 * (logLik(model_full) - logLik(model_null))
lrt_p_value <- pchisq(lrt_statistic, df = 1, lower.tail = FALSE)
cat("Likelihood Ratio Test Statistic:", lrt_statistic, "\n")
cat("Likelihood Ratio Test p-value:", lrt_p_value, "\n")
end_time <- Sys.time()
time_taken_LR_glm_package <- end_time - start_time


# Wald Test
beta_1 <- coef(summary(model_full))["x_original2", "Estimate"]
se_beta_1 <- coef(summary(model_full))["x_original2", "Std. Error"]
wald_statistic <- (beta_1 / se_beta_1) ^ 2
wald_p_value <- pchisq(wald_statistic, df = 1, lower.tail = FALSE)
cat("Wald Test Statistic:", wald_statistic, "\n")
cat("Wald Test p-value:", wald_p_value, "\n")

time_taken_beta_glm_package <- as.numeric(time_taken_beta_glm_package, units = "secs")
time_taken_beta_my_code <- as.numeric(time_taken_beta_my_code, units = "secs")
time_taken_LR_glm_package <- as.numeric(time_taken_LR_glm_package, units = "secs")
time_taken_LR_my_code <- as.numeric(time_taken_LR_my_code, units = "secs")

cat("How faster is my code compared to GLM package for Beta estimation:", time_taken_beta_glm_package/time_taken_beta_my_code, "\n")
cat("How faster is my code compared to GLM package for performing the Likelihood Ratio test:", time_taken_LR_glm_package/time_taken_LR_my_code, "\n")
