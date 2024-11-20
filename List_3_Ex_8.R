# Define the predictor and response variables
x <- c(2, 4, 6, 9, 10)
y <- c(2.3, 1.8, 1.5, 1.3, 1.2)

# Fit the GLM with inverse Gaussian family and canonical link function
model <- glm(y ~ x, family = inverse.gaussian(link = "1/mu^2"))

# Display the summary of the model
print(summary(model))


# Create a sequence of x values for plotting
x_new <- seq(min(x), max(x), length.out = 100)

# Predict the fitted values on the original scale
y_pred <- predict(model, newdata = data.frame(x = x_new), type = "response")

# Plot the original data
plot(x, y, main = "Inverse Gaussian GLM with Canonical Link", xlab = "x", ylab = "y")
lines(x_new, y_pred, col = "blue", lwd = 2)

# Extract and display the estimated dispersion parameter (phi)
phi_estimate <- summary(model)$dispersion
cat("Estimated dispersion parameter (phi):", phi_estimate, "\n")
