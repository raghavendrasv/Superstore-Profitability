setwd("C://Users/raghavendrasv/Documents/MBA/ISTM660 - Applied Analytics - R/Project")

library(tidyverse)
library(lubridate)
library(neuralnet)
library(caret)

# Read the CSV file
superstore <- read.csv("SuperstoreFlatdisc.csv", stringsAsFactors = FALSE)
head(superstore)

# Convert date and create time-based features
superstore <- superstore %>%
  mutate(
    OrderDate = as.Date(OrderDate, format = "%m/%d/%Y"),
    Year = year(OrderDate),
    Month = month(OrderDate),
    Quarter = quarter(OrderDate)
  )


# Aggregate data to monthly level
monthly_data <- superstore %>%
  group_by(Year, Month) %>%
  summarise(
    Profit = sum(Profit),
    Sales = sum(Sales),
    Quantity = sum(Quantity),
    Discount = mean(Discount),
    DaystoShipActual = mean(DaystoShipActual),
    DaystoShipScheduled = mean(DaystoShipScheduled),
    .groups = 'drop'
  ) %>%
  mutate(Date = as.Date(paste(Year, Month, "01", sep = "-")))

# Create lag features
for(i in 1:3) {
  monthly_data[[paste0("Profit_Lag", i)]] <- lag(monthly_data$Profit, i)
  monthly_data[[paste0("Sales_Lag", i)]] <- lag(monthly_data$Sales, i)
}

# Remove rows with NA values
monthly_data <- na.omit(monthly_data)

head(monthly_data)

# One-hot encode categorical variables
#dummy <- dummyVars(" ~ Category + SubCategory + Segment + ShipMode + ShipStatus + Region", data = monthly_data)
#categorical_dummy <- predict(dummy, newdata = monthly_data)
#monthly_data <- cbind(monthly_data, categorical_dummy)


# Normalize numerical variables
normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

numerical_vars <- c("Profit", "Sales", "Quantity", "Discount", "DaystoShipActual" ,  "DaystoShipScheduled" ,
                    "Profit_Lag1", "Profit_Lag2", "Profit_Lag3",
                    "Sales_Lag1", "Sales_Lag2", "Sales_Lag3")

monthly_data[numerical_vars] <- lapply(monthly_data[numerical_vars], normalize)

# Split data into training and testing sets
set.seed(123)
train_indices <- sample(1:nrow(monthly_data), 0.8 * nrow(monthly_data))
train_data <- monthly_data[train_indices, ]
test_data <- monthly_data[-train_indices, ]

# Define the formula for the neural network
#formula_vars <- c(numerical_vars[-1], colnames(categorical_dummy))
formula <- as.formula(paste("Profit ~", paste(numerical_vars[-1], collapse = " + ")))

print(formula)

# Train the neural network
nn_model <- neuralnet(
  formula,
  data = train_data,
  hidden = c(9, 6, 3), #(6, 3) works with original dataset, (8,5,4) works best with strat_Disocunt dataset
  linear.output = TRUE,
  threshold = 0.01,
  stepmax = 1e5
)

# Plot the neural network
plot(nn_model)

# Make predictions on test data
predictions <- compute(nn_model, test_data[, numerical_vars[-1]])

# Denormalize predictions and actual values
denormalize <- function(x, orig) {
  x * (max(orig) - min(orig)) + min(orig)
}

predictions_denormalized <- denormalize(predictions$net.result, superstore$Profit)
actual_denormalized <- denormalize(test_data$Profit, superstore$Profit)

head(actual_denormalized)

# Calculate RMSE
rmse <- sqrt(mean((predictions_denormalized - actual_denormalized)^2))
print(paste("RMSE:", rmse))

# Calculate R-squared
ss_res <- sum((test_data$Profit - predictions$net.result)^2)
ss_tot <- sum((test_data$Profit - mean(test_data$Profit))^2)
r_squared <- 1 - (ss_res / ss_tot)
print(paste("R-squared:", r_squared))

# Plot actual vs predicted values
plot(actual_denormalized, predictions_denormalized, 
     main = "Actual vs Predicted Profit", 
     xlab = "Actual Profit", ylab = "Predicted Profit")
abline(a = 0, b = 1, col = "red")

# Forecast for the next 12 months
last_date <- max(monthly_data$Date)
future_dates <- seq(last_date %m+% months(1), by = "month", length.out = 12)

# Calculate weighted averages for forecasting
weighted_avg <- function(x, weights) {
  sum(x * weights) / sum(weights)
}

forecast_data <- data.frame(
  Date = future_dates,
  Year = year(future_dates),
  Month = month(future_dates)
)

for (var in c("Sales", "Quantity", "Discount", "DaystoShipActual" ,  "DaystoShipScheduled" ,
              "Profit_Lag1", "Profit_Lag2", "Profit_Lag3",
              "Sales_Lag1", "Sales_Lag2", "Sales_Lag3")) {
  for (m in 1:12) {
    historical_data <- monthly_data[monthly_data$Month == m, var]
    weights <- 1:length(historical_data)
    forecast_data[m, var] <- weighted_avg(historical_data, weights)
  }
}

# Add lag features for forecasting
#for (i in 1:3) {
#  forecast_data[[paste0("Profit_Lag", i)]] <- tail(monthly_data$Profit, 3)[3-i+1]
#  forecast_data[[paste0("Sales_Lag", i)]] <- tail(monthly_data$Sales, 3)[3-i+1]
#}

print(forecast_data)

# Add categorical variables (using mode from last year)
#last_year_data <- tail(monthly_data, 12)
#for (var in c("Category", "Segment", "ShipMode", "Region")) {
#  forecast_data[[var]] <- names(which.max(table(last_year_data[[var]])))
#}

# One-hot encode categorical variables for forecast data
#forecast_categorical <- predict(dummy, newdata = forecast_data)
#forecast_data <- cbind(forecast_data, forecast_categorical)

# Normalize forecast data
forecast_data[numerical_vars[-1]] <- lapply(forecast_data[numerical_vars[-1]], normalize)
head(forecast_data)
# Make predictions for future dates
future_predictions <- compute(nn_model, forecast_data)
head(future_predictions)
# Denormalize future predictions
future_predictions_denormalized <- denormalize(future_predictions$net.result, superstore$Profit)
print(future_predictions_denormalized)

# Plot forecasted profits
plot(monthly_data$Date, denormalize(monthly_data$Profit, superstore$Profit), type = "l", 
     xlim = c(min(monthly_data$Date), max(future_dates)),
     ylim = c(min(superstore$Profit), max(c(superstore$Profit, future_predictions_denormalized))),
     main = "Profit Forecast", xlab = "Date", ylab = "Profit")
lines(future_dates, future_predictions_denormalized, col = "red")
legend("topleft", legend = c("Historical", "Forecast"), col = c("black", "red"), lty = 1)

# Print forecasted profits
forecast_results <- data.frame(Date = future_dates, Forecasted_Profit = future_predictions_denormalized)
print(forecast_results)
