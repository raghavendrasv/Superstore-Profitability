setwd("C://Users/raghavendrasv/Documents/MBA/ISTM660 - Applied Analytics - R/Project")
# Load required libraries
library(tidyverse)
library(lubridate)
library(forecast)
library(neuralnet)
library(caret)

# Step 1: Load and preprocess the data
superstore <- read.csv("SuperstoreFlatdisc.csv", stringsAsFactors = FALSE)
head(superstore)

# Convert date columns to Date type
superstore$OrderDate <- as.Date(superstore$OrderDate, format = "%m/%d/%Y")
superstore$ShipDate <- as.Date(superstore$ShipDate, format = "%m/%d/%Y")

# Create year and month columns
superstore$Year <- year(superstore$OrderDate)
superstore$Month <- month(superstore$OrderDate)

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
    #    Category_Furniture = sum(Category == "Furniture") / n(),
    #    Category_OfficeSupplies = sum(Category == "Office Supplies") / n(),
    #    Category_Technology = sum(Category == "Technology") / n(),
    #    Segment_Consumer = sum(Segment == "Consumer") / n(),
    #    Segment_Corporate = sum(Segment == "Corporate") / n(),
    #    Segment_HomeOffice = sum(Segment == "Home Office") / n(),
    #    ShipMode_FirstClass = sum(ShipMode == "First Class") / n(),
    #    ShipMode_SameDay = sum(ShipMode == "Same Day") / n(),
    #    ShipMode_SecondClass = sum(ShipMode == "Second Class") / n(),
    #    ShipMode_StandardClass = sum(ShipMode == "Standard Class") / n(),
    #    ShipStatus_ShippedOnTime = sum(ShipStatus == "ShippedOnTime") / n(),
    #    ShipStatus_ShippedLate = sum(ShipStatus == "ShippedLate") / n(),
    #    ShipStatus_ShippedEarly = sum(ShipStatus == "ShippedEarly") / n()
  ) %>%
  mutate(Date = as.Date(paste(Year, Month, "01", sep = "-")))

str(monthly_data)

# Step 2: Prepare time series data for SARIMA
ts_profit <- ts(monthly_data$Profit, frequency = 12, start = c(min(monthly_data$Year), 1))

# Step 3: Fit SARIMA model
sarima_model <- ets(ts_profit, model = "AAA")
print(summary(sarima_model))

# Step 4: Generate SARIMA forecasts
sarima_forecast <- forecast(sarima_model, h = 12)
head(sarima_model)

# Step 5: Prepare data for neural network
# Create lag features
for (i in 1:6) {
  monthly_data[[paste0("Profit_Lag", i)]] <- lag(monthly_data$Profit, i)
  monthly_data[[paste0("Sales_Lag", i)]] <- lag(monthly_data$Sales, i)
}

# Remove rows with NA values
monthly_data <- na.omit(monthly_data)
head(monthly_data)

# Export the data frame to a CSV file
write.csv(monthly_data, "monthly_data_DS.csv", row.names = FALSE)

# Normalize the data
normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

monthly_data_normalized <- as.data.frame(lapply(monthly_data[, -which(names(monthly_data) %in% c("Year", "Month", "Date"))], normalize))

# Split data into training and testing sets
train_size <- nrow(monthly_data_normalized) - 12
train_data <- monthly_data_normalized[1:train_size, ]
test_data <- monthly_data_normalized[(train_size + 1):nrow(monthly_data_normalized), ]

any(is.na(train_data))
any(is.infinite(as.matrix(train_data)))

str(train_data)

# Step 6: Train the neural network
nn_formula <- as.formula(paste("Profit ~", paste(setdiff(names(train_data), "Profit"), collapse = " + ")))
head(nn_formula)

nn_model <- neuralnet(
  nn_formula,
  data = train_data,
  hidden = c(9, 7, 6, 3), #(9,7,6,5(3 for DS)) works for original data
  linear.output = TRUE,
  threshold = 0.01,
  stepmax = 1e5
)
plot(nn_model)

# Step 7: Make predictions using the neural network
nn_predictions <- compute(nn_model, test_data[, setdiff(names(test_data), "Profit")])$net.result

# Convert nn_predictions to a time series
nn_predictions_denormalized <- nn_predictions * (max(monthly_data$Profit) - min(monthly_data$Profit)) + min(monthly_data$Profit)
nn_predictions_ts <- ts(as.vector(nn_predictions_denormalized), frequency = 12)

print(nn_predictions_ts)

# Step 8: Combine SARIMA and neural network predictions
hybrid_forecast <- (sarima_forecast$mean + nn_predictions_denormalized) / 2

# Step 9: Evaluate the models
actual_values <- monthly_data$Profit[(nrow(monthly_data) - 11):nrow(monthly_data)]

Holt_Winters_accuracy <- accuracy(sarima_forecast, actual_values)
nn_accuracy <- accuracy(nn_predictions_ts, actual_values)
hybrid_accuracy <- accuracy(hybrid_forecast, actual_values)

print("Holt-Winters Accuracy:")
print(Holt_Winters_accuracy)
print("Neural Network Accuracy:")
print(nn_accuracy)
print("Hybrid Model Accuracy:")
print(hybrid_accuracy)

# Step 10: Plot the results
plot(monthly_data$Date, monthly_data$Profit, type = "l", col = "black", 
     main = "Profit Forecast: SARIMA vs Neural Network vs Hybrid",
     xlab = "Date", ylab = "Profit")
lines(monthly_data$Date[(nrow(monthly_data) - 11):nrow(monthly_data)], sarima_forecast$mean, col = "blue")
lines(monthly_data$Date[(nrow(monthly_data) - 11):nrow(monthly_data)], 
      nn_predictions * (max(monthly_data$Profit) - min(monthly_data$Profit)) + min(monthly_data$Profit), col = "red")
lines(monthly_data$Date[(nrow(monthly_data) - 11):nrow(monthly_data)], hybrid_forecast, col = "green")
legend("topleft", legend = c("Actual", "SARIMA", "Neural Network", "Hybrid"), 
       col = c("black", "blue", "red", "green"), lty = 1)

# Create a data frame with the plot data
plot_data <- data.frame(
  Date = monthly_data$Date[(nrow(monthly_data) - 11):nrow(monthly_data)],
  Actual = monthly_data$Profit[(nrow(monthly_data) - 11):nrow(monthly_data)],
  SARIMA = sarima_forecast$mean,
  NeuralNetwork = nn_predictions * (max(monthly_data$Profit) - min(monthly_data$Profit)) + min(monthly_data$Profit),
  Hybrid = hybrid_forecast
)

# Export the data frame to a CSV file
write.csv(plot_data, "profit_forecast_comparison_SD.csv", row.names = FALSE)

# Step 11: Generate future forecasts
future_dates <- seq(max(monthly_data$Date) + months(1), by = "month", length.out = 12)

# Calculate weighted averages for forecasting
weighted_avg <- function(x, weights) {
  sum(x * weights) / sum(weights)
}

#future_data <- data.frame(
#  Date = future_dates,
#  Year = year(future_dates),
#  Month = month(future_dates)
#)

for (var in c("Profit", "Sales", "Quantity", "Discount", "DaystoShipActual" ,  "DaystoShipScheduled")) 
{
  for (m in 1:12) {
    historical_data <- monthly_data[monthly_data$Month == m, var]
    weights <- 1:length(historical_data)
    future_data[m, var] <- weighted_avg(historical_data, weights)
  }
}

head(future_data)

# Prepare future data for neural network
#future_data <- data.frame(
#  Sales = rep(mean(monthly_data$Sales), 12),
#  Quantity = rep(mean(monthly_data$Quantity), 12),
#  Discount = rep(mean(monthly_data$Discount), 12),
#  ShippingCost = rep(mean(monthly_data$ShippingCost), 12),
#  OrderCount = rep(mean(monthly_data$OrderCount), 12),
#  CustomerCount = rep(mean(monthly_data$CustomerCount), 12)
#)


# Add lag features
for (i in 1:6) {
  future_data[[paste0("Profit_Lag", i)]] <- tail(monthly_data$Profit, 6)[6:1]
  future_data[[paste0("Sales_Lag", i)]] <- tail(monthly_data$Sales, 6)[6:1]
}

head(future_data)

# Normalize future data
future_data_normalized <- as.data.frame(lapply(future_data[, -which(names(future_data) %in% c("Year", "Month", "Date"))], normalize))

# Generate future forecasts
sarima_future <- forecast(sarima_model, h = 12)
nn_future <- compute(nn_model, future_data_normalized[, setdiff(names(test_data), "Profit")])$net.result
nn_future_denormalized <- nn_future * (max(monthly_data$Profit) - min(monthly_data$Profit)) + min(monthly_data$Profit)
hybrid_future <- (sarima_future$mean + nn_future_denormalized) / 2

# Plot future forecasts
plot(c(monthly_data$Date, future_dates), c(monthly_data$Profit, hybrid_future), type = "l", col = "black",
     main = "Future Profit Forecast: Hybrid Model", xlab = "Date", ylab = "Profit")
lines(future_dates, sarima_future$mean, col = "blue")
lines(future_dates, nn_future_denormalized, col = "red")
lines(future_dates, hybrid_future, col = "green")
legend("topleft", legend = c("Historical", "SARIMA", "Neural Network", "Hybrid"),
       col = c("black", "blue", "red", "green"), lty = 1)

# Print future forecasts
hybrid_forecast_df <- data.frame(
  Date = future_dates,
  SARIMA = sarima_future$mean,
  NeuralNetwork = nn_future_denormalized,
  Hybrid = hybrid_future
)
print(hybrid_forecast_df)

# Export the data frame to a CSV file
write.csv(hybrid_forecast_df, "hybrid_forecast_SD_df.csv", row.names = FALSE)