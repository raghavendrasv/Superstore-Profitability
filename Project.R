setwd("C://Users/raghavendrasv/Documents/MBA/ISTM660 - Applied Analytics - R/Project")
superstore<-read.csv("Superstore.csv")
#View(superstore)

# Load required libraries
library(dplyr)
library(lubridate)
library(xts)
library(TTR) 
library(forecast)
library(ggplot2)
library(pracma)
library(tidyverse)
library(lubridate)
library(xts)
library(forecast)
library(tree)
library(cluster)
library(Clustering)
library(ClusterR)
library(neuralnet)
library(rpart)
library(rpart.plot)


ss_profit<-lm(formula = Profit ~ Category + SubCategory + Discount + Quantity,data = superstore)

summary(ss_profit)

ss_sales<-lm(formula = Sales ~ Category + SubCategory + Discount + Quantity,data = superstore)

summary(ss_sales)

# Convert "Order Date" to a proper date format
superstore$OrderDate <- mdy(superstore$OrderDate)

# Extract month from the date
superstore$Month <- month(superstore$OrderDate)

# Create Spring column
superstore$Spring <- ifelse(superstore$Month >= 3 & superstore$Month <= 5, 1, 0)
superstore$Summer <- ifelse(superstore$Month >= 6 & superstore$Month <= 8, 1, 0)
superstore$Fall <- ifelse(superstore$Month >= 9 & superstore$Month <= 11, 1, 0)
superstore$Winter <- ifelse(superstore$Month >= 12 & superstore$Month <= 2, 1, 0)

superstore$ProfitBinary <- ifelse(superstore$Profit > 0, 1, 0)


# Create the model
model <- glm(ProfitBinary ~ Discount + as.factor(superstore$Category) + as.factor(superstore$SubCategory), 
             data = superstore, 
             family = binomial(link = "logit"))



# Summary of the model
summary(model)

# Odds ratios
exp(coef(model))

# Predict probabilities
probabilities <- predict(model, type = "response")

# Convert probabilities to binary predictions
predictions <- ifelse(probabilities > 0.5, 1, 0)

# Create confusion matrix
table(Actual = superstore$ProfitBinary, Predicted = predictions)

mean(predictions == superstore$ProfitBinary)


# Convert Order Date to date format and ensure Profit and Sales are numeric
superstore$Order_Date <- parse_date_time(superstore$OrderDate, orders = c("mdy", "dmy", "ymd"))
superstore$Profit <- as.numeric(superstore$Profit)
superstore$Sales <- as.numeric(superstore$Sales)

# Aggregate data to monthly level
monthly_data <- superstore %>%
  group_by(Year_Month = floor_date(Order_Date, "month")) %>%
  summarise(
    Total_Profit = sum(Profit),
    Total_Sales = sum(Sales)
  )

# Create xts objects
profit_xts <- xts(monthly_data$Total_Profit, order.by = monthly_data$Year_Month)
sales_xts <- xts(monthly_data$Total_Sales, order.by = monthly_data$Year_Month)

# Convert xts to ts objects for forecasting
profit_ts <- as.ts(profit_xts)
sales_ts <- as.ts(sales_xts)

# Function to fit SARIMA model and forecast
fit_and_forecast <- function(ts_data, title) {
  # Fit the SARIMA model
  fit <- auto.arima(ts_data, seasonal = TRUE)
  
  # Forecast next 12 months
  forecast <- forecast(fit, h = 12)
  
  # Plot the forecast
  plot(forecast, main = paste("Forecast of", title), xlab = "Time", ylab = title)
  
  return(forecast)
}


# Fit models and generate forecasts
profit_forecast <- fit_and_forecast(profit_ts, "Monthly Profit")
sales_forecast <- fit_and_forecast(sales_ts, "Monthly Sales")

# Print summary of forecasts
# summary(profit_forecast)
# summary(sales_forecast)

# Combine forecasts into a data frame
forecast_df <- data.frame(
  Date = seq(max(monthly_data$Year_Month) + months(1), by = "month", length.out = 12),
  Profit_Forecast = profit_forecast$mean,
  Profit_Lower = profit_forecast$lower[, 2],
  Profit_Upper = profit_forecast$upper[, 2],
  Sales_Forecast = sales_forecast$mean,
  Sales_Lower = sales_forecast$lower[, 2],
  Sales_Upper = sales_forecast$upper[, 2]
)

# Print the forecast results
print(forecast_df)

# Optionally, save the forecasts to a CSV file
write.csv(forecast_df, "forecast_results.csv", row.names = FALSE)

# Fit models and generate forecasts
profit_forecast <- fit_and_forecast(profit_ts, "Monthly Profit")
sales_forecast <- fit_and_forecast(sales_ts, "Monthly Sales")

# Print summary of forecasts
summary(profit_forecast)
summary(sales_forecast)

# Combine forecasts into a data frame
forecast_df <- data.frame(
  Date = seq(max(monthly_data$Year_Month) + months(1), by = "month", length.out = 12),
  Profit_Forecast = profit_forecast$mean,
  Profit_Lower = profit_forecast$lower[, 2],
  Profit_Upper = profit_forecast$upper[, 2],
  Sales_Forecast = sales_forecast$mean,
  Sales_Lower = sales_forecast$lower[, 2],
  Sales_Upper = sales_forecast$upper[, 2]
)

# Print the forecast results
print(forecast_df)

# Optionally, save the forecasts to a CSV file
write.csv(forecast_df, "forecast_results.csv", row.names = FALSE)

# Create time series objects
ts_sales <- ts(monthly_data$Total_Sales, frequency = 12, 
               start = c(year(min(monthly_data$Year_Month)), 
                         month(min(monthly_data$Year_Month))))
ts_profit <- ts(monthly_data$Total_Profit, frequency = 12, 
                start = c(year(min(monthly_data$Year_Month)), 
                          month(min(monthly_data$Year_Month))))

# Function to fit ETS models and choose the best one
fit_best_ets <- function(ts_data, title) {
  # Fit ETS models
  model_ANN <- ets(ts_data, model = "ANN")
  model_ANA <- ets(ts_data, model = "ANA")
  model_AAA <- ets(ts_data, model = "AAA")
  
  # Compare models
  models <- list(ANN = model_ANN, ANA = model_ANA, AAA = model_AAA)
  aics <- sapply(models, AIC)
  best_model <- models[[which.min(aics)]]
  
  # Print summary of the best model
  cat("Best model for", title, ":", names(which.min(aics)), "\n")
  print(summary(best_model))
  
  # Forecast using the best model
  forecast <- forecast(best_model, h = 12)
  
  # Plot the forecast
  plot(forecast, main = paste("Forecast of", title), xlab = "Time", ylab = title)
  
  return(list(model = best_model, forecast = forecast))
}

# Fit models and generate forecasts for Sales and Profit
sales_result <- fit_best_ets(ts_sales, "Monthly Sales")
profit_result <- fit_best_ets(ts_profit, "Monthly Profit")

# Combine forecasts into a data frame
forecast_sma_df <- data.frame(
  Date = seq(max(monthly_data$Year_Month) + months(1), by = "month", length.out = 12),
  Sales_Forecast = sales_result$forecast$mean,
  Sales_Lower = sales_result$forecast$lower[, 2],
  Sales_Upper = sales_result$forecast$upper[, 2],
  Profit_Forecast = profit_result$forecast$mean,
  Profit_Lower = profit_result$forecast$lower[, 2],
  Profit_Upper = profit_result$forecast$upper[, 2]
)

# Print the forecast results
print(forecast_sma_df)

# Optionally, save the forecasts to a CSV file
write.csv(forecast_sma_df, "forecast_sma_results.csv", row.names = FALSE)

# Calculate accuracy measures
sales_accuracy <- accuracy(sales_result$forecast)
profit_accuracy <- accuracy(profit_result$forecast)

cat("\nAccuracy measures for Sales forecast:\n")
print(sales_accuracy)

cat("\nAccuracy measures for Profit forecast:\n")
print(profit_accuracy)

profit_ts

# ----------------------- Clulsters and Trees ----------------------------------

set.seed(500)
train = sample(1:nrow(superstore), nrow(superstore)/2)
tree.superstore = tree(Profit ~ Discount+Sales+Quantity+as.factor(superstore$Segment)+ DaystoShipActual + DaystoShipScheduled+ Spring+ Summer+Fall+Winter+as.factor(superstore$Category) + as.factor(superstore$SubCategory) + 
                         as.factor(superstore$Region) + as.factor(superstore$ShipMode) + as.factor(superstore$ShipStatus) , superstore, subset = train)
cv.superstore = cv.tree(tree.superstore)
plot(cv.superstore$size, cv.superstore$dev, type='b')
cv.superstore

tree2.superstore = rpart(Profit ~ Discount+Sales+Quantity+as.factor(superstore$Segment)+ DaystoShipActual + DaystoShipScheduled+ Spring+ Summer+Fall+Winter+as.factor(superstore$Category) + as.factor(superstore$SubCategory) + 
                               as.factor(superstore$Region) + as.factor(superstore$ShipMode) + as.factor(superstore$ShipStatus) , data=superstore, control = rpart.control(maxdepth = 7))
prp(tree2.superstore, type = 1, extra = 1, split.font = 1, varlen = -10)

dim(superstore)
superstore.cp = rpart(Profit ~ Discount+Sales+Quantity+as.factor(superstore$Segment)+ DaystoShipActual + DaystoShipScheduled+ Spring+ Summer+Fall+Winter+as.factor(superstore$Category) + as.factor(superstore$SubCategory) + 
                            as.factor(superstore$Region) + as.factor(superstore$ShipMode) + as.factor(superstore$ShipStatus) , data=superstore, cp = 0.001, minsplit = 2, xval = 1000)
prp(superstore.cp, type = 1, extra = 1, split.font = 1,varlen = -10)
printcp(superstore.cp)

#------------------------ Random Forests ---------------------------------------
#profit.rf <- randomforest(Profit + Discount + Sales + Quantity + as.factor(superstore$Segment) + DaystoShipActual + DaystoShipScheduled+ Spring+ Summer+Fall+Winter+as.factor(superstore$Category) + as.factor(superstore$SubCategory) + 
#                            as.factor(superstore$Region) + as.factor(superstore$ShipMode) + as.factor(superstore$ShipStatus) , subset = train, mtry = 6, ntree = 1000, importance = true)

#------------------------ Neural Netwrorks -------------------------------------

# Load required libraries
library(neuralnet)
library(tidyverse)
library(lubridate)

# Read the CSV file
superstore <- read.csv("Superstore.csv", stringsAsFactors = FALSE)
summary(superstore)

# Data preprocessing
superstore <- superstore %>%
  mutate(
    OrderDate = mdy(OrderDate),
#    Profit = as.numeric(gsub("[$,]", "", Profit)),
#    Discount = as.numeric(gsub("%", "", Discount)) / 100,
    Month = month(OrderDate),
    Spring = as.integer(Month %in% c(3, 4, 5)),
    Summer = as.integer(Month %in% c(6, 7, 8)),
    Fall = as.integer(Month %in% c(9, 10, 11)),
    Winter = as.integer(Month %in% c(12, 1, 2))
  ) %>%
  select(Profit, Category, Region, Segment, ShipMode, ShipStatus, SubCategory, 
         Discount, Sales, Quantity, Spring, Summer, Fall, Winter)

# Convert categorical variables to factors
superstore$Category <- as.factor(superstore$Category)
superstore$Region <- as.factor(superstore$Region)
superstore$Segment <- as.factor(superstore$Segment)
superstore$ShipMode <- as.factor(superstore$ShipMode)
superstore$ShipStatus <- as.factor(superstore$ShipStatus)
superstore$SubCategory <- as.factor(superstore$SubCategory)

normalize <- function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}

superstore_normalized <- superstore %>%
  mutate(across(where(is.numeric), normalize))

# Check the structure of the normalized data
str(superstore_normalized)

# If you need to create dummy variables for categorical columns
#superstore_dummy <- model.matrix(~ . , data = superstore_normalized)

# Convert back to data frame
superstore_dummy <- as.data.frame(superstore_dummy)

# Print the first few rows of the final dataset
head(superstore_dummy)


# Split data into training and testing sets
set.seed(123)
sample_size <- floor(0.8 * nrow(superstore_dummy))
train_indices <- sample(seq_len(nrow(superstore_dummy)), size = sample_size)
train_data <- superstore_dummy[train_indices, ]
test_data <- superstore_dummy[-train_indices, ]
head(train_data)
names(train_data)

# Define formula for neural network
formula <- Profit ~ CategoryOfficeSupplies + CategoryTechnology + RegionEast + RegionSouth + RegionWest +
  SegmentCorporate + SegmentHomeOffice + ShipModeSecondClass + ShipModeStandardClass +
  ShipStatusShippedLate + ShipStatusShippedOnTime +
  SubCategoryAppliances + SubCategoryArt +
  SubCategoryBinders + SubCategoryBookcases + SubCategoryChairs +
  SubCategoryCopiers + SubCategoryEnvelopes + SubCategoryFasteners +
  SubCategoryFurnishings + SubCategoryLabels + SubCategoryMachines +
  SubCategoryPaper + SubCategoryPhones + SubCategoryStorage +
  SubCategorySupplies + SubCategoryTables +
  Discount + Sales + Quantity + Spring + Summer + Fall + Winter

# Train the neural network
nn_model <- neuralnet(
  formula,
  data = train_data,
  hidden = c(5, 3),  # Three hidden layers with 9, 5, and 2 neurons
  linear.output = TRUE,
  threshold = 0.01,
  stepmax = 1e5
)

# Plot the neural network
plot(nn_model)
#--------------------------- Pretty Plot --------------------------------------
library(neuralnet)
library(NeuralNetTools)

# Assuming you have already trained your neural network model named 'nn_model'

# Generate a basic plot
plotnet(nn_model)

#------------------------------------------------------------------------------

# Make predictions on test data
predictions <- compute(nn_model, test_data)
# Print the forecast results
print(predictions)


# Calculate Mean Squared Error (MSE)
mse <- mean((predictions$net.result - test_data$Profit)^2)
print(paste("Mean Squared Error:", mse))

# Calculate R-squared
ss_res <- sum((test_data$Profit - predictions$net.result)^2)
ss_tot <- sum((test_data$Profit - mean(test_data$Profit))^2)
r_squared <- 1 - (ss_res / ss_tot)
print(paste("R-squared:", r_squared))

# Plot predicted vs actual values
plot(test_data$Profit, predictions$net.result, 
     main = "Predicted vs Actual Profit",
     xlab = "Actual Profit", ylab = "Predicted Profit")
abline(0, 1, col = "red")

print(summary(nn_model))

#cv_results <- train(Profit ~ ., data = train_data, method = "nnet", trControl = trainControl(method = "cv", number = 5))
#print(cv_results)

#---------------- Pretty neural network ----------------------------------------
library(neuralnet)
library(ggplot2)
library(NeuralNetTools)

# Assuming you have already trained your neural network model named 'nn_model'

# Create a basic plot
plot_net <- plotnet(nn_model)

# Customize the plot
gg_net <- plot_net +
  theme_minimal() +
  ggtitle("Neural Network Architecture") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        panel.grid = element_blank()) +
  scale_x_continuous(name = "Layer", breaks = NULL) +
  scale_y_continuous(name = "Neuron", breaks = NULL)

# Add color to nodes based on layer
gg_net$data$color <- factor(gg_net$data$layer)
gg_net <- gg_net +
  geom_point(aes(color = color), size = 10) +
  scale_color_brewer(palette = "Set1", guide = "none")

# Adjust line aesthetics
gg_net <- gg_net +
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend),
               arrow = arrow(length = unit(0.1, "cm"), type = "closed"),
               color = "gray50", alpha = 0.5)

# Add labels
gg_net <- gg_net +
  geom_text(aes(label = label), size = 3, fontface = "bold")

# Display the plot
print(gg_net)

# Save the plot (optional)
ggsave("neural_network_visualization.png", gg_net, width = 10, height = 8, dpi = 300)


