# pkgs = c("nycflights13", "skimr", "tidymodels")
# install.packages(pkgs)

library(nycflights13)
library(skimr)
library(tidymodels)

# data wrangling ----------------------------------------------------------
set.seed(68)
flight_data <- flights |>
  dplyr::mutate(
    arr_delay = dplyr::if_else(arr_delay >= 30, "late", "on_time"),
    arr_delay = factor(arr_delay),
    date = lubridate::as_date(time_hour)
  ) |>
  dplyr::inner_join(weather, by = c("origin", "time_hour")) |>
  dplyr::select(
    dep_time, flight, origin, dest, air_time, distance,
    carrier, date, arr_delay, time_hour
  ) |>
  na.omit() |>
  dplyr::mutate_if(is.character, as.factor)

dplyr::glimpse(flight_data)

data_split <- rsample::initial_split(flight_data, prop = 3 / 4)
flights_train <- rsample::training(data_split)
flights_test <- rsample::testing(data_split)

# recipes -----------------------------------------------------------------
# create recipe with flight and time_hour in ID role
flights_rec <- recipes::recipe(arr_delay ~ ., data = flights_train) |>
  recipes::update_role(flight, time_hour, new_role = "ID")


# feature engineering using date predictor
flights_rec <- flights_rec |>
  recipes::step_date(date, features = c("dow", "month")) |>
  recipes::step_holiday(date,
    holidays = timeDate::listHolidays("US"),
    keep_original_cols = FALSE
  )

# create dummy variables from factors
flights_rec <- flights_rec |>
  recipes::step_dummy(recipes::all_nominal_predictors())

# remove infrequent factor levels that might not appear in
# training data
flights_rec <- flights_rec |>
  recipes::step_zv(all_predictors())

# fit model with recipe ---------------------------------------------------
lr_model <- parsnip::logistic_reg() |>
  parsnip::set_engine("glm")

flights_workflow <- workflows::workflow() |>
  workflows::add_model(lr_model) |>
  workflows::add_recipe(flights_rec)

flights_fit <- flights_workflow |>
  fit(data = flights_train)

flights_pred <- predict(flights_fit, flights_test, type = "prob")

flights_aug <- parsnip::augment(flights_fit, flights_test)

flights_aug |>
  yardstick::roc_curve(truth = arr_delay, .pred_late) |>
  autoplot()

flights_aug |>
  yardstick::roc_auc(truth = arr_delay, .pred_late)
