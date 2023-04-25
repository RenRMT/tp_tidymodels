# pkgs = c("readr", "tidymodels", "vip")
# install.packages(pkgs)

library(readr)
library(tidymodels)
library(vip)

set.seed(68)


# variables ---------------------------------------------------------------
hotels_url <- "https://tidymodels.org/start/case-study/hotels.csv"

holidays <- c(
  "AllSouls", "AshWednesday", "ChristmasEve", "Easter",
  "ChristmasDay", "GoodFriday", "NewYearsDay", "PalmSunday"
)
cores <- parallel::detectCores()
# data --------------------------------------------------------------------
hotels <- readr::read_csv(hotels_url) |>
  dplyr::mutate(
    dplyr::across(
      dplyr::where(is.character), as.factor
    )
  )

splits <- rsample::initial_split(hotels, strata = children)
splits_val <- rsample::validation_split(hotel_train, strata = children, prop = 0.80)

hotel_train <- rsample::training(splits)
hotel_test <- rsample::testing(splits)

# model -------------------------------------------------------------------
logreg_model <- parsnip::logistic_reg(
  penalty = tune::tune(),
  mixture = 1
) |>
  parsnip::set_engine("glmnet")

logreg_recipe <- recipes::recipe(children ~ ., data = hotel_train) |>
  recipes::step_date(arrival_date) |>
  recipes::step_holiday(arrival_date, holidays = holidays) |>
  recipes::step_rm(arrival_date) |>
  recipes::step_dummy(recipes::all_nominal_predictors()) |>
  recipes::step_zv(recipes::all_predictors()) |>
  recipes::step_normalize(recipes::all_predictors())

logreg_workflow <- workflows::workflow() |>
  workflows::add_model(logreg_model) |>
  workflows::add_recipe(logreg_recipe)

# tuning grid for penalty values in lasso
logreg_tuning_grid <- tibble::tibble(
  penalty = 10^seq(-4, -1, length.out = 30)
)

# training model ----------------------------------------------------------
logreg_res <- logreg_workflow |>
  tune::tune_grid(splits_val,
    grid = logreg_tuning_grid,
    control = tune::control_grid(
      save_pred = TRUE
    ),
    metrics = yardstick::metric_set(roc_auc)
  )

logreg_best <- logreg_res |>
  tune::collect_metrics() |>
  dplyr::arrange(penalty) |>
  dplyr::slice(12)

logreg_auc <- logreg_res |>
  tune::collect_predictions(parameters = logreg_best) |>
  yardstick::roc_curve(children, .pred_children) |>
  dplyr::mutate(model = "Logistic Regression")

# random forest -----------------------------------------------------------
rf_model <- parsnip::rand_forest(
  mtry = tune::tune(),
  min_n = tune::tune(),
  trees = 1000
) |>
  parsnip::set_engine("ranger", num.threads = cores) |>
  parsnip::set_mode("classification")

rf_recipe <- recipes::recipe(children ~ ., data = hotel_train) |>
  recipes::step_date(arrival_date) |>
  recipes::step_holiday(arrival_date) |>
  recipes::step_rm(arrival_date)

rf_workflow <- workflows::workflow() |>
  workflows::add_model(rf_model) |>
  workflows::add_recipe(rf_recipe)

rf_res <- rf_workflow |>
  tune::tune_grid(splits_val,
    grid = 25,
    control = tune::control_grid(save_pred = TRUE),
    metrics = yardstick::metric_set(roc_auc)
  )

rf_best <- tune::select_best(rf_res, metric = "roc_auc")

rf_auc <- tune::collect_predictions(rf_res, parameters = rf_best) |>
  yardstick::roc_curve(children, .pred_children) |>
  dplyr::mutate(model = "Random Forest")

dplyr::bind_rows(rf_auc, logreg_auc) |>
  ggplot2::ggplot(
    ggplot2::aes(x = 1 - specificity, y = sensitivity, col = model)
  ) +
  ggplot2::geom_path(linewidth = 1.5, alpha = 0.7) +
  ggplot2::geom_abline(lty = 3) +
  ggplot2::coord_equal() +
  ggplot2::scale_colour_viridis_d(option = "plasma", end = .6)

# last fit ----------------------------------------------------------------
last_rf_model <- parsnip::rand_forest(mtry = 8, min_n = 7, trees = 1000) |>
  parsnip::set_engine("ranger", num.threads = cores, importance = "impurity") |>
  parsnip::set_mode("classification")

last_rf_workflow <- rf_workflow |>
  workflows::update_model(last_rf_model)

last_rf_fit <- last_rf_workflow |>
  last_fit(splits)

tune::collect_metrics(last_rf_fit)

workflowsets::extract_fit_parsnip(last_rf_fit) |>
  vip::vip(num_features = 20)

tune::collect_predictions(last_rf_fit) |>
  yardstick::roc_curve(children, .pred_children) |>
  ggplot2::autoplot()
