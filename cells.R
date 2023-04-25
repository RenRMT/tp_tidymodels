# pkgs = c("modeldata", "ranger", tidymodels")
# install.packages(pkgs)

library(modeldata)
library(tidymodels)

# data --------------------------------------------------------------------
set.seed(68)
data(cells, package = "modeldata")

cells_train <- cells[cells$case == "Train", ]
cells_test <- cells[cells$case == "Test", ]


# modeling ----------------------------------------------------------------
rf_model <- parsnip::rand_forest(trees = 1000) |>
  parsnip::set_engine("ranger") |>
  parsnip::set_mode("classification")

rf_fit <- rf_model |>
  fit(class ~ ., data = cells_train)

rf_pred <- predict(rf_fit, cells_test) |>
  dplyr::bind_cols(predict(rf_fit, cells_test, type = "prob")) |>
  dplyr::bind_cols(dplyr::select(cells_test, class))

rf_pred |>
  yardstick::roc_auc(truth = class, .pred_PS)

# resampling  -------------------------------------------------------------
folds <- rsample::vfold_cv(cells_train, v = 10)

rf_workflow <- workflows::workflow() |>
  workflows::add_model(rf_model) |>
  workflows::add_formula(class ~ .)

rf_fit_cv <- rf_workflow |>
  tune::fit_resamples(folds)

tune::collect_metrics(rf_fit_cv)
