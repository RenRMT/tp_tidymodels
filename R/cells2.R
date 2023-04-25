# pkgs = c("modeldata", rpart.plot", "tidymodels", "vip")
# install.packages(pkgs)

library(rpart.plot)
library(tidymodels)
library(vip)

# data --------------------------------------------------------------------
set.seed(68)
data(cells, package = "modeldata")

cells_split <- rsample::initial_split(
  dplyr::select(cells, -case),
  strata = class
)

cells_train <- rsample::training(cells_split)
cells_test <- rsample::testing(cells_split)

# tuning hyperparameters --------------------------------------------------
tune_specs <- parsnip::decision_tree(
  cost_complexity = tune::tune(),
  tree_depth = tune::tune()
) |>
  parsnip::set_engine("rpart") |>
  parsnip::set_mode("classification")

tree_grid <- dials::grid_regular(dials::cost_complexity(),
  dials::tree_depth(),
  levels = 5
)

cell_folds <- rsample::vfold_cv(cells_train)

tree_workflow <- workflows::workflow() |>
  workflows::add_model(tune_specs) |>
  workflows::add_formula(class ~ .)

tree_res <- tree_workflow |>
  tune::tune_grid(
    resamples = cell_folds,
    grid = tree_grid
  )

tune::collect_metrics(tree_res) |>
  dplyr::mutate(tree_depth = factor(tree_depth)) |>
  ggplot2::ggplot(mapping = ggplot2::aes(cost_complexity, mean, color = tree_depth)) +
  ggplot2::geom_line(linewidth = 1.5, alpha = 0.5) +
  ggplot2::geom_point(size = 2) +
  ggplot2::facet_wrap(~.metric, scales = "free", nrow = 2) +
  ggplot2::scale_x_log10(labels = scales::label_number()) +
  ggplot2::scale_color_viridis_d(option = "plasma", begin = .9, end = 0)

best_tree <- tune::select_best(tree_res, "accuracy")

# finalize workflow -------------------------------------------------------

final_workflow <- tree_workflow |>
  tune::finalize_workflow(best_tree)

final_fit <- tune::last_fit(final_workflow, cells_split)

tune::collect_metrics(final_fit)

tune::collect_predictions(final_fit) |>
  yardstick::roc_curve(class, .pred_PS) |>
  ggplot2::autoplot()

final_tree <- tune::extract_workflow(final_fit)

tune::extract_fit_engine(final_tree) |>
  rpart.plot::rpart.plot(roundint = FALSE)
