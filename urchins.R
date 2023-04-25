# pkg = c("broom.mixed", "dotwhisker", "readr", "tidymodels")
# install.packages(pkg)

library(broom.mixed)
library(dotwhisker)
library(readr)
library(tidymodels)

# load data ---------------------------------------------------------------
urchins_url <- "https://tidymodels.org/start/models/urchins.csv"
urchins_names <- c("food_regime", "initial_volume", "width")
urchins <- readr::read_csv(urchins_url) |>
  setNames(urchins_names) |>
  dplyr::mutate(
    food_regime = factor(
      food_regime,
      levels = c("Initial", "Low", "High")
    )
  )

p <- ggplot2::ggplot(
  urchins,
  ggplot2::aes(
    x = initial_volume,
    y = width,
    group = food_regime,
    col = food_regime
  )
) +
  ggplot2::geom_point() +
  ggplot2::geom_smooth(method = lm, se = FALSE) +
  ggplot2::scale_color_viridis_d(option = "plasma", end = .7)

# linear model ------------------------------------------------------------
train <- sample(1:nrow(urchins), 0.8 * nrow(urchins))
urchins_train <- urchins[train, ]
urchins_test <- urchins[-train, ]

lm_model <- parsnip::linear_reg()
lm_fit <- lm_model |>
  fit(width ~ initial_volume * food_regime, data = urchins_train)

urchins_pred <- predict(lm_fit, new_data = urchins_test)

urchins_mse <- mean((urchins_pred$.pred - urchins_test$width)^2)
plot(urchins_pred$.pred, urchins_test$width)
