# pkgs = c("parsnip", "broom", "broom.mixed", "readr", "dotwhisker", "dplyr", "tensorflow", "reticulate")
# install.packages(pkgs)
# reticulate::install_miniconda()
tensorflow::install_tensorflow()

# hard_coded variables & functions ----------------------------------------
# variables for urchins dataframe
urchins_url = "https://tidymodels.org/start/models/urchins.csv"
urchins_colnames = c("food_regime", "initial_volume", "width")
urchins_food_factor_levels = c("Initial", "Low", "High")
urchins_lm_formula = formula(width ~ initial_volume * food_regime)
urchins = readr::read_csv(urchins_url) |> 
  setNames(urchins_colnames) |> 
  dplyr::mutate(food_regime = factor(food_regime, levels = urchins_food_factor_levels))

# data exploration --------------------------------------------------------
str(urchins)

ggplot2::ggplot(urchins, 
                mapping = ggplot2::aes(
                  initial_volume, 
                  width, 
                  group = food_regime,
                  col = food_regime)) +
  ggplot2::geom_point() +
  ggplot2::geom_smooth(method = lm, se = FALSE) +
  ggplot2::scale_colour_viridis_d()

urchins_lm_model = parsnip::linear_reg() |> 
  parsnip::set_engine("keras")

urchins_lm_fit <- 
  urchins_lm_model |> 
  parsnip::fit(urchins_lm_formula, urchins)
