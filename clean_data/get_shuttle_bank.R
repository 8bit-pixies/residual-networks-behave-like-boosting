# install.packages(c("tidyverse", "tidymodel", "mlbench", "rsample"))
library(tidyverse)
library(rsample)
library(recipes)
library(mlbench)

data("Shuttle")

Shuttle <- Shuttle %>%
  mutate(Class = as.factor(as.numeric(as.factor(Class)) - 1))

set.seed(100)
train_test_split <- initial_split(Shuttle, prop = 0.7)
train_test_split

train_dat <- training(train_test_split)
test_dat  <- testing(train_test_split) 

train_data %>% write_csv("shuttle_train.csv")
test_data %>% write_csv("shuttle_test.csv")

rec_obj <- recipe(Class ~ ., data = train_dat) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())

trained_rec <- prep(rec_obj, training = train_dat)

train_data <- bake(trained_rec, newdata = train_dat)
test_data  <- bake(trained_rec, newdata = test_dat)

train_data %>% write_csv("shuttle_train_scale.csv")
test_data %>% write_csv("shuttle_test_scale.csv")

bank_additional_full <- read_delim("H:/temp/data-uci/bank-additional-full.csv", ";", escape_double = FALSE, trim_ws = TRUE)
bank_additional_full <- bank_additional_full %>%
  mutate(y=as.numeric(as.factor(y))-1) %>%
  mutate_if(is.character, as.factor)

set.seed(100)
train_test_split <- initial_split(bank_additional_full, prop = 0.7)
train_test_split

train_dat <- training(train_test_split)
test_dat  <- testing(train_test_split) 

rec_obj <- recipe(y ~ ., data = train_dat) %>%
  step_meanimpute(all_numeric()) %>%
  step_modeimpute(all_predictors(), -all_numeric()) %>%
  step_dummy(all_predictors(), -all_numeric()) %>%
  step_nzv(all_predictors())

trained_rec <- prep(rec_obj, training = train_dat)

train_data <- bake(trained_rec, newdata = train_dat)
test_data  <- bake(trained_rec, newdata = test_dat)

train_data %>% write_csv("bank_train.csv")
test_data %>% write_csv("bank_test.csv")

rec_obj <- recipe(y ~ ., data = train_dat) %>%
  step_meanimpute(all_numeric()) %>%
  step_modeimpute(all_predictors(), -all_numeric()) %>%
  step_dummy(all_predictors(), -all_numeric()) %>%
  step_nzv(all_predictors()) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())

trained_rec <- prep(rec_obj, training = train_dat)

train_data <- bake(trained_rec, newdata = train_dat)
test_data  <- bake(trained_rec, newdata = test_dat)

train_data %>% write_csv("bank_train_scale.csv")
test_data %>% write_csv("bank_test_scale.csv")
