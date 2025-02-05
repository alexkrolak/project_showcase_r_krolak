

# Libraries
{
  setwd("~/project_showcase_r_krolak")
  # Need to install xgboostExplainer from my github repo fork with fixes. It is no longer supported on CRAN and breaks due to XGBoost package Updates
  # remotes::install_github("alexkrolak/xgboostExplainer")
  
  library(tidyverse)
  library(data.table)
  library(skimr)
  library(DataExplorer)
  library(caret)
  library(GGally)
  library(data.table)
  library(xgboost)
  library(xgboostExplainer)
  library(officer)
  library(ggplot2)
  library(caret)
}

# UDFs
{
  # Function to clean column names
  make_names <- function(col_names){
    # Transform the column names
    cleaned_names <- col_names %>%
      tolower() %>%                   # Convert to lowercase
      gsub("[^a-z0-9_]", "_", .) %>%  # Replace spaces and special characters with underscores
      gsub("_+", "_", .) %>%            # Replace multiple underscores with a single underscore
      sapply(function(col) {            # Add "x" to column names starting with numbers
        if (grepl("^[0-9]", col)) paste0("x", col) else col
      })
    
    # Return the transformed column names
    return(cleaned_names)
  }
  
  # Function to find the mode of a column of data
  get_mode <- function(column) {
    uniq_vals <- unique(column)
    uniq_vals[which.max(tabulate(match(column, uniq_vals)))]
  }
  
  # Recode missing to UNK
  code_missing_to_UNK <- function(df) {
    # Ensure the input is a dataframe
    if (!is.data.frame(df)) {
      stop("Input must be a dataframe.")
    }
    
    # Iterate over character columns and replace NAs with "UNK"
    df[] <- lapply(df, function(col) {
      if (is.character(col)) {
        col[is.na(col)] <- "UNK"
      }
      else if (is.numeric(col)){
        col <- col %>% as.character()
        col[is.na(col)] <- "UNK"
      }
      return(col)
    })
    
    return(df)
  }
  
  # Convert my column to factor with smallest value as baseline
  convert_to_factor_with_mode <- function(column) {
    # Ensure the input is numeric
    if (!is.numeric(column)) {
      stop("Input column must be numeric.")
    }
    
    # Find the most frequent value (mode)
    mode_value <- names(which.max(table(column)))
    
    # Convert the column to a factor
    factor_column <- factor(column)
    
    # Set the mode as the baseline
    factor_column <- relevel(factor_column, ref = as.character(mode_value))
    
    return(factor_column)
  }
  
  # Changes all columns to use their mode as the baseline
  update_columns_to_mode_factor <- function(df, columns) {
    # Ensure the input is a dataframe
    if (!is.data.frame(df)) {
      stop("Input must be a dataframe.")
    }
    
    # Check that all specified columns exist in the dataframe
    if (!all(columns %in% colnames(df))) {
      stop("Some specified columns do not exist in the dataframe.")
    }
    
    # Apply the conversion to each specified column
    for (col in columns) {
      if (is.numeric(df[[col]])) {
        df[[col]] <- convert_to_factor_with_mode(df[[col]])
      } else {
        warning(paste("Skipping column", col, "because it is not numeric."))
      }
  }
  
  return(df)
}
}

# Data Import
{
  # set working directory to local file path
  setwd("~/project_showcase_r_krolak")
  
  # Train import
  train_dt <- fread("./train.csv") %>% data.table
  names(train_dt) %<>% {.} %>% make_names()
  cols_to_convert <- c("mssubclass", "overallcond", "overallqual" ) %>% union(train_dt %>% dplyr::select_if(is.character) %>% names)
  # convert columns to factors and baseline at the mode:
  train_dt <- train_dt %>% update_columns_to_mode_factor(., cols_to_convert)
  # Fix more columns
  bsmtqual_mode_train <- train_dt$bsmtqual %>% get_mode()
  train_dt[, ":=" (# Convert month sold to a factor value
                   mosold = factor(as.character(mosold), levels = as.character(1:12)),
                   # Recode the missings in bsmtqual to the mode
                   bsmtqual = ifelse(is.na(bsmtqual), bsmtqual_mode_train, bsmtqual))]
  
  
  
  # Test import  
  test_dt <- fread("./test.csv") %>% data.table
  names(test_dt) %<>% {.} %>% make_names()
  # convert columns to factors and baseline at the mode:
  test_dt <- test_dt %>% update_columns_to_mode_factor(., cols_to_convert)
  # Fix more columns
  bsmtqual_mode_test <- test_dt$bsmtqual %>% get_mode()
  test_dt[, ":=" (# Convert month sold to a factor value
                  mosold = factor(as.character(mosold), levels = as.character(1:12)),
                  # Recode the missings in bsmtqual to the mode
                  bsmtqual = ifelse(is.na(bsmtqual), bsmtqual_mode_test, bsmtqual))]
  
  # Import sample submission to validate final export
  sample_submission <- fread("./sample_submission.csv") %>% data.table

}

# EDA
{
  # Skim and DataExplorer
  {
    # How many columns of each type?
    train_dt %>% sapply(class) %>% table
    # Look at columns in groupings of "character", "factor", "integer"
    train_dt %>% sapply(class) %>% sort
    # View first few entries per column
    train_dt %>% glimpse
    # Summarize each column 
    train_dt %>% skim # Many missing values in some character columns, only missing a handful of values from a couple of numeric columns though
    # Plots
    train_dt %>% plot_bar() # No categorical variable has > 50 uniques, many have a ton of missing values though
    train_dt %>% plot_histogram() # Will delve deeper into some of these numeric variables later after XGBoost model EDA since there are so many of them. For now...
    
    # EDA on Y variable and the month/year date variables
    {
      # y-variable
        # saleprice: is not normal, almost looks chi-squared-like. will need to transform this for a general linear model/regression
      # date vars
        # yrsold: is only about 1/2 as populated for 2010, so houses predicted from that year may have worst fit
        # mosold: shows almost a normal distribution of counts peaking around June. Need to look at this plot by-year
        # Are month and year sold fairly equally represented?
        # A: Similar count/proportion of rows by month per year, so each year's data has the same chance of inducing variance. In other words, it's not immediately obvious that a month-year interaction term would be necessary in a linear model.
        train_dt[,.(mosold, yrsold)]  %>% 
          ggplot(aes(x=mosold)) + 
          geom_bar() + 
          facet_wrap(.~yrsold) 
    }
   
    # Run a quick XGBoost to see which variables pop out as most important - this can be useful when dealing with new data and when subject matter experts are absent/unavailable.
    # Explicitly not doing any CV here since it's too much overhead for this simple EDA
    {
      # Define target variable
      target_var <- "saleprice"  # Adjust as needed
      id_var <- "id"
      
      # Convert categorical variables to numeric encoding
      categorical_cols <- names(train_dt)[sapply(train_dt, is.character) | sapply(train_dt, is.factor)]
      if (length(categorical_cols) > 0) {
        for (col in categorical_cols) {
          train_dt[[col]] <- as.integer(as.factor(train_dt[[col]]))  # Convert to integer factors
        }
      }
      
      # Prepare data (EXCLUDE `id` and `target_var`)
      X <- as.matrix(train_dt[, setdiff(names(train_dt), c(target_var, id_var)), with = FALSE])
      y <- train_dt[[target_var]]
      
      # Create DMatrix and create column names
      dtrain <- xgb.DMatrix(data = X, label = y)
      colnames(dtrain) <- colnames(X)  # Ensure column names match
      
      # Train XGBoost Model - use basic hyperparameters for now
      params <- list(objective = "reg:squarederror", eval_metric = "rmse", max_depth = 6, eta = 0.1, subsample = 0.8, colsample_bytree = 0.8)
      xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100, verbose = 0)
      
      # Generate Feature Importance
      importance_matrix <- xgb.importance(model = xgb_model)
      top_features <- importance_matrix[1:20, ]  # Select top 20 most important features
      
      # Create explainer
      explainer <- buildExplainer(xgb_model, dtrain, type = "regression", base_score = mean(y))
      
      # Generate explanations
      expl_plot <- explainPredictions(xgb_model, explainer, dtrain)
      
      # Create PowerPoint
      ppt <- read_pptx()
      
      plot_list <- list()
      # Loop through each top feature and generate a plot
      for (feature in top_features$Feature) {
        
        # Extract feature impact (Y-axis) from explainer output
        expl_plot_feature <- data.table(Effect = expl_plot[[feature]])
        
        # Extract actual feature values (X-axis) from the dataset
        expl_plot_feature$x_value <- train_dt[[feature]]
        
        # Remove missing values before plotting
        expl_plot_feature <- na.omit(expl_plot_feature)
        
        
        # Generate ggplot for the feature
        p <- ggplot(expl_plot_feature, aes(x = x_value, y = Effect)) +
          geom_point() +
          labs(
            title = paste("Effect of", feature, "on", target_var),
            x = feature,  # Use feature name for x-axis label
            y = "Effect on Prediction"
          ) +
          theme_minimal()
        
        # Store these plots in a list for later
        index <- grep(feature, top_features$Feature)
        plot_list[[index]] <- p
        names(plot_list)[index] <- feature
        
        # Add slide to PowerPoint
        ppt <- add_slide(ppt, layout = "Title and Content", master = "Office Theme")
        ppt <- ph_with(ppt, value = feature, location = ph_location_type(type = "title"))
        ppt <- ph_with(ppt, value = p, location = ph_location_fullsize())
        
      }
      # Save PowerPoint
      output_file <- "XGBoost_Explainer_Presentation.pptx"
      print(ppt, target = output_file)
      
      cat("PowerPoint exported successfully as:", output_file, "\n")
        
    }
    
    # XGB Findings: Let's start with the top 7 variables since their impact plots are simple enough, and we can add more variables as we go along
    {
      # Top variable, overallqual, has a non-linear impact on y-hat (needs stratifying in a linear model, e.g. 0-5, 6, 7, 8, 9-10)
      plot_list[[1]]
      # 2nd var, grlivarea, seems fine for now with a few outliers
      plot_list[[2]]
      # 3rd var, garagecars, also nonlinear, looks like 0-2 and 3-4 can be strata
      plot_list[[3]]
      # 4th var, totalbsmtsf, looks ok-ish. Lots of 0's in the data causes an odd pattern, but should overall be fine for basic linear model
      plot_list[[4]]
      # 5th var, x1stflrsf, seems fine for now - not perfect, but nothing too horrible
      plot_list[[5]]
      # 6th var, bsmtfinsf1, similar to 4th variable with many 0's + some outliers, but that's not the end of the world
      plot_list[[6]]
      # 7th var, bsmtqual, might be ok, could benefit from stratification e.g. 1, 2-4
      plot_list[[7]]
      # 8th var, bsmtqual, this plot has a few outliers - useable, but we can stop for now since it'll probably need cleaning
      plot_list[[8]]
    }
      
    # Vars to include in basic linear model 
    x_vars <- names(plot_list)[1:7] # grab top 7 XGBoost vars for now
    # want year and month sold for interpretability
    date_vars <- names(train_dt) %>% grep("mosold|yrsold",.,value=T)
    x_vars <- union(date_vars, x_vars) # add date_vars to x's
    # also a good idea to add in a location variable since Location, Location, Location is important for real estate pricing
    x_vars <- union(x_vars, "neighborhood") 
    
    # Let's investigate a few easy-to-understand variables and try a basic linear model for a first attempt:
    {
      # Define what columns should be in the model's explanatory variables
      x_vars
      y_var <- "saleprice"
      
      # Let's make sure they don't have any data issues before modeling:
      train_dt %>% dplyr::select(c(x_vars, y_var)) %>% skim
      # Investigate any NAs and clean data
      
      # Initial modeling
      {
        # Define what columns should be in the model's explanatory variables
        # x_vars <- c("overallqual", "mosold", "yrsold", "housestyle", "exterior1st", "exterior2nd", "x1stflrsf", "mssubclass", "neighborhood") %>% unique()
        x_vars
        # Recode mosold to character for linear modeling
        train_dt_linear = data.table(train_dt)
        cols_to_char <- grep("mosold|overallqual|neighborhood", names(train_dt_linear), value=T)
        # Convert selected columns to character
        train_dt_linear[, (cols_to_char) := lapply(.SD, function(x) as.character(x)), .SDcols = cols_to_char]
        
        # Define the model formula as just the y variable with basic linear behavior for all explanatory vars
        formula <- paste0("saleprice ~ ", paste0(x_vars, collapse = " + ")) %>% as.formula
        
        # Split the data into 80% training and 20% validation
        data_split <- sample(1:nrow(train_dt_linear), size = 0.8 * nrow(train_dt_linear))
        training_data <- train_dt_linear[data_split, ]
        validation_data <- train_dt_linear[-data_split, ]
        
        # Handle categorical columns
        {
          # Categorical columns
          categorical_col_names <- train_dt_linear %>% 
            dplyr::select(x_vars) %>%
            dplyr::select_if(~!is.numeric(.)) %>%
            names()
          
          # If there are categorical vars, handle them
          if (length(categorical_col_names) > 0) {
            
            # Ensure all categorical columns in both training and validation datasets
            for (col in categorical_col_names) {
              # Convert to character first
              training_data[[col]] <- as.character(training_data[[col]])
              validation_data[[col]] <- as.character(validation_data[[col]])
              
              # Replace NAs with "UNK"
              training_data[[col]][is.na(training_data[[col]])] <- "UNK"
              validation_data[[col]][is.na(validation_data[[col]])] <- "UNK"
            }
            
            # Ensure consistent levels for all categorical variables across both training and validation data
            all_levels <- lapply(categorical_col_names, function(cat) {
              unique(training_data[[cat]])  # Use only training data levels
            })
            names(all_levels) <- categorical_col_names
            
            # Apply the consistent factor levels to both training and validation data
            for (cat in categorical_col_names) {
              # Convert to factor and set levels to include all observed levels in the training data only
              training_data[[cat]] <- factor(training_data[[cat]], levels = all_levels[[cat]])
              
              # For validation data, remove any levels not seen in training data
              validation_data[[cat]] <- factor(validation_data[[cat]], levels = all_levels[[cat]])
              
              # Remove rows from the validation set where there are unseen factor levels
              unseen_levels <- !(validation_data[[cat]] %in% levels(training_data[[cat]]))
              validation_data <- validation_data[!unseen_levels, ]
            }
          }
        }
        
        # Fit the linear model on the training data
        model <- lm(formula, data = training_data)
        
        # Predict on the validation data
        validation_predictions <- predict(model, newdata = validation_data)
        
        # Compare predictions vs true values
        validation_results <- data.table(y_vals = validation_data$saleprice, predicted = validation_predictions)
        
        # Calculate performance metrics (including R-squared)
        validation_metrics <- postResample(validation_results$predicted, validation_results$y_vals) %>% round(4)
        
        # Display the R-squared value along with other metrics (RMSE, MAE)
        print(validation_metrics) # the validation data's R^2 is quite high, over 85%!
        
        # Display the summary of the model
        summary(model) # In-sample R^2 is lower than validation data
      }
      
    }
    
    
    
  }
  
  
  
  
  
 
}











