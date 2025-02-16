# This script is based on the kaggle dataset located here: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

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
    mode_value <- as.numeric(names(which.max(table(column))))  # Convert mode to numeric explicitly
    
    # Convert the column to a factor
    factor_column <- factor(column)
    
    # Ensure mode_value exists in the factor levels before releveling
    if (mode_value %in% levels(factor_column)) {
      factor_column <- relevel(factor_column, ref = as.character(mode_value))
    } else {
      warning(paste("Mode value", mode_value, "not found in factor levels. Skipping relevel."))
    }
    
    return(factor_column)
  }
  
  # Changes all columns to use their mode as the baseline
  update_columns_to_mode_factor <- function(df, columns) {
    # Ensure the input is a dataframe
    if (!is.data.table(df)) {
      stop("Input must be a dataframe.")
    }
    
    # Check that all specified columns exist in the dataframe
    if (!all(columns %in% colnames(df))) {
      stop("Some specified columns do not exist in the dataframe.")
    }
    
    # Apply the conversion to each specified column
    for (col in columns) {
      if (is.numeric(df[[col]])) {
        set(df, j = col, value = convert_to_factor_with_mode(df[[col]]))
        # df[[col]] <- convert_to_factor_with_mode(df[[col]])
      } else {
        warning(paste("Skipping column", col, "because it is not numeric."))
      }
  }
  
  return(df)
  }
  
  # Define a helper function to apply consistent factor levels to a dataset
  apply_factor_levels_to_dataset <- function(dataset, categorical_col_names, all_levels) {
    # Apply factor conversion to each column individually
    for (col in categorical_col_names) {
      # Apply factor levels based on 'all_levels' for each column
      dataset[[col]] <- factor(dataset[[col]], levels = all_levels[[col]])
    }
    return(dataset)
  }
  
  # Function to clean categorical columns
  clean_categorical_columns <- function(dataset, categorical_col_names) {
    # Ensure categorical columns are characters and replace NAs with "UNK"
    dataset[, (categorical_col_names) := lapply(.SD, function(x) {
      x <- as.character(x)
      x[is.na(x)] <- "UNK"
      return(x)
    }), .SDcols = categorical_col_names]
    
    # Return the updated dataset
    return(dataset)
  }
  
  
  
  # Define ETL function to preprocess the data
  etl_process <- function(data, create_encoded_vars = TRUE) {
    # Make sure column names are standardized
    setDT(data)
    setnames(data, make_names(names(data)))
    # Fix overallqual
    data[, ":=" (overallqual = fifelse(as.integer(as.character(overallqual)) <= 2, 2, overallqual))]
    
    names(data) %<>% {.} %>% make_names()
    
    # Define columns to convert to factors (you can adjust these as needed)
    cols_to_convert <- c("mssubclass", "overallcond", "overallqual") %>% 
      union(data %>% dplyr::select_if(is.character) %>% names)
    
    # Convert specified columns to factors and baseline at the mode
    data <- data %>% update_columns_to_mode_factor(., cols_to_convert)
    
    # Fix more columns (mode-based imputation, for example)
    bsmtqual_mode <- data$bsmtqual %>% get_mode()
    
    # Feature engineering and transformations
    data[, ":=" ( 
      # Convert month sold to a factor value
      mosold = factor(as.character(mosold), levels = as.character(1:12)),
     
      # Recode missing bsmtqual values to the mode
      bsmtqual = fifelse(is.na(bsmtqual), bsmtqual_mode, bsmtqual),
      
      # Add new feature 'total_sqft' based on existing features
      total_sqft = grlivarea + totalbsmtsf
    )]
    
    # Check if 'saleprice' exists and create new features based on it
    if ("saleprice" %in% names(data)) {
      # Create new, logged Y variable (log-transformed saleprice)
      data[, saleprice_log := fifelse(is.na(saleprice) | saleprice < 1, 0, log(saleprice))]
    }
    
    # Convert categorical variables to numeric encoding (creating new columns)
    if (create_encoded_vars) {
      categorical_cols <- names(data)[sapply(data, is.character) | sapply(data, is.factor)]
      
      if (length(categorical_cols) > 0) {
        for (col in categorical_cols) {
          new_col_name <- paste0(col, "_encoded")
          data[[new_col_name]] <- as.integer(as.factor(data[[col]]))
        }
      }
    }
    
    # Return processed data
    return(data)
  }
  
 
}

# Data Import
{
  # set working directory to local file path
  setwd("~/project_showcase_r_krolak")
  
  # Train import
  train_dt <- fread("./train.csv") 
  setDT(train_dt)
  setnames(train_dt, make_names(names(train_dt)))
  names(train_dt) %<>% {.} %>% make_names()
  # Fix overallqual
  train_dt[, ":=" (overallqual = fifelse(as.integer(as.character(overallqual)) <= 2, 2, overallqual))]
  
  cols_to_convert <- c("mssubclass", "overallcond", "overallqual" ) %>% union(train_dt %>% dplyr::select_if(is.character) %>% names)
  # convert columns to factors and baseline at the mode:
  train_dt <- train_dt %>% update_columns_to_mode_factor(., cols_to_convert)
  # Fix more columns
  bsmtqual_mode_train <- train_dt$bsmtqual %>% get_mode()
  # Feature additions
  train_dt[, ":=" (# Convert month sold to a factor value
                   mosold = factor(as.character(mosold), levels = as.character(1:12)),
                   # Recode the missings in bsmtqual to the mode
                   bsmtqual = fifelse(is.na(bsmtqual), bsmtqual_mode_train, bsmtqual),
                   total_sqft = grlivarea + totalbsmtsf,
                   saleprice_log = fifelse(is.na(saleprice) | saleprice < 1, 0, log(saleprice)))]
  
  
  
  # Test import  
  test_dt <- fread("./test.csv") %>% data.table
  test_dt <- etl_process(test_dt, create_encoded_vars = F)
  
  # names(test_dt) %<>% {.} %>% make_names()
  # # convert columns to factors and baseline at the mode:
  # test_dt <- test_dt %>% update_columns_to_mode_factor(., cols_to_convert)
  # # Fix more columns
  # bsmtqual_mode_test <- test_dt$bsmtqual %>% get_mode()
  # test_dt[, ":=" (# Convert month sold to a factor value
  #                 mosold = factor(as.character(mosold), levels = as.character(1:12)),
  #                 # Recode the missings in bsmtqual to the mode
  #                 bsmtqual = ifelse(is.na(bsmtqual), bsmtqual_mode_test, bsmtqual))]
  
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
    
  }
  
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
 
  # XGBoost as EDA
  {
    # Run a quick XGBoost to see which variables pop out as most important - this can be useful when dealing with new data and when subject matter experts are absent/unavailable.
    # Explicitly not doing any CV here since it's too much overhead for this simple EDA
    {
      # Define target variable
      target_var <- "saleprice_log"  # Adjust as needed
      id_var <- "id"
      
      # Convert categorical variables to numeric encoding (creating new columns)
      categorical_cols <- names(train_dt)[sapply(train_dt, is.character) | sapply(train_dt, is.factor)]
      
      if (length(categorical_cols) > 0) {
        for (col in categorical_cols) {
          # Create new column with encoded values
          new_col_name <- paste0(col, "_encoded")
          
          # Convert to integer factors and store in the new column
          train_dt[[new_col_name]] <- as.integer(as.factor(train_dt[[col]]))
        }
      }
      
      # Prepare data (EXCLUDE `id` and `target_var` and original categorical columns)
      # Create a list of columns to exclude (original categorical columns and target/id)
      exclude_cols <- c(target_var, id_var, categorical_cols)  # Original categorical columns are excluded
      # Make sure to catch all Y var look-alikes
      exclude_cols <- union(exclude_cols, names(train_dt) %>% grep("saleprice",.,value=T))
      # Include only the newly encoded columns in X
      X <- as.matrix(train_dt[, setdiff(names(train_dt), exclude_cols), with = FALSE])
      
      y <- train_dt[[target_var]]
      
      # Create DMatrix and create column names
      dtrain <- xgb.DMatrix(data = X, label = y)
      colnames(dtrain) <- colnames(X)  # Ensure column names match
      
      # Train XGBoost Model
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
  }
 
  # Basic Linear model to check how well we already fit
  {
    # Vars to include in basic linear model 
    x_vars <- names(plot_list)[1:7] # grab top 7 XGBoost vars for now
    # If any "_encoded vars were important, gsub out the encoded part for linear modeling.
    x_vars <- x_vars %>% gsub("_encoded","",.)
    # want year and month sold for interpretability
    date_vars <- names(train_dt) %>% grep("mosold$|yrsold$",.,value=T)
    x_vars <- union(date_vars, x_vars) # add date_vars to x's
    # also a good idea to add in a location variable since Location, Location, Location is important for real estate pricing
    x_vars <- union(x_vars, "neighborhood") 
    
    # Let's investigate a few easy-to-understand variables and try a basic linear model for a first attempt:
    {
      # Define what columns should be in the model's explanatory variables
      x_vars
      y_var <- target_var
      
      # Let's make sure they don't have any data issues before modeling:
      train_dt %>% dplyr::select(c(x_vars, y_var)) %>% skim
      # Investigate any NAs and clean data
      
      # Initial modeling ETL
      {
        # Define what columns should be in the model's explanatory variables
        x_vars
        
        # Copy training dataset to run ETL on while retaining original dataset
        train_dt_linear <- data.table(train_dt)
        # Don't include any encoded vars from XGB ETL
        encoded_cols <- names(train_dt_linear) %>% grep("encode",.,value=T) %>% unique
        # Convert selected columns to character
        cols_to_char <- grep("mosold|overallqual|neighborhood", names(train_dt_linear), value=T) %>% setdiff(encoded_cols)
        train_dt_linear[, (cols_to_char) := lapply(.SD, function(x) as.character(x)), .SDcols = cols_to_char]
        
        # Define the model formula as just the y variable with basic linear behavior for all explanatory vars
        formula <- paste0(y_var, " ~ ", paste0(x_vars, collapse = " + ")) %>% as.formula
        
        # Split the data into 80% training and 20% validation
        data_split <- sample(1:nrow(train_dt_linear), size = 0.8 * nrow(train_dt_linear))
        training_data <- train_dt_linear[data_split, ]
        validation_data <- train_dt_linear[-data_split, ]
        
        # Categorical columns can be tricky if any of them have levels which are only present in train/validation/test datasets. Ensure this doesn't happen.
        # Handle categorical columns
        {
          # Categorical columns
          categorical_col_names <- train_dt_linear %>% 
            dplyr::select(x_vars) %>%
            dplyr::select_if(~!is.numeric(.)) %>%
            names()
          
          # If there are categorical vars, handle them
          if (length(categorical_col_names) > 0) {
            
            # Clean the columns
            training_data <- clean_categorical_columns(training_data, categorical_col_names)
            validation_data <- clean_categorical_columns(validation_data, categorical_col_names)
            
            # Ensure consistent levels for all categorical variables across both training and validation data
            all_levels <- lapply(categorical_col_names, function(cat) {
              unique(training_data[[cat]])  # Use only training data levels
            })
            names(all_levels) <- categorical_col_names
            
            # Apply the consistent factor levels to both training and validation data
            training_data <- apply_factor_levels_to_dataset(training_data, categorical_col_names, all_levels)
            validation_data <- apply_factor_levels_to_dataset(validation_data, categorical_col_names, all_levels)
            
            # Remove rows from the validation set where there are unseen factor levels
            unseen_levels <- lapply(categorical_col_names, function(cat) {
              !(validation_data[[cat]] %in% levels(training_data[[cat]]))
            })
            
            # Apply the unseen levels filtering to the validation dataset
            rows_to_keep <- Reduce("&", unseen_levels)
            validation_data <- validation_data[!rows_to_keep, ]
          }
        }
      }
      
      # Initial linear model fitting/exporting
      {
        # Fit the linear model on the training data
        model <- lm(formula, data = training_data)
        
        # Predict on the validation data
        validation_predictions <- predict(model, newdata = validation_data)
        
        # Compare predictions vs true y-values
        validation_results <- data.table(y_vals = validation_data$saleprice, predicted = validation_predictions)
        
        # Calculate performance metrics
        validation_metrics <- postResample(validation_results$predicted, validation_results$y_vals) %>% round(4)
        
        # Display the R-squared value along with other metrics (RMSE, MAE)
        print(validation_metrics) # the validation data's R^2 is quite high, over 80%!
        
        # Display the summary of the model
        print(summary(model)) # In-sample R^2 very close to out-of-sample
        
        # Save the model to an RDS file
        saveRDS(model, "linear_model.rds")
      }
      
    }
  }
 
   
}




# Test export for kaggle
{
  # Load the model from the RDS file
  loaded_model <- readRDS("linear_model.rds")
 
  # # Preprocess the test data if necessary (this might include handling missing values, encoding, etc.)
  # # You would need to apply the same preprocessing steps you used to train the model
  # test_data_processed <- preprocess(test_data)  # Ensure this matches your training data's preprocessing

  # Generate predictions using the loaded model
  predictions <- predict(model, newdata = test_dt)
  
  # Prepare the submission format (e.g., the competition might require an 'id' column and a 'SalePrice' column)
  submission <- data.table(id = test_data$id, SalePrice = predictions)
  
  # Save the predictions as a CSV file
  write.csv(submission, "submission.csv", row.names = FALSE)
  
  # At this point, you can go to Kaggle and submit the CSV file
  
}







