library(shiny)
library(dplyr)
library(randomForest)
library(caret)
library(caTools)
library(ggplot2)
library(plotly)
library(knitr)
library(glmnet)
library(rpart)
library(rpart.plot)


ui <- fluidPage(
  titlePanel("Moustapha Kebe"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "CHARGEMENT DES DONNEES", accept = c(".csv",".txt","xls")),
      selectInput("delimiter","Séparateur",choices = c(",",";","\t"," "),selected = ","),
      actionButton("go", "Load"),
      radioButtons("missing_values_action", "Traitement des valeurs manquantes :",
                   choices = list("Supprimer" = "remove",
                                  "Remplacer" = "remplacer",
                                  "Conserver" = "keep"),
                   selected = "keep"),
      radioButtons("modeles", "Modèles",
                   choices = c("Decision Tree","Logistic Regression","Random Forest"),
                   selected = "Random Forest")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Data", 
                 dataTableOutput("contents"),
                 textOutput("Dimensions"),
                 downloadButton("save", "Download data")
        ),
        tabPanel("infos-Cleaned",
                 verbatimTextOutput("var"),
                 verbatimTextOutput("NA_"),
                 verbatimTextOutput('clean'),
                 verbatimTextOutput("class"),
                 verbatimTextOutput("summary")
                 
        ),
        tabPanel("Data Cleaned",
                 verbatimTextOutput("summary2"),
                 verbatimTextOutput("NA_1"),
                 dataTableOutput("data1_clean"),
                 textOutput("Dimensions1"),
                 downloadButton("save1", "Download data cleaned")
        ),
        tabPanel("Univariée",
                 fluidRow(
                   uiOutput("quantlist"),
                   tableOutput("centreDisp")
                 ),
                 fluidRow(
                   column(6,plotOutput("effectifsHist")),
                   column(6,plotOutput("effectifsCumCurve")),
                   plotOutput("boiteMoustaches")
                 )
                 
        ),
        tabPanel("Bivariée",
                 uiOutput("quantlistbi1"),
                 uiOutput("quantlistbi2"),
                 fluidRow(
                   textOutput("correlation"),
                   tableOutput("caract"),
                   titlePanel("Nuage de points & histogramme"),
                   plotOutput("nuagePoints"),
                   column(6,
                          plotOutput("histX")
                   ),
                   column(6,
                          plotOutput("histY"))
                   
                 )
                 
        ),
        tabPanel("Target",
                 uiOutput("select_target"),
                 plotOutput("target_plot")
        )
        ,
        tabPanel(
          "Modele",
          conditionalPanel(condition = "input.modeles=='Random Forest'",
                           verbatimTextOutput("model_summary"),
                           verbatimTextOutput("result_model"),
                           verbatimTextOutput("feature"),
                           plotOutput("feature_importance"),
                           verbatimTextOutput("confusion_matrix")
          ),
          conditionalPanel(
            condition = "input.modeles=='Logistic Regression'",
            verbatimTextOutput("reg_model_summary"),
            verbatimTextOutput("reg_result_model"),
            verbatimTextOutput("coefficients"),
            verbatimTextOutput("reg_confusion_matrix")
            
          ),
          conditionalPanel(
            condition = "input.modeles=='Decision Tree'",
            verbatimTextOutput("tree_model"),
            plotOutput("tree_plot"),
            verbatimTextOutput("tree_model_summary")
          )
          
        )
      )
    )
  )
)

server <- function(input, output) {
  
  data <- eventReactive(input$go, {
    req(input$file)
    
    data<-read.csv(input$file$datapath, header = TRUE, sep = input$delimiter)
    #data <- data %>%
    # mutate_all(~ ifelse(. == '?', NA, .))
    return(data)
  })
  
  output$NA_ <- renderPrint({
    percent_missing <- colMeans(is.na(data())) * 100
    print(percent_missing)
  })
  
  find_constants_and_duplicate_rows <- function(data) {
    duplicate_rows <- duplicated(data) | duplicated(data, fromLast = TRUE)
    constant_columns <- sapply(data, function(col) length(unique(col)) == 1)
    list(duplicate_rows = which(duplicate_rows), constant_columns = which(constant_columns))
  }
  
  
  cleaned_data <- reactive({
    req(data())
    #if(length(find_constants_and_duplicate_rows()$duplicate_rows)!=0){
    #data()<-data()[-find_constants_and_duplicate_rows()$duplicate_rows,]
    #}
    #if(length(find_constants_and_duplicate_rows()$constant_columns)!=0){
    #  data()<-data()[,-find_constants_and_duplicate_rows()$constant_columns]
    #}
    cleaned <- data()
    
    if (input$missing_values_action == "remove") {
      cleaned <- cleaned[complete.cases(cleaned), ]
    } else if (input$missing_values_action == "remplacer") {
      # Remplacer les valeurs manquantes dans les colonnes numériques par la médiane
      cleaned <- cleaned %>%
        mutate_if(is.numeric, ~ifelse(is.na(.), median(., na.rm = TRUE), .))
      
      # Remplacer les valeurs manquantes dans les colonnes catégorielles par le mode
      #cleaned <- cleaned %>%
      #  mutate_if(is.character, ~ ifelse(is.na(.), names(sort(table(.), decreasing = TRUE))[1], .))
      cleaned <- cleaned %>%
        mutate_if(function(col) is.character(col) || is.factor(col),
                  ~ ifelse(is.na(.), names(sort(table(.[!is.na(.)]), decreasing = TRUE))[1], .))
      
      
    } else {
      # Retourner le jeu de données original si l'action spécifiée n'est ni "remove" ni "remplacer"
      return(cleaned)
    }
    
    return(cleaned)
  })
  #Univariée
  output$quantlist = renderUI({
    selectInput('qnt', 'Le choix de la variable', names(cleaned_data())[!grepl('factor|logical|character',sapply(cleaned_data(),class))])
  })
  tabCentreDisp <- reactive({
    req(input$qnt)
    # Noms des caractéristiques
    dt =cleaned_data()[,input$qnt]
    #df=cleaned_data()
    names.tmp <- c("Maximum", "Minimum", "Moyenne", "Médiane",
                   "1e quartile", "3e quartile", "Variance", "Ecart-type")
    # Calcul des caractéristiques
    
    summary.tmp <- c(max(dt), min(dt), mean(dt), median(dt),
                     quantile((dt))[2], quantile((dt))[4],
                     var(dt), sqrt(var(dt)))
    # Ajout des nomes au vecteur de valeurs
    summary.tmp <- cbind.data.frame(names.tmp, summary.tmp)
    # Ajout des noms de colonnes
    colnames(summary.tmp) <- c("Caractéristique", "Valeur")
    
    summary.tmp
  })
  output$centreDisp <- renderTable({tabCentreDisp()})
  
  # Commande pour l'affichage du plot des effectifs
  output$effectifsDiag <- renderPlot({ 
    dt = cleaned_data()
    plot(table(data.frame(dt[,input$qnt])), col ="blue", xlab =sym(input$qnt), ylab ="Effectifs", 
         main ="Distribution des effectifs")
  })
  # boite aux moustaches
  output$boiteMoustaches <- renderPlot({
    dt = cleaned_data()
    boxplot( data.frame(as.numeric(as.character(dt[,input$qnt]))), col = "blue", 
             main = paste("Boite à moustaches de:",input$qnt),
             ylab = "", las = 1)
    rug(cleaned_data()[,input$qnt], side = 2)
  })
  
  output$effectifsHist <- renderPlot({
    dt = cleaned_data()
    # Histogramme des effectifs
    hist(as.numeric(as.character(dt[,input$qnt])) , freq = TRUE, cex.axis = 1.5, cex.main = 1.5,
         main = "Histogramme", col = "blue",
         xlab = sym(input$qnt), ylab = "Effectifs", las = 1,
         right = FALSE, cex.lab = 1.5)
  })
  
  
  output$effectifsCumCurve <- renderPlot({
    dt = cleaned_data()
    tmp.hist <- hist(as.numeric(as.character(dt[,input$qnt])) , plot = FALSE,
                     right = FALSE)
    # Courbe cumulative (effectifs)
    plot(x = tmp.hist$breaks[-1], y = cumsum(tmp.hist$counts),
         xlab =sym(input$qnt),
         ylab = "Effectifs cumulés", cex.axis = 1.5, cex.lab = 1.5,
         main = "Courbe cumulative ",
         type = "o", col = "green", lwd = 2, cex.main = 1.5)
    
  })
  
  
  
  #BIVARIEE
  
  output$quantlistbi1 = renderUI({
    selectInput('quantlistbi1', 'Le choix de la variable X', names(cleaned_data())[!grepl('factor|logical|character',sapply(cleaned_data(),class))])
  })
  output$quantlistbi2 = renderUI({
    df2 <- cleaned_data()[,!names(cleaned_data()) %in% c(input$quantlistbi1)]
    selectInput('quantlistbi2', 'Le choix de la variable Y', names(df2)[!grepl('factor|logical|character',sapply(df2,class))])
  })
  
  output$caract <- renderTable({
    req(input$quantlistbi1)
    req(input$quantlistbi2)
    req(cleaned_data)
    var.names <-c(input$quantlistbi1,input$quantlistbi2)
    df<-cleaned_data()
    caract.df <- data.frame()
    for(strCol in var.names){
      caract.vect <- c(min(df[, strCol]), max(df[,strCol]), 
                       mean(var(df[,strCol])), sqrt(var(df[,strCol])))
      caract.df <- rbind.data.frame(caract.df, caract.vect)
    }
    rownames(caract.df) <- var.names
    colnames(caract.df) <- c("Minimum", "Maximum", "Moyenne", "Ecart-type")
    caract.df
  }, rownames = TRUE, digits = 0)
  
  
  output$nuagePoints <- renderPlot({
    req(input$quantlistbi1)
    req(input$quantlistbi2)
    options(digits=1, col="blue")
    dt = cleaned_data()
    dt2 =dt[,input$quantlistbi1]
    dt2 = as.numeric(dt2)
    dt = cleaned_data()
    dt3 =dt[,input$quantlistbi2]
    dt3 = as.numeric(dt3)
    X = dt2; 
    Y = dt3;
    ggplot(cleaned_data(),mapping = aes_string(x = X, y = Y)) +
      geom_point(color = "blue") +
      labs(title =paste("Nuage de points entre:",paste(c(input$quantlistbi1,input$quantlistbi2),collapse = " et ")))
    
    
  })
  
  output$correlation<-renderText({
    req(cleaned_data())
    dt = cleaned_data()
    dt2 =dt[,input$quantlistbi1]
    dt2 = as.numeric(dt2)
    dt = cleaned_data()
    dt3 =dt[,input$quantlistbi2]
    dt3 = as.numeric(dt3)
    #x.var = input$choix ; y.var = input$choixx;
    coeff.tmp <- cov(dt2, dt3)/(sqrt(var(dt2)*var(dt3)))
    paste("\nCoefficient de corrélation linéaire =", round(coeff.tmp,digits = 2))
    
  })
  
  output$histX <- renderPlot({
    req(input$quantlistbi1)
    dt <- cleaned_data()
    dt2 <- dt[, input$quantlistbi1]
    dt2 <- as.numeric(dt2)
    X = dt2; 
    ggplot(dt, aes_string(x = X)) +
      geom_histogram(col = "red") +
      labs(title = paste("Histogramme de", input$quantlistbi1))
  })
  
  output$histY <- renderPlot({
    req(input$quantlistbi2)
    dt <- cleaned_data()
    dt2 <- dt[, input$quantlistbi2]
    dt2 <- as.numeric(dt2)
    X = dt2; 
    ggplot(dt, aes_string(x = X)) +
      geom_histogram(col = "red") +
      labs(title = paste("Histogramme de", input$quantlistbi2))
  })
  
  
  #Les types de variables
  get_variable_types <- function(data) {
    categorical_vars <- lapply(data, function(col) {
      if (is.factor(col) || is.character(col)) {
        return(colnames(data)[sapply(data, identical, col)])
      }
    })
    numerical_vars <- lapply(data, function(col) {
      if (is.numeric(col) && !is.factor(col)) {
        return(colnames(data)[sapply(data, identical, col)])
      }
    })
    categorical_vars <- unlist(categorical_vars)
    categorical_vars <- categorical_vars[!is.na(categorical_vars)]
    categorical_vars <- unique(categorical_vars)
    numerical_vars <- unlist(numerical_vars)
    numerical_vars <- numerical_vars[!is.na(numerical_vars)]
    numerical_vars <- unique(numerical_vars)
    return(list(categorical = categorical_vars, numerical = numerical_vars))
    
  }
  
  output$var <- renderPrint({
    variable_types <- get_variable_types(data())
    formatted_output <- paste(
      "Variables catégoriques :",
      paste(variable_types$categorical, collapse = ", "),
      "\n\nVariables numériques :",
      paste(variable_types$numerical, collapse = ", "),
      "\n"
    )
    cat(formatted_output) 
  })
  
  
  
  output$clean <- renderPrint({
    var_duplicate<-find_constants_and_duplicate_rows(data())
    View_format<-paste(
      "duplicate rows:",
      paste(var_duplicate$duplicate_rows,collapse= ","),
      "\n\nconstant columns:",
      paste(var_duplicate$constant_columns,collapse = ","),
      "\n"
    )
    cat(View_format)
  })
  
  
  
  # Information sur la variable cible et analyse univariée et bivariée
  
  
  output$select_target <- renderUI({
    if (input$modeles == "Logistic Regression" || input$modeles == "Random Forest" || input$modeles=="Decision Tree") {
      selectInput('target', 'Select Target', choices = names(cleaned_data()), selected = NULL)
    }
  })
  output$target_plot<-renderPlot({
    req(input$target)
    dt <- cleaned_data()
    dt2 <- dt[, input$target]
    dt2 <- as.numeric(dt2)
    X = dt2; 
    ggplot(dt, aes_string(x = X)) +
      geom_histogram(col = "red") +
      labs(title = paste("Histogramme de", input$target))
  })
  
  #MODELES
  
  trained_model <- reactive({
    req(input$target)
    req(input$modeles)
    target_variable <- input$target
    train_data <- cleaned_data()
    dummy <- dummyVars(~ ., data = train_data)
    train_data_transformed <- data.frame(predict(dummy, newdata = train_data))
    train_data_transformed[, target_variable] <- as.factor(train_data_transformed[, target_variable])
    
    inTrain <- createDataPartition(train_data_transformed[, target_variable], p = 0.8, list = FALSE)
    training <- train_data_transformed[inTrain,]
    testing <- train_data_transformed[-inTrain,]
    testing_features <- testing[, !colnames(testing) %in% target_variable]
    
    model <- randomForest(as.formula(paste(target_variable, "~ .")), data = training)
    
    predictions <- predict(model, newdata = testing_features)
    
    precision <- precision(predictions, as.factor(testing[, target_variable]))
    recall<- recall(predictions,as.factor(testing[, target_variable]))
    #AIC<- AIC(model)
    #BIC<- BIC(model)
    confusion_matrix<-confusionMatrix(predictions, as.factor(testing[, target_variable]))
    f1_score <- confusion_matrix$byClass["F1"]
    list(
      model = model,
      model_summary = capture.output(print(model)),
      rf=capture.output(print(summary(model))),
      precision = precision,
      recall=recall,
      f1_score=f1_score,
      confusionmatrix=confusion_matrix,
      feature_importance = importance(model)
    )
  })
  
  reg_trained_model <- reactive({
    req(input$target)
    req(input$modeles)
    target_variable <- input$target
    train_data <- cleaned_data()
    
    # Convertir les variables catégoriques en facteurs
    categorical_vars <- which(sapply(train_data, is.character) | sapply(train_data, is.factor))
    train_data[categorical_vars] <- lapply(train_data[categorical_vars], as.factor)
    
    dummy <- dummyVars(~ ., data = train_data)
    train_data_transformed <- data.frame(predict(dummy, newdata = train_data))
    train_data_transformed[, target_variable] <- as.factor(train_data_transformed[, target_variable])
    
    inTrain <- createDataPartition(train_data_transformed[, target_variable], p = 0.8, list = FALSE)
    training <- train_data_transformed[inTrain,]
    testing <- train_data_transformed[-inTrain,]
    testing_features <- testing[, !colnames(testing) %in% target_variable]
    
    model <- glm(as.formula(paste(target_variable, "~ .")), data = training, family = "binomial")
    predictions <- predict(model, newdata = testing_features, type = "response")
    predictions_factor <- ifelse(predictions > 0.5, "1", "2")
    predictions_factor <- as.factor(predictions_factor)
    actual_values <- as.factor(testing[, target_variable]) 
    
    precision <- precision(predictions_factor, actual_values)
    recall<- recall(predictions_factor,actual_values)
    confusion_matrix <- confusionMatrix(predictions_factor, actual_values)
    f1_score <- confusion_matrix$byClass["F1"]
    list(
      model = model,
      model_summary = capture.output(print(summary(model))),
      precision = precision,
      recall=recall,
      f1_score=f1_score,
      confusionmatrix = confusion_matrix,
      coefficients = coef(model)
    )
  })
  
  rpart_model<- reactive({
    req(input$modeles)
    req(cleaned_data())
    target_variable <- input$target
    train_data <- cleaned_data()
    categorical_vars <- which(sapply(train_data, is.character) | sapply(train_data, is.factor))
    train_data[categorical_vars] <- lapply(train_data[categorical_vars], as.factor)
    
    dummy <- dummyVars(~ ., data = train_data)
    train_data_transformed <- data.frame(predict(dummy, newdata = train_data))
    train_data_transformed[, target_variable] <- as.factor(train_data_transformed[, target_variable])
    
    inTrain <- createDataPartition(train_data_transformed[, target_variable], p = 0.8, list = FALSE)
    training <- train_data_transformed[inTrain,]
    testing <- train_data_transformed[-inTrain,]
    testing_features <- testing[, !colnames(testing) %in% target_variable]
    model<- rpart(as.formula(paste(target_variable,"~ .")),data=training,method = 'class',parms = list(split = 'gini'))
    predictions <- predict(model, newdata = testing_features)
    
    actual_values <- testing[, target_variable]
    accuracy <- sum(predictions == actual_values) / length(actual_values)
    
    true_positives <- sum(predictions == 1 & actual_values == 1)
    false_positives <- sum(predictions == 1 & actual_values == 0)
    precision <- true_positives / (true_positives + false_positives)
    
    
    
    list(
      model=model,
      model_print=capture.output(print(model)),
      model_summary=capture.output(summary(model)),
      accuracy=accuracy,
      precision=precision
    )
    
  })
  
  output$tree_model<-renderPrint({
    if (!is.null(rpart_model())) {
      cat(rpart_model()$model_print,sep="\n")
    }
  })
  output$tree_plot <- renderPlot({
    if (!is.null(rpart_model())) {
      plot(rpart_model()$model,cex=0.5,main="Decision tree with rpart")
      text(rpart_model()$model,pretty=0.5)
    }
  })
  
  output$tree_model_summary<-renderPrint({
    if (!is.null(rpart_model())) {
      cat(rpart_model()$model_summary,sep="\n")
    }
  })
  
  
  
  
  #1
  output$model_summary <- renderPrint({
    if (!is.null(trained_model())) {
      cat(trained_model()$model_summary,sep="\n")
    }
  })
  output$result_model<-renderPrint({
    if (!is.null(trained_model())) {
      result<- paste(
        paste("Précision ",round(trained_model()$precision,digits = 2),sep = " = "),
        "\n",
        paste("Recall = ",round(trained_model()$precision,digits = 2)),
        "\n",
        paste("F1-SCORE = ",round(trained_model()$f1_score,digits = 2))
      )
      cat(result,sep="\n")
    }
  })
  output$confusion_matrix<-renderPrint({
    trained_model()$confusionmatrix
  })
  output$feature <- renderPrint({
    if (!is.null(trained_model())) {
      trained_model()$feature_importance
    }
  })
  
  output$feature_importance <- renderPlot({
    if (!is.null(trained_model())) {
      varImpPlot(trained_model()$model,main = "features importances")
    }
  })
  
  #2
  output$reg_model_summary<-renderPrint({
    if (!is.null(trained_model())) {
      cat(reg_trained_model()$model_summary,sep="\n")
    }
  })
  output$reg_result_model<-renderPrint({
    if (!is.null(reg_trained_model())) {
      result<- paste(
        paste("Précision ",round(reg_trained_model()$precision,digits = 2),sep = " = "),
        "\n",
        paste("Recall = ",round(reg_trained_model()$recall,digits = 2)),
        "\n",
        paste("F1-SCORE = ",round(reg_trained_model()$f1_score,digits = 2))
      )
      cat(result,sep="\n")
    }
  })
  output$reg_confusion_matrix<-renderPrint({
    reg_trained_model()$confusionmatrix
  })
  output$coefficients<-renderPrint({
    if (!is.null(reg_trained_model())) {
      reg_trained_model()$coefficients
    }
  })
  
  
  
  
  
  #affichage  
  output$NA_1 <- renderPrint({
    percent_missing1 <- colMeans(is.na(cleaned_data())) * 100
    print(percent_missing1)
  })
  
  output$data1_clean <- renderDataTable({
    cleaned_data()
  })
  
  
  
  
  output$contents <- renderDataTable({
    data()
  })
  
  output$Dimensions <- renderText({
    if (!is.null(data())) {
      paste("Number of rows:", nrow(data()), "\nNumber of columns:", ncol(data()))
    }
  })
  
  output$Dimensions1 <- renderText({
    if (!is.null(cleaned_data())) {
      paste("Number of rows:", nrow(cleaned_data()), "Number of columns:", ncol(cleaned_data()))
    }
  })
  
  output$save <- downloadHandler(
    filename = function() {
      paste("data_", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      write.csv(data(), file,fileEncoding = "UTF-8",row.names = FALSE)
    }
  )
  
  output$save1 <- downloadHandler(
    filename = function() {
      paste("data_cleaned_", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      write.csv(cleaned_data(), file,fileEncoding = "UTF-8",row.names = FALSE)
    }
  )
  
  output$class <- renderPrint({
    class(data())
  })
  
  output$summary <- renderPrint({
    summary(data())
  })
  
  output$summary2 <- renderPrint({
    summary(cleaned_data())
  })
}

shinyApp(ui = ui, server = server)