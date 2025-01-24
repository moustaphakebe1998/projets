library(shiny)
library(ggplot2)
library(shinydashboard)
library(bslib)
library(dplyr)
library(ggExtra)
library(bsicons)
library(ggcorrplot)
library(formattable)
library(caret)
library(randomForest)
library(randomForest)  # Assurez-vous que ce package est chargé
library(rpart)
library(DT)
library(e1071)
library(rpart.plot)



# Interface utilisateur
options(shiny.maxRequestSize = 100*1024^2)
ui <- dashboardPage(
  dashboardHeader(title = "Moustapha Kebe"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Charger Données", tabName = "upload", icon = icon("upload")),
      menuItem("Statistiques", tabName = "stats", icon = icon("info-circle")),
      menuItem("Configuration du modèle", tabName = "config", icon = icon("cogs")),
      menuItem("Résultats du modèle", tabName = "results", icon = icon("chart-bar")),
      menuItem("Prédiction", tabName = "predict", icon = icon("calculator"))
    )
  ),
  
  dashboardBody(
    tabItems(
      tabItem(tabName = "upload",
              value_box(
                title = "The current time",
                value = textOutput("time"),
                showcase = bs_icon("clock")
              ),
              fluidRow(
                column(12,
                       box(title = "Télécharger un fichier", status = "primary", solidHeader = TRUE, width = 12,
                           fileInput('file', 'Input file', accept = c(".csv", ".txt", "xls")),
                           selectInput("delimiter", "Séparateur", 
                                       choices = c("," = ",", ";" = ";", "\t" = "\t", " " = " "), 
                                       selected = ","),
                           checkboxInput("header", "Entête", value = TRUE),
                           selectInput("missing_values_action", "Action sur les valeurs manquantes", 
                                       choices = list("Aucune action" = "none", 
                                                      "Supprimer les lignes" = "remove", 
                                                      "Remplacer par médiane/mode" = "remplacer")),
                           actionButton("go", "Load", icon = icon("paper-plane"))
                       )
                )
              ),
              fluidRow(
                column(12,
                       box(title = "Aperçu des données", status = "info", solidHeader = TRUE, width = 12,
                           dataTableOutput("data_table"),
                           textOutput('dimension'),
                           downloadButton('download', "Télécharger")
                       )
                )
              )
      ),
      tabItem(tabName = "stats",
              h2("Statistiques Descriptives"),
              fluidRow(
                column(12,
                       box(title = "Résumé des données", status = "info", solidHeader = TRUE, width = 12,
                           verbatimTextOutput("summary", placeholder = TRUE)
                       )
                )
              ),
              fluidRow(
                column(6,
                       box(title = "Analyse univariée", status = "info", solidHeader = TRUE, width = 12,
                           uiOutput("quantlist"),
                           tableOutput("centreDisp"),
                           plotOutput("effectifsHist", width = "100%"),
                           fluidRow(
                             valueBoxOutput("mean_box"),
                             valueBoxOutput("median_box")
                           ),
                           plotOutput("effectifsCumCurve", width = "100%")
                       )
                ),
                column(6,
                       box(title = "Analyse bivariée", status = "info", solidHeader = TRUE, width = 12,
                           
                           # Sélections des variables quantitatives
                           fluidRow(
                             column(6, uiOutput("quantlistbi1")),
                             column(6, uiOutput("quantlistbi2"))
                           ),
                           
                           # Graphique du nuage de points
                           plotOutput("nuagePoints", width = "100%",height = "500px"),
                           
                           # Affichage du coefficient de corrélation
                           fluidRow(column(12, valueBoxOutput("correlationBox", width = "100%"))),
                           
                           # Tableau des statistiques descriptives avec formattable
                           #formattableOutput("summaryStatsTable", width = "100%")
                           div(style = 'height: 602px; overflow-y: scroll;',  # Augmenter la hauteur ici
                               formattableOutput("summaryStatsTable", width = "80%")
                           )
                       )
                )
              ),
              fluidRow(
                box(title = "Matrice des Corrélations", 
                    width = 12,
                    status = "primary", 
                    solidHeader = TRUE,
                    plotOutput("corrPlot", height = "900px")
                ))
      ),
      tabItem(tabName = "config",
              fluidPage(
                box(
                  title = "Configuration du modèle", 
                  status = "warning", 
                  solidHeader = TRUE, 
                  width = 12,
                  uiOutput("select_target"),
                  selectInput("modeles", "Sélectionnez un modèle", 
                              choices = list("Decision Tree" = "rpart",
                                             "Random Forest" = "rf", 
                                             "SVM" = "svm")),
                  sliderInput("ratio", "Ratio de partitionnement", min = 0.1, max = 0.9, value = 0.8),
                  selectInput("sampling_method", "Méthode de sampling", 
                              choices = list("Échantillonnage aléatoire" = "random", 
                                             "Échantillonnage stratifié" = "stratified")),
                  actionButton("train", "Entraîner le modèle", icon = icon("play")),
                  br(),
                  verbatimTextOutput("status_message"),  # Message d'état
                  uiOutput("progress")  # Indicateur de progression
                )
              )
      ),
      # Onglet des résultats du modèle
      tabItem(tabName = "results",
              fluidPage(
                box(
                  title = "Résumé du modèle", 
                  status = "success", 
                  solidHeader = TRUE, 
                  width = 12,
                  collapsible = TRUE,
                  verbatimTextOutput("model_summary")
                ),
                fluidRow(
                  box(
                    title = "Métriques de performance", 
                    status = "primary", 
                    solidHeader = TRUE, 
                    width = 6,
                    verbatimTextOutput("result_model")
                  ),
                  box(
                    title = "Matrice de confusion", 
                    status = "danger", 
                    solidHeader = TRUE, 
                    width = 6,
                    collapsible = TRUE,
                    verbatimTextOutput("confusion_matrix")
                  )
                ),
                fluidRow(
                  box(
                    title = "Importance des variables", 
                    status = "info", 
                    solidHeader = TRUE, 
                    width = 12,
                    plotOutput("feature_importance")
                  )
                )
              )
      ),
      tabItem(
        tabName = "predict",  # Correspond au menuItem de prédiction dans le sidebar
        fluidPage(
          titlePanel("Prédiction basée sur les valeurs saisies"),
          
          sidebarLayout(
            sidebarPanel(
              numericInput("AGE", "Âge:", value = 35, min = 1, max = 100),
              numericInput("SEX", "Sexe (1=Homme, 2=Femme):", value = 1, min = 1, max = 2),
              numericInput("STEROID", "Stéroïdes (1=Oui, 2=Non):", value = 2, min = 1, max = 2),
              numericInput("ANTIVIRALS", "Antiviraux (1=Oui, 2=Non):", value = 2, min = 1, max = 2),
              numericInput("FATIGUE", "Fatigue (1=Oui, 2=Non):", value = 1, min = 1, max = 2),
              numericInput("MALAISE", "Malaise (1=Oui, 2=Non):", value = 1, min = 1, max = 2),
              numericInput("ANOREXIA", "Anorexie (1=Oui, 2=Non):", value = 1, min = 1, max = 2),
              numericInput("LIVER.BIG", "Foie Gros (1=Oui, 2=Non):", value = 1, min = 1, max = 2),
              numericInput("LIVER.FIRM", "Foie Dur (1=Oui, 2=Non):", value = 1, min = 1, max = 2),
              numericInput("SPLEEN.PALPABLE", "Rate palpable (1=Oui, 2=Non):", value = 1, min = 1, max = 2),
              numericInput("SPIDERS", "Spiders (1=Oui, 2=Non):", value = 1, min = 1, max = 2),
              numericInput("ASCITES", "Ascite (1=Oui, 2=Non):", value = 1, min = 1, max = 2),
              numericInput("VARICES", "Varices (1=Oui, 2=Non):", value = 1, min = 1, max = 2),
              numericInput("BILIRUBIN", "Bilirubine:", value = 1.2, min = 0, max = 10),
              numericInput("ALK.PHOSPHATE", "Alk. Phosphate:", value = 85, min = 0, max = 200),
              numericInput("SGOT", "SGOT:", value = 35, min = 0, max = 500),
              numericInput("ALBUMIN", "Albumine:", value = 4.0, min = 0, max = 5),
              numericInput("PROTIME", "Temps de Prothrombine:", value = 80, min = 0, max = 100),
              numericInput("HISTOLOGY", "Histologie (1=Présente, 2=Absente):", value = 1, min = 1, max = 2),
              actionButton("predict", "Prédire", icon = icon("check"))
            ),
            
            mainPanel(
              valueBoxOutput("prediction_result") # Résultat de la prédiction
            )
          )
        )
      )
      
      
      
    )
  )
)

# Serveur
server <- function(input, output) {
  output$time <- renderText({
    invalidateLater(1000)
    format(Sys.time())
  })
  # Charger les données lorsque le bouton est cliqué
  data0 <- eventReactive(input$go, {
    req(input$file)  # Assurez-vous qu'un fichier est téléchargé
    df <- read.csv(input$file$datapath, header = input$header, sep = input$delimiter)
    df  # Retourner le dataframe
  })
  
  data <- reactive({
    req(data0())
    cleaned <- data0()
    
    if (input$missing_values_action == "remove") {
      cleaned <- cleaned[complete.cases(cleaned), ]
    } else if (input$missing_values_action == "remplacer") {
      cleaned <- cleaned %>%
        mutate_if(is.numeric, ~ifelse(is.na(.), median(., na.rm = TRUE), .)) %>%
        mutate_if(function(col) is.character(col) || is.factor(col),
                  ~ ifelse(is.na(.), names(sort(table(.[!is.na(.)]), decreasing = TRUE))[1], .))
    }
    
    return(cleaned)
  })
  
  
  # Afficher les données dans un tableau
  output$data_table <- renderDataTable({
    datatable(data(), options = list(pageLength = 10, autoWidth = TRUE, scrollX = TRUE))
  })
  
  # Afficher les dimensions des données
  output$dimension <- renderText({
    req(data())  # Assurez-vous que les données ne sont pas NULL
    paste("Lignes: ", nrow(data()), " Colonnes: ", ncol(data()))  # Affiche le nombre de lignes et colonnes
  })
  
  # Fonction de téléchargement des données
  output$download <- downloadHandler(
    filename = function() {
      paste("data_download", Sys.Date(), '.csv', sep = "")  # Nom du fichier avec la date
    },
    content = function(file) {
      write.csv(data(), file, fileEncoding = "UTF-8", row.names = FALSE)  # Écriture du fichier
    }
  )
  
  # Résumé des données
  output$summary <- renderPrint({
    req(data())
    summary(data())
  })
  
  
  ## Univariée - Choix de la variable
  
  output$quantlist = renderUI({
    selectInput('qnt', 'Le choix de la variable', names(data())[!grepl('factor|logical|character',sapply(data(),class))])
  })
  
  tabCentreDisp <- reactive({
    req(input$qnt)
    # Noms des caractéristiques
    dt =data()[,input$qnt]
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
  
  output$effectifsDiag <- renderPlot({ 
    req(data())
    req(input$qnt)
    dt = data()
    plot(table(data.frame(dt[,input$qnt])), col ="blue", xlab =sym(input$qnt), ylab ="Effectifs", 
         main ="Distribution des effectifs")
  })
  
  output$effectifsHist <- renderPlot({
    req(input$qnt)
    dt <- data()
    
    ggplot(dt, aes_string(x = input$qnt)) +
      geom_histogram(fill = "green", color = "black", bins = 30) +
      labs(title = paste("Histogramme de", input$qnt), x = input$qnt, y = "Effectifs") +
      theme_minimal()
  })
  
  
  output$effectifsCumCurve <- renderPlot({
    dt <- data()
    req(input$qnt)
    # Convertir la colonne choisie en numérique
    data_values <- as.numeric(as.character(dt[,input$qnt]))
    
    # Calcul des effectifs cumulés croissants et décroissants
    tmp_hist <- hist(data_values, plot = FALSE, right = FALSE)
    cum_counts <- cumsum(tmp_hist$counts)
    cum_counts_dec <- rev(cumsum(rev(tmp_hist$counts)))
    
    # Calcul de la moyenne et de la médiane
    mean_value <- mean(data_values, na.rm = TRUE)
    median_value <- median(data_values, na.rm = TRUE)
    
    # Créer un data frame pour les courbes
    df <- data.frame(
      breaks = tmp_hist$breaks[-1],  # Exclure le dernier intervalle
      cum_counts = cum_counts,
      cum_counts_dec = cum_counts_dec
    )
    
    # Graphique avec ggplot2
    ggplot(df, aes(x = breaks)) +
      # Courbe cumulative croissante
      geom_line(aes(y = cum_counts), color = "green", size = 1.5, linetype = "solid") +
      geom_point(aes(y = cum_counts), color = "darkgreen", size = 3) +
      
      # Courbe cumulative décroissante
      geom_line(aes(y = cum_counts_dec), color = "red", size = 1.5, linetype = "dashed") +
      geom_point(aes(y = cum_counts_dec), color = "darkred", size = 3) +
      
      # Ajouter la moyenne et la médiane avec des lignes verticales
      geom_vline(xintercept = median_value, linetype = "dotted", color = "purple", size = 1) +
      # Labels pour la médiane
      annotate("text", x = median_value, y = max(cum_counts), label = paste("Médiane= ",round(median_value, 1)),
               color = "purple", angle = 0, vjust = -0.2, size = 6) +
      
      # Ajustements des axes et titres
      labs(title = paste("Courbes cumulatives de", input$qnt),
           x = input$qnt, y = "Effectifs cumulés") +
      
      scale_y_continuous(sec.axis = dup_axis(name = "Effectifs cumulés décroissants")) +
      
      # Style général du graphique
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5, size = 18),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12),
        legend.position = "bottom"
      )
  })
  output$mean_box <- renderValueBox({
    req(input$qnt)
    
    dt <- data()
    x <- as.numeric(as.character(dt[,input$qnt]))
    
    mean_value <- mean(x, na.rm = TRUE)
    
    valueBox(
      round(mean_value, 2),
      subtitle = "Moyenne",
      icon = icon("calculator"),
      color = "blue"
    )
  })
  
  output$median_box <- renderValueBox({
    req(input$quantlistbi1)
    
    dt <- data()
    x <- as.numeric(as.character(dt[,input$qnt]))
    
    median_value <- median(x, na.rm = TRUE)
    
    valueBox(
      round(median_value, 2),
      subtitle = "Médiane",
      icon = icon("sort-numeric-up"),
      color = "green"
    )
  })
  
  
  
  
  ## Bivariée
  
  output$quantlistbi1 = renderUI({
    selectInput('quantlistbi1', 'Le choix de la variable X', names(data())[!grepl('factor|logical|character',sapply(data(),class))])
  })
  output$quantlistbi2 = renderUI({
    df2 <- data()[,!names(data()) %in% c(input$quantlistbi1)]
    selectInput('quantlistbi2', 'Le choix de la variable Y', names(df2)[!grepl('factor|logical|character',sapply(df2,class))])
  })
  
  output$nuagePoints <- renderPlot({
    req(input$quantlistbi1)
    req(input$quantlistbi2)
    dt <- data()
    
    ggplot(dt, aes_string(x = input$quantlistbi1, y = input$quantlistbi2)) +
      geom_point(color = "purple") +
      labs(
        title = paste("Nuage de points entre:", paste(c(input$quantlistbi1, input$quantlistbi2), collapse = " et ")),
        x = input$quantlistbi1,
        y = input$quantlistbi2
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5, size = 18),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12),
        legend.position = "bottom"
      )
  })
  
  
  # Render the correlation value box
  output$correlationBox <- renderValueBox({
    req(input$quantlistbi1)
    req(input$quantlistbi2)
    
    dt <- data()
    dt2 <- as.numeric(dt[, input$quantlistbi1])
    dt3 <- as.numeric(dt[, input$quantlistbi2])
    # Calculer le coefficient de corrélation
    coeff.tmp <- cor(dt2, dt3, use = "complete.obs")
    # Créer la boîte de valeur pour la corrélation
    valueBox(
      value = round(coeff.tmp, digits = 2),
      subtitle = "Coefficient de corrélation linéaire",
      icon = icon("bar-chart"),
      color = "purple"  # Choisis la couleur que tu préfères
    )
  })
  
  output$corrPlot <- renderPlot({
    dt <- data()
    
    # Sélectionner uniquement les colonnes numériques pour la matrice de corrélation
    numeric_columns <- dt[, sapply(dt, is.numeric)]
    
    # Calculer la matrice de corrélation
    corr_matrix <- cor(numeric_columns, use = "complete.obs")
    
    # Générer un graphique de corrélation avec des couleurs
    ggcorrplot(
      corr_matrix, 
      method = "circle",    # Utiliser des cercles pour représenter les corrélations
      hc.order = TRUE,      # Réorganisation des variables avec clustering hiérarchique
      type = "lower",       # Afficher uniquement le triangle inférieur
      lab = TRUE,           # Ajouter les coefficients de corrélation dans les cellules
      colors = c("#6D9EC1", "white", "#E46726"),  # Dégradé de couleurs
      title = "Matrice de Corrélation des Variables Numériques",
      lab_size = 4,         # Taille de police des coefficients
      ggtheme = theme_minimal()  # Thème minimaliste pour un affichage clair
    )
  })
  
  tabDisp <- reactive({
    req(input$quantlistbi1, input$quantlistbi2)  # Vérification des inputs
    
    # Obtenir les données des deux variables sélectionnées
    dt_x <- data()[, input$quantlistbi1]
    dt_y <- data()[, input$quantlistbi2]
    
    # Noms des statistiques descriptives
    stats_names <- c("Maximum", "Minimum", "Moyenne", "Médiane", 
                     "1er quartile", "3e quartile", "Variance", "Écart-type")
    
    # Calcul des statistiques pour X
    summary_x <- c(max(dt_x, na.rm = TRUE), 
                   min(dt_x, na.rm = TRUE), 
                   mean(dt_x, na.rm = TRUE), 
                   median(dt_x, na.rm = TRUE), 
                   quantile(dt_x, 0.25, na.rm = TRUE), 
                   quantile(dt_x, 0.75, na.rm = TRUE), 
                   var(dt_x, na.rm = TRUE), 
                   sd(dt_x, na.rm = TRUE))
    
    # Calcul des statistiques pour Y
    summary_y <- c(max(dt_y, na.rm = TRUE), 
                   min(dt_y, na.rm = TRUE), 
                   mean(dt_y, na.rm = TRUE), 
                   median(dt_y, na.rm = TRUE), 
                   quantile(dt_y, 0.25, na.rm = TRUE), 
                   quantile(dt_y, 0.75, na.rm = TRUE), 
                   var(dt_y, na.rm = TRUE), 
                   sd(dt_y, na.rm = TRUE))
    
    # Création de la table avec les deux variables, gardant "X" et "Y"
    summary_table <- data.frame(
      Caractéristique = stats_names,  # Noms des statistiques
      X = round(summary_x, 2),        # Résultats pour la première variable
      Y = round(summary_y, 2)         # Résultats pour la deuxième variable
    )
    
    # Retourner le tableau formaté avec des mini-barres colorées pour X et Y
    return(formattable(summary_table, 
                       list(
                         X = color_bar("lightblue"),
                         Y = color_bar("lightgreen")
                       )))
  })
  
  # Serveur : Afficher le tableau dans l'application Shiny
  output$summaryStatsTable <- renderFormattable({
    tabDisp()  # Appel de la fonction qui génère le tableau
  })
  
  
  #Modéles
  
  output$select_target <- renderUI({
    selectInput('target', 'Select Target', choices = names(data()), selected = NULL)
  })
  
  trained_model <- reactiveVal(NULL)
  
  # Sélection de la variable cible
  output$select_target <- renderUI({
    selectInput('target', 'Sélectionnez la variable cible', choices = names(data()), selected = NULL)
  })
  
  # Affichage des champs pour saisir les valeurs de prédiction
  output$prediction_input <- renderUI({
    req(input$target)
    features <- setdiff(names(data()), input$target)
    lapply(features, function(feature) {
      numericInput(feature, paste("Entrez la valeur pour", feature), value = NULL)
    })
  })
  
  observeEvent(input$train, {
    output$status_message <- renderText("Entraînement du modèle en cours...")
    
    output$progress <- renderUI({
      withProgress(message = 'Entraînement du modèle', value = 0, {
        for (i in 1:10) {
          Sys.sleep(0.5)
          incProgress(1/10)
        }
      })
    })
    
    showNotification("L'entraînement du modèle a commencé...", type = "message", duration = 5)
    
    model_info <- tryCatch({
      req(input$target, input$modeles)
      train_data <- data()
      target_variable <- input$target
      
      # Gérer les valeurs manquantes
      if (anyNA(train_data)) {
        numeric_cols <- sapply(train_data, is.numeric)
        train_data[numeric_cols] <- lapply(train_data[numeric_cols], function(x) {
          ifelse(is.na(x), mean(x, na.rm = TRUE), x)
        })
        
        categorical_cols <- sapply(train_data, is.factor)
        train_data[categorical_cols] <- lapply(train_data[categorical_cols], function(x) {
          x[is.na(x)] <- as.factor(names(sort(table(x), decreasing = TRUE))[1])
          return(x)
        })
      }
      
      is_classification <- input$modeles %in% c("rf", "rpart", "svm")
      
      if (is_classification) {
        train_data[[target_variable]] <- as.factor(train_data[[target_variable]])
      } else {
        train_data[[target_variable]] <- as.numeric(train_data[[target_variable]])
      }
      
      dummy <- dummyVars(~ ., data = train_data[, !colnames(train_data) %in% target_variable])
      train_data_transformed <- data.frame(predict(dummy, newdata = train_data))
      train_data_transformed[[target_variable]] <- train_data[[target_variable]]
      
      inTrain <- createDataPartition(train_data_transformed[[target_variable]], p = input$ratio, list = FALSE)
      training <- train_data_transformed[inTrain, ]
      testing <- train_data_transformed[-inTrain, ]
      testing_features <- testing[, !colnames(testing) %in% target_variable]
      
      model <- NULL
      if (input$modeles == "rf") {
        model <- randomForest(as.formula(paste(target_variable, "~ .")), data = training)
      } else if (input$modeles == "svm") {
        model <- svm(as.formula(paste(target_variable, "~ .")), data = training, probability = TRUE)
      } else if (input$modeles == "rpart") {
        method <- if (is_classification) "class" else "anova"
        model <- rpart(as.formula(paste(target_variable, "~ .")), data = training, method = method)
      }
      
      predictions <- if (is_classification) {
        predict(model, newdata = testing_features, type = "class")
      } else {
        predict(model, newdata = testing_features)
      }
      
      if (is_classification) {
        actual_values <- as.factor(testing[[target_variable]])
        confusion_matrix <- confusionMatrix(predictions, actual_values)
        accuracy <- confusion_matrix$overall["Accuracy"]
        precision <- caret::precision(predictions, actual_values)
        recall <- caret::recall(predictions, actual_values)
        f1_score <- confusion_matrix$byClass["F1"]
        
        list(
          model = model,
          model_summary = capture.output(print(summary(model))),
          confusionmatrix = confusion_matrix,
          accuracy = accuracy,
          precision = precision,
          recall = recall,
          f1_score = f1_score
        )
      } else {
        actual_values <- testing[[target_variable]]
        mse <- mean((predictions - actual_values)^2)
        rmse <- sqrt(mse)
        r_squared <- cor(predictions, actual_values)^2
        
        list(
          model = model,
          model_summary = capture.output(print(summary(model))),
          mse = mse,
          rmse = rmse,
          r_squared = r_squared
        )
      }
      
    }, error = function(e) {
      showNotification(paste("Erreur lors de l'entraînement :", e$message), type = "error", duration = 10)
      output$status_message <- renderText(paste("Erreur lors de l'entraînement :", e$message))
      NULL
    })
    
    if (!is.null(model_info)) {
      trained_model(model_info)  # Stocker le modèle entraîné dans la variable réactive
      output$status_message <- renderText("Modèle entraîné avec succès !")
      output$model_summary <- renderPrint({ cat(model_info$model_summary, sep = "\n") })
      
      if (input$modeles == "rpart") {
        output$feature_importance <- renderPlot({ rpart.plot(model_info$model, main = "Arbre de décision") })
      }
      
      if (input$modeles == "rf") {
        output$feature_importance <- renderPlot({ varImpPlot(model_info$model, main = "Importance des variables") })
      }
      
      if (is_classification) {
        output$result_model <- renderPrint({
          result <- paste(
            paste("Précision = ", round(model_info$precision, 2)),
            paste("Recall = ", round(model_info$recall, 2)),
            paste("F1-SCORE = ", round(model_info$f1_score, 2)),
            paste("Accuracy = ", round(model_info$accuracy, 2)),
            sep = "\n"
          )
          cat(result, sep = "\n")
        })
        output$confusion_matrix <- renderPrint({ model_info$confusionmatrix })
      } else {
        output$result_model <- renderPrint({
          result <- paste(
            paste("MSE = ", round(model_info$mse, 2)),
            paste("RMSE = ", round(model_info$rmse, 2)),
            paste("R² = ", round(model_info$r_squared, 2)),
            sep = "\n"
          )
          cat(result, sep = "\n")
        })
      }
    }
  })
  
  
  observeEvent(input$predict, {
    req(trained_model())  # S'assurer qu'un modèle a bien été entraîné avant la prédiction
    
    new_data <- data.frame(
      AGE = input$AGE,
      SEX = input$SEX,
      STEROID = input$STEROID,
      ANTIVIRALS = input$ANTIVIRALS,
      FATIGUE = input$FATIGUE,
      MALAISE = input$MALAISE,
      ANOREXIA = input$ANOREXIA,
      LIVER.BIG = input$LIVER.BIG,
      LIVER.FIRM = input$LIVER.FIRM,
      SPLEEN.PALPABLE = input$SPLEEN.PALPABLE,
      SPIDERS = input$SPIDERS,
      ASCITES = input$ASCITES,
      VARICES = input$VARICES,
      BILIRUBIN = input$BILIRUBIN,
      ALK.PHOSPHATE = input$ALK.PHOSPHATE,
      SGOT = input$SGOT,
      ALBUMIN = input$ALBUMIN,
      PROTIME = input$PROTIME,
      HISTOLOGY = input$HISTOLOGY
    )
    
    # Effectuer la prédiction
    prediction <- tryCatch({
      if (input$modeles == "svm") {
        pred_probs <- predict(trained_model()$model, newdata = new_data, probability = TRUE)
        pred_class <- ifelse(attr(pred_probs, "probabilities")[,1] > 0.5, 1, 2)  # Si probabilité > 0.5, classe 1, sinon classe 2
        pred_class
      } else if (input$modeles %in% c("rf", "rpart")) {
        predict(trained_model()$model, newdata = new_data, type = "class")
      } else {
        predict(trained_model()$model, newdata = new_data)
      }
    }, error = function(e) {
      showNotification(paste("Erreur lors de la prédiction :", e$message), type = "error", duration = 10)
      NULL
    })
    
    # Afficher le résultat de la prédiction
    output$prediction_result <- renderValueBox({
      if (!is.null(prediction)) {
        valueBox(
          value = prediction,  # La prédiction obtenue
          subtitle = "Résultat de la prédiction",  # Titre pour l'affichage
          color = "blue",  # Couleur du valueBox
          icon = icon("chart-line")  # Icône à afficher (facultatif)
        )
      } else {
        valueBox(
          value = "Erreur",
          subtitle = "Erreur lors de la prédiction",
          color = "red",
          icon = icon("exclamation-triangle")
        )
      }
    })
    
  })
  
  
  
  
  
  
}

# Lancer l'application Shiny
shinyApp(ui, server)
