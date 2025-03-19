#! /usr/bin/env Rscript
if(!require("httr")) install.packages("httr")
library(httr)
library(jsonlite)

time_r<-format(Sys.time(),"%Y-%m-%dT%H-%M-%S+00:00")
reponse<-GET("https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-disponibilite-en-temps-reel/exports/json?lang=fr&timezone=Europe%2FBerlin")
reponse2<-GET("https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-disponibilite-en-temps-reel/exports/csv?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B")

get_content<-content(reponse,"text")
get_content2<-content(reponse2,"text")

file_json<-paste0("velib_R_",gsub("[:\\+]","_",time_r),".json")
fiel_csv<-paste0("velib_csv_",gsub("[:\\+]","_",time_r),".csv")

writeLines(get_content,file_json)
writeLine(get_content2,file_csv)
