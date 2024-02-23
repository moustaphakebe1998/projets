setwd("C:/Users/KEBE/Documents/A-DataScience/Dev-Web-R/Shiny/hepatitis")
getwd()
data<-read.csv("data_hep.csv",header=TRUE,sep=",")
new_data<-data.frame(
  Class=as.character(data$Class),
  AGE=as.integer(data$AGE),
  SEX=as.character(data$SEX),
  STEROID=as.character(data$STEROID),
  ANTIVIRALS=as.character(data$ANTIVIRALS),
  FATIGUE=as.character(data$FATIGUE),
  MALAISE=as.character(data$MALAISE),
  ANOREXIA=as.character(data$ANOREXIA),
  Liver.Big=as.character(data$LIVER.BIG),
  Liver.Firm=as.character(data$LIVER.FIRM),
  SPLEEN.PALPABLE=as.character(data$SPLEEN.PALPABLE),
  SPIDERS=as.character(data$SPIDERS),
  ASCITES=as.character(data$ASCITES),
  VARICES=as.character(data$VARICES),
  BILIRUBIN=data$BILIRUBIN,
  ALK.PHOSPHATE=as.integer(data$ALK.PHOSPHATE),
  SGOT=as.integer(data$SGOT),
  ALBUMIN=as.integer(data$ALBUMIN),
  PROTIME=as.integer(data$PROTIME),
  HISTOLOGY=as.integer(data$HISTOLOGY),
  stringsAsFactors = FALSE 
  )

write.csv(new_data,"naw_data_hepatitis.csv",row.names = FALSE)
df<-read.csv("naw_data_hepatitis.csv",header = TRUE,stringsAsFactors = TRUE)
str(df)
summary(df)
library(Factoshiny)
res.ca=Factoshiny(df)
