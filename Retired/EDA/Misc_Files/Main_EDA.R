## ----setup, include=FALSE-----------------------------------------------------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = T, warning = FALSE)
options(java.parameters = c("-XX:+UseConcMarkSweepGC", "-Xmx8192m"))
#options(java.parameters = "-Xmx8g")


## ---- echo=F, message=FALSE, warning=FALSE------------------------------------------------------------------------------------------------------------------------------------------------
# Base Packages
library(dplyr) # Additional Base level functionality of R 
library(stats) # Base Statistic Funtionality

# Extended Base Functionality
library(reshape2) # Melt Function Correlation Matrix
library(rlist) #list.append
library(lubridate) # For working with dates
library(caret) # MachineLearning package used for createDataPartition

# Visualisation Packages
library(ggplot2) # Comprehensive Graphical Output Package 
library(ggthemes) # More themes for GGPLOT2
library(scales) # Control over Axis formatting
library(gghighlight) # GGplot Highlight function
library(gridExtra) # Plot in Grids

# Latex Packages
library(xtable) # Latex Tables
library(papeR) # Paper Writing Module
library(reporttools) # Latex Sumarry Tables


## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
download.file(
  "https://github.com/Kydoimos97/CapstoneMSBA2020/raw/main/Data/CapstoneProjectInfoRevised.rds", 
  destfile = "CapstoneProjectInfoRevised.rds")

download.file(
  "https://github.com/Kydoimos97/CapstoneMSBA2020/raw/main/Data/CapstoneProjectProducts.rds", 
  destfile= "CapstoneProjectProducts.rds")

df <- readRDS("CapstoneProjectInfoRevised.rds")
products <- readRDS("CapstoneProjectProducts.rds")


## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Factorization
df$Site_ID <- as.factor(df$Site_ID)
df$Location_ID <- as.factor(df$Location_ID)
df$Locale <- as.factor(df$Locale)
df$Fiscal_Period <- as.factor(df$Fiscal_Period)
df$MPDS <- as.factor(df$MPDS)
df$Project <- as.factor(df$Project)

# Numeration
df$Quantity_Sold <- as.numeric(df$Quantity_Sold)
df$SQ_Footage <- as.numeric(df$SQ_Footage)# Shows Factor like tendencies
df$Periodic_GBV <- as.numeric(df$Periodic_GBV)
df$Current_GBV <- as.numeric(df$Current_GBV)

# Re-origin of Dates at 06/22/1998

x <- min(df$Open_Date) 
df$Open_Date <- as_date(df$Open_Date, origin = x)
df$DATE <- as_date(df$DATE, origin = x)

rm(x)

# enables DATE to be used in prediction algorithms



## ---- warning=FALSE, message=FALSE--------------------------------------------------------------------------------------------------------------------------------------------------------
candy_vector <- c(products$Item_Desc)

id_vector <- c(products$Item_ID)

df$Item_desc <- df$Item_ID
df$Item_desc <- plyr::mapvalues(df$Item_desc, id_vector, candy_vector)

df$Item_ID <- as.factor(df$Item_ID)

# Reodering Data
df <- df[, c(1,2,6,17,7,3,12,13,14,10,11,8,9,15,16,4,5)]

#Rename Variables
names(df) <- tolower(make.names(names(df)))

rm(products, candy_vector, id_vector)



## ---- results="asis"----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# summarize numeric variables
tableContinuous(df[,sapply(df, is.numeric)], comment = FALSE,
                stats = c("n", "min", "q1", "median", "mean", "q3", "max", "na"),
                cap = "Summary of Numeric variables")


## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
new_df <- df[is.na(df$maxtemp),]


## ---- eval=F------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## levels(as.factor(new_df$site_id))
## 
## levels(as.factor(new_df$date))


## ---- results="asis"----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# summarize factorized variables
tableNominal(df[c(12,13,14,15)], cap = "Summary of Factorized variables", cumsum = FALSE,
             comment = FALSE)



## ---- fig.cap='Pre-engineering, Correlation Matrix'---------------------------------------------------------------------------------------------------------------------------------------
df2 <- df
df2$open_date <- as.numeric(df2$open_date)
df2$date <- as.numeric(df2$date)

nums <- unlist(lapply(df2, is.numeric))
Integers <- na.omit(df2[ , nums]) # Remove NA's due to temp

melted_cormat <- melt(cor(Integers))

names(melted_cormat)[3] <- "Correlation"

melted_cormat <- melted_cormat %>% 
    arrange(Var1) %>%
    group_by(Var1) %>%
    filter(row_number() >= which(Var2 == Var1))

ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=Correlation)) + 
  geom_tile(aes(fill=Correlation), col = "Black") + 
  theme_fivethirtyeight () +
  theme(legend.position="right",legend.direction = "vertical",
        legend.text = element_text(size=7)) +
  geom_text(label = round(melted_cormat$Correlation, digits = 2),
            color="Black", size=3, alpha=0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust=1, size=10),
        axis.text.y = element_text(angle=0,vjust = 0.5, hjust=1, size=10)) +
  scale_fill_gradient2_tableau("Classic Orange-White-Blue Light", limits = c(-1,1)) +
  theme(axis.title = element_text()) + 
  ylab("Feature") + 
  xlab("Feature") + 
  labs(title = "Correlation Matrix")
  

rm(Integers, melted_cormat, nums, df2)


## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# creation of tempdiff
df$temp_diff <- as.numeric(df$maxtemp-df$mintemp)

# Creation of cgbv_sqf
df$cgbv_sqf <- as.numeric(df$current_gbv/df$sq_footage)

# creation of diff_gbv
df$diff_gbv <- df$current_gbv - df$periodic_gbv

#Days open
x <- min(df$open_date) 
df$days_open <- as.numeric(df$date-df$open_date)

rm(x)

df <- df[, c(1,2,3,4,21,6,7,19,20,10,18,12,13,14,15,16,17,8,9,11,5)]

df <- df[,-c(12,18,19,20,21)]


## ---- fig.cap='Post-engineering, Correlation Matrix'--------------------------------------------------------------------------------------------------------------------------------------
df2 <- df
df2$date <- as.numeric(df2$date)

nums <- unlist(lapply(df2, is.numeric))
Integers <- na.omit(df2[ , nums]) # Remove NA's due to temp

melted_cormat <- melt(cor(Integers))

names(melted_cormat)[3] <- "Correlation"

melted_cormat <- melted_cormat %>% 
    arrange(Var1) %>%
    group_by(Var1) %>%
    filter(row_number() >= which(Var2 == Var1))

ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=Correlation)) + 
  geom_tile(aes(fill=Correlation), col = "Black") + 
  theme_fivethirtyeight () +
  theme(legend.position="right",legend.direction = "vertical",
        legend.text = element_text(size=7)) +
  geom_text(label = round(melted_cormat$Correlation, digits = 2),
            color="Black", size=3, alpha=0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust=1, size=10),
        axis.text.y = element_text(angle=0,vjust = 0.5, hjust=1, size=10)) +
  scale_fill_gradient2_tableau("Classic Orange-White-Blue Light", limits = c(-1,1)) +
  theme(axis.title = element_text()) + 
  ylab("Feature") + 
  xlab("Feature") + 
  labs(title = "Correlation Matrix")
  

rm(Integers, melted_cormat, nums, df2)


## ---- fig.height = 9, fig.width = 7, fig.cap='Relationships between Target and Numeric Variables seperated by Site_ID', warning=FALSE, message=FALSE--------------------------------------
num_list <- colnames(df[sapply(df, is.numeric)])
num_list <- num_list[-c(4,5,6,7)]
plot_list <- list(1)

counter = 0

resample <- caret::createDataPartition(y=df$sales, p = 0.01, list=FALSE) 
#This is due to a limitation on PDF's
df2 <-df[resample,]

mycolors <- grDevices::colorRampPalette(RColorBrewer::brewer.pal(8, "Set1"))(10)

for (i in num_list) {
  
  counter = counter+1
  x = ggplot(data=na.omit(df2[df2$sales < 100,]), aes_string(x = "sales", y = i, color="site_id")) +
      geom_point(position = "identity", stat = "identity", alpha = .35, size=1, stroke=.05) + 
      theme_fivethirtyeight() + 
      scale_color_manual(values = mycolors) +
      scale_y_continuous(labels = comma) + scale_x_continuous(labels = dollar) + coord_flip() + 
      ggtitle(paste0("sales vs ", as.character(i), sep= " ")) + 
      theme(plot.title = element_text(size=10), legend.position = "right", legend.direction = "vertical") + 
      guides(color = guide_legend(override.aes = list(size=2.5, alpha = 1) ,title = "Legend", title.position = "top"))
  
  
  assign(paste0("plt",i), x)
  plot_list <- rlist::list.append(plot_list, x)
}

plot_list <- plot_list[-1]

nCol <- floor(sqrt(length(plot_list)))

grid.arrange(grobs=plot_list, widths = c(1,1), ncol=2, layout_matrix =  rbind(c(1,1),
                                                                                 c(2,2), 
                                                                                 c(3,3)), 
             top = "Relationships between Target and Numeric Variables seperated by Site_ID")

for (i in num_list) {
  rm(list = paste0("plt",i))
}

rm(plot_list, counter, num_list, x, i, nCol, df2, mycolors, resample)


## ---- fig.height = 9, fig.width = 7, fig.cap='Relationships between Target and Numeric Variables', warning=FALSE, message=FALSE-----------------------------------------------------------
num_list <- colnames(df[sapply(df, is.numeric)])
num_list <- num_list[-c(1,2,3,7)]
plot_list <- list(1)

counter = 0
colcounter = 0

resample <- caret::createDataPartition(y=df$sales, p = 0.01, list=FALSE)
df2 <-df[resample,]

mycolors <- grDevices::colorRampPalette(RColorBrewer::brewer.pal(8, "Set1"))(10)

for (i in num_list) {
  
  counter = counter+1
  colcounter = counter*3
  x = ggplot(data=na.omit(df2[df2$sales < 150,]), aes_string(x = "sales", y = i)) +
      geom_point(position = "identity", stat = "identity", alpha = .75, color="grey", size=0.75)  + 
      scale_y_continuous(labels = comma) + scale_x_continuous(labels= dollar) + 
      geom_smooth(method = lm, se=TRUE, color=mycolors[colcounter], size=1, fullrange = TRUE) + 
      ggtitle(paste0("sales vs ", as.character(i), sep= " ")) +
      theme_fivethirtyeight() + theme(plot.title = element_text(size=10)) 
  
  
  assign(paste0("plt",i), x)
  plot_list <- rlist::list.append(plot_list, x)
}

plot_list <- plot_list[-1]

nCol <- floor(sqrt(length(plot_list)))

grid.arrange(grobs=plot_list, widths = c(1,1), ncol=2, layout_matrix =  rbind(c(1,1),
                                                                                 c(2,2), 
                                                                                 c(3,3)), 
             top = "Relationships between Target and Numeric Variables")


for (i in num_list) {
  rm(list = paste0("plt",i))
}

rm(plot_list, counter, num_list, x, i, nCol, df2, resample, mycolors, colcounter)


## ---- results='asis', warning=FALSE, message=FALSE----------------------------------------------------------------------------------------------------------------------------------------
table <- top_n(df[c(1,2,4,15,16)], 5)

xtable(table, caption="Top 5 Sales values in the data set", comment=FALSE)


## ---- eval=TRUE, fig.height =3, fig.width =5, fig.align = "center", results="asis", fig.cap='Relationship between Sales and Locale'-------------------------------------------------------
i = "locale"
o <- c(summary(df$locale))

  x = ggplot(data=df, aes_string(x = "sales", y = paste0("`",as.character(i),"`", sep=""))) +
  geom_boxplot() + 
  ggtitle(paste0("Relationship of sales and ", as.character(i), sep= " ")) +
  coord_flip() + 
  theme_fivethirtyeight() + theme(axis.text.x = element_text(angle =0)) + 
  scale_y_discrete(guide = guide_axis(n.dodge=2))+
  theme(plot.title = element_text(size = 10, face = "bold")) + 
  scale_x_continuous(labels = dollar, limits = c(0,15))
  print(x)


## ---- eval=TRUE, fig.height =3, fig.width =5, fig.align = "center", results="asis"--------------------------------------------------------------------------------------------------------
y <- as.data.frame(ggplot_build(x)$data)
  y <- y[,c(1,2,3,4,5,6)]
  y$xmax <- y$xmax-y$xmin
  y$xmin <- levels(df[,which( colnames(df)==i)])
  y$outliers <- ifelse(
    lengths(regmatches(as.character(y$outliers), gregexpr(",", as.character(y$outliers)))) != 0, 
    lengths(regmatches(as.character(y$outliers), gregexpr(",", as.character(y$outliers)))) + 1,
    ifelse(regmatches(as.character(y$outliers), gregexpr("[0-9]", as.character(y$outliers))) == "0", 
    NA, 1))
  y$sample_size <- o
  names(y) <- c("Factor Levels", "25Q", "Median", "75Q", "1.5XIQ", "# Outliers", "Ns")
  y <- y[order(abs(y$Median), decreasing = TRUE),]

print(xtable(y, caption="Relationship between Sales and Locale between different factor levels"), 
      comment = FALSE)



## ---- eval=TRUE, fig.height =3, fig.width =5, fig.align = "center", results="asis", fig.cap='Relationship between Sales and MPDS'---------------------------------------------------------
i = "mpds"
o <- c(summary(df$mpds))

  x = ggplot(data=df, aes_string(x = "sales", y = paste0("`",as.character(i),"`", sep=""))) +
  geom_boxplot() +   ggtitle(paste0("Relationship of sales and ", as.character(i), sep= " ")) +
  coord_flip() + 
  theme_fivethirtyeight() + 
  theme(axis.text.x = element_text(angle =0)) + 
  scale_y_discrete(guide = guide_axis(n.dodge=1))+
  theme(plot.title = element_text(size = 10, face = "bold")) + 
  scale_x_continuous(labels = dollar, limits = c(0,15))
  print(x)


## ---- eval=TRUE, fig.height =3, fig.width =5, fig.align = "center", results="asis"--------------------------------------------------------------------------------------------------------
y <- as.data.frame(ggplot_build(x)$data)
  y <- y[,c(1,2,3,4,5,6)]
  y$xmax <- y$xmax-y$xmin
  y$xmin <- levels(df[,which( colnames(df)==i)])
  y$outliers <- ifelse(
    lengths(regmatches(as.character(y$outliers), gregexpr(",", as.character(y$outliers)))) != 0, 
    lengths(regmatches(as.character(y$outliers), gregexpr(",", as.character(y$outliers)))) + 1,
    ifelse(regmatches(as.character(y$outliers), gregexpr("[0-9]", as.character(y$outliers))) == "0", 
    NA, 1))
  y$sample_size <- o
  names(y) <- c("Factor Levels", "25Q", "Median", "75Q", "1.5XIQ", "# Outliers", "Ns")
  y <- y[order(abs(y$Median), decreasing = TRUE),]

print(xtable(y, caption="Relationship between Sales and MPDS between different factor levels"), comment = FALSE)



## ---- eval=TRUE, fig.height =3, fig.width =5, fig.align = "center", results="asis", fig.cap='Relationship between Sales and Project'------------------------------------------------------
i = "project"
o <- c(summary(df$project))

  x = ggplot(data=df, aes_string(x = "sales", y = paste0("`",as.character(i),"`", sep=""))) +
  geom_boxplot() + 
  ggtitle(paste0("Relationship of sales and ", as.character(i), sep= " ")) +
  coord_flip() + 
  theme_fivethirtyeight() + 
  theme(axis.text.x = element_text(angle =0)) + 
  scale_y_discrete(guide = guide_axis(n.dodge=2)) +
  theme(plot.title = element_text(size = 10, face = "bold")) + 
  scale_x_continuous(labels = dollar, limits = c(0,15))
  print(x)


## ---- eval=TRUE, fig.height =3, fig.width =5, fig.align = "center", results="asis"--------------------------------------------------------------------------------------------------------
y <- as.data.frame(ggplot_build(x)$data)
  y <- y[,c(1,2,3,4,5,6)]
  y$xmax <- y$xmax-y$xmin
  y$xmin <- levels(df[,which( colnames(df)==i)])
  y$outliers <- ifelse(
    lengths(regmatches(as.character(y$outliers), gregexpr(",", as.character(y$outliers)))) != 0, 
    lengths(regmatches(as.character(y$outliers), gregexpr(",", as.character(y$outliers)))) + 1,
    ifelse(regmatches(as.character(y$outliers), gregexpr("[0-9]", as.character(y$outliers))) == "0", 
    NA, 1))
  y$sample_size <- o
  names(y) <- c("Factor Levels", "25Q", "Median", "75Q", "1.5XIQ", "# Outliers", "Ns")
  y <- y[order(abs(y$Median), decreasing = TRUE),]

print(xtable(y, caption="Relationship between Sales and Project between different factor levels"),
             comment = FALSE)

  
rm(i, o, y, x)


## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
sum_sales <- aggregate(df$sales, by=list(df$date, df$site_id), FUN=sum)
names(sum_sales) <- c("Date", "Site_ID", "Sum_of_Sales")


## ---- fig.height = 9, fig.width = 7, fig.cap='Sales over time', warning=FALSE, message=FALSE----------------------------------------------------------------------------------------------
mycolors <- grDevices::colorRampPalette(RColorBrewer::brewer.pal(8, "Set1"))(10)

ggplot(sum_sales, aes(x=Date, y=Sum_of_Sales, color=Site_ID)) +
  facet_wrap(~Site_ID, scale="free_y", nrow = 5, ncol = 2) +
  geom_point(position = "identity", stat = "identity", alpha = .8, color="grey", size=0.75) +
  geom_smooth(se=F) +
  theme_fivethirtyeight() +
  scale_color_manual(values = mycolors) +
  scale_y_continuous(labels = scales::dollar, limits = c(0,1500)) +
  theme(legend.position = ("none"), strip.text = element_text(size=15), plot.title = element_text(size=22),axis.title.x = element_text(size = 15), axis.title.y =     element_text(size = 15)) +
  labs(x = "Site_ID", y = "Occupation Growth in percentage", title = "Total Sales per Site_ID ")



## ---- results='asis'----------------------------------------------------------------------------------------------------------------------------------------------------------------------
counter = 0

list_siteid <- NULL
list_mpds <- NULL
df2 <- NULL

for (i in c(levels(df$site_id))) {
   counter = counter+1
   list_siteid[counter] <- i
   df2 <- df %>% filter(site_id == i)
   list_mpds[counter] <- mean(as.numeric(as.character(df2$mpds)))
  
}

#Latex does not work nice with _ so site_id Site_ID due to results= asis
table2 <-data.frame(SiteID = list_siteid, MPDS = list_mpds)




## ---- results='asis', warning=FALSE, message=FALSE----------------------------------------------------------------------------------------------------------------------------------------

print.xtable(xtable(table2, caption = "Amount of Pumps per SiteID"), comment=FALSE)

rm(table2, list_mpds, list_siteid, i, counter, df2)


## ---- results='asis', echo=FALSE----------------------------------------------------------------------------------------------------------------------------------------------------------

cat(paste("\\item", c(levels(as.factor(new_df$site_id))), sep = "\n"))



## ---- results='asis', echo=FALSE----------------------------------------------------------------------------------------------------------------------------------------------------------

cat(paste("\\item", c(levels(as.factor(new_df$date))), sep = "\n"))


