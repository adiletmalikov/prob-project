if(!require('readODS')) {
  install.packages('readODS')
  library('readODS')
}

names <- read_ods("data-2020/names.ods", sheet=2)

write.csv(names, "data-2020/names.csv", row.names=FALSE)
