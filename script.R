if (!require(xlsx)) {
  install.packages("xlsx")
}
library(xlsx)

pathes_to_grades <- lapply(6:11, function(grade) {
  path_template <- "data-2020/%d-grade.xlsx"
  path <- sprintf(path_template, grade)
  return(path)
})

data <- lapply(pathes_to_grades, function(path) {
  read.xlsx(path, 1, colNames = TRUE)
})

grade6 <- c()
sum <- 0.0

for (n in data[[1]][[17]]) {
  grade6 <- append(n, grade6)
  sum <- sum + n
}

grade6
