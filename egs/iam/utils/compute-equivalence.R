#!/usr/bin/env Rscript
library(boot);
library(equivalence);

args = commandArgs(trailingOnly=TRUE)
if(length(args) != 2) {
  stop('Provide the errors files, computed with utils/compute-errors.sh')
}

df1 <- read.table(args[1], header=FALSE)
df2 <- read.table(args[2], header=FALSE)
df <- merge(as.data.frame(df1), as.data.frame(df2), by = 'V1')

f <- function(data, indices) {
  sum_err1 <- sum(data[indices, 'V2.x'])
  sum_sub1 <- sum(data[indices, 'V3.x'])
  sum_del1 <- sum(data[indices, 'V4.x'])
  sum_ins1 <- sum(data[indices, 'V5.x'])

  sum_err2 <- sum(data[indices, 'V2.y'])
  sum_sub2 <- sum(data[indices, 'V3.y'])
  sum_del2 <- sum(data[indices, 'V4.y'])
  sum_ins2 <- sum(data[indices, 'V5.y'])

  sum_len <- sum(data[indices, 'V6.x'])

  cer1 = sum_err1 / sum_len * 100
  cer2 = sum_err2 / sum_len * 100

  return (c(cer1 - cer2, cer1, cer2))

}

results <- boot(data=df, statistic=f, R=1000)
ci1 <- boot.ci(results, type = 'perc', conf = 0.90, index = 1)
ci2 <- boot.ci(results, type = 'perc', conf = 0.95, index = 2)
ci3 <- boot.ci(results, type = 'perc', conf = 0.95, index = 3)


sd(results$t[,1])
mean(results$t[,2])
results$t0
ci1$t0

ptte.stat(ci1$t0, sd(results$t[,1]), nrow(results$t), alpha = 0.05, Epsilon = 0.9)


cat(sprintf('%%ERR %.2f [%.2f -- %.2f]\n',
            ci1['t0'], ci1['percent'][[1]][4], ci1['percent'][[1]][5]))
cat(sprintf('%%ERR %.2f [%.2f -- %.2f]\n',
            ci2['t0'], ci2['percent'][[1]][4], ci2['percent'][[1]][5]))
cat(sprintf('%%ERR %.2f [%.2f -- %.2f]\n',
            ci3['t0'], ci3['percent'][[1]][4], ci3['percent'][[1]][5]))
