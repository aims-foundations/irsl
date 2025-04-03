library("IsingSampler")
library("Rcpp")

N <- 7126 # Number of nodes
nSample <- 84 # Number of samples
# Ising parameters:
Graph <- matrix(sample(0:1,N^2,TRUE,prob = c(0.7, 0.3)),N,N) * rnorm(N^2)
Graph <- pmax(Graph,t(Graph)) / N
diag(Graph) <- 0
Thresh <- -(rnorm(N)^2)
Beta <- 1
# Response options (0,1 or -1,1):
Resp <- c(0L,1L)
# Simulate with metropolis:
MetData <- IsingSampler(nSample, Graph, Thresh, Beta, 1000/N, responses = Resp, method = "MH")

write.csv(MetData, file = "MetData.csv", row.names = FALSE)
write.csv(Graph, file = "IsingGraph.csv", row.names = FALSE)
write.csv(data.frame(Thresh), file = "Thresholds.csv", row.names = FALSE)