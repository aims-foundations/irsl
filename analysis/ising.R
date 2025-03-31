library("dplyr")
library("qgraph")
library("psychonetrics")

sapa <- read.csv("../data/deval_resmat.csv", header = TRUE)

network_sapa_4 <- psychonetrics::ggm(data = sapa,
                                     omega = "full", # or "empty" to set all elements to zero
                                     delta = "full", # or "empty" to set all elements to zero
                                     estimator = "FIML", # or "ML", "ULS", or "DWLS"
                                     verbose = TRUE) %>%

psychonetrics::runmodel() # Run the model

net <- getmatrix(network_sapa_4, "omega")

var_names <- colnames(sapa)
base_names <- sub("\\..*$", "", var_names)
unique_base_names <- unique(base_names)
colors <- rainbow(length(unique_base_names))
color_map <- setNames(colors, unique_base_names)
node_colors <- color_map[base_names]

png("../result/deval_network.png", width = 8, height = 8, units = "in", res = 300)
qgraph::qgraph(net, 
               layout = "spring", 
               theme = "colorblind",
               labels = var_names,
               color = node_colors)
dev.off()