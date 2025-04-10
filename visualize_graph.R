library("dplyr")
library("qgraph")
library("jsonlite")

# Obtain the network plot matrix from your model
ggm_mat <- read.csv("W.csv", header = TRUE)

# Column names
var_names <- colnames(ggm_mat)

# Step 1: Read the JSON mapping var_names -> z
z_mapping <- jsonlite::fromJSON("diffeasy_to_z_997.json")
# Ensure that z values are extracted in the order of var_names
z_vals <- sapply(var_names, function(x) z_mapping[[x]])

# Step 2: Define a function to "darken" a base color.
darken_color <- function(color, darkness) {
  # Get the base color's RGB values (range 0-255)
  rgb_vals <- col2rgb(color)
  # Multiply each channel by (1 - darkness): darkness==0 returns full color; darkness==1 returns black.
  new_vals <- rgb_vals * (1 - darkness)
  # Convert the new RGB values back to a hexadecimal color string.
  new_color <- rgb(new_vals[1], new_vals[2], new_vals[3], maxColorValue = 255)
  return(new_color)
}

# Step 3: Assign a single base color "green" to all nodes.
base_color <- rep("green", length(var_names))

# Step 4: Compute a darkness factor from the actual z values.
min_z <- min(z_vals)
max_z <- max(z_vals)
# Normalize z values so that the minimum is 0 and the maximum is 1.
normalized_darkness <- (z_vals - min_z) / (max_z - min_z)
# Scale the darkness to range between 0.3 and 0.7:
scaled_darkness <- 0.3 + normalized_darkness * 0.4

# Generate the final node colors by applying the darkening to the base color.
node_colors <- mapply(darken_color, base_color, scaled_darkness)

# Plot saving and layout definition: one large panel for the network, one for the single color bar.
png("network_gsm.png", width = 10, height = 8, units = "in", res = 300)
layout(matrix(c(1,2), ncol = 2, byrow = TRUE), widths = c(0.8, 0.2))

# Panel 1: Plot the network with qgraph.
qgraph::qgraph(
  ggm_mat, 
  layout = "spring", 
  theme = "colorblind",
  labels = var_names,
  color = node_colors
)

# Panel 2: Plot a single color bar for the green scale.
par(mar = c(5, 4, 4, 2) + 0.1)  # Set margins for the panel
# Create a sequence of z values for the color bar.
seq_z <- seq(min_z, max_z, length.out = 100)
# Compute the corresponding scaled darkness values.
seq_normalized_darkness <- (seq_z - min_z) / (max_z - min_z)
seq_scaled_darkness <- 0.3 + seq_normalized_darkness * 0.4
# Generate the color bar colors based on green.
colorbar_green <- sapply(seq_scaled_darkness, function(dark) darken_color("green", dark))
# Plot the color bar.
image(x = 1, y = seq_z, z = matrix(seq_z, nrow = 1),
      col = colorbar_green, xlab = "", ylab = "z", axes = FALSE)
axis(2, at = seq(min_z, max_z, length.out = 5), 
     labels = round(seq(min_z, max_z, length.out = 5), 2))
title("z")

dev.off()