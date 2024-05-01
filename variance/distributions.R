# Variance and Distributions

dist_hi_var <- data.frame(value = rnorm(n = 1000, mean = 50, sd = 15), group = "High Variance")
dist_low_var <- data.frame(value = rnorm(n = 1000, mean = 50, sd = 5), group = "Low Variance")

dist_data <- rbind(dist_hi_var, dist_low_var)

ggplot2::ggplot(data = dist_data, ggplot2::aes(x = value, fill = group)) +
  ggplot2::geom_density(alpha = 0.5) +
  ggplot2::geom_vline(xintercept = 50,
                 linetype = "dashed",
                 linewidth = 0.25) +
  ggplot2::labs(title = "Distributions with Different Variance",
                x = "Value",
                y = "Density",
                fill = NULL) +
  ggplot2::coord_cartesian(expand = FALSE,
                           ylim = c(0, 0.1),
                           xlim = c(0, 100)) +
  ggplot2::theme_bw() +
  ggplot2::theme(
    text = ggplot2::element_text(family = "Roboto"),
    # bold title
    plot.title = ggplot2::element_text(face = "bold"),
  )
