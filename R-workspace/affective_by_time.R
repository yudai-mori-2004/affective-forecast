suppressPackageStartupMessages({
  library(ggplot2)
})

in_csv <- "affective_by_continuous_time.csv"
setwd("C:/Users/mori/Programs/R-workspace")
dat <- read.csv(in_csv)

dat$arousal_jitter <- dat$arousal + runif(nrow(dat), -0.5, 0.5)
dat$valence_jitter <- dat$valence + runif(nrow(dat), -0.5, 0.5)

time_breaks <- seq(0, 1, by = 1/8)
time_labels <- c("0:00", "3:00", "6:00", "9:00", "12:00", "15:00", "18:00", "21:00", "24:00")

ggplot(dat, aes(x = time_continuous, y = arousal_jitter)) +
  geom_point(size = 0.5) +
  scale_x_continuous(
    breaks = time_breaks,
    labels = time_labels,
    limits = c(0, 1)
  ) +
  labs(
    x = "Time",
    y = "Arousal (±0.5 jitter)", 
    title = "Arousal by Time"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggplot(dat, aes(x = time_continuous, y = valence_jitter)) +
  geom_point(size = 0.5) +
  scale_x_continuous(
    breaks = time_breaks,
    labels = time_labels,
    limits = c(0, 1)
  ) +
  scale_y_continuous(breaks = function(x) seq(floor(min(x)), ceiling(max(x)), by = 1)) +
  labs(
    x = "Time",
    y = "Valence (±0.5 jitter)",
    title = "Valence by Time"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

