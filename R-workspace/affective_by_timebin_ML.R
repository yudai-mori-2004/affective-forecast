suppressPackageStartupMessages({
  library(lme4) 
  library(jsonlite)
  library(ggplot2)
})


in_csv   <- "va_long.csv"
out_json <- "va_lmm.json"

setwd("/Users/forest/Research/R-workspace")
dat <- read.csv(in_csv)


all_bins <- c(
  "00:00-03:00", "03:00-06:00", "06:00-09:00",
  "09:00-12:00", "12:00-15:00", "15:00-18:00",
  "18:00-21:00", "21:00-00:00"
)


dat$subject_id <- factor(dat$subject_id) 
dat$time_bin   <- factor(dat$time_bin, levels = all_bins) 

aro_stats <- list()
val_stats <- list()

for(bin in all_bins){
  range_dat <- subset(dat, time_bin == bin)
  
  # ① ある時間帯における感情レベルの「全体平均+個人のランダム効果」とするLMMモデル
  arousal_model <- lmer(arousal ~ 1 + (1|subject_id), range_dat, REML = FALSE)
  valence_model <- lmer(valence ~ 1 + (1|subject_id), range_dat, REML = FALSE)
  
  aro_stats[[bin]] <- c(
    mean = unname(fixef(arousal_model)["(Intercept)"]),
    sd   = sigma(arousal_model)
  )
  val_stats[[bin]] <- c(
    mean = unname(fixef(valence_model)["(Intercept)"]),
    sd   = sigma(valence_model)
  )
}


print(aro_stats)
print(val_stats)


aro_df <- data.frame(
  time_bin = factor(names(aro_stats), levels = all_bins),
  mean     = as.numeric(sapply(aro_stats, `[`, "mean")),
  sd       = as.numeric(sapply(aro_stats, `[`, "sd"))
)

val_df <- data.frame(
  time_bin = factor(names(aro_stats), levels = all_bins),
  mean     = as.numeric(sapply(val_stats, `[`, "mean")),
  sd       = as.numeric(sapply(val_stats, `[`, "sd"))
)


ggplot(aro_df, aes(x = time_bin, y = mean, group = 1)) +
  geom_line() +                            # 線
  geom_point(size = 3) +                   # 点
  geom_errorbar(                           # エラーバー
    aes(ymin = mean - sd, ymax = mean + sd),
    width = 0.1
  ) +
  labs(
    x     = "Time Bin",
    y     = "Mean Arousal",
    title = "Arousal ± SD by Time Bin (ML)"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


ggplot(val_df, aes(x = time_bin, y = mean, group = 1)) +
  geom_line() +                            # 線
  geom_point(size = 3) +                   # 点
  geom_errorbar(                           # エラーバー
    aes(ymin = mean - sd, ymax = mean + sd),
    width = 0.1
  ) +
  labs(
    x     = "Time Bin",
    y     = "Mean Valence",
    title = "Valence ± SD by Time Bin  (ML)"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

