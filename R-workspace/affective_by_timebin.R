suppressPackageStartupMessages({
  library(lme4) 
  library(jsonlite)
  library(ggplot2)
  library(emmeans)
})


in_csv   <- "affective_by_discrete_time.csv"

setwd("C:/Users/mori/Programs/R-workspace")
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

summary_to_df <- function(model) {
  emms <- emmeans(model, ~ time_bin)
  df <- as.data.frame(emms)
  df2 <- df[, c("time_bin", "emmean", "SE")]
  names(df2) <- c("time_bin", "mean", "sd")
  df2
}



#--------------------------------------------------------------------

# モデル①　全被験者の感情スコアを、時間帯ごとの平均（固定効果）のプロットと被験者ごとのズレ（ランダム効果）で説明するモデル
arousal_model <- lmer(arousal ~ 1 + time_bin +(1|subject_id),dat, REML = FALSE)
valence_model <- lmer(valence ~ 1 + time_bin +(1|subject_id),dat, REML = FALSE)
aro_df <- summary_to_df(arousal_model)
val_df <- summary_to_df(valence_model)

# モデル①について、各時間帯ごとに固定効果の推定値をプロットし、その推定値の標準誤差をエラーバーで示したグラフ
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
    title = "Arousal ± SD by Time Bin"
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
    title = "Valence ± SD by Time Bin"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

#--------------------------------------------------------------------
















