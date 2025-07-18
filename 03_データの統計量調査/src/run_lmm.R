#!/usr/bin/env Rscript
# run_lmm.R ---------------------------------------------------------------
# Usage: Rscript run_lmm.R <in_csv> <out_json>
suppressPackageStartupMessages({
  library(lme4)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Need <input_csv> <output_json>", call. = FALSE)
}
in_csv   <- args[1]
out_json <- args[2]

# ---- CSV 読み込み ----
dat <- read.csv(in_csv, stringsAsFactors = TRUE)

# ---- time_bin の全レベルを明示 ----
all_bins <- c(
  "00:00-03:00","03:00-06:00","06:00-09:00",
  "09:00-12:00","12:00-15:00","15:00-18:00",
  "18:00-21:00","21:00-24:00"
)
dat$subject_id <- factor(dat$subject_id)
dat$time_bin   <- factor(dat$time_bin, levels = all_bins)

# ---- デバッグ：レベル確認 ----
cat("time_bin levels:", paste(levels(dat$time_bin), collapse = ", "), "\n")

# ---- モデルあてはめ関数 ----
fit_one <- function(resp) {
  frm <- as.formula(paste(resp, "~ time_bin + (1 + time_bin | subject)"))
  fit <- lmer(frm, data = dat, REML = TRUE)
  coefs     <- fixef(fit)
  intercept <- coefs["(Intercept)"]
  levs      <- levels(dat$time_bin)
  means     <- numeric(length(levs))
  for (i in seq_along(levs)) {
    term     <- paste0("time_bin", levs[i])
    means[i] <- intercept + ifelse(term %in% names(coefs), coefs[term], 0)
  }
  list(
    mean = means,
    sd   = rep(as.numeric(sigma(fit)), length(levs))
  )
}

# ---- Valence, Arousal の LMM 推定 ----
val  <- fit_one("valence")
aro  <- fit_one("arousal")

# ---- JSON 出力 ----
out <- list(
  time_bins = levels(dat$time_bin),
  valence   = val,
  arousal   = aro
)
write(
  toJSON(out, digits = 7, pretty = TRUE, auto_unbox = TRUE),
  out_json
)
cat("[R] JSON written:", out_json, "\n")
