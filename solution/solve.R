# ==============================================================================
# solve.R —— 参考解：上古生代湖盆双井旋回地层学（R + astrochron）
#
# 说明：本参考解演示"正确方法论路径"。交付前必须在 Harbor 环境内实际运行、
# 按运行结果完成最终标定（滤波带宽、峰值阈值等），并保留满分运行日志。
# 核心科学软件：astrochron（MTM 谱分析、Taner 带通、峰值提取、Hilbert 解调）。
# ==============================================================================

library(astrochron)
library(jsonlite)

set.seed(1)

A  <- read.csv('/workspace/data/well_A.csv')
B  <- read.csv('/workspace/data/well_B.csv')

dA  <- A$Depth_m
gA  <- A$GR_API
cA  <- A$CARB_pct
dB  <- B$Depth_m
gB  <- B$GR_API

# ------------------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------------------
# 稳健去趋势（低通 + 残差）
detrend_loess <- function(x, y, span = 0.35) {
  y - predict(loess(y ~ x, span = span, degree = 1))
}
# 滑动窗正弦拟合相位（用于检测间断导致的相位跳变）
phase_track <- function(depth, y, wl, win = 100) {
  n <- length(depth)
  ph <- rep(NA_real_, n)
  for (i in seq(1, n - win, by = win / 2)) {
    ii <- i:(i + win - 1)
    z  <- 2 * pi * depth[ii] / wl
    b  <- coef(lm(y[ii] ~ sin(z) + cos(z)))
    ph[ii] <- atan2(b[3], b[2])  # 相位
  }
  ph
}
# 从相位序列找跳变（间断位置）
find_jumps <- function(depth, ph) {
  dph <- diff(ph)
  dph <- (dph + pi) %% (2 * pi) - pi          # 折叠到 [-pi, pi]
  thr <- 3 * sd(dph, na.rm = TRUE)
  idx <- which(abs(dph) > thr)
  if (length(idx) == 0) return(data.frame())
  # 聚类相邻跳变点
  grp <- cumsum(c(TRUE, diff(idx) > 3))
  do.call(rbind, lapply(split(idx, grp), function(g) {
    data.frame(depth_m = depth[g[length(g)]], dphase = sum(dph[g]))
  }))
}

# ------------------------------------------------------------------------------
# 1. 频谱分析（MTM + AR1 红噪声）——GR 与 CARB
# ------------------------------------------------------------------------------
gA_det <- detrend_loess(dA, gA)
cA_det <- detrend_loess(dA, cA)

png('/workspace/figures/spectrum_GR.png', width = 1600, height = 800, res = 150)
mtm(data.frame(dA, gA_det), ar1 = TRUE, output = 1, genplot = TRUE)
dev.off()
png('/workspace/figures/spectrum_CARB.png', width = 1600, height = 800, res = 150)
mtm(data.frame(dA, cA_det), ar1 = TRUE, output = 1, genplot = TRUE)
dev.off()

spec_gr   <- mtm(data.frame(dA, gA_det), ar1 = TRUE, output = 1, genplot = FALSE)
spec_carb <- mtm(data.frame(dA, cA_det), ar1 = TRUE, output = 1, genplot = FALSE)

# 帮助函数：某深度域波长带的谱峰是否显著
band_peak <- function(spec, wl_lo, wl_hi) {
  wl <- 1 / spec[, 1]
  m <- wl >= wl_lo & wl <= wl_hi
  if (!any(m)) return(FALSE)
  any(spec[m, 2] > spec[m, 3])  # power 超过该处 AR1 置信线
}
# Q6：GR 中的 ~60 kyr（约 4.1-4.3 m）与 ~112 kyr（约 8-9 m）峰，在 CARB 中是否存在
gr_60  <- band_peak(spec_gr,   4.0, 4.4)
gr_112 <- band_peak(spec_gr,   8.0, 9.5)
ca_60  <- band_peak(spec_carb, 4.0, 4.4)
ca_112 <- band_peak(spec_carb, 8.0, 9.5)

# ------------------------------------------------------------------------------
# 2. 405 kyr 长偏心率旋回提取（Taner 带通，波段按 SR 6-9.5 cm/kyr 取 20-40 m）
# ------------------------------------------------------------------------------
f405 <- taner(data.frame(dA, gA_det), padfac = 10, flow = 1/40, fhigh = 1/20,
              roll = 10^4, demean = TRUE, detrend = TRUE, addmean = FALSE,
              genplot = FALSE)
png('/workspace/figures/filter_405.png', width = 1600, height = 800, res = 150)
plot(dA, gA_det, type = 'l', col = 'grey80', xlab = 'Depth (m)', ylab = 'GR (detrended)')
lines(f405[, 1], f405[, 2], col = 'red')
dev.off()

# 405 旋回峰提取
pk <- peaks(f405[, 2], genplot = FALSE)
pk_d <- f405[pk, 1]
png('/workspace/figures/cycles_405.png', width = 1600, height = 800, res = 150)
plot(dA, gA_det, type = 'l', col = 'grey80')
lines(f405[, 1], f405[, 2], col = 'red')
abline(v = pk_d, col = 'blue', lty = 2)
dev.off()

# 每个旋回的局部波长 -> 局部沉积速率（cm/kyr）
cyc_wl <- diff(pk_d)                      # m/cycle
local_sr <- cyc_wl / 0.405                # cm/kyr（0.405 Myr = 405 kyr）
png('/workspace/figures/sedimentation_rate.png', width = 1600, height = 800, res = 150)
plot(pk_d[-1], local_sr, type = 'o', xlab = 'Depth (m)', ylab = 'Local SR (cm/kyr)')
dev.off()

# Q1：总沉积时长 = 完整 405 kyr 旋回数 × 0.405 Myr
n_cycles <- length(pk_d) - 1
dur_Myr  <- n_cycles * 0.405
# Q4：平均沉积速率
mean_sr <- (max(dA) - min(dA)) * 100 / (dur_Myr * 1000)

# ------------------------------------------------------------------------------
# 3. 间断检测：405 相位跳变
# ------------------------------------------------------------------------------
ph <- phase_track(dA, f405[, 2], wl = 30, win = 120)
jumps <- find_jumps(dA, ph)
# 每个跳变的相位差 -> 缺失时间（kyr）
jumps$duration_kyr <- abs(jumps$dphase) / (2 * pi) * 405

png('/workspace/figures/tuning.png', width = 1600, height = 800, res = 150)
plot(dA, ph, type = 'l', xlab = 'Depth (m)', ylab = 'Phase (rad)')
abline(v = jumps$depth_m, col = 'red', lty = 2)
dev.off()

# ------------------------------------------------------------------------------
# 4. 年代-深度模型与 Q2（250 m 年龄）
# 策略：逐段把 405 旋回调谐到时间轴；间断处时间直接跳跃
# ------------------------------------------------------------------------------
age_model <- function(depth, pk_d, jumps, dur_kyr) {
  # 按峰值计数赋时（405 kyr/cycle），并在间断处加上缺失时间
  n <- length(pk_d)
  t_pk <- cumsum(c(0, rep(405, n - 1)))                 # kyr（未含间断）
  if (nrow(jumps) > 0) {
    for (j in seq_len(nrow(jumps))) {
      below <- pk_d >= jumps$depth_m[j]
      t_pk[below] <- t_pk[below] + jumps$duration_kyr[j]
    }
  }
  # 顶部锚 0 kyr（0 m = 300 Ma，向下增龄）
  approx(pk_d, t_pk, xout = depth, rule = 2)$y
}
age250_kyr <- age_model(250.0, pk_d, jumps, dur_Myr)
age250_Ma  <- 300.0 + age250_kyr / 1000

# ------------------------------------------------------------------------------
# 5. 斜率周期（Q5）与 170 kyr 调幅（Q7）：对 CARB 的斜率带做解调
# ------------------------------------------------------------------------------
# 深度域斜率波长随 SR 变化（2.1-3.4 m），直接对 CARB 谱读斜率主峰周期：
# 先用局部 SR 把深度转时间（粗略调谐），再做时间域谱
depth2time <- approxfun(pk_d, cumsum(c(0, rep(405, length(pk_d) - 1))) +
                        c(0, if (nrow(jumps) > 0) cumsum(jumps$duration_kyr) else 0))

cA_t <- data.frame(time_kyr = depth2time(dA), value = cA_det)
cA_t <- cA_t[order(cA_t$time_kyr), ]
cA_ev <- linterp(cA_t, dt = 1, genplot = FALSE)

# 斜率带（30-40 kyr）带通 + Hilbert 包络
ob <- taner(data.frame(cA_ev[, 1], cA_ev[, 2]), flow = 1/40, fhigh = 1/30,
            roll = 10^5, demean = TRUE, detrend = TRUE, addmean = FALSE, genplot = FALSE)
ob_env <- hilbert(ob, demean = FALSE, genplot = FALSE)

spec_env <- periodogram(ob_env, output = 1, genplot = FALSE)
wl_env <- 1 / spec_env[, 1]
# 170 kyr 调幅峰
am_170 <- any(spec_env[wl_env > 140 & wl_env < 200, 2] > 1.5 * median(spec_env[, 2]))
gr_factor <- sd(ob_env[, 2], na.rm = TRUE) / mean(abs(ob_env[, 2]), na.rm = TRUE)
carb_factor <- gr_factor  # 占位：实际取 CARB 包络因子（见下）

# 对 GR 同样解调比较包络强度
gA_t <- data.frame(time_kyr = depth2time(dA), value = gA_det)
gA_t <- gA_t[order(gA_t$time_kyr), ]
gA_ev <- linterp(gA_t, dt = 1, genplot = FALSE)
ob_g <- taner(data.frame(gA_ev[, 1], gA_ev[, 2]), flow = 1/40, fhigh = 1/30,
              roll = 10^5, demean = TRUE, detrend = TRUE, addmean = FALSE, genplot = FALSE)
obg_env <- hilbert(ob_g, demean = FALSE, genplot = FALSE)
gr_factor  <- sd(obg_env[, 2], na.rm = TRUE) / mean(abs(obg_env[, 2]), na.rm = TRUE)
carb_factor <- sd(ob_env[, 2], na.rm = TRUE) / mean(abs(ob_env[, 2]), na.rm = TRUE)

png('/workspace/figures/obliquity_AM.png', width = 1600, height = 800, res = 150)
plot(ob_env[, 1], ob_env[, 2], type = 'l', col = 'red', xlab = 'Time (kyr)',
     ylab = 'Obliquity amplitude envelope')
lines(obg_env[, 1], obg_env[, 2], col = 'blue')
legend('topright', legend = c('CARB', 'GR'), col = c('red', 'blue'), lty = 1)
dev.off()

# Q5：斜率主周期（时间域，从 CARB 调谐谱读峰）
spec_t <- periodogram(data.frame(cA_ev[, 1], cA_ev[, 2]), output = 1, genplot = FALSE)
wt <- 1 / spec_t[, 1]
obl_kyr <- wt[which.max(spec_t[wt > 20 & wt < 45, 2]) + which(wt > 20 & wt < 45)[1] - 1]

# ------------------------------------------------------------------------------
# 6. 井 B：相同流程，找其间断并与 A 对比（Q8）
# ------------------------------------------------------------------------------
gB_det <- detrend_loess(dB, gB)
f405B <- taner(data.frame(dB, gB_det), padfac = 10, flow = 1/40, fhigh = 1/20,
               roll = 10^4, demean = TRUE, detrend = TRUE, addmean = FALSE, genplot = FALSE)
phB <- phase_track(dB, f405B[, 2], wl = 30, win = 100)
jumpsB <- find_jumps(dB, phB)
jumpsB$duration_kyr <- abs(jumpsB$dphase) / (2 * pi) * 405

png('/workspace/figures/tuning_B.png', width = 1600, height = 800, res = 150)
plot(dB, phB, type = 'l', xlab = 'Depth (m)', ylab = 'Phase (rad)')
abline(v = jumpsB$depth_m, col = 'red', lty = 2)
dev.off()

# 区域性间断 = A 与 B 中间断时长匹配者
regional <- NA
for (j in seq_len(nrow(jumpsB))) {
  if (any(abs(jumpsB$duration_kyr[j] - jumps$duration_kyr) < 50)) {
    regional <- j
    break
  }
}

# ------------------------------------------------------------------------------
# 7. 汇总答案
# ------------------------------------------------------------------------------
answers <- list(
  Q1_total_duration_Myr = round(dur_Myr, 2),
  Q2_age_250m_Ma = round(age250_Ma, 3),
  Q3_hiatuses = if (nrow(jumps) > 0) {
    lapply(seq_len(nrow(jumps)), function(i) list(depth_m = round(jumps$depth_m[i], 2),
                                                  duration_kyr = round(jumps$duration_kyr[i], 0)))
  } else list(),
  Q4_mean_SR_cm_kyr = round(mean_sr, 2),
  Q5_obliquity_kyr = round(obl_kyr, 1),
  Q6 = list(p60_autogenic = gr_60 && !ca_60,
            p112_autogenic = gr_112 && !ca_112,
            carb_has_60kyr_peak = ca_60,
            carb_has_112kyr_peak = ca_112),
  Q7 = list(exists = am_170,
            stronger_in = if (carb_factor > gr_factor) 'CARB' else 'GR',
            GR_factor = round(gr_factor, 2),
            CARB_factor = round(carb_factor, 2)),
  Q8 = if (!is.na(regional)) list(wellA_depth_m = round(jumps$depth_m[which.min(abs(jumps$duration_kyr - jumpsB$duration_kyr[regional]))], 2),
                                   wellB_depth_m = round(jumpsB$depth_m[regional], 2),
                                   duration_kyr = round(jumpsB$duration_kyr[regional], 0))
       else list(wellA_depth_m = 0, wellB_depth_m = 0, duration_kyr = 0)
)

write_json(answers, '/workspace/answers.json', pretty = TRUE, auto_unbox = TRUE)
cat('DONE\n')
