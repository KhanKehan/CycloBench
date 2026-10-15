# 求解环境镜像（网络在构建时可用，运行时由平台断网）

FROM rocker/verse:4.4.1

# astrochron 及其依赖（multitaper 谱分析、信号处理等）
RUN R -e "install.packages(c('astrochron','multitaper','gsignal','pracma','signal','jsonlite'), repos='https://cloud.r-project.org')"

# 基础工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pandas && rm -rf /var/lib/apt/lists/*

# 工作目录
WORKDIR /workspace
RUN mkdir -p /workspace/figures /workspace/data

# 关键版本（交付时固化）：
#   R 4.4.1
#   astrochron >= 1.0（CRAN）
#   multitaper（astrochron 的 MTM 后端）
# 备注：构建时需联网拉取 CRAN 依赖；运行阶段完全离线。
