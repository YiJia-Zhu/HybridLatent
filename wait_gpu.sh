#!/bin/bash

# GPU监控脚本
# 功能：检测GPU 6和7是否空闲超过5分钟，满足条件后启动 one_way.sh
# 最大等待时间：10小时

# 配置参数
TARGET_GPUS="6,7"                    # 目标GPU编号
IDLE_THRESHOLD=600                   # 空闲阈值（秒），5分钟=300秒
CHECK_INTERVAL=60                    # 检查间隔（秒）
MAX_WAIT_TIME=36000                  # 最大等待时间（秒），10小时=36000秒
SCRIPT_TO_RUN="bash one_way.sh"      # 满足条件后要运行的脚本
LOG_FILE="gpu_monitor.log"           # 日志文件

# 记录空闲开始时间的文件
IDLE_START_FILE="/tmp/gpu_67_idle_start"

# 脚本启动时间
SCRIPT_START_TIME=$(date +%s)

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 检查GPU是否有进程在运行
check_gpu_processes() {
    local gpu_id=$1
    # 获取指定GPU上的计算进程数量
    local process_count=$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v "^$" | wc -l)
    echo "$process_count"
}

# 检查GPU 6和7是否都空闲
are_gpus_idle() {
    local gpu6_procs=$(check_gpu_processes 6)
    local gpu7_procs=$(check_gpu_processes 7)
    
    if [ "$gpu6_procs" -eq 0 ] && [ "$gpu7_procs" -eq 0 ]; then
        return 0  # 都空闲
    else
        return 1  # 有进程在运行
    fi
}

# 获取GPU使用情况（用于日志）
get_gpu_status() {
    local gpu6_procs=$(check_gpu_processes 6)
    local gpu7_procs=$(check_gpu_processes 7)
    local gpu6_mem=$(nvidia-smi -i 6 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    local gpu7_mem=$(nvidia-smi -i 7 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    echo "GPU6: ${gpu6_procs}进程/${gpu6_mem}MiB, GPU7: ${gpu7_procs}进程/${gpu7_mem}MiB"
}

# 格式化时间显示
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    printf "%d小时%d分%d秒" $hours $minutes $secs
}

# 检查是否超过最大等待时间
check_timeout() {
    local current_time=$(date +%s)
    local elapsed=$((current_time - SCRIPT_START_TIME))
    
    if [ "$elapsed" -ge "$MAX_WAIT_TIME" ]; then
        log "=========================================="
        log "已等待 $(format_duration $elapsed)，超过最大等待时间 $(format_duration $MAX_WAIT_TIME)"
        log "GPU未能在规定时间内空闲，监控脚本退出"
        log "=========================================="
        rm -f "$IDLE_START_FILE"
        exit 1
    fi
    
    echo "$elapsed"
}

# 主逻辑
main() {
    log "=========================================="
    log "GPU监控脚本启动"
    log "监控GPU: $TARGET_GPUS"
    log "空闲阈值: ${IDLE_THRESHOLD}秒 ($(($IDLE_THRESHOLD/60))分钟)"
    log "最大等待: ${MAX_WAIT_TIME}秒 ($(($MAX_WAIT_TIME/3600))小时)"
    log "检查间隔: ${CHECK_INTERVAL}秒"
    log "=========================================="
    
    # 清除之前的空闲记录
    rm -f "$IDLE_START_FILE"
    
    while true; do
        current_time=$(date +%s)
        status=$(get_gpu_status)
        
        # 检查是否超时
        elapsed=$(check_timeout)
        remaining=$((MAX_WAIT_TIME - elapsed))
        
        if are_gpus_idle; then
            # GPU空闲
            if [ ! -f "$IDLE_START_FILE" ]; then
                # 首次检测到空闲，记录开始时间
                echo "$current_time" > "$IDLE_START_FILE"
                log "检测到GPU 6和7空闲，开始计时... [$status] (剩余等待: $(format_duration $remaining))"
            else
                # 计算空闲时长
                idle_start=$(cat "$IDLE_START_FILE")
                idle_duration=$((current_time - idle_start))
                
                log "GPU 6和7持续空闲: $(format_duration $idle_duration) [$status] (剩余等待: $(format_duration $remaining))"
                
                if [ "$idle_duration" -ge "$IDLE_THRESHOLD" ]; then
                    log "=========================================="
                    log "空闲时间达到阈值！启动脚本: $SCRIPT_TO_RUN"
                    log "=========================================="
                    
                    # 删除空闲记录文件
                    rm -f "$IDLE_START_FILE"
                    
                    # 执行目标脚本
                    $SCRIPT_TO_RUN
                    
                    log "脚本执行完成，监控脚本退出"
                    exit 0
                fi
            fi
        else
            # GPU正在使用
            if [ -f "$IDLE_START_FILE" ]; then
                log "GPU被占用，重置空闲计时器 [$status] (剩余等待: $(format_duration $remaining))"
                rm -f "$IDLE_START_FILE"
            else
                log "GPU正在使用中 [$status] (剩余等待: $(format_duration $remaining))"
            fi
        fi
        
        sleep "$CHECK_INTERVAL"
    done
}

# 捕获中断信号
trap 'log "收到中断信号，监控脚本退出"; rm -f "$IDLE_START_FILE"; exit 0' SIGINT SIGTERM

# 运行主程序
main