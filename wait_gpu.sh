#!/bin/bash

# GPU监控脚本
# 功能：检测指定GPU是否空闲超过阈值，满足条件后启动目标脚本

# ============ 配置参数 ============
GPU_IDS=(2 3)                        # 目标GPU编号（数组形式）
IDLE_THRESHOLD=600                   # 空闲阈值（秒）
CHECK_INTERVAL=60                    # 检查间隔（秒）
MAX_WAIT_TIME=36000                  # 最大等待时间（秒）
SCRIPT_TO_RUN="bash one_way2.sh"     # 满足条件后要运行的脚本
LOG_FILE="gpu_monitor.log"           # 日志文件
NVIDIA_SMI_TIMEOUT=10                # nvidia-smi 超时时间（秒）
# ==================================

IDLE_START_FILE="/tmp/gpu_${GPU_IDS[*]// /_}_idle_start"
SCRIPT_START_TIME=$(date +%s)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 检查单个GPU是否有进程在运行
check_gpu_processes() {
    local gpu_id=$1
    local process_count
    process_count=$(timeout "$NVIDIA_SMI_TIMEOUT" nvidia-smi -i "$gpu_id" \
        --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v "^$" | wc -l)
    
    if [ $? -ne 0 ]; then
        log "警告: nvidia-smi 查询 GPU $gpu_id 超时或失败"
        echo "-1"
        return
    fi
    echo "$process_count"
}

# 检查所有目标GPU是否都空闲
are_gpus_idle() {
    for gpu_id in "${GPU_IDS[@]}"; do
        local procs
        procs=$(check_gpu_processes "$gpu_id")
        if [ "$procs" -eq -1 ]; then
            return 2  # nvidia-smi 失败
        elif [ "$procs" -ne 0 ]; then
            return 1  # 有进程在运行
        fi
    done
    return 0  # 都空闲
}

# 获取GPU使用情况（用于日志）
get_gpu_status() {
    local status=""
    for gpu_id in "${GPU_IDS[@]}"; do
        local procs mem
        procs=$(check_gpu_processes "$gpu_id")
        mem=$(timeout "$NVIDIA_SMI_TIMEOUT" nvidia-smi -i "$gpu_id" \
            --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        status+="GPU${gpu_id}: ${procs}进程/${mem:-N/A}MiB, "
    done
    echo "${status%, }"  # 移除末尾逗号
}

format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    printf "%d小时%d分%d秒" $hours $minutes $secs
}

check_timeout() {
    local current_time=$(date +%s)
    local elapsed=$((current_time - SCRIPT_START_TIME))
    
    if [ "$elapsed" -ge "$MAX_WAIT_TIME" ]; then
        log "=========================================="
        log "已等待 $(format_duration $elapsed)，超过最大等待时间"
        log "GPU未能在规定时间内空闲，监控脚本退出"
        log "=========================================="
        rm -f "$IDLE_START_FILE"
        exit 1
    fi
    echo "$elapsed"
}

main() {
    log "=========================================="
    log "GPU监控脚本启动"
    log "监控GPU: ${GPU_IDS[*]}"
    log "空闲阈值: ${IDLE_THRESHOLD}秒 ($(($IDLE_THRESHOLD/60))分钟)"
    log "最大等待: ${MAX_WAIT_TIME}秒 ($(($MAX_WAIT_TIME/3600))小时)"
    log "检查间隔: ${CHECK_INTERVAL}秒"
    log "=========================================="
    
    rm -f "$IDLE_START_FILE"
    
    while true; do
        current_time=$(date +%s)
        status=$(get_gpu_status)
        elapsed=$(check_timeout)
        remaining=$((MAX_WAIT_TIME - elapsed))
        
        are_gpus_idle
        idle_result=$?
        
        if [ "$idle_result" -eq 0 ]; then
            if [ ! -f "$IDLE_START_FILE" ]; then
                echo "$current_time" > "$IDLE_START_FILE"
                log "检测到GPU ${GPU_IDS[*]} 空闲，开始计时... [$status]"
            else
                idle_start=$(cat "$IDLE_START_FILE")
                idle_duration=$((current_time - idle_start))
                
                log "GPU持续空闲: $(format_duration $idle_duration) [$status]"
                
                if [ "$idle_duration" -ge "$IDLE_THRESHOLD" ]; then
                    log "=========================================="
                    log "空闲时间达到阈值！启动脚本: $SCRIPT_TO_RUN"
                    log "=========================================="
                    rm -f "$IDLE_START_FILE"
                    $SCRIPT_TO_RUN
                    log "脚本执行完成，监控脚本退出"
                    exit 0
                fi
            fi
        elif [ "$idle_result" -eq 2 ]; then
            log "nvidia-smi 异常，跳过本次检查 (剩余: $(format_duration $remaining))"
        else
            if [ -f "$IDLE_START_FILE" ]; then
                log "GPU被占用，重置计时器 [$status]"
                rm -f "$IDLE_START_FILE"
            else
                log "GPU使用中 [$status] (剩余: $(format_duration $remaining))"
            fi
        fi
        
        sleep "$CHECK_INTERVAL"
    done
}

trap 'log "收到中断信号，退出"; rm -f "$IDLE_START_FILE"; exit 0' SIGINT SIGTERM

main