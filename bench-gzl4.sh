#!/usr/bin/env bash
# =============================================================================
# bench_gzl4.sh  gzl4 performance sweep
# Sweeps backends, thread counts, compression levels, and GPU tuning knobs.
# Outputs a ranked terminal summary and a detailed JSON results file.
#
# Usage: ./bench_gzl4.sh [OPTIONS]
#
#   --gzl4 PATH     path to gzl4 binary       (default: auto-detect)
#   --file PATH     existing file to benchmark (default: generate synthetic)
#   --size-mb N     synthetic file MB          (default: 256)
#   --runs N        timed runs per config      (default: 3)
#   --warmup N      warmup runs                (default: 1)
#   --out FILE      JSON output path           (default: auto-named)
#   --quick         shorter sweep
#   --no-hc         skip HC levels (-10/-11/-12)
#   --no-gpu        skip GPU backends
#   --help
# =============================================================================
set -uo pipefail

# ── ANSI ──────────────────────────────────────────────────────────────────────
R=$'\033[0m'
BOLD=$'\033[1m'
DIM=$'\033[2m'
RED=$'\033[1;31m'
GRN=$'\033[1;32m'
YLW=$'\033[1;33m'
BLU=$'\033[1;34m'
CYN=$'\033[1;36m'
WHT=$'\033[1;37m'
MGN=$'\033[1;35m'

# ── All display goes to stderr so \r overwrites work on a single stream ────────
# Redirect our own stdout to stderr; real data (JSON path) uses fd3.
exec 3>&1 1>&2

# ── Ctrl-C / cleanup ──────────────────────────────────────────────────────────
WORKDIR=""
CHILD_PID=""

cleanup() {
    echo ""
    echo ""
    echo "  ${YLW}⚠  Interrupted  cleaning up${R}"
    # Kill any running gzl4 child
    [[ -n "$CHILD_PID" ]] && kill "$CHILD_PID" 2>/dev/null || true
    [[ -n "$WORKDIR" && -d "$WORKDIR" ]] && rm -rf "$WORKDIR"
    exit 130
}
trap cleanup INT TERM

# ── Defaults ──────────────────────────────────────────────────────────────────
GZL4=""
BENCH_FILE=""
SIZE_MB=256
RUNS=3
WARMUP=1
OUT_JSON=""
QUICK=0
NO_HC=0
NO_GPU=0

usage() {
    cat <<EOF
${BOLD}bench_gzl4.sh${R}  gzl4 performance sweep

  ${CYN}--gzl4 PATH${R}    path to gzl4 binary
  ${CYN}--file PATH${R}    existing file to benchmark
  ${CYN}--size-mb N${R}    synthetic file size in MB   (default: 256)
  ${CYN}--runs N${R}       timed runs per config        (default: 3)
  ${CYN}--warmup N${R}     warmup runs                  (default: 1)
  ${CYN}--out FILE${R}     JSON output path
  ${CYN}--quick${R}        shorter sweep
  ${CYN}--no-hc${R}        skip HC levels
  ${CYN}--no-gpu${R}       skip GPU backends
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gzl4)    GZL4="$2";       shift 2 ;;
        --file)    BENCH_FILE="$2"; shift 2 ;;
        --size-mb) SIZE_MB="$2";    shift 2 ;;
        --runs)    RUNS="$2";       shift 2 ;;
        --warmup)  WARMUP="$2";     shift 2 ;;
        --out)     OUT_JSON="$2";   shift 2 ;;
        --quick)   QUICK=1;         shift   ;;
        --no-hc)   NO_HC=1;         shift   ;;
        --no-gpu)  NO_GPU=1;        shift   ;;
        --help|-h) usage ;;
        *) echo "${RED}Unknown option: $1${R}"; exit 1 ;;
    esac
done

# ── Locate gzl4 ───────────────────────────────────────────────────────────────
if [[ -z "$GZL4" ]]; then
    for c in ./gzl4 ./build/gzl4 /usr/local/bin/gzl4 /usr/bin/gzl4; do
        [[ -x "$c" ]] && { GZL4="$c"; break; }
    done
fi
command -v gzl4 &>/dev/null && [[ -z "$GZL4" ]] && GZL4=$(command -v gzl4)
[[ -z "$GZL4" || ! -x "$GZL4" ]] && { echo "${RED}gzl4 not found  use --gzl4 PATH${R}"; exit 1; }

GZL4_VERSION=$("$GZL4" -V 2>/dev/null | head -1 || echo "unknown")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
[[ -z "$OUT_JSON" ]] && OUT_JSON="bench_gzl4_${TIMESTAMP}.json"

WORKDIR=$(mktemp -d /tmp/bench_gzl4_XXXXXX)
COMPRESSED="$WORKDIR/bench.lz4"
GZL4_STDERR="$WORKDIR/gzl4_stderr"
START_TIME=$(date +%s)

# ── Display helpers ───────────────────────────────────────────────────────────
TERMW=80
header() {
    echo ""
    echo "${BOLD}${BLU}${R}"
    printf "${BOLD}${BLU}${R}  %-56s${BOLD}${BLU}${R}\n" "$1"
    echo "${BOLD}${BLU}${R}"
}
section() { echo ""; echo "${BOLD}${CYN}── $1 ──${R}"; }
info()    { printf "  ${DIM}▸${R} %s\n" "$*"; }
ok()      { printf "  ${GRN}✓${R} %s\n" "$*"; }
warn()    { printf "  ${YLW}⚠${R} %s\n" "$*"; }
die()     { echo "  ${RED}✗ $*${R}"; exit 1; }

BARW=28   # visible bar width
_bar_full=""
_bar_empty=""
for (( _i=0; _i<BARW; _i++ )); do _bar_full+="="; _bar_empty+="-"; done

# GZL4_SUB_PCT: current gzl4 internal progress (0-100), updated by run_compress
GZL4_SUB_PCT=""
_current_label=""

draw_bar() {
    # draw_bar CURRENT TOTAL "label"
    local cur=$1 tot=$2 label="${3:0:26}"
    local pct=$(( cur * 100 / tot ))
    local fill=$(( cur * BARW / tot ))
    local emp=$(( BARW - fill ))
    # Show gzl4 sub-progress if available  numeric = timed run, text = warmup
    local sub=""
    if [[ -n "$GZL4_SUB_PCT" ]]; then
        if [[ "$GZL4_SUB_PCT" =~ ^[0-9]+$ ]]; then
            sub=$(printf "${DIM} run:${YLW}%3s%%${R}" "$GZL4_SUB_PCT")
        else
            sub=$(printf "${DIM} %s%%${R}" "$GZL4_SUB_PCT")
        fi
    fi
    printf "\r  ${CYN}[${GRN}%.${fill}s${DIM}%.${emp}s${CYN}]${R} ${YLW}%3d%%${R}  ${DIM}%-26s${R}%s    " \
        "$_bar_full" "$_bar_empty" "$pct" "$label" "$sub"
}

erase_bar() {
    printf "\r%-${TERMW}s\r" ""
}

# ── Result row ────────────────────────────────────────────────────────────────
# All output is already going to stderr (exec 3>&1 1>&2 above)
print_row() {
    # print_row LABEL MBPS RATIO PHASE_BEST_MBPS
    local label="$1" mbps="$2" ratio="$3" best="$4"
    local pct=100
    awk "BEGIN{exit !($best+0 > 0)}" 2>/dev/null && \
        pct=$(awk "BEGIN{printf \"%d\", ($mbps+0)/($best+0)*100}" 2>/dev/null) || true
    local color
    if   [[ $pct -ge 95 ]]; then color="${GRN}${BOLD}"
    elif [[ $pct -ge 75 ]]; then color="${YLW}"
    elif [[ $pct -ge 50 ]]; then color="${WHT}"
    else                         color="${DIM}"
    fi
    printf "  %-36s  %s%8.1f${R}  ${DIM}MB/s${R}  ${YLW}%6.2f%%${R}\n" \
        "${label:0:36}" "$color" "$mbps" "$ratio"
}

table_header() {
    printf "  %s%-36s  %8s  %-4s  %6s%s\n" "$DIM" "Configuration" "" "MB/s" "Ratio" "$R"
    printf "  %s%s%s\n" "$DIM" "$(printf '─%.0s' {1..61})" "$R"
}

# ── Timing ────────────────────────────────────────────────────────────────────
now_ms() { date +%s%3N; }

# ── run_compress: warmup + timed runs ─────────────────────────────────────────
# run_compress gzl4args...
# Compresses BENCH_FILE to stdout -> COMPRESSED
# Results in globals RC_MBPS and RC_RATIO; reads _current_label for progress display
RC_MBPS="0"
RC_RATIO="0"
run_compress() {
    RC_MBPS="0"; RC_RATIO="0"
    local _in _out _total_ms _r _t0 _t1 _avg
    _in=$(stat -c%s "$BENCH_FILE")

    # warmup  tap stderr so the bar animates with "warmup NN%" during the pause
    local _w
    for (( _w=0; _w<WARMUP; _w++ )); do
        > "$GZL4_STDERR"
        CHILD_PID=""
        "$GZL4" "$@" -c "$BENCH_FILE" > "$COMPRESSED" 2>"$GZL4_STDERR" &
        CHILD_PID=$!
        while kill -0 "$CHILD_PID" 2>/dev/null; do
            local _pct
            _pct=$(grep -oP '(?<![.\d])\d{1,3}(?=%)' "$GZL4_STDERR" 2>/dev/null \
                   | awk '$1+0<=100' | tail -1 || echo "")
            GZL4_SUB_PCT="${_pct:+warmup ${_pct}}"
            draw_bar "$CONFIG_NUM" "$TOTAL_CONFIGS" "$_current_label"
            sleep 0.1
        done
        wait "$CHILD_PID" || { CHILD_PID=""; GZL4_SUB_PCT=""; return 1; }
        CHILD_PID=""
        GZL4_SUB_PCT=""
    done

    # timed runs  tap gzl4 stderr for live progress
    _total_ms=0
    GZL4_SUB_PCT="0"
    for (( _r=0; _r<RUNS; _r++ )); do
        > "$GZL4_STDERR"   # clear tap file
        _t0=$(now_ms)
        # Run without -q so gzl4 emits progress; capture stderr to tap file
        "$GZL4" "$@" -c "$BENCH_FILE" > "$COMPRESSED" 2>"$GZL4_STDERR" &
        CHILD_PID=$!
        # Poll tap file while gzl4 runs; update GZL4_SUB_PCT each tick
        while kill -0 "$CHILD_PID" 2>/dev/null; do
            local _pct
            _pct=$(grep -oP '(?<![.\d])\d{1,3}(?=%)' "$GZL4_STDERR" 2>/dev/null \
                   | awk '$1+0<=100' | tail -1 || echo "")
            [[ -n "$_pct" ]] && GZL4_SUB_PCT="$_pct"
            draw_bar "$CONFIG_NUM" "$TOTAL_CONFIGS" "$_current_label"
            sleep 0.1
        done
        wait "$CHILD_PID" || { CHILD_PID=""; GZL4_SUB_PCT=""; return 1; }
        CHILD_PID=""
        _t1=$(now_ms)
        _total_ms=$(( _total_ms + _t1 - _t0 ))
    done
    GZL4_SUB_PCT=""

    _out=$(stat -c%s "$COMPRESSED")
    _avg=$(( _total_ms / RUNS ))
    [[ $_avg -lt 1 ]] && _avg=1
    RC_MBPS=$(awk  "BEGIN{printf \"%.1f\", ($_in/1048576)/($_avg/1000)}")
    RC_RATIO=$(awk "BEGIN{printf \"%.2f\", $_out/$_in*100}")
    return 0
}

# ── run_decompress: compress once, then time decompression ────────────────────
run_decompress() {
    RC_MBPS="0"
    local _in _total_ms _r _t0 _t1 _avg _w
    _in=$(stat -c%s "$BENCH_FILE")

    "$GZL4" -q "$@" -c "$BENCH_FILE" > "$COMPRESSED" 2>/dev/null || \
        return 1

    for (( _w=0; _w<WARMUP; _w++ )); do
        "$GZL4" -q -dc "$COMPRESSED" > /dev/null 2>/dev/null || true
    done

    _total_ms=0
    for (( _r=0; _r<RUNS; _r++ )); do
        _t0=$(now_ms)
        "$GZL4" -q -dc "$COMPRESSED" > /dev/null 2>/dev/null &
        CHILD_PID=$!
        wait "$CHILD_PID" || true
        CHILD_PID=""
        _t1=$(now_ms)
        _total_ms=$(( _total_ms + _t1 - _t0 ))
    done

    _avg=$(( _total_ms / RUNS ))
    [[ $_avg -lt 1 ]] && _avg=1
    RC_MBPS=$(awk "BEGIN{printf \"%.1f\", ($_in/1048576)/($_avg/1000)}")
    return 0
}

# ── Result tracking ───────────────────────────────────────────────────────────
BEST_COMP_MBPS="0"
BEST_COMP_LABEL=""
BEST_COMP_ARGS=""
BEST_DECOMP_MBPS="0"
BEST_DECOMP_LABEL=""
CONFIG_NUM=0
JSON_RESULTS=()

_gt() { awk "BEGIN{exit !($1+0 > $2+0)}"; }   # float greater-than

# ── Core per-config runner ────────────────────────────────────────────────────
# run_one PHASE LABEL BACKEND LEVEL THREADS BATCH STREAMS PHASE_BEST_VAR -- gzl4args...
run_one() {
    local phase="$1" label="$2" backend="$3" level="$4" threads="$5" \
          batch="$6" streams="$7" pb_var="$8"
    shift 8
    local args_str="$*"

    (( CONFIG_NUM++ )) || true
    draw_bar "$CONFIG_NUM" "$TOTAL_CONFIGS" "$label"

    local mbps="" ratio=""
    _current_label="$label"
    if run_compress "$@"; then
        mbps="$RC_MBPS"; ratio="$RC_RATIO"
        erase_bar
        local pb_val="${!pb_var}"
        print_row "$label" "$mbps" "$ratio" "$pb_val"

        # update phase best
        _gt "$mbps" "$pb_val" && printf -v "$pb_var" '%s' "$mbps" || true
        # update global best
        _gt "$mbps" "$BEST_COMP_MBPS" && {
            BEST_COMP_MBPS="$mbps"
            BEST_COMP_LABEL="$label"
            BEST_COMP_ARGS="$args_str"
        } || true

        # append JSON entry
        local jargs jentry
        jargs=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$args_str")
        jentry="{\"phase\":\"$phase\",\"backend\":\"$backend\","
        jentry+="\"level\":\"$level\",\"threads\":$threads,"
        jentry+="\"batch_size\":\"$batch\",\"streams_per_gpu\":\"$streams\","
        jentry+="\"mbps\":$mbps,\"ratio_pct\":$ratio,\"args\":$jargs}"
        JSON_RESULTS+=("$jentry")
    else
        erase_bar
        printf "  ${DIM}%-36s  skipped${R}\n" "${label:0:36}"
    fi
}

# ── Decompress runner ─────────────────────────────────────────────────────────
decomp_run() {
    local label="$1" backend="$2" pb_var="$3"; shift 3

    (( CONFIG_NUM++ )) || true
    draw_bar "$CONFIG_NUM" "$TOTAL_CONFIGS" "decomp: $label"

    if run_decompress "$@"; then
        local mbps="$RC_MBPS"
        erase_bar
        local pb_val="${!pb_var}"
        local pct=100
        _gt "$pb_val" "0" && pct=$(awk "BEGIN{printf \"%d\", $mbps/$pb_val*100}") || true
        local col
        if   [[ $pct -ge 95 ]]; then col="${GRN}${BOLD}"
        elif [[ $pct -ge 75 ]]; then col="${YLW}"
        else                         col="${DIM}"
        fi
        printf "  %-36s  %s%8.1f${R}  ${DIM}MB/s${R}\n" "${label:0:36}" "$col" "$mbps"
        _gt "$mbps" "$pb_val" && printf -v "$pb_var" '%s' "$mbps" || true
        _gt "$mbps" "$BEST_DECOMP_MBPS" && {
            BEST_DECOMP_MBPS="$mbps"
            BEST_DECOMP_LABEL="$label"
        } || true
        local jargs jentry
        jargs=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$*")
        jentry="{\"phase\":\"decompress\",\"backend\":\"$backend\","
        jentry+="\"level\":\"N/A\",\"threads\":0,"
        jentry+="\"batch_size\":\"auto\",\"streams_per_gpu\":\"auto\","
        jentry+="\"mbps\":$mbps,\"ratio_pct\":0,\"args\":$jargs}"
        JSON_RESULTS+=("$jentry")
    else
        erase_bar
        printf "  ${DIM}%-36s  skipped${R}\n" "${label:0:36}"
    fi
}

# =============================================================================
# BANNER
# =============================================================================
clear
echo ""
echo "${BOLD}${MGN}  ${R}"
echo "${BOLD}${MGN}           gzl4  Performance  Benchmark             ${R}"
echo "${BOLD}${MGN}     backends · levels · threads · GPU tuning        ${R}"
echo "${BOLD}${MGN}  ${R}"
echo ""
info "Press ${BOLD}Ctrl-C${R} at any time to abort"

# =============================================================================
# SYSTEM DETECTION
# =============================================================================
header "System Detection"
CPU_MODEL=$(grep 'model name' /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || echo "unknown")
CPU_CORES=$(nproc 2>/dev/null || echo 4)
MEM_GB=$(awk '/MemTotal/{printf "%.1f", $2/1048576}' /proc/meminfo 2>/dev/null || echo "?")
HOSTNAME=$(hostname 2>/dev/null || echo "unknown")

info "Host:   ${BOLD}$HOSTNAME${R}"
info "CPU:    ${BOLD}$CPU_MODEL${R}  (${CPU_CORES} cores)"
info "RAM:    ${BOLD}${MEM_GB} GB${R}"
info "gzl4:   ${BOLD}$GZL4${R}"
info "        ${DIM}$GZL4_VERSION${R}"

GPU_COUNT=0
GPU_NAMES=()
GPU_WORKS=0

if [[ $NO_GPU -eq 0 ]] && command -v nvidia-smi &>/dev/null; then
    while IFS= read -r line; do
        [[ -n "$line" ]] && { GPU_NAMES+=("$line"); (( GPU_COUNT++ )) || true; }
    done < <(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)
fi

if [[ $GPU_COUNT -gt 0 ]]; then
    info "GPUs:   ${BOLD}${GPU_COUNT}x${R}"
    for g in "${GPU_NAMES[@]}"; do info "        ${DIM}$g${R}"; done
    printf "  ${DIM}▸${R} Probing GPU compress ... "
    if echo "probe" | "$GZL4" -q --gpu-only -c - > /dev/null 2>/dev/null; then
        GPU_WORKS=1; echo "${GRN}OK${R}"
    else
        echo "${YLW}unavailable  skipping GPU phases${R}"
    fi
else
    info "GPUs:   ${DIM}none / --no-gpu${R}"
fi

# =============================================================================
# TEST DATA
# =============================================================================
header "Preparing Test Data"

if [[ -n "$BENCH_FILE" ]]; then
    [[ -f "$BENCH_FILE" ]] || die "File not found: $BENCH_FILE"
    FILE_SIZE=$(stat -c%s "$BENCH_FILE")
    FILE_MB=$(awk "BEGIN{printf \"%.1f\", $FILE_SIZE/1048576}")
    ok "Using: ${BOLD}$BENCH_FILE${R}  (${FILE_MB} MB)"
else
    BENCH_FILE="$WORKDIR/bench_data.bin"
    printf "  ${DIM}▸${R} Generating ${BOLD}${SIZE_MB} MB${R} of synthetic data ... "
    python3 - "$BENCH_FILE" "$SIZE_MB" <<'PEOF'
import sys, os
path, size_mb = sys.argv[1], int(sys.argv[2])
target = size_mb * 1024 * 1024
chunk_size = 65536
lines = []
i = 0
while sum(len(l) for l in lines) < int(chunk_size * 0.7):
    lines.append(f'line {i:08d} abcdefghijklmnopqrstuvwxyz0123456789\n'.encode())
    i += 1
structured = b''.join(lines)[:int(chunk_size * 0.7)]
noise = os.urandom(chunk_size - len(structured))
chunk = structured + noise
written = 0
with open(path, 'wb') as f:
    while written < target:
        take = min(len(chunk), target - written)
        f.write(chunk[:take])
        written += take
PEOF
    FILE_SIZE=$(stat -c%s "$BENCH_FILE")
    FILE_MB=$(awk "BEGIN{printf \"%.1f\", $FILE_SIZE/1048576}")
    echo "${GRN}done${R}  (${FILE_MB} MB)"
fi

# Smoke test
printf "  ${DIM}▸${R} Smoke test ... "
if "$GZL4" -q --cpu-only -1 -c "$BENCH_FILE" > "$COMPRESSED" 2>/dev/null; then
    SMOKED=$(awk "BEGIN{printf \"%.1f\", $(stat -c%s "$COMPRESSED")/1048576}")
    echo "${GRN}OK${R}  (${SMOKED} MB compressed)"
else
    echo "${RED}FAILED${R}"
    die "Cannot compress test file. Is gzl4 working?"
fi

# =============================================================================
# SWEEP PLAN
# =============================================================================
header "Sweep Plan"

if [[ $QUICK -eq 1 ]]; then
    LEVELS=(-1 -6 -9)
    HC_LEVELS=(-10 -12)
    THREAD_COUNTS=(1 4 "$CPU_CORES")
    BATCH_SIZES=(auto 8 32 64)
    STREAM_COUNTS=(auto 2 4)
else
    LEVELS=(-1 -3 -6 -9)
    HC_LEVELS=(-10 -11 -12)
    THREAD_COUNTS=(1 2 4 8 16 "$CPU_CORES")
    BATCH_SIZES=(auto 4 8 16 32 64 128)
    STREAM_COUNTS=(auto 1 2 4 8)
fi
[[ $NO_HC -eq 1 ]] && HC_LEVELS=()
THREAD_COUNTS=($(printf '%s\n' "${THREAD_COUNTS[@]}" | sort -nu))

n_cpu=$(( ${#LEVELS[@]} * ${#THREAD_COUNTS[@]} ))
n_hc=$(( ${#HC_LEVELS[@]} * ${#THREAD_COUNTS[@]} ))
n_gpu=0; n_hyb=0; n_batch=0; n_streams=0
if [[ $GPU_WORKS -eq 1 ]]; then
    n_gpu=${#LEVELS[@]}
    n_hyb=$(( ${#LEVELS[@]} * ${#THREAD_COUNTS[@]} ))
    n_batch=${#BATCH_SIZES[@]}
    n_streams=${#STREAM_COUNTS[@]}
fi
n_decomp=1; [[ $GPU_WORKS -eq 1 ]] && n_decomp=3
# Convergence phase: ~7 batch steps + ~4 stream steps + ~7 thread steps
n_conv=0; [[ $GPU_WORKS -eq 1 ]] && n_conv=18
TOTAL_CONFIGS=$(( n_cpu + n_hc + n_gpu + n_hyb + n_batch + n_streams + n_decomp + n_conv ))

info "CPU fast:      ${BOLD}$n_cpu${R} configs"
[[ $n_hc -gt 0 ]]      && info "CPU HC:        ${BOLD}$n_hc${R} configs"
[[ $GPU_WORKS -eq 1 ]] && info "GPU/Hybrid:    ${BOLD}$(( n_gpu + n_hyb + n_batch + n_streams ))${R} configs"
info "Decompression: ${BOLD}$n_decomp${R} configs"
[[ $GPU_WORKS -eq 1 ]] && info "Convergence:   ${BOLD}~$n_conv${R} configs (adaptive)"
info "Total:         ${BOLD}~$TOTAL_CONFIGS${R}  (warmup=$WARMUP, timed runs=$RUNS)"

# =============================================================================
# PHASES
# =============================================================================

# Phase-best variables (named so run_one can use indirect reference)
PB_CPU_FAST="0"
PB_CPU_HC="0"
PB_GPU="0"
PB_HYB="0"
PB_BATCH="0"
PB_STREAMS="0"
PB_DECOMP="0"

# ── Phase 1: CPU fast ─────────────────────────────────────────────────────────
section "Phase 1 · CPU-only  fast levels"
table_header
for lvl in "${LEVELS[@]}"; do
    for thr in "${THREAD_COUNTS[@]}"; do
        run_one "cpu_fast" "cpu-only $lvl  -T $thr" \
            "cpu-only" "$lvl" "$thr" "N/A" "N/A" \
            PB_CPU_FAST \
            --cpu-only "$lvl" -T "$thr"
    done
done
echo "  ${DIM}Phase best: ${GRN}${BOLD}${PB_CPU_FAST} MB/s${R}"

# ── Phase 2: CPU HC ───────────────────────────────────────────────────────────
if [[ ${#HC_LEVELS[@]} -gt 0 ]]; then
    section "Phase 2 · CPU-only  HC levels"
    table_header
    for lvl in "${HC_LEVELS[@]}"; do
        for thr in "${THREAD_COUNTS[@]}"; do
            run_one "cpu_hc" "cpu-only $lvl  -T $thr" \
                "cpu-only" "$lvl" "$thr" "N/A" "N/A" \
                PB_CPU_HC \
                --cpu-only "$lvl" -T "$thr"
        done
    done
    echo "  ${DIM}Phase best: ${GRN}${BOLD}${PB_CPU_HC} MB/s${R}"
fi

# ── Phase 3: GPU-only ─────────────────────────────────────────────────────────
if [[ $GPU_WORKS -eq 1 ]]; then
    section "Phase 3 · GPU-only  fast levels"
    table_header
    for lvl in "${LEVELS[@]}"; do
        run_one "gpu_fast" "gpu-only $lvl" \
            "gpu-only" "$lvl" "0" "auto" "auto" \
            PB_GPU \
            --gpu-only "$lvl"
    done
    echo "  ${DIM}Phase best: ${GRN}${BOLD}${PB_GPU} MB/s${R}"
fi

# ── Phase 4: Hybrid ───────────────────────────────────────────────────────────
if [[ $GPU_WORKS -eq 1 ]]; then
    section "Phase 4 · Hybrid  fast levels"
    table_header
    for lvl in "${LEVELS[@]}"; do
        for thr in "${THREAD_COUNTS[@]}"; do
            run_one "hybrid_fast" "hybrid $lvl  -T $thr" \
                "hybrid" "$lvl" "$thr" "auto" "auto" \
                PB_HYB \
                --hybrid "$lvl" -T "$thr"
        done
    done
    echo "  ${DIM}Phase best: ${GRN}${BOLD}${PB_HYB} MB/s${R}"
fi

# ── Phase 5: GPU batch-size ───────────────────────────────────────────────────
if [[ $GPU_WORKS -eq 1 ]]; then
    section "Phase 5 · GPU-only  --batch-size sweep  (level -1)"
    table_header
    for bs in "${BATCH_SIZES[@]}"; do
        if [[ "$bs" == "auto" ]]; then
            run_one "gpu_batch" "gpu-only -1  (batch=auto)" \
                "gpu-only" "-1" "0" "auto" "auto" \
                PB_BATCH \
                --gpu-only -1
        else
            run_one "gpu_batch" "gpu-only -1  --batch-size $bs" \
                "gpu-only" "-1" "0" "$bs" "auto" \
                PB_BATCH \
                --gpu-only -1 --batch-size "$bs"
        fi
    done
    echo "  ${DIM}Phase best: ${GRN}${BOLD}${PB_BATCH} MB/s${R}"
fi

# ── Phase 6: GPU streams ──────────────────────────────────────────────────────
if [[ $GPU_WORKS -eq 1 ]]; then
    section "Phase 6 · GPU-only  --streams-per-gpu sweep  (level -1)"
    table_header
    for sp in "${STREAM_COUNTS[@]}"; do
        if [[ "$sp" == "auto" ]]; then
            run_one "gpu_streams" "gpu-only -1  (streams=auto)" \
                "gpu-only" "-1" "0" "auto" "auto" \
                PB_STREAMS \
                --gpu-only -1
        else
            run_one "gpu_streams" "gpu-only -1  --streams-per-gpu $sp" \
                "gpu-only" "-1" "0" "auto" "$sp" \
                PB_STREAMS \
                --gpu-only -1 --streams-per-gpu "$sp"
        fi
    done
    echo "  ${DIM}Phase best: ${GRN}${BOLD}${PB_STREAMS} MB/s${R}"
fi

# ── Phase 7: Decompression ────────────────────────────────────────────────────
section "Phase 7 · Decompression"
printf "  ${DIM}%-36s  %8s  %-4s${R}\n" "Configuration" "" "MB/s"
printf "  ${DIM}%s${R}\n" "$(printf '─%.0s' {1..52})"

decomp_run "cpu-only"     "cpu-only" PB_DECOMP  --cpu-only
if [[ $GPU_WORKS -eq 1 ]]; then
    decomp_run "gpu-only"     "gpu-only" PB_DECOMP  --gpu-only
    decomp_run "hybrid"       "hybrid"   PB_DECOMP  --hybrid
fi
echo "  ${DIM}Phase best: ${GRN}${BOLD}${PB_DECOMP} MB/s${R}"

# =============================================================================
# PHASE 8: Best Performance Convergence
# Binary-searches the optimal --batch-size, --streams-per-gpu, and -T
# using the phase 5/6/4 sweep results as starting brackets.
# Works on whatever backend was fastest  GPU or CPU-only.
# =============================================================================

# ── Measure helper ────────────────────────────────────────────────────────────
# cm VARNAME gzl4args...   sets VARNAME to MB/s (never empty  "0" on failure)
CONV_RESULT="0"
cm() {
    local _v="$1"; shift
    run_compress "$@" && CONV_RESULT="$RC_MBPS" || CONV_RESULT="0"
    printf -v "$_v" '%s' "$CONV_RESULT"
}

# ── Float greater-than with empty-safe defaults ───────────────────────────────
# fgt A B → returns 0 (true) if A > B, else 1
fgt() { awk "BEGIN{exit !(${1:-0}+0 > ${2:-0}+0)}"; }

# ── Single binary-search pass ─────────────────────────────────────────────────
# converge FLAG LO HI BASE_ARGS...
# Prints each probe, sets CONV_BEST_VAL and CONV_BEST_MBPS when done.
CONV_BEST_VAL=""
CONV_BEST_MBPS="0"

_probe() {
    # _probe FLAG VAL BASE_ARGS...   measure, print, update best
    local _flag="$1" _val="$2"; shift 2
    local _m="0"
    (( CONFIG_NUM++ )) || true
    _current_label="${_flag}=${_val}"
    draw_bar "$CONFIG_NUM" "$TOTAL_CONFIGS" "$_current_label"
    cm _m "$@" "$_flag" "$_val"
    erase_bar
    # color: green if best, yellow otherwise
    local _col="$DIM"
    fgt "$_m" "$CONV_BEST_MBPS" && {
        _col="${GRN}${BOLD}"
        CONV_BEST_MBPS="$_m"
        CONV_BEST_VAL="$_val"
    }
    printf "  %-34s  %s%8.1f${R}  ${DIM}MB/s${R}\n"         "${_flag}=${_val}" "$_col" "$_m"
    # return the measurement via global so callers can read it
    CONV_RESULT="$_m"
}

converge() {
    local flag="$1" lo="$2" hi="$3"; shift 3
    local base=("$@")
    CONV_BEST_VAL="$lo"
    CONV_BEST_MBPS="0"

    # measure both endpoints
    local m_lo m_hi
    _probe "$flag" "$lo" "${base[@]}"; m_lo="$CONV_RESULT"
    _probe "$flag" "$hi" "${base[@]}"; m_hi="$CONV_RESULT"

    # binary search until gap <= 1
    while (( hi - lo > 1 )); do
        local mid=$(( (lo + hi) / 2 ))
        local m_mid
        _probe "$flag" "$mid" "${base[@]}"; m_mid="$CONV_RESULT"

        # keep the half with higher average  compare sums directly
        if fgt "$(awk 'BEGIN{printf "%.3f", ('${m_lo:-0}'+'${m_mid:-0}')/2}')" \
               "$(awk 'BEGIN{printf "%.3f", ('${m_mid:-0}'+'${m_hi:-0}')/2}')"; then
            hi="$mid"; m_hi="$m_mid"
        else
            lo="$mid"; m_lo="$m_mid"
        fi
    done
}

# ── Extract top-2 distinct values for a field from a given sweep phase ─────────
# top2 PHASE FIELD → sets TOP2_LO and TOP2_HI (integers, lo < hi), returns 1 if
# fewer than 2 distinct non-"auto" values found.
TOP2_LO="" TOP2_HI=""
top2() {
    local want_phase="$1" want_field="$2"
    # Collect: "value mbps" pairs, sort by mbps desc, pick top-2 distinct values
    local pairs=""
    local entry
    for entry in "${JSON_RESULTS[@]}"; do
        local p v m
        read -r p v m < <(python3 -c "
import json,sys
try:
    d=json.loads(sys.argv[1])
    print(d.get('phase',''), d.get('$want_field',''), d.get('mbps',0))
except: print('','',0)
" "$entry" 2>/dev/null) || continue
        [[ "$p" != "$want_phase" ]] && continue
        [[ "$v" == "auto" || "$v" == "N/A" || -z "$v" ]] && continue
        pairs+="$v $m"$'
'
    done
    [[ -z "$pairs" ]] && { TOP2_LO=""; TOP2_HI=""; return 1; }

    # sort by mbps descending, pick top-2 distinct integer values
    local top
    top=$(echo "$pairs" | sort -k2 -rn | awk '!seen[$1]++ && ++n<=2 {print $1}')
    local a b
    read -r a < <(echo "$top" | head -1)
    read -r b < <(echo "$top" | tail -1 | grep -v "^$a$" || true)
    [[ -z "$a" || -z "$b" || "$a" == "$b" ]] && { TOP2_LO=""; TOP2_HI=""; return 1; }
    # ensure lo < hi
    if (( a < b )); then TOP2_LO="$a"; TOP2_HI="$b"
    else                 TOP2_LO="$b"; TOP2_HI="$a"; fi
    return 0
}

# ── Run Phase 8 ───────────────────────────────────────────────────────────────
section "Phase 8 · Best Performance Convergence"
echo "  ${DIM}Binary-searching optimal knobs using sweep results as brackets${R}"
printf "  %s%-34s  %8s  %-4s%s\n" "$DIM" "Configuration" "" "MB/s" "$R"
printf "  %s%s%s\n" "$DIM" "$(printf '─%.0s' {1..50})" "$R"

# Each step runs based on whether that sweep phase produced data 
# independent of which backend happened to win the overall sweep.

# Convergence results per backend
CONV_BATCH="auto"           # best --batch-size  (gpu-only)
CONV_STREAMS="auto"         # best --streams-per-gpu  (gpu-only)
CONV_THR_CPU="$CPU_CORES"   # best -T  (cpu-only)
CONV_THR_HYB="$CPU_CORES"   # best -T  (hybrid)
CONV_GPU_MBPS="0"
CONV_HYB_MBPS="0"
CONV_CPU_MBPS="0"
CONV_GPU_ARGS="--gpu-only -1"
CONV_HYB_ARGS="--hybrid -1"
CONV_CPU_ARGS="--cpu-only -1"

# ── Step 1: --batch-size  (gpu-only sweep data) ───────────────────────────────
echo ""
echo "  ${BOLD}${YLW}▸ Step 1:${R}  --batch-size  (gpu-only -1)"
if [[ $GPU_WORKS -eq 1 ]]; then
    echo "  ${DIM}  --streams-per-gpu: auto (not yet converged)${R}"
    if top2 "gpu_batch" "batch_size"; then
        converge "--batch-size" "$TOP2_LO" "$TOP2_HI" --gpu-only -1
        CONV_BATCH="$CONV_BEST_VAL"
        CONV_GPU_MBPS="$CONV_BEST_MBPS"
        CONV_GPU_ARGS="--gpu-only -1 --batch-size $CONV_BATCH"
        echo "  ${GRN}✓${R}  Best --batch-size: ${BOLD}${CONV_BATCH}${R}  @ ${GRN}${BOLD}${CONV_GPU_MBPS}${R} MB/s"
    else
        echo "  ${YLW}  Not enough sweep data  run without --no-gpu and without --quick${R}"
    fi
else
    echo "  ${DIM}  Skipped (no GPU available)${R}"
fi

# ── Step 2: --streams-per-gpu  (gpu-only sweep data, batch locked) ────────────
echo ""
echo "  ${BOLD}${YLW}▸ Step 2:${R}  --streams-per-gpu  (gpu-only -1)"
if [[ $GPU_WORKS -eq 1 ]]; then
    batch_lock=()
    [[ "$CONV_BATCH" != "auto" ]] && batch_lock=(--batch-size "$CONV_BATCH")
    if [[ "$CONV_BATCH" == "auto" ]]; then
        echo "  ${DIM}  --batch-size: auto (step 1 skipped or inconclusive)${R}"
    else
        echo "  ${DIM}  --batch-size: ${WHT}${BOLD}${CONV_BATCH}${R}${DIM}  (locked from step 1)${R}"
    fi
    if top2 "gpu_streams" "streams_per_gpu"; then
        converge "--streams-per-gpu" "$TOP2_LO" "$TOP2_HI"             --gpu-only -1 "${batch_lock[@]}"
        CONV_STREAMS="$CONV_BEST_VAL"
        fgt "$CONV_BEST_MBPS" "$CONV_GPU_MBPS" && {
            CONV_GPU_MBPS="$CONV_BEST_MBPS"
            CONV_GPU_ARGS="--gpu-only -1 ${batch_lock[*]+"${batch_lock[*]}"} --streams-per-gpu $CONV_STREAMS"
        }
        echo "  ${GRN}✓${R}  Best --streams-per-gpu: ${BOLD}${CONV_STREAMS}${R}  @ ${GRN}${BOLD}${CONV_BEST_MBPS}${R} MB/s"
    else
        echo "  ${YLW}  Not enough sweep data  run without --no-gpu and without --quick${R}"
    fi
else
    echo "  ${DIM}  Skipped (no GPU available)${R}"
fi

# ── Step 3a: -T  (cpu-only) ───────────────────────────────────────────────────
echo ""
echo "  ${BOLD}${YLW}▸ Step 3:${R}  -T thread count  (cpu-only -1)"
if top2 "cpu_fast" "threads"; then
    converge "-T" "$TOP2_LO" "$TOP2_HI" --cpu-only -1
    CONV_THR_CPU="$CONV_BEST_VAL"
    CONV_CPU_MBPS="$CONV_BEST_MBPS"
    CONV_CPU_ARGS="--cpu-only -1 -T $CONV_THR_CPU"
    echo "  ${GRN}✓${R}  Best -T (cpu-only): ${BOLD}${CONV_THR_CPU}${R}  @ ${GRN}${BOLD}${CONV_CPU_MBPS}${R} MB/s"
else
    echo "  ${YLW}  Not enough sweep data${R}"
fi

# ── Step 4: -T  (hybrid, with steps 1+2 knobs locked in) ────────────────────
if [[ $GPU_WORKS -eq 1 ]]; then
    echo ""
    echo "  ${BOLD}${YLW}▸ Step 4:${R}  -T thread count  (hybrid -1, using best batch/streams from steps 1+2)"
    # Build the fixed GPU knobs discovered in steps 1 and 2
    hyb_gpu_knobs=()
    [[ "$CONV_BATCH"   != "auto" ]] && hyb_gpu_knobs+=(--batch-size "$CONV_BATCH")
    [[ "$CONV_STREAMS" != "auto" ]] && hyb_gpu_knobs+=(--streams-per-gpu "$CONV_STREAMS")
    [[ ${#hyb_gpu_knobs[@]} -gt 0 ]]         && echo "  ${DIM}  locking in: ${hyb_gpu_knobs[*]}${R}"         || echo "  ${DIM}  no GPU knobs to lock (steps 1+2 skipped or kept auto)${R}"
    if top2 "hybrid_fast" "threads"; then
        converge "-T" "$TOP2_LO" "$TOP2_HI" --hybrid -1 "${hyb_gpu_knobs[@]}"
        CONV_THR_HYB="$CONV_BEST_VAL"
        CONV_HYB_MBPS="$CONV_BEST_MBPS"
        CONV_HYB_ARGS="--hybrid -1 ${hyb_gpu_knobs[*]+"${hyb_gpu_knobs[*]}"} -T $CONV_THR_HYB"
        echo "  ${GRN}✓${R}  Best -T (hybrid): ${BOLD}${CONV_THR_HYB}${R}  @ ${GRN}${BOLD}${CONV_HYB_MBPS}${R} MB/s"
    else
        echo "  ${YLW}  Not enough sweep data${R}"
    fi
fi

# ── Convergence summary ───────────────────────────────────────────────────────
echo ""
echo "  ${BOLD}${MGN} Convergence Results ${R}"
if [[ $GPU_WORKS -eq 1 ]]; then
    printf "  ${BOLD}${MGN}${R}  ${DIM}%-18s${R} ${WHT}${BOLD}%-28s${R} ${BOLD}${MGN}${R}
"         "--batch-size"       "$CONV_BATCH"
    printf "  ${BOLD}${MGN}${R}  ${DIM}%-18s${R} ${WHT}${BOLD}%-28s${R} ${BOLD}${MGN}${R}
"         "--streams-per-gpu"  "$CONV_STREAMS"
    printf "  ${BOLD}${MGN}${R}  ${DIM}%-18s${R} ${GRN}${BOLD}%-28s${R} ${BOLD}${MGN}${R}
"         "gpu-only peak"      "${CONV_GPU_MBPS} MB/s"
    printf "  ${BOLD}${MGN}${R}  ${DIM}%-18s${R} ${WHT}${BOLD}%-28s${R} ${BOLD}${MGN}${R}
"         "-T (hybrid)"        "$CONV_THR_HYB"
    printf "  ${BOLD}${MGN}${R}  ${GRN}${BOLD}%-18s${R} ${GRN}${BOLD}%-28s${R} ${BOLD}${MGN}${R}
"         "hybrid peak"        "${CONV_HYB_MBPS} MB/s"
fi
printf "  ${BOLD}${MGN}${R}  ${DIM}%-18s${R} ${WHT}${BOLD}%-28s${R} ${BOLD}${MGN}${R}
"     "-T (cpu-only)"      "$CONV_THR_CPU"
printf "  ${BOLD}${MGN}${R}  ${GRN}${BOLD}%-18s${R} ${GRN}${BOLD}%-28s${R} ${BOLD}${MGN}${R}
"     "cpu-only peak"      "${CONV_CPU_MBPS} MB/s"
echo "  ${BOLD}${MGN}${R}"
echo ""
if [[ $GPU_WORKS -eq 1 ]]; then
    echo "  ${DIM}GPU:    gzl4 ${CONV_GPU_ARGS} <file>${R}"
    echo "  ${DIM}Hybrid: gzl4 ${CONV_HYB_ARGS} <file>${R}"
fi
echo "  ${DIM}CPU:    gzl4 ${CONV_CPU_ARGS} <file>${R}"
echo ""

# Update global best with the best convergence result across all backends
_conv_mbps_list=("$CONV_GPU_MBPS" "$CONV_HYB_MBPS" "$CONV_CPU_MBPS")
_conv_args_list=("$CONV_GPU_ARGS" "$CONV_HYB_ARGS" "$CONV_CPU_ARGS")
for _ci in 0 1 2; do
    fgt "${_conv_mbps_list[$_ci]}" "$BEST_COMP_MBPS" && {
        BEST_COMP_MBPS="${_conv_mbps_list[$_ci]}"
        BEST_COMP_LABEL="converged"
        BEST_COMP_ARGS="${_conv_args_list[$_ci]}"
    }
done


# =============================================================================
# SUMMARY
# =============================================================================
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
ELAPSED_FMT=$(printf '%dm%02ds' $(( ELAPSED/60 )) $(( ELAPSED%60 )))

header "Results Summary"
echo "  ${BOLD}${GRN}Best compression:${R}"
echo "    ${WHT}${BOLD}${BEST_COMP_MBPS} MB/s${R}    ${CYN}${BEST_COMP_LABEL}${R}"
echo "    ${DIM}Recommended: gzl4 ${BEST_COMP_ARGS} <file>${R}"
echo ""
echo "  ${BOLD}${GRN}Best decompression:${R}"
echo "    ${WHT}${BOLD}${BEST_DECOMP_MBPS} MB/s${R}    ${CYN}${BEST_DECOMP_LABEL}${R}"
echo ""
echo "  ${DIM}Configs tested: ${CONFIG_NUM}  ·  Elapsed: ${ELAPSED_FMT}${R}"

# =============================================================================
# JSON
# =============================================================================
header "Writing Results"

GPU_NAMES_JSON="["
for (( g=0; g<${#GPU_NAMES[@]}; g++ )); do
    [[ $g -gt 0 ]] && GPU_NAMES_JSON+=","
    GPU_NAMES_JSON+=$(python3 -c \
        "import json,sys; print(json.dumps(sys.argv[1]))" "${GPU_NAMES[$g]}")
done
GPU_NAMES_JSON+="]"

RESULTS_JOINED=""
first=1
for entry in "${JSON_RESULTS[@]}"; do
    [[ $first -eq 0 ]] && RESULTS_JOINED+=","
    RESULTS_JOINED+="$entry"
    first=0
done

python3 - \
    "$OUT_JSON" "$TIMESTAMP" "$HOSTNAME" \
    "$CPU_MODEL" "$CPU_CORES" "$MEM_GB" \
    "$GZL4" "$GZL4_VERSION" \
    "$GPU_COUNT" "$GPU_NAMES_JSON" \
    "$BENCH_FILE" "$FILE_MB" \
    "$RUNS" "$WARMUP" "$ELAPSED" \
    "$BEST_COMP_MBPS" "$BEST_COMP_LABEL" "$BEST_COMP_ARGS" \
    "$BEST_DECOMP_MBPS" "$BEST_DECOMP_LABEL" \
    "[$RESULTS_JOINED]" <<'PEOF'
import json, sys
(out_path, ts, host, cpu, cores, mem,
 g4path, g4ver, gpucount, gpujson,
 bfile, fmb, runs, warmup, elapsed,
 bcmbps, bclabel, bcargs,
 bdmbps, bdlabel, rjson) = sys.argv[1:]
doc = {
    "benchmark": {
        "timestamp": ts, "elapsed_seconds": int(elapsed),
        "runs_per_config": int(runs), "warmup_runs": int(warmup),
        "file": bfile, "file_mb": float(fmb),
    },
    "system": {
        "hostname": host, "cpu_model": cpu,
        "cpu_cores": int(cores), "ram_gb": mem,
        "gpu_count": int(gpucount), "gpus": json.loads(gpujson),
    },
    "gzl4": {"path": g4path, "version": g4ver},
    "best": {
        "compression":   {"mbps": float(bcmbps), "label": bclabel, "args": bcargs},
        "decompression": {"mbps": float(bdmbps), "label": bdlabel},
    },
    "results": json.loads(rjson),
}
with open(out_path, 'w') as f:
    json.dump(doc, f, indent=2)
PEOF

ok "JSON written: ${BOLD}${OUT_JSON}${R}"
echo ""
echo "  ${BOLD}${MGN}Benchmark complete.${R}  ${DIM}${ELAPSED_FMT}${R}"
echo ""
