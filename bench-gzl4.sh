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
#   --sparse        use zero-heavy synthetic data to exercise sparse-write
#                   and zero-chunk routing optimisations (alternating
#                   256 MB zero / 256 MB random blocks)
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
exec 3>&1 1>&2

# ── Ctrl-C / cleanup ──────────────────────────────────────────────────────────
WORKDIR=""
CHILD_PID=""

cleanup() {
    echo ""
    echo ""
    echo "  ${YLW}⚠  Interrupted  cleaning up${R}"
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
SPARSE=0       # --sparse: generate zero-heavy synthetic data

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
  ${CYN}--sparse${R}       zero-heavy data: exercises sparse-write (decompress)
                 and zero-chunk GPU bypass (hybrid compress).
                 Generates alternating 256 MB zero / 256 MB random blocks.
                 Use with ${BOLD}--size-mb${R} to control total size (must be a
                 multiple of 512 MB for clean alternation; rounded up if not).
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
        --sparse)  SPARSE=1;        shift   ;;
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

BARW=28
_bar_full=""
_bar_empty=""
for (( _i=0; _i<BARW; _i++ )); do _bar_full+="="; _bar_empty+="-"; done

GZL4_SUB_PCT=""
_current_label=""

draw_bar() {
    local cur=$1 tot=$2 label="${3:0:26}"
    local pct=$(( cur * 100 / tot ))
    local fill=$(( cur * BARW / tot ))
    local emp=$(( BARW - fill ))
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

print_row() {
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

# ── Sparse-specific helpers ───────────────────────────────────────────────────

# After each decompress run, parse gzl4 -v output for sparse stats.
# Returns the sparse percentage (0 if none reported).
LAST_SPARSE_PCT="0"
parse_sparse_pct() {
    local log="$1"
    # Look for: "Sparse holes: X.X GB skipped (YY.Y% of output)"
    local line
    line=$(grep -oP 'Sparse holes:.*?\(\K[0-9.]+(?=%)' "$log" 2>/dev/null | tail -1 || echo "")
    LAST_SPARSE_PCT="${line:-0}"
}

# print_sparse_row: like print_row but appends the sparse percentage.
print_sparse_row() {
    local label="$1" mbps="$2" ratio="$3" best="$4" sparse_pct="$5"
    local pct=100
    awk "BEGIN{exit !($best+0 > 0)}" 2>/dev/null && \
        pct=$(awk "BEGIN{printf \"%d\", ($mbps+0)/($best+0)*100}" 2>/dev/null) || true
    local color
    if   [[ $pct -ge 95 ]]; then color="${GRN}${BOLD}"
    elif [[ $pct -ge 75 ]]; then color="${YLW}"
    elif [[ $pct -ge 50 ]]; then color="${WHT}"
    else                         color="${DIM}"
    fi
    local sparse_col=""
    if awk "BEGIN{exit !($sparse_pct+0 > 0)}" 2>/dev/null; then
        sparse_col=$(printf "  ${CYN}sparse:${GRN}%5.1f%%${R}" "$sparse_pct")
    fi
    printf "  %-36s  %s%8.1f${R}  ${DIM}MB/s${R}  ${YLW}%6.2f%%${R}%s\n" \
        "${label:0:36}" "$color" "$mbps" "$ratio" "$sparse_col"
}

# ── Timing ────────────────────────────────────────────────────────────────────
now_ms() { date +%s%3N; }

# ── run_compress ──────────────────────────────────────────────────────────────
RC_MBPS="0"
RC_RATIO="0"
run_compress() {
    RC_MBPS="0"; RC_RATIO="0"
    local _in _out _total_ms _r _t0 _t1 _avg
    _in=$(stat -c%s "$BENCH_FILE")

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

    _total_ms=0
    GZL4_SUB_PCT="0"
    for (( _r=0; _r<RUNS; _r++ )); do
        > "$GZL4_STDERR"
        _t0=$(now_ms)
        "$GZL4" "$@" -c "$BENCH_FILE" > "$COMPRESSED" 2>"$GZL4_STDERR" &
        CHILD_PID=$!
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

# ── run_decompress ────────────────────────────────────────────────────────────
# For sparse mode we decompress to a real temp file so the OS actually creates
# the sparse file and we can measure it.  For normal mode we decompress to
# /dev/null as before.
DECOMP_OUT=""
run_decompress() {
    RC_MBPS="0"
    local _in _total_ms _r _t0 _t1 _avg _w
    _in=$(stat -c%s "$BENCH_FILE")

    "$GZL4" -q "$@" -c "$BENCH_FILE" > "$COMPRESSED" 2>/dev/null || return 1

    # Determine output target.
    # Non-sparse mode: decompress to /dev/null  we only care about throughput,
    # not the output content, and canSparse is always false for /dev/null
    # (lseek on /dev/null is harmless but pointless).
    #
    # Sparse mode: decompress to a real named file so that:
    #   1. gzl4's lseek() probe on the fd succeeds (canSparse = true)
    #   2. The OS actually creates sparse holes we can measure with du
    # We use stdout redirect (-c > file) since the v3.27.1 canSparse fix
    # correctly handles seekable stdout via lseek(STDOUT_FILENO, 0, SEEK_CUR).
    if [[ $SPARSE -eq 1 ]]; then
        DECOMP_OUT="$WORKDIR/decomp_out"
    else
        DECOMP_OUT="/dev/null"
    fi

    for (( _w=0; _w<WARMUP; _w++ )); do
        [[ $SPARSE -eq 1 ]] && rm -f "$DECOMP_OUT"
        "$GZL4" -q -dc "$COMPRESSED" > "$DECOMP_OUT" 2>/dev/null || true
    done

    _total_ms=0
    for (( _r=0; _r<RUNS; _r++ )); do
        [[ $SPARSE -eq 1 ]] && rm -f "$DECOMP_OUT"
        _t0=$(now_ms)
        "$GZL4" -q -dc "$COMPRESSED" > "$DECOMP_OUT" 2>/dev/null &
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

# ── Sparse verification helper ────────────────────────────────────────────────
# Reports logical size vs physical disk usage for the last decomp output.
# Only meaningful when DECOMP_OUT is a real file (not /dev/null).
check_sparse_savings() {
    [[ $SPARSE -eq 0 || "$DECOMP_OUT" == "/dev/null" || ! -f "$DECOMP_OUT" ]] && return
    local logical physical savings_pct
    logical=$(stat -c%s "$DECOMP_OUT" 2>/dev/null || echo 0)
    # stat -c%b gives 512-byte blocks actually allocated on disk.
    # Multiplying by 512 gives the true physical footprint.
    # du -b would give apparent/logical size  useless for detecting holes.
    physical=$(( $(stat -c%b "$DECOMP_OUT" 2>/dev/null || echo 0) * 512 ))
    if [[ $logical -gt 0 ]]; then
        savings_pct=$(awk "BEGIN{printf \"%.1f\", 100*(1 - $physical/$logical)}")
        local phys_fmt logical_fmt
        phys_fmt=$(numfmt --to=iec-i --suffix=B "$physical" 2>/dev/null || echo "${physical}B")
        logical_fmt=$(numfmt --to=iec-i --suffix=B "$logical" 2>/dev/null || echo "${logical}B")
        printf "  ${DIM}  sparse check: %s logical, %s physical  (${GRN}%.1f%% holes${R})\n" \
            "$logical_fmt" "$phys_fmt" "$savings_pct"
    fi
}

# ── Result tracking ───────────────────────────────────────────────────────────
BEST_COMP_MBPS="0"
BEST_COMP_LABEL=""
BEST_COMP_ARGS=""
BEST_DECOMP_MBPS="0"
BEST_DECOMP_LABEL=""
CONFIG_NUM=0
JSON_RESULTS=()

_gt() { awk "BEGIN{exit !($1+0 > $2+0)}"; }

# ── Core per-config runner ────────────────────────────────────────────────────
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

        _gt "$mbps" "$pb_val" && printf -v "$pb_var" '%s' "$mbps" || true
        _gt "$mbps" "$BEST_COMP_MBPS" && {
            BEST_COMP_MBPS="$mbps"
            BEST_COMP_LABEL="$label"
            BEST_COMP_ARGS="$args_str"
        } || true

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

        # In sparse mode, show the sparse savings for this backend
        [[ $SPARSE -eq 1 ]] && check_sparse_savings

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
[[ $SPARSE -eq 1 ]] && info "${CYN}Sparse mode:${R} zero-heavy data  exercises sparse-write + zero-chunk routing"

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
    if [[ $SPARSE -eq 1 ]]; then
        warn "--sparse ignored when --file is specified (using provided file as-is)"
        SPARSE=0
    fi
elif [[ $SPARSE -eq 1 ]]; then
    # ── Sparse synthetic data ──────────────────────────────────────────────────
    # Generate alternating 256 MB zero blocks and 256 MB random blocks.
    # This exercises two v3.27.0 optimisations simultaneously:
    #
    #   Compression (hybrid):   zero chunks are routed to CPU workers,
    #     bypassing the GPU PCIe round-trip.  Random chunks go to the GPU.
    #     -v output will show GPU%/CPU% split; expect ~50/50.
    #
    #   Decompression (all backends):   zero blocks trigger lseek() instead
    #     of write(), punching sparse holes in the output file.  The result
    #     file has the correct logical size but ~50% physical disk usage.
    #
    # Block size: 256 MB is large enough to hold many gzl4 chunks and
    # small enough that the generator finishes in a few seconds.
    # SIZE_MB is rounded up to the nearest 512 MB for clean alternation.
    SPARSE_BLOCK_MB=256
    PAIRS=$(( (SIZE_MB + 511) / 512 ))   # ceiling division
    ACTUAL_MB=$(( PAIRS * 512 ))
    BENCH_FILE="$WORKDIR/bench_sparse.bin"
    printf "  ${DIM}▸${R} Generating ${BOLD}${ACTUAL_MB} MB${R} of sparse synthetic data "
    printf "(${PAIRS}x 256 MB zero + ${PAIRS}x 256 MB random) ... "
    python3 - "$BENCH_FILE" "$PAIRS" "$SPARSE_BLOCK_MB" <<'PEOF'
import sys, os
path, pairs, block_mb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
block = block_mb * 1024 * 1024
with open(path, 'wb') as f:
    for _ in range(pairs):
        # Zero block: exercises lseek sparse-write on decompress,
        # and zero-chunk GPU bypass on hybrid compress.
        f.write(b'\x00' * block)
        # Random block: incompressible, ensures the ratio stat is meaningful
        # and that GPU compression actually gets exercised.
        f.write(os.urandom(block))
PEOF
    FILE_SIZE=$(stat -c%s "$BENCH_FILE")
    FILE_MB=$(awk "BEGIN{printf \"%.1f\", $FILE_SIZE/1048576}")
    echo "${GRN}done${R}  (${FILE_MB} MB)"
    info "  Zero regions:   ${BOLD}$(( ACTUAL_MB / 2 )) MB${R}  (~50%% of file)"
    info "  Random regions: ${BOLD}$(( ACTUAL_MB / 2 )) MB${R}  (~50%% of file)"
else
    # ── Normal synthetic data (original generator) ─────────────────────────────
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

# ── Sparse smoke test ──────────────────────────────────────────────────────────
# Verify that the sparse-write optimisation actually fires by decompressing
# to a real file and checking that physical disk usage is less than logical size.
# Skip when /dev/null is the decomp target (non-sparse mode) or on filesystems
# that don't support sparse files (rare but possible in Docker/tmpfs setups).
if [[ $SPARSE -eq 1 ]]; then
    printf "  ${DIM}▸${R} Sparse-write smoke test ... "
    SPARSE_SMOKE="$WORKDIR/sparse_smoke_out"
    if "$GZL4" -q --cpu-only -dc "$COMPRESSED" > "$SPARSE_SMOKE" 2>/dev/null; then
        LOGICAL=$(stat -c%s "$SPARSE_SMOKE" 2>/dev/null || echo 0)
        PHYSICAL=$(( $(stat -c%b "$SPARSE_SMOKE" 2>/dev/null || echo 0) * 512 ))
        if [[ $LOGICAL -gt 0 ]] && [[ $PHYSICAL -lt $LOGICAL ]]; then
            HOLE_PCT=$(awk "BEGIN{printf \"%.0f\", 100*(1 - $PHYSICAL/$LOGICAL)}")
            echo "${GRN}OK${R}  (${HOLE_PCT}% sparse on this filesystem)"
        elif [[ $LOGICAL -eq $PHYSICAL ]]; then
            echo "${YLW}filesystem may not support sparse files  holes will still be skipped but no disk savings${R}"
        else
            echo "${GRN}OK${R}"
        fi
        rm -f "$SPARSE_SMOKE"
    else
        echo "${YLW}skipped (decompress failed  sparse stats will still be shown in results)${R}"
    fi
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
n_conv=0; [[ $GPU_WORKS -eq 1 ]] && n_conv=18
TOTAL_CONFIGS=$(( n_cpu + n_hc + n_gpu + n_hyb + n_batch + n_streams + n_decomp + n_conv ))

info "CPU fast:      ${BOLD}$n_cpu${R} configs"
[[ $n_hc -gt 0 ]]      && info "CPU HC:        ${BOLD}$n_hc${R} configs"
[[ $GPU_WORKS -eq 1 ]] && info "GPU/Hybrid:    ${BOLD}$(( n_gpu + n_hyb + n_batch + n_streams ))${R} configs"
info "Decompression: ${BOLD}$n_decomp${R} configs"
[[ $GPU_WORKS -eq 1 ]] && info "Convergence:   ${BOLD}~$n_conv${R} configs (adaptive)"
info "Total:         ${BOLD}~$TOTAL_CONFIGS${R}  (warmup=$WARMUP, timed runs=$RUNS)"
[[ $SPARSE -eq 1 ]] && info "Sparse mode:   decompression results include hole-punch savings"

# =============================================================================
# PHASES
# =============================================================================
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
    if [[ $SPARSE -eq 1 ]]; then
        printf "  ${DIM}(zero chunks routed to CPU  expect higher CPU%% vs normal data)${R}\n"
    fi
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
if [[ $SPARSE -eq 1 ]]; then
    printf "  ${DIM}(sparse mode: decompressing to real files; hole-punch savings shown per backend)${R}\n"
    printf "  ${DIM}%-36s  %8s  %-4s  %s${R}\n" "Configuration" "" "MB/s" "Sparse savings"
else
    printf "  ${DIM}%-36s  %8s  %-4s${R}\n" "Configuration" "" "MB/s"
fi
printf "  ${DIM}%s${R}\n" "$(printf '─%.0s' {1..52})"

decomp_run "cpu-only"     "cpu-only" PB_DECOMP  --cpu-only
if [[ $GPU_WORKS -eq 1 ]]; then
    decomp_run "gpu-only"     "gpu-only" PB_DECOMP  --gpu-only
    decomp_run "hybrid"       "hybrid"   PB_DECOMP  --hybrid
fi
echo "  ${DIM}Phase best: ${GRN}${BOLD}${PB_DECOMP} MB/s${R}"

# =============================================================================
# PHASE 8: Best Performance Convergence
# =============================================================================
CONV_RESULT="0"
cm() {
    local _v="$1"; shift
    run_compress "$@" && CONV_RESULT="$RC_MBPS" || CONV_RESULT="0"
    printf -v "$_v" '%s' "$CONV_RESULT"
}

fgt() { awk "BEGIN{exit !(${1:-0}+0 > ${2:-0}+0)}"; }

CONV_BEST_VAL=""
CONV_BEST_MBPS="0"

_probe() {
    local _flag="$1" _val="$2"; shift 2
    local _m="0"
    (( CONFIG_NUM++ )) || true
    _current_label="${_flag}=${_val}"
    draw_bar "$CONFIG_NUM" "$TOTAL_CONFIGS" "$_current_label"
    cm _m "$@" "$_flag" "$_val"
    erase_bar
    local _col="$DIM"
    fgt "$_m" "$CONV_BEST_MBPS" && {
        _col="${GRN}${BOLD}"
        CONV_BEST_MBPS="$_m"
        CONV_BEST_VAL="$_val"
    }
    printf "  %-34s  %s%8.1f${R}  ${DIM}MB/s${R}\n" "${_flag}=${_val}" "$_col" "$_m"
    CONV_RESULT="$_m"
}

converge() {
    local flag="$1" lo="$2" hi="$3"; shift 3
    local base=("$@")
    CONV_BEST_VAL="$lo"
    CONV_BEST_MBPS="0"

    local m_lo m_hi
    _probe "$flag" "$lo" "${base[@]}"; m_lo="$CONV_RESULT"
    _probe "$flag" "$hi" "${base[@]}"; m_hi="$CONV_RESULT"

    while (( hi - lo > 1 )); do
        local mid=$(( (lo + hi) / 2 ))
        local m_mid
        _probe "$flag" "$mid" "${base[@]}"; m_mid="$CONV_RESULT"

        if fgt "$(awk 'BEGIN{printf "%.3f", ('${m_lo:-0}'+'${m_mid:-0}')/2}')" \
               "$(awk 'BEGIN{printf "%.3f", ('${m_mid:-0}'+'${m_hi:-0}')/2}')"; then
            hi="$mid"; m_hi="$m_mid"
        else
            lo="$mid"; m_lo="$m_mid"
        fi
    done
}

TOP2_LO="" TOP2_HI=""
top2() {
    local want_phase="$1" want_field="$2"
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
        pairs+="$v $m"$'\n'
    done
    [[ -z "$pairs" ]] && { TOP2_LO=""; TOP2_HI=""; return 1; }
    local top
    top=$(echo "$pairs" | sort -k2 -rn | awk '!seen[$1]++ && ++n<=2 {print $1}')
    local a b
    read -r a < <(echo "$top" | head -1)
    read -r b < <(echo "$top" | tail -1 | grep -v "^$a$" || true)
    [[ -z "$a" || -z "$b" || "$a" == "$b" ]] && { TOP2_LO=""; TOP2_HI=""; return 1; }
    if (( a < b )); then TOP2_LO="$a"; TOP2_HI="$b"
    else                 TOP2_LO="$b"; TOP2_HI="$a"; fi
    return 0
}

section "Phase 8 · Best Performance Convergence"
echo "  ${DIM}Binary-searching optimal knobs using sweep results as brackets${R}"
printf "  %s%-34s  %8s  %-4s%s\n" "$DIM" "Configuration" "" "MB/s" "$R"
printf "  %s%s%s\n" "$DIM" "$(printf '─%.0s' {1..50})" "$R"

CONV_BATCH="auto"
CONV_STREAMS="auto"
CONV_THR_CPU="$CPU_CORES"
CONV_THR_HYB="$CPU_CORES"
CONV_GPU_MBPS="0"
CONV_HYB_MBPS="0"
CONV_CPU_MBPS="0"
CONV_GPU_ARGS="--gpu-only -1"
CONV_HYB_ARGS="--hybrid -1"
CONV_CPU_ARGS="--cpu-only -1"

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
        echo "  ${YLW}  Not enough sweep data${R}"
    fi
else
    echo "  ${DIM}  Skipped (no GPU available)${R}"
fi

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
        converge "--streams-per-gpu" "$TOP2_LO" "$TOP2_HI" \
            --gpu-only -1 "${batch_lock[@]}"
        CONV_STREAMS="$CONV_BEST_VAL"
        fgt "$CONV_BEST_MBPS" "$CONV_GPU_MBPS" && {
            CONV_GPU_MBPS="$CONV_BEST_MBPS"
            CONV_GPU_ARGS="--gpu-only -1 ${batch_lock[*]+"${batch_lock[*]}"} --streams-per-gpu $CONV_STREAMS"
        }
        echo "  ${GRN}✓${R}  Best --streams-per-gpu: ${BOLD}${CONV_STREAMS}${R}  @ ${GRN}${BOLD}${CONV_BEST_MBPS}${R} MB/s"
    else
        echo "  ${YLW}  Not enough sweep data${R}"
    fi
else
    echo "  ${DIM}  Skipped (no GPU available)${R}"
fi

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

if [[ $GPU_WORKS -eq 1 ]]; then
    echo ""
    echo "  ${BOLD}${YLW}▸ Step 4:${R}  -T thread count  (hybrid -1, using best batch/streams from steps 1+2)"
    hyb_gpu_knobs=()
    [[ "$CONV_BATCH"   != "auto" ]] && hyb_gpu_knobs+=(--batch-size "$CONV_BATCH")
    [[ "$CONV_STREAMS" != "auto" ]] && hyb_gpu_knobs+=(--streams-per-gpu "$CONV_STREAMS")
    [[ ${#hyb_gpu_knobs[@]} -gt 0 ]] \
        && echo "  ${DIM}  locking in: ${hyb_gpu_knobs[*]}${R}" \
        || echo "  ${DIM}  no GPU knobs to lock${R}"
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

echo ""
echo "  ${BOLD}${MGN} Convergence Results ${R}"
if [[ $GPU_WORKS -eq 1 ]]; then
    printf "  ${BOLD}${MGN}${R}  ${DIM}%-18s${R} ${WHT}${BOLD}%-28s${R} ${BOLD}${MGN}${R}\n" "--batch-size"       "$CONV_BATCH"
    printf "  ${BOLD}${MGN}${R}  ${DIM}%-18s${R} ${WHT}${BOLD}%-28s${R} ${BOLD}${MGN}${R}\n" "--streams-per-gpu"  "$CONV_STREAMS"
    printf "  ${BOLD}${MGN}${R}  ${DIM}%-18s${R} ${GRN}${BOLD}%-28s${R} ${BOLD}${MGN}${R}\n" "gpu-only peak"      "${CONV_GPU_MBPS} MB/s"
    printf "  ${BOLD}${MGN}${R}  ${DIM}%-18s${R} ${WHT}${BOLD}%-28s${R} ${BOLD}${MGN}${R}\n" "-T (hybrid)"        "$CONV_THR_HYB"
    printf "  ${BOLD}${MGN}${R}  ${GRN}${BOLD}%-18s${R} ${GRN}${BOLD}%-28s${R} ${BOLD}${MGN}${R}\n" "hybrid peak"        "${CONV_HYB_MBPS} MB/s"
fi
printf "  ${BOLD}${MGN}${R}  ${DIM}%-18s${R} ${WHT}${BOLD}%-28s${R} ${BOLD}${MGN}${R}\n" "-T (cpu-only)"      "$CONV_THR_CPU"
printf "  ${BOLD}${MGN}${R}  ${GRN}${BOLD}%-18s${R} ${GRN}${BOLD}%-28s${R} ${BOLD}${MGN}${R}\n" "cpu-only peak"      "${CONV_CPU_MBPS} MB/s"
echo "  ${BOLD}${MGN}${R}"
echo ""
if [[ $GPU_WORKS -eq 1 ]]; then
    echo "  ${DIM}GPU:    gzl4 ${CONV_GPU_ARGS} <file>${R}"
    echo "  ${DIM}Hybrid: gzl4 ${CONV_HYB_ARGS} <file>${R}"
fi
echo "  ${DIM}CPU:    gzl4 ${CONV_CPU_ARGS} <file>${R}"
echo ""

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
if [[ $SPARSE -eq 1 ]]; then
    echo "  ${BOLD}${CYN}Sparse mode notes:${R}"
    echo "    ${DIM}Decompression ran to real files.  du vs ls -l on output shows${R}"
    echo "    ${DIM}actual hole savings.  Expect ~50%% sparse with this dataset.${R}"
    echo "    ${DIM}Hybrid compress: zero chunks routed to CPU, random to GPU.${R}"
    echo ""
fi
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
    "$SPARSE" \
    "[$RESULTS_JOINED]" <<'PEOF'
import json, sys
(out_path, ts, host, cpu, cores, mem,
 g4path, g4ver, gpucount, gpujson,
 bfile, fmb, runs, warmup, elapsed,
 bcmbps, bclabel, bcargs,
 bdmbps, bdlabel, sparse_mode, rjson) = sys.argv[1:]
doc = {
    "benchmark": {
        "timestamp": ts, "elapsed_seconds": int(elapsed),
        "runs_per_config": int(runs), "warmup_runs": int(warmup),
        "file": bfile, "file_mb": float(fmb),
        "sparse_mode": sparse_mode == "1",
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
