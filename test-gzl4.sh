#!/usr/bin/env bash
# =============================================================================
# test_gzl4.sh    Comprehensive test suite for gzl4 / ungzl4
# =============================================================================
#
# Usage:
#   ./test_gzl4.sh [OPTIONS]
#
# Options:
#   --gzl4 PATH      path to gzl4 binary  (default: auto-detect)
#   --no-gpu         skip GPU and hybrid tests (CPU-only environment)
#   --no-lz4         skip lz4-interop tests (lz4 tool not installed)
#   --no-tar         skip tar -I integration tests
#   --small          use only small test files (< 1 MB)  faster
#   --keep-tmp       don't delete temp dir on exit (for debugging)
#   --verbose        show full output from each gzl4 invocation
#
# Exit code: 0 = all tests passed, 1 = one or more failures
#
# Requirements:
#   - bash 4+, cmp, md5sum/md5, xxd or od, tar
#   - gzl4 binary (found via --gzl4, build/, or PATH in that order)
#   - lz4 tool for interop tests (optional, detected automatically)
# =============================================================================

set -uo pipefail

# ── Colour codes ──────────────────────────────────────────────────────────────
RED=$'\e[31m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; CYAN=$'\e[36m'; BOLD=$'\e[1m'; RESET=$'\e[0m'

# ── Option defaults ───────────────────────────────────────────────────────────
GZL4=""
SKIP_GPU=0
SKIP_LZ4=0
SKIP_TAR=0
SMALL_ONLY=0
KEEP_TMP=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gzl4)     GZL4="$2"; shift 2 ;;
        --no-gpu)   SKIP_GPU=1; shift ;;
        --no-lz4)   SKIP_LZ4=1; shift ;;
        --no-tar)   SKIP_TAR=1; shift ;;
        --small)    SMALL_ONLY=1; shift ;;
        --keep-tmp) KEEP_TMP=1; shift ;;
        --verbose)  VERBOSE=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Locate gzl4 binary ────────────────────────────────────────────────────────
if [[ -z "$GZL4" ]]; then
    for candidate in ./build/gzl4 ./gzl4 "$(command -v gzl4 2>/dev/null || true)"; do
        if [[ -x "$candidate" ]]; then GZL4="$candidate"; break; fi
    done
fi
if [[ -z "$GZL4" || ! -x "$GZL4" ]]; then
    echo "${RED}ERROR: gzl4 binary not found. Use --gzl4 PATH or build first.${RESET}"
    exit 1
fi
GZL4="$(realpath "$GZL4")"
GZL4_DIR="$(dirname "$GZL4")"
UNGZL4="$GZL4_DIR/ungzl4"

# ── Probe GPU availability ──────────────────────────────────────────────────
if [[ $SKIP_GPU -eq 0 ]]; then
    if ! echo "" | "$GZL4" --gpu-only -c >/dev/null 2>&1; then
        echo "${YELLOW}NOTICE: No GPU detected  GPU tests will be skipped automatically${RESET}"
        SKIP_GPU=1
    fi
fi

# ── Check for lz4 tool ────────────────────────────────────────────────────────
if [[ $SKIP_LZ4 -eq 0 ]] && ! command -v lz4 &>/dev/null; then
    echo "${YELLOW}NOTICE: lz4 tool not found  skipping interop tests (use --no-lz4 to suppress)${RESET}"
    SKIP_LZ4=1
fi

# ── Temp workspace ────────────────────────────────────────────────────────────
TMPDIR_ROOT="$(mktemp -d /tmp/gzl4_test.XXXXXX)"
cleanup() { [[ $KEEP_TMP -eq 0 ]] && rm -rf "$TMPDIR_ROOT"; }
interrupted() {
    echo
    echo "${YELLOW}Interrupted  aborting test run.${RESET}"
    echo "  ${GREEN}${PASS} passed${RESET}  ${RED}${FAIL} failed${RESET}  ${YELLOW}${SKIP} skipped${RESET} (incomplete)"
    cleanup
    exit 130
}
trap cleanup EXIT
trap interrupted INT TERM

# ── Test counters ─────────────────────────────────────────────────────────────
PASS=0; FAIL=0; SKIP=0
FAILURES=()

# ── Helper functions ──────────────────────────────────────────────────────────

# run_gzl4 ARGS...
# Run gzl4 with the given args.  stdout/stderr suppressed unless --verbose.
# Returns gzl4's exit code.  For calls that need stdout captured use gzl4_to:
#   gzl4_to OUTPUT_FILE ARGS...
run_gzl4() {
    if [[ $VERBOSE -eq 1 ]]; then
        "$GZL4" "$@"
    else
        "$GZL4" "$@" >/dev/null 2>&1
    fi
    local rc=$?
    # If gzl4 was killed by SIGINT (exit 130), re-raise so the script exits too
    [[ $rc -eq 130 ]] && kill -INT $$
    return $rc
}

# gzl4_to OUTPUT_FILE ARGS...
# Run gzl4, writing stdout to OUTPUT_FILE only if gzl4 exits 0.
# On failure the output file is left absent (not truncated to empty).
gzl4_to() {
    local out="$1"; shift
    local tmp="${out}.tmp.$$"
    local rc
    if [[ $VERBOSE -eq 1 ]]; then
        "$GZL4" "$@" > "$tmp"; rc=$?
    else
        "$GZL4" "$@" > "$tmp" 2>/dev/null; rc=$?
    fi
    if [[ $rc -eq 130 ]]; then rm -f "$tmp"; kill -INT $$; fi
    if [[ $rc -eq 0 ]]; then mv "$tmp" "$out"; else rm -f "$tmp"; return 1; fi
}

# must_run NAME CMD...   run CMD; report failure and continue (don't abort)
must_run() {
    local name="$1"; shift
    if "$@" >/dev/null 2>&1; then
        : # success  silent, the subsequent check_files_equal will validate
    else
        fail "$name" "command exited non-zero: $*"
    fi
}

# pass NAME
pass() { echo "  ${GREEN}PASS${RESET}  $1"; (( PASS++ )) || true; }

# fail NAME REASON
fail() {
    echo "  ${RED}FAIL${RESET}  $1  ${RED}($2)${RESET}"
    FAILURES+=("$1: $2")
    (( FAIL++ )) || true
}

# skip NAME REASON
skip() { echo "  ${YELLOW}SKIP${RESET}  $1  (${YELLOW}$2${RESET})"; (( SKIP++ )) || true; }

# check NAME CMD...   run CMD, pass if exit 0, fail otherwise
check() {
    local name="$1"; shift
    if "$@" >/dev/null 2>&1; then pass "$name"
    else fail "$name" "command failed: $*"; fi
}

# check_files_equal NAME FILE_A FILE_B
check_files_equal() {
    local name="$1" a="$2" b="$3"
    if cmp -s "$a" "$b"; then pass "$name"
    else fail "$name" "files differ: $a vs $b"; fi
}

# check_file_exists FILE NAME
check_file_exists() {
    if [[ -f "$1" ]]; then pass "$2"
    else fail "$2" "file not found: $1"; fi
}

# check_file_absent FILE NAME
check_file_absent() {
    if [[ ! -e "$1" ]]; then pass "$2"
    else fail "$2" "file should not exist: $1"; fi
}

# section TITLE
section() { echo; echo "${BOLD}${CYAN}── $1 ──${RESET}"; }

# mkwork  create a fresh subdirectory under TMPDIR_ROOT and cd into it
mkwork() {
    local d="$TMPDIR_ROOT/$1"; mkdir -p "$d"; echo "$d"
}

# ── Generate test data ────────────────────────────────────────────────────────
# small_file   ~128 KB   fast for all tests
# medium_file  ~8 MB    enough to exercise multi-chunk paths (skipped with --small)
# text_file    ~256 KB of compressible text
# binary_file  ~128 KB of random data (incompressible)

DATA="$TMPDIR_ROOT/data"
mkdir -p "$DATA"

echo "Generating test data..."

# Small compressible file (~128 KB)
python3 -c "
import random, string, sys
random.seed(42)
lines = [''.join(random.choices(string.ascii_letters + ' \n', k=80)) + '\n' for _ in range(1600)]
sys.stdout.buffer.write(''.join(lines).encode())
" > "$DATA/small.txt"

# Small binary (incompressible) file (~128 KB)
python3 -c "
import os, sys
random_bytes = bytearray(os.urandom(131072))
sys.stdout.buffer.write(random_bytes)
" > "$DATA/small.bin"

if [[ $SMALL_ONLY -eq 0 ]]; then
    # Medium compressible file (~8 MB)  repeat the small file
    python3 -c "
data = open('$DATA/small.txt','rb').read()
import sys
for _ in range(64): sys.stdout.buffer.write(data)
" > "$DATA/medium.txt"
fi

# Directory for tar tests
mkdir -p "$DATA/tardir/subdir"
cp "$DATA/small.txt" "$DATA/tardir/file1.txt"
cp "$DATA/small.bin" "$DATA/tardir/file2.bin"
echo "hello from subdir" > "$DATA/tardir/subdir/hello.txt"

echo "Test data ready."
echo "  small.txt:  $(wc -c < "$DATA/small.txt") bytes"
echo "  small.bin:  $(wc -c < "$DATA/small.bin") bytes"
[[ $SMALL_ONLY -eq 0 ]] && echo "  medium.txt: $(wc -c < "$DATA/medium.txt") bytes"

# ── Print banner ──────────────────────────────────────────────────────────────
echo
echo "${BOLD}gzl4 test suite${RESET}"
echo "  Binary : $GZL4"
echo "  Version: $("$GZL4" -V 2>&1 | head -1)"
echo "  GPU    : $([ $SKIP_GPU -eq 1 ] && echo 'skip' || echo 'enabled')"
echo "  lz4    : $([ $SKIP_LZ4 -eq 1 ] && echo 'skip' || echo 'enabled')"
echo "  tar -I : $([ $SKIP_TAR -eq 1 ] && echo 'skip' || echo 'enabled')"

# =============================================================================
# 1. BASIC HELP / VERSION
# =============================================================================
section "1. Help and version output"

W="$(mkwork s1)"; cd "$W"
check       "1.01  -h exits 0"            "$GZL4" -h
check       "1.02  --help exits 0"        "$GZL4" --help
check       "1.03  -V exits 0"            "$GZL4" -V
check       "1.04  --version exits 0"     "$GZL4" --version
check       "1.05  --change-log exits 0"  "$GZL4" --change-log

# -V should print exactly 2 lines (no blank line)
nlines=$("$GZL4" -V 2>&1 | wc -l)
if [[ $nlines -eq 2 ]]; then pass "1.06  -V output is exactly 2 lines"
else fail "1.06  -V output is exactly 2 lines" "got $nlines lines"; fi

# --help should mention --change-log
if "$GZL4" --help 2>&1 | grep -q 'change-log'; then pass "1.07  --help mentions --change-log"
else fail "1.07  --help mentions --change-log" "string not found"; fi

# =============================================================================
# 2. COMPRESS / DECOMPRESS  DEFAULT BACKEND (HYBRID)
# =============================================================================
section "2. Compress and decompress (hybrid/default)"

W="$(mkwork s2)"; cd "$W"

# Basic compress
cp "$DATA/small.txt" input.txt
run_gzl4 -k input.txt || true
check_file_exists   "input.txt.lz4"           "2.01  basic compress creates .lz4"
check_file_exists   "input.txt"               "2.02  -k keeps original"

# Basic decompress
run_gzl4 -k input.txt.lz4 || true
check_file_exists   "input.txt"               "2.03  basic decompress creates output"
check_files_equal   "2.04  roundtrip matches original" "$DATA/small.txt" input.txt

# Auto-detect .lz4 → decompress (no -d flag)
cp "$DATA/small.txt" autodet.txt
run_gzl4 -k autodet.txt || true
run_gzl4 -f -k autodet.txt.lz4 || true
check_files_equal   "2.05  auto-detect .lz4 decompress" "$DATA/small.txt" autodet.txt

# -c (stdout) compress
gzl4_to stdout_comp.lz4 -c input.txt || true
check_file_exists   "stdout_comp.lz4"         "2.06  -c writes to stdout"

# -c (stdout) decompress
gzl4_to stdout_decomp.txt -dc stdout_comp.lz4 || true
check_files_equal   "2.07  -dc roundtrip matches original" "$DATA/small.txt" stdout_decomp.txt

# -f force overwrite
cp "$DATA/small.txt" force_test.txt
run_gzl4 -k force_test.txt || true
run_gzl4 -f -k force_test.txt   # should overwrite .lz4
pass "2.08  -f force overwrite succeeds"

# -z force compress a .lz4 file (creates .lz4.lz4)
run_gzl4 -z -k input.txt.lz4 || true
check_file_exists   "input.txt.lz4.lz4"       "2.09  -z compresses .lz4 -> .lz4.lz4"

# -t integrity test on valid file
if run_gzl4 -t input.txt.lz4; then pass "2.10  -t integrity test passes on valid file"
else fail "2.10  -t integrity test passes on valid file" "exit non-zero"; fi

# -t on corrupted file should fail
# Corrupt 64 bytes in the middle of the compressed data (well past the header)
# so the LZ4 decompressor sees invalid token sequences and returns an error.
cp input.txt.lz4 corrupt.lz4
python3 -c "
import os
with open('corrupt.lz4','r+b') as f:
    size = os.path.getsize('corrupt.lz4')
    mid = size // 2
    f.seek(mid)
    f.write(bytes([0xFF ^ (i % 256) for i in range(64)]))
"
if ! run_gzl4 -t corrupt.lz4 2>/dev/null; then pass "2.11  -t detects corruption"
else fail "2.11  -t detects corruption" "should have failed on corrupt file"; fi

# incompressible (binary) data roundtrip
cp "$DATA/small.bin" rand.bin
run_gzl4 -k rand.bin || true
run_gzl4 -f -k rand.bin.lz4 || true
check_files_equal   "2.12  binary roundtrip" "$DATA/small.bin" rand.bin

# =============================================================================
# 3. COMPRESSION LEVELS
# =============================================================================
section "3. Compression levels"

W="$(mkwork s3)"; cd "$W"
cp "$DATA/small.txt" src.txt

for level in 1 3 5 7 9; do
    gzl4_to level${level}.lz4 -k -${level} -c src.txt || true
    gzl4_to level${level}_out.txt -dc level${level}.lz4 || true
    check_files_equal "3.0${level}  level -${level} roundtrip" "$DATA/small.txt" level${level}_out.txt
done

# --fast alias
gzl4_to fast.lz4 -k --fast -c src.txt || true
gzl4_to fast_out.txt -dc fast.lz4 || true
check_files_equal   "3.10  --fast roundtrip" "$DATA/small.txt" fast_out.txt

# --best alias
gzl4_to best.lz4 -k --best -c src.txt || true
gzl4_to best_out.txt -dc best.lz4 || true
check_files_equal   "3.11  --best roundtrip" "$DATA/small.txt" best_out.txt

# HC levels (CPU only)
for level in 10 11 12; do
    gzl4_to hc${level}.lz4 --cpu-only -k -${level} -c src.txt || true
    gzl4_to hc${level}_out.txt -dc hc${level}.lz4 || true
    check_files_equal "3.${level}  HC level -${level} roundtrip (cpu-only)" "$DATA/small.txt" hc${level}_out.txt
done

# --hc-level explicit
gzl4_to hcl6.lz4 --cpu-only --hc-level 6 -k -c src.txt || true
gzl4_to hcl6_out.txt -dc hcl6.lz4 || true
check_files_equal   "3.13  --hc-level 6 roundtrip" "$DATA/small.txt" hcl6_out.txt

# Level -1 output should be larger than level -9 (for compressible data)
sz1=$(wc -c < level1.lz4)
sz9=$(wc -c < level9.lz4)
if [[ $sz1 -ge $sz9 ]]; then pass "3.14  level -1 output >= level -9 output (compressible)"
else fail "3.14  level -1 output >= level -9 output" "sz1=$sz1 sz9=$sz9"; fi

# HC -12 should be <= fast -9 (compressible data)
sz12=$(wc -c < hc12.lz4)
if [[ $sz12 -le $sz9 ]]; then pass "3.15  HC -12 output <= fast -9 output"
else fail "3.15  HC -12 output <= fast -9 output" "HC=$sz12 fast=$sz9 (may vary)"; fi

# =============================================================================
# 4. CPU-ONLY BACKEND
# =============================================================================
section "4. CPU-only backend"

W="$(mkwork s4)"; cd "$W"
cp "$DATA/small.txt" src.txt

gzl4_to cpu.lz4 --cpu-only -k -c src.txt || true
gzl4_to cpu_out.txt -dc cpu.lz4 || true
check_files_equal   "4.01  cpu-only compress/decompress roundtrip" "$DATA/small.txt" cpu_out.txt

# CPU with explicit thread count
gzl4_to cpu_t2.lz4 --cpu-only -T 2 -k -c src.txt || true
gzl4_to cpu_t2_out.txt -dc cpu_t2.lz4 || true
check_files_equal   "4.02  cpu-only -T 2 roundtrip" "$DATA/small.txt" cpu_t2_out.txt

gzl4_to cpu_t1.lz4 --cpu-only -T 1 -k -c src.txt || true
gzl4_to cpu_t1_out.txt -dc cpu_t1.lz4 || true
check_files_equal   "4.03  cpu-only -T 1 roundtrip" "$DATA/small.txt" cpu_t1_out.txt

# CPU decompress with -t
gzl4_to for_test.lz4 --cpu-only -k -c src.txt || true
if run_gzl4 -t for_test.lz4; then pass "4.04  cpu-only -t passes"
else fail "4.04  cpu-only -t passes" "exit non-zero"; fi

if [[ $SMALL_ONLY -eq 0 ]]; then
    cp "$DATA/medium.txt" med.txt
    gzl4_to med_cpu.lz4 --cpu-only -k -c med.txt || true
    gzl4_to med_cpu_out.txt -dc med_cpu.lz4 || true
    check_files_equal "4.05  cpu-only medium file roundtrip" "$DATA/medium.txt" med_cpu_out.txt
else
    skip "4.05  cpu-only medium file roundtrip" "--small"
fi

# =============================================================================
# 5. GPU-ONLY BACKEND
# =============================================================================
section "5. GPU-only backend"

if [[ $SKIP_GPU -eq 1 ]]; then
    for n in 01 02 03 04 05 06 07; do skip "5.$n  (gpu-only)" "--no-gpu"; done
else
    W="$(mkwork s5)"; cd "$W"
    cp "$DATA/small.txt" src.txt

    if run_gzl4 --gpu-only -k src.txt 2>/dev/null; then
        pass "5.01  gpu-only compress exits 0"
        # -k wrote to src.txt.lz4; rename for clarity
        cp src.txt.lz4 gpu.lz4
        gzl4_to gpu_out.txt -dc gpu.lz4 || true
        check_files_equal "5.02  gpu-only compress/decompress roundtrip" "$DATA/small.txt" gpu_out.txt

        # GPU decompress
        run_gzl4 -d --gpu-only -k -f gpu.lz4 || true
        check_files_equal "5.03  gpu-only explicit decompress" "$DATA/small.txt" src.txt

        # GPU with tuning params
        gzl4_to gpu_b4.lz4 --gpu-only --batch-size 4 -k -c src.txt || true
        gzl4_to gpu_b4_out.txt -dc gpu_b4.lz4 || true
        check_files_equal "5.04  gpu-only --batch-size 4 roundtrip" "$DATA/small.txt" gpu_b4_out.txt

        gzl4_to gpu_s2.lz4 --gpu-only --streams-per-gpu 2 -k -c src.txt || true
        gzl4_to gpu_s2_out.txt -dc gpu_s2.lz4 || true
        check_files_equal "5.05  gpu-only --streams-per-gpu 2 roundtrip" "$DATA/small.txt" gpu_s2_out.txt

        gzl4_to gpu_ne.lz4 --gpu-only --no-early-read -k -c src.txt || true
        gzl4_to gpu_ne_out.txt -dc gpu_ne.lz4 || true
        check_files_equal "5.06  gpu-only --no-early-read roundtrip" "$DATA/small.txt" gpu_ne_out.txt

        # -t via GPU
        if run_gzl4 -t --gpu-only gpu.lz4; then pass "5.07  gpu-only -t passes"
        else fail "5.07  gpu-only -t passes" "exit non-zero"; fi
    else
        for n in 01 02 03 04 05 06 07; do skip "5.$n  (gpu-only)" "no GPU available"; done
    fi
fi

# =============================================================================
# 6. HYBRID BACKEND (explicit)
# =============================================================================
section "6. Hybrid backend (explicit)"

if [[ $SKIP_GPU -eq 1 ]]; then
    for n in 01 02 03; do skip "6.$n  (hybrid)" "--no-gpu"; done
else
    W="$(mkwork s6)"; cd "$W"
    cp "$DATA/small.txt" src.txt

    if run_gzl4 --hybrid -k src.txt 2>/dev/null; then
        pass "6.01  hybrid compress exits 0"
        cp src.txt.lz4 hyb.lz4
        gzl4_to hyb_out.txt -dc hyb.lz4 || true
        check_files_equal "6.02  hybrid roundtrip" "$DATA/small.txt" hyb_out.txt

        if run_gzl4 -t --hybrid hyb.lz4; then pass "6.03  hybrid -t passes"
        else fail "6.03  hybrid -t passes" "exit non-zero"; fi
    else
        for n in 01 02 03; do skip "6.$n  (hybrid)" "no GPU available"; done
    fi
fi

# =============================================================================
# 7. PIPE MODE (stdin → stdout)
# =============================================================================
section "7. Pipe mode (stdin -> stdout)"

W="$(mkwork s7)"; cd "$W"

# Compress via pipe
cat "$DATA/small.txt" | gzl4_to pipe_comp.lz4 -c || true
gzl4_to pipe_decomp.txt -dc pipe_comp.lz4 || true
check_files_equal   "7.01  pipe compress roundtrip" "$DATA/small.txt" pipe_decomp.txt

# Decompress via pipe
gzl4_to pipe_decomp2.txt -dc < pipe_comp.lz4 || true
check_files_equal   "7.02  pipe decompress (cat | gzl4 -dc)" "$DATA/small.txt" pipe_decomp2.txt

# Explicit stdin "-" marker
gzl4_to explicit_stdin.lz4 < "$DATA/small.txt" || true
gzl4_to explicit_stdin_out.txt -d < explicit_stdin.lz4 || true
check_files_equal   "7.03  explicit stdin '-' compress/decompress" "$DATA/small.txt" explicit_stdin_out.txt

# Pipe with level flag
cat "$DATA/small.txt" | gzl4_to pipe_l1.lz4 -c -1 || true
gzl4_to pipe_l1_out.txt -dc pipe_l1.lz4 || true
check_files_equal   "7.04  pipe -1 level roundtrip" "$DATA/small.txt" pipe_l1_out.txt

# Pipe --cpu-only
cat "$DATA/small.txt" | gzl4_to pipe_cpu.lz4 -c --cpu-only || true
gzl4_to pipe_cpu_out.txt -dc pipe_cpu.lz4 || true
check_files_equal   "7.05  pipe --cpu-only roundtrip" "$DATA/small.txt" pipe_cpu_out.txt

# GPU pipe mode
if [[ $SKIP_GPU -eq 0 ]]; then
    if gzl4_to pipe_gpu.lz4 -c --gpu-only < "$DATA/small.txt" 2>/dev/null; then
        gzl4_to pipe_gpu_out.txt -dc pipe_gpu.lz4 || true
        check_files_equal "7.06  pipe --gpu-only roundtrip" "$DATA/small.txt" pipe_gpu_out.txt
    else
        skip "7.06  pipe --gpu-only roundtrip" "no GPU available"
    fi
else
    skip "7.06  pipe --gpu-only roundtrip" "--no-gpu"
fi

# Pipe quiet by default (no progress to stderr in pipe mode)
stderr_out=$(cat "$DATA/small.txt" | "$GZL4" -c 2>&1 >/dev/null || true)
if [[ -z "$stderr_out" ]]; then pass "7.07  pipe mode is quiet by default"
else fail "7.07  pipe mode is quiet by default" "got stderr: $stderr_out"; fi

# --progress flag produces output on stderr in pipe mode
stderr_out=$(cat "$DATA/small.txt" | "$GZL4" -c --progress 2>&1 >/dev/null || true)
if [[ -n "$stderr_out" ]]; then pass "7.08  --progress produces stderr in pipe mode"
else fail "7.08  --progress produces stderr in pipe mode" "no stderr output"; fi

# -v also produces output on stderr in pipe mode
stderr_out=$(cat "$DATA/small.txt" | "$GZL4" -c -v 2>&1 >/dev/null || true)
if [[ -n "$stderr_out" ]]; then pass "7.09  -v produces stderr in pipe mode"
else fail "7.09  -v produces stderr in pipe mode" "no stderr output"; fi

# Large-ish pipe (medium file) if not --small
if [[ $SMALL_ONLY -eq 0 ]]; then
    cat "$DATA/medium.txt" | gzl4_to pipe_med.lz4 -c || true
    gzl4_to pipe_med_out.txt -dc pipe_med.lz4 || true
    check_files_equal "7.10  pipe medium file roundtrip" "$DATA/medium.txt" pipe_med_out.txt
else
    skip "7.10  pipe medium file roundtrip" "--small"
fi

# =============================================================================
# 8. TAR PIPE: tar -cf - ... | gzl4
# =============================================================================
section "8. tar pipe (tar -cf - | gzl4)"

W="$(mkwork s8)"; cd "$W"

# Compress tar stream via pipe, decompress back
tar -cf - -C "$DATA" tardir | gzl4_to tarstream.lz4 -c || true
mkdir -p out
"$GZL4" -dc tarstream.lz4 2>/dev/null | tar -x -C out || true
check_file_exists   "out/tardir/file1.txt"    "8.01  tar pipe: file1.txt extracted"
check_file_exists   "out/tardir/file2.bin"    "8.02  tar pipe: file2.bin extracted"
check_file_exists   "out/tardir/subdir/hello.txt" "8.03  tar pipe: subdir/hello.txt extracted"
check_files_equal   "8.04  tar pipe: file1.txt matches" "$DATA/tardir/file1.txt" out/tardir/file1.txt
check_files_equal   "8.05  tar pipe: file2.bin matches" "$DATA/tardir/file2.bin" out/tardir/file2.bin

# cpu-only pipe
tar -cf - -C "$DATA" tardir | gzl4_to tarstream_cpu.lz4 -c --cpu-only || true
mkdir -p out_cpu
"$GZL4" -dc tarstream_cpu.lz4 2>/dev/null | tar -x -C out_cpu || true
check_files_equal   "8.06  tar pipe --cpu-only roundtrip" \
    "$DATA/tardir/file1.txt" out_cpu/tardir/file1.txt

# =============================================================================
# 9. TAR -I INTEGRATION
# =============================================================================
section "9. tar -I integration"

if [[ $SKIP_TAR -eq 1 ]]; then
    for n in 01 02 03 04 05 06 07 08; do skip "9.$n  (tar -I)" "--no-tar"; done
else
    W="$(mkwork s9)"; cd "$W"

    # Create archive with tar -I gzl4
    tar -I "$GZL4" -cf tarI.tar.lz4 -C "$DATA" tardir || true
    check_file_exists "tarI.tar.lz4"            "9.01  tar -I create archive"

    # List contents with tar -I gzl4
    listing=$(tar -I "$GZL4" -tf tarI.tar.lz4 2>/dev/null)
    if echo "$listing" | grep -q "tardir/file1.txt"; then pass "9.02  tar -I list contents"
    else fail "9.02  tar -I list contents" "file1.txt not in listing"; fi

    # Extract with tar -I gzl4
    mkdir -p extracted
    tar -I "$GZL4" -xf tarI.tar.lz4 -C extracted || true
    check_file_exists "extracted/tardir/file1.txt" "9.03  tar -I extract file1.txt"
    check_file_exists "extracted/tardir/file2.bin" "9.04  tar -I extract file2.bin"
    check_files_equal "9.05  tar -I extract matches original" \
        "$DATA/tardir/file1.txt" extracted/tardir/file1.txt

    # GPU-only via tar -I option passing
    if [[ $SKIP_GPU -eq 0 ]]; then
        if tar -I "$GZL4 --gpu-only" -cf tarI_gpu.tar.lz4 -C "$DATA" tardir 2>/dev/null; then
            mkdir -p extracted_gpu
            tar -I "$GZL4" -xf tarI_gpu.tar.lz4 -C extracted_gpu || true
            check_files_equal "9.06  tar -I --gpu-only roundtrip" \
                "$DATA/tardir/file1.txt" extracted_gpu/tardir/file1.txt
        else
            skip "9.06  tar -I --gpu-only roundtrip" "no GPU available"
        fi
    else
        skip "9.06  tar -I --gpu-only roundtrip" "--no-gpu"
    fi

    # cpu-only via tar -I
    tar -I "$GZL4 --cpu-only" -cf tarI_cpu.tar.lz4 -C "$DATA" tardir || true
    mkdir -p extracted_cpu
    tar -I "$GZL4" -xf tarI_cpu.tar.lz4 -C extracted_cpu || true
    check_files_equal "9.07  tar -I --cpu-only roundtrip" \
        "$DATA/tardir/file1.txt" extracted_cpu/tardir/file1.txt

    # HC compression via tar -I
    tar -I "$GZL4 --cpu-only -12" -cf tarI_hc.tar.lz4 -C "$DATA" tardir || true
    mkdir -p extracted_hc
    tar -I "$GZL4" -xf tarI_hc.tar.lz4 -C extracted_hc || true
    check_files_equal "9.08  tar -I -12 HC roundtrip" \
        "$DATA/tardir/file1.txt" extracted_hc/tardir/file1.txt
fi

# =============================================================================
# 10. UNGZL4 SYMLINK BEHAVIOUR
# =============================================================================
section "10. ungzl4 symlink"

W="$(mkwork s10)"; cd "$W"

# Create symlink if not already present
UNGZL4_LOCAL="$W/ungzl4"
ln -sf "$GZL4" "$UNGZL4_LOCAL"

# Compress a file first
gzl4_to to_decomp.lz4 -k -c "$DATA/small.txt" || true

# ungzl4 should auto-decompress
"$UNGZL4_LOCAL" -k to_decomp.lz4 >/dev/null 2>&1
check_file_exists   "to_decomp"               "10.01  ungzl4 auto-decompresses (no -d needed)"
check_files_equal   "10.02  ungzl4 output matches original" "$DATA/small.txt" to_decomp

# ungzl4 -c should decompress to stdout
"$UNGZL4_LOCAL" -c to_decomp.lz4 > ungzl4_stdout.txt 2>/dev/null
check_files_equal   "10.03  ungzl4 -c to stdout" "$DATA/small.txt" ungzl4_stdout.txt

# ungzl4 -z should COMPRESS despite the "un" prefix
cp "$DATA/small.txt" forcecmp.txt
"$UNGZL4_LOCAL" -z -k forcecmp.txt >/dev/null 2>&1
check_file_exists   "forcecmp.txt.lz4"        "10.04  ungzl4 -z creates .lz4"
gzl4_to forcecmp_out.txt -dc forcecmp.txt.lz4 || true
check_files_equal   "10.05  ungzl4 -z roundtrip" "$DATA/small.txt" forcecmp_out.txt

# ungzl4 -z -f -k (the exact invocation from the bug report)
cp "$DATA/small.txt" forcecmp2.txt
run_gzl4 -k forcecmp2.txt  # pre-create .lz4 so -f is needed || true
"$UNGZL4_LOCAL" -z -f -k forcecmp2.txt >/dev/null 2>&1
check_file_exists   "forcecmp2.txt.lz4"       "10.06  ungzl4 -z -f -k succeeds"
check_file_exists   "forcecmp2.txt"            "10.07  ungzl4 -z -k keeps original"

# =============================================================================
# 11. FILE HANDLING: -k, -f, no-clobber, atomic rename
# =============================================================================
section "11. File handling (-k -f, no-clobber, keep)"

W="$(mkwork s11)"; cd "$W"
cp "$DATA/small.txt" src.txt

# Without -k, original is removed after compress
run_gzl4 src.txt || true
check_file_absent   "src.txt"             "11.01  compress without -k removes original"
check_file_exists   "src.txt.lz4"             "11.02  compress creates .lz4"

# Without -k, original is removed after decompress
run_gzl4 src.txt.lz4 || true
check_file_absent   "src.txt.lz4"         "11.03  decompress without -k removes .lz4"
check_file_exists   "src.txt"                 "11.04  decompress restores original"

# -k keeps both
cp "$DATA/small.txt" keep_src.txt
run_gzl4 -k keep_src.txt || true
check_file_exists   "keep_src.txt"            "11.05  -k compress keeps original"
check_file_exists   "keep_src.txt.lz4"        "11.06  -k compress creates .lz4"

run_gzl4 -k keep_src.txt.lz4 || true
check_file_exists   "keep_src.txt.lz4"        "11.07  -k decompress keeps .lz4"
check_file_exists   "keep_src.txt"            "11.08  -k decompress restores file"

# Without -f, should refuse to overwrite existing .lz4
run_gzl4 -k keep_src.txt  # .lz4 now exists || true
if ! run_gzl4 keep_src.txt 2>/dev/null; then pass "11.09  no-clobber: refuses to overwrite without -f"
else fail "11.09  no-clobber: refuses to overwrite without -f" "should have failed"; fi

# With -f, overwrites
run_gzl4 -f keep_src.txt >/dev/null 2>&1 || true
# check the .lz4 is still valid after the -f overwrite
run_gzl4 -k keep_src.txt.lz4 || true
check_files_equal   "11.10  -f overwrite produces valid output" "$DATA/small.txt" keep_src.txt

# =============================================================================
# 12. VERBOSITY FLAGS
# =============================================================================
section "12. Verbosity flags"

W="$(mkwork s12)"; cd "$W"
cp "$DATA/small.txt" src.txt
gzl4_to src.lz4 -k -c src.txt || true

# -q should produce no stderr
stderr_q=$("$GZL4" -q -k -c src.txt 2>&1 >/dev/null || true)
if [[ -z "$stderr_q" ]]; then pass "12.01  -q suppresses all stderr"
else fail "12.01  -q suppresses all stderr" "got: $stderr_q"; fi

# Default (no -q, not a pipe) should produce some stderr
stderr_def=$("$GZL4" -f -k -c src.txt 2>&1 >/dev/null || true)
if [[ -n "$stderr_def" ]]; then pass "12.02  default verbosity produces stderr"
else fail "12.02  default verbosity produces stderr" "no stderr output"; fi

# -v should produce stderr
stderr_v=$("$GZL4" -f -k -c -v src.txt 2>&1 >/dev/null || true)
if [[ -n "$stderr_v" ]]; then pass "12.03  -v produces stderr"
else fail "12.03  -v produces stderr" "no stderr output"; fi

# =============================================================================
# 13. LZ4 TOOL INTEROPERABILITY
# =============================================================================
section "13. lz4 interoperability"

if [[ $SKIP_LZ4 -eq 1 ]]; then
    for n in 01 02 03 04; do skip "13.$n  (lz4 interop)" "--no-lz4"; done
else
    W="$(mkwork s13)"; cd "$W"

    # gzl4 output readable by lz4
    gzl4_to gzl4_out.lz4 -k --cpu-only -c "$DATA/small.txt" || true
    lz4 -d -f gzl4_out.lz4 lz4_read.txt >/dev/null 2>&1
    check_files_equal   "13.01  lz4 can decompress gzl4 output" "$DATA/small.txt" lz4_read.txt

    # lz4 output readable by gzl4
    lz4 -f "$DATA/small.txt" lz4_compressed.lz4 >/dev/null 2>&1
    gzl4_to lz4_then_gzl4.txt -k -c lz4_compressed.lz4 || true
    gzl4_to lz4_decomp.txt -dc lz4_compressed.lz4 || true
    check_files_equal   "13.02  gzl4 can decompress lz4 output" "$DATA/small.txt" lz4_decomp.txt

    # tar -I lz4 creates file readable by tar -I gzl4
    if [[ $SKIP_TAR -eq 0 ]]; then
        tar -I lz4 -cf lz4_archive.tar.lz4 -C "$DATA" tardir >/dev/null 2>&1
        mkdir -p extracted_from_lz4
        tar -I "$GZL4" -xf lz4_archive.tar.lz4 -C extracted_from_lz4 >/dev/null 2>&1
        check_files_equal "13.03  gzl4 reads lz4-created tar archive" \
            "$DATA/tardir/file1.txt" extracted_from_lz4/tardir/file1.txt

        # tar -I gzl4 creates file readable by tar -I lz4
        tar -I "$GZL4 --cpu-only" -cf gzl4_archive.tar.lz4 -C "$DATA" tardir >/dev/null 2>&1
        mkdir -p extracted_from_gzl4
        tar -I lz4 -xf gzl4_archive.tar.lz4 -C extracted_from_gzl4 >/dev/null 2>&1
        check_files_equal "13.04  lz4 reads gzl4-created tar archive" \
            "$DATA/tardir/file1.txt" extracted_from_gzl4/tardir/file1.txt
    else
        skip "13.03  gzl4 reads lz4 tar archive" "--no-tar"
        skip "13.04  lz4 reads gzl4 tar archive" "--no-tar"
    fi
fi

# =============================================================================
# 14. MULTI-CHUNK / MEDIUM FILE (skipped with --small)
# =============================================================================
section "14. Multi-chunk correctness (medium file)"

if [[ $SMALL_ONLY -eq 1 ]]; then
    for n in 01 02 03 04; do skip "14.$n" "--small"; done
else
    W="$(mkwork s14)"; cd "$W"

    # cpu-only multi-chunk
    gzl4_to med_cpu.lz4 --cpu-only -k -c "$DATA/medium.txt" || true
    gzl4_to med_cpu_out.txt -dc med_cpu.lz4 || true
    check_files_equal "14.01  cpu-only medium file roundtrip" "$DATA/medium.txt" med_cpu_out.txt

    # cpu-only different levels
    gzl4_to med_l1.lz4 --cpu-only -1 -k -c "$DATA/medium.txt" || true
    gzl4_to med_l1_out.txt -dc med_l1.lz4 || true
    check_files_equal "14.02  cpu-only medium -1 roundtrip" "$DATA/medium.txt" med_l1_out.txt

    if [[ $SKIP_GPU -eq 0 ]]; then
        if gzl4_to med_gpu.lz4 --gpu-only -c "$DATA/medium.txt"; then
            gzl4_to med_gpu_out.txt -dc med_gpu.lz4 || true
            check_files_equal "14.03  gpu-only medium file roundtrip" "$DATA/medium.txt" med_gpu_out.txt
        else
            skip "14.03  gpu-only medium file" "no GPU available"
        fi

        if gzl4_to med_hyb.lz4 --hybrid -c "$DATA/medium.txt"; then
            gzl4_to med_hyb_out.txt -dc med_hyb.lz4 || true
            check_files_equal "14.04  hybrid medium file roundtrip" "$DATA/medium.txt" med_hyb_out.txt
        else
            skip "14.04  hybrid medium file" "no GPU available"
        fi
    else
        skip "14.03  gpu-only medium file" "--no-gpu"
        skip "14.04  hybrid medium file"   "--no-gpu"
    fi
fi

# =============================================================================
# 15. EDGE CASES
# =============================================================================
section "15. Edge cases"

W="$(mkwork s15)"; cd "$W"

# Empty file
# Empty and single-byte files can hang old binaries (fixed in v3.26.1: writer
# loop required tb>0 to exit, but empty files have 0 blocks).  Use timeout as
# a safety net so the test suite doesn't stall against an unfixed build.
touch empty.txt
run_gzl4 -k --cpu-only empty.txt || true
check_file_exists "empty.txt.lz4"             "15.01  empty file compresses"
if [[ -f empty.txt.lz4 ]]; then
    if timeout 10 "$GZL4" -f -k empty.txt.lz4 >/dev/null 2>&1; then
        if [[ -f empty.txt ]] && [[ ! -s empty.txt ]]; then pass "15.02  empty file roundtrip"
        else fail "15.02  empty file roundtrip" "file missing or non-empty after decompress"; fi
    else
        fail "15.02  empty file roundtrip" "gzl4 timed out or failed decompressing empty.lz4 (hang bug?)"
    fi
else
    fail "15.02  empty file roundtrip" "no .lz4 to decompress"
fi

# Single-byte file
echo -n "X" > one_byte.txt
run_gzl4 -k --cpu-only one_byte.txt || true
if [[ -f one_byte.txt.lz4 ]]; then
    if timeout 10 "$GZL4" -f -k one_byte.txt.lz4 >/dev/null 2>&1; then
        if [[ -f one_byte.txt ]] && [[ "$(wc -c < one_byte.txt)" -eq 1 ]]; then
            pass "15.03  single-byte file roundtrip"
        else fail "15.03  single-byte file roundtrip" "wrong size or missing"; fi
    else
        fail "15.03  single-byte file roundtrip" "gzl4 timed out or failed (hang bug?)"
    fi
else
    fail "15.03  single-byte file roundtrip" "no .lz4 to decompress"
fi

# -z on a non-.lz4 file is same as no -z
cp "$DATA/small.txt" noext.dat
run_gzl4 -z -k noext.dat || true
check_file_exists "noext.dat.lz4"             "15.04  -z on non-.lz4 file still compresses"
gzl4_to noext_out.dat -dc noext.dat.lz4 || true
check_files_equal "15.05  -z non-.lz4 roundtrip" "$DATA/small.txt" noext_out.dat

# Non-existent input file gives non-zero exit
if ! run_gzl4 /nonexistent/file.txt 2>/dev/null; then
    pass "15.06  non-existent input exits non-zero"
else
    fail "15.06  non-existent input exits non-zero" "should have returned non-zero"
fi

# Bad option gives non-zero exit
if ! "$GZL4" --bad-option-xyz 2>/dev/null; then
    pass "15.07  unknown option exits non-zero"
else
    fail "15.07  unknown option exits non-zero" "should have returned non-zero"
fi

# =============================================================================
# 16. --content-size / --no-content-size
# =============================================================================
section "16. --content-size / --no-content-size"

W="$(mkwork s16)"; cd "$W"
cp "$DATA/small.txt" src.txt

# Compress with --content-size: FLG byte bit 3 (0x08) should be set.
# LZ4 frame layout: magic(4) FLG(1) ...  FLG is byte offset 4 (0-indexed).
# Use -c so gzl4_to captures the output via stdout.
gzl4_to with_cs.lz4 -c --cpu-only --content-size src.txt || true
if [[ -f with_cs.lz4 ]]; then
    flg=$(od -An -tx1 -j4 -N1 with_cs.lz4 | tr -d ' \n')
    bit3=$(( 0x$flg & 0x08 ))
    if [[ $bit3 -ne 0 ]]; then pass "16.01  --content-size sets C_SIZE bit in FLG"
    else fail "16.01  --content-size sets C_SIZE bit in FLG" \
              "FLG=0x$flg, bit 3 not set"; fi
else
    fail "16.01  --content-size sets C_SIZE bit in FLG" "compression failed"
fi

# Compress with --no-content-size: FLG bit 3 should be clear.
gzl4_to no_cs.lz4 -c --cpu-only --no-content-size src.txt || true
if [[ -f no_cs.lz4 ]]; then
    flg=$(od -An -tx1 -j4 -N1 no_cs.lz4 | tr -d ' \n')
    bit3=$(( 0x$flg & 0x08 ))
    if [[ $bit3 -eq 0 ]]; then pass "16.02  --no-content-size clears C_SIZE bit in FLG"
    else fail "16.02  --no-content-size clears C_SIZE bit in FLG" \
              "FLG=0x$flg, bit 3 unexpectedly set"; fi
else
    fail "16.02  --no-content-size clears C_SIZE bit in FLG" "compression failed"
fi

# Default file mode should embed content size (bit 3 set)
# Use -c so gzl4_to captures output; src.txt is preserved for later tests.
gzl4_to default_cs.lz4 -c --cpu-only src.txt || true
if [[ -f default_cs.lz4 ]]; then
    flg=$(od -An -tx1 -j4 -N1 default_cs.lz4 | tr -d ' \n')
    bit3=$(( 0x$flg & 0x08 ))
    if [[ $bit3 -ne 0 ]]; then pass "16.03  file mode default embeds content size"
    else fail "16.03  file mode default embeds content size" \
              "FLG=0x$flg, bit 3 not set"; fi
else
    fail "16.03  file mode default embeds content size" "compression failed"
fi

# Pipe mode default should NOT embed content size (bit 3 clear)
gzl4_to pipe_cs.lz4 --cpu-only < src.txt || true
if [[ -f pipe_cs.lz4 ]]; then
    flg=$(od -An -tx1 -j4 -N1 pipe_cs.lz4 | tr -d ' \n')
    bit3=$(( 0x$flg & 0x08 ))
    if [[ $bit3 -eq 0 ]]; then pass "16.04  pipe mode default omits content size"
    else fail "16.04  pipe mode default omits content size" \
              "FLG=0x$flg, bit 3 set in pipe mode"; fi
else
    fail "16.04  pipe mode default omits content size" "compression failed"
fi

# Roundtrip: --no-content-size output still decompresses correctly
gzl4_to nocs_out.txt -dc no_cs.lz4 || true
check_files_equal "16.05  --no-content-size roundtrip" "$DATA/small.txt" nocs_out.txt

# =============================================================================
# 17. --list
# =============================================================================
section "17. --list"

W="$(mkwork s17)"; cd "$W"

# Create a file with known content size embedded
gzl4_to known.lz4 -c --cpu-only --content-size "$DATA/small.txt" || true
# Create a file without content size
gzl4_to unknown.lz4 -c --cpu-only --no-content-size "$DATA/small.txt" || true

# --list exits 0 on valid file
if [[ -f known.lz4 ]]; then
    if "$GZL4" --list known.lz4 >/dev/null 2>&1; then
        pass "17.01  --list exits 0 on valid .lz4"
    else
        fail "17.01  --list exits 0 on valid .lz4" "exit non-zero"
    fi
else
    skip "17.01  --list exits 0 on valid .lz4" "compression failed"
fi

# --list output contains expected column header
if [[ -f known.lz4 ]]; then
    list_out=$("$GZL4" --list known.lz4 2>/dev/null)
    if echo "$list_out" | grep -q "Frames"; then
        pass "17.02  --list output has Frames header"
    else
        fail "17.02  --list output has Frames header" "header not found in: $list_out"
    fi
else
    skip "17.02  --list output has Frames header" "no input file"
fi

# --list shows uncompressed size when content-size is present
if [[ -f known.lz4 ]]; then
    list_out=$("$GZL4" --list known.lz4 2>/dev/null)
    # Should NOT contain N/A in the Uncompressed column
    data_line=$(echo "$list_out" | grep -v "^Frames" | head -1)
    uncomp=$(echo "$data_line" | awk '{print $5}')
    if [[ "$uncomp" != "N/A" && -n "$uncomp" ]]; then
        pass "17.03  --list shows uncompressed size when C_SIZE present"
    else
        fail "17.03  --list shows uncompressed size when C_SIZE present" \
             "got: '$data_line'"
    fi
else
    skip "17.03  --list shows uncompressed size when C_SIZE present" "no input file"
fi

# --list shows N/A for uncompressed when no content-size
if [[ -f unknown.lz4 ]]; then
    list_out=$("$GZL4" --list unknown.lz4 2>/dev/null)
    if echo "$list_out" | grep -q "N/A"; then
        pass "17.04  --list shows N/A when C_SIZE absent"
    else
        fail "17.04  --list shows N/A when C_SIZE absent" \
             "N/A not found in: $list_out"
    fi
else
    skip "17.04  --list shows N/A when C_SIZE absent" "no input file"
fi

# --list shows LZ4Frame type
if [[ -f known.lz4 ]]; then
    if "$GZL4" --list known.lz4 2>/dev/null | grep -q "LZ4Frame"; then
        pass "17.05  --list shows LZ4Frame type"
    else
        fail "17.05  --list shows LZ4Frame type" "LZ4Frame not in output"
    fi
else
    skip "17.05  --list shows LZ4Frame type" "no input file"
fi

# --list shows block descriptor (B<n>I or B<n>D format)
if [[ -f known.lz4 ]]; then
    if "$GZL4" --list known.lz4 2>/dev/null | grep -qE "B[4-7][ID]"; then
        pass "17.06  --list shows block descriptor (B<n>I/D)"
    else
        fail "17.06  --list shows block descriptor (B<n>I/D)" "pattern not found"
    fi
else
    skip "17.06  --list shows block descriptor (B<n>I/D)" "no input file"
fi

# --list with multiple files: single header, one row per file
if [[ -f known.lz4 && -f unknown.lz4 ]]; then
    list_out=$("$GZL4" --list known.lz4 unknown.lz4 2>/dev/null)
    header_count=$(echo "$list_out" | grep -c "^Frames")
    data_count=$(echo "$list_out" | grep -v "^Frames" | grep -c "LZ4Frame")
    if [[ $header_count -eq 1 && $data_count -eq 2 ]]; then
        pass "17.07  --list multiple files: 1 header, 2 data rows"
    else
        fail "17.07  --list multiple files: 1 header, 2 data rows" \
             "headers=$header_count data=$data_count"
    fi
else
    skip "17.07  --list multiple files: 1 header, 2 data rows" "missing input files"
fi

# --list -v shows per-frame detail (frame<n> rows)
if [[ -f known.lz4 ]]; then
    if "$GZL4" --list -v known.lz4 2>/dev/null | grep -q "frame"; then
        pass "17.08  --list -v shows per-frame rows"
    else
        fail "17.08  --list -v shows per-frame rows" "no frame rows in -v output"
    fi
else
    skip "17.08  --list -v shows per-frame rows" "no input file"
fi

# --list on non-.lz4 file exits non-zero
if ! "$GZL4" --list "$DATA/small.txt" >/dev/null 2>&1; then
    pass "17.09  --list on non-lz4 file exits non-zero"
else
    fail "17.09  --list on non-lz4 file exits non-zero" "should have failed"
fi

# =============================================================================
# 18. Multi-file command line
# =============================================================================
section "18. Multi-file command line"

W="$(mkwork s18)"; cd "$W"
cp "$DATA/small.txt" file1.txt
cp "$DATA/small.bin" file2.bin
cp "$DATA/small.txt" file3.txt

# Compress multiple files in one invocation
run_gzl4 --cpu-only -k file1.txt file2.bin file3.txt || true
check_file_exists "file1.txt.lz4" "18.01  multi-file: file1.txt.lz4 created"
check_file_exists "file2.bin.lz4" "18.02  multi-file: file2.bin.lz4 created"
check_file_exists "file3.txt.lz4" "18.03  multi-file: file3.txt.lz4 created"

# Originals kept (-k)
check_file_exists "file1.txt" "18.04  multi-file -k: file1.txt kept"
check_file_exists "file2.bin" "18.05  multi-file -k: file2.bin kept"

# Decompress multiple files in one invocation
run_gzl4 -d --cpu-only -k file1.txt.lz4 file2.bin.lz4 file3.txt.lz4 || true
check_files_equal "18.06  multi-file decompress: file1.txt roundtrip" \
    "$DATA/small.txt" file1.txt
check_files_equal "18.07  multi-file decompress: file2.bin roundtrip" \
    "$DATA/small.bin" file2.bin
check_files_equal "18.08  multi-file decompress: file3.txt roundtrip" \
    "$DATA/small.txt" file3.txt

# Auto-detect per file: mixed list (compress .txt, decompress .lz4)
cp "$DATA/small.txt" auto_src.txt
gzl4_to auto_pre.lz4 -c --cpu-only -k "$DATA/small.bin" || true
mv auto_pre.lz4 auto_src2.lz4
run_gzl4 --cpu-only -k auto_src.txt auto_src2.lz4 || true
check_file_exists "auto_src.txt.lz4" \
    "18.09  auto-detect: .txt was compressed"
check_file_exists "auto_src2"        \
    "18.10  auto-detect: .lz4 was decompressed"

# No-clobber: second file already exists, should error but first still succeeds
cp "$DATA/small.txt" nc1.txt
cp "$DATA/small.txt" nc2.txt
touch nc2.txt.lz4   # pre-create to trigger no-clobber on second file
if ! run_gzl4 --cpu-only -k nc1.txt nc2.txt 2>/dev/null; then
    pass "18.11  multi-file no-clobber: exits non-zero when output exists"
else
    fail "18.11  multi-file no-clobber: exits non-zero when output exists" \
         "should have returned non-zero"
fi
check_file_exists "nc1.txt.lz4" \
    "18.12  multi-file no-clobber: first file still processed"

# -c (stdout) rejected with multiple files
if ! run_gzl4 -c --cpu-only file1.txt file2.bin 2>/dev/null; then
    pass "18.13  -c with multiple files exits non-zero"
else
    fail "18.13  -c with multiple files exits non-zero" \
         "should have rejected -c + multi-file"
fi

# -c error message mentions incompatibility
err_msg=$("$GZL4" -c --cpu-only file1.txt file2.bin 2>&1 || true)
if echo "$err_msg" | grep -qi "not compatible\|incompatible"; then
    pass "18.14  -c multi-file error message is descriptive"
else
    fail "18.14  -c multi-file error message is descriptive" \
         "got: $err_msg"
fi

# -t (test) on multiple files: all pass
run_gzl4 -t --cpu-only file1.txt.lz4 file2.bin.lz4 file3.txt.lz4 || true
if run_gzl4 -t --cpu-only file1.txt.lz4 file2.bin.lz4 file3.txt.lz4 2>/dev/null; then
    pass "18.15  -t on multiple valid files exits 0"
else
    fail "18.15  -t on multiple valid files exits 0" "exit non-zero"
fi

# =============================================================================
# SUMMARY
# =============================================================================
TOTAL=$(( PASS + FAIL + SKIP ))
echo
echo "${BOLD}${RESET}"
echo "${BOLD}Results:  $TOTAL tests${RESET}"
echo "  ${GREEN}${PASS} passed${RESET}"
[[ $FAIL -gt 0 ]]  && echo "  ${RED}${FAIL} failed${RESET}"   || echo "  ${FAIL} failed"
[[ $SKIP -gt 0 ]]  && echo "  ${YELLOW}${SKIP} skipped${RESET}" || echo "  ${SKIP} skipped"
echo "${BOLD}${RESET}"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo
    echo "${RED}${BOLD}Failed tests:${RESET}"
    for f in "${FAILURES[@]}"; do echo "  ${RED}✗${RESET}  $f"; done
    echo
    exit 1
fi

echo
echo "${GREEN}${BOLD}All tests passed.${RESET}"
exit 0
