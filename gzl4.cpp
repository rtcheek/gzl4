/*
 * gzl4 - GPU-Accelerated LZ4 Compression/Decompression Tool
 * 
 * A gzip-like utility that leverages NVIDIA GPUs and nvCOMP 5.1.x
 * for high-performance LZ4 compression using batched operations.
 * 
 * Features:
 * - Multi-GPU support with automatic detection
 * - TRUE batched compression for maximum throughput
 * - Multiple CUDA streams for parallel processing
 * - Memory-aware chunk sizing
 * - gzip-compatible command-line interface
 * - Comprehensive error handling and verbosity levels
 */

#include <cuda_runtime.h>
#include <nvcomp/lz4.h>
#include <nvcomp.hpp>
#include <lz4.h>
#include <lz4hc.h>
// NVML for GPU utilization queries  used to route work away from busy GPUs
// (e.g. GPUs already running LLM training).  Loaded via dlopen so the binary
// still runs on systems without libnvidia-ml.so.
#include <dlfcn.h>
#include <nvml.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <inttypes.h>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <iomanip>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <map>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>   // sysinfo() for available RAM query
#include <sys/uio.h>       // writev(), struct iovec, IOV_MAX
#include <fcntl.h>
#include <getopt.h>
#include <signal.h>

// SIMD intrinsics for isAllZeros()  included at file scope, guarded by
// the same feature macros used in the function body below.
#if defined(__AVX2__)
#  include <immintrin.h>
#elif defined(__SSE2__)
#  include <emmintrin.h>
#endif

// Configuration constants
constexpr const char* VERSION = "3.32.3";

// Compression backend modes
enum class BackendMode {
    GPU_ONLY,   // GPU-only compression (original implementation)
    CPU_ONLY,   // Multi-threaded CPU compression  
    HYBRID      // CPU + GPU simultaneously (default, best performance)
};

constexpr size_t MIN_CHUNK_SIZE = 16 * 1024;            // 16KB minimum
constexpr size_t MAX_CHUNK_SIZE = 4 * 1024 * 1024;      // 4MB maximum (LZ4 frame limit)
// Pipeline depth per GPU is computed from cudaDeviceProp.asyncEngineCount at runtime.
constexpr double GPU_MEM_SAFETY_FACTOR = 0.5;           // Use 50% of available memory for batches
constexpr size_t CPU_THREADS_AUTO = 0;                   // Auto-detect CPU thread count

// LZ4 Frame Format constants
constexpr uint32_t LZ4_MAGIC = 0x184D2204;               // LZ4 frame magic number
constexpr uint8_t LZ4_VERSION = 0x40;                    // Version 01 in bits 7-6
constexpr uint32_t XXH32_SEED = 0;                       // Default seed for XXH32

// Chunk sizes by compression level (optimized for GPU performance)
// Levels 1-4 now start at 256KB minimum to avoid CPU overhead bottleneck
constexpr size_t CHUNK_SIZE_LEVEL_1 = 256 * 1024;       // 256KB (was 16KB)
constexpr size_t CHUNK_SIZE_LEVEL_2 = 512 * 1024;       // 512KB (was 32KB)
constexpr size_t CHUNK_SIZE_LEVEL_3 = 1 * 1024 * 1024;  // 1MB (was 64KB)
constexpr size_t CHUNK_SIZE_LEVEL_4 = 2 * 1024 * 1024;  // 2MB (was 128KB)
constexpr size_t CHUNK_SIZE_LEVEL_5 = 2 * 1024 * 1024;  // 2MB (was 256KB)
constexpr size_t CHUNK_SIZE_LEVEL_6 = 3 * 1024 * 1024;  // 3MB (was 512KB)
constexpr size_t CHUNK_SIZE_LEVEL_7 = 3 * 1024 * 1024 + 512 * 1024;  // 3.5MB (was 1MB)
constexpr size_t CHUNK_SIZE_LEVEL_8 = 4 * 1024 * 1024;  // 4MB (was 2MB)
constexpr size_t CHUNK_SIZE_LEVEL_9 = 4 * 1024 * 1024;  // 4MB (max for LZ4, now default)

// Verbosity levels - unified system
enum VerbosityLevel {
    QUIET = 0,        // -q: errors only
    NORMAL = 1,       // default: progress, completion messages
    VERBOSE = 2,      // -v: extra details
    VERY_VERBOSE = 3, // -vv: more details
    DEBUG = 4         // -vvv: debug info
};

// Global verbosity setting (default: NORMAL)
int g_verbosity = NORMAL;

// ── ANSI color support ────────────────────────────────────────────────────────
// Detected once at startup: true if stderr is a TTY and $TERM/COLORTERM
// indicate a color-capable terminal.
bool g_color = false;

inline void detectColor() {
    if (!isatty(STDERR_FILENO)) { g_color = false; return; }
    const char* term      = getenv("TERM");
    const char* colorterm = getenv("COLORTERM");
    const char* termProg  = getenv("TERM_PROGRAM");
    // COLORTERM=truecolor/24bit/yes → definitely supports ANSI colors
    if (colorterm && (strstr(colorterm, "truecolor") ||
                      strstr(colorterm, "24bit") ||
                      strcmp(colorterm, "yes") == 0)) {
        g_color = true; return;
    }
    // Known color-capable TERM values
    if (term && (strstr(term, "color") || strstr(term, "256") ||
                 strstr(term, "xterm") || strstr(term, "screen") ||
                 strstr(term, "tmux")  || strstr(term, "rxvt") ||
                 strcmp(term, "linux") == 0)) {
        g_color = true; return;
    }
    // Known color-capable TERM_PROGRAM values (macOS Terminal, iTerm2, etc.)
    if (termProg && (strstr(termProg, "iTerm") ||
                     strstr(termProg, "Terminal") ||
                     strstr(termProg, "Hyper") ||
                     strstr(termProg, "vscode"))) {
        g_color = true; return;
    }
    g_color = false;
}

// ANSI escape helpers  return empty string when color is disabled.
// All codes reset with \033[0m so they can't bleed into adjacent text.
static inline const char* C(const char* code) {
    return g_color ? code : "";
}

// Color macros for use in fprintf format strings:
#define CC_RESET    (g_color ? "\033[0m"     : "")
#define CC_BOLD     (g_color ? "\033[1m"     : "")
#define CC_DIM      (g_color ? "\033[2m"     : "")
#define CC_GREEN    (g_color ? "\033[32m"    : "")
#define CC_BGREEN   (g_color ? "\033[1;32m"  : "")
#define CC_YELLOW   (g_color ? "\033[33m"    : "")
#define CC_BYELLOW  (g_color ? "\033[1;33m"  : "")
#define CC_CYAN     (g_color ? "\033[36m"    : "")
#define CC_BCYAN    (g_color ? "\033[1;36m"  : "")
#define CC_BLUE     (g_color ? "\033[34m"    : "")
#define CC_BBLUE    (g_color ? "\033[1;34m"  : "")
#define CC_RED      (g_color ? "\033[31m"    : "")
#define CC_BRED     (g_color ? "\033[1;31m"  : "")
#define CC_WHITE    (g_color ? "\033[37m"    : "")
#define CC_BWHITE   (g_color ? "\033[1;37m"  : "")
// Erase to End of Line: clears from cursor to end of line without overwriting
// with spaces.  Used on completion lines to remove any leftover progress text.
#define CC_EL       (g_color ? "\033[K"      : "    ")

// Set to true while a \r progress bar is live on stderr.
// VLOG checks this to print \r\033[K before its message so the log line
// appears on a clean line rather than appended to the progress bar.
static std::atomic<bool> g_progressActive{false};

// Macro for verbosity-aware output
#define VLOG(level, ...) do { \
    if (g_verbosity >= level) { \
        if (g_progressActive.load(std::memory_order_relaxed)) \
            fprintf(stderr, "\r\033[K"); \
        fprintf(stderr, __VA_ARGS__); \
    } \
} while(0)

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

// CUDA error checking macro with custom error message
#define CUDA_CHECK_MSG(call, msg) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d - %s\n", \
                msg, __FILE__, __LINE__, cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

// nvCOMP error checking macro
#define NVCOMP_CHECK(call) do { \
    nvcompStatus_t status = call; \
    if (status != nvcompSuccess) { \
        fprintf(stderr, "nvCOMP error at %s:%d - status %d\n", \
                __FILE__, __LINE__, static_cast<int>(status)); \
        return false; \
    } \
} while(0)

/*
 * XXH32 - 32-bit xxHash implementation for LZ4 checksums
 * Based on xxHash specification: https://github.com/Cyan4973/xxHash
 */
namespace XXH {
    constexpr uint32_t PRIME32_1 = 2654435761U;
    constexpr uint32_t PRIME32_2 = 2246822519U;
    constexpr uint32_t PRIME32_3 = 3266489917U;
    constexpr uint32_t PRIME32_4 =  668265263U;
    constexpr uint32_t PRIME32_5 =  374761393U;
    
    inline uint32_t rotl32(uint32_t x, int r) {
        return (x << r) | (x >> (32 - r));
    }
    
    inline uint32_t read32(const void* ptr) {
        uint32_t val;
        std::memcpy(&val, ptr, sizeof(val));
        return val;
    }
    
    // Streaming state for incremental hashing
    struct State {
        uint32_t seed;
        uint64_t totalLen;
        uint32_t v1, v2, v3, v4;
        uint8_t buffer[16];
        size_t bufferSize;
        
        State(uint32_t seed = 0) : seed(seed), totalLen(0), bufferSize(0) {
            v1 = seed + PRIME32_1 + PRIME32_2;
            v2 = seed + PRIME32_2;
            v3 = seed + 0;
            v4 = seed - PRIME32_1;
        }
        
        void update(const void* input, size_t length) {
            const uint8_t* p = (const uint8_t*)input;
            totalLen += length;
            
            // If we have buffered data, try to fill the buffer
            if (bufferSize > 0) {
                size_t toCopy = std::min(16 - bufferSize, length);
                std::memcpy(buffer + bufferSize, p, toCopy);
                bufferSize += toCopy;
                p += toCopy;
                length -= toCopy;
                
                // Process buffer if full
                if (bufferSize == 16) {
                    v1 += read32(buffer) * PRIME32_2; v1 = rotl32(v1, 13); v1 *= PRIME32_1;
                    v2 += read32(buffer + 4) * PRIME32_2; v2 = rotl32(v2, 13); v2 *= PRIME32_1;
                    v3 += read32(buffer + 8) * PRIME32_2; v3 = rotl32(v3, 13); v3 *= PRIME32_1;
                    v4 += read32(buffer + 12) * PRIME32_2; v4 = rotl32(v4, 13); v4 *= PRIME32_1;
                    bufferSize = 0;
                }
            }
            
            // Process 16-byte blocks
            const uint8_t* const limit = p + (length & ~15);
            while (p < limit) {
                v1 += read32(p) * PRIME32_2; v1 = rotl32(v1, 13); v1 *= PRIME32_1; p += 4;
                v2 += read32(p) * PRIME32_2; v2 = rotl32(v2, 13); v2 *= PRIME32_1; p += 4;
                v3 += read32(p) * PRIME32_2; v3 = rotl32(v3, 13); v3 *= PRIME32_1; p += 4;
                v4 += read32(p) * PRIME32_2; v4 = rotl32(v4, 13); v4 *= PRIME32_1; p += 4;
            }
            
            // Buffer remaining bytes
            size_t remaining = length & 15;
            if (remaining > 0) {
                std::memcpy(buffer + bufferSize, p, remaining);
                bufferSize += remaining;
            }
        }
        
        uint32_t digest() const {
            uint32_t h32;
            
            if (totalLen >= 16) {
                h32 = rotl32(v1, 1) + rotl32(v2, 7) + rotl32(v3, 12) + rotl32(v4, 18);
            } else {
                h32 = seed + PRIME32_5;
            }
            
            h32 += (uint32_t)totalLen;
            
            // Process buffered data
            const uint8_t* p = buffer;
            const uint8_t* const bEnd = buffer + bufferSize;
            
            while (p + 4 <= bEnd) {
                h32 += read32(p) * PRIME32_3;
                h32 = rotl32(h32, 17) * PRIME32_4;
                p += 4;
            }
            
            while (p < bEnd) {
                h32 += (*p) * PRIME32_5;
                h32 = rotl32(h32, 11) * PRIME32_1;
                p++;
            }
            
            h32 ^= h32 >> 15;
            h32 *= PRIME32_2;
            h32 ^= h32 >> 13;
            h32 *= PRIME32_3;
            h32 ^= h32 >> 16;
            
            return h32;
        }
    };
    
    uint32_t XXH32(const void* input, size_t length, uint32_t seed) {
        const uint8_t* p = (const uint8_t*)input;
        const uint8_t* const bEnd = p + length;
        uint32_t h32;
        
        if (length >= 16) {
            const uint8_t* const limit = bEnd - 15;
            uint32_t v1 = seed + PRIME32_1 + PRIME32_2;
            uint32_t v2 = seed + PRIME32_2;
            uint32_t v3 = seed + 0;
            uint32_t v4 = seed - PRIME32_1;
            
            do {
                v1 += read32(p) * PRIME32_2; v1 = rotl32(v1, 13); v1 *= PRIME32_1; p += 4;
                v2 += read32(p) * PRIME32_2; v2 = rotl32(v2, 13); v2 *= PRIME32_1; p += 4;
                v3 += read32(p) * PRIME32_2; v3 = rotl32(v3, 13); v3 *= PRIME32_1; p += 4;
                v4 += read32(p) * PRIME32_2; v4 = rotl32(v4, 13); v4 *= PRIME32_1; p += 4;
            } while (p < limit);
            
            h32 = rotl32(v1, 1) + rotl32(v2, 7) + rotl32(v3, 12) + rotl32(v4, 18);
        } else {
            h32 = seed + PRIME32_5;
        }
        
        h32 += (uint32_t)length;
        
        while (p + 4 <= bEnd) {
            h32 += read32(p) * PRIME32_3;
            h32 = rotl32(h32, 17) * PRIME32_4;
            p += 4;
        }
        
        while (p < bEnd) {
            h32 += (*p) * PRIME32_5;
            h32 = rotl32(h32, 11) * PRIME32_1;
            p++;
        }
        
        h32 ^= h32 >> 15;
        h32 *= PRIME32_2;
        h32 ^= h32 >> 13;
        h32 *= PRIME32_3;
        h32 ^= h32 >> 16;
        
        return h32;
    }
}

/*
 * LZ4 Frame Format Helper Functions
 */
namespace LZ4Frame {
    
    struct FrameDescriptor {
        uint8_t FLG;
        uint8_t BD;
        uint64_t contentSize;
        bool hasContentSize;
        bool hasContentChecksum;
        bool hasBlockChecksum;
        bool blockIndependence;
        uint8_t blockMaxSize;
        
        FrameDescriptor() 
            : FLG(0), BD(0), contentSize(0)
            , hasContentSize(true)
            , hasContentChecksum(true)
            , hasBlockChecksum(false)
            , blockIndependence(true)
            , blockMaxSize(7) {} // 7 = 4MB
        
        void buildFlags() {
            FLG = 0x40;  // Version 01
            if (blockIndependence) FLG |= 0x20;
            if (hasBlockChecksum) FLG |= 0x10;
            if (hasContentSize) FLG |= 0x08;
            if (hasContentChecksum) FLG |= 0x04;
            
            BD = (blockMaxSize & 0x07) << 4;
        }
        
        uint8_t computeHeaderChecksum() const {
            uint8_t header[15];
            size_t idx = 0;
            
            header[idx++] = FLG;
            header[idx++] = BD;
            
            if (hasContentSize) {
                std::memcpy(&header[idx], &contentSize, 8);
                idx += 8;
            }
            
            uint32_t xxh = XXH::XXH32(header, idx, XXH32_SEED);
            return (xxh >> 8) & 0xFF;  // Use second byte
        }
    };
    
    void writeU32(std::ostream& out, uint32_t value) {
        out.write(reinterpret_cast<const char*>(&value), 4);
    }
    
    void writeU64(std::ostream& out, uint64_t value) {
        out.write(reinterpret_cast<const char*>(&value), 8);
    }
    
    uint32_t readU32(std::istream& in) {
        uint32_t value;
        in.read(reinterpret_cast<char*>(&value), 4);
        return value;
    }
    
    uint64_t readU64(std::istream& in) {
        uint64_t value;
        in.read(reinterpret_cast<char*>(&value), 8);
        return value;
    }
    
    /*
     * Calculate blockMaxSize value for LZ4 header from actual chunk size
     * blockMaxSize encoding: size = 1 << (8 + 2*blockMaxSize)
     * 4: 64KB, 5: 256KB, 6: 1MB, 7: 4MB
     */
    uint8_t calculateBlockMaxSize(size_t chunkSizeBytes) {
        if (chunkSizeBytes <= 64 * 1024) return 4;        // 64KB
        if (chunkSizeBytes <= 256 * 1024) return 5;       // 256KB
        if (chunkSizeBytes <= 1024 * 1024) return 6;      // 1MB
        if (chunkSizeBytes <= 4 * 1024 * 1024) return 7;  // 4MB (includes 2MB, 3MB, 3.5MB)
        return 7;                                          // 4MB (LZ4 frame limit)
    }
    
    bool writeFrameHeader(std::ostream& out, size_t contentSize, size_t chunkSizeBytes,
                          bool storeContentSize = true) {
        // Write magic number
        writeU32(out, LZ4_MAGIC);
        
        // Build frame descriptor
        FrameDescriptor desc;
        desc.hasContentSize = storeContentSize && (contentSize > 0);
        desc.contentSize = desc.hasContentSize ? contentSize : 0;
        desc.hasContentChecksum = true;
        desc.hasBlockChecksum = false;
        desc.blockIndependence = true;
        desc.blockMaxSize = calculateBlockMaxSize(chunkSizeBytes);
        desc.buildFlags();
        
        // Write descriptor
        out.put(desc.FLG);
        out.put(desc.BD);
        
        if (desc.hasContentSize) {
            writeU64(out, desc.contentSize);
        }
        
        // Write header checksum
        uint8_t hc = desc.computeHeaderChecksum();
        out.put(hc);
        
        VLOG(DEBUG, "Wrote LZ4 frame header: magic=0x%08X, blockMaxSize=%d (%zu bytes), size=%zu, HC=0x%02X\n",
             LZ4_MAGIC, desc.blockMaxSize, chunkSizeBytes, contentSize, hc);
        
        return out.good();
    }
    
    bool readFrameHeader(std::istream& in, FrameDescriptor& desc) {
        // Read and verify magic
        uint32_t magic = readU32(in);
        if (magic != LZ4_MAGIC) {
            fprintf(stderr, "Error: Invalid LZ4 magic number: 0x%08X\n", magic);
            return false;
        }
        
        // Read descriptor
        desc.FLG = in.get();
        desc.BD = in.get();
        
        // Parse flags
        uint8_t version = (desc.FLG >> 6) & 0x03;
        if (version != 0x01) {
            fprintf(stderr, "Error: Unsupported LZ4 version: %d\n", version);
            return false;
        }
        
        desc.blockIndependence = (desc.FLG & 0x20) != 0;
        desc.hasBlockChecksum = (desc.FLG & 0x10) != 0;
        desc.hasContentSize = (desc.FLG & 0x08) != 0;
        desc.hasContentChecksum = (desc.FLG & 0x04) != 0;
        desc.blockMaxSize = (desc.BD >> 4) & 0x07;
        
        // Read optional content size
        if (desc.hasContentSize) {
            desc.contentSize = readU64(in);
        }
        
        // Read and verify header checksum
        uint8_t hc_read = in.get();
        uint8_t hc_calc = desc.computeHeaderChecksum();
        
        if (hc_read != hc_calc) {
            fprintf(stderr, "Error: LZ4 header checksum mismatch: got 0x%02X, expected 0x%02X\n",
                    hc_read, hc_calc);
            return false;
        }
        
        VLOG(DEBUG, "Read LZ4 frame header: version=%d, blockMaxSize=%d, contentSize=%zu\n",
             version, 1 << (8 + 2 * desc.blockMaxSize), desc.contentSize);
        
        return true;
    }
    
    void writeBlock(std::ostream& out, const std::vector<uint8_t>& compressedData,
                    const std::vector<uint8_t>& originalData, size_t blockNumber = 0) {
        uint32_t compSize = compressedData.size();
        uint32_t origSize = originalData.size();
        
        // If compressed is larger than original, store uncompressed
        if (compSize >= origSize) {
            VLOG(DEBUG, "Block %zu incompressible, storing uncompressed (%u bytes)\n", 
                 blockNumber, origSize);
            uint32_t blockSize = origSize | 0x80000000;  // Set high bit
            writeU32(out, blockSize);
            out.write(reinterpret_cast<const char*>(originalData.data()), origSize);
        } else {
            writeU32(out, compSize);
            out.write(reinterpret_cast<const char*>(compressedData.data()), compSize);
        }
    }
    
    bool writeEndMark(std::ostream& out) {
        writeU32(out, 0x00000000);
        return out.good();
    }
    
    void writeContentChecksum(std::ostream& out, uint32_t checksum) {
        writeU32(out, checksum);
    }
}

/*
 * Structure to hold GPU information and resources
 */
struct GPUDevice {
    int deviceId;
    cudaDeviceProp properties;
    size_t availableMemory;
    size_t totalMemory;
    std::vector<cudaStream_t> streams;

    // Derived from hardware at init time
    int    pipelineDepth;   // asyncEngineCount + 1: H2D + compute + D2H concurrency
    size_t optimalBatch;    // chunks/slot that keeps SMs busy (recomputed dynamically)
    size_t smCount;         // multiProcessorCount

    // NVML load snapshot  updated periodically by the load monitor thread.
    // smUtilPct:  SM utilization 0-100 (from nvmlDeviceGetUtilizationRates)
    // memUtilPct: memory bandwidth utilization 0-100
    // loadScore:  composite 0-100; lower = more available for our work
    std::atomic<uint32_t> smUtilPct{0};
    std::atomic<uint32_t> memUtilPct{0};
    std::atomic<uint32_t> loadScore{0};  // 0 = fully idle, 100 = fully busy

    // Capacity fractions derived from loadScore  set by load monitor,
    // read by dispatcher and GPU workers.
    // streamFrac: fraction of full SC_h to allocate (0.25 / 0.50 / 0.75 / 1.0)
    // batchFrac:  fraction of full batch size
    std::atomic<uint32_t> streamPct{100}; // percent of full stream count
    std::atomic<uint32_t> batchPct{100};  // percent of full batch size

    GPUDevice(int id) : deviceId(id), pipelineDepth(3), optimalBatch(64), smCount(1) {}

    // Atomic members delete the copy constructor  provide explicit move.
    GPUDevice(GPUDevice&& o) noexcept
        : deviceId(o.deviceId)
        , properties(o.properties)
        , availableMemory(o.availableMemory)
        , totalMemory(o.totalMemory)
        , streams(std::move(o.streams))
        , pipelineDepth(o.pipelineDepth)
        , optimalBatch(o.optimalBatch)
        , smCount(o.smCount)
        , smUtilPct(o.smUtilPct.load())
        , memUtilPct(o.memUtilPct.load())
        , loadScore(o.loadScore.load())
        , streamPct(o.streamPct.load())
        , batchPct(o.batchPct.load())
    {}
    GPUDevice& operator=(GPUDevice&&) = delete;
    GPUDevice(const GPUDevice&)        = delete;
    GPUDevice& operator=(const GPUDevice&) = delete;

    ~GPUDevice() {}
};

/*
 * Structure to hold batch compression state
 */
struct BatchCompressState {
    size_t batch_size;                        // Number of chunks in batch
    std::vector<uint8_t*> d_inputs;           // Multiple input buffers
    const void** d_input_ptrs;                // Pointers array (device)
    size_t* d_input_sizes;                    // Sizes array (device)
    void* d_temp;
    std::vector<void*> d_outputs;             // Multiple output buffers
    void** d_output_ptrs;                     // Output pointers array (device)
    size_t* d_output_sizes;                   // Output sizes array (device)
    nvcompStatus_t* d_statuses;               // Status array (device)
    std::vector<size_t> max_output_sizes;
    std::vector<size_t> input_sizes;
    // Pinned host staging buffers - must outlive async copies
    void* h_input_ptrs_pinned  = nullptr;
    void* h_input_sizes_pinned = nullptr;
    void* h_output_ptrs_pinned = nullptr;
};

// ── Pre-allocated GPU compression slot ──────────────────────────────────────
// All device and pinned-host memory is allocated once at startup.
// Zero cudaMalloc / cudaFree calls during compression hot path.

/*
 * PinnedInputPool  pre-allocated pool of pinned host memory slots.
 */
class PinnedInputPool {
public:
    struct Handle {
        uint8_t*        data     = nullptr;
        size_t          size     = 0;
        size_t          chunkIdx = 0;
        int             slotId   = -1;
        PinnedInputPool* pool    = nullptr;

        Handle()  = default;
        ~Handle() { release(); }

        Handle(Handle&& o) noexcept
            : data(o.data), size(o.size), chunkIdx(o.chunkIdx),
              slotId(o.slotId), pool(o.pool)
        { o.slotId = -1; o.pool = nullptr; }

        Handle& operator=(Handle&& o) noexcept {
            release();
            data=o.data; size=o.size; chunkIdx=o.chunkIdx;
            slotId=o.slotId; pool=o.pool;
            o.slotId=-1; o.pool=nullptr;
            return *this;
        }

        Handle(const Handle&)            = delete;
        Handle& operator=(const Handle&) = delete;

        bool valid() const { return pool != nullptr && slotId >= 0; }

        void release() {
            if (pool && slotId >= 0) {
                pool->releaseSlot(slotId);
                slotId = -1; pool = nullptr;
            }
        }
    };

private:
    uint8_t*              base_     = nullptr;
    size_t                slotSz_   = 0;
    size_t                nSlots_   = 0;
    std::vector<int>      freeList_;
    std::mutex            mu_;
    std::condition_variable cv_;
    std::atomic<bool>     shutdown_{false};

    void releaseSlot(int id) {
        { std::lock_guard<std::mutex> lk(mu_); freeList_.push_back(id); }
        cv_.notify_one();
    }
    friend struct Handle;

public:
    ~PinnedInputPool() { destroy(); }

    bool init(size_t nSlots, size_t slotSize) {
        slotSz_  = slotSize;
        nSlots_  = nSlots;
        cudaError_t err = cudaHostAlloc(&base_, nSlots * slotSize, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            fprintf(stderr, "PinnedInputPool: cudaHostAlloc(%zu Ã %zu MB) failed: %s\n",
                    nSlots, slotSize>>20, cudaGetErrorString(err));
            return false;
        }
        freeList_.resize(nSlots);
        std::iota(freeList_.begin(), freeList_.end(), 0);
        VLOG(DEBUG, "PinnedInputPool: %zu Ã %.1f MB = %.1f GB pinned\n",
             nSlots, slotSize/(1024.0*1024.0),
             (nSlots*slotSize)/(1024.0*1024.0*1024.0));
        return true;
    }

    void destroy() {
        if (base_) { cudaFreeHost(base_); base_ = nullptr; }
        nSlots_ = 0;
        freeList_.clear();
    }

    void shutdown() {
        shutdown_.store(true);
        cv_.notify_all();
    }

    // Acquire a free slot  blocks until one is available or shutdown
    Handle acquire() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this]{ return !freeList_.empty() || shutdown_.load(); });
        if (freeList_.empty()) return {};   // shutdown
        int id = freeList_.back(); freeList_.pop_back();
        Handle h;
        h.data   = base_ + id * slotSz_;
        h.size   = slotSz_;
        h.slotId = id;
        h.pool   = this;
        return h;
    }

    size_t slotSize()  const { return slotSz_; }
    size_t numSlots()  const { return nSlots_;  }
    size_t numFree()   const {
        std::lock_guard<std::mutex> lk(const_cast<std::mutex&>(mu_));
        return freeList_.size();
    }
};

struct PreallocSlot {
    int      deviceId    = -1;
    size_t   capacity    = 0;        // max chunks per batch
    size_t   chunkStride = 0;        // bytes per input slot  (= chunkSize)
    size_t   outStride   = 0;        // max output bytes per chunk (LZ4 bound)
    size_t   tempBytes   = 0;        // nvCOMP scratch size

    // Contiguous device pools (allocated once)
    uint8_t*        d_input  = nullptr;   // capacity * chunkStride
    uint8_t*        d_output = nullptr;   // capacity * outStride
    void*           d_temp   = nullptr;

    // Pre-filled device pointer arrays  point into d_input / d_output
    const void**    d_iPtrs  = nullptr;   // [capacity] -> d_input[i]
    void**          d_oPtrs  = nullptr;   // [capacity] -> d_output[i]

    // Per-batch device metadata (small, written each batch)
    size_t*         d_iSizes = nullptr;
    size_t*         d_oSizes = nullptr;
    nvcompStatus_t* d_stats  = nullptr;

    // Pinned host buffers
    size_t*         h_iSizes = nullptr;   // H→D staging for input sizes
    size_t*         h_oSizes = nullptr;   // D→H result sizes
    nvcompStatus_t* h_stats  = nullptr;   // D→H result statuses
    uint8_t*        h_output = nullptr;   // D→H output data (capacity * outStride)

    cudaStream_t    stream          = 0;
    bool            ready           = false;

    // Per-batch state (set by worker thread each iteration)
    size_t                            batchSize  = 0;
    bool                              hasPending = false;
    std::vector<size_t>               indices;
    std::vector<size_t>               origSizes;
    // Input data: held until after D→H completes, then released back to pool
    std::vector<PinnedInputPool::Handle> origHandles;  // pooled path
    std::vector<std::vector<uint8_t>>    origData;     // fallback path

    void release() {
        if (!ready) return;
        if (deviceId >= 0) cudaSetDevice(deviceId);
        cudaFree(d_input);  cudaFree(d_output); cudaFree(d_temp);
        cudaFree(d_iPtrs);  cudaFree(d_oPtrs);
        cudaFree(d_iSizes); cudaFree(d_oSizes); cudaFree(d_stats);
        cudaFreeHost(h_iSizes); cudaFreeHost(h_oSizes);
        cudaFreeHost(h_stats);  cudaFreeHost(h_output);
        if (stream) { cudaStreamDestroy(stream); stream = 0; }
        *this = PreallocSlot{};
    }
};

// Thread-safe queue used to pass completed batches from GPU worker threads
// to the main thread / AsyncWriter.
template<typename T>
class TsQueue {
    std::queue<T>           q_;
    mutable std::mutex      m_;
    std::condition_variable cv_;
    std::atomic<bool>       closed_{false};
public:
    void push(T&& item) {
        { std::lock_guard<std::mutex> lk(m_); q_.push(std::move(item)); }
        cv_.notify_one();
    }

    // Blocking pop: sleeps until an item is available or the queue is closed.
    // Returns true with item, or false if the queue is closed and empty.
    // No timeout  wakes exactly when work arrives or close() is called.
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&]{ return !q_.empty() || closed_.load(); });
        if (q_.empty()) return false;
        out = std::move(q_.front()); q_.pop();
        return true;
    }

    // Timed pop: used only by callers that need to check a secondary
    // termination condition (e.g. a workerAbort flag) that the queue
    // itself cannot observe.  Prefer the blocking pop() where possible.
    bool pop_for(T& out, int timeoutMs) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait_for(lk, std::chrono::milliseconds(timeoutMs),
                     [&]{ return !q_.empty() || closed_.load(); });
        if (q_.empty()) return false;
        out = std::move(q_.front()); q_.pop();
        return true;
    }

    void close()         { closed_.store(true); cv_.notify_all(); }
    bool isClosed() const{ return closed_.load(); }
    size_t size()  const { std::lock_guard<std::mutex> lk(m_); return q_.size(); }
};

/*
 * Structure to hold batch decompression state
 */
struct BatchDecompressState {
    size_t batch_size;                        // Number of chunks in batch
    std::vector<uint8_t*> d_inputs;           // Multiple input buffers
    const void** d_input_ptrs;                // Pointers array (device)
    size_t* d_input_sizes;                    // Sizes array (device)
    size_t* d_output_sizes;                   // Expected output sizes (device)
    void* d_temp;
    std::vector<void*> d_outputs;             // Multiple output buffers
    void** d_output_ptrs;                     // Output pointers array (device)
    size_t* d_actual_output_sizes;            // Actual output sizes (device)
    nvcompStatus_t* d_statuses;               // Status array (device)
    std::vector<size_t> uncompressed_sizes;
};

/*
 * Asynchronous Reader with Advanced I/O
 */

class AsyncReader {
public:
    struct ReadChunk {
        size_t chunkIndex = 0;
        size_t size       = 0;

        // Storage: either a pinned pool handle OR a heap vector.
        PinnedInputPool::Handle poolHandle;   // valid when using pool
        std::vector<uint8_t>    heapData;     // used in fallback mode

        uint8_t*       data()       { return poolHandle.valid() ? poolHandle.data : heapData.data(); }
        const uint8_t* data() const { return poolHandle.valid() ? poolHandle.data : heapData.data(); }
    };

private:
    std::thread  readerThread;

    std::queue<ReadChunk>    readQueue;
    std::mutex               queueMutex;
    std::condition_variable  queueCV;

    std::atomic<bool>   shouldStop{false};
    std::atomic<bool>   finished{false};
    std::atomic<size_t> bytesRead{0};
    std::atomic<double> totalReadTime{0.0};

    int    inputFd  = -1;
    size_t fileSize = 0;
    size_t chunkSize = 0;

    // Pool mode: non-null when using pinned memory
    PinnedInputPool* pool_        = nullptr;
    // Fallback mode: limit queue depth to bound RAM
    size_t           maxQueuedChunks = 0;

    // Read exactly `count` bytes from fd, looping over short reads.
    static ssize_t readFull(int fd, void* buf, size_t count) {
        size_t total = 0;
        uint8_t* p   = static_cast<uint8_t*>(buf);
        while (total < count) {
            ssize_t n = ::read(fd, p + total, count - total);
            if (n < 0) return (total > 0) ? (ssize_t)total : -1;
            if (n == 0) break; // EOF
            total += (size_t)n;
        }
        return (ssize_t)total;
    }

    void readerLoop() {
        size_t chunkIndex = 0;
        size_t totalRead  = 0;
        auto   t0         = std::chrono::high_resolution_clock::now();
        // fileSize==0 means stdin/pipe: read until EOF, chunk size is fixed.
        const bool pipeMode = (fileSize == 0);

        while (!shouldStop.load()) {
            if (!pipeMode && totalRead >= fileSize) break;

            size_t toRead = pipeMode ? chunkSize
                                     : std::min(chunkSize, fileSize - totalRead);
            ReadChunk chunk;
            chunk.chunkIndex = chunkIndex;

            if (pool_) {
                chunk.poolHandle = pool_->acquire();
                if (!chunk.poolHandle.valid()) break;  // shutdown

                auto rs = std::chrono::high_resolution_clock::now();
                ssize_t n = pipeMode
                    ? readFull(inputFd, chunk.poolHandle.data, toRead)
                    : ::read(inputFd, chunk.poolHandle.data, toRead);
                auto re = std::chrono::high_resolution_clock::now();
                totalReadTime = totalReadTime.load() +
                    std::chrono::duration<double>(re - rs).count();
                if (pipeMode) {
                    if (n <= 0) { chunk.poolHandle.release(); break; } // EOF
                    chunk.poolHandle.size     = (size_t)n;
                    chunk.poolHandle.chunkIdx = chunkIndex;
                    chunk.size = (size_t)n;
                } else {
                    if (n != (ssize_t)toRead) {
                        fprintf(stderr, "Reader: read error chunk %zu: %s\n",
                                chunkIndex, strerror(errno));
                        chunk.poolHandle.release();
                        break;
                    }
                    chunk.poolHandle.size     = toRead;
                    chunk.poolHandle.chunkIdx = chunkIndex;
                    chunk.size = toRead;
                }
            } else {
                {
                    std::unique_lock<std::mutex> lk(queueMutex);
                    queueCV.wait(lk, [this]{
                        return readQueue.size() < maxQueuedChunks || shouldStop.load();
                    });
                    if (shouldStop.load()) break;
                }
                chunk.heapData.resize(toRead);

                auto rs = std::chrono::high_resolution_clock::now();
                ssize_t n = pipeMode
                    ? readFull(inputFd, chunk.heapData.data(), toRead)
                    : ::read(inputFd, chunk.heapData.data(), toRead);
                auto re = std::chrono::high_resolution_clock::now();
                totalReadTime = totalReadTime.load() +
                    std::chrono::duration<double>(re - rs).count();
                if (pipeMode) {
                    if (n <= 0) break; // EOF
                    chunk.heapData.resize((size_t)n); // trim to actual bytes read
                    chunk.size = (size_t)n;
                } else {
                    if (n != (ssize_t)toRead) {
                        fprintf(stderr, "Reader: read error chunk %zu: %s\n",
                                chunkIndex, strerror(errno));
                        break;
                    }
                    chunk.size = toRead;
                }
            }

            totalRead  += chunk.size;
            bytesRead  += chunk.size;
            chunkIndex++;

            { std::lock_guard<std::mutex> lk(queueMutex); readQueue.push(std::move(chunk)); }
            queueCV.notify_one();
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        finished.store(true);
        queueCV.notify_all();
        VLOG(VERBOSE, "Reader thread finished: read %zu bytes in %.2fs"
             " (%.2fs actual I/O, %.2fs waiting)\n",
             bytesRead.load(),
             std::chrono::duration<double>(t1-t0).count(),
             totalReadTime.load(),
             std::chrono::duration<double>(t1-t0).count() - totalReadTime.load());
    }

public:
    AsyncReader()  = default;
    ~AsyncReader() { stop(); }

    // ── Pooled start: reader uses pre-allocated pinned slots ─────────────────
    bool startPooled(const std::string& filename, size_t chunk_size,
                     PinnedInputPool* pool) {
        pool_      = pool;
        chunkSize  = chunk_size;
        return openAndLaunch(filename);
    }

    // ── Fallback start: heap allocation with queue cap ────────────────────────
    bool start(const std::string& filename, size_t chunk_size,
               size_t max_queued = 128) {
        pool_           = nullptr;
        chunkSize       = chunk_size;
        maxQueuedChunks = max_queued;
        return openAndLaunch(filename);
    }

private:
    bool openAndLaunch(const std::string& filename) {
        if (filename == "-") {
            inputFd  = STDIN_FILENO;
            fileSize = 0;
            fcntl(STDIN_FILENO, F_SETPIPE_SZ, 1 << 20);
            VLOG(VERBOSE, "AsyncReader: reading from stdin (pipe)\n");
        } else {
            inputFd = open(filename.c_str(), O_RDONLY);
            if (inputFd < 0) {
                fprintf(stderr, "Error opening %s: %s\n", filename.c_str(), strerror(errno));
                return false;
            }
            struct stat st;
            fstat(inputFd, &st);
            fileSize = st.st_size;
            posix_fadvise(inputFd, 0, fileSize, POSIX_FADV_SEQUENTIAL);
            posix_fadvise(inputFd, 0, fileSize, POSIX_FADV_WILLNEED);
            VLOG(VERBOSE, "AsyncReader: opened %s (%.2f MB) for reading\n",
                 filename.c_str(), fileSize/(1024.0*1024.0));
        }
        shouldStop.store(false);
        finished.store(false);
        readerThread = std::thread(&AsyncReader::readerLoop, this);
        return true;
    }

public:
    bool getChunk(ReadChunk& chunk) {
        std::unique_lock<std::mutex> lk(queueMutex);
        queueCV.wait(lk, [this]{ return !readQueue.empty() || finished.load(); });
        if (readQueue.empty()) return false;
        chunk = std::move(readQueue.front());
        readQueue.pop();
        queueCV.notify_one();
        return true;
    }

    void stop() {
        shouldStop.store(true);
        if (pool_) pool_->shutdown();
        queueCV.notify_all();
        if (readerThread.joinable()) readerThread.join();
        if (inputFd >= 0) { if (inputFd != STDIN_FILENO) close(inputFd); inputFd = -1; }
    }

    bool   isFinished()    const { return finished.load() && getQueueDepth() == 0; }
    bool   isPooled()      const { return pool_ != nullptr; }
    size_t getQueueDepth() const {
        std::lock_guard<std::mutex> lk(const_cast<std::mutex&>(queueMutex));
        return readQueue.size();
    }
    size_t getFileSize()  const { return fileSize; }
    size_t getBytesRead() const { return bytesRead.load(); }
    double getReadTime()  const { return totalReadTime.load(); }
};

/*
 * Asynchronous Writer with Advanced I/O
 * Writes completed batches in background thread while GPUs continue working.
 *
 * fsync() is skipped by default (syncOutput=false) because consumer NVMe
 * drives have an SLC write cache that causes 4-20x timing variance on
 * consecutive runs.  Use --sync-output to enable fsync() for durability.
 * The output is still written atomically via .tmp + rename, so it is always
 * either complete or absent after a crash regardless of this setting.
 */
class AsyncWriter {
private:
    // ── Write buffer ──────────────────────────────────────────────────────────
    // Used only for the LZ4 frame header (written by the main thread before
    // AsyncWriter starts) and the end-of-stream marker.  Per-chunk data is
    // sent directly via writev() to avoid a full memcpy.
    static constexpr size_t WRITE_BUF_SIZE = 256ULL * 1024 * 1024;

    std::vector<uint8_t> writeBuf;
    size_t               writeBufUsed = 0;
    bool                 isPipe       = false;
    size_t               flushSize    = WRITE_BUF_SIZE;
    bool                 syncOutput_  = false;

    void bufAppend(const void* data, size_t len) {
        const uint8_t* src = reinterpret_cast<const uint8_t*>(data);
        while (len > 0) {
            size_t space = WRITE_BUF_SIZE - writeBufUsed;
            size_t copy  = std::min(len, space);
            memcpy(writeBuf.data() + writeBufUsed, src, copy);
            writeBufUsed += copy;
            src += copy; len -= copy;
            if (writeBufUsed >= flushSize) bufFlush();
        }
    }

    void bufFlushU32(uint32_t v) {
        uint8_t buf[4] = { uint8_t(v), uint8_t(v>>8), uint8_t(v>>16), uint8_t(v>>24) };
        bufAppend(buf, 4);
    }

    void bufFlush() {
        if (writeBufUsed == 0) return;
        auto t0 = std::chrono::high_resolution_clock::now();
        ssize_t written = ::write(outputFd, writeBuf.data(), writeBufUsed);
        totalWriteTime = totalWriteTime.load() +
            std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t0).count();
        if (written != (ssize_t)writeBufUsed)
            fprintf(stderr, "Write error: %s\n", strerror(errno));
        bytesWritten += writeBufUsed;
        if (!isPipe) {
            off_t pos = lseek(outputFd, 0, SEEK_CUR);
            if (pos >= (off_t)writeBufUsed)
                posix_fadvise(outputFd, pos - writeBufUsed, writeBufUsed,
                              POSIX_FADV_DONTNEED);
        }
        writeBufUsed = 0;
    }

    // ── writev helpers ────────────────────────────────────────────────────────
    // Write all iovecs, retrying on partial writes (can happen on pipes).
    void writevAll(struct iovec* iov, int iovcnt, size_t totalBytes) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // ── Diagnostic validation before writev ──────────────────────────────
        size_t sumLen = 0;
        for (int vi = 0; vi < iovcnt; vi++) {
            sumLen += iov[vi].iov_len;
            if (iov[vi].iov_base == nullptr && iov[vi].iov_len > 0) {
                fprintf(stderr, "DIAG writevAll: iov[%d] null base, len=%zu, "
                        "totalBytes=%zu iovcnt=%d\n",
                        vi, iov[vi].iov_len, totalBytes, iovcnt);
            }
            // Sanity: iov_len should never be 0 (would be a header-only entry)
            if (iov[vi].iov_len == 0) {
                fprintf(stderr, "DIAG writevAll: iov[%d] zero len, base=%p, "
                        "totalBytes=%zu iovcnt=%d\n",
                        vi, iov[vi].iov_base, totalBytes, iovcnt);
            }
            // Sanity: individual entries should never exceed 1 GB
            if (iov[vi].iov_len > (size_t)1 << 30) {
                fprintf(stderr, "DIAG writevAll: iov[%d] suspiciously large len=%zu "
                        "base=%p\n", vi, iov[vi].iov_len, iov[vi].iov_base);
            }
        }
        if (sumLen != totalBytes) {
            fprintf(stderr, "DIAG writevAll: sumLen=%zu != totalBytes=%zu iovcnt=%d\n",
                    sumLen, totalBytes, iovcnt);
        }

        size_t remaining = totalBytes;
        int origIovcnt = iovcnt;
        while (iovcnt > 0 && remaining > 0) {
            ssize_t n = ::writev(outputFd, iov, std::min(iovcnt, IOV_MAX));
            if (n <= 0) {
                fprintf(stderr, "Write error: %s  (errno=%d fd=%d iovcnt=%d/%d "
                        "remaining=%zu totalBytes=%zu)\n",
                        strerror(errno), errno, outputFd, iovcnt, origIovcnt,
                        remaining, totalBytes);
                int dump = std::min(iovcnt, 8);
                for (int vi = 0; vi < dump; vi++)
                    fprintf(stderr, "  iov[%d]: base=%p len=%zu\n",
                            vi, iov[vi].iov_base, iov[vi].iov_len);
                writeError_.store(true);
                shouldStop.store(true);
                queueCV.notify_all();
                break;
            }
            bytesWritten += (size_t)n;
            remaining    -= (size_t)n;
            // Advance iov past written bytes
            size_t skip = (size_t)n;
            while (skip > 0 && iovcnt > 0) {
                if (iov->iov_len <= skip) {
                    skip -= iov->iov_len;
                    iov++; iovcnt--;
                } else {
                    iov->iov_base = (char*)iov->iov_base + skip;
                    iov->iov_len -= skip;
                    skip = 0;
                }
            }
        }
        totalWriteTime = totalWriteTime.load() +
            std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t0).count();
        if (!isPipe) {
            off_t pos = lseek(outputFd, 0, SEEK_CUR);
            if (pos >= (off_t)totalBytes)
                posix_fadvise(outputFd, pos - totalBytes, totalBytes,
                              POSIX_FADV_DONTNEED);
        }
    }

    // ── Per-chunk write task ──────────────────────────────────────────────────
    struct WriteTask {
        size_t chunkIndex;
        std::vector<std::vector<uint8_t>> compressedChunks;
        std::vector<std::vector<uint8_t>> originalChunks;
        std::vector<size_t> chunkIndices;
        std::vector<size_t> originalSizes;
    };

    struct HashWork {
        std::vector<uint8_t> data;
        size_t               origSize = 0;
    };

    TsQueue<HashWork> hashQueue_;
    std::thread       hashThread_;

    void hashLoop() {
        HashWork work;
        while (hashQueue_.pop(work))
            xxhState->update(work.data.data(), work.origSize);
    }

    // ── Pending write queue ───────────────────────────────────────────────────
    // unordered_map: O(1) insert/find vs std::map O(log n)
    std::thread writerThread;
    std::unordered_map<size_t, WriteTask> pendingWrites;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool>   shouldStop{false};
    std::atomic<bool>   writeError_{false};  // set on writev failure
    std::atomic<size_t> bytesWritten{0};
    std::atomic<double> totalWriteTime{0.0};
    std::atomic<size_t> nextChunkToWrite{0};
    std::atomic<bool>   writerDone{false};
    std::atomic<size_t> totalExpectedChunks{SIZE_MAX};

    int         outputFd  = -1;
    std::string outputFile;
    XXH::State* xxhState  = nullptr;

    void writerLoop() {
        VLOG(DEBUG, "Writer thread started\n");
        auto threadStart = std::chrono::high_resolution_clock::now();

        // Per-batch metadata and iovec storage  rebuilt each iteration.
        std::vector<uint32_t>     headers;
        std::vector<struct iovec> iovecs;

        while (true) {
            WriteTask task;
            bool hasTask = false;

            {
                std::unique_lock<std::mutex> lock(queueMutex);
                // Wake on: next chunk ready, stop requested, OR workerAbort
                // (missing chunk due to error  avoids infinite wait).
                queueCV.wait(lock, [this] {
                    return pendingWrites.count(nextChunkToWrite.load()) > 0
                           || shouldStop.load();
                });
                bool allDone = nextChunkToWrite.load() >= totalExpectedChunks.load();
                if (pendingWrites.empty() && (shouldStop.load() || allDone)) break;
                auto it = pendingWrites.find(nextChunkToWrite.load());
                if (it != pendingWrites.end()) {
                    task    = std::move(it->second);
                    pendingWrites.erase(it);
                    hasTask = true;
                } else if (shouldStop.load()) {
                    // shouldStop set but next chunk not present  worker aborted,
                    // drain what we have and exit rather than waiting forever.
                    break;
                }
            }  // ← lock released before any I/O

            while (hasTask) {
                // ── Write one task per writev call ───────────────────────
                // Process one task per iteration: build iovecs for all chunks
                // in this task and issue one writev() call.  No cross-task
                // coalescing  a single GPU batch at level 9 is already 128
                // chunks × ~4MB = ~512MB per writev, which is plenty.
                // Using a stable header array (sized before any &headers[i]
                // is taken) prevents the reallocation-invalidates-pointer bug.
                size_t nChunks = task.originalChunks.size();
                headers.resize(nChunks);
                iovecs.clear();
                iovecs.reserve(nChunks * 2);
                size_t totalBytes = 0;

                for (size_t i = 0; i < nChunks; i++) {
                    size_t origSize = task.originalSizes[i];
                    bool hasComp = i < task.compressedChunks.size()
                                   && !task.compressedChunks[i].empty()
                                   && task.compressedChunks[i].size() < origSize;
                    if (hasComp) {
                        headers[i] = (uint32_t)task.compressedChunks[i].size();
                        iovecs.push_back({(void*)&headers[i], 4});
                        iovecs.push_back({task.compressedChunks[i].data(),
                                          task.compressedChunks[i].size()});
                        totalBytes += 4 + task.compressedChunks[i].size();
                    } else {
                        headers[i] = (uint32_t)origSize | 0x80000000u;
                        iovecs.push_back({(void*)&headers[i], 4});
                        iovecs.push_back({task.originalChunks[i].data(), origSize});
                        totalBytes += 4 + origSize;
                    }
                }

                if (!iovecs.empty())
                    writevAll(iovecs.data(), (int)iovecs.size(), totalBytes);

                // Hash after writev  move data out of task (no copy needed)
                if (xxhState) {
                    for (size_t i = 0; i < nChunks; i++) {
                        HashWork w;
                        w.origSize = task.originalSizes[i];
                        w.data     = std::move(task.originalChunks[i]);
                        hashQueue_.push(std::move(w));
                    }
                }

                // Advance and check for next task
                nextChunkToWrite += task.chunkIndices.size();
                {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    auto it = pendingWrites.find(nextChunkToWrite.load());
                    if (it != pendingWrites.end()) {
                        task    = std::move(it->second);
                        pendingWrites.erase(it);
                        hasTask = true;
                    } else {
                        hasTask = false;
                        lock.unlock();
                        queueCV.notify_all();
                    }
                }
            }
        }

        bufFlush();  // flush any trailing header bytes

        if (xxhState) hashQueue_.close();
        writerDone.store(true);

        auto threadEnd = std::chrono::high_resolution_clock::now();
        double total = std::chrono::duration<double>(threadEnd - threadStart).count();
        VLOG(VERBOSE, "Writer thread finished: wrote %zu bytes in %.2fs "
             "(%.2fs actual I/O, %.2fs waiting)\n",
             bytesWritten.load(), total, totalWriteTime.load(),
             total - totalWriteTime.load());
    }

public:
    AsyncWriter() = default;
    ~AsyncWriter() { stop(); }

    bool start(const std::string& filename, XXH::State* xxh, bool syncOutput = false) {
        outputFile  = filename;
        xxhState    = xxh;
        syncOutput_ = syncOutput;

        writeBuf.resize(WRITE_BUF_SIZE);
        writeBufUsed = 0;

        if (filename == "-") {
            outputFd = STDOUT_FILENO;
            isPipe   = true;
            flushSize = 4ULL * 1024 * 1024;
            // Increase pipe buffer to reduce blocking frequency
#ifdef F_SETPIPE_SZ
            fcntl(outputFd, F_SETPIPE_SZ, 1 << 20);  // request 1 MB pipe buffer
#endif
            VLOG(VERBOSE, "AsyncWriter: writing to stdout (pipe mode)\n");
        } else {
            outputFd = open(filename.c_str(), O_WRONLY | O_APPEND, 0644);
            if (outputFd < 0) {
                fprintf(stderr, "Error opening output file %s: %s\n",
                        filename.c_str(), strerror(errno));
                return false;
            }
            posix_fadvise(outputFd, 0, 0, POSIX_FADV_SEQUENTIAL);
            isPipe    = (lseek(outputFd, 0, SEEK_CUR) < 0);
            flushSize = isPipe ? 4ULL * 1024 * 1024 : WRITE_BUF_SIZE;
            if (isPipe) {
#ifdef F_SETPIPE_SZ
                fcntl(outputFd, F_SETPIPE_SZ, 1 << 20);
#endif
            }
            VLOG(VERBOSE, "AsyncWriter: opened %s for appending chunks%s\n",
                 filename.c_str(), isPipe ? " (pipe/fifo)" : "");
        }

        shouldStop.store(false);
        writerThread = std::thread(&AsyncWriter::writerLoop, this);
        if (xxhState)
            hashThread_ = std::thread(&AsyncWriter::hashLoop, this);

        return true;
    }

    void enqueue(size_t chunkIndex,
                 std::vector<std::vector<uint8_t>> compressedChunks,
                 std::vector<std::vector<uint8_t>> originalChunks,
                 std::vector<size_t> chunkIndices,
                 std::vector<size_t> originalSizes) {
        WriteTask task;
        task.chunkIndex       = chunkIndex;
        task.compressedChunks = std::move(compressedChunks);
        task.originalChunks   = std::move(originalChunks);
        task.chunkIndices     = std::move(chunkIndices);
        task.originalSizes    = std::move(originalSizes);
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            pendingWrites[chunkIndex] = std::move(task);
        }
        queueCV.notify_one();
    }

    void enqueueBatch(std::vector<std::vector<uint8_t>>& compressedChunks,
                      std::vector<std::vector<uint8_t>>& originalChunks,
                      const std::vector<size_t>&         chunkIndices,
                      const std::vector<size_t>&         originalSizes) {
        // ── Diagnostic validation ─────────────────────────────────────────────
        for (size_t i = 0; i < chunkIndices.size(); i++) {
            bool hasComp = i < compressedChunks.size() && !compressedChunks[i].empty();
            size_t origSz = i < originalSizes.size() ? originalSizes[i] : 0;
            if (hasComp && compressedChunks[i].data() == nullptr) {
                fprintf(stderr, "DIAG enqueueBatch: chunk %zu compressedChunks[%zu] "
                        "non-empty but null ptr, size=%zu\n",
                        chunkIndices[i], i, compressedChunks[i].size());
            }
            if (!hasComp) {
                if (i >= originalChunks.size() || originalChunks[i].empty()) {
                    fprintf(stderr, "DIAG enqueueBatch: chunk %zu NO comp AND "
                            "originalChunks[%zu] empty! origSz=%zu\n",
                            chunkIndices[i], i, origSz);
                } else if (originalChunks[i].size() != origSz && origSz > 0) {
                    fprintf(stderr, "DIAG enqueueBatch: chunk %zu size mismatch "
                            "orig.size=%zu origSizes[%zu]=%zu\n",
                            chunkIndices[i], originalChunks[i].size(), i, origSz);
                }
            }
        }
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            for (size_t i = 0; i < chunkIndices.size(); i++) {
                WriteTask task;
                task.chunkIndex    = chunkIndices[i];
                task.chunkIndices  = { chunkIndices[i] };
                task.originalSizes = { originalSizes[i] };
                task.compressedChunks.push_back(std::move(compressedChunks[i]));
                task.originalChunks  .push_back(std::move(originalChunks[i]));
                pendingWrites[chunkIndices[i]] = std::move(task);
            }
        }
        queueCV.notify_all();
    }

    void stop() {
        shouldStop.store(true);
        queueCV.notify_one();
        if (writerThread.joinable()) writerThread.join();
        if (hashThread_.joinable())  hashThread_.join();
        if (outputFd >= 0 && outputFd != STDOUT_FILENO) {
            if (syncOutput_) fsync(outputFd);
            close(outputFd);
            outputFd = -1;
        }
    }

    size_t getBytesWritten()     const { return bytesWritten.load(); }
    double getWriteTime()        const { return totalWriteTime.load(); }
    size_t getNextChunkToWrite() const { return nextChunkToWrite.load(); }
    bool   isDone()              const { return writerDone.load(); }
    bool   hasWriteError()       const { return writeError_.load(); }
    void   setTotalChunks(size_t n)    { totalExpectedChunks.store(n); }

    size_t getQueueDepth() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(queueMutex));
        return pendingWrites.size();
    }
};

/*
 * Pre-decompression reader: reads LZ4 compressed blocks from disk into a
 * queue concurrently with GPU context initialization.  The decompressor
 * drains this queue first, then continues reading from the fd directly.
 * This eliminates the dead time where the disk is idle while CUDA contexts
 * are being created (which can take 1-2 s per GPU).
 */
class PreDecompReader {
public:
    struct Block {
        uint32_t             sizeField; // raw 4-byte prefix (high bit = uncompressed)
        std::vector<uint8_t> data;
    };

    TsQueue<Block>        queue;
    std::atomic<size_t>   bytesRead{0};           // bytes read from fd (for progress display)
    std::atomic<size_t>   bytesConsumedFromFd{0}; // exact bytes consumed including size headers
    size_t                hdrConsumedBytes{0};     // bytes of LZ4 frame header consumed before blocks
    std::atomic<bool>     done{false};
    std::atomic<bool>     error{false};
    std::vector<uint8_t>  headerBytes;   // raw LZ4 frame header (already consumed)
    size_t                compressedFileSize{0};
    int                   fd{-1};        // fd positioned after headerBytes
    bool                  stdinMode{false};

    // Start reading: opens file, reads+stores header, then fills queue in a thread.
    // headerSize bytes have already been consumed from fd by the caller (header parse).
    bool start(int openedFd, bool isStdin, size_t fileSize,
               const std::vector<uint8_t>& hdrBytes, size_t hdrConsumed) {
        fd                 = openedFd;
        stdinMode          = isStdin;
        compressedFileSize = fileSize;
        headerBytes        = hdrBytes;
        hdrConsumedBytes   = hdrConsumed;

        thread_ = std::thread([this]() {
            while (true) {
                Block blk;
                ssize_t n = ::read(fd, &blk.sizeField, 4);
                if (n == 0) break;
                if (n != 4) { error.store(true); break; }
                // Treat 0-size as EOF sentinel (end-of-frame marker)
                if (blk.sizeField == 0) {
                    bytesConsumedFromFd.fetch_add(4, std::memory_order_relaxed);
                    queue.push(std::move(blk));
                    break;
                }
                uint32_t dataSize = blk.sizeField & 0x7FFFFFFFu;
                if (dataSize > 256u * 1024 * 1024) { error.store(true); break; }
                blk.data.resize(dataSize);
                ssize_t got = ::read(fd, blk.data.data(), dataSize);
                if (got != (ssize_t)dataSize) { error.store(true); break; }
                size_t blockTotal = 4 + dataSize;
                bytesRead.fetch_add(blockTotal, std::memory_order_relaxed);
                bytesConsumedFromFd.fetch_add(blockTotal, std::memory_order_relaxed);
                try {
                    queue.push(std::move(blk));
                } catch (const std::bad_alloc&) {
                    // OOM: stop pre-reading; decompressor will read from fd directly
                    break;
                }
            }
            done.store(true);
            queue.close();
        });
        return true;
    }

    // Drain: wait for thread to finish and return whether any data was buffered.
    // Called if GPU init fails and we fall back before the decompressor runs.
    void stop() {
        queue.close();
        if (thread_.joinable()) thread_.join();
    }

    bool isStarted() const { return thread_.joinable() || done.load(); }

    ~PreDecompReader() { stop(); }

private:
    std::thread thread_;
};

/*
 * Multi-threaded CPU Compression Pool
 * Parallel LZ4 compression using CPU threads
 */
class CPUCompressionPool {
public:
    struct CompressResult {
        size_t chunkIndex;
        std::vector<uint8_t> compressedData;
        std::vector<uint8_t> originalData;   // Always populated
        size_t originalSize;
        bool success;
    };
    
private:
    struct CompressJob {
        size_t chunkIndex;
        std::vector<uint8_t> inputData;
        size_t maxOutputSize;
        int    hcLevel      = 0;      // 0=fast, 1-12=HC
    };
    
    std::vector<std::thread> workers;
    std::queue<CompressJob> jobQueue;
    std::map<size_t, CompressResult> completedJobs;
    std::mutex jobMutex;
    std::mutex resultMutex;
    std::condition_variable jobCV;
    std::condition_variable resultCV;
    std::atomic<bool> shouldStop{false};
    // Set by finish() once all jobs are submitted.  Unlike shouldStop, this
    // does NOT tell workers to exit  they keep running until the job queue
    // is drained.  waitForResult() uses it to know it's safe to return false
    // only after the job queue and completedJobs are both empty.
    std::atomic<bool> allJobsSubmitted{false};
    std::atomic<size_t> activeJobs{0};
    std::atomic<size_t> totalJobsProcessed{0};
    
    size_t numThreads;
    
    void workerThread() {
        while (true) {
            CompressJob job;
            {
                std::unique_lock<std::mutex> lock(jobMutex);
                jobCV.wait(lock, [this] {
                    return !jobQueue.empty() || shouldStop.load();
                });
                if (shouldStop.load() && jobQueue.empty()) break;
                if (jobQueue.empty()) continue;
                job = std::move(jobQueue.front());
                jobQueue.pop();
                activeJobs++;
            }
            
            CompressResult result;
            result.chunkIndex = job.chunkIndex;
            result.originalSize = job.inputData.size();
            result.originalData = job.inputData;
            result.compressedData.resize(job.maxOutputSize);
            
            int compSize = (job.hcLevel > 0)
                ? LZ4_compress_HC(
                    reinterpret_cast<const char*>(job.inputData.data()),
                    reinterpret_cast<char*>(result.compressedData.data()),
                    job.inputData.size(),
                    job.maxOutputSize,
                    job.hcLevel)
                : LZ4_compress_default(
                    reinterpret_cast<const char*>(job.inputData.data()),
                    reinterpret_cast<char*>(result.compressedData.data()),
                    job.inputData.size(),
                    job.maxOutputSize);

            if (compSize > 0 && (size_t)compSize < job.inputData.size()) {
                result.compressedData.resize(compSize);
                result.success = true;
            } else {
                result.compressedData.clear();
                result.success = false;
            }
            
            {
                size_t storedIdx = result.chunkIndex;
                std::lock_guard<std::mutex> lock(resultMutex);
                completedJobs[storedIdx] = std::move(result);
                VLOG(DEBUG, "CPU worker: stored chunk %zu (pool now has %zu results)\n",
                     storedIdx, completedJobs.size());
            }
            activeJobs--;
            totalJobsProcessed++;
            // Notify after decrementing activeJobs so waitForResult's
            // "all done" predicate sees the correct count.
            resultCV.notify_all();
        }
    }
    
public:
    CPUCompressionPool(size_t threads = 0) {
        if (threads == 0) {
            numThreads = std::thread::hardware_concurrency();
            if (numThreads == 0) numThreads = 4;
        } else {
            numThreads = threads;
        }
        VLOG(VERBOSE, "CPU compression pool: %zu threads\n", numThreads);
        for (size_t i = 0; i < numThreads; i++) {
            workers.emplace_back(&CPUCompressionPool::workerThread, this);
        }
    }
    
    ~CPUCompressionPool() { stop(); }
    
    void submitJob(size_t chunkIndex, std::vector<uint8_t> data, int hcLevel = 0) {
        CompressJob job;
        job.chunkIndex    = chunkIndex;
        job.inputData     = std::move(data);
        job.maxOutputSize = LZ4_compressBound(job.inputData.size());
        job.hcLevel       = hcLevel;
        {
            std::lock_guard<std::mutex> lock(jobMutex);
            jobQueue.push(std::move(job));
        }
        jobCV.notify_one();
    }
    
    bool getResult(size_t chunkIndex, CompressResult& result) {
        std::lock_guard<std::mutex> lock(resultMutex);
        auto it = completedJobs.find(chunkIndex);
        if (it != completedJobs.end()) {
            result = std::move(it->second);
            completedJobs.erase(it);
            return true;
        }
        return false;
    }
    
    // Blocking wait: sleeps until chunkIndex is in completedJobs, or all work
    // is provably done (all jobs submitted, queue drained, no active workers).
    // Returns false only when no more results will ever arrive.
    bool waitForResult(size_t chunkIndex, CompressResult& result) {
        std::unique_lock<std::mutex> lock(resultMutex);
        resultCV.wait(lock, [this, chunkIndex] {
            if (completedJobs.count(chunkIndex) > 0) return true;
            // No more results will arrive when: all jobs were submitted AND
            // no worker is actively compressing AND the job queue is empty.
            // activeJobs is atomic so no extra lock needed.  jobQueue.empty()
            // is read under resultMutex which is already held  safe because
            // workers pop from jobQueue under jobMutex alone, and we only need
            // a conservative "is it plausibly empty" check here; the worst
            // case is a spurious wake which re-evaluates the predicate.
            if (!allJobsSubmitted.load()) return false;
            if (activeJobs.load() > 0)    return false;
            std::lock_guard<std::mutex> jlk(jobMutex);
            return jobQueue.empty();
        });
        auto it = completedJobs.find(chunkIndex);
        if (it != completedJobs.end()) {
            result = std::move(it->second);
            completedJobs.erase(it);
            return true;
        }
        return false;
    }
    
    size_t getQueueDepth() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(jobMutex));
        return jobQueue.size();
    }
    size_t getActiveJobs()     const { return activeJobs.load(); }
    size_t getTotalProcessed() const { return totalJobsProcessed.load(); }
    size_t getCompletedCount() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(resultMutex));
        return completedJobs.size();
    }
    size_t getSmallestCompletedIndex() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(resultMutex));
        if (completedJobs.empty()) return SIZE_MAX;
        return completedJobs.begin()->first;
    }
    bool drainOne(CompressResult& result) {
        std::lock_guard<std::mutex> lock(resultMutex);
        if (completedJobs.empty()) return false;
        auto it = completedJobs.begin();
        result = std::move(it->second);
        completedJobs.erase(it);
        return true;
    }
    
    // finish(): called by the reader thread once all jobs are submitted.
    // Sets allJobsSubmitted so waitForResult() can detect when all work is
    // done without racing against workers still compressing.  Does NOT set
    // shouldStop  workers keep running until the job queue is empty.
    // stop()/destructor handles the final join after the drain loop exits.
    void finish() {
        allJobsSubmitted.store(true);
        resultCV.notify_all(); // wake waitForResult in case queue already empty
    }

    void stop() {
        shouldStop.store(true);
        jobCV.notify_all();
        resultCV.notify_all();
        for (auto& worker : workers) {
            if (worker.joinable()) worker.join();
        }
        workers.clear();
    }
};

/*
 * Main application class
 */
/*
 * NVML wrapper  loaded dynamically so the binary runs on systems without
 * libnvidia-ml.so.  Falls back gracefully (all utilization reported as 0).
 */
struct NVMLHandle {
    void* lib = nullptr;

    // Function pointers
    nvmlReturn_t (*Init)()                                            = nullptr;
    nvmlReturn_t (*Shutdown)()                                        = nullptr;
    nvmlReturn_t (*DeviceGetHandleByIndex)(unsigned, nvmlDevice_t*)  = nullptr;
    nvmlReturn_t (*DeviceGetUtilizationRates)(nvmlDevice_t,
                      nvmlUtilization_t*)                            = nullptr;
    nvmlReturn_t (*DeviceGetMemoryInfo)(nvmlDevice_t,
                      nvmlMemory_t*)                                  = nullptr;

    bool available = false;

    NVMLHandle() {
        lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!lib) lib = dlopen("libnvidia-ml.so", RTLD_LAZY | RTLD_LOCAL);
        if (!lib) return;

        #define LOAD(sym) sym = (decltype(sym))dlsym(lib, "nvml" #sym); \
                          if (!sym) { dlclose(lib); lib=nullptr; return; }
        LOAD(Init) LOAD(Shutdown) LOAD(DeviceGetHandleByIndex)
        LOAD(DeviceGetUtilizationRates) LOAD(DeviceGetMemoryInfo)
        #undef LOAD

        if (Init() != NVML_SUCCESS) { dlclose(lib); lib = nullptr; return; }
        available = true;
    }

    ~NVMLHandle() {
        if (available && Shutdown) Shutdown();
        if (lib) dlclose(lib);
    }
};

/*
 * GPU load monitor  runs a background thread that periodically queries NVML
 * for SM and memory utilization on each GPU.  Updates GPUDevice::smUtilPct,
 * memUtilPct, loadScore, streamPct, and batchPct atomically so the dispatcher
 * and GPU workers can read them without locking.
 *
 * loadScore = 0.7 * smUtil + 0.3 * memUtil  (weighted toward SM, which is
 * the primary bottleneck for LZ4 decompression and LLM training alike).
 *
 * Capacity fractions:
 *   loadScore 0-15:   idle     → streamPct=100, batchPct=100
 *   loadScore 15-40:  light    → streamPct=75,  batchPct=75
 *   loadScore 40-70:  moderate → streamPct=50,  batchPct=50
 *   loadScore 70-85:  heavy    → streamPct=25,  batchPct=25
 *   loadScore 85+:    overload → streamPct=0  (dispatcher skips this GPU)
 */
class GPULoadMonitor {
public:
    GPULoadMonitor() {}

    void start(std::vector<GPUDevice>& gpus, int intervalMs = 2000) {
        gpus_     = &gpus;
        interval_ = intervalMs;
        stop_.store(false);
        if (!nvml_.available) {
            VLOG(VERBOSE, "NVML not available  GPU load balancing disabled\n");
            return;
        }
        thread_ = std::thread([this]() { run(); });
        VLOG(VERBOSE, "GPU load monitor started (interval %d ms)\n", interval_);
    }

    void stop() {
        stop_.store(true);
        if (thread_.joinable()) thread_.join();
    }

    // Snapshot current load for all GPUs and log it.
    void logCurrentLoad() {
        if (!nvml_.available || !gpus_) return;
        for (size_t i = 0; i < gpus_->size(); i++) {
            GPUDevice& g = (*gpus_)[i];
            VLOG(VERBOSE, "  GPU%d: SM %2u%%  MEM %2u%%  load %2u%%  "
                 "streams %3u%%  batch %3u%%\n",
                 g.deviceId,
                 g.smUtilPct.load(), g.memUtilPct.load(),
                 g.loadScore.load(),
                 g.streamPct.load(), g.batchPct.load());
        }
    }

    ~GPULoadMonitor() { stop(); }

private:
    NVMLHandle              nvml_;
    std::vector<GPUDevice>* gpus_     = nullptr;
    int                     interval_ = 2000;
    std::atomic<bool>       stop_{false};
    std::thread             thread_;

    void run() {
        while (!stop_.load()) {
            poll();
            // Sleep in short increments so we can exit promptly on stop()
            for (int i = 0; i < interval_ / 50 && !stop_.load(); i++)
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    void poll() {
        if (!gpus_ || !nvml_.available) return;
        for (size_t i = 0; i < gpus_->size(); i++) {
            GPUDevice& g = (*gpus_)[i];
            nvmlDevice_t dev;
            if (nvml_.DeviceGetHandleByIndex((unsigned)g.deviceId, &dev)
                    != NVML_SUCCESS) continue;

            nvmlUtilization_t util{};
            if (nvml_.DeviceGetUtilizationRates(dev, &util) == NVML_SUCCESS) {
                g.smUtilPct.store(util.gpu,    std::memory_order_relaxed);
                g.memUtilPct.store(util.memory, std::memory_order_relaxed);
            }

            // Composite load score: 70% SM, 30% memory bandwidth
            uint32_t score = (uint32_t)(0.7 * g.smUtilPct.load()
                                      + 0.3 * g.memUtilPct.load());
            g.loadScore.store(score, std::memory_order_relaxed);

            // Capacity fractions
            uint32_t sp, bp;
            if      (score < 15) { sp = 100; bp = 100; }
            else if (score < 40) { sp = 75;  bp = 75;  }
            else if (score < 70) { sp = 50;  bp = 50;  }
            else if (score < 85) { sp = 25;  bp = 25;  }
            else                 { sp = 0;   bp = 0;   }

            g.streamPct.store(sp, std::memory_order_release);
            g.batchPct.store (bp, std::memory_order_release);
        }
    }
};

class GZL4Compressor {
private:
    std::vector<GPUDevice> gpus;
    size_t chunkSize;
    size_t batchSize;
    int compressionLevel;
    bool decompress;
    bool keepOriginal;
    bool forceOverwrite;
    bool forceMode;         // -z: force compression mode
    bool stdoutMode;
    bool testMode;
    bool listMode;
    std::vector<std::string> listFileArgs;
    std::vector<std::string> extraInputFiles;
    bool storeContentSize;
    bool contentSizeExplicit;
    bool syncOutput;        // --sync-output: call fsync() before closing output file

    static void signalHandler(int signum) {
        (void)signum;
        if (g_instance && !g_instance->tempOutputFile.empty()) {
            fprintf(stderr, "\nInterrupted - cleaning up temporary file...\n");
            unlink(g_instance->tempOutputFile.c_str());
        }
        exit(EXIT_FAILURE);
    }
    
    void setupSignalHandlers() {
        g_instance = this;
        signal(SIGINT, signalHandler);
        signal(SIGTERM, signalHandler);
    }
    
    void cleanupTempFile() {
        if (!tempOutputFile.empty() && !stdoutMode) {
            unlink(tempOutputFile.c_str());
            tempOutputFile.clear();
        }
    }
    
    bool renameTempToFinal() {
        if (tempOutputFile.empty() || stdoutMode) return true;
        VLOG(VERBOSE, "Renaming: %s -> %s\n",
             tempOutputFile.c_str(), outputFile.c_str());
        if (rename(tempOutputFile.c_str(), outputFile.c_str()) != 0) {
            fprintf(stderr, "Error: Failed to rename %s to %s: %s\n",
                    tempOutputFile.c_str(), outputFile.c_str(), strerror(errno));
            cleanupTempFile();
            return false;
        }
        tempOutputFile.clear();
        return true;
    }
    
    const char* getActualOutputPath() const {
        return tempOutputFile.empty() ? outputFile.c_str() : tempOutputFile.c_str();
    }
    int  hcLevel;
    BackendMode backendMode;
    size_t cpuThreads;
    std::string inputFile;
    std::string outputFile;
    std::string tempOutputFile;

    static GZL4Compressor* g_instance;
    
    size_t slotCapacity;
    size_t pipelineDepth;
    bool   disableEarlyRead;
    bool   forceProgress;
    bool   earlyExit;

    AsyncReader      earlyReader;
    bool             earlyReaderStarted = false;
    PreDecompReader  preDecompReader;
    bool             preDecompReaderStarted = false;
    GPULoadMonitor   loadMonitor;
    PinnedInputPool  inputPool;
    
public:
    GZL4Compressor() 
        : chunkSize(CHUNK_SIZE_LEVEL_9)
        , batchSize(1)
        , compressionLevel(9)
        , decompress(false)
        , keepOriginal(false)
        , forceOverwrite(false)
        , forceMode(false)
        , stdoutMode(false)
        , testMode(false)
        , listMode(false)
        , storeContentSize(true)
        , contentSizeExplicit(false)
        , syncOutput(false)
        , hcLevel(0)
        , backendMode(BackendMode::HYBRID)
        , cpuThreads(CPU_THREADS_AUTO)
        , slotCapacity(8)
        , pipelineDepth(0)
        , disableEarlyRead(false)
        , forceProgress(false)
        , earlyExit(false)
    {}
    
    ~GZL4Compressor() {}
    
    /*
     * Initialize and enumerate all available CUDA GPUs
     */
    // Fast GPU enumeration for decompression: populates gpus[] using only
    // cudaGetDeviceCount + cudaGetDeviceProperties.  No cudaSetDevice, no
    // context creation  takes <1ms regardless of GPU count.
    // GPU workers do their own cudaSetDevice + cudaMemGetInfo to get accurate
    // free VRAM for slot sizing.  We use totalGlobalMem as a conservative
    // proxy for N_STREAMS_H sizing here.
    bool enumerateGPUs() {
        auto t0 = std::chrono::high_resolution_clock::now();
        // Force CUDA driver initialization  without this, cudaGetDeviceCount
        // can return 0 or cudaErrorNoDevice in some environments even when
        // GPUs are present.  cudaFree(0) is the lightest possible call that
        // triggers driver init without allocating anything.
        cudaFree(0);

        int deviceCount = 0;
        if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
            fprintf(stderr, "Error: No CUDA-capable GPUs found\n");
            return false;
        }
        VLOG(VERBOSE, "Found %d CUDA-capable GPU(s)\n", deviceCount);

        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp props;
            if (cudaGetDeviceProperties(&props, i) != cudaSuccess) continue;
            int cc = props.major * 10 + props.minor;
            if (cc < 35) continue;

            GPUDevice gpu(i);
            gpu.properties      = props;
            // Use 80% of total VRAM as conservative free estimate  GPU workers
            // query cudaMemGetInfo themselves for accurate slot sizing.
            gpu.availableMemory = static_cast<size_t>(props.totalGlobalMem * 0.80);
            gpu.totalMemory     = props.totalGlobalMem;
            gpu.smCount         = props.multiProcessorCount;
            gpu.pipelineDepth   = (pipelineDepth > 0) ? pipelineDepth : 1;
            VLOG(VERBOSE, "GPU%d: %s  %.1f GB VRAM  %d SMs\n",
                 i, props.name, props.totalGlobalMem / (1024.0*1024.0*1024.0), (int)gpu.smCount);
            gpus.push_back(std::move(gpu));
        }

        if (gpus.empty()) {
            fprintf(stderr, "Error: No suitable GPUs found\n");
            return false;
        }

        // Auto-tune slotCapacity from VRAM with latency cap.
        if (slotCapacity == 8) {
            const size_t memPerChunk = chunkSize * 5;
            size_t minBatch = SIZE_MAX;
            for (auto& gpu : gpus) {
                size_t perSlot = static_cast<size_t>(gpu.availableMemory * 0.85) / 3;
                size_t vramBatch = perSlot / std::max(memPerChunk, size_t(1));
                vramBatch = std::max(size_t(4), std::min(size_t(8192), vramBatch));
                minBatch = std::min(minBatch, vramBatch);
            }
            if (minBatch == SIZE_MAX) minBatch = 128;
            const size_t latencyCap = std::max(size_t(4),
                size_t(512ULL * 1024 * 1024) / chunkSize);
            slotCapacity = std::min(minBatch, latencyCap);
        }

        double ms = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count() * 1000.0;
        VLOG(VERBOSE, "Enumerated %zu GPU(s) in %.1f ms\n", gpus.size(), ms);

        // Start load monitor  polls NVML every 2s to track GPU utilization.
        // This lets the dispatcher route work away from GPUs already busy
        // with other tasks (e.g. LLM training).
        loadMonitor.start(gpus);
        return true;
    }

    bool initializeGPUs() {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto elapsed = [&]() {
            return std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t0).count() * 1000.0;
        };

        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error: Failed to get CUDA device count: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        if (deviceCount == 0) {
            fprintf(stderr, "Error: No CUDA-capable GPUs found\n");
            return false;
        }
        VLOG(VERBOSE, "Found %d CUDA-capable GPU(s)\n", deviceCount);
        VLOG(DEBUG,   "  [%.1f ms] cudaGetDeviceCount\n", elapsed());

        // ── Phase 1: enumerate GPUs (fast  no context creation) ────────────
        // cudaGetDeviceProperties does NOT create a CUDA context. Only
        // cudaSetDevice + the first alloc/stream-create on that device does.
        struct GPUInfo {
            int            id;
            cudaDeviceProp props;
            size_t         freeMem, totalMem;
            bool           valid = false;
        };
        std::vector<GPUInfo> infos(deviceCount);
        for (int i = 0; i < deviceCount; i++) {
            infos[i].id = i;
            if (cudaGetDeviceProperties(&infos[i].props, i) != cudaSuccess) {
                VLOG(VERBOSE, "Skipping GPU %d - cudaGetDeviceProperties failed\n", i);
                continue;
            }
            int cc = infos[i].props.major * 10 + infos[i].props.minor;
            if (cc < 35) {
                VLOG(VERBOSE, "Skipping GPU %d (%s) - compute capability %d.%d < 3.5\n",
                     i, infos[i].props.name, infos[i].props.major, infos[i].props.minor);
                continue;
            }
            infos[i].valid = true;
        }
        VLOG(DEBUG, "  [%.1f ms] device property enumeration done\n", elapsed());

        // ── Phase 2: per-GPU context init in parallel threads ────────────────
        // cudaSetDevice on a new device creates the CUDA context for that
        // device.  With 8 GPUs this used to take 8× the single-GPU init time
        // when done serially.  Parallel init overlaps all context creations.
        std::vector<std::unique_ptr<GPUDevice>> candidates(deviceCount);
        std::vector<std::thread> initThreads;
        std::mutex               initMutex;  // guards VLOG output ordering

        for (int i = 0; i < deviceCount; i++) {
            if (!infos[i].valid) continue;
            initThreads.emplace_back([&, i]() {
                auto ti = std::chrono::high_resolution_clock::now();

                GPUDevice gpu(i);
                cudaError_t e = cudaSetDevice(i);  // ← context creation happens here
                if (e != cudaSuccess) {
                    std::lock_guard<std::mutex> lk(initMutex);
                    VLOG(VERBOSE, "GPU%d: cudaSetDevice failed: %s\n",
                         i, cudaGetErrorString(e));
                    return;
                }

                double ctxMs = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - ti).count() * 1000.0;
                VLOG(DEBUG, "  GPU%d: context created in %.1f ms\n", i, ctxMs);

                size_t freeMem, totalMem;
                if (cudaMemGetInfo(&freeMem, &totalMem) != cudaSuccess) {
                    VLOG(VERBOSE, "GPU%d: cudaMemGetInfo failed\n", i);
                    return;
                }

                gpu.availableMemory = static_cast<size_t>(freeMem * GPU_MEM_SAFETY_FACTOR);
                gpu.totalMemory     = totalMem;
                gpu.smCount         = infos[i].props.multiProcessorCount;
                gpu.properties      = infos[i].props;
                gpu.pipelineDepth   = (pipelineDepth > 0) ? pipelineDepth : 1;

                gpu.streams.resize(gpu.pipelineDepth);
                for (int s = 0; s < gpu.pipelineDepth; s++) {
                    if (cudaStreamCreate(&gpu.streams[s]) != cudaSuccess) {
                        for (int c = 0; c < s; c++) cudaStreamDestroy(gpu.streams[c]);
                        VLOG(VERBOSE, "GPU%d: stream creation failed\n", i);
                        return;
                    }
                }

                double totalMs = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - ti).count() * 1000.0;
                {
                    std::lock_guard<std::mutex> lk(initMutex);
                    VLOG(VERBOSE, "GPU%d: %s  %.1f GB VRAM  %d SMs  %d copy engine%s  "
                         "init %.1f ms\n",
                         i, gpu.properties.name,
                         totalMem / (1024.0*1024.0*1024.0),
                         (int)gpu.smCount,
                         gpu.properties.asyncEngineCount,
                         gpu.properties.asyncEngineCount == 1 ? "" : "s",
                         totalMs);
                    candidates[i] = std::make_unique<GPUDevice>(std::move(gpu));
                }
            });
        }

        VLOG(DEBUG, "  [%.1f ms] launched %zu GPU init threads\n",
             elapsed(), initThreads.size());

        for (auto& t : initThreads) t.join();

        VLOG(DEBUG, "  [%.1f ms] all GPU init threads joined\n", elapsed());

        // ── Phase 3: collect valid GPUs ──────────────────────────────────────
        for (int i = 0; i < deviceCount; i++) {
            if (candidates[i]) gpus.push_back(std::move(*candidates[i]));
        }

        // ── Phase 4: auto-tune pipeline depth and slot capacity ──────────────
        if (pipelineDepth <= 0 && !gpus.empty()) {
            // H100/A100 have 3 copy engines (H2D, D2H, P2P); 34 streams
            // lets H2D copy, kernel, and D2H overlap fully.  GPU count has
            // no bearing on per-GPU stream depth.
            int autoDepth = 3;
            for (auto& gpu : gpus) {
                for (auto& s : gpu.streams) cudaStreamDestroy(s);
                gpu.pipelineDepth = autoDepth;
                gpu.streams.resize(autoDepth);
                for (int s = 0; s < autoDepth; s++) cudaStreamCreate(&gpu.streams[s]);
                VLOG(DEBUG, "GPU%d: auto-tuned pipeline depth to %d\n",
                     gpu.deviceId, autoDepth);
            }
        }

        if (slotCapacity == 8 && !gpus.empty()) {
            // VRAM ceiling: don't over-allocate device memory.
            const size_t memPerChunk = chunkSize * 5;
            size_t minBatch = SIZE_MAX;
            for (auto& gpu : gpus) {
                size_t usable = static_cast<size_t>(gpu.availableMemory * 0.85);
                size_t perSlot = usable / std::max(1, gpu.pipelineDepth);
                size_t vramBatch = perSlot / memPerChunk;
                vramBatch = std::max(size_t(4), std::min(size_t(8192), vramBatch));
                minBatch = std::min(minBatch, vramBatch);
            }
            if (minBatch == SIZE_MAX) minBatch = 128;

            // Latency cap: ~512 MB input per batch keeps GPU startup fast.
            // At 4MB chunks = 128, at 256KB chunks = 2048.
            const size_t latencyCap = std::max(size_t(4),
                size_t(512ULL * 1024 * 1024) / chunkSize);
            slotCapacity = std::min(minBatch, latencyCap);

            VLOG(DEBUG, "Auto-tuned batch size to %zu "
                 "(VRAM ceil %zu, latency cap %zu, %.1f MB chunks, %zu GPU(s))\n",
                 slotCapacity, minBatch, latencyCap,
                 chunkSize / (1024.0*1024.0), gpus.size());
        }

        if (gpus.empty()) {
            fprintf(stderr, "Error: No suitable GPUs found\n");
            return false;
        }

        VLOG(VERBOSE, "Initialized %zu GPU(s) in %.1f ms\n", gpus.size(), elapsed());
        VLOG(DEBUG,   "  [%.1f ms] initializeGPUs complete\n", elapsed());
        return true;
    }

    /*
     * Calculate optimal chunk size based on compression level
     */
    void setChunkSizeFromLevel() {
        if (compressionLevel >= 10) {
            chunkSize = CHUNK_SIZE_LEVEL_9;
            static const int hcMap[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8, 12};
            if (hcLevel == 0)
                hcLevel = hcMap[std::min(compressionLevel, 12)];
        } else {
            switch (compressionLevel) {
                case 1: chunkSize = CHUNK_SIZE_LEVEL_1; break;
                case 2: chunkSize = CHUNK_SIZE_LEVEL_2; break;
                case 3: chunkSize = CHUNK_SIZE_LEVEL_3; break;
                case 4: chunkSize = CHUNK_SIZE_LEVEL_4; break;
                case 5: chunkSize = CHUNK_SIZE_LEVEL_5; break;
                case 6: chunkSize = CHUNK_SIZE_LEVEL_6; break;
                case 7: chunkSize = CHUNK_SIZE_LEVEL_7; break;
                case 8: chunkSize = CHUNK_SIZE_LEVEL_8; break;
                case 9:
                default: chunkSize = CHUNK_SIZE_LEVEL_9; break;
            }
        }

        if (hcLevel > 0) {
            VLOG(VERBOSE, "Compression level %d: LZ4 HC level %d, chunk size %.2f KB\n",
                 compressionLevel, hcLevel, chunkSize / 1024.0);
        } else if (!decompress) {
            VLOG(VERBOSE, "Compression level %d: LZ4 fast, chunk size %.2f KB\n",
                 compressionLevel, chunkSize / 1024.0);
        }
    }

    /*
     * Calculate optimal batch size based on GPU memory
     */
    size_t calculateBatchSize(size_t gpuMemory, size_t numChunks = 0) {
        // VRAM-driven ceiling: each chunk needs ~5× chunkSize device memory.
        const size_t memPerChunk = chunkSize * 5;
        const size_t usable      = static_cast<size_t>(gpuMemory * 0.90);
        size_t batchSize         = usable / memPerChunk;

        // Latency cap: limit how much data must be H2D-transferred before
        // the first kernel fires.  Targeting ~512 MB input per batch keeps
        // GPU startup fast regardless of chunk size.  At 4MB chunks = 128,
        // at 256KB chunks = 2048  scales naturally with compression level.
        const size_t targetInputBytes = size_t(512) * 1024 * 1024;  // 512 MB
        size_t latencyCap = std::max(size_t(4), targetInputBytes / chunkSize);
        if (latencyCap < batchSize)
            batchSize = latencyCap;

        // File-size cap: no point allocating slots for more work than exists.
        if (numChunks > 0 && !gpus.empty()) {
            size_t totalSlots = gpus.size() * std::max(1, gpus[0].pipelineDepth);
            size_t targetBatch = (numChunks + totalSlots * 8 - 1) / (totalSlots * 8);
            if (targetBatch < batchSize)
                batchSize = std::max(targetBatch, size_t(4));
        }

        batchSize = std::max(size_t(4), std::min(size_t(8192), batchSize));
        VLOG(DEBUG, "Init-time batch estimate: %zu chunks/slot "
             "(%.1f GB/slot, %.1f MB chunks%s)\n",
             batchSize,
             (batchSize * memPerChunk) / (1024.0*1024.0*1024.0),
             chunkSize / (1024.0*1024.0),
             numChunks > 0 ? ", file-size capped" : "");
        return batchSize;
    }

    size_t refreshGPUMemoryAndBatchSize(GPUDevice& gpu) {
        cudaSetDevice(gpu.deviceId);

        size_t freeMem, totalMem;
        if (cudaMemGetInfo(&freeMem, &totalMem) != cudaSuccess) return 0;
        gpu.availableMemory = freeMem;

        const size_t memPerChunk    = chunkSize * 5;
        const int    slots          = std::max(1, gpu.pipelineDepth);
        size_t perSlotMem = static_cast<size_t>(freeMem * 0.90) / slots;

        if (perSlotMem < memPerChunk) {
            VLOG(DEBUG, "GPU%d: %.1f GB free / %d slots -> only %.0f MB/slot, need %.0f MB\n",
                 gpu.deviceId, freeMem/(1024.0*1024.0*1024.0), slots,
                 perSlotMem/(1024.0*1024.0), memPerChunk/(1024.0*1024.0));
            return 0;
        }

        size_t vramBatch = perSlotMem / memPerChunk;
        // Clamp to sensible range  no SM floor since chunk size already
        // determines minimum useful parallelism better than smCount heuristic.
        size_t batch = std::max(size_t(4), std::min(vramBatch, size_t(8192)));
        gpu.optimalBatch = batch;

        VLOG(DEBUG, "GPU%d: %.1f GB free / %d slots -> %zu chunks/slot "
             "(VRAM ceiling %zu, %.1f MB chunks)\n",
             gpu.deviceId, freeMem/(1024.0*1024.0*1024.0),
             slots, batch, vramBatch, chunkSize/(1024.0*1024.0));

        return batch;
    }

    /*
     * Launch batch compression asynchronously
     */
    bool compressBatchAsync(const std::vector<std::vector<uint8_t>>& inputs,
                           BatchCompressState& state,
                           GPUDevice& gpu,
                           cudaStream_t stream = 0) {
        
        if (inputs.empty()) {
            fprintf(stderr, "Error: Cannot compress empty batch\n");
            return false;
        }

        state.batch_size = inputs.size();
        state.input_sizes.resize(state.batch_size);
        state.max_output_sizes.resize(state.batch_size);

        VLOG(DEBUG, "Launching async batch compression of %zu chunks on GPU %d stream %p\n",
             state.batch_size, gpu.deviceId, (void*)stream);

        cudaError_t err = cudaSetDevice(gpu.deviceId);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: Failed to set device %d - %s\n",
                    gpu.deviceId, cudaGetErrorString(err));
            return false;
        }

        cudaGetLastError();

        state.d_inputs.resize(state.batch_size);
        std::vector<const void*> h_input_ptrs(state.batch_size);
        std::vector<size_t>      h_input_sizes(state.batch_size);

        size_t max_chunk_size   = 0;
        size_t total_input_size = 0;

        for (size_t i = 0; i < state.batch_size; i++) {
            size_t size = inputs[i].size();
            state.input_sizes[i] = size;
            h_input_sizes[i]     = size;
            max_chunk_size   = std::max(max_chunk_size, size);
            total_input_size += size;

            CUDA_CHECK_MSG(cudaMalloc(&state.d_inputs[i], size),
                           "Failed to allocate input buffer");
            CUDA_CHECK_MSG(cudaMemcpyAsync(state.d_inputs[i], inputs[i].data(), size,
                                           cudaMemcpyHostToDevice, stream),
                           "Failed to async copy input to device");
            h_input_ptrs[i] = state.d_inputs[i];
        }

        size_t ptr_bytes  = state.batch_size * sizeof(void*);
        size_t size_bytes = state.batch_size * sizeof(size_t);

        CUDA_CHECK_MSG(cudaHostAlloc(&state.h_input_ptrs_pinned,  ptr_bytes,  cudaHostAllocDefault),
                       "Failed to alloc pinned input ptrs");
        CUDA_CHECK_MSG(cudaHostAlloc(&state.h_input_sizes_pinned, size_bytes, cudaHostAllocDefault),
                       "Failed to alloc pinned input sizes");

        memcpy(state.h_input_ptrs_pinned,  h_input_ptrs.data(),  ptr_bytes);
        memcpy(state.h_input_sizes_pinned, h_input_sizes.data(), size_bytes);

        CUDA_CHECK_MSG(cudaMalloc(&state.d_input_ptrs,  ptr_bytes),
                       "Failed to allocate input pointers array");
        CUDA_CHECK_MSG(cudaMalloc(&state.d_input_sizes, size_bytes),
                       "Failed to allocate input sizes array");

        CUDA_CHECK_MSG(cudaMemcpyAsync(state.d_input_ptrs,  state.h_input_ptrs_pinned,  ptr_bytes,
                                       cudaMemcpyHostToDevice, stream),
                       "Failed to async copy input pointers");
        CUDA_CHECK_MSG(cudaMemcpyAsync(state.d_input_sizes, state.h_input_sizes_pinned, size_bytes,
                                       cudaMemcpyHostToDevice, stream),
                       "Failed to async copy input sizes");

        nvcompBatchedLZ4CompressOpts_t opts = nvcompBatchedLZ4CompressDefaultOpts;
        size_t temp_bytes = 0;
        NVCOMP_CHECK(nvcompBatchedLZ4CompressGetTempSizeSync(
            state.d_input_ptrs, state.d_input_sizes,
            state.batch_size, max_chunk_size, opts,
            &temp_bytes, total_input_size, stream));

        VLOG(DEBUG, "  Temp buffer size: %.2f MB\n", temp_bytes / (1024.0*1024.0));
        CUDA_CHECK_MSG(cudaMalloc(&state.d_temp, temp_bytes),
                       "Failed to allocate temp buffer");

        state.d_outputs.resize(state.batch_size);
        std::vector<void*> h_output_ptrs(state.batch_size);

        for (size_t i = 0; i < state.batch_size; i++) {
            size_t max_output_size = 0;
            NVCOMP_CHECK(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
                state.input_sizes[i], opts, &max_output_size));
            state.max_output_sizes[i] = max_output_size;
            CUDA_CHECK_MSG(cudaMalloc(&state.d_outputs[i], max_output_size),
                           "Failed to allocate output buffer");
            h_output_ptrs[i] = state.d_outputs[i];
        }

        CUDA_CHECK_MSG(cudaMalloc(&state.d_output_ptrs,  state.batch_size * sizeof(void*)),
                       "Failed to allocate output pointers array");
        CUDA_CHECK_MSG(cudaMalloc(&state.d_output_sizes, state.batch_size * sizeof(size_t)),
                       "Failed to allocate output sizes array");
        CUDA_CHECK_MSG(cudaMalloc(&state.d_statuses,     state.batch_size * sizeof(nvcompStatus_t)),
                       "Failed to allocate statuses array");

        CUDA_CHECK_MSG(cudaHostAlloc(&state.h_output_ptrs_pinned, state.batch_size * sizeof(void*),
                                     cudaHostAllocDefault),
                       "Failed to alloc pinned output ptrs");
        memcpy(state.h_output_ptrs_pinned, h_output_ptrs.data(), state.batch_size * sizeof(void*));

        CUDA_CHECK_MSG(cudaMemcpyAsync(state.d_output_ptrs, state.h_output_ptrs_pinned,
                                       state.batch_size * sizeof(void*),
                                       cudaMemcpyHostToDevice, stream),
                       "Failed to async copy output pointers");

        VLOG(DEBUG, "  Launching nvcompBatchedLZ4CompressAsync batch_size=%zu\n", state.batch_size);
        NVCOMP_CHECK(nvcompBatchedLZ4CompressAsync(
            state.d_input_ptrs,
            state.d_input_sizes,
            max_chunk_size,
            state.batch_size,
            state.d_temp,
            temp_bytes,
            state.d_output_ptrs,
            state.d_output_sizes,
            opts,
            state.d_statuses,
            stream));

        return true;
    }
    
    /*
     * Get batch compression results
     */
    bool getBatchCompressResults(BatchCompressState& state,
                                std::vector<std::vector<uint8_t>>& outputs,
                                GPUDevice& gpu,
                                cudaStream_t stream = 0) {

        cudaSetDevice(gpu.deviceId);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<nvcompStatus_t> statuses(state.batch_size);
        CUDA_CHECK_MSG(cudaMemcpy(statuses.data(), state.d_statuses,
                                  state.batch_size * sizeof(nvcompStatus_t),
                                  cudaMemcpyDeviceToHost),
                       "Failed to copy statuses");

        auto cleanup = [&]() {
            for (auto ptr : state.d_inputs)  cudaFree(ptr);
            for (auto ptr : state.d_outputs) cudaFree(ptr);
            cudaFree(state.d_input_ptrs);
            cudaFree(state.d_input_sizes);
            cudaFree(state.d_temp);
            cudaFree(state.d_output_ptrs);
            cudaFree(state.d_output_sizes);
            cudaFree(state.d_statuses);
            if (state.h_input_ptrs_pinned)  cudaFreeHost(state.h_input_ptrs_pinned);
            if (state.h_input_sizes_pinned) cudaFreeHost(state.h_input_sizes_pinned);
            if (state.h_output_ptrs_pinned) cudaFreeHost(state.h_output_ptrs_pinned);
            state.h_input_ptrs_pinned = state.h_input_sizes_pinned =
                state.h_output_ptrs_pinned = nullptr;
        };

        for (size_t i = 0; i < state.batch_size; i++) {
            if (statuses[i] != nvcompSuccess) {
                fprintf(stderr, "Compression failed for chunk %zu with status %d\n",
                        i, static_cast<int>(statuses[i]));
                cleanup();
                return false;
            }
        }

        std::vector<size_t> output_sizes(state.batch_size);
        CUDA_CHECK_MSG(cudaMemcpy(output_sizes.data(), state.d_output_sizes,
                                  state.batch_size * sizeof(size_t),
                                  cudaMemcpyDeviceToHost),
                       "Failed to copy output sizes");

        outputs.resize(state.batch_size);
        for (size_t i = 0; i < state.batch_size; i++) {
            outputs[i].resize(output_sizes[i]);
            CUDA_CHECK_MSG(cudaMemcpy(outputs[i].data(), state.d_outputs[i],
                                      output_sizes[i], cudaMemcpyDeviceToHost),
                           "Failed to copy output data");
            VLOG(DEBUG, "  Chunk %zu: %zu -> %zu bytes (%.1f%%)\n",
                 i, state.input_sizes[i], output_sizes[i],
                 100.0 * output_sizes[i] / state.input_sizes[i]);
        }

        cleanup();
        return true;
    }
    
    /*
     * Launch batch decompression asynchronously
     */
    bool decompressBatchAsync(const std::vector<std::vector<uint8_t>>& inputs,
                             const std::vector<size_t>& uncompressed_sizes,
                             BatchDecompressState& state,
                             GPUDevice& gpu,
                             cudaStream_t stream = 0) {
        
        if (inputs.empty() || inputs.size() != uncompressed_sizes.size()) {
            fprintf(stderr, "Error: Invalid batch decompression parameters\n");
            return false;
        }
        
        state.batch_size = inputs.size();
        state.uncompressed_sizes = uncompressed_sizes;
        
        VLOG(DEBUG, "Launching async batch decompression of %zu chunks on GPU %d\n",
             state.batch_size, gpu.deviceId);

        cudaError_t err = cudaSetDevice(gpu.deviceId);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: Failed to set device %d - %s\n",
                    gpu.deviceId, cudaGetErrorString(err));
            return false;
        }
        
        cudaGetLastError();
        
        state.d_inputs.resize(state.batch_size);
        std::vector<const void*> h_input_ptrs(state.batch_size);
        std::vector<size_t> h_input_sizes(state.batch_size);
        std::vector<size_t> h_output_sizes(state.batch_size);
        
        for (size_t i = 0; i < state.batch_size; i++) {
            size_t size = inputs[i].size();
            h_input_sizes[i] = size;
            h_output_sizes[i] = uncompressed_sizes[i];
            
            CUDA_CHECK_MSG(cudaMalloc(&state.d_inputs[i], size),
                          "Failed to allocate input buffer");
            CUDA_CHECK_MSG(cudaMemcpy(state.d_inputs[i], inputs[i].data(), size,
                                     cudaMemcpyHostToDevice),
                          "Failed to copy input to device");
            
            h_input_ptrs[i] = state.d_inputs[i];
        }
        
        CUDA_CHECK_MSG(cudaMalloc(&state.d_input_ptrs, state.batch_size * sizeof(void*)),
                      "Failed to allocate input pointers array");
        CUDA_CHECK_MSG(cudaMalloc(&state.d_input_sizes, state.batch_size * sizeof(size_t)),
                      "Failed to allocate input sizes array");
        CUDA_CHECK_MSG(cudaMalloc(&state.d_output_sizes, state.batch_size * sizeof(size_t)),
                      "Failed to allocate output sizes array");
        
        CUDA_CHECK_MSG(cudaMemcpy(state.d_input_ptrs, h_input_ptrs.data(),
                                 state.batch_size * sizeof(void*), cudaMemcpyHostToDevice),
                      "Failed to copy input pointers");
        CUDA_CHECK_MSG(cudaMemcpy(state.d_input_sizes, h_input_sizes.data(),
                                 state.batch_size * sizeof(size_t), cudaMemcpyHostToDevice),
                      "Failed to copy input sizes");
        CUDA_CHECK_MSG(cudaMemcpy(state.d_output_sizes, h_output_sizes.data(),
                                 state.batch_size * sizeof(size_t), cudaMemcpyHostToDevice),
                      "Failed to copy output sizes");
        
        size_t max_uncompressed = *std::max_element(uncompressed_sizes.begin(),
                                                     uncompressed_sizes.end());
        size_t total_uncompressed = 0;
        for (size_t size : uncompressed_sizes) {
            total_uncompressed += size;
        }
        
        size_t temp_bytes = 0;
        nvcompBatchedLZ4DecompressOpts_t decomp_opts = nvcompBatchedLZ4DecompressDefaultOpts;
        
        CUDA_CHECK_MSG(cudaMalloc(&state.d_statuses, state.batch_size * sizeof(nvcompStatus_t)),
                      "Failed to allocate statuses array");
        
        NVCOMP_CHECK(nvcompBatchedLZ4DecompressGetTempSizeSync(
            state.d_input_ptrs,
            state.d_input_sizes,
            state.batch_size,
            max_uncompressed,
            &temp_bytes,
            total_uncompressed,
            decomp_opts,
            state.d_statuses,
            stream
        ));
        
        VLOG(DEBUG, "  Temp buffer size: %.2f MB\n", temp_bytes / (1024.0 * 1024.0));

        CUDA_CHECK_MSG(cudaMalloc(&state.d_temp, temp_bytes),
                      "Failed to allocate temp buffer");
        
        state.d_outputs.resize(state.batch_size);
        std::vector<void*> h_output_ptrs(state.batch_size);
        
        for (size_t i = 0; i < state.batch_size; i++) {
            CUDA_CHECK_MSG(cudaMalloc(&state.d_outputs[i], uncompressed_sizes[i]),
                          "Failed to allocate output buffer");
            h_output_ptrs[i] = state.d_outputs[i];
        }
        
        CUDA_CHECK_MSG(cudaMalloc(&state.d_output_ptrs, state.batch_size * sizeof(void*)),
                      "Failed to allocate output pointers array");
        CUDA_CHECK_MSG(cudaMalloc(&state.d_actual_output_sizes, state.batch_size * sizeof(size_t)),
                      "Failed to allocate actual output sizes array");
        
        CUDA_CHECK_MSG(cudaMemcpy(state.d_output_ptrs, h_output_ptrs.data(),
                                 state.batch_size * sizeof(void*), cudaMemcpyHostToDevice),
                      "Failed to copy output pointers");
        
        VLOG(DEBUG, "  Launching decompression with batch_size=%zu\n", state.batch_size);
        NVCOMP_CHECK(nvcompBatchedLZ4DecompressAsync(
            state.d_input_ptrs,
            state.d_input_sizes,
            state.d_output_sizes,
            state.d_actual_output_sizes,
            state.batch_size,
            state.d_temp,
            temp_bytes,
            state.d_output_ptrs,
            decomp_opts,
            state.d_statuses,
            stream
        ));
        
        return true;
    }
    
    /*
     * Get batch decompression results
     */
    bool getBatchDecompressResults(BatchDecompressState& state,
                                  std::vector<std::vector<uint8_t>>& outputs,
                                  GPUDevice& gpu) {
        
        cudaSetDevice(gpu.deviceId);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        std::vector<nvcompStatus_t> statuses(state.batch_size);
        CUDA_CHECK_MSG(cudaMemcpy(statuses.data(), state.d_statuses,
                                 state.batch_size * sizeof(nvcompStatus_t),
                                 cudaMemcpyDeviceToHost),
                      "Failed to copy statuses");
        
        for (size_t i = 0; i < state.batch_size; i++) {
            if (statuses[i] != nvcompSuccess) {
                fprintf(stderr, "Decompression failed for chunk %zu with status %d\n",
                        i, static_cast<int>(statuses[i]));
                for (auto ptr : state.d_inputs) cudaFree(ptr);
                for (auto ptr : state.d_outputs) cudaFree(ptr);
                cudaFree(state.d_input_ptrs);
                cudaFree(state.d_input_sizes);
                cudaFree(state.d_output_sizes);
                cudaFree(state.d_temp);
                cudaFree(state.d_output_ptrs);
                cudaFree(state.d_actual_output_sizes);
                cudaFree(state.d_statuses);
                return false;
            }
        }
        
        std::vector<size_t> actualSizes(state.batch_size);
        CUDA_CHECK_MSG(cudaMemcpy(actualSizes.data(), state.d_actual_output_sizes,
                                 state.batch_size * sizeof(size_t),
                                 cudaMemcpyDeviceToHost),
                      "Failed to copy actual output sizes");
        
        outputs.resize(state.batch_size);
        for (size_t i = 0; i < state.batch_size; i++) {
            size_t actualSize = actualSizes[i];
            outputs[i].resize(actualSize);
            CUDA_CHECK_MSG(cudaMemcpy(outputs[i].data(), state.d_outputs[i],
                                     actualSize, cudaMemcpyDeviceToHost),
                          "Failed to copy output data");
            
            VLOG(DEBUG, "  Chunk %zu decompressed to %zu bytes (estimated %zu)\n",
                 i, actualSize, state.uncompressed_sizes[i]);
        }
        
        for (auto ptr : state.d_inputs) cudaFree(ptr);
        for (auto ptr : state.d_outputs) cudaFree(ptr);
        cudaFree(state.d_input_ptrs);
        cudaFree(state.d_input_sizes);
        cudaFree(state.d_output_sizes);
        cudaFree(state.d_temp);
        cudaFree(state.d_output_ptrs);
        cudaFree(state.d_actual_output_sizes);
        cudaFree(state.d_statuses);
        
        return true;
    }
    
    /*
     * Compress a file using CPU-only multi-threaded compression
     */
    bool compressFileCPU() {
        VLOG(NORMAL, "%sCompressing%s (CPU-only): %s -> %s\n",
                CC_BCYAN, CC_RESET,
                inputFile.c_str(), outputFile.c_str());

        double timeCompressing = 0;
        
        size_t fileSize = 0;
        const bool stdinMode = (inputFile == "-");
        if (!stdinMode) {
            struct stat st;
            if (stat(inputFile.c_str(), &st) != 0) {
                fprintf(stderr, "Error: Cannot stat input file: %s\n", inputFile.c_str());
                return false;
            }
            fileSize = st.st_size;
            VLOG(VERBOSE, "Input file size: %.2f MB\n", fileSize / (1024.0 * 1024.0));
        } else {
            VLOG(VERBOSE, "Input: stdin (size unknown)\n");
        }

        size_t numChunks = fileSize > 0
            ? (fileSize + chunkSize - 1) / chunkSize
            : SIZE_MAX;
        if (!stdinMode)
            VLOG(VERBOSE, "Processing %zu chunk(s) of size %.2f MB\n",
                 numChunks, chunkSize / (1024.0 * 1024.0));

        size_t effectiveThreads = cpuThreads;
        if (effectiveThreads == 0) {
            effectiveThreads = std::thread::hardware_concurrency();
            if (effectiveThreads == 0) effectiveThreads = 4;
            if (effectiveThreads > 64) effectiveThreads = 64;
        }
        VLOG(VERBOSE, "  %zu worker threads, chunk size %zu KB\n",
             effectiveThreads, chunkSize / 1024);

        AsyncReader asyncReader;
        size_t maxReadQueue = std::min(size_t(64), numChunks);
        if (!asyncReader.start(inputFile, chunkSize, maxReadQueue)) {
            fprintf(stderr, "Error: Failed to start async reader\n");
            return false;
        }
        
        std::vector<uint8_t> headerBuffer;
        {
            std::ostringstream headerStream(std::ios::binary);
            if (!LZ4Frame::writeFrameHeader(headerStream, fileSize, chunkSize,
                                            storeContentSize && !stdinMode)) {
                fprintf(stderr, "Error: Failed to write LZ4 frame header\n");
                return false;
            }
            std::string headerStr = headerStream.str();
            headerBuffer.assign(headerStr.begin(), headerStr.end());
        }
        
        {
            int outputFd = stdoutMode ? STDOUT_FILENO
                                      : open(getActualOutputPath(),
                                             O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (outputFd < 0) {
                fprintf(stderr, "Error: Cannot create output file: %s\n", getActualOutputPath());
                return false;
            }
            ssize_t written = ::write(outputFd, headerBuffer.data(), headerBuffer.size());
            if (written != (ssize_t)headerBuffer.size()) {
                fprintf(stderr, "Error writing header\n");
                if (!stdoutMode) close(outputFd);
                return false;
            }
            if (!stdoutMode) close(outputFd);
        }
        
        XXH::State xxhState(XXH32_SEED);

        AsyncWriter asyncWriter;
        if (!asyncWriter.start(stdoutMode ? "-" : getActualOutputPath(), &xxhState, syncOutput)) {
            fprintf(stderr, "Error: Failed to start async writer\n");
            return false;
        }

        CPUCompressionPool cpuPool(effectiveThreads);

        size_t nextChunkToWrite = 0;
        size_t totalCompressed  = 0;
        size_t chunksSubmitted  = 0;
        size_t chunksExpanded   = 0;

        auto startTime = std::chrono::high_resolution_clock::now();

        // Dedicated progress thread: samples chunksSubmitted every 150 ms.
        // The main loop runs at pool/reader speed and can complete in <200 ms
        // for cached files, making inline progress updates invisible.  A
        // separate thread with a fixed sleep interval guarantees the bar
        // is visible for the full duration of compression regardless of speed.
        std::atomic<bool> stopCompressionProgress{false};
        std::thread compressionProgressThread;
        if (g_verbosity != QUIET && (stdinMode || numChunks > 10)) {
            compressionProgressThread = std::thread([&]() {
                g_progressActive.store(true, std::memory_order_relaxed);
                while (!stopCompressionProgress.load()) {
                    size_t submitted = chunksSubmitted;
                    if (!stdinMode) {
                        size_t bytesQueued = std::min(submitted * chunkSize, fileSize);
                        int progress = numChunks > 0
                            ? (int)((100 * submitted) / numChunks) : 0;
                        std::string cpuBytes = formatBytes(bytesQueued);
                        fprintf(stderr, "\r%sCompressing:%s %s%3d%%%s  %sCPU:%s %s%s%s%s",
                                CC_BCYAN, CC_RESET,
                                CC_BYELLOW, progress, CC_RESET,
                                CC_CYAN, CC_RESET, CC_BLUE, cpuBytes.c_str(), CC_RESET, CC_EL);
                    } else {
                        std::string cpuBytes = formatBytes(submitted * chunkSize);
                        fprintf(stderr, "\r%sCompressing:%s  %sCPU:%s %s%s%s%s",
                                CC_BCYAN, CC_RESET,
                                CC_CYAN, CC_RESET, CC_BLUE, cpuBytes.c_str(), CC_RESET, CC_EL);
                    }
                    fflush(stderr);
                    std::this_thread::sleep_for(std::chrono::milliseconds(150));
                }
            });
        }

        // Reader thread: submits chunks to the CPU pool as fast as the reader
        // produces them.  Runs concurrently with the drain loop below so the
        // pool always has work queued and the writer always has results to
        // consume  no polling, no alternating phases, true pipeline.
        std::thread readerSubmitThread([&]() {
            AsyncReader::ReadChunk chunk;
            while (asyncReader.getChunk(chunk)) {
                cpuPool.submitJob(chunk.chunkIndex, std::move(chunk.heapData), hcLevel);
                chunksSubmitted++;
                VLOG(DEBUG, "Submitted chunk %zu to CPU pool\n", chunk.chunkIndex);
            }
            // All chunks submitted. Signal finish() so waitForResult() in the
            // drain loop unblocks once the last result is posted by a worker.
            // Do not call stop() here  that would join workers prematurely
            // before they finish processing the remaining queued jobs.
            cpuPool.finish();
        });

        // Drain loop: blocks on waitForResult until each successive chunk is
        // ready, then enqueues it to the async writer immediately.  Wakes
        // exactly when the pool posts the result  no polling, no sleep.
        {
            auto compressStart = std::chrono::high_resolution_clock::now();
            CPUCompressionPool::CompressResult result;
            while (cpuPool.waitForResult(nextChunkToWrite, result)) {
                std::vector<std::vector<uint8_t>> compressedChunks;
                std::vector<std::vector<uint8_t>> originalChunks;

                compressedChunks.push_back(std::move(result.compressedData));
                originalChunks.push_back(std::move(result.originalData));

                if (!compressedChunks[0].empty()) {
                    totalCompressed += compressedChunks[0].size();
                } else {
                    totalCompressed += result.originalSize;
                    chunksExpanded++;
                }

                asyncWriter.enqueue(
                    nextChunkToWrite,
                    std::move(compressedChunks),
                    std::move(originalChunks),
                    {nextChunkToWrite},
                    {result.originalSize}
                );

                nextChunkToWrite++;
                VLOG(DEBUG, "Drained chunk %zu to writer\n", nextChunkToWrite - 1);
            }
            timeCompressing += std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - compressStart).count();
        }

        readerSubmitThread.join();

        // Stop compression progress thread before Writing: phase takes over
        stopCompressionProgress.store(true);
        if (compressionProgressThread.joinable()) compressionProgressThread.join();

        if (stdinMode) asyncWriter.setTotalChunks(nextChunkToWrite);
        
        {
            std::atomic<bool> stopProgress{false};
            std::thread progressThread;
            if (g_verbosity != QUIET && (stdinMode || numChunks > 10)) {
                progressThread = std::thread([&]() {
                    g_progressActive.store(true, std::memory_order_relaxed);
                    while (!stopProgress.load()) {
                        std::string written = formatBytes(asyncWriter.getBytesWritten());
                        if (stdinMode) {
                            VLOG(NORMAL, "\r%sWriting:%s  %s%s%s%s",
                                    CC_BGREEN, CC_RESET,
                                    CC_BGREEN, written.c_str(), CC_RESET, CC_EL);
                        } else {
                            size_t w = asyncWriter.getNextChunkToWrite();
                            std::string total = formatBytes(fileSize);
                            VLOG(NORMAL, "\r%sWriting:%s       %s%3d%%%s  %s[%s %s%s%s / %s%s%s ]%s%s",
                                    CC_BGREEN, CC_RESET,
                                    CC_BYELLOW, (int)(100 * w / numChunks), CC_RESET,
                                    CC_DIM, CC_RESET,
                                    CC_BGREEN, written.c_str(), CC_RESET,
                                    CC_WHITE, total.c_str(), CC_RESET,
                                    CC_DIM, CC_EL);
                        }
                        fflush(stderr);
                        std::this_thread::sleep_for(std::chrono::milliseconds(150));
                    }
                    std::string fb = formatBytes((size_t)asyncWriter.getBytesWritten());
                    if (stdinMode) {
                        VLOG(NORMAL, "\r%sWriting:%s  %s%s%s%s\n",
                                CC_BGREEN, CC_RESET, CC_BGREEN, fb.c_str(), CC_RESET, CC_EL);
                    } else {
                        std::string total = formatBytes(fileSize);
                        VLOG(NORMAL, "\r%sWriting:%s       %s100%%%s  %s[%s %s%s%s / %s%s%s ]%s\n",
                                CC_BGREEN, CC_RESET,
                                CC_BYELLOW, CC_RESET,
                                CC_DIM, CC_RESET,
                                CC_BGREEN, total.c_str(), CC_RESET,
                                CC_WHITE, total.c_str(), CC_RESET,
                                CC_DIM);
                    }
                });
            }
            
            VLOG(VERBOSE, "Waiting for async writer to complete...\n");
            asyncWriter.stop();
            
            stopProgress.store(true);
            if (progressThread.joinable()) progressThread.join();
            g_progressActive.store(false, std::memory_order_relaxed);
        }
        
        // Write footer (end mark + checksum)
        {
            int footerFd = stdoutMode ? STDOUT_FILENO
                                      : open(getActualOutputPath(), O_WRONLY | O_APPEND);
            if (footerFd >= 0) {
                uint32_t endMark = 0;
                ssize_t bytesWritten = ::write(footerFd, &endMark, 4);
                if (bytesWritten != 4)
                    fprintf(stderr, "Error writing end mark\n");

                uint32_t contentChecksum = xxhState.digest();
                uint8_t checksumBuf[4] = {
                    (uint8_t)(contentChecksum),
                    (uint8_t)(contentChecksum >> 8),
                    (uint8_t)(contentChecksum >> 16),
                    (uint8_t)(contentChecksum >> 24)
                };
                bytesWritten = ::write(footerFd, checksumBuf, 4);
                if (bytesWritten != 4)
                    fprintf(stderr, "Error writing checksum\n");

                if (!stdoutMode) {
                    if (syncOutput) fsync(footerFd);
                    close(footerFd);
                }
                VLOG(VERBOSE, "Computed content checksum: 0x%08X\n", contentChecksum);
            }
        }

        asyncReader.stop();
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        double totalBytesWritten = asyncWriter.getBytesWritten();
        double ratio = (!stdinMode && fileSize > 0)
            ? 100.0 * totalBytesWritten / fileSize : 0.0;
        double throughputMBps = (!stdinMode && fileSize > 0 && duration.count() > 0)
            ? (fileSize / (1024.0 * 1024.0)) / (duration.count() / 1000.0)
            : (totalBytesWritten / (1024.0 * 1024.0)) / (duration.count() / 1000.0);
        
        double cpuReadTime  = asyncReader.getReadTime();
        double cpuWriteTime = asyncWriter.getWriteTime();

        std::string inputSize  = stdinMode ? "(stdin)" : formatBytes(fileSize);
        std::string outputSize = formatBytes((size_t)totalBytesWritten);
        if (stdinMode)
            VLOG(NORMAL, "\r%sCompression complete:%s stdin -> %s%s%s in %.2f s\n",
                    CC_BGREEN, CC_RESET,
                    CC_CYAN, outputSize.c_str(), CC_RESET,
                    duration.count() / 1000.0);
        else
        VLOG(NORMAL, "\r%sCompression complete:%s %s%s%s -> %s%s%s %s(%.2f%%)%s in %.2f s\n",
                CC_BGREEN, CC_RESET,
                CC_WHITE, inputSize.c_str(), CC_RESET,
                CC_CYAN, outputSize.c_str(), CC_RESET,
                CC_BYELLOW, ratio, CC_RESET,
                duration.count() / 1000.0);
        VLOG(VERBOSE, "Throughput: %.2f MB/s\n", throughputMBps);
        VLOG(VERBOSE, "  Read:    %.2f s  |  CPU compress (%zu threads): %.2f s  |  Write: %.2f s\n",
             cpuReadTime, effectiveThreads, timeCompressing, cpuWriteTime);
        VLOG(VERBOSE, "  Uncompressed chunks: %zu / %zu (%.1f%%)\n",
             chunksExpanded, numChunks, 100.0 * chunksExpanded / numChunks);

        if (!keepOriginal && !stdoutMode) {
            if (unlink(inputFile.c_str()) != 0) {
                fprintf(stderr, "Warning: Could not remove input file: %s\n",
                        inputFile.c_str());
            }
        }

        return true;
    }

    /*
     * Compress a file using GPU acceleration (original implementation)
     */
    bool compressFileGPU() {
        for (auto& g : gpus)
            VLOG(VERBOSE, "  GPU%d: %s  %.1f GB  %zu SMs  %d pipeline slots\n",
                 g.deviceId, g.properties.name,
                 g.totalMemory/(1024.0*1024.0*1024.0), g.smCount, g.pipelineDepth);
        VLOG(NORMAL, "%sCompressing%s (GPU-only, %zu GPU%s): %s -> %s\n",
                CC_BCYAN, CC_RESET,
                gpus.size(), gpus.size()==1?"":"s",
                inputFile.c_str(), outputFile.c_str());
        if (hcLevel > 0)
            fprintf(stderr, "Note: --best / HC compression is not supported by nvCOMP; "
                            "GPU path uses fast LZ4 (-9). Use --hybrid or --cpu-only for HC.\n");

        const bool stdinMode = (inputFile == "-");
        size_t fileSize = 0;
        if (!stdinMode) {
            struct stat st;
            if (stat(inputFile.c_str(), &st) != 0) {
                fprintf(stderr, "Error: Cannot stat input file: %s\n", inputFile.c_str());
                return false;
            }
            fileSize = st.st_size;
            VLOG(VERBOSE, "Input file size: %.2f MB\n", fileSize / (1024.0 * 1024.0));
        } else {
            VLOG(VERBOSE, "Input: stdin (size unknown)\n");
        }

        size_t numChunks = fileSize > 0
            ? (fileSize + chunkSize - 1) / chunkSize
            : SIZE_MAX;
        if (!stdinMode)
            VLOG(VERBOSE, "Processing %zu chunk(s) of size %.2f MB\n",
                 numChunks, chunkSize / (1024.0 * 1024.0));

        size_t totalPipelineSlots = 0;
        for (auto& g : gpus) totalPipelineSlots += g.pipelineDepth;
        size_t estBatch    = gpus.empty() ? 64 :
            std::max(size_t(64), static_cast<size_t>(
                gpus[0].availableMemory * 0.90 / gpus[0].pipelineDepth / (chunkSize * 5)));
        size_t maxReadQueue = stdinMode
            ? std::max(size_t(256), totalPipelineSlots * estBatch)
            : std::min(numChunks, std::max(size_t(256), totalPipelineSlots * estBatch));

        AsyncReader localReader;
        AsyncReader* asyncReaderPtr = nullptr;

        if (earlyReaderStarted) {
            asyncReaderPtr = &earlyReader;
            VLOG(VERBOSE, "Pre-warmed reader: %.2f MB buffered so far\n",
                 earlyReader.getBytesRead() / (1024.0*1024.0));
        } else {
            bool started = inputPool.numSlots()
                ? localReader.startPooled(inputFile, chunkSize, &inputPool)
                : localReader.start(inputFile, chunkSize, maxReadQueue);
            if (!started) {
                fprintf(stderr, "Error: Failed to start async reader\n");
                return false;
            }
            asyncReaderPtr = &localReader;
        }
        AsyncReader& asyncReader = *asyncReaderPtr;

        VLOG(VERBOSE, "Reader queue: %zu chunks (%.2f GB RAM) to feed %zu pipeline slot%s\n",
             maxReadQueue, (maxReadQueue * chunkSize) / (1024.0*1024.0*1024.0),
             totalPipelineSlots, totalPipelineSlots == 1 ? "" : "s");

        const size_t SLOT_CAPACITY = slotCapacity;

        if (!inputPool.numSlots() && !earlyReaderStarted) {
            size_t poolSlots = stdinMode
                ? std::max(size_t(64), 2 * totalPipelineSlots * slotCapacity)
                : std::min(numChunks, std::max(size_t(64), 2 * totalPipelineSlots * slotCapacity));
            if (!inputPool.init(poolSlots, chunkSize)) {
                fprintf(stderr, "Warning: pinned pool alloc failed, using heap\n");
            } else {
                VLOG(VERBOSE, "PinnedInputPool: %zu slots Ã %.0f MB = %.1f GB\n",
                     poolSlots, chunkSize/1024.0/1024.0,
                     poolSlots*chunkSize/1024.0/1024.0/1024.0);
            }
        }
        nvcompBatchedLZ4CompressOpts_t nvOpts = nvcompBatchedLZ4CompressDefaultOpts;
        size_t maxOutPerChunk = 0;
        nvcompBatchedLZ4CompressGetMaxOutputChunkSize(chunkSize, nvOpts, &maxOutPerChunk);

        size_t sharedTempBytes = 0;
        {
            cudaSetDevice(gpus[0].deviceId);
            const void** qiPtrs = nullptr; size_t* qiSizes = nullptr;
            cudaMalloc(&qiPtrs,  SLOT_CAPACITY * sizeof(void*));
            cudaMalloc(&qiSizes, SLOT_CAPACITY * sizeof(size_t));
            std::vector<const void*> hip(SLOT_CAPACITY, nullptr);
            std::vector<size_t>      hisz(SLOT_CAPACITY, chunkSize);
            cudaMemcpy(qiPtrs, hip.data(),  SLOT_CAPACITY*sizeof(void*),  cudaMemcpyHostToDevice);
            cudaMemcpy(qiSizes,hisz.data(), SLOT_CAPACITY*sizeof(size_t), cudaMemcpyHostToDevice);
            nvcompBatchedLZ4CompressGetTempSizeSync(
                qiPtrs, qiSizes, SLOT_CAPACITY, chunkSize, nvOpts,
                &sharedTempBytes, SLOT_CAPACITY*chunkSize, 0);
            cudaFree(qiPtrs); cudaFree(qiSizes);
        }

        std::vector<std::vector<PreallocSlot>> gpuSlots(gpus.size());
        std::vector<bool> gpuInitOk(gpus.size(), true);

        auto freeAllSlots = [&]() {
            for (auto& gSlots : gpuSlots)
                for (auto& s : gSlots) s.release();
        };

        auto initSlotT0 = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> slotInitThreads;
        for (size_t g = 0; g < gpus.size(); g++) {
            slotInitThreads.emplace_back([&, g]() {
                cudaSetDevice(gpus[g].deviceId);
                int depth = gpus[g].pipelineDepth;
                gpuSlots[g].resize(depth);
                std::vector<const void*> hip(SLOT_CAPACITY);
                std::vector<void*>       hop(SLOT_CAPACITY);
                for (int si = 0; si < depth; si++) {
                    PreallocSlot& sl = gpuSlots[g][si];
                    sl.deviceId    = gpus[g].deviceId;
                    sl.capacity    = SLOT_CAPACITY;
                    sl.chunkStride = chunkSize;
                    sl.outStride   = maxOutPerChunk;
                    sl.tempBytes   = sharedTempBytes;
                    bool ok = true;
                    ok = ok && cudaMalloc(&sl.d_input,  SLOT_CAPACITY*chunkSize)      == cudaSuccess;
                    ok = ok && cudaMalloc(&sl.d_output, SLOT_CAPACITY*maxOutPerChunk) == cudaSuccess;
                    ok = ok && cudaMalloc(&sl.d_iPtrs,  SLOT_CAPACITY*sizeof(void*))  == cudaSuccess;
                    ok = ok && cudaMalloc(&sl.d_oPtrs,  SLOT_CAPACITY*sizeof(void*))  == cudaSuccess;
                    ok = ok && cudaMalloc(&sl.d_iSizes, SLOT_CAPACITY*sizeof(size_t))         == cudaSuccess;
                    ok = ok && cudaMalloc(&sl.d_oSizes, SLOT_CAPACITY*sizeof(size_t))         == cudaSuccess;
                    ok = ok && cudaMalloc(&sl.d_stats,  SLOT_CAPACITY*sizeof(nvcompStatus_t)) == cudaSuccess;
                    ok = ok && cudaMalloc(&sl.d_temp,   sharedTempBytes)                      == cudaSuccess;
                    if (ok) {
                        for (size_t k = 0; k < SLOT_CAPACITY; k++) {
                            hip[k] = sl.d_input  + k*chunkSize;
                            hop[k] = sl.d_output + k*maxOutPerChunk;
                        }
                        cudaMemcpy(sl.d_iPtrs, hip.data(), SLOT_CAPACITY*sizeof(void*), cudaMemcpyHostToDevice);
                        cudaMemcpy(sl.d_oPtrs, hop.data(), SLOT_CAPACITY*sizeof(void*), cudaMemcpyHostToDevice);
                    }
                    ok = ok && cudaHostAlloc(&sl.h_iSizes, SLOT_CAPACITY*sizeof(size_t),         cudaHostAllocDefault) == cudaSuccess;
                    ok = ok && cudaHostAlloc(&sl.h_oSizes, SLOT_CAPACITY*sizeof(size_t),         cudaHostAllocDefault) == cudaSuccess;
                    ok = ok && cudaHostAlloc(&sl.h_stats,  SLOT_CAPACITY*sizeof(nvcompStatus_t), cudaHostAllocDefault) == cudaSuccess;
                    ok = ok && cudaHostAlloc(&sl.h_output, SLOT_CAPACITY*maxOutPerChunk,         cudaHostAllocDefault) == cudaSuccess;
                    ok = ok && cudaStreamCreate(&sl.stream) == cudaSuccess;
                    if (ok) { sl.ready = true; }
                    else    {
                              fprintf(stderr, "GPU%d: failed to init slot %d\n", gpus[g].deviceId, si);
                              gpuInitOk[g] = false; break; }
                }
            });
        }

        std::vector<uint8_t> headerBuffer;
        {
            std::ostringstream headerStream(std::ios::binary);
            if (!LZ4Frame::writeFrameHeader(headerStream, fileSize, chunkSize,
                                            storeContentSize && !stdinMode)) {
                fprintf(stderr, "Error: Failed to write LZ4 frame header\n");
                return false;
            }
            std::string headerStr = headerStream.str();
            headerBuffer.assign(headerStr.begin(), headerStr.end());
        }
        
        {
            int hdrFd = stdoutMode ? STDOUT_FILENO
                                   : open(getActualOutputPath(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (hdrFd < 0) {
                fprintf(stderr, "Error: Cannot create output file: %s\n", getActualOutputPath());
                return false;
            }
            ssize_t written = ::write(hdrFd, headerBuffer.data(), headerBuffer.size());
            if (written != (ssize_t)headerBuffer.size()) {
                fprintf(stderr, "Error writing header: %s\n", strerror(errno));
                if (!stdoutMode) close(hdrFd);
                return false;
            }
            if (!stdoutMode) close(hdrFd);
        }
        
        XXH::State xxhState(XXH32_SEED);
        
        for (auto& t : slotInitThreads) t.join();

        bool slotsOk = true;
        for (bool ok : gpuInitOk) if (!ok) { slotsOk = false; break; }

        double initMs = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - initSlotT0).count();
        VLOG(VERBOSE, "Slot init: %.2f s (parallel, overlapped with file setup)\n",
             initMs / 1000.0);

        AsyncWriter asyncWriter;
        if (!asyncWriter.start(stdoutMode ? "-" : getActualOutputPath(), &xxhState, syncOutput)) {
            fprintf(stderr, "Error: Failed to start async writer\n");
            freeAllSlots(); asyncReader.stop(); return false;
        }

        if (!slotsOk) { freeAllSlots(); asyncWriter.stop(); asyncReader.stop(); return false; }

        size_t totalSlots = 0;
        for (auto& gSlots : gpuSlots) totalSlots += gSlots.size();
        VLOG(VERBOSE, "GPU pipeline ready: %zu GPUs × %d streams/GPU × %zu chunks/slot  =  %.1f GB VRAM/slot\n",
             gpus.size(), gpus[0].pipelineDepth, SLOT_CAPACITY,
             (SLOT_CAPACITY * chunkSize * 5) / (1024.0*1024.0*1024.0));

        std::atomic<bool>   workerAbort{false};

        std::atomic<size_t> chunksSubmitted{0};

        auto gpuWorker = [&](size_t gpuIdx) {
            GPUDevice& gpu = gpus[gpuIdx];
            cudaSetDevice(gpu.deviceId);
            std::vector<PreallocSlot>& slots = gpuSlots[gpuIdx];
            const int nSlots = (int)slots.size();
            int sIdx = 0;
            bool firstRound = true;

            while (!workerAbort.load()) {
                PreallocSlot& slot = slots[sIdx];
                sIdx = (sIdx + 1) % nSlots;

                cudaStreamSynchronize(slot.stream);

                if (!firstRound && slot.hasPending) {
                    std::vector<std::vector<uint8_t>> cChunks, oChunks;
                    cChunks.reserve(slot.batchSize);
                    oChunks.reserve(slot.batchSize);
                    for (size_t i = 0; i < slot.batchSize; i++) {
                        size_t outSz  = slot.h_oSizes[i];
                        size_t origSz = slot.origSizes[i];
                        const uint8_t* origPtr = slot.origHandles.size() > i
                            ? slot.origHandles[i].data : slot.origData[i].data();
                        if (outSz < origSz) {
                            std::vector<uint8_t> buf(outSz);
                            memcpy(buf.data(), slot.h_output + i * slot.outStride, outSz);
                            cChunks.push_back(std::move(buf));
                        } else {
                            cChunks.push_back({});
                        }
                        oChunks.emplace_back(origPtr, origPtr + origSz);
                    }
                    slot.origHandles.clear(); slot.origData.clear();
                    asyncWriter.enqueueBatch(cChunks, oChunks, slot.indices, slot.origSizes);
                    chunksSubmitted += slot.batchSize;
                    slot.hasPending = false;
                }
                firstRound = false;

                slot.indices.clear();
                slot.origSizes.clear();
                slot.origData.clear();
                slot.origHandles.clear();
                slot.batchSize = 0;

                while (slot.batchSize < slot.capacity) {
                    AsyncReader::ReadChunk chunk;
                    if (!asyncReader.getChunk(chunk)) break;
                    cudaMemcpyAsync(
                        slot.d_input + slot.batchSize * slot.chunkStride,
                        chunk.data(), chunk.size,
                        cudaMemcpyHostToDevice, slot.stream);
                    slot.h_iSizes[slot.batchSize] = chunk.size;
                    slot.indices.push_back(chunk.chunkIndex);
                    slot.origSizes.push_back(chunk.size);
                    if (chunk.poolHandle.valid())
                        slot.origHandles.push_back(std::move(chunk.poolHandle));
                    else
                        slot.origData.push_back(std::move(chunk.heapData));
                    slot.batchSize++;
                }

                if (slot.batchSize == 0) break;

                cudaMemcpyAsync(slot.d_iSizes, slot.h_iSizes,
                                slot.batchSize * sizeof(size_t),
                                cudaMemcpyHostToDevice, slot.stream);
                nvcompStatus_t nErr = nvcompBatchedLZ4CompressAsync(
                    slot.d_iPtrs, slot.d_iSizes, slot.chunkStride,
                    slot.batchSize, slot.d_temp, slot.tempBytes,
                    slot.d_oPtrs, slot.d_oSizes, nvOpts, slot.d_stats, slot.stream);
                if (nErr != nvcompSuccess) {
                    fprintf(stderr, "GPU%d: nvcomp error %d\n", gpu.deviceId, (int)nErr);
                    workerAbort.store(true); break;
                }
                cudaMemcpyAsync(slot.h_output, slot.d_output,
                                slot.batchSize * slot.outStride,
                                cudaMemcpyDeviceToHost, slot.stream);
                cudaMemcpyAsync(slot.h_oSizes, slot.d_oSizes,
                                slot.batchSize * sizeof(size_t),
                                cudaMemcpyDeviceToHost, slot.stream);
                cudaMemcpyAsync(slot.h_stats, slot.d_stats,
                                slot.batchSize * sizeof(nvcompStatus_t),
                                cudaMemcpyDeviceToHost, slot.stream);
                slot.hasPending = true;
            }

            for (int si = 0; si < nSlots; si++) {
                PreallocSlot& sl = slots[si];
                cudaStreamSynchronize(sl.stream);
                if (!sl.hasPending) continue;
                std::vector<std::vector<uint8_t>> cChunks, oChunks;
                cChunks.reserve(sl.batchSize);
                oChunks.reserve(sl.batchSize);
                for (size_t i = 0; i < sl.batchSize; i++) {
                    size_t outSz  = sl.h_oSizes[i];
                    size_t origSz = sl.origSizes[i];
                    const uint8_t* origPtr = sl.origHandles.size() > i
                        ? sl.origHandles[i].data : sl.origData[i].data();
                    if (outSz < origSz) {
                        std::vector<uint8_t> buf(outSz);
                        memcpy(buf.data(), sl.h_output + i * sl.outStride, outSz);
                        cChunks.push_back(std::move(buf));
                    } else {
                        cChunks.push_back({});
                    }
                    oChunks.emplace_back(origPtr, origPtr + origSz);
                }
                sl.origHandles.clear(); sl.origData.clear();
                asyncWriter.enqueueBatch(cChunks, oChunks, sl.indices, sl.origSizes);
                chunksSubmitted += sl.batchSize;
                sl.hasPending = false;
            }
        };

        asyncWriter.setTotalChunks(numChunks);

        auto startTime = std::chrono::high_resolution_clock::now();
        std::atomic<int> activeWorkers{(int)gpus.size()};
        std::vector<std::thread> workers;
        for (size_t g = 0; g < gpus.size(); g++) {
            workers.emplace_back([&, g]() {
                gpuWorker(g);
                --activeWorkers;
            });
        }

        while (activeWorkers.load() > 0) {
            if (g_verbosity != QUIET && numChunks > 10) {
                size_t submitted = chunksSubmitted.load();
                size_t bytesProcessed = stdinMode
                    ? submitted * chunkSize
                    : std::min(submitted * chunkSize, fileSize);
                std::string gpuBytes = formatBytes(bytesProcessed);
                if (stdinMode) {
                    fprintf(stderr, "\r%sCompressing:%s  %sGPU:%s %s%s%s%s",
                            CC_BCYAN, CC_RESET,
                            CC_CYAN, CC_RESET, CC_GREEN, gpuBytes.c_str(), CC_RESET, CC_EL);
                } else {
                    int progress = (int)(100 * submitted / numChunks);
                    fprintf(stderr, "\r%sCompressing:%s %s%3d%%%s  %sGPU:%s %s%s%s%s",
                            CC_BCYAN, CC_RESET,
                            CC_BYELLOW, progress, CC_RESET,
                            CC_CYAN, CC_RESET, CC_GREEN, gpuBytes.c_str(), CC_RESET, CC_EL);
                }
                fflush(stderr);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        for (auto& w : workers) if (w.joinable()) w.join();

        if (stdinMode) asyncWriter.setTotalChunks(chunksSubmitted.load());

        {
            std::atomic<bool> stopProgress{false};
            std::thread progressThread;
            if (g_verbosity != QUIET && (stdinMode || numChunks > 10)) {
                progressThread = std::thread([&]() {
                    g_progressActive.store(true, std::memory_order_relaxed);
                    while (!stopProgress.load()) {
                        std::string written = formatBytes(asyncWriter.getBytesWritten());
                        if (stdinMode) {
                            VLOG(NORMAL, "\r%sWriting:%s  %s%s%s%s",
                                    CC_BGREEN, CC_RESET,
                                    CC_BGREEN, written.c_str(), CC_RESET, CC_EL);
                        } else {
                            size_t w = asyncWriter.getNextChunkToWrite();
                            std::string total = formatBytes(fileSize);
                            VLOG(NORMAL, "\r%sWriting:%s       %s%3d%%%s  %s[%s %s%s%s / %s%s%s ]%s%s",
                                    CC_BGREEN, CC_RESET,
                                    CC_BYELLOW, (int)(100 * w / numChunks), CC_RESET,
                                    CC_DIM, CC_RESET,
                                    CC_BGREEN, written.c_str(), CC_RESET,
                                    CC_WHITE, total.c_str(), CC_RESET,
                                    CC_DIM, CC_EL);
                        }
                        fflush(stderr);
                        std::this_thread::sleep_for(std::chrono::milliseconds(150));
                    }
                    std::string fb = formatBytes((size_t)asyncWriter.getBytesWritten());
                    if (stdinMode) {
                        VLOG(NORMAL, "\r%sWriting:%s  %s%s%s%s\n",
                                CC_BGREEN, CC_RESET, CC_BGREEN, fb.c_str(), CC_RESET, CC_EL);
                    } else {
                        std::string total = formatBytes(fileSize);
                        VLOG(NORMAL, "\r%sWriting:%s       %s100%%%s  %s[%s %s%s%s / %s%s%s ]%s\n",
                                CC_BGREEN, CC_RESET,
                                CC_BYELLOW, CC_RESET,
                                CC_DIM, CC_RESET,
                                CC_BGREEN, total.c_str(), CC_RESET,
                                CC_WHITE, total.c_str(), CC_RESET,
                                CC_DIM);
                    }
                });
            }

            VLOG(VERBOSE, "Waiting for async writer to complete...\n");
            asyncWriter.stop();

            stopProgress.store(true);
            if (progressThread.joinable()) progressThread.join();
            g_progressActive.store(false, std::memory_order_relaxed);
        }

        freeAllSlots();
        
        // Append footer synchronously (end mark + checksum)
        {
            int footerFd = stdoutMode ? STDOUT_FILENO
                                      : open(getActualOutputPath(), O_WRONLY | O_APPEND);
            if (footerFd < 0) {
                fprintf(stderr, "Error reopening file for footer: %s\n", strerror(errno));
                return false;
            }
            uint32_t endMark = 0;
            if (::write(footerFd, &endMark, 4) != 4)
                fprintf(stderr, "Error writing end mark: %s\n", strerror(errno));

            uint32_t contentChecksum = xxhState.digest();
            uint8_t checksumBuf[4] = {
                (uint8_t)(contentChecksum),
                (uint8_t)(contentChecksum >> 8),
                (uint8_t)(contentChecksum >> 16),
                (uint8_t)(contentChecksum >> 24)
            };
            if (::write(footerFd, checksumBuf, 4) != 4)
                fprintf(stderr, "Error writing checksum: %s\n", strerror(errno));

            if (!stdoutMode) {
                if (syncOutput) fsync(footerFd);
                close(footerFd);
            }
            VLOG(VERBOSE, "Computed content checksum: 0x%08X (from %zu bytes)\n",
                 contentChecksum, xxhState.totalLen);
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        asyncReader.stop();
        
        double totalBytesWritten = asyncWriter.getBytesWritten();
        double ratio = (!stdinMode && fileSize > 0) ? 100.0 * totalBytesWritten / fileSize : 0.0;
        double throughputMBps = (!stdinMode && fileSize > 0 && duration.count() > 0)
            ? (fileSize / (1024.0 * 1024.0)) / (duration.count() / 1000.0)
            : (totalBytesWritten / (1024.0 * 1024.0)) / (duration.count() / 1000.0);
        
        double asyncReadTime  = asyncReader.getReadTime();
        double asyncWriteTime = asyncWriter.getWriteTime();

        size_t finalSlots = 0;
        for (auto& gSlots : gpuSlots) finalSlots += gSlots.size();
        std::string inputSize  = stdinMode ? "(stdin)" : formatBytes(fileSize);
        std::string outputSize = formatBytes((size_t)totalBytesWritten);
        if (stdinMode)
            VLOG(NORMAL, "\r%sCompression complete:%s stdin -> %s%s%s in %.2f s\n",
                    CC_BGREEN, CC_RESET,
                    CC_CYAN, outputSize.c_str(), CC_RESET,
                    duration.count() / 1000.0);
        else
            VLOG(NORMAL, "\r%sCompression complete:%s %s%s%s -> %s%s%s %s(%.2f%%)%s in %.2f s\n",
                    CC_BGREEN, CC_RESET,
                    CC_WHITE, inputSize.c_str(), CC_RESET,
                    CC_CYAN, outputSize.c_str(), CC_RESET,
                    CC_BYELLOW, ratio, CC_RESET,
                    duration.count() / 1000.0);
        VLOG(VERBOSE, "Throughput: %.2f MB/s\n", throughputMBps);
        VLOG(VERBOSE, "  Read: %.2f s  |  Write: %.2f s\n",
             asyncReadTime, asyncWriteTime);
        VLOG(VERBOSE, "  %zu GPU%s × %d streams/GPU × %zu chunks/slot  =  %.1f GB VRAM/slot\n",
             gpus.size(), gpus.size() == 1 ? "" : "s",
             gpus.empty() ? 0 : gpus[0].pipelineDepth,
             SLOT_CAPACITY,
             (SLOT_CAPACITY * chunkSize * 5) / (1024.0*1024.0*1024.0));

        if (!keepOriginal && !stdoutMode) {
            if (unlink(inputFile.c_str()) != 0) {
                fprintf(stderr, "Warning: Could not remove input file: %s\n", 
                        inputFile.c_str());
            }
        }

        return true;
    }

    /*
     * Compress file - dispatcher to appropriate backend
     */
    bool compressFile() {
        switch (backendMode) {
            case BackendMode::CPU_ONLY:  return compressFileCPU();
            case BackendMode::GPU_ONLY:  return compressFileGPU();
            case BackendMode::HYBRID:    return compressFileHybrid();
            default:
                fprintf(stderr, "Error: Unknown backend mode\n");
                return false;
        }
    }

    /*
     * Hybrid compression: GPU-priority dispatcher + CPU fill-in.
     */
    bool compressFileHybrid() {
        const bool stdinMode = (inputFile == "-");
        size_t fileSize = 0;
        if (!stdinMode) {
            struct stat st;
            if (stat(inputFile.c_str(), &st) != 0) {
                fprintf(stderr, "Error: Cannot stat input file: %s\n", inputFile.c_str());
                return false;
            }
            fileSize = st.st_size;
        }
        size_t numChunks = fileSize > 0
            ? (fileSize + chunkSize - 1) / chunkSize
            : SIZE_MAX;

        size_t effectiveThreads = cpuThreads ? cpuThreads
                                             : std::thread::hardware_concurrency();
        if (!effectiveThreads) effectiveThreads = 4;

        VLOG(NORMAL, "%sCompressing%s (Hybrid, %zu GPU%s + %zu thread%s): %s -> %s\n",
                CC_BCYAN, CC_RESET,
                gpus.size(),      gpus.size()==1?"":"s",
                effectiveThreads, effectiveThreads==1?"":"s",
                inputFile.c_str(), outputFile.c_str());

        size_t totalPipelineSlots = 0;
        for (auto& g : gpus) totalPipelineSlots += g.pipelineDepth;
        size_t estBatch = gpus.empty() ? 64 :
            std::max(size_t(64), static_cast<size_t>(
                gpus[0].availableMemory * 0.90 / gpus[0].pipelineDepth / (chunkSize * 5)));
        size_t maxReadQueue = stdinMode
            ? std::max(size_t(256), totalPipelineSlots * estBatch)
            : std::min(numChunks, std::max(size_t(256), totalPipelineSlots * estBatch));

        AsyncReader  localReader;
        AsyncReader* asyncReaderPtr = nullptr;
        if (earlyReaderStarted) {
            asyncReaderPtr = &earlyReader;
        } else {
            bool started = inputPool.numSlots()
                ? localReader.startPooled(inputFile, chunkSize, &inputPool)
                : localReader.start(inputFile, chunkSize, maxReadQueue);
            if (!started) { fprintf(stderr,"Error: reader start failed\n"); return false; }
            asyncReaderPtr = &localReader;
        }
        AsyncReader& asyncReader = *asyncReaderPtr;

        const size_t SLOT_CAPACITY = slotCapacity;

        if (!inputPool.numSlots() && !earlyReaderStarted) {
            // Use up to 60% of available system RAM for the pinned input pool.
            // The reader blocks only when the pool is full, so larger = less
            // reader stalling.  cudaHostAlloc is tried first; if it fails we
            // halve and retry down to a minimum floor.
            size_t availRam = 0;
            {
                struct sysinfo si;
                if (sysinfo(&si) == 0)
                    availRam = (size_t)si.freeram * si.mem_unit;
            }
            // Floor: enough for 2× the GPU pipeline so GPUs are never starved.
            const size_t floorSlots = std::max(size_t(64),
                                               2 * totalPipelineSlots * slotCapacity);
            // Target: 60% of free RAM, capped at total file chunks.
            size_t targetSlots = floorSlots;
            if (availRam > 0 && chunkSize > 0) {
                size_t ramSlots = (size_t)(availRam * 0.60) / chunkSize;
                targetSlots = std::min(numChunks > 0 ? numChunks : SIZE_MAX,
                                       std::max(floorSlots, ramSlots));
            }
            VLOG(VERBOSE, "Input pool target: %zu slots × %.1f MB = %.2f GB "
                 "(%.1f%% of %.2f GB free RAM)\n",
                 targetSlots, chunkSize / (1024.0*1024.0),
                 targetSlots * chunkSize / (1024.0*1024.0*1024.0),
                 availRam > 0 ? 60.0 : 0.0,
                 availRam / (1024.0*1024.0*1024.0));
            // Try to allocate; halve on failure until we hit the floor.
            while (targetSlots >= floorSlots) {
                if (inputPool.init(targetSlots, chunkSize)) break;
                VLOG(VERBOSE, "Pool alloc failed at %zu slots, trying %zu\n",
                     targetSlots, targetSlots / 2);
                inputPool.destroy();
                targetSlots /= 2;
            }
            if (!inputPool.numSlots()) {
                if (!inputPool.init(floorSlots, chunkSize))
                    fprintf(stderr, "Warning: pinned pool alloc failed, using heap\n");
            }
        }

        nvcompBatchedLZ4CompressOpts_t nvOpts = nvcompBatchedLZ4CompressDefaultOpts;
        size_t maxOutPerChunk = 0;
        nvcompBatchedLZ4CompressGetMaxOutputChunkSize(chunkSize, nvOpts, &maxOutPerChunk);

        size_t sharedTempBytes = 0;
        {
            cudaSetDevice(gpus[0].deviceId);
            const void** qiPtrs = nullptr; size_t* qiSizes = nullptr;
            cudaMalloc(&qiPtrs,  SLOT_CAPACITY * sizeof(void*));
            cudaMalloc(&qiSizes, SLOT_CAPACITY * sizeof(size_t));
            std::vector<const void*> hip(SLOT_CAPACITY, nullptr);
            std::vector<size_t>      hisz(SLOT_CAPACITY, chunkSize);
            cudaMemcpy(qiPtrs, hip.data(),  SLOT_CAPACITY*sizeof(void*),  cudaMemcpyHostToDevice);
            cudaMemcpy(qiSizes,hisz.data(), SLOT_CAPACITY*sizeof(size_t), cudaMemcpyHostToDevice);
            nvcompBatchedLZ4CompressGetTempSizeSync(
                qiPtrs, qiSizes, SLOT_CAPACITY, chunkSize, nvOpts,
                &sharedTempBytes, SLOT_CAPACITY*chunkSize, 0);
            cudaFree(qiPtrs); cudaFree(qiSizes);
        }

        std::vector<std::vector<PreallocSlot>> gpuSlots(gpus.size());
        std::vector<bool>   gpuInitOk(gpus.size(), true);
        std::atomic<size_t> gpuFreeSlots{0};   // free GPU slots; set after slot init
        std::atomic<size_t> totalGpuSlots{0};  // total slots; set after slot init

        auto freeAllSlots = [&]() {
            for (auto& gSlots : gpuSlots)
                for (auto& s : gSlots) s.release();
        };

        std::vector<std::thread> slotInitThreads;
        for (size_t g = 0; g < gpus.size(); g++) {
            slotInitThreads.emplace_back([&, g]() {
                cudaSetDevice(gpus[g].deviceId);
                int depth = gpus[g].pipelineDepth;
                gpuSlots[g].resize(depth);
                std::vector<const void*> hip2(SLOT_CAPACITY);
                std::vector<void*>       hop2(SLOT_CAPACITY);
                for (int si = 0; si < depth; si++) {
                    PreallocSlot& sl = gpuSlots[g][si];
                    sl.deviceId    = gpus[g].deviceId;
                    sl.capacity    = SLOT_CAPACITY;
                    sl.chunkStride = chunkSize;
                    sl.outStride   = maxOutPerChunk;
                    sl.tempBytes   = sharedTempBytes;
                    bool ok = true;
                    ok = ok && cudaMalloc(&sl.d_input,  SLOT_CAPACITY*chunkSize)               == cudaSuccess;
                    ok = ok && cudaMalloc(&sl.d_output, SLOT_CAPACITY*maxOutPerChunk)          == cudaSuccess;
                    ok = ok && cudaMalloc(&sl.d_iPtrs,  SLOT_CAPACITY*sizeof(void*))           == cudaSuccess;
                    ok = ok && cudaMalloc(&sl.d_oPtrs,  SLOT_CAPACITY*sizeof(void*))           == cudaSuccess;
                    ok = ok && cudaMalloc(&sl.d_iSizes, SLOT_CAPACITY*sizeof(size_t))          == cudaSuccess;
                    ok = ok && cudaMalloc(&sl.d_oSizes, SLOT_CAPACITY*sizeof(size_t))          == cudaSuccess;
                    ok = ok && cudaMalloc(&sl.d_stats,  SLOT_CAPACITY*sizeof(nvcompStatus_t))  == cudaSuccess;
                    ok = ok && cudaMalloc(&sl.d_temp,   sharedTempBytes)                       == cudaSuccess;
                    if (ok) {
                        for (size_t k = 0; k < SLOT_CAPACITY; k++) {
                            hip2[k] = sl.d_input  + k*chunkSize;
                            hop2[k] = sl.d_output + k*maxOutPerChunk;
                        }
                        cudaMemcpy(sl.d_iPtrs, hip2.data(), SLOT_CAPACITY*sizeof(void*), cudaMemcpyHostToDevice);
                        cudaMemcpy(sl.d_oPtrs, hop2.data(), SLOT_CAPACITY*sizeof(void*), cudaMemcpyHostToDevice);
                    }
                    ok = ok && cudaHostAlloc(&sl.h_iSizes, SLOT_CAPACITY*sizeof(size_t),         cudaHostAllocDefault) == cudaSuccess;
                    ok = ok && cudaHostAlloc(&sl.h_oSizes, SLOT_CAPACITY*sizeof(size_t),         cudaHostAllocDefault) == cudaSuccess;
                    ok = ok && cudaHostAlloc(&sl.h_stats,  SLOT_CAPACITY*sizeof(nvcompStatus_t), cudaHostAllocDefault) == cudaSuccess;
                    ok = ok && cudaHostAlloc(&sl.h_output, SLOT_CAPACITY*maxOutPerChunk,         cudaHostAllocDefault) == cudaSuccess;
                    ok = ok && cudaStreamCreate(&sl.stream) == cudaSuccess;
                    if (ok) { sl.ready = true; }
                    else    { fprintf(stderr, "GPU%d: failed to init slot %d\n", gpus[g].deviceId, si);
                              gpuInitOk[g] = false; break; }
                }
            });
        }

        // Write LZ4 frame header
        {
            std::ostringstream hs(std::ios::binary);
            if (!LZ4Frame::writeFrameHeader(hs, fileSize, chunkSize,
                                            storeContentSize && !stdinMode)) {
                fprintf(stderr,"Error: Failed to write LZ4 frame header\n"); return false;
            }
            std::string hstr = hs.str();
            int fd = stdoutMode ? STDOUT_FILENO
                                : open(getActualOutputPath(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd < 0) { fprintf(stderr,"Error: Cannot create output file\n"); return false; }
            if (::write(fd, hstr.data(), hstr.size()) != (ssize_t)hstr.size()) {
                fprintf(stderr,"Error writing header\n");
                if (!stdoutMode) close(fd);
                return false;
            }
            if (!stdoutMode) close(fd);
        }

        XXH::State xxhState(XXH32_SEED);

        // Declare all shared state before cpuWorker so the lambda can capture it.
        AsyncWriter asyncWriter;
        if (!asyncWriter.start(stdoutMode ? "-" : getActualOutputPath(), &xxhState, syncOutput)) {
            fprintf(stderr,"Error: Failed to start async writer\n");
            freeAllSlots(); asyncReader.stop(); return false;
        }

        if (!stdinMode) asyncWriter.setTotalChunks(numChunks);
        auto startTime = std::chrono::high_resolution_clock::now();
        std::atomic<size_t> chunksSubmitted{0};
        std::atomic<bool>   workerAbort{false};
        std::atomic<size_t> gpuChunkCount{0};
        std::atomic<size_t> cpuChunkCount{0};

        TsQueue<AsyncReader::ReadChunk> gpuWorkQueue;
        TsQueue<AsyncReader::ReadChunk> cpuWorkQueue;

        // Shared compress-and-enqueue helper used by both phases of cpuWorker.
        auto compressAndSend = [&](AsyncReader::ReadChunk& chunk) {
            size_t origSz = chunk.size;
            const uint8_t* src = chunk.data();
            size_t maxOut = LZ4_compressBound((int)origSz);
            std::vector<uint8_t> compressed(maxOut);
            int compSz = (hcLevel > 0)
                ? LZ4_compress_HC(reinterpret_cast<const char*>(src),
                                  reinterpret_cast<char*>(compressed.data()),
                                  (int)origSz, (int)maxOut, hcLevel)
                : LZ4_compress_default(reinterpret_cast<const char*>(src),
                                       reinterpret_cast<char*>(compressed.data()),
                                       (int)origSz, (int)maxOut);
            std::vector<uint8_t> origVec(src, src + origSz);
            std::vector<uint8_t> compVec;
            if (compSz > 0 && (size_t)compSz < origSz) {
                compressed.resize(compSz);
                compVec = std::move(compressed);
            }
            std::vector<std::vector<uint8_t>> cBatch = {std::move(compVec)};
            std::vector<std::vector<uint8_t>> oBatch = {std::move(origVec)};
            std::vector<size_t> idxBatch = {chunk.chunkIndex};
            std::vector<size_t>  szBatch = {origSz};
            asyncWriter.enqueueBatch(cBatch, oBatch, idxBatch, szBatch);
            cpuChunkCount++;
            chunksSubmitted++;
        };

        auto cpuWorker = [&]() {
            // ── Phase 1: GPU slot init window ─────────────────────────────────
            // While totalGpuSlots==0 GPU cudaMalloc hasn't finished.
            // Pull directly from the reader and compress immediately.
            while (!workerAbort.load() &&
                   totalGpuSlots.load(std::memory_order_acquire) == 0) {
                AsyncReader::ReadChunk chunk;
                if (!asyncReader.getChunk(chunk)) return;
                compressAndSend(chunk);
            }
            // ── Phase 2: normal  drain cpuWorkQueue ──────────────────────────
            while (!workerAbort.load()) {
                AsyncReader::ReadChunk chunk;
                if (!cpuWorkQueue.pop(chunk)) break;
                compressAndSend(chunk);
            }
        };

        // Launch CPU workers BEFORE slot init join so Phase 1 runs during GPU init.
        std::vector<std::thread> allWorkers;
        allWorkers.reserve(gpus.size() + effectiveThreads);
        for (size_t t = 0; t < effectiveThreads; t++)
            allWorkers.emplace_back(cpuWorker);

        for (auto& t : slotInitThreads) t.join();

        bool slotsOk = true;
        for (bool ok : gpuInitOk) if (!ok) { slotsOk = false; break; }

        // Count total GPU slots and publish atomically  dispatcher transitions
        // from CPU-only routing (totalGpuSlots==0) to normal routing.
        size_t slotCount = 0;
        for (auto& gSlots : gpuSlots) slotCount += gSlots.size();
        gpuFreeSlots.store(slotCount, std::memory_order_release);
        totalGpuSlots.store(slotCount, std::memory_order_release);
        VLOG(DEBUG, "Total GPU slots: %zu across %zu GPUs\n", slotCount, gpus.size());

        if (!slotsOk) {
            fprintf(stderr,"GPU slot init failed, falling back to CPU-only\n");
            workerAbort.store(true);
            cpuWorkQueue.close(); gpuWorkQueue.close();
            for (auto& w : allWorkers) if (w.joinable()) w.join();
            freeAllSlots(); asyncWriter.stop(); asyncReader.stop();
            return compressFileCPU();
        }

        std::atomic<bool> dispatcherDone{false};
        std::thread dispatcherThread([&]() {
            while (true) {
                AsyncReader::ReadChunk chunk;
                if (!asyncReader.getChunk(chunk)) break;
                if (isAllZeros(chunk.data(), chunk.size)) {
                    // Zero chunks always go to CPU  trivial to compress there
                    cpuWorkQueue.push(std::move(chunk));
                } else if (totalGpuSlots.load(std::memory_order_acquire) == 0) {
                    // GPU slots not yet allocated (still in cudaMalloc) 
                    // send everything to CPU so it starts compressing immediately
                    // while GPUs initialize.  Same Phase 1 overlap as decompressor.
                    cpuWorkQueue.push(std::move(chunk));
                } else if (gpuFreeSlots.load(std::memory_order_acquire) == 0) {
                    // All GPU streams are in-flight  overflow to CPU so it
                    // doesn't sit idle while waiting for a GPU slot to free up.
                    cpuWorkQueue.push(std::move(chunk));
                } else {
                    gpuWorkQueue.push(std::move(chunk));
                }
            }
            gpuWorkQueue.close();
            cpuWorkQueue.close();
            dispatcherDone.store(true);
        });

        auto gpuWorker = [&](size_t gpuIdx) {
            GPUDevice& gpu = gpus[gpuIdx];
            cudaSetDevice(gpu.deviceId);
            std::vector<PreallocSlot>& slots = gpuSlots[gpuIdx];
            const int nSlots = (int)slots.size();
            size_t sIdx = 0;
            bool firstRound = true;

            while (!workerAbort.load()) {
                PreallocSlot& slot = slots[sIdx];
                sIdx = (sIdx + 1) % nSlots;
                cudaStreamSynchronize(slot.stream);
                if (!firstRound && slot.hasPending) {
                    std::vector<std::vector<uint8_t>> cChunks, oChunks;
                    cChunks.reserve(slot.batchSize);
                    oChunks.reserve(slot.batchSize);
                    for (size_t i = 0; i < slot.batchSize; i++) {
                        size_t outSz  = slot.h_oSizes[i];
                        size_t origSz = slot.origSizes[i];
                        const uint8_t* origPtr = slot.origHandles.size() > i
                            ? slot.origHandles[i].data : slot.origData[i].data();
                        if (outSz < origSz) {
                            std::vector<uint8_t> buf(outSz);
                            memcpy(buf.data(), slot.h_output + i * slot.outStride, outSz);
                            cChunks.push_back(std::move(buf));
                        } else {
                            cChunks.push_back({});
                        }
                        oChunks.emplace_back(origPtr, origPtr + origSz);
                    }
                    slot.origHandles.clear(); slot.origData.clear();
                    asyncWriter.enqueueBatch(cChunks, oChunks, slot.indices, slot.origSizes);
                    gpuChunkCount += slot.batchSize;
                    chunksSubmitted += slot.batchSize;
                    slot.hasPending = false;
                }
                firstRound = false;

                slot.indices.clear(); slot.origSizes.clear();
                slot.origData.clear(); slot.origHandles.clear();
                slot.batchSize = 0;

                // This slot is now free  signal the dispatcher so it knows
                // at least one GPU stream is available and can route to GPU
                // instead of overflowing to CPU.
                gpuFreeSlots.fetch_add(1, std::memory_order_release);

                while (slot.batchSize < slot.capacity) {
                    AsyncReader::ReadChunk chunk;
                    if (!gpuWorkQueue.pop(chunk)) break;  // closed + empty
                    cudaMemcpyAsync(
                        slot.d_input + slot.batchSize * slot.chunkStride,
                        chunk.data(), chunk.size,
                        cudaMemcpyHostToDevice, slot.stream);
                    slot.h_iSizes[slot.batchSize] = chunk.size;
                    slot.indices.push_back(chunk.chunkIndex);
                    slot.origSizes.push_back(chunk.size);
                    if (chunk.poolHandle.valid())
                        slot.origHandles.push_back(std::move(chunk.poolHandle));
                    else
                        slot.origData.push_back(std::move(chunk.heapData));
                    slot.batchSize++;
                }
                if (slot.batchSize == 0) break;

                cudaMemcpyAsync(slot.d_iSizes, slot.h_iSizes,
                                slot.batchSize * sizeof(size_t),
                                cudaMemcpyHostToDevice, slot.stream);
                nvcompStatus_t nErr = nvcompBatchedLZ4CompressAsync(
                    slot.d_iPtrs, slot.d_iSizes, slot.chunkStride,
                    slot.batchSize, slot.d_temp, slot.tempBytes,
                    slot.d_oPtrs, slot.d_oSizes, nvOpts, slot.d_stats, slot.stream);
                if (nErr != nvcompSuccess) {
                    fprintf(stderr, "GPU%d: nvcomp error %d\n", gpu.deviceId, (int)nErr);
                    workerAbort.store(true); break;
                }
                // Slot is now in-flight  one fewer free slot available.
                gpuFreeSlots.fetch_sub(1, std::memory_order_release);
                cudaMemcpyAsync(slot.h_output, slot.d_output,
                                slot.batchSize * slot.outStride,
                                cudaMemcpyDeviceToHost, slot.stream);
                cudaMemcpyAsync(slot.h_oSizes, slot.d_oSizes,
                                slot.batchSize * sizeof(size_t),
                                cudaMemcpyDeviceToHost, slot.stream);
                cudaMemcpyAsync(slot.h_stats, slot.d_stats,
                                slot.batchSize * sizeof(nvcompStatus_t),
                                cudaMemcpyDeviceToHost, slot.stream);
                slot.hasPending = true;
            }

            for (int si = 0; si < nSlots; si++) {
                PreallocSlot& sl = slots[si];
                if (!sl.hasPending) continue;
                cudaStreamSynchronize(sl.stream);
                std::vector<std::vector<uint8_t>> cChunks, oChunks;
                for (size_t i = 0; i < sl.batchSize; i++) {
                    size_t outSz  = sl.h_oSizes[i];
                    size_t origSz = sl.origSizes[i];
                    const uint8_t* origPtr = sl.origHandles.size() > i
                        ? sl.origHandles[i].data : sl.origData[i].data();
                    if (outSz < origSz) {
                        std::vector<uint8_t> buf(outSz);
                        memcpy(buf.data(), sl.h_output + i * sl.outStride, outSz);
                        cChunks.push_back(std::move(buf));
                    } else {
                        cChunks.push_back({});
                    }
                    oChunks.emplace_back(origPtr, origPtr + origSz);
                }
                sl.origHandles.clear(); sl.origData.clear();
                asyncWriter.enqueueBatch(cChunks, oChunks, sl.indices, sl.origSizes);
                gpuChunkCount += sl.batchSize;
                chunksSubmitted += sl.batchSize;
            }
        };

        // GPU workers launched after slot init  they need their slots ready.
        // CPU workers were already launched before slotInitThreads.join().
        for (size_t g = 0; g < gpus.size(); g++)
            allWorkers.emplace_back(gpuWorker, g);

        while (stdinMode ? !dispatcherDone.load() : (chunksSubmitted.load() < numChunks)) {
            // Abort if writer hit an unrecoverable error
            if (asyncWriter.hasWriteError()) {
                workerAbort.store(true);
                gpuWorkQueue.close();
                cpuWorkQueue.close();
                break;
            }
            if (g_verbosity != QUIET && (stdinMode || numChunks > 10)) {
                size_t done = asyncWriter.getNextChunkToWrite();
                std::string gpuStr = formatBytes(gpuChunkCount.load() * chunkSize);
                std::string cpuStr = formatBytes(cpuChunkCount.load() * chunkSize);
                if (stdinMode) {
                    fprintf(stderr, "\r%sCompressing:%s  %sGPU:%s %s%s%s  %sCPU:%s %s%s%s%s",
                            CC_BCYAN, CC_RESET,
                            CC_CYAN, CC_RESET, CC_GREEN, gpuStr.c_str(), CC_RESET,
                            CC_CYAN, CC_RESET, CC_BLUE,  cpuStr.c_str(), CC_RESET, CC_EL);
                } else {
                    fprintf(stderr, "\r%sCompressing:%s %s%3zu%%%s  %sGPU:%s %s%s%s  %sCPU:%s %s%s%s%s",
                            CC_BCYAN, CC_RESET,
                            CC_BYELLOW, (100 * done) / numChunks, CC_RESET,
                            CC_CYAN, CC_RESET, CC_GREEN, gpuStr.c_str(), CC_RESET,
                            CC_CYAN, CC_RESET, CC_BLUE,  cpuStr.c_str(), CC_RESET, CC_EL);
                }
                fflush(stderr);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        dispatcherThread.join();
        for (auto& t : allWorkers) t.join();

        if (stdinMode) asyncWriter.setTotalChunks(chunksSubmitted.load());

        {
            std::atomic<bool> stopProgress{false};
            std::thread progressThread;
            if (g_verbosity != QUIET && (stdinMode || numChunks > 10)) {
                progressThread = std::thread([&]() {
                    g_progressActive.store(true, std::memory_order_relaxed);
                    while (!stopProgress.load()) {
                        size_t w = asyncWriter.getNextChunkToWrite();
                        size_t bytesWritten = w * chunkSize;
                        if (!stdinMode && bytesWritten > fileSize) bytesWritten = fileSize;
                        std::string written = formatBytes(bytesWritten);
                        if (stdinMode) {
                            VLOG(NORMAL, "\r%sWriting:%s  %s%s%s%s",
                                    CC_BGREEN, CC_RESET,
                                    CC_BGREEN, written.c_str(), CC_RESET, CC_EL);
                        } else {
                            std::string total = formatBytes(fileSize);
                            VLOG(NORMAL, "\r%sWriting:%s       %s%3d%%%s  %s[%s %s%s%s / %s%s%s ]%s%s",
                                    CC_BGREEN, CC_RESET,
                                    CC_BYELLOW, (int)(100 * w / numChunks), CC_RESET,
                                    CC_DIM, CC_RESET,
                                    CC_BGREEN, written.c_str(), CC_RESET,
                                    CC_WHITE, total.c_str(), CC_RESET,
                                    CC_DIM, CC_EL);
                        }
                        fflush(stderr);
                        std::this_thread::sleep_for(std::chrono::milliseconds(150));
                    }
                    if (stdinMode) {
                        std::string fb = formatBytes((size_t)asyncWriter.getBytesWritten());
                        VLOG(NORMAL, "\r%sWriting:%s  %s%s%s%s\n",
                                CC_BGREEN, CC_RESET, CC_BGREEN, fb.c_str(), CC_RESET, CC_EL);
                    } else {
                        std::string total = formatBytes(fileSize);
                        VLOG(NORMAL, "\r%sWriting:%s       %s100%%%s  %s[%s %s%s%s / %s%s%s ]%s\n",
                                CC_BGREEN, CC_RESET,
                                CC_BYELLOW, CC_RESET,
                                CC_DIM, CC_RESET,
                                CC_BGREEN, total.c_str(), CC_RESET,
                                CC_WHITE, total.c_str(), CC_RESET,
                                CC_DIM);
                    }
                });
            }

            VLOG(VERBOSE, "Waiting for async writer to complete...\n");
            asyncWriter.stop();

            stopProgress.store(true);
            if (progressThread.joinable()) progressThread.join();
            g_progressActive.store(false, std::memory_order_relaxed);
        }

        // Write end mark and checksum
        {
            int fd = stdoutMode ? STDOUT_FILENO
                                : open(getActualOutputPath(), O_WRONLY | O_APPEND);
            if (fd >= 0) {
                uint32_t endMark = 0;
                if (::write(fd, &endMark, 4) != 4)
                    fprintf(stderr, "Error writing end mark\n");
                uint32_t cs = xxhState.digest();
                uint8_t cb[4] = { (uint8_t)(cs), (uint8_t)(cs>>8),
                                  (uint8_t)(cs>>16), (uint8_t)(cs>>24) };
                if (::write(fd, cb, 4) != 4)
                    fprintf(stderr, "Error writing checksum\n");
                if (!stdoutMode) {
                    if (syncOutput) fsync(fd);
                    close(fd);
                }
                VLOG(VERBOSE, "Content checksum: 0x%08X\n", cs);
            }
        }

        asyncReader.stop();
        freeAllSlots();

        auto endTime  = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        double bw   = asyncWriter.getBytesWritten();
        double mbps = duration.count() > 0
            ? (stdinMode ? bw : (double)fileSize) / 1048576.0 / (duration.count()/1000.0)
            : 0.0;

        std::string inputSize  = stdinMode ? "(stdin)" : formatBytes(fileSize);
        std::string outputSize = formatBytes((size_t)bw);
        if (stdinMode) {
            VLOG(NORMAL, "\r%sCompression complete:%s stdin -> %s%s%s in %.2f s\n",
                    CC_BGREEN, CC_RESET,
                    CC_CYAN, outputSize.c_str(), CC_RESET,
                    duration.count()/1000.0);
        } else {
            VLOG(NORMAL, "\r%sCompression complete:%s %s%s%s -> %s%s%s %s(%.2f%%)%s in %.2f s\n",
                    CC_BGREEN, CC_RESET,
                    CC_WHITE, inputSize.c_str(), CC_RESET,
                    CC_CYAN, outputSize.c_str(), CC_RESET,
                    CC_BYELLOW, 100.0*bw/fileSize, CC_RESET,
                    duration.count()/1000.0);
        }
        VLOG(VERBOSE, "Throughput: %.2f MB/s\n", mbps);
        size_t totalDone = gpuChunkCount.load() + cpuChunkCount.load();
        VLOG(VERBOSE, "  GPU: %zu chunks (%.1f%%)  CPU: %zu chunks (%.1f%%)\n",
             gpuChunkCount.load(),
             totalDone > 0 ? 100.0*gpuChunkCount.load()/totalDone : 0.0,
             cpuChunkCount.load(),
             totalDone > 0 ? 100.0*cpuChunkCount.load()/totalDone : 0.0);
        VLOG(VERBOSE, "  Write: %.2f s\n", asyncWriter.getWriteTime());

        if (!keepOriginal && !stdoutMode) {
            if (unlink(inputFile.c_str()) != 0)
                fprintf(stderr, "Warning: Could not remove input file: %s\n",
                        inputFile.c_str());
        }
        return true;
    }

    bool decompressBlockGPU(const uint8_t* compData, size_t compSize,
                            uint8_t* outData, size_t outSize,
                            GPUDevice& gpu, cudaStream_t& stream,
                            size_t& actualSize) {
        if (cudaSetDevice(gpu.deviceId) != cudaSuccess) return false;
        void *d_comp=nullptr, *d_decomp=nullptr, **d_inPtrs=nullptr, **d_outPtrs=nullptr;
        size_t *d_inSizes=nullptr, *d_outBufSizes=nullptr, *d_actualSizes=nullptr;
        nvcompStatus_t* d_statuses=nullptr;
        void* d_temp=nullptr;
        auto cleanup = [&]() {
            if (d_temp)        cudaFree(d_temp);
            if (d_statuses)    cudaFree(d_statuses);
            if (d_actualSizes) cudaFree(d_actualSizes);
            if (d_outBufSizes) cudaFree(d_outBufSizes);
            if (d_inSizes)     cudaFree(d_inSizes);
            if (d_outPtrs)     cudaFree(d_outPtrs);
            if (d_inPtrs)      cudaFree(d_inPtrs);
            if (d_decomp)      cudaFree(d_decomp);
            if (d_comp)        cudaFree(d_comp);
        };
        if (cudaMalloc(&d_comp,        compSize)               != cudaSuccess ||
            cudaMalloc(&d_decomp,      outSize)                != cudaSuccess ||
            cudaMalloc(&d_inPtrs,      sizeof(void*))          != cudaSuccess ||
            cudaMalloc(&d_outPtrs,     sizeof(void*))          != cudaSuccess ||
            cudaMalloc(&d_inSizes,     sizeof(size_t))         != cudaSuccess ||
            cudaMalloc(&d_outBufSizes, sizeof(size_t))         != cudaSuccess ||
            cudaMalloc(&d_actualSizes, sizeof(size_t))         != cudaSuccess ||
            cudaMalloc(&d_statuses,    sizeof(nvcompStatus_t)) != cudaSuccess) {
            cleanup(); return false;
        }
        const void* h_inPtr  = d_comp;
        void*       h_outPtr = d_decomp;
        cudaMemcpy(d_comp,        compData,    compSize,       cudaMemcpyHostToDevice);
        cudaMemcpy(d_inPtrs,      &h_inPtr,    sizeof(void*),  cudaMemcpyHostToDevice);
        cudaMemcpy(d_outPtrs,     &h_outPtr,   sizeof(void*),  cudaMemcpyHostToDevice);
        cudaMemcpy(d_inSizes,     &compSize,   sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_outBufSizes, &outSize,    sizeof(size_t), cudaMemcpyHostToDevice);
        nvcompBatchedLZ4DecompressOpts_t opts = nvcompBatchedLZ4DecompressDefaultOpts;
        size_t tempBytes = outSize;
        if (cudaMalloc(&d_temp, tempBytes) != cudaSuccess) { cleanup(); return false; }
        nvcompStatus_t st = nvcompBatchedLZ4DecompressAsync(
            (const void* const*)d_inPtrs, d_inSizes, d_outBufSizes, d_actualSizes,
            (size_t)1, d_temp, tempBytes,
            (void* const*)d_outPtrs, opts, d_statuses, stream);
        if (st != nvcompSuccess) { cleanup(); return false; }
        cudaStreamSynchronize(stream);
        nvcompStatus_t itemSt = nvcompSuccess;
        cudaMemcpy(&itemSt, d_statuses, sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost);
        if (itemSt != nvcompSuccess) { cleanup(); return false; }
        size_t actualOut = 0;
        cudaMemcpy(&actualOut, d_actualSizes, sizeof(size_t), cudaMemcpyDeviceToHost);
        if (actualOut == 0 || actualOut > outSize) { cleanup(); return false; }
        cudaMemcpy(outData, d_decomp, actualOut, cudaMemcpyDeviceToHost);
        actualSize = actualOut;
        VLOG(VERY_VERBOSE, "decompressBlockGPU: ok, actualSize=%zu\n", actualSize);
        cleanup();
        return true;
    }

    bool decompressFileGPU() {
        if (testMode) {
            VLOG(NORMAL, "Testing (GPU+fallback, %zu GPU%s): %s\n",
                    gpus.size(), gpus.size() == 1 ? "" : "s", inputFile.c_str());
        } else {
            VLOG(NORMAL, "%sDecompressing%s (GPU+fallback, %zu GPU%s): %s -> %s\n",
                    CC_BCYAN, CC_RESET,
                    gpus.size(), gpus.size() == 1 ? "" : "s",
                    inputFile.c_str(), outputFile.c_str());
        }

        const bool stdinMode = (inputFile == "-");
        int inputFd = stdinMode ? STDIN_FILENO : open(inputFile.c_str(), O_RDONLY);
        if (inputFd < 0) {
            fprintf(stderr, "Error: Cannot open input file: %s\n", inputFile.c_str());
            return false;
        }
        if (!stdinMode) {
            posix_fadvise(inputFd, 0, 0, POSIX_FADV_SEQUENTIAL);
            posix_fadvise(inputFd, 0, 0, POSIX_FADV_WILLNEED);
        }

        uint8_t headerBuf[32];
        ssize_t headerRead = ::read(inputFd, headerBuf, 32);
        if (headerRead < 15) {
            fprintf(stderr, "Error: File too small to be valid LZ4\n");
            if (inputFd != STDIN_FILENO) close(inputFd);
            return false;
        }
        LZ4Frame::FrameDescriptor desc;
        size_t headerBytes = 0;
        {
            std::string headerStr((char*)headerBuf, headerRead);
            std::istringstream headerStream(headerStr, std::ios::binary);
            if (!LZ4Frame::readFrameHeader(headerStream, desc)) {
                fprintf(stderr, "Error: Failed to read LZ4 frame header\n");
                if (inputFd != STDIN_FILENO) close(inputFd);
                return false;
            }
            headerBytes = headerStream.tellg();
        }
        if (lseek(inputFd, (off_t)headerBytes, SEEK_SET) == (off_t)-1) {
            fprintf(stderr, "Error: Failed to seek past header: %s\n", strerror(errno));
            if (inputFd != STDIN_FILENO) close(inputFd);
            return false;
        }
        VLOG(DEBUG, "LZ4 header consumed %zu bytes, seeking fd to byte %zu\n",
             headerBytes, headerBytes);

        size_t originalFileSize = desc.contentSize;
        size_t chunkSize = static_cast<size_t>(1) << (8 + 2 * desc.blockMaxSize);
        size_t estimatedBlocks = (originalFileSize + chunkSize - 1) / chunkSize;

        VLOG(DEBUG, "[GPU decomp] header parsed: origSize=%.2f GB  chunkSize=%zu KB  ~%zu blocks\n",
             originalFileSize / (1024.0*1024.0*1024.0), chunkSize / 1024, estimatedBlocks);

        size_t compressedFileSize = 0;
        {
            struct stat cst;
            if (fstat(inputFd, &cst) == 0 && (size_t)cst.st_size > 0) {
                compressedFileSize = (size_t)cst.st_size;
                // Only use the compressed-file-size fallback when the header
                // did not provide a valid contentSize (estimatedBlocks == 0 or 1).
                // The fallback formula divides by the LZ4 block frame minimum:
                // 4-byte size header + at least chunkSize/256 bytes of data
                // (LZ4 worst-case expansion is < 0.4%, so even incompressible
                // data compresses to ≥ 99.6% of input  but we use chunkSize/256
                // as a very conservative floor so we don't over-allocate).
                // Never let the fallback shrink a valid contentSize estimate.
                if (estimatedBlocks <= 1 && compressedFileSize > 0) {
                    size_t minBlockBytes = 4 + chunkSize / 256 + 1;
                    size_t compBlocks    = compressedFileSize / minBlockBytes + 1;
                    if (compBlocks > estimatedBlocks) {
                        VLOG(VERBOSE, "  contentSize unknown  using compressed-file "
                             "estimate: %zu chunks\n", compBlocks);
                        estimatedBlocks = compBlocks;
                    }
                }
            }
        }
        if (estimatedBlocks == 0) estimatedBlocks = 1;

        VLOG(VERBOSE, "  %.2f MB  |  chunk size %zu KB  |  ~%zu chunks\n",
             originalFileSize / (1024.0*1024.0), chunkSize / 1024, estimatedBlocks);

        XXH::State xxhState(XXH32_SEED);
        std::atomic<size_t> cksumConsumed{0};

        int outputFd = -1;
        bool outIsPipe = false;
        if (!testMode) {
            if (stdoutMode) {
                outputFd = STDOUT_FILENO;
                outIsPipe = true;
            } else {
                outputFd = open(getActualOutputPath(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
                if (outputFd < 0) {
                    fprintf(stderr, "Error: Cannot create output file: %s\n", getActualOutputPath());
                    if (inputFd != STDIN_FILENO) close(inputFd);
                    return false;
                }
                posix_fadvise(outputFd, 0, 0, POSIX_FADV_SEQUENTIAL);
                outIsPipe = (lseek(outputFd, 0, SEEK_CUR) < 0);
            }
#ifdef F_SETPIPE_SZ
            if (outIsPipe) fcntl(outputFd, F_SETPIPE_SZ, 1 << 20);
#endif
        }

        struct DecompBlock {
            std::vector<uint8_t> data;
            bool gpuPath = false;
        };

        std::vector<DecompBlock>          results(estimatedBlocks);
        std::vector<std::atomic<uint8_t>> ready(estimatedBlocks);
        for (auto& fl : ready) fl.store(0, std::memory_order_relaxed);

        std::mutex              resultMutex;
        std::condition_variable resultCV;
        auto growResults = [&](size_t needed) {
            if (needed < results.size()) return;
            std::lock_guard<std::mutex> lk(resultMutex);
            if (needed < results.size()) return;
            size_t newSz = std::max(needed + 1, results.size() * 2);
            results.resize(newSz);
            std::vector<std::atomic<uint8_t>> newReady(newSz);
            for (size_t i = 0; i < ready.size(); i++)
                newReady[i].store(ready[i].load(std::memory_order_relaxed),
                                  std::memory_order_relaxed);
            ready = std::move(newReady);
            VLOG(VERBOSE, "  growResults: resized results/ready to %zu\n", newSz);
        };
        std::atomic<size_t>     blocksQueued{0};
        std::atomic<size_t>     blocksDone{0};
        std::atomic<bool>       readDone{false};
        std::atomic<bool>       decompError{false};
        std::atomic<size_t>     gpuBlocks{0}, cpuFallbackBlocks{0};
        std::atomic<size_t>     totalBlocks{0};

        struct RawBlock {
            size_t idx;
            std::vector<uint8_t> compData;
            std::vector<uint8_t> rawData;
            size_t origSize;
        };
        std::queue<RawBlock>    blockQueue;
        std::mutex              blockQueueMutex;
        std::condition_variable blockQueueCV;

        size_t N_STREAMS;
        if (pipelineDepth > 0) {
            N_STREAMS = (size_t)pipelineDepth;
            VLOG(DEBUG, "N_STREAMS = %zu (user --streams-per-gpu)\n", N_STREAMS);
        } else {
            const size_t maxComp_g     = chunkSize + (chunkSize / 255) + 16;
            const size_t perStreamVRAM = slotCapacity * (maxComp_g + chunkSize + chunkSize);
            const size_t perStreamPin  = slotCapacity * chunkSize;
            const size_t pinnedCap     = 4ULL * 1024 * 1024 * 1024;
            size_t freeVRAM    = gpus[0].availableMemory;
            size_t autoByVRAM  = (freeVRAM / 2) / perStreamVRAM;
            if (autoByVRAM < 1)  autoByVRAM = 1;
            if (autoByVRAM > 32) autoByVRAM = 32;
            size_t autoByPin = pinnedCap / perStreamPin;
            if (autoByPin < 1)  autoByPin = 1;
            if (autoByPin > 32) autoByPin = 32;
            N_STREAMS = std::min(autoByVRAM, autoByPin);
            VLOG(DEBUG, "N_STREAMS auto: freeVRAM=%.1f GB  perStreamVRAM=%.0f MB  "
                 "perStreamPin=%.0f MB  byVRAM=%zu  byPin=%zu  -> %zu\n",
                 freeVRAM / (1024.0*1024.0*1024.0),
                 perStreamVRAM / (1024.0*1024.0),
                 perStreamPin  / (1024.0*1024.0),
                 autoByVRAM, autoByPin, N_STREAMS);
        }
        VLOG(VERBOSE, "GPU decompressor: %zu streams/GPU%s  ×  %zu chunks/slot  =  %.1f GB VRAM/slot\n",
             N_STREAMS, pipelineDepth > 0 ? " (user-specified)" : " (auto)",
             slotCapacity,
             (slotCapacity * (chunkSize + (chunkSize/255+16) + chunkSize)) / (1024.0*1024.0*1024.0));

        struct DecompSlot {
            cudaStream_t    stream         = nullptr;
            uint8_t*        d_comp         = nullptr;
            uint8_t*        d_decomp       = nullptr;
            void*           d_temp         = nullptr;
            void**          d_inPtr        = nullptr;
            void**          d_outPtr       = nullptr;
            size_t*         d_inSize       = nullptr;
            size_t*         d_outBufSize   = nullptr;
            size_t*         d_actualSize   = nullptr;
            nvcompStatus_t* d_status       = nullptr;
            size_t*         h_inSizes      = nullptr;
            size_t*         h_actualSize   = nullptr;
            nvcompStatus_t* h_status       = nullptr;
            uint8_t*        h_decomp       = nullptr;
            bool            inFlight       = false;
            size_t          batchCount     = 0;
            std::vector<size_t> blockIdxs;
            size_t          origSize       = 0;
        };

        auto gpuWorker = [&](size_t gpuIdx) {
            GPUDevice& gpu = gpus[gpuIdx];
            auto wt0 = std::chrono::high_resolution_clock::now();
            auto wElapsed = [&]() {
                return std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - wt0).count() * 1000.0;
            };

            VLOG(DEBUG, "[GPU%d decompWorker] started, calling cudaSetDevice\n", (int)gpuIdx);
            cudaSetDevice(gpu.deviceId);
            VLOG(DEBUG, "[GPU%d decompWorker] cudaSetDevice done (%.1f ms)\n",
                 (int)gpuIdx, wElapsed());
            // Refresh availableMemory with actual free VRAM now that context exists.
            // enumerateGPUs() used totalGlobalMem as a proxy since it avoids cudaSetDevice.
            { size_t fr, tot; if (cudaMemGetInfo(&fr, &tot) == cudaSuccess) gpu.availableMemory = fr; }

            const size_t maxCompSize = chunkSize + (chunkSize / 255) + 16;
            const size_t tempBytes   = chunkSize;
            nvcompBatchedLZ4DecompressOpts_t opts = nvcompBatchedLZ4DecompressDefaultOpts;
            const size_t SC = slotCapacity;
            std::vector<DecompSlot> slots(N_STREAMS);
            size_t allocatedSlots = 0;
            for (auto& sl : slots) {
                if (cudaStreamCreate(&sl.stream)                                        != cudaSuccess) break;
                if (cudaMalloc(&sl.d_comp,       SC * maxCompSize)                     != cudaSuccess) break;
                if (cudaMalloc(&sl.d_decomp,     SC * chunkSize)                       != cudaSuccess) break;
                if (cudaMalloc(&sl.d_temp,       SC * tempBytes)                       != cudaSuccess) break;
                if (cudaMalloc(&sl.d_inPtr,      SC * sizeof(void*))                   != cudaSuccess) break;
                if (cudaMalloc(&sl.d_outPtr,     SC * sizeof(void*))                   != cudaSuccess) break;
                if (cudaMalloc(&sl.d_inSize,     SC * sizeof(size_t))                  != cudaSuccess) break;
                if (cudaMalloc(&sl.d_outBufSize, SC * sizeof(size_t))                  != cudaSuccess) break;
                if (cudaMalloc(&sl.d_actualSize, SC * sizeof(size_t))                  != cudaSuccess) break;
                if (cudaMalloc(&sl.d_status,     SC * sizeof(nvcompStatus_t))          != cudaSuccess) break;
                if (cudaMallocHost(&sl.h_inSizes,    SC * sizeof(size_t))              != cudaSuccess) break;
                if (cudaMallocHost(&sl.h_actualSize, SC * sizeof(size_t))              != cudaSuccess) break;
                if (cudaMallocHost(&sl.h_status,     SC * sizeof(nvcompStatus_t))      != cudaSuccess) break;
                if (cudaMallocHost(&sl.h_decomp,     SC * chunkSize)                   != cudaSuccess) break;
                std::vector<void*>  hInPtrs(SC), hOutPtrs(SC);
                std::vector<size_t> hOutBuf(SC, chunkSize);
                for (size_t i = 0; i < SC; i++) {
                    hInPtrs[i]  = sl.d_comp   + i * maxCompSize;
                    hOutPtrs[i] = sl.d_decomp + i * chunkSize;
                }
                cudaMemcpy(sl.d_inPtr,      hInPtrs.data(),  SC * sizeof(void*),  cudaMemcpyHostToDevice);
                cudaMemcpy(sl.d_outPtr,     hOutPtrs.data(), SC * sizeof(void*),  cudaMemcpyHostToDevice);
                cudaMemcpy(sl.d_outBufSize, hOutBuf.data(),  SC * sizeof(size_t), cudaMemcpyHostToDevice);
                allocatedSlots++;
            }
            if (allocatedSlots == 0) {
                fprintf(stderr, "GPU%zu: failed to allocate any decompression slots\n", gpuIdx);
                decompError = true;
                goto cleanup_slots;
            }
            VLOG(VERBOSE, "GPU%zu: %zu streams x %zu chunks/batch  (slot alloc %.1f ms)\n",
                 gpuIdx, allocatedSlots, SC, wElapsed());
            VLOG(DEBUG, "[GPU%d decompWorker] slot allocation done (%.1f ms), entering work loop\n",
                 (int)gpuIdx, wElapsed());
            {
                size_t nextSlot = 0;
                std::vector<std::vector<std::vector<uint8_t>>>
                    slotCompData(allocatedSlots, std::vector<std::vector<uint8_t>>(SC));

                auto collectSlot = [&](size_t si) -> bool {
                    DecompSlot& sl = slots[si];
                    if (!sl.inFlight) return true;
                    cudaStreamSynchronize(sl.stream);
                    sl.inFlight = false;
                    for (size_t j = 0; j < sl.batchCount; j++) {
                        nvcompStatus_t st = sl.h_status[j];
                        size_t actualOut  = sl.h_actualSize[j];
                        DecompBlock out;
                        if (st != nvcompSuccess || actualOut == 0) {
                            VLOG(VERBOSE, "Block %zu: nvCOMP status %d, CPU fallback\n",
                                 sl.blockIdxs[j], (int)st);
                            out.data.resize(sl.origSize);
                            int r = LZ4_decompress_safe(
                                reinterpret_cast<const char*>(slotCompData[si][j].data()),
                                reinterpret_cast<char*>(out.data.data()),
                                (int)slotCompData[si][j].size(), (int)sl.origSize);
                            if (r < 0) {
                                fprintf(stderr, "Error: block %zu GPU st=%d CPU fallback also failed\n",
                                        sl.blockIdxs[j], (int)st);
                                decompError = true; return false;
                            }
                            out.data.resize(r);
                            out.gpuPath = false;
                            cpuFallbackBlocks++;
                        } else {
                            out.data.resize(actualOut);
                            memcpy(out.data.data(), sl.h_decomp + j * chunkSize, actualOut);
                            out.gpuPath = true;
                            gpuBlocks++;
                        }
                        slotCompData[si][j].clear();
                        results[sl.blockIdxs[j]] = std::move(out);
                        ready[sl.blockIdxs[j]].store(1, std::memory_order_release);
                        blocksDone++;
                        resultCV.notify_one();
                    }
                    return true;
                };

                auto dispatchSlot = [&](size_t si, std::vector<RawBlock>& batch) {
                    DecompSlot& sl   = slots[si];
                    sl.batchCount    = batch.size();
                    sl.origSize      = batch[0].origSize;
                    sl.blockIdxs.resize(sl.batchCount);
                    for (size_t j = 0; j < sl.batchCount; j++) {
                        sl.blockIdxs[j]    = batch[j].idx;
                        size_t csz         = batch[j].compData.size();
                        sl.h_inSizes[j]    = csz;
                        slotCompData[si][j] = std::move(batch[j].compData);
                        cudaMemcpyAsync(sl.d_comp + j * maxCompSize,
                                        slotCompData[si][j].data(), csz,
                                        cudaMemcpyHostToDevice, sl.stream);
                    }
                    cudaMemcpyAsync(sl.d_inSize, sl.h_inSizes,
                                    sl.batchCount * sizeof(size_t),
                                    cudaMemcpyHostToDevice, sl.stream);
                    nvcompStatus_t apiSt = nvcompBatchedLZ4DecompressAsync(
                        (const void* const*)sl.d_inPtr,
                        sl.d_inSize, sl.d_outBufSize, sl.d_actualSize,
                        sl.batchCount, sl.d_temp, SC * tempBytes,
                        (void* const*)sl.d_outPtr, opts, sl.d_status, sl.stream);
                    if (apiSt != nvcompSuccess) {
                        for (size_t j = 0; j < sl.batchCount; j++) sl.h_status[j] = apiSt;
                        cudaStreamSynchronize(sl.stream);
                    } else {
                        cudaMemcpyAsync(sl.h_actualSize, sl.d_actualSize,
                                        sl.batchCount * sizeof(size_t),
                                        cudaMemcpyDeviceToHost, sl.stream);
                        cudaMemcpyAsync(sl.h_status, sl.d_status,
                                        sl.batchCount * sizeof(nvcompStatus_t),
                                        cudaMemcpyDeviceToHost, sl.stream);
                        cudaMemcpyAsync(sl.h_decomp, sl.d_decomp,
                                        sl.batchCount * chunkSize,
                                        cudaMemcpyDeviceToHost, sl.stream);
                    }
                    sl.inFlight = true;
                };

                while (!decompError) {
                    std::vector<RawBlock> batch;
                    batch.reserve(SC);
                    bool reachedEOF = false;
                    while (batch.size() < SC && !decompError) {
                        RawBlock block;
                        {
                            std::unique_lock<std::mutex> lk(blockQueueMutex);
                            blockQueueCV.wait(lk, [&]{
                                return !blockQueue.empty() || (readDone && blockQueue.empty());
                            });
                            if (blockQueue.empty()) { reachedEOF = true; break; }
                            block = std::move(blockQueue.front());
                            blockQueue.pop();
                        }
                        blockQueueCV.notify_all();
                        if (block.compData.empty()) {
                            DecompBlock out;
                            out.data    = std::move(block.rawData);
                            out.gpuPath = false;
                            growResults(block.idx);
                            results[block.idx] = std::move(out);
                            ready[block.idx].store(1, std::memory_order_release);
                            blocksDone++; gpuBlocks++;
                            resultCV.notify_one();
                            continue;
                        }
                        batch.push_back(std::move(block));
                    }
                    if (batch.empty()) break;
                    if (!collectSlot(nextSlot)) break;
                    dispatchSlot(nextSlot, batch);
                    nextSlot = (nextSlot + 1) % allocatedSlots;
                    if (reachedEOF) break;
                }
                for (size_t i = 0; i < allocatedSlots && !decompError; i++)
                    collectSlot((nextSlot + i) % allocatedSlots);
            }
        cleanup_slots:
            for (auto& sl : slots) {
                if (sl.stream)       cudaStreamDestroy(sl.stream);
                if (sl.d_comp)       cudaFree(sl.d_comp);
                if (sl.d_decomp)     cudaFree(sl.d_decomp);
                if (sl.d_temp)       cudaFree(sl.d_temp);
                if (sl.d_inPtr)      cudaFree(sl.d_inPtr);
                if (sl.d_outPtr)     cudaFree(sl.d_outPtr);
                if (sl.d_inSize)     cudaFree(sl.d_inSize);
                if (sl.d_outBufSize) cudaFree(sl.d_outBufSize);
                if (sl.d_actualSize) cudaFree(sl.d_actualSize);
                if (sl.d_status)     cudaFree(sl.d_status);
                if (sl.h_inSizes)    cudaFreeHost(sl.h_inSizes);
                if (sl.h_actualSize) cudaFreeHost(sl.h_actualSize);
                if (sl.h_status)     cudaFreeHost(sl.h_status);
                if (sl.h_decomp)     cudaFreeHost(sl.h_decomp);
            }
        };

        // ── Variable declarations (all referenced by showProgress lambda) ──────
        std::atomic<int64_t>  readUs{0};
        std::atomic<size_t>   readBytesRead{0};
        size_t nextBlockToWrite  = 0;
        size_t totalBytesWritten = 0;
        size_t sparseBytes = 0;
        bool   canSparse   = (outputFd >= 0 && !testMode &&
                  (outputFd != STDOUT_FILENO || lseek(outputFd, 0, SEEK_CUR) >= 0));
        auto   startTime = std::chrono::high_resolution_clock::now();
        int64_t writeUs    = 0;
        int64_t waitUs     = 0;
        size_t  drainCalls = 0;
        size_t  drainBlocks= 0;
        size_t  maxPending = 0;

        // ── writev coalescing helpers ─────────────────────────────────────────
        // Accumulate non-sparse blocks and flush in one writev() call.
        // Sparse (all-zero) blocks force a flush then punch a hole.
        std::vector<struct iovec> wiovecs;
        size_t wiovecBytes = 0;
        wiovecs.reserve(64);

        auto flushWriteVec = [&]() -> bool {
            if (wiovecs.empty()) return true;
            auto t0 = std::chrono::steady_clock::now();
            size_t remaining = wiovecBytes;
            struct iovec* iov = wiovecs.data();
            int cnt = (int)wiovecs.size();
            while (cnt > 0 && remaining > 0) {
                ssize_t n = ::writev(outputFd, iov, std::min(cnt, IOV_MAX));
                if (n <= 0) {
                    fprintf(stderr, "Error: writev failed: %s\n", strerror(errno));
                    wiovecs.clear(); wiovecBytes = 0;
                    return false;
                }
                remaining -= (size_t)n;
                size_t skip = (size_t)n;
                while (skip > 0 && cnt > 0) {
                    if (iov->iov_len <= skip) { skip -= iov->iov_len; iov++; cnt--; }
                    else { iov->iov_base = (char*)iov->iov_base + skip;
                           iov->iov_len -= skip; skip = 0; }
                }
            }
            writeUs += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count();
            if (!outIsPipe) {
                off_t pos = lseek(outputFd, 0, SEEK_CUR);
                if (pos >= (off_t)wiovecBytes)
                    posix_fadvise(outputFd, pos - wiovecBytes, wiovecBytes,
                                  POSIX_FADV_DONTNEED);
            }
            wiovecs.clear(); wiovecBytes = 0;
            return true;
        };

        // writeBlock: queue a block for coalesced writev (flushes if sparse).
        // Returns false on write error.
        auto writeBlock = [&](DecompBlock& blk, bool& err) {
            if (outputFd < 0) return;
            if (canSparse && isAllZeros(blk.data.data(), blk.data.size())) {
                if (!flushWriteVec()) { err = true; return; }
                auto t0 = std::chrono::steady_clock::now();
                if (lseek(outputFd, (off_t)blk.data.size(), SEEK_CUR) < 0) {
                    fprintf(stderr, "Warning: lseek for sparse hole failed: %s"
                            "  falling back to write\n", strerror(errno));
                    struct iovec iov{blk.data.data(), blk.data.size()};
                    if (::writev(outputFd, &iov, 1) != (ssize_t)blk.data.size())
                        err = true;
                    writeUs += std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - t0).count();
                } else {
                    writeUs += std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - t0).count();
                    sparseBytes += blk.data.size();
                }
            } else {
                // Queue for coalesced write; flush if we'd exceed IOV_MAX
                if ((int)wiovecs.size() >= IOV_MAX) flushWriteVec();
                wiovecs.push_back({blk.data.data(), blk.data.size()});
                wiovecBytes += blk.data.size();
            }
        };
        // Called exclusively by the dedicated progress thread below at a
        // fixed 150ms interval, so the bar is visible regardless of how
        // fast the writer loop runs.
        auto showProgress = [&]() {
            if (g_verbosity == QUIET || estimatedBlocks <= 10) return;
            size_t rb    = readBytesRead.load(std::memory_order_relaxed);
            size_t done  = blocksDone.load();
            size_t tb    = totalBlocks.load();
            size_t knownTotal = tb > 0 ? tb * chunkSize : originalFileSize;
            size_t written = totalBytesWritten;
            size_t pct = knownTotal > 0 ? std::min(size_t(99), written * 100 / knownTotal) : 0;
            std::string ws = formatBytes(written);
            std::string ts = knownTotal > 0 ? formatBytes(knownTotal) : "?";
            if (testMode) {
                fprintf(stderr, "\r%sTesting:%s       %s%3zu%%%s  %s[%s %s%s%s / %s%s%s ]%s%s%s",
                        CC_BCYAN, CC_RESET, CC_BYELLOW, pct, CC_RESET,
                        CC_DIM, CC_RESET, CC_BWHITE, ws.c_str(), CC_RESET,
                        CC_WHITE, ts.c_str(), CC_RESET, CC_DIM, CC_RESET, CC_EL);
            } else if (done == 0 && rb == 0) {
                // GPU workers still initializing  no data moving yet
                fprintf(stderr, "\r%sInitializing:%s  %s[%s %s%s%s / %s%s%s ]%s%s%s",
                        CC_DIM, CC_RESET, CC_DIM, CC_RESET,
                        CC_WHITE, formatBytes(0).c_str(), CC_RESET,
                        CC_WHITE, (compressedFileSize > 0 ? formatBytes(compressedFileSize) : std::string("?")).c_str(), CC_RESET,
                        CC_DIM, CC_RESET, CC_EL);
            } else if (done == 0 && rb > 0) {
                // Reader active, GPU workers still spinning up
                std::string rs  = formatBytes(rb);
                std::string cfs = compressedFileSize > 0 ? formatBytes(compressedFileSize) : "?";
                size_t rpct = compressedFileSize > 0
                    ? std::min(size_t(99), rb * 100 / compressedFileSize) : 0;
                fprintf(stderr, "\r%sReading:%s       %s%3zu%%%s  %s[%s %s%s%s / %s%s%s ]%s%s%s",
                        CC_BCYAN, CC_RESET, CC_BYELLOW, rpct, CC_RESET,
                        CC_DIM, CC_RESET, CC_WHITE, rs.c_str(), CC_RESET,
                        CC_WHITE, cfs.c_str(), CC_RESET, CC_DIM, CC_RESET, CC_EL);
            } else {
                std::string gpuStr = formatBytes(gpuBlocks.load() * chunkSize);
                std::string cpuStr = formatBytes(cpuFallbackBlocks.load() * chunkSize);
                fprintf(stderr,
                    "\r%sDecompressing:%s %s%3zu%%%s  %s[%s %s%s%s / %s%s%s ]%s"
                    "  %sGPU:%s %s%s%s  %sCPU:%s %s%s%s%s",
                    CC_BCYAN, CC_RESET, CC_BYELLOW, pct, CC_RESET,
                    CC_DIM, CC_RESET, CC_BGREEN, ws.c_str(), CC_RESET,
                    CC_WHITE, ts.c_str(), CC_RESET, CC_DIM, CC_RESET,
                    CC_CYAN, CC_RESET, CC_GREEN, gpuStr.c_str(), CC_RESET,
                    CC_CYAN, CC_RESET, CC_BLUE,  cpuStr.c_str(), CC_RESET);
            }
            fflush(stderr);
        };

        // ── Thread launch order ─────────────────────────────────────────────
        // 1. Progress thread   starts displaying immediately
        // 2. Reader thread     starts filling blockQueue from disk/pipe
        // 3. GPU worker threads  init CUDA (cudaMalloc etc.) then drain the
        //    already-filling blockQueue.  By the time context init completes,
        //    compressed blocks are waiting in the queue.
        //
        std::atomic<bool> stopDecompProgress{false};
        std::thread decompProgressThread;
        if (g_verbosity != QUIET && estimatedBlocks > 10) {
            decompProgressThread = std::thread([&]() {
                g_progressActive.store(true, std::memory_order_relaxed);
                while (!stopDecompProgress.load()) {
                    showProgress();
                    std::this_thread::sleep_for(std::chrono::milliseconds(150));
                }
                g_progressActive.store(false, std::memory_order_relaxed);
            });
        }

        std::thread readerThread([&]() {
            VLOG(VERBOSE, "[GPU decomp reader] thread started\n");
            size_t blockIdx = 0;

            // ── Phase 1: drain pre-read queue ────────────────────────────────
            // Blocks that were read during GPU init are already in the queue.
            // Consume them at memory speed before touching the fd again.
            if (preDecompReaderStarted) {
                VLOG(VERBOSE, "[GPU decomp reader] draining pre-read queue "
                     "(%.2f GB buffered)\n",
                     preDecompReader.bytesRead.load() / (1024.0*1024.0*1024.0));
                PreDecompReader::Block preBlk;
                while (preDecompReader.queue.pop(preBlk)) {
                    if (preBlk.sizeField == 0) {
                        // EOF sentinel  seek inputFd to the checksum position
                        // before returning so the decompressor reads the actual
                        // stored checksum, not stale block data at hdrConsumed.
                        off_t seekTarget = (off_t)(preDecompReader.hdrConsumedBytes
                                                   + preDecompReader.bytesConsumedFromFd.load());
                        if (lseek(inputFd, seekTarget, SEEK_SET) == (off_t)-1)
                            fprintf(stderr, "Error: failed to seek inputFd to checksum: %s\n",
                                    strerror(errno));
                        VLOG(DEBUG, "[GPU decomp reader] sentinel: seeked inputFd to %zu\n",
                             (size_t)seekTarget);
                        totalBlocks.store(blockIdx);
                        readDone = true;
                        blockQueueCV.notify_all();
                        resultCV.notify_all();
                        return;
                    }
                    readBytesRead.fetch_add(4 + preBlk.data.size(), std::memory_order_relaxed);
                    bool isUncomp = (preBlk.sizeField & 0x80000000u) != 0;
                    RawBlock rb;
                    rb.idx      = blockIdx;
                    rb.origSize = chunkSize;
                    if (isUncomp) rb.rawData  = std::move(preBlk.data);
                    else          rb.compData = std::move(preBlk.data);
                    blocksQueued++;
                    { std::lock_guard<std::mutex> lk(blockQueueMutex); blockQueue.push(std::move(rb)); }
                    blockQueueCV.notify_one();
                    blockIdx++;
                }
                VLOG(VERBOSE, "[GPU decomp reader] pre-read queue drained (%zu blocks), "
                     "continuing from fd\n", blockIdx);
                // Seek inputFd past the bytes already consumed by the pre-reader.
                // Without this, inputFd is still positioned at hdrConsumed and
                // Phase 2 would re-read every pre-read block, producing duplicate
                // data and a checksum mismatch.
                off_t seekTarget = (off_t)(preDecompReader.hdrConsumedBytes
                                           + preDecompReader.bytesConsumedFromFd.load());
                if (lseek(inputFd, seekTarget, SEEK_SET) == (off_t)-1) {
                    fprintf(stderr, "Error: failed to seek inputFd past pre-read data: %s\n",
                            strerror(errno));
                    decompError = true;
                    totalBlocks.store(blockIdx);
                    readDone = true;
                    blockQueueCV.notify_all();
                    resultCV.notify_all();
                    return;
                }
                VLOG(DEBUG, "[GPU decomp reader] seeked inputFd to byte %zu\n",
                     (size_t)seekTarget);
            }

            // ── Phase 2: continue reading from fd ────────────────────────────
            while (!decompError) {
                uint32_t blockSize32 = 0;
                auto rt0 = std::chrono::steady_clock::now();
                ssize_t nr = ::read(inputFd, &blockSize32, 4);
                readUs.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - rt0).count());
                if (nr == 0) break;
                if (blockIdx == 0)
                    VLOG(VERBOSE, "[GPU decomp reader] first block received (%u bytes compressed)\n",
                         blockSize32 & 0x7FFFFFFFu);
                if (nr != 4) { fprintf(stderr, "Error: truncated chunk-size field at chunk %zu\n", blockIdx); decompError = true; break; }
                if (blockSize32 == 0) { estimatedBlocks = blockIdx; break; }
                bool isUncomp    = (blockSize32 & 0x80000000u) != 0;
                uint32_t blockSize = blockSize32 & 0x7FFFFFFFu;
                if (blockSize > 128u * 1024 * 1024) {
                    fprintf(stderr, "Error: implausibly large chunk %u at chunk %zu\n", blockSize, blockIdx);
                    decompError = true; break;
                }
                std::vector<uint8_t> raw(blockSize);
                {
                    auto rt1 = std::chrono::steady_clock::now();
                    if (::read(inputFd, raw.data(), blockSize) != (ssize_t)blockSize) {
                        fprintf(stderr, "Error: truncated chunk data at chunk %zu\n", blockIdx);
                        decompError = true; break;
                    }
                    readUs.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - rt1).count());
                }
                readBytesRead.fetch_add(4 + blockSize, std::memory_order_relaxed);
                try {
                    RawBlock rb;
                    rb.idx      = blockIdx;
                    rb.origSize = chunkSize;
                    if (isUncomp) rb.rawData  = std::move(raw);
                    else          rb.compData = std::move(raw);
                    blocksQueued++;
                    { std::lock_guard<std::mutex> lk(blockQueueMutex); blockQueue.push(std::move(rb)); }
                    blockQueueCV.notify_one();
                } catch (const std::bad_alloc&) {
                    fprintf(stderr, "Error: out of memory buffering compressed data\n");
                    decompError = true; break;
                }
                blockIdx++;
            }
            totalBlocks.store(blockIdx);
            readDone = true;
            blockQueueCV.notify_all();
            resultCV.notify_all();
            VLOG(DEBUG, "Reader: done, totalBlocks=%zu\n", blockIdx);
        });

        // Checksum thread: runs in parallel with the writer, hashing each
        // block as it becomes ready. Declared here so it captures
        // results/ready/decompError which are all now in scope.
        std::thread cksumThread([&]() {
            size_t nextToHash = 0;
            while (!decompError) {
                if (nextToHash >= estimatedBlocks) break;
                if (!ready[nextToHash].load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                    continue;
                }
                xxhState.update(results[nextToHash].data.data(),
                                results[nextToHash].data.size());
                cksumConsumed.fetch_add(1, std::memory_order_release);
                nextToHash++;
            }
        });

        // GPU workers start after reader  by the time cudaMalloc completes,
        // blockQueue already has data waiting.
        std::vector<std::thread> workerThreads;
        workerThreads.reserve(gpus.size());
        for (size_t g = 0; g < gpus.size(); g++)
            workerThreads.emplace_back(gpuWorker, g);

        while (true) {
            bool flushedAny = false;
            while (!decompError &&
                   nextBlockToWrite < estimatedBlocks &&
                   ready[nextBlockToWrite].load(std::memory_order_acquire)) {
                DecompBlock& blk = results[nextBlockToWrite];
                while (cksumConsumed.load(std::memory_order_acquire)
                       <= nextBlockToWrite && !decompError)
                    std::this_thread::yield();
                bool err = false;
                writeBlock(blk, err);
                if (err) decompError = true;
                totalBytesWritten += blk.data.size();
                nextBlockToWrite++;
                flushedAny = true;
                // Flush every 64 blocks (~256MB at 4MB chunks) to avoid
                // accumulating a giant writev call that stalls the writer
                if (wiovecs.size() >= 128) {  // 2 iovecs per block, so 64 blocks
                    if (!flushWriteVec()) { decompError = true; break; }
                    for (size_t i = drainBlocks; i < nextBlockToWrite; i++) {
                        results[i].data.clear(); results[i].data.shrink_to_fit();
                    }
                    drainBlocks = nextBlockToWrite;
                }
            }
            // Flush coalesced blocks, THEN clear (iovecs point into blk.data)
            if (flushedAny) {
                if (!flushWriteVec()) decompError = true;
                for (size_t i = drainBlocks; i < nextBlockToWrite; i++) {
                    results[i].data.clear(); results[i].data.shrink_to_fit();
                }
                drainCalls++;
                size_t pending = blocksDone.load() - nextBlockToWrite;
                if (pending > maxPending) maxPending = pending;
            }
            drainBlocks = nextBlockToWrite;
            if (decompError) break;
            size_t tb = totalBlocks.load();
            size_t bd = blocksDone.load();
            if (readDone.load() && bd >= tb) {
                VLOG(DEBUG, "Writer: done  totalBlocks=%zu  blocksDone=%zu  written=%zu\n",
                     tb, bd, nextBlockToWrite);
                break;
            }
            if (!flushedAny) {
                std::unique_lock<std::mutex> lk(resultMutex);
                auto wt0 = std::chrono::steady_clock::now();
                resultCV.wait(lk, [&]{
                    return (nextBlockToWrite < estimatedBlocks &&
                            ready[nextBlockToWrite].load(std::memory_order_acquire) != 0)
                        || decompError.load()
                        || (readDone.load() && blocksDone.load() >= totalBlocks.load());
                });
                waitUs += std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - wt0).count();
            }
        }

        readerThread.join();
        readDone = true;
        blockQueueCV.notify_all();
        for (auto& t : workerThreads) t.join();

        // Final drain after all workers done
        while (!decompError &&
               nextBlockToWrite < estimatedBlocks &&
               ready[nextBlockToWrite].load(std::memory_order_acquire)) {
            DecompBlock& blk = results[nextBlockToWrite];
            while (cksumConsumed.load(std::memory_order_acquire)
                   <= nextBlockToWrite && !decompError)
                std::this_thread::yield();
            bool err = false;
            writeBlock(blk, err);
            if (err) decompError = true;
            totalBytesWritten += blk.data.size();
            nextBlockToWrite++;
        }
        if (!flushWriteVec()) decompError = true;
        // Clear after flush  iovecs pointed into this data
        for (size_t i = drainBlocks; i < nextBlockToWrite; i++) {
            if (i < results.size()) { results[i].data.clear(); results[i].data.shrink_to_fit(); }
        }


        cksumThread.join();
        stopDecompProgress.store(true);
        if (decompProgressThread.joinable()) decompProgressThread.join();
        g_progressActive.store(false, std::memory_order_relaxed);

        auto endTime  = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        if (g_verbosity != QUIET && estimatedBlocks > 10) {
            std::string ws     = formatBytes(totalBytesWritten);
            std::string gpuStr = formatBytes(gpuBlocks.load() * chunkSize);
            std::string cpuStr = formatBytes(cpuFallbackBlocks.load() * chunkSize);
            if (testMode)
                fprintf(stderr, "\r%sTesting:%s       %s100%%%s  %s[%s %s%s%s / %s%s%s ]%s%s%s",
                        CC_BCYAN, CC_RESET, CC_BYELLOW, CC_RESET,
                        CC_DIM, CC_RESET, CC_BGREEN, ws.c_str(), CC_RESET,
                        CC_WHITE, ws.c_str(), CC_RESET, CC_DIM, CC_RESET, CC_EL);
            else
                fprintf(stderr, "\r%sDecompressing:%s %s100%%%s"
                    "  %sGPU:%s %s%s%s  %sCPU:%s %s%s%s%s\n",
                        CC_BCYAN, CC_RESET, CC_BYELLOW, CC_RESET,
                        CC_CYAN, CC_RESET, CC_GREEN, gpuStr.c_str(), CC_RESET,
                        CC_CYAN, CC_RESET, CC_BLUE,  cpuStr.c_str(), CC_RESET, CC_EL);
            fflush(stderr);
        }

        uint32_t computedChecksum = xxhState.digest();
        uint32_t storedChecksum   = 0;
        bool     checksumOk       = false;
        if (::read(inputFd, &storedChecksum, 4) == 4) {
            checksumOk = (computedChecksum == storedChecksum);
            if (!checksumOk) {
                fprintf(stderr, "%sWarning: Checksum mismatch  file may be corrupted!%s\n", CC_BRED, CC_RESET);
                fprintf(stderr, "  Stored:   0x%08X\n", storedChecksum);
                fprintf(stderr, "  Computed: 0x%08X\n", computedChecksum);
            }
        } else {
            fprintf(stderr, "Warning: Could not read stored checksum (truncated file?)\n");
        }

        if (inputFd != STDIN_FILENO) close(inputFd);
        if (outputFd >= 0 && outputFd != STDOUT_FILENO) {
            if (sparseBytes > 0) {
                // Extend the file to cover any trailing sparse holes.
                // lseek() only moves the file cursor; ftruncate() sets the
                // official file size so the decompressed output is complete.
                if (ftruncate(outputFd, (off_t)totalBytesWritten) != 0)
                    fprintf(stderr, "Warning: ftruncate failed: %s\n", strerror(errno));
            }
            if (syncOutput) fsync(outputFd);
            close(outputFd);
        } else if (outputFd == STDOUT_FILENO && canSparse && sparseBytes > 0) {
            // stdout was redirected to a seekable file (canSparse proved this
            // via lseek probe at startup).  ftruncate on STDOUT_FILENO extends
            // the file to cover any trailing sparse holes, just as we do for
            // a named output file.  We do not close or fsync stdout.
            if (ftruncate(outputFd, (off_t)totalBytesWritten) != 0)
                fprintf(stderr, "Warning: ftruncate(stdout) failed: %s\n", strerror(errno));
        }

        if (testMode) {
            struct stat st;
            size_t compressedSize = (stat(inputFile.c_str(), &st) == 0) ? st.st_size : 0;
            double ratio = compressedSize > 0 ? (100.0 * compressedSize / totalBytesWritten) : 0.0;
            std::string outputSize = formatBytes(totalBytesWritten);
            VLOG(NORMAL, "\r%sTest complete:%s %s%s%s in %.2f s%s\n",
                    CC_BGREEN, CC_RESET, CC_BGREEN, outputSize.c_str(), CC_RESET,
                    duration.count() / 1000.0, CC_EL);
            if (checksumOk)
                VLOG(NORMAL, "%sTest OK:%s %s  %sratio:%s %s%.1f%%%s\n",
                        CC_BGREEN, CC_RESET, inputFile.c_str(),
                        CC_CYAN, CC_RESET, CC_BYELLOW, ratio, CC_RESET);
            else
                VLOG(NORMAL, "%sTest FAILED:%s %s (checksum mismatch)\n",
                        CC_BRED, CC_RESET, inputFile.c_str());
            VLOG(VERBOSE, "  Compressed:   %.2f MB\n", compressedSize / (1024.0*1024.0));
            VLOG(VERBOSE, "  Uncompressed: %.2f MB  (ratio %.2f%%)\n",
                 totalBytesWritten / (1024.0*1024.0), ratio);
            VLOG(VERBOSE, "  Time: %.2f s  Throughput: %.2f MB/s\n",
                 duration.count() / 1000.0,
                 (totalBytesWritten / (1024.0*1024.0)) / (duration.count() / 1000.0));
            VLOG(VERBOSE, "  GPU chunks: %zu  CPU-fallback chunks: %zu\n",
                 gpuBlocks.load(), cpuFallbackBlocks.load());
        } else {
            double mbps = (totalBytesWritten / (1024.0*1024.0)) / (duration.count() / 1000.0);
            std::string outputSize = formatBytes(totalBytesWritten);
            VLOG(NORMAL, "\r%sDecompression complete:%s %s%s%s in %.2f s%s\n",
                    CC_BGREEN, CC_RESET, CC_BGREEN, outputSize.c_str(), CC_RESET,
                    duration.count() / 1000.0, CC_EL);
            VLOG(VERBOSE, "Throughput: %.2f MB/s\n", mbps);
            VLOG(VERBOSE, "  GPU chunks: %zu  CPU-fallback: %zu  pass-through: %zu\n",
                 gpuBlocks.load(), cpuFallbackBlocks.load(),
                 nextBlockToWrite - gpuBlocks.load() - cpuFallbackBlocks.load());
            if (sparseBytes > 0)
                VLOG(VERBOSE, "  Sparse holes: %s skipped (%.1f%% of output)\n",
                     formatBytes(sparseBytes).c_str(),
                     100.0 * sparseBytes / std::max(size_t(1), totalBytesWritten));
            VLOG(VERBOSE, "  Timing breakdown:\n");
            VLOG(VERBOSE, "    read(compressed):  %6.3f s\n",  readUs.load()  / 1e6);
            VLOG(VERBOSE, "    write(decomp out): %6.3f s  (%zu calls, avg %.2f ms each)\n",
                 writeUs / 1e6, nextBlockToWrite,
                 writeUs / 1e3 / std::max(size_t(1), nextBlockToWrite));
            VLOG(VERBOSE, "    wait(result stall):%6.3f s\n",  waitUs / 1e6);
            VLOG(VERBOSE, "    drain efficiency:  %.2f chunks/call  (max out-of-order: %zu chunks)\n",
                 drainCalls > 0 ? (double)drainBlocks / drainCalls : 0.0, maxPending);
        }

        if (!keepOriginal && !stdoutMode && !testMode) {
            if (unlink(inputFile.c_str()) != 0)
                fprintf(stderr, "Warning: Could not remove compressed file: %s\n", inputFile.c_str());
        }
        return checksumOk;
    }

    bool decompressFile() {
        switch (backendMode) {
            case BackendMode::GPU_ONLY:
                if (gpus.empty()) {
                    fprintf(stderr, "Warning: No GPUs available, falling back to CPU decompression\n");
                    return decompressFileCPU();
                }
                return decompressFileGPU();
            case BackendMode::HYBRID:
                if (gpus.empty()) {
                    VLOG(VERBOSE, "Hybrid decompression: no GPUs, using CPU-only\n");
                    return decompressFileCPU();
                }
                return decompressFileHybrid();
            case BackendMode::CPU_ONLY:
            default:
                return decompressFileCPU();
        }
    }

    bool decompressFileHybrid() {
        const bool stdinMode = (inputFile == "-");
        int inputFd = stdinMode ? STDIN_FILENO
                                : ::open(inputFile.c_str(), O_RDONLY | O_LARGEFILE);
        if (inputFd < 0) {
            fprintf(stderr, "Error opening input file: %s\n", strerror(errno));
            return false;
        }
        if (!stdinMode) posix_fadvise(inputFd, 0, 0, POSIX_FADV_SEQUENTIAL);
        posix_fadvise(inputFd, 0, 0, POSIX_FADV_WILLNEED);

        uint8_t headerBuf[32];
        ssize_t headerRead = ::read(inputFd, headerBuf, 32);
        if (headerRead < 15) {
            fprintf(stderr, "Error: File too small to be valid LZ4\n");
            { if (inputFd != STDIN_FILENO) close(inputFd); return false; }
        }
        LZ4Frame::FrameDescriptor desc;
        size_t headerBytes = 0;
        {
            std::string hs((char*)headerBuf, headerRead);
            std::istringstream hstream(hs, std::ios::binary);
            if (!LZ4Frame::readFrameHeader(hstream, desc)) {
                fprintf(stderr, "Error: Failed to read LZ4 frame header\n");
                { if (inputFd != STDIN_FILENO) close(inputFd); return false; }
            }
            headerBytes = hstream.tellg();
        }
        lseek(inputFd, (off_t)headerBytes, SEEK_SET);

        size_t originalFileSize = desc.contentSize;
        size_t chunkSize        = static_cast<size_t>(1) << (8 + 2 * desc.blockMaxSize);
        size_t estimatedBlocks  = (originalFileSize + chunkSize - 1) / chunkSize;

        size_t compressedFileSize = 0;
        {
            struct stat cst;
            if (fstat(inputFd, &cst) == 0 && (size_t)cst.st_size > 0) {
                compressedFileSize = (size_t)cst.st_size;
                // Only use the compressed-file-size fallback when the header
                // did not provide a valid contentSize (estimatedBlocks == 0 or 1).
                // The fallback formula divides by the LZ4 block frame minimum:
                // 4-byte size header + at least chunkSize/256 bytes of data
                // (LZ4 worst-case expansion is < 0.4%, so even incompressible
                // data compresses to ≥ 99.6% of input  but we use chunkSize/256
                // as a very conservative floor so we don't over-allocate).
                // Never let the fallback shrink a valid contentSize estimate.
                if (estimatedBlocks <= 1 && compressedFileSize > 0) {
                    size_t minBlockBytes = 4 + chunkSize / 256 + 1;
                    size_t compBlocks    = compressedFileSize / minBlockBytes + 1;
                    if (compBlocks > estimatedBlocks) {
                        VLOG(VERBOSE, "  contentSize unknown  using compressed-file "
                             "estimate: %zu chunks\n", compBlocks);
                        estimatedBlocks = compBlocks;
                    }
                }
            }
        }
        if (estimatedBlocks == 0) estimatedBlocks = 1;

        size_t effectiveThreads = cpuThreads ? cpuThreads : std::thread::hardware_concurrency();
        if (!effectiveThreads) effectiveThreads = 4;

        if (testMode) {
            VLOG(NORMAL, "Testing (hybrid, %zu GPU%s + %zu thread%s): %s\n",
                    gpus.size(), gpus.size() == 1 ? "" : "s",
                    effectiveThreads, effectiveThreads == 1 ? "" : "s",
                    inputFile.c_str());
        } else {
            VLOG(NORMAL, "%sDecompressing%s (hybrid, %zu GPU%s + %zu thread%s): %s -> %s\n",
                    CC_BCYAN, CC_RESET,
                    gpus.size(), gpus.size() == 1 ? "" : "s",
                    effectiveThreads, effectiveThreads == 1 ? "" : "s",
                    inputFile.c_str(), outputFile.c_str());
        }
        VLOG(VERBOSE, "  %.2f MB  |  chunk size %zu KB  |  ~%zu chunks\n",
             originalFileSize / (1024.0*1024.0), chunkSize/1024, estimatedBlocks);

        int outputFd = -1;
        bool outIsPipe = false;
        if (!testMode) {
            if (stdoutMode) {
                outputFd = STDOUT_FILENO;
                outIsPipe = true;
            } else {
                outputFd = open(getActualOutputPath(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
                if (outputFd < 0) {
                    fprintf(stderr, "Error: Cannot create output file: %s\n", getActualOutputPath());
                    { if (inputFd != STDIN_FILENO) close(inputFd); return false; }
                }
                posix_fadvise(outputFd, 0, 0, POSIX_FADV_SEQUENTIAL);
                outIsPipe = (lseek(outputFd, 0, SEEK_CUR) < 0);
            }
#ifdef F_SETPIPE_SZ
            if (outIsPipe) fcntl(outputFd, F_SETPIPE_SZ, 1 << 20);
#endif
        }
        XXH::State xxhState(XXH32_SEED);
        std::atomic<size_t> cksumConsumed{0};
        std::thread cksumThread;

        struct DecompBlock { std::vector<uint8_t> data; };
        struct RawBlock {
            size_t               idx;
            std::vector<uint8_t> compData;
            std::vector<uint8_t> rawData;
            size_t               origSize;
        };

        auto _at0 = std::chrono::high_resolution_clock::now();
        VLOG(DEBUG, "[Hybrid] allocating results[%zu] + ready[%zu]\n",
             estimatedBlocks, estimatedBlocks);
        std::vector<DecompBlock>          results(estimatedBlocks);
        std::vector<std::atomic<uint8_t>> ready(estimatedBlocks);
        for (auto& fl : ready) fl.store(0, std::memory_order_relaxed);
        VLOG(DEBUG, "[Hybrid] allocation done in %.1f ms\n",
             std::chrono::duration<double>(
                 std::chrono::high_resolution_clock::now() - _at0).count() * 1000.0);

        std::mutex              resultMutex;
        std::condition_variable resultCV;
        auto growResults = [&](size_t needed) {
            if (needed < results.size()) return;
            std::lock_guard<std::mutex> lk(resultMutex);
            if (needed < results.size()) return;
            size_t newSz = std::max(needed + 1, results.size() * 2);
            results.resize(newSz);
            std::vector<std::atomic<uint8_t>> newReady(newSz);
            for (size_t i = 0; i < ready.size(); i++)
                newReady[i].store(ready[i].load(std::memory_order_relaxed),
                                  std::memory_order_relaxed);
            ready = std::move(newReady);
            VLOG(VERBOSE, "  growResults: resized results/ready to %zu\n", newSz);
        };
        std::atomic<bool>       decompError{false};
        std::atomic<size_t>     gpuBlocks{0}, cpuBlocks{0};
        std::atomic<size_t>     blocksDone{0};
        // Incremented by each GPU worker after cudaMalloc slot allocation
        // completes.  CPU workers use this to know when all GPUs are hot.
        std::atomic<size_t>     gpusReady{0};
        std::atomic<size_t>     passthroughBlocks{0};

        const size_t SC_h          = slotCapacity;
        const size_t maxComp_h     = chunkSize + (chunkSize / 255) + 16;
        const size_t perStreamVRAM = SC_h * (maxComp_h + chunkSize + chunkSize);

        size_t N_STREAMS_H;
        if (pipelineDepth > 0) {
            N_STREAMS_H = (size_t)pipelineDepth;
        } else {
            size_t freeVRAM    = gpus[0].availableMemory;
            size_t target      = freeVRAM / 2;
            size_t autoStreams = target / perStreamVRAM;
            if (autoStreams < 1)  autoStreams = 1;
            if (autoStreams > 32) autoStreams = 32;
            N_STREAMS_H = autoStreams;
        }
        VLOG(VERBOSE, "Hybrid decompressor: %zu streams/GPU%s  ×  %zu chunks/slot  =  %.1f GB VRAM/slot"
             "  (%zu CPU threads)\n",
             N_STREAMS_H, pipelineDepth > 0 ? " (user-specified)" : " (auto)",
             SC_h,
             (SC_h * (maxComp_h + chunkSize + chunkSize)) / (1024.0*1024.0*1024.0),
             effectiveThreads);
        // Do one immediate NVML poll so we have load scores before dispatching.
        // The background thread will keep them updated every 2s thereafter.
        if (g_verbosity >= VERBOSE) loadMonitor.logCurrentLoad();

        // Per-GPU work queues  each GPU worker pops from its own queue.
        // The dispatcher routes to the least-loaded GPU so slower/busier GPUs
        // get fewer blocks, reducing out-of-order straggler delays at the writer.
        std::vector<TsQueue<RawBlock>> gpuQueues(gpus.size());
        TsQueue<RawBlock> cpuWorkQueue;

        std::atomic<bool>   dispatcherDone{false};
        std::atomic<size_t> totalBlocks{0};
        std::atomic<size_t> readBytesRead{0};
        std::thread dispatcherThread([&]() {
            VLOG(VERBOSE, "[Hybrid dispatcher] thread started\n");
            size_t blockIdx = 0;
            size_t nGpu = 0, nCpu = 0, nPass = 0;

            // Routing lambda  shared by pre-read drain and live read phases.
            const size_t zeroThreshold = chunkSize / 50;  // 2%

            // Pick the least-loaded GPU that has capacity (streamPct > 0).
            // Returns gpus.size() if all GPUs are overloaded (fall back to CPU).
            auto pickGPU = [&]() -> size_t {
                size_t best     = gpus.size();
                uint32_t bestScore = UINT32_MAX;
                for (size_t i = 0; i < gpus.size(); i++) {
                    uint32_t sp = gpus[i].streamPct.load(std::memory_order_relaxed);
                    if (sp == 0) continue;  // overloaded  skip
                    // Combine load score with current queue depth so we don't
                    // pile work onto a GPU that already has a large backlog.
                    uint32_t ls   = gpus[i].loadScore.load(std::memory_order_relaxed);
                    uint32_t qdepth = (uint32_t)std::min(gpuQueues[i].size(), size_t(100));
                    uint32_t composite = ls + qdepth;
                    if (composite < bestScore) { bestScore = composite; best = i; }
                }
                return best;
            };

            auto routeBlock = [&](RawBlock rb, bool isUncomp,
                                  std::vector<uint8_t> raw) {
                rb.origSize = chunkSize;
                if (isUncomp) {
                    growResults(rb.idx);
                    DecompBlock out; out.data = std::move(raw);
                    results[rb.idx] = std::move(out);
                    ready[rb.idx].store(1, std::memory_order_release);
                    blocksDone++; cpuBlocks++; passthroughBlocks++; resultCV.notify_one(); nPass++;
                } else {
                    rb.compData = std::move(raw);
                    try {
                        if (rb.compData.size() < zeroThreshold) {
                            cpuWorkQueue.push(std::move(rb)); nCpu++;
                        } else {
                            size_t gpuIdx = pickGPU();
                            if (gpuIdx < gpus.size()) {
                                gpuQueues[gpuIdx].push(std::move(rb)); nGpu++;
                            } else {
                                // All GPUs overloaded  route to CPU
                                cpuWorkQueue.push(std::move(rb)); nCpu++;
                            }
                        }
                    } catch (const std::bad_alloc&) {
                        fprintf(stderr, "Error: out of memory in dispatcher\n");
                        decompError = true;
                    }
                }
            };

            // ── Phase 1: drain pre-read queue ────────────────────────────────
            if (preDecompReaderStarted) {
                VLOG(VERBOSE, "[Hybrid dispatcher] draining pre-read queue "
                     "(%.2f GB buffered)\n",
                     preDecompReader.bytesRead.load() / (1024.0*1024.0*1024.0));
                PreDecompReader::Block preBlk;
                while (!decompError && preDecompReader.queue.pop(preBlk)) {
                    if (preBlk.sizeField == 0) {
                        // EOF sentinel  seek inputFd to the checksum position.
                        off_t seekTarget = (off_t)(preDecompReader.hdrConsumedBytes
                                                   + preDecompReader.bytesConsumedFromFd.load());
                        if (lseek(inputFd, seekTarget, SEEK_SET) == (off_t)-1)
                            fprintf(stderr, "Error: failed to seek inputFd to checksum: %s\n",
                                    strerror(errno));
                        VLOG(DEBUG, "[Hybrid dispatcher] sentinel: seeked inputFd to %zu\n",
                             (size_t)seekTarget);
                        totalBlocks.store(blockIdx);
                        for (auto& q : gpuQueues) q.close();
                        cpuWorkQueue.close();
                        dispatcherDone.store(true);
                        resultCV.notify_all();
                        return;
                    }
                    bool isUncomp = (preBlk.sizeField & 0x80000000u) != 0;
                    readBytesRead.fetch_add(4 + preBlk.data.size(), std::memory_order_relaxed);
                    RawBlock rb; rb.idx = blockIdx++;
                    routeBlock(std::move(rb), isUncomp, std::move(preBlk.data));
                }
                VLOG(VERBOSE, "[Hybrid dispatcher] pre-read queue drained "
                     "(%zu blocks), continuing from fd\n", blockIdx);
                // Seek inputFd past bytes already consumed by the pre-reader.
                off_t seekTarget = (off_t)(preDecompReader.hdrConsumedBytes
                                           + preDecompReader.bytesConsumedFromFd.load());
                if (lseek(inputFd, seekTarget, SEEK_SET) == (off_t)-1) {
                    fprintf(stderr, "Error: failed to seek inputFd past pre-read data: %s\n",
                            strerror(errno));
                    decompError = true;
                    for (auto& q : gpuQueues) q.close();
                    cpuWorkQueue.close();
                    dispatcherDone.store(true);
                    resultCV.notify_all();
                    return;
                }
                VLOG(DEBUG, "[Hybrid dispatcher] seeked inputFd to byte %zu\n",
                     (size_t)seekTarget);
            }

            // ── Phase 2: continue reading from fd ────────────────────────────
            while (!decompError) {
                uint32_t bs32 = 0;
                ssize_t nr = ::read(inputFd, &bs32, 4);
                if (nr == 0 || bs32 == 0) break;
                if (blockIdx == 0)
                    VLOG(VERBOSE, "[Hybrid dispatcher] first block received (%u bytes compressed)\n",
                         bs32 & 0x7FFFFFFFu);
                if (nr != 4) { fprintf(stderr, "Dispatcher: short read at chunk %zu\n", blockIdx); decompError = true; break; }
                bool isUncomp = (bs32 & 0x80000000u) != 0;
                uint32_t bs   =  bs32 & 0x7FFFFFFFu;
                if (bs > 128u * 1024 * 1024) {
                    fprintf(stderr, "Error: implausibly large chunk %u at chunk %zu\n", bs, blockIdx);
                    decompError = true; break;
                }
                std::vector<uint8_t> raw(bs);
                if (::read(inputFd, raw.data(), bs) != (ssize_t)bs) {
                    fprintf(stderr, "Error: truncated chunk data at chunk %zu\n", blockIdx);
                    decompError = true; break;
                }
                readBytesRead.fetch_add(4 + bs, std::memory_order_relaxed);
                RawBlock rb; rb.idx = blockIdx++;
                routeBlock(std::move(rb), isUncomp, std::move(raw));
            }
            totalBlocks.store(blockIdx);
            for (auto& q : gpuQueues) q.close();
            cpuWorkQueue.close();
            dispatcherDone.store(true);
            resultCV.notify_all();
            VLOG(DEBUG, "Dispatcher: done. total=%zu  gpu=%zu cpu=%zu pass=%zu\n", blockIdx, nGpu, nCpu, nPass);
        });

        // GPU worker
        auto gpuWorker = [&](size_t gpuIdx) {
            GPUDevice& gpu = gpus[gpuIdx];
            auto hwt0 = std::chrono::high_resolution_clock::now();
            auto hwElapsed = [&]() {
                return std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - hwt0).count() * 1000.0;
            };
            VLOG(DEBUG, "[Hybrid GPU%d decompWorker] started, calling cudaSetDevice\n", (int)gpuIdx);
            cudaSetDevice(gpu.deviceId);
            VLOG(DEBUG, "[Hybrid GPU%d decompWorker] cudaSetDevice done (%.1f ms)\n",
                 (int)gpuIdx, hwElapsed());
            { size_t fr, tot; if (cudaMemGetInfo(&fr, &tot) == cudaSuccess) gpu.availableMemory = fr; }

            // Scale slot count and batch size based on current GPU load.
            // A GPU already busy with LLM training gets fewer slots so it
            // finishes its smaller share quickly  reducing writer stalls
            // from straggler GPUs.  streamPct/batchPct are set by the load
            // monitor and default to 100 when NVML is unavailable.
            uint32_t sp = gpu.streamPct.load(std::memory_order_relaxed);
            uint32_t bp = gpu.batchPct.load(std::memory_order_relaxed);
            if (sp == 0) {
                // GPU fully overloaded  drain our queue into CPU and exit
                VLOG(VERBOSE, "GPU%d fully loaded (score %u%%)  routing to CPU\n",
                     (int)gpuIdx, gpu.loadScore.load());
                RawBlock block;
                while (gpuQueues[gpuIdx].pop(block)) cpuWorkQueue.push(std::move(block));
                return;
            }
            const size_t SC = std::max(size_t(1),
                                       (SC_h * sp + 99) / 100);  // round up, min 1
            const size_t effectiveBatchSz = std::max(size_t(1),
                                       (SC_h * bp + 99) / 100);
            // Scale CUDA stream count proportionally to streamPct
            const size_t effectiveStreams = std::max(size_t(1),
                                       (N_STREAMS_H * sp + 99) / 100);
            if (sp < 100)
                VLOG(VERBOSE, "GPU%d load %u%%  using %zu/%zu slots, "
                     "%zu/%zu streams, %zu batch\n",
                     (int)gpuIdx, gpu.loadScore.load(),
                     SC, SC_h, effectiveStreams, N_STREAMS_H, effectiveBatchSz);
            const size_t maxCompSize = maxComp_h;
            const size_t tempBytes   = chunkSize;
            nvcompBatchedLZ4DecompressOpts_t opts = nvcompBatchedLZ4DecompressDefaultOpts;

            struct HDecompSlot {
                cudaStream_t    stream       = nullptr;
                uint8_t*        d_comp       = nullptr;
                uint8_t*        d_decomp     = nullptr;
                void*           d_temp       = nullptr;
                void**          d_inPtr      = nullptr;
                void**          d_outPtr     = nullptr;
                size_t*         d_inSize     = nullptr;
                size_t*         d_outBufSize = nullptr;
                size_t*         d_actualSize = nullptr;
                nvcompStatus_t* d_status     = nullptr;
                size_t*         h_inSizes    = nullptr;
                size_t*         h_actualSize = nullptr;
                nvcompStatus_t* h_status     = nullptr;
                uint8_t*        h_decomp     = nullptr;
                bool            inFlight     = false;
                size_t          batchCount   = 0;
                std::vector<size_t> blockIdxs;
                size_t          origSize     = 0;
            };

            std::vector<HDecompSlot> slots;
            slots.reserve(effectiveStreams);
            for (size_t si = 0; si < effectiveStreams; si++) {
                HDecompSlot sl;
                bool ok = true;
                auto tryAlloc = [&](cudaError_t err, const char* what) {
                    if (err != cudaSuccess) {
                        VLOG(VERBOSE, "GPU%zu slot %zu: %s failed  stopping at %zu slots\n",
                             gpuIdx, si, what, slots.size());
                        cudaGetLastError(); ok = false;
                    }
                };
                tryAlloc(cudaStreamCreate(&sl.stream),                                   "cudaStreamCreate");
                if (ok) tryAlloc(cudaMalloc(&sl.d_comp,       SC * maxCompSize),         "d_comp");
                if (ok) tryAlloc(cudaMalloc(&sl.d_decomp,     SC * chunkSize),           "d_decomp");
                if (ok) tryAlloc(cudaMalloc(&sl.d_temp,       SC * tempBytes),           "d_temp");
                if (ok) tryAlloc(cudaMalloc(&sl.d_inPtr,      SC * sizeof(void*)),       "d_inPtr");
                if (ok) tryAlloc(cudaMalloc(&sl.d_outPtr,     SC * sizeof(void*)),       "d_outPtr");
                if (ok) tryAlloc(cudaMalloc(&sl.d_inSize,     SC * sizeof(size_t)),      "d_inSize");
                if (ok) tryAlloc(cudaMalloc(&sl.d_outBufSize, SC * sizeof(size_t)),      "d_outBufSize");
                if (ok) tryAlloc(cudaMalloc(&sl.d_actualSize, SC * sizeof(size_t)),      "d_actualSize");
                if (ok) tryAlloc(cudaMalloc(&sl.d_status,     SC * sizeof(nvcompStatus_t)), "d_status");
                if (ok) tryAlloc(cudaMallocHost(&sl.h_inSizes,    SC * sizeof(size_t)),  "h_inSizes");
                if (ok) tryAlloc(cudaMallocHost(&sl.h_actualSize, SC * sizeof(size_t)),  "h_actualSize");
                if (ok) tryAlloc(cudaMallocHost(&sl.h_status, SC * sizeof(nvcompStatus_t)), "h_status");
                if (ok) tryAlloc(cudaMallocHost(&sl.h_decomp,     SC * chunkSize),           "h_decomp");
                if (!ok) {
                    if (sl.h_decomp)     cudaFreeHost(sl.h_decomp);
                    if (sl.h_status)     cudaFreeHost(sl.h_status);
                    if (sl.h_actualSize) cudaFreeHost(sl.h_actualSize);
                    if (sl.h_inSizes)    cudaFreeHost(sl.h_inSizes);
                    if (sl.d_status)     cudaFree(sl.d_status);
                    if (sl.d_actualSize) cudaFree(sl.d_actualSize);
                    if (sl.d_outBufSize) cudaFree(sl.d_outBufSize);
                    if (sl.d_inSize)     cudaFree(sl.d_inSize);
                    if (sl.d_outPtr)     cudaFree(sl.d_outPtr);
                    if (sl.d_inPtr)      cudaFree(sl.d_inPtr);
                    if (sl.d_temp)       cudaFree(sl.d_temp);
                    if (sl.d_decomp)     cudaFree(sl.d_decomp);
                    if (sl.d_comp)       cudaFree(sl.d_comp);
                    if (sl.stream)       cudaStreamDestroy(sl.stream);
                    break;
                }
                std::vector<void*>  hInPtrs(SC), hOutPtrs(SC);
                std::vector<size_t> hOutBuf(SC, chunkSize);
                for (size_t i = 0; i < SC; i++) {
                    hInPtrs[i]  = sl.d_comp   + i * maxCompSize;
                    hOutPtrs[i] = sl.d_decomp + i * chunkSize;
                }
                cudaMemcpy(sl.d_inPtr,      hInPtrs.data(),  SC * sizeof(void*),  cudaMemcpyHostToDevice);
                cudaMemcpy(sl.d_outPtr,     hOutPtrs.data(), SC * sizeof(void*),  cudaMemcpyHostToDevice);
                cudaMemcpy(sl.d_outBufSize, hOutBuf.data(),  SC * sizeof(size_t), cudaMemcpyHostToDevice);
                slots.push_back(std::move(sl));
            }

            const size_t allocatedSlots = slots.size();
            if (allocatedSlots == 0) {
                fprintf(stderr, "GPU%zu: zero slots allocated  re-routing GPU queue to CPU\n", gpuIdx);
                RawBlock block;
                // Drain the entire GPU queue into the CPU queue.  Blocking
                // pop() returns false only when the queue is closed+empty.
                while (gpuQueues[gpuIdx].pop(block)) cpuWorkQueue.push(std::move(block));
                return;
            }
            VLOG(DEBUG, "[Hybrid GPU%d decompWorker] %zu slots allocated (%.1f ms), entering work loop\n",
                 (int)gpuIdx, allocatedSlots, hwElapsed());
            gpusReady.fetch_add(1, std::memory_order_release);
            VLOG(VERBOSE, "GPU%d ready for decompression (%zu/%zu GPUs hot)\n",
                 (int)gpuIdx, gpusReady.load(), gpus.size());

            std::vector<std::vector<std::vector<uint8_t>>>
                slotCompData(allocatedSlots, std::vector<std::vector<uint8_t>>(SC));
            size_t nextSlot = 0;

            auto collectSlot = [&](size_t si) -> bool {
                HDecompSlot& sl = slots[si];
                if (!sl.inFlight) return true;
                cudaStreamSynchronize(sl.stream);
                sl.inFlight = false;
                for (size_t j = 0; j < sl.batchCount; j++) {
                    nvcompStatus_t st = sl.h_status[j];
                    size_t actualOut  = sl.h_actualSize[j];
                    DecompBlock out;
                    if (st != nvcompSuccess || actualOut == 0) {
                        out.data.resize(sl.origSize);
                        int r = LZ4_decompress_safe(
                            reinterpret_cast<const char*>(slotCompData[si][j].data()),
                            reinterpret_cast<char*>(out.data.data()),
                            (int)slotCompData[si][j].size(), (int)sl.origSize);
                        if (r < 0) { fprintf(stderr, "Error: GPU%zu block %zu fallback failed\n", gpuIdx, sl.blockIdxs[j]); decompError = true; return false; }
                        out.data.resize(r); cpuBlocks++;
                    } else {
                        out.data.resize(actualOut);
                        memcpy(out.data.data(), sl.h_decomp + j * chunkSize, actualOut);
                        gpuBlocks++;
                    }
                    slotCompData[si][j].clear();
                    results[sl.blockIdxs[j]] = std::move(out);
                    ready[sl.blockIdxs[j]].store(1, std::memory_order_release);
                    blocksDone++; resultCV.notify_one();
                }
                return true;
            };

            auto dispatchSlot = [&](size_t si, std::vector<RawBlock>& batch) {
                HDecompSlot& sl = slots[si];
                sl.batchCount   = batch.size();
                sl.origSize     = batch[0].origSize;
                sl.blockIdxs.resize(sl.batchCount);
                for (size_t j = 0; j < sl.batchCount; j++) {
                    sl.blockIdxs[j]     = batch[j].idx;
                    size_t csz          = batch[j].compData.size();
                    sl.h_inSizes[j]     = csz;
                    slotCompData[si][j] = std::move(batch[j].compData);
                    cudaMemcpyAsync(sl.d_comp + j * maxCompSize, slotCompData[si][j].data(), csz, cudaMemcpyHostToDevice, sl.stream);
                }
                cudaMemcpyAsync(sl.d_inSize, sl.h_inSizes, sl.batchCount * sizeof(size_t), cudaMemcpyHostToDevice, sl.stream);
                nvcompStatus_t apiSt = nvcompBatchedLZ4DecompressAsync(
                    (const void* const*)sl.d_inPtr, sl.d_inSize, sl.d_outBufSize, sl.d_actualSize,
                    sl.batchCount, sl.d_temp, SC * tempBytes, (void* const*)sl.d_outPtr, opts, sl.d_status, sl.stream);
                if (apiSt != nvcompSuccess) {
                    for (size_t j = 0; j < sl.batchCount; j++) sl.h_status[j] = apiSt;
                    cudaStreamSynchronize(sl.stream);
                } else {
                    cudaMemcpyAsync(sl.h_actualSize, sl.d_actualSize, sl.batchCount * sizeof(size_t), cudaMemcpyDeviceToHost, sl.stream);
                    cudaMemcpyAsync(sl.h_status, sl.d_status, sl.batchCount * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost, sl.stream);
                    cudaMemcpyAsync(sl.h_decomp, sl.d_decomp, sl.batchCount * chunkSize, cudaMemcpyDeviceToHost, sl.stream);
                }
                sl.inFlight = true;
            };

            bool queueDone = false;
            while (!decompError && !queueDone) {
                std::vector<RawBlock> batch;
                batch.reserve(effectiveBatchSz);
                while ((int)batch.size() < (int)effectiveBatchSz && !decompError) {
                    RawBlock block;
                    // Blocking pop: returns false when queue is closed+empty.
                    if (!gpuQueues[gpuIdx].pop(block)) { queueDone = true; break; }
                    batch.push_back(std::move(block));
                }
                if (batch.empty()) break;
                if (!collectSlot(nextSlot)) break;
                dispatchSlot(nextSlot, batch);
                nextSlot = (nextSlot + 1) % allocatedSlots;
            }
            for (size_t i = 0; i < allocatedSlots && !decompError; i++)
                collectSlot((nextSlot + i) % allocatedSlots);

            for (auto& sl : slots) {
                if (sl.stream)       cudaStreamDestroy(sl.stream);
                if (sl.d_comp)       cudaFree(sl.d_comp);
                if (sl.d_decomp)     cudaFree(sl.d_decomp);
                if (sl.d_temp)       cudaFree(sl.d_temp);
                if (sl.d_inPtr)      cudaFree(sl.d_inPtr);
                if (sl.d_outPtr)     cudaFree(sl.d_outPtr);
                if (sl.d_inSize)     cudaFree(sl.d_inSize);
                if (sl.d_outBufSize) cudaFree(sl.d_outBufSize);
                if (sl.d_actualSize) cudaFree(sl.d_actualSize);
                if (sl.d_status)     cudaFree(sl.d_status);
                if (sl.h_inSizes)    cudaFreeHost(sl.h_inSizes);
                if (sl.h_actualSize) cudaFreeHost(sl.h_actualSize);
                if (sl.h_status)     cudaFreeHost(sl.h_status);
                if (sl.h_decomp)     cudaFreeHost(sl.h_decomp);
            }
        };

        // GPU queue backlog threshold for CPU stealing once all GPUs are hot.
        // If gpuWorkQueue depth exceeds this, GPUs are falling behind and
        // CPU workers should steal to help drain.  2 full batches per GPU so
        // we only steal when there's a meaningful backlog.
        const size_t gpuStealThreshold = gpus.size() * SC_h * 2;

        auto cpuWorker = [&](size_t tidx) {
            // Decompress a block and post the result.  Used by both phases.
            auto decompBlock = [&](RawBlock& block) -> bool {
                DecompBlock out;
                out.data.resize(block.origSize);
                int r = LZ4_decompress_safe(
                    reinterpret_cast<const char*>(block.compData.data()),
                    reinterpret_cast<char*>(out.data.data()),
                    (int)block.compData.size(), (int)block.origSize);
                if (r < 0) {
                    fprintf(stderr, "Error: CPU worker %zu block %zu\n", tidx, block.idx);
                    decompError = true; return false;
                }
                out.data.resize(r);
                growResults(block.idx);
                results[block.idx] = std::move(out);
                ready[block.idx].store(1, std::memory_order_release);
                cpuBlocks++; blocksDone++; resultCV.notify_one();
                return true;
            };

            // ── Phase 1: GPU init window ──────────────────────────────────────
            // While not all GPUs have finished cudaMalloc, service gpuWorkQueue
            // directly.  Blocking pop() wakes immediately when a block arrives
            // or the queue closes  no polling, no sleeps, no timers.
            while (gpusReady.load(std::memory_order_acquire) < gpus.size()
                   && !decompError) {
                RawBlock block;
                // Drain dedicated CPU queue first (zero-threshold blocks)
                if (cpuWorkQueue.pop_for(block, 0)) {
                    if (!decompBlock(block)) return;
                    continue;
                }
                // Try all GPU queues  pick first one with available work
                {
                    bool gotBlock = false;
                    for (auto& q : gpuQueues) {
                        if (q.pop_for(block, 0)) { gotBlock = true; break; }
                    }
                    if (gotBlock) {
                        if (!decompBlock(block)) return;
                        continue;
                    }
                    // No work yet  check if all queues closed (phase 2)
                    bool allClosed = true;
                    for (auto& q : gpuQueues) if (!q.isClosed()) { allClosed = false; break; }
                    if (allClosed) break;  // → phase 2
                    std::this_thread::yield();
                }
            }

            // ── Phase 2: normal protocol ──────────────────────────────────────
            // All GPUs hot.  CPUs handle their dedicated queue and steal from
            // the GPU queue only when it's backlogged beyond the threshold.
            while (!decompError) {
                RawBlock block;

                // Dedicated CPU queue first (non-blocking)
                if (cpuWorkQueue.pop_for(block, 0)) {
                    if (!decompBlock(block)) return;
                    continue;
                }

                // Steal from GPU queues when backlogged
                {
                    size_t totalDepth = 0;
                    for (auto& q : gpuQueues) totalDepth += q.size();
                    if (totalDepth > gpuStealThreshold) {
                        // Pop from the deepest queue to equalize
                        size_t deepest = 0; size_t deepestSz = 0;
                        for (size_t qi = 0; qi < gpuQueues.size(); qi++) {
                            size_t sz = gpuQueues[qi].size();
                            if (sz > deepestSz) { deepestSz = sz; deepest = qi; }
                        }
                        if (gpuQueues[deepest].pop_for(block, 0)) {
                            if (!decompBlock(block)) return;
                            continue;
                        }
                    }
                }

                // Block on cpuWorkQueue until work arrives or it closes
                if (!cpuWorkQueue.pop(block)) break;
                if (!decompBlock(block)) return;
            }
        };

        // ── Variables declared before showProgress so lambda can capture them ─
        auto   startTime         = std::chrono::high_resolution_clock::now();
        size_t nextBlockToWrite  = 0;
        size_t totalBytesWritten = 0;
        size_t maxPending        = 0;
        size_t sparseBytes       = 0;
        bool   canSparse         = (outputFd >= 0 && !testMode &&
                  (outputFd != STDOUT_FILENO || lseek(outputFd, 0, SEEK_CUR) >= 0));

        // ── writev coalescing helpers ─────────────────────────────────────────
        std::vector<struct iovec> wiovecs;
        size_t wiovecBytes = 0;
        int64_t writeUs = 0;
        wiovecs.reserve(64);

        auto flushWriteVec = [&]() -> bool {
            if (wiovecs.empty()) return true;
            auto t0 = std::chrono::steady_clock::now();
            size_t remaining = wiovecBytes;
            struct iovec* iov = wiovecs.data();
            int cnt = (int)wiovecs.size();
            while (cnt > 0 && remaining > 0) {
                ssize_t n = ::writev(outputFd, iov, std::min(cnt, IOV_MAX));
                if (n <= 0) {
                    fprintf(stderr, "Error: writev failed: %s\n", strerror(errno));
                    wiovecs.clear(); wiovecBytes = 0;
                    return false;
                }
                remaining -= (size_t)n;
                size_t skip = (size_t)n;
                while (skip > 0 && cnt > 0) {
                    if (iov->iov_len <= skip) { skip -= iov->iov_len; iov++; cnt--; }
                    else { iov->iov_base = (char*)iov->iov_base + skip;
                           iov->iov_len -= skip; skip = 0; }
                }
            }
            writeUs += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count();
            if (!outIsPipe) {
                off_t pos = lseek(outputFd, 0, SEEK_CUR);
                if (pos >= (off_t)wiovecBytes)
                    posix_fadvise(outputFd, pos - wiovecBytes, wiovecBytes,
                                  POSIX_FADV_DONTNEED);
            }
            wiovecs.clear(); wiovecBytes = 0;
            return true;
        };

        auto writeBlock = [&](DecompBlock& blk, bool& err) {
            if (outputFd < 0) return;
            if (canSparse && isAllZeros(blk.data.data(), blk.data.size())) {
                if (!flushWriteVec()) { err = true; return; }
                auto t0 = std::chrono::steady_clock::now();
                if (lseek(outputFd, (off_t)blk.data.size(), SEEK_CUR) < 0) {
                    fprintf(stderr, "Warning: lseek for sparse hole failed: %s"
                            "  falling back to write\n", strerror(errno));
                    struct iovec iov{blk.data.data(), blk.data.size()};
                    if (::writev(outputFd, &iov, 1) != (ssize_t)blk.data.size())
                        err = true;
                    writeUs += std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - t0).count();
                } else {
                    writeUs += std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - t0).count();
                    sparseBytes += blk.data.size();
                }
            } else {
                if ((int)wiovecs.size() >= IOV_MAX) flushWriteVec();
                wiovecs.push_back({blk.data.data(), blk.data.size()});
                wiovecBytes += blk.data.size();
            }
        };

        auto showProgress = [&]() {
            if (g_verbosity == QUIET || estimatedBlocks <= 10) return;
            size_t rb    = readBytesRead.load(std::memory_order_relaxed);
            size_t tb    = totalBlocks.load();
            size_t gpu   = gpuBlocks.load();
            size_t cpu   = cpuBlocks.load();
            size_t knownTotal = tb > 0 ? tb * chunkSize : originalFileSize;
            size_t written = totalBytesWritten;
            size_t pct = knownTotal > 0 ? std::min(size_t(99), written * 100 / knownTotal) : 0;
            std::string ws = formatBytes(written);
            std::string ts = knownTotal > 0 ? formatBytes(knownTotal) : "?";
            if (testMode) {
                fprintf(stderr, "\r%sTesting:%s       %s%3zu%%%s  %s[%s %s%s%s / %s%s%s ]%s%s%s",
                        CC_BCYAN, CC_RESET, CC_BYELLOW, pct, CC_RESET,
                        CC_DIM, CC_RESET, CC_BWHITE, ws.c_str(), CC_RESET,
                        CC_WHITE, ts.c_str(), CC_RESET, CC_DIM, CC_RESET, CC_EL);
            } else if (gpu == 0 && cpu == 0) {
                // Nothing decompressed yet  show read progress
                if (rb == 0) {
                    fprintf(stderr, "\r%sInitializing:%s  %s[%s %s%s%s / %s%s%s ]%s%s%s",
                            CC_DIM, CC_RESET, CC_DIM, CC_RESET,
                            CC_WHITE, formatBytes(0).c_str(), CC_RESET,
                            CC_WHITE, (compressedFileSize > 0 ? formatBytes(compressedFileSize) : std::string("?")).c_str(), CC_RESET,
                            CC_DIM, CC_RESET, CC_EL);
                } else {
                    std::string rs  = formatBytes(rb);
                    std::string cfs = compressedFileSize > 0 ? formatBytes(compressedFileSize) : "?";
                    size_t rpct = compressedFileSize > 0
                        ? std::min(size_t(99), rb * 100 / compressedFileSize) : 0;
                    fprintf(stderr, "\r%sReading:%s       %s%3zu%%%s  %s[%s %s%s%s / %s%s%s ]%s%s%s",
                            CC_BCYAN, CC_RESET, CC_BYELLOW, rpct, CC_RESET,
                            CC_DIM, CC_RESET, CC_WHITE, rs.c_str(), CC_RESET,
                            CC_WHITE, cfs.c_str(), CC_RESET, CC_DIM, CC_RESET, CC_EL);
                }
            } else {
                size_t done = blocksDone.load();
                size_t tb2  = totalBlocks.load();
                bool decompressionDone = dispatcherDone.load() && done >= tb2 && tb2 > 0;

                std::string gpuStr  = formatBytes(gpu * chunkSize);
                std::string cpuStr  = formatBytes(cpu * chunkSize);

                if (decompressionDone) {
                    fprintf(stderr,
                        "\r%sWriting:%s       %s%3zu%%%s  %s[%s %s%s%s / %s%s%s ]%s"
                        "  %sGPU:%s %s%s%s  %sCPU:%s %s%s%s%s%s",
                        CC_BGREEN, CC_RESET, CC_BYELLOW, pct, CC_RESET,
                        CC_DIM, CC_RESET, CC_BGREEN, ws.c_str(), CC_RESET,
                        CC_WHITE, ts.c_str(), CC_RESET, CC_DIM, CC_RESET,
                        CC_CYAN, CC_RESET, CC_GREEN,  gpuStr.c_str(),  CC_RESET,
                        CC_CYAN, CC_RESET, CC_BLUE,   cpuStr.c_str(),  CC_RESET, CC_EL);
                } else {
                    fprintf(stderr,
                        "\r%sDecompressing:%s %s%3zu%%%s  %s[%s %s%s%s / %s%s%s ]%s"
                        "  %sGPU:%s %s%s%s  %sCPU:%s %s%s%s%s%s",
                        CC_BCYAN, CC_RESET, CC_BYELLOW, pct, CC_RESET,
                        CC_DIM, CC_RESET, CC_BGREEN, ws.c_str(), CC_RESET,
                        CC_WHITE, ts.c_str(), CC_RESET, CC_DIM, CC_RESET,
                        CC_CYAN, CC_RESET, CC_GREEN,  gpuStr.c_str(),  CC_RESET,
                        CC_CYAN, CC_RESET, CC_BLUE,   cpuStr.c_str(),  CC_RESET, CC_EL);
                }
            }
            fflush(stderr);
        };

        // ── Thread launch order for hybrid decompressor ─────────────────────
        // 1. Progress thread    starts displaying immediately
        // 2. Dispatcher thread  starts reading from disk/pipe immediately,
        //    routing blocks to gpuWorkQueue and cpuWorkQueue
        // 3. CPU worker threads  start immediately; drain cpuWorkQueue while
        //    GPU workers are still doing CUDA context initialization
        // 4. GPU worker threads  initialize CUDA (cudaMalloc etc.) then
        //    drain gpuWorkQueue.  CPU workers have already been consuming
        //    the cpuWorkQueue (zero-threshold blocks) since step 3, so GPU
        //    init time is fully overlapped with useful CPU decompression work.

        std::atomic<bool> stopDecompProgress{false};
        std::thread decompProgressThread;
        if (g_verbosity != QUIET && estimatedBlocks > 10) {
            decompProgressThread = std::thread([&]() {
                g_progressActive.store(true, std::memory_order_relaxed);
                while (!stopDecompProgress.load()) {
                    showProgress();
                    std::this_thread::sleep_for(std::chrono::milliseconds(150));
                }
                g_progressActive.store(false, std::memory_order_relaxed);
            });
        }

        // dispatcherThread was launched above and is already running 
        // it began reading from disk immediately when created.

        // ── Launch CPU workers FIRST  before progress thread ───────────────
        // CPU workers begin Phase 1 (steal from GPU queue while GPUs init)
        // immediately.  Launching them before the progress thread ensures the
        // very first showProgress() tick sees non-zero CPU counters.
        std::vector<std::thread> allWorkers;
        allWorkers.reserve(gpus.size() + effectiveThreads);
        for (size_t t = 0; t < effectiveThreads; t++) allWorkers.emplace_back(cpuWorker, t);
        // GPU workers after  they init CUDA, then join the GPU queue.
        for (size_t g = 0; g < gpus.size(); g++) allWorkers.emplace_back(gpuWorker, g);

        cksumThread = std::thread([&]() {
            size_t nextToHash = 0;
            while (!decompError) {
                if (nextToHash >= estimatedBlocks) break;
                if (!ready[nextToHash].load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                    continue;
                }
                xxhState.update(results[nextToHash].data.data(),
                                results[nextToHash].data.size());
                cksumConsumed.fetch_add(1, std::memory_order_release);
                nextToHash++;
            }
        });

        size_t drainBlocks = 0;  // tracks start of current drain burst for clear-after-flush
        // Flush threshold: max blocks to accumulate before issuing writev.
        // 64 blocks × 4MB = 256MB per writev  large enough to amortize
        // syscall overhead, small enough to keep writes flowing steadily.
        static constexpr size_t FLUSH_EVERY = 64;

        while (true) {
            bool flushedAny = false;
            while (!decompError && nextBlockToWrite < estimatedBlocks &&
                   ready[nextBlockToWrite].load(std::memory_order_acquire)) {
                DecompBlock& blk = results[nextBlockToWrite];
                while (cksumConsumed.load(std::memory_order_acquire)
                       <= nextBlockToWrite && !decompError)
                    std::this_thread::yield();
                bool err = false;
                writeBlock(blk, err);
                if (err) decompError = true;
                totalBytesWritten += blk.data.size();
                nextBlockToWrite++; flushedAny = true;

                // Flush periodically to avoid accumulating a giant writev call
                if (wiovecs.size() >= FLUSH_EVERY * 2) {  // *2 because 2 iovecs/block
                    if (!flushWriteVec()) { decompError = true; break; }
                    for (size_t i = drainBlocks; i < nextBlockToWrite; i++) {
                        results[i].data.clear(); results[i].data.shrink_to_fit();
                    }
                    drainBlocks = nextBlockToWrite;
                }
            }
            if (flushedAny) {
                if (!flushWriteVec()) decompError = true;
                for (size_t i = drainBlocks; i < nextBlockToWrite; i++) {
                    results[i].data.clear(); results[i].data.shrink_to_fit();
                }
                drainBlocks = nextBlockToWrite;
                size_t pending = blocksDone.load() - nextBlockToWrite;
                if (pending > maxPending) maxPending = pending;
            }
            if (decompError) break;
            size_t tb = totalBlocks.load();
            size_t bd = blocksDone.load();
            if (dispatcherDone.load() && bd >= tb) break;
            if (!flushedAny) {
                std::unique_lock<std::mutex> lk(resultMutex);
                resultCV.wait(lk, [&]{
                    return (nextBlockToWrite < estimatedBlocks &&
                            ready[nextBlockToWrite].load(std::memory_order_acquire) != 0)
                        || decompError.load()
                        || (dispatcherDone.load() && blocksDone.load() >= totalBlocks.load());
                });
            }
        }

        if (dispatcherThread.joinable()) dispatcherThread.join();
        for (auto& t : allWorkers) t.join();

        // Final drain
        while (!decompError && nextBlockToWrite < estimatedBlocks &&
               ready[nextBlockToWrite].load(std::memory_order_acquire)) {
            DecompBlock& blk = results[nextBlockToWrite];
            while (cksumConsumed.load(std::memory_order_acquire)
               <= nextBlockToWrite && !decompError)
                std::this_thread::yield();
            bool err = false;
            writeBlock(blk, err);
            if (err) decompError = true;
            totalBytesWritten += blk.data.size();
            nextBlockToWrite++;
        }
        if (!flushWriteVec()) decompError = true;
        // Clear after flush  iovecs pointed into this data
        for (size_t i = drainBlocks; i < nextBlockToWrite; i++) {
            if (i < results.size()) { results[i].data.clear(); results[i].data.shrink_to_fit(); }
        }

        stopDecompProgress.store(true);
        if (decompProgressThread.joinable()) decompProgressThread.join();
        g_progressActive.store(false, std::memory_order_relaxed);

        auto endTime  = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        if (g_verbosity != QUIET && estimatedBlocks > 10) {
            std::string ws      = formatBytes(totalBytesWritten);
            std::string gpuStr  = formatBytes(gpuBlocks.load() * chunkSize);
            std::string cpuStr  = formatBytes(cpuBlocks.load() * chunkSize);
            if (testMode)
                fprintf(stderr, "\r%sTesting:%s       %s100%%%s  %s[%s %s%s%s / %s%s%s ]%s%s%s",
                        CC_BCYAN, CC_RESET, CC_BYELLOW, CC_RESET,
                        CC_DIM, CC_RESET, CC_BGREEN, ws.c_str(), CC_RESET,
                        CC_WHITE, ws.c_str(), CC_RESET, CC_DIM, CC_RESET, CC_EL);
            else
                fprintf(stderr, "\r%sDecompressing:%s %s100%%%s"
                    "  %sGPU:%s %s%s%s  %sCPU:%s %s%s%s%s\n",
                        CC_BCYAN, CC_RESET, CC_BYELLOW, CC_RESET,
                        CC_CYAN, CC_RESET, CC_GREEN,  gpuStr.c_str(),  CC_RESET,
                        CC_CYAN, CC_RESET, CC_BLUE,   cpuStr.c_str(),  CC_RESET, CC_EL);
            fflush(stderr);
        }

        if (cksumThread.joinable()) cksumThread.join();
        uint32_t computedCS = xxhState.digest();
        uint32_t storedCS   = 0;
        bool     csOk       = false;
        if (::read(inputFd, &storedCS, 4) == 4) {
            csOk = (computedCS == storedCS);
            if (!csOk)
                fprintf(stderr, "%sWarning: checksum mismatch  stored 0x%08X computed 0x%08X%s\n",
                        CC_BRED, storedCS, computedCS, CC_RESET);
        } else {
            fprintf(stderr, "Warning: could not read stored checksum\n");
        }
        if (inputFd != STDIN_FILENO) close(inputFd);
        if (outputFd >= 0 && outputFd != STDOUT_FILENO) {
            if (sparseBytes > 0) {
                // Cover any trailing sparse holes so the file has the correct
                // total size.  lseek() alone does not extend the file size.
                if (ftruncate(outputFd, (off_t)totalBytesWritten) != 0)
                    fprintf(stderr, "Warning: ftruncate failed: %s\n", strerror(errno));
            }
            if (syncOutput) fsync(outputFd);
            close(outputFd);
        } else if (outputFd == STDOUT_FILENO && canSparse && sparseBytes > 0) {
            // stdout redirected to a seekable file  extend to cover trailing holes.
            if (ftruncate(outputFd, (off_t)totalBytesWritten) != 0)
                fprintf(stderr, "Warning: ftruncate(stdout) failed: %s\n", strerror(errno));
        }

        double mbps = totalBytesWritten > 0 && duration.count() > 0
            ? (totalBytesWritten / (1024.0*1024.0)) / (duration.count() / 1000.0) : 0.0;
        std::string outputSize = formatBytes(totalBytesWritten);
        size_t finalPassthrough = passthroughBlocks.load();

        if (testMode) {
            double ratio = compressedFileSize > 0 ? (100.0 * compressedFileSize / totalBytesWritten) : 0.0;
            VLOG(NORMAL, "\r%sTest complete:%s %s%s%s in %.2f s%s\n",
                    CC_BGREEN, CC_RESET, CC_BGREEN, outputSize.c_str(), CC_RESET,
                    duration.count() / 1000.0, CC_EL);
            if (csOk)
                VLOG(NORMAL, "%sTest OK:%s %s  %sratio:%s %s%.1f%%%s\n",
                        CC_BGREEN, CC_RESET, inputFile.c_str(),
                        CC_CYAN, CC_RESET, CC_BYELLOW, ratio, CC_RESET);
            else
                VLOG(NORMAL, "%sTest FAILED:%s %s (checksum mismatch)\n",
                        CC_BRED, CC_RESET, inputFile.c_str());
        } else {
            VLOG(NORMAL, "\r%sDecompression complete:%s %s%s%s in %.2f s%s\n",
                    CC_BGREEN, CC_RESET, CC_BGREEN, outputSize.c_str(), CC_RESET,
                    duration.count() / 1000.0, CC_EL);
        }
        VLOG(VERBOSE, "  GPU: %zu chunks (%.1f%%)  CPU: %zu chunks (%.1f%%)"
             "  pass-through: %zu  throughput: %.2f MB/s\n",
             gpuBlocks.load(),
             nextBlockToWrite > 0 ? 100.0 * gpuBlocks.load()  / nextBlockToWrite : 0.0,
             cpuBlocks.load(),
             nextBlockToWrite > 0 ? 100.0 * cpuBlocks.load()  / nextBlockToWrite : 0.0,
             finalPassthrough, mbps);
        if (sparseBytes > 0)
            VLOG(VERBOSE, "  Sparse holes: %s skipped (%.1f%% of output)\n",
                 formatBytes(sparseBytes).c_str(),
                 100.0 * sparseBytes / std::max(size_t(1), totalBytesWritten));
        VLOG(VERBOSE, "  Result store: max out-of-order depth: %zu chunks\n", maxPending);

        if (!keepOriginal && !stdoutMode && !testMode) unlink(inputFile.c_str());

        loadMonitor.stop();
        return !decompError && csOk;
    }

    bool decompressFileCPU() {
        const bool stdinMode = (inputFile == "-");
        int inputFd = stdinMode ? STDIN_FILENO
                                : ::open(inputFile.c_str(), O_RDONLY | O_LARGEFILE);
        if (inputFd < 0) { fprintf(stderr, "Error opening input file: %s\n", strerror(errno)); return false; }
        if (!stdinMode) {
            posix_fadvise(inputFd, 0, 0, POSIX_FADV_SEQUENTIAL);
            posix_fadvise(inputFd, 0, 0, POSIX_FADV_WILLNEED);
        }
        uint8_t headerBuf[32];
        ssize_t headerRead = ::read(inputFd, headerBuf, 32);
        if (headerRead < 15) { fprintf(stderr, "Error: File too small to be valid LZ4\n"); { if (inputFd != STDIN_FILENO) close(inputFd); return false; } }
        LZ4Frame::FrameDescriptor desc;
        size_t headerBytes = 0;
        {
            std::string hs((char*)headerBuf, headerRead);
            std::istringstream hstream(hs, std::ios::binary);
            if (!LZ4Frame::readFrameHeader(hstream, desc)) { fprintf(stderr, "Error: Failed to read LZ4 frame header\n"); { if (inputFd != STDIN_FILENO) close(inputFd); return false; } }
            headerBytes = hstream.tellg();
        }
        if (lseek(inputFd, (off_t)headerBytes, SEEK_SET) == (off_t)-1) { fprintf(stderr, "Error seeking past header: %s\n", strerror(errno)); { if (inputFd != STDIN_FILENO) close(inputFd); return false; } }
        size_t originalFileSize = desc.contentSize;
        size_t chunkSize = static_cast<size_t>(1) << (8 + 2 * desc.blockMaxSize);
        size_t estimatedBlocks = originalFileSize > 0 ? (originalFileSize + chunkSize - 1) / chunkSize : 0;
        size_t numWorkers = (cpuThreads > 0) ? cpuThreads : std::thread::hardware_concurrency();
        if (numWorkers == 0) numWorkers = 4;
        if (testMode) {
            VLOG(NORMAL, "Testing (CPU, %zu thread%s): %s\n", numWorkers, numWorkers == 1 ? "" : "s", inputFile.c_str());
        } else {
            VLOG(NORMAL, "%sDecompressing%s (CPU, %zu thread%s): %s -> %s\n",
                    CC_BCYAN, CC_RESET, numWorkers, numWorkers == 1 ? "" : "s",
                    inputFile.c_str(), outputFile.c_str());
        }
        VLOG(VERBOSE, "  %.2f MB source  |  chunk size %zu KB  |  ~%zu chunks\n",
             originalFileSize/(1024.0*1024.0), chunkSize/1024, estimatedBlocks);
        VLOG(VERBOSE, "CPU decompression: %zu worker threads\n", numWorkers);

        int outputFd = -1;
        bool outIsPipe = false;
        if (!testMode) {
            if (stdoutMode) {
                outputFd = STDOUT_FILENO;
                outIsPipe = true;
            } else {
                outputFd = ::open(getActualOutputPath(), O_WRONLY | O_CREAT | O_TRUNC | O_LARGEFILE, 0644);
                if (outputFd < 0) { fprintf(stderr, "Error opening output '%s': %s\n", getActualOutputPath(), strerror(errno)); { if (inputFd != STDIN_FILENO) close(inputFd); return false; } }
                posix_fadvise(outputFd, 0, 0, POSIX_FADV_SEQUENTIAL);
                outIsPipe = (lseek(outputFd, 0, SEEK_CUR) < 0);
            }
#ifdef F_SETPIPE_SZ
            if (outIsPipe) fcntl(outputFd, F_SETPIPE_SZ, 1 << 20);
#endif
        }

        struct RawBlock { size_t blockIdx; bool isUncompressed; std::vector<uint8_t> data; };
        std::queue<RawBlock>    rawQueue;
        std::mutex              rawMutex;
        std::condition_variable rawCV;
        std::atomic<bool>       readerDone{false};
        std::atomic<bool>       readError{false};

        size_t compressedFileSize = 0;
        { struct stat cst; if (fstat(inputFd, &cst) == 0) compressedFileSize = (size_t)cst.st_size; }
        std::atomic<size_t> readBytesRead{0};

        struct DecompResult { bool ok = true; std::vector<uint8_t> data; };
        size_t resultCapacity = estimatedBlocks > 0 ? estimatedBlocks : size_t(16384);
        std::vector<DecompResult>          resultVec(resultCapacity);
        std::vector<std::atomic<uint8_t>>  ready2(resultCapacity);
        for (auto& fl : ready2) fl.store(0, std::memory_order_relaxed);

        std::mutex              resultMutex;
        std::condition_variable resultCV;
        std::atomic<bool>       writeError{false};
        std::atomic<uint32_t>   storedFooterCS{0};
        std::atomic<size_t>     blocksSubmitted{0};
        XXH::State xxhState(XXH32_SEED);
        std::atomic<size_t> cksumConsumed2{0};
        std::atomic<size_t>     totalBytesWritten{0};
        int64_t writeUs = 0, waitUs = 0;
        size_t drainCalls = 0, maxPending = 0, drainStart = 0;
        // Sparse-file optimisation  same design as GPU/hybrid decompressors.
        // See isAllZeros() for full explanation.
        size_t sparseBytes = 0;
        bool   canSparse   = (outputFd >= 0 && !testMode &&
                  (outputFd != STDOUT_FILENO || lseek(outputFd, 0, SEEK_CUR) >= 0));
        auto startTime = std::chrono::high_resolution_clock::now();

        // ── writev coalescing helpers ─────────────────────────────────────────
        std::vector<struct iovec> wiovecs;
        size_t wiovecBytes = 0;
        wiovecs.reserve(64);

        auto flushWriteVec = [&]() -> bool {
            if (wiovecs.empty()) return true;
            auto t0 = std::chrono::steady_clock::now();
            size_t remaining = wiovecBytes;
            struct iovec* iov = wiovecs.data();
            int cnt = (int)wiovecs.size();
            while (cnt > 0 && remaining > 0) {
                ssize_t n = ::writev(outputFd, iov, std::min(cnt, IOV_MAX));
                if (n <= 0) {
                    fprintf(stderr, "Error: writev failed: %s\n", strerror(errno));
                    wiovecs.clear(); wiovecBytes = 0;
                    return false;
                }
                remaining -= (size_t)n;
                size_t skip = (size_t)n;
                while (skip > 0 && cnt > 0) {
                    if (iov->iov_len <= skip) { skip -= iov->iov_len; iov++; cnt--; }
                    else { iov->iov_base = (char*)iov->iov_base + skip;
                           iov->iov_len -= skip; skip = 0; }
                }
            }
            writeUs += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count();
            if (!outIsPipe) {
                off_t pos = lseek(outputFd, 0, SEEK_CUR);
                if (pos >= (off_t)wiovecBytes)
                    posix_fadvise(outputFd, pos - wiovecBytes, wiovecBytes,
                                  POSIX_FADV_DONTNEED);
            }
            wiovecs.clear(); wiovecBytes = 0;
            return true;
        };

        auto writeBlock = [&](std::vector<uint8_t>& data, bool& err) {
            if (outputFd < 0) return;
            if (canSparse && isAllZeros(data.data(), data.size())) {
                if (!flushWriteVec()) { err = true; return; }
                auto t0 = std::chrono::steady_clock::now();
                if (lseek(outputFd, (off_t)data.size(), SEEK_CUR) < 0) {
                    fprintf(stderr, "Warning: lseek for sparse hole failed: %s"
                            "  falling back to write\n", strerror(errno));
                    struct iovec iov{data.data(), data.size()};
                    if (::writev(outputFd, &iov, 1) != (ssize_t)data.size())
                        err = true;
                    writeUs += std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - t0).count();
                } else {
                    writeUs += std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - t0).count();
                    sparseBytes += data.size();
                }
            } else {
                if ((int)wiovecs.size() >= IOV_MAX) flushWriteVec();
                wiovecs.push_back({data.data(), data.size()});
                wiovecBytes += data.size();
            }
        };

        std::thread readerThread([&]() {
            size_t blockIdx = 0;
            while (true) {
                uint32_t rawSz;
                ssize_t n = ::read(inputFd, &rawSz, 4);
                if (n == 0 || rawSz == 0) {
                    uint32_t footerCS = 0;
                    if (::read(inputFd, &footerCS, 4) == 4) storedFooterCS.store(footerCS, std::memory_order_relaxed);
                    break;
                }
                if (n != 4) { readError.store(true); break; }
                bool isUncomp = (rawSz & 0x80000000) != 0;
                uint32_t bsz  =  rawSz & 0x7FFFFFFF;
                if (bsz > 256*1024*1024) { fprintf(stderr, "Implausible chunk size=%u at chunk %zu\n", bsz, blockIdx); readError.store(true); break; }
                RawBlock blk;
                blk.blockIdx       = blockIdx++;
                blk.isUncompressed = isUncomp;
                blk.data.resize(bsz);
                n = ::read(inputFd, blk.data.data(), bsz);
                if (n != (ssize_t)bsz) { fprintf(stderr, "Short read chunk %zu: wanted %u got %zd\n", blk.blockIdx, bsz, n); readError.store(true); break; }
                readBytesRead.fetch_add(4 + bsz, std::memory_order_relaxed);
                try {
                {
                    std::lock_guard<std::mutex> lk(rawMutex);
                    if (readError.load() || writeError.load()) break;
                    if (isUncomp) {
                        resultVec[blk.blockIdx].ok   = true;
                        resultVec[blk.blockIdx].data = std::move(blk.data);
                        ready2[blk.blockIdx].store(1, std::memory_order_release);
                        resultCV.notify_one();
                    } else {
                        rawQueue.push(std::move(blk));
                    }
                    blocksSubmitted++;
                }
                } catch (const std::bad_alloc&) {
                    fprintf(stderr, "Error: out of memory buffering decompressed data\n");
                    readError.store(true); break;
                }
                rawCV.notify_one();
            }
            readerDone.store(true);
            rawCV.notify_all();
        });

        std::vector<std::thread> workers2;
        for (size_t t = 0; t < numWorkers; t++) {
            workers2.emplace_back([&]() {
                while (true) {
                    RawBlock blk;
                    {
                        std::unique_lock<std::mutex> lk(rawMutex);
                        rawCV.wait(lk, [&]{ return !rawQueue.empty() || (readerDone.load() && rawQueue.empty()) || readError.load() || writeError.load(); });
                        if (rawQueue.empty()) break;
                        blk = std::move(rawQueue.front()); rawQueue.pop();
                    }
                    rawCV.notify_one();
                    DecompResult res; res.ok = true;
                    if (blk.isUncompressed) {
                        res.data = std::move(blk.data);
                    } else {
                        res.data.resize(chunkSize);
                        int dsz = LZ4_decompress_safe((const char*)blk.data.data(), (char*)res.data.data(), (int)blk.data.size(), (int)chunkSize);
                        if (dsz < 0) { fprintf(stderr, "LZ4_decompress_safe failed block %zu (compSz=%zu ret=%d)\n", blk.blockIdx, blk.data.size(), dsz); res.ok = false; }
                        else res.data.resize(dsz);
                    }
                    resultVec[blk.blockIdx] = std::move(res);
                    ready2[blk.blockIdx].store(1, std::memory_order_release);
                    resultCV.notify_one();
                }
            });
        }

        bool ok = true;
        size_t nextBlockToWrite = 0;
        auto   lastProgress     = std::chrono::steady_clock::now();

        auto showProgress = [&]() {
            if (g_verbosity == QUIET || estimatedBlocks <= 10) return;
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastProgress).count() < 200) return;
            lastProgress = now;
            size_t tbw = totalBytesWritten.load();
            size_t knownTotal = originalFileSize > 0 ? originalFileSize : 0;
            size_t pct = knownTotal > 0 ? std::min(size_t(99), tbw * 100 / knownTotal) : 0;
            std::string ws = formatBytes(tbw);
            std::string ts = knownTotal > 0 ? formatBytes(knownTotal) : "?";
            if (testMode) {
                fprintf(stderr, "\r%sTesting:%s       %s%3zu%%%s  %s[%s %s%s%s / %s%s%s ]%s%s%s",
                        CC_BCYAN, CC_RESET, CC_BYELLOW, pct, CC_RESET,
                        CC_DIM, CC_RESET, CC_BWHITE, ws.c_str(), CC_RESET,
                        CC_WHITE, ts.c_str(), CC_RESET, CC_DIM, CC_RESET, CC_EL);
            } else if (nextBlockToWrite == 0) {
                size_t rb2 = readBytesRead.load(std::memory_order_relaxed);
                std::string rs = formatBytes(rb2);
                std::string cs2 = compressedFileSize > 0 ? formatBytes(compressedFileSize) : "?";
                fprintf(stderr, "\r%sReading:%s              %s[%s %s%s%s / %s%s%s ]%s%s%s",
                        CC_DIM, CC_RESET, CC_DIM, CC_RESET,
                        CC_WHITE, rs.c_str(), CC_RESET,
                        CC_WHITE, cs2.c_str(), CC_RESET, CC_DIM, CC_RESET, CC_EL);
            } else if (!readerDone.load()) {
                std::string cpuStr = formatBytes(blocksSubmitted.load() * chunkSize);
                fprintf(stderr,
                    "\r%sDecompressing:%s %s%3zu%%%s  %s[%s %s%s%s / %s%s%s ]%s"
                    "  %sCPU:%s %s%s%s%s",
                    CC_BCYAN, CC_RESET, CC_BYELLOW, pct, CC_RESET,
                    CC_DIM, CC_RESET, CC_BGREEN, ws.c_str(), CC_RESET,
                    CC_WHITE, ts.c_str(), CC_RESET, CC_DIM, CC_RESET,
                    CC_CYAN, CC_RESET, CC_BLUE, cpuStr.c_str(), CC_RESET);
            } else {
                fprintf(stderr, "\r%sWriting:%s       %s%3zu%%%s  %s[%s %s%s%s / %s%s%s ]%s%s%s",
                        CC_BGREEN, CC_RESET, CC_BYELLOW, pct, CC_RESET,
                        CC_DIM, CC_RESET, CC_BGREEN, ws.c_str(), CC_RESET,
                        CC_WHITE, ts.c_str(), CC_RESET, CC_DIM, CC_RESET, CC_EL);
            }
            fflush(stderr);
        };

        std::thread cksumThread2([&]() {
            size_t nextToHash = 0;
            while (true) {
                if (nextToHash >= resultCapacity) break;
                if (!ready2[nextToHash].load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                    continue;
                }
                if (resultVec[nextToHash].ok)
                    xxhState.update(resultVec[nextToHash].data.data(),
                                    resultVec[nextToHash].data.size());
                cksumConsumed2.fetch_add(1, std::memory_order_release);
                nextToHash++;
            }
        });

        while (ok && !writeError.load()) {
            bool flushedAny = false;
            while (!writeError.load() && nextBlockToWrite < resultCapacity &&
                   ready2[nextBlockToWrite].load(std::memory_order_acquire)) {
                DecompResult& res = resultVec[nextBlockToWrite];
                if (!res.ok) { writeError.store(true); ok = false; break; }
                while (cksumConsumed2.load(std::memory_order_acquire)
                       <= nextBlockToWrite)
                    std::this_thread::yield();
                bool err = false;
                writeBlock(res.data, err);
                if (err) { writeError.store(true); ok = false; break; }
                totalBytesWritten += res.data.size();
                nextBlockToWrite++; flushedAny = true; showProgress();
                // Flush every 64 blocks to avoid giant writev stalls
                if (wiovecs.size() >= 128) {
                    if (!flushWriteVec()) { writeError.store(true); ok = false; break; }
                    for (size_t i = drainStart; i < nextBlockToWrite; i++) {
                        resultVec[i % resultCapacity].data.clear();
                        resultVec[i % resultCapacity].data.shrink_to_fit();
                    }
                    drainStart = nextBlockToWrite;
                }
            }
            if (flushedAny) {
                if (!flushWriteVec()) { writeError.store(true); ok = false; }
                // Clear after flush  iovecs pointed into this data
                for (size_t i = drainStart; i < nextBlockToWrite; i++) {
                    resultVec[i % resultCapacity].data.clear();
                    resultVec[i % resultCapacity].data.shrink_to_fit();
                }
                drainStart = nextBlockToWrite;
                drainCalls++;
                size_t submitted = blocksSubmitted.load();
                size_t pending   = submitted > nextBlockToWrite ? submitted - nextBlockToWrite : 0;
                if (pending > maxPending) maxPending = pending;
            }
            if (ok && readError.load()) { ok = false; break; }
            if (ok && readerDone.load()) { if (nextBlockToWrite >= blocksSubmitted.load()) break; }
            if (!flushedAny) {
                std::unique_lock<std::mutex> lk(resultMutex);
                auto wt0 = std::chrono::steady_clock::now();
                resultCV.wait(lk, [&]{
                    return (nextBlockToWrite < resultCapacity && ready2[nextBlockToWrite].load(std::memory_order_acquire) != 0)
                        || writeError.load() || readError.load()
                        || (readerDone.load() && nextBlockToWrite >= blocksSubmitted.load());
                });
                waitUs += std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - wt0).count();
                showProgress();
            }
        }
        if (!flushWriteVec()) { writeError.store(true); ok = false; }

        rawCV.notify_all(); resultCV.notify_all();
        readerThread.join();
        for (auto& w : workers2) w.join();

        if (inputFd != STDIN_FILENO) close(inputFd);
        if (outputFd >= 0 && outputFd != STDOUT_FILENO) {
            if (sparseBytes > 0) {
                // Cover any trailing sparse holes so the file has the correct
                // total size.  lseek() alone does not extend the file size.
                if (ftruncate(outputFd, (off_t)totalBytesWritten.load()) != 0)
                    fprintf(stderr, "Warning: ftruncate failed: %s\n", strerror(errno));
            }
            if (syncOutput) fsync(outputFd);
            close(outputFd);
        } else if (outputFd == STDOUT_FILENO && canSparse && sparseBytes > 0) {
            // stdout redirected to a seekable file  extend to cover trailing holes.
            if (ftruncate(outputFd, (off_t)totalBytesWritten.load()) != 0)
                fprintf(stderr, "Warning: ftruncate(stdout) failed: %s\n", strerror(errno));
        }

        auto elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - startTime).count();

        cksumThread2.join();
        if (ok) {
            uint32_t computedChecksum = xxhState.digest();
            uint32_t storedChecksum   = storedFooterCS.load(std::memory_order_relaxed);
            bool checksumOk = (storedChecksum == 0) ? true : (computedChecksum == storedChecksum);
            if (!checksumOk) {
                fprintf(stderr, "%sWarning: Checksum mismatch  file may be corrupted!%s\n", CC_BRED, CC_RESET);
                fprintf(stderr, "  Stored:   0x%08X\n", storedChecksum);
                fprintf(stderr, "  Computed: 0x%08X\n", computedChecksum);
            }
            ok = ok && checksumOk;
            std::string outputSize = formatBytes(totalBytesWritten.load());
            if (g_verbosity != QUIET && estimatedBlocks > 10) {
                std::string ws = formatBytes(totalBytesWritten.load());
                fprintf(stderr, "\r%sWriting:%s       %s100%%%s  %s[%s %s%s%s / %s%s%s ]%s%s%s",
                        CC_BGREEN, CC_RESET, CC_BYELLOW, CC_RESET,
                        CC_DIM, CC_RESET, CC_BGREEN, ws.c_str(), CC_RESET,
                        CC_WHITE, ws.c_str(), CC_RESET, CC_DIM, CC_RESET, CC_EL);
                fflush(stderr);
            }
            fprintf(stderr, "\r%s%s%s %s%s%s in %.2f s%s\n",
                    CC_BGREEN,
                    testMode ? "Test complete:" : "Decompression complete:",
                    CC_RESET,
                    CC_BGREEN, outputSize.c_str(), CC_RESET, elapsed,
                    CC_EL);
            if (testMode) {
                double ratio = compressedFileSize > 0 ? (100.0 * compressedFileSize / totalBytesWritten.load()) : 0.0;
                if (checksumOk) VLOG(NORMAL, "%sTest OK:%s %s  %sratio:%s %s%.1f%%%s\n", CC_BGREEN, CC_RESET, inputFile.c_str(), CC_CYAN, CC_RESET, CC_BYELLOW, ratio, CC_RESET);
                else            VLOG(NORMAL, "%sTest FAILED:%s %s (checksum mismatch)\n", CC_BRED, CC_RESET, inputFile.c_str());
            }
            double mbps = totalBytesWritten.load() > 0 && elapsed > 0 ? (totalBytesWritten.load() / (1024.0*1024.0)) / elapsed : 0.0;
            VLOG(VERBOSE, "  Throughput: %.2f MB/s\n", mbps);
            VLOG(VERBOSE, "  Timing breakdown:\n");
            VLOG(VERBOSE, "    write(decomp out): %6.3f s  (%zu chunks, avg %.2f ms each)\n",
                 writeUs / 1e6, nextBlockToWrite, writeUs / 1e3 / std::max(size_t(1), nextBlockToWrite));
            VLOG(VERBOSE, "    wait(result stall):%6.3f s\n", waitUs / 1e6);
            VLOG(VERBOSE, "    drain efficiency:  %.2f chunks/call  (high-water pending: %zu chunks)\n",
                 drainCalls > 0 ? (double)nextBlockToWrite / drainCalls : 0.0, maxPending);
            if (sparseBytes > 0)
                VLOG(VERBOSE, "  Sparse holes: %s skipped (%.1f%% of output)\n",
                     formatBytes(sparseBytes).c_str(),
                     100.0 * sparseBytes / std::max(size_t(1), (size_t)totalBytesWritten.load()));
        }
        return ok;
    }

    void preprocessArgv(int& argc, char**& argv) {
        static std::vector<char*> newArgv;
        static std::vector<std::string> allocatedStrings;
        newArgv.clear(); allocatedStrings.clear();
        newArgv.push_back(argv[0]);
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-10") {
                allocatedStrings.push_back("--hc-level"); allocatedStrings.push_back("4");
                newArgv.push_back(const_cast<char*>(allocatedStrings[allocatedStrings.size()-2].c_str()));
                newArgv.push_back(const_cast<char*>(allocatedStrings[allocatedStrings.size()-1].c_str()));
            } else if (arg == "-11") {
                allocatedStrings.push_back("--hc-level"); allocatedStrings.push_back("8");
                newArgv.push_back(const_cast<char*>(allocatedStrings[allocatedStrings.size()-2].c_str()));
                newArgv.push_back(const_cast<char*>(allocatedStrings[allocatedStrings.size()-1].c_str()));
            } else if (arg == "-12") {
                allocatedStrings.push_back("--hc-level"); allocatedStrings.push_back("12");
                newArgv.push_back(const_cast<char*>(allocatedStrings[allocatedStrings.size()-2].c_str()));
                newArgv.push_back(const_cast<char*>(allocatedStrings[allocatedStrings.size()-1].c_str()));
            } else {
                newArgv.push_back(argv[i]);
            }
        }
        argc = (int)newArgv.size();
        argv = newArgv.data();
    }

    /*
     * isAllZeros: check whether a block of memory is entirely zero bytes.
     *
     * Used by:
     *   - Compression (hybrid): route zero-content chunks to the CPU queue
     *     instead of the GPU queue.  Zeros compress trivially in microseconds
     *     on a single CPU core; sending them to the GPU wastes PCIe bandwidth.
     *
     *   - Decompression (all backends): after a block is decompressed, if it
     *     is all zeros we lseek past it instead of calling write().  On
     *     filesystems that support sparse files (ext4, xfs, btrfs, APFS)
     *     this punches a hole and avoids both the I/O bandwidth and the
     *     physical disk space for the zero region.
     *
     * Correctness guarantee:
     *   This function checks the ACTUAL decompressed bytes.  The compressed
     *   size heuristic (<2%) used for routing is a performance hint only and
     *   never substitutes for this check.  A block of all 0x01 bytes compresses
     *   to roughly the same size as all 0x00 bytes; only scanning the real
     *   output bytes is safe.
     *
     * SIMD implementation strategy:
     *   Three tiers, selected at compile time via preprocessor:
     *
     *   1. AVX2 (32-byte vectors, RTX 5090 host CPUs almost certainly support
     *      this): ORs 32 bytes per iteration into an accumulator, tests the
     *      accumulator for non-zero with _mm256_testz_si256.  Early-exit via
     *      a second accumulator that is checked every 4 iterations (128 bytes)
     *      to keep the branch predictor happy without adding per-vector tests.
     *      Speed: ~8× faster than the scalar fallback for large zero blocks.
     *
     *   2. SSE2 (16-byte vectors, universal on x86-64): same OR-accumulate
     *      pattern, 16 bytes per iteration.  ~4× faster than scalar.
     *
     *   3. Scalar (any architecture): align to 8 bytes then scan 8 bytes per
     *      iteration.  Fastest possible without SIMD.
     *
     *   All tiers share the same early-exit property: the moment a non-zero
     *   byte is encountered the function returns false immediately.  For real
     *   data (the common case) this fires on the first or second iteration,
     *   making the overhead of calling this function negligible.
     *
     *   For a 256 KB zero block (one gzl4 chunk at -1):
     *     Scalar:  ~500 µs
     *     SSE2:    ~125 µs
     *     AVX2:    ~62 µs
     *   All are much less than the ~1 ms write() they replace.
     *   For a 500 GB file with 50% zeros: AVX2 saves ~240s vs scalar.
     */
#if defined(__AVX2__)
    static bool isAllZeros(const void* data, size_t len) {
        const uint8_t* p = static_cast<const uint8_t*>(data);

        // Scalar prefix: align to 32-byte boundary for AVX2
        while (len > 0 && (reinterpret_cast<uintptr_t>(p) & 31)) {
            if (*p++) return false;
            len--;
        }

        // AVX2 main loop: OR 32 bytes per iteration into two accumulators.
        // Two accumulators allow the CPU to execute two 256-bit OR operations
        // in parallel (one per execution port on modern Intel/AMD).
        // Early-exit check every 128 bytes keeps branch misprediction cost low
        // without testing on every iteration.
        const __m256i* v   = reinterpret_cast<const __m256i*>(p);
        __m256i        acc = _mm256_setzero_si256();
        size_t         n   = len / 32;

        while (n >= 4) {
            acc = _mm256_or_si256(acc, _mm256_loadu_si256(v));
            acc = _mm256_or_si256(acc, _mm256_loadu_si256(v + 1));
            acc = _mm256_or_si256(acc, _mm256_loadu_si256(v + 2));
            acc = _mm256_or_si256(acc, _mm256_loadu_si256(v + 3));
            // _mm256_testz_si256(a, a) returns 1 iff all bits of (a AND a) are 0,
            // i.e. iff a is all zeros.  Returns 0 (false) as soon as any bit is set.
            if (!_mm256_testz_si256(acc, acc)) return false;
            v   += 4;
            n   -= 4;
            acc  = _mm256_setzero_si256();
        }
        // Drain remaining full 32-byte vectors
        while (n--) {
            acc = _mm256_or_si256(acc, _mm256_loadu_si256(v++));
        }
        if (!_mm256_testz_si256(acc, acc)) return false;

        // Scalar tail: bytes that didn't fill a 32-byte vector
        p   = reinterpret_cast<const uint8_t*>(v);
        len = len & 31;
        while (len--) {
            if (*p++) return false;
        }
        return true;
    }

#elif defined(__SSE2__)
    static bool isAllZeros(const void* data, size_t len) {
        const uint8_t* p = static_cast<const uint8_t*>(data);

        // Scalar prefix: align to 16-byte boundary
        while (len > 0 && (reinterpret_cast<uintptr_t>(p) & 15)) {
            if (*p++) return false;
            len--;
        }

        // SSE2 main loop: OR 16 bytes per iteration, check every 64 bytes
        const __m128i* v   = reinterpret_cast<const __m128i*>(p);
        __m128i        acc = _mm_setzero_si128();
        size_t         n   = len / 16;

        while (n >= 4) {
            acc = _mm_or_si128(acc, _mm_loadu_si128(v));
            acc = _mm_or_si128(acc, _mm_loadu_si128(v + 1));
            acc = _mm_or_si128(acc, _mm_loadu_si128(v + 2));
            acc = _mm_or_si128(acc, _mm_loadu_si128(v + 3));
            // SSE2 has no testz; compare each byte lane to zero,
            // then movemask to get a 16-bit mask  any set bit = non-zero.
            if (_mm_movemask_epi8(_mm_cmpeq_epi8(acc, _mm_setzero_si128())) != 0xFFFF)
                return false;
            v   += 4;
            n   -= 4;
            acc  = _mm_setzero_si128();
        }
        while (n--) {
            acc = _mm_or_si128(acc, _mm_loadu_si128(v++));
        }
        if (_mm_movemask_epi8(_mm_cmpeq_epi8(acc, _mm_setzero_si128())) != 0xFFFF)
            return false;

        // Scalar tail
        p   = reinterpret_cast<const uint8_t*>(v);
        len = len & 15;
        while (len--) {
            if (*p++) return false;
        }
        return true;
    }

#else
    // Scalar fallback: no SIMD available (non-x86 or SIMD explicitly disabled)
    static bool isAllZeros(const void* data, size_t len) {
        const uint8_t* p = static_cast<const uint8_t*>(data);

        // Align to 8-byte boundary
        while (len > 0 && (reinterpret_cast<uintptr_t>(p) & 7)) {
            if (*p++) return false;
            len--;
        }
        // 8 bytes per iteration
        const uint64_t* w = reinterpret_cast<const uint64_t*>(p);
        while (len >= 8) {
            if (*w++) return false;
            len -= 8;
        }
        // Trailing bytes
        p = reinterpret_cast<const uint8_t*>(w);
        while (len--) {
            if (*p++) return false;
        }
        return true;
    }
#endif

    static std::string formatBytes(size_t bytes) {
        const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        int unitIdx = 0;
        double size = (double)bytes;
        while (size >= 1024.0 && unitIdx < 4) { size /= 1024.0; unitIdx++; }
        char buf[32];
        if (unitIdx == 0)       snprintf(buf, sizeof(buf), "%zu B", bytes);
        else if (size >= 100.0) snprintf(buf, sizeof(buf), "%.0f %s", size, units[unitIdx]);
        else if (size >= 10.0)  snprintf(buf, sizeof(buf), "%.1f %s", size, units[unitIdx]);
        else                    snprintf(buf, sizeof(buf), "%.2f %s", size, units[unitIdx]);
        return std::string(buf);
    }

    bool parseArguments(int argc, char* argv[]) {
        if (argc > 0 && argv[0] != nullptr) {
            const char* progName = strrchr(argv[0], '/');
            progName = progName ? progName + 1 : argv[0];
            if (strlen(progName) >= 2 && progName[0] == 'u' && progName[1] == 'n') {
                decompress = true;
                VLOG(VERBOSE, "Auto-enabled decompression mode (program name: %s)\n", progName);
            }
        }

        const char* short_opts = "cdfhkqT:tvVzZ123456789";
        const struct option long_opts[] = {
            {"stdout",            no_argument,       nullptr, 'c'},
            {"to-stdout",         no_argument,       nullptr, 'c'},
            {"decompress",        no_argument,       nullptr, 'd'},
            {"uncompress",        no_argument,       nullptr, 'd'},
            {"force",             no_argument,       nullptr, 'f'},
            {"help",              no_argument,       nullptr, 2000},
            {"keep",              no_argument,       nullptr, 'k'},
            {"quiet",             no_argument,       nullptr, 'q'},
            {"threads",           required_argument, nullptr, 'T'},
            {"test",              no_argument,       nullptr, 't'},
            {"verbose",           no_argument,       nullptr, 'v'},
            {"version",           no_argument,       nullptr, 2001},
            {"fast",              no_argument,       nullptr, '1'},
            {"best",              no_argument,       nullptr, 1013},
            {"cpu-only",          no_argument,       nullptr, 1001},
            {"gpu-only",          no_argument,       nullptr, 1002},
            {"hybrid",            no_argument,       nullptr, 1003},
            {"slot-capacity",     required_argument, nullptr, 1004},
            {"batch-size",        required_argument, nullptr, 1004},
            {"chunks-per-batch",  required_argument, nullptr, 1004},
            {"pipeline-depth",    required_argument, nullptr, 1005},
            {"streams-per-gpu",   required_argument, nullptr, 1005},
            {"slots-per-gpu",     required_argument, nullptr, 1005},
            {"no-early-read",     no_argument,       nullptr, 1006},
            {"force-compress",    no_argument,       nullptr, 'z'},
            {"hc-level",          required_argument, nullptr, 1007},
            {"progress",          no_argument,       nullptr, 1008},
            {"change-log",        no_argument,       nullptr, 1009},
            {"list",              no_argument,       nullptr, 1012},
            {"content-size",      no_argument,       nullptr, 1010},
            {"no-content-size",   no_argument,       nullptr, 1011},
            {"sync-output",       no_argument,       nullptr, 1014},
            {nullptr, 0, nullptr, 0}
        };

        int opt;
        int option_index = 0;
        while ((opt = getopt_long(argc, argv, short_opts, long_opts, &option_index)) != -1) {
            switch (opt) {
                case 'c': stdoutMode = true; keepOriginal = true; break;
                case 'd': decompress = true; break;
                case 'f': forceOverwrite = true; break;
                case 'h': printShortHelp(); earlyExit = true; return false;
                case 'k': keepOriginal = true; break;
                case 't': testMode = true; decompress = true; break;
                case 'q': g_verbosity = QUIET; break;
                case 'v': g_verbosity++; break;
                case 'V': printVersion(); earlyExit = true; return false;
                case 'T': {
                    char* endptr;
                    long threads = strtol(optarg, &endptr, 10);
                    if (*endptr != '\0' || threads < 1 || threads > 1024) {
                        fprintf(stderr, "Error: Invalid thread count: %s\n", optarg); return false;
                    }
                    cpuThreads = threads;
                    VLOG(DEBUG, "CPU threads set to %zu\n", cpuThreads);
                    break;
                }
                case '1': case '2': case '3': case '4': case '5':
                case '6': case '7': case '8': case '9':
                    compressionLevel = opt - '0';
                    VLOG(DEBUG, "Compression level %d specified\n", compressionLevel);
                    break;
                case 1001: backendMode = BackendMode::CPU_ONLY; VLOG(DEBUG, "Backend mode: CPU-only\n"); break;
                case 1002: backendMode = BackendMode::GPU_ONLY; VLOG(DEBUG, "Backend mode: GPU-only\n"); break;
                case 1003: backendMode = BackendMode::HYBRID;   VLOG(DEBUG, "Backend mode: Hybrid\n");   break;
                case 1004: {
                    char* endptr;
                    long cap = strtol(optarg, &endptr, 10);
                    if (*endptr != '\0' || cap < 1 || cap > 1024) { fprintf(stderr, "Error: --slot-capacity must be 1-1024\n"); return false; }
                    slotCapacity = cap;
                    VLOG(DEBUG, "Batch size set to %zu\n", slotCapacity);
                    break;
                }
                case 1005: {
                    char* endptr;
                    long depth = strtol(optarg, &endptr, 10);
                    if (*endptr != '\0' || depth < 1 || depth > 128) { fprintf(stderr, "Error: --pipeline-depth must be 1-128\n"); return false; }
                    pipelineDepth = depth;
                    VLOG(DEBUG, "Streams per GPU set to %zu\n", pipelineDepth);
                    break;
                }
                case 1006: disableEarlyRead = true; VLOG(DEBUG, "Early reader disabled\n"); break;
                case 'z': case 'Z': forceMode = true; VLOG(DEBUG, "Force compression mode enabled\n"); break;
                case 1007: {
                    char* endptr;
                    long hlv = strtol(optarg, &endptr, 10);
                    if (*endptr != '\0' || hlv < 1 || hlv > 12) { fprintf(stderr, "Error: --hc-level must be 1-12\n"); return false; }
                    hcLevel = (int)hlv;
                    break;
                }
                case 1008: forceProgress = true; if (g_verbosity < NORMAL) g_verbosity = NORMAL; break;
                case 1009: printChangelog(); earlyExit = true; return false;
                case 1013:
                    compressionLevel = 9; hcLevel = 12;
                    VLOG(DEBUG, "--best: compressionLevel=9, hcLevel=12\n");
                    break;
                case 1012: listMode = true; break;
                case 1010: storeContentSize = true;  contentSizeExplicit = true; break;
                case 1011: storeContentSize = false; contentSizeExplicit = true; break;
                case 1014: syncOutput = true; VLOG(DEBUG, "--sync-output enabled\n"); break;
                case 2000: printHelp();    earlyExit = true; return false;
                case 2001: printVersion(); earlyExit = true; return false;
                default:
                    fprintf(stderr, "Try 'gzl4 --help' for more information.\n");
                    return false;
            }
        }

        if (forceMode) decompress = false;

        if (listMode) {
            if (optind >= argc) { fprintf(stderr, "Error: --list requires at least one filename\n"); return false; }
            for (int i = optind; i < argc; i++) listFileArgs.push_back(argv[i]);
            inputFile = listFileArgs[0];
            return true;
        }

        if (optind < argc) {
            inputFile = argv[optind];
            for (int i = optind + 1; i < argc; i++) extraInputFiles.push_back(argv[i]);
            if (stdoutMode && !extraInputFiles.empty()) {
                fprintf(stderr, "Error: -c (stdout) is not compatible with multiple input files\n");
                fprintf(stderr, "Use -c with a single file, or omit -c to compress each file in place\n");
                return false;
            }
        } else if (!isatty(STDIN_FILENO)) {
            inputFile  = "-";
            stdoutMode = true;
            keepOriginal = true;
        } else {
            fprintf(stderr, "Error: No input file specified\n");
            fprintf(stderr, "Try 'gzl4 --help' for more information.\n");
            return false;
        }

        if (inputFile == "-") {
            if (outputFile.empty()) outputFile = "-";
            return true;
        }

        struct stat st;
        if (stat(inputFile.c_str(), &st) != 0) { fprintf(stderr, "Error: Cannot access input file: %s\n", inputFile.c_str()); return false; }
        if (!S_ISREG(st.st_mode)) { fprintf(stderr, "Error: Input is not a regular file: %s\n", inputFile.c_str()); return false; }

        bool hasLz4Extension = (inputFile.size() > 4 && inputFile.substr(inputFile.size() - 4) == ".lz4");
        if (!decompress && !forceMode && hasLz4Extension) {
            decompress = true;
            VLOG(VERBOSE, "Auto-detected decompression mode (input has .lz4 extension)\n");
        } else if (!decompress && forceMode && hasLz4Extension) {
            fprintf(stderr, "Warning: Compressing .lz4 file (use -z to override auto-detection)\n");
            fprintf(stderr, "         Output will be: %s.lz4\n", inputFile.c_str());
        }

        if (stdoutMode) {
            outputFile = "-";
        } else if (decompress) {
            if (inputFile.size() > 4 && inputFile.substr(inputFile.size() - 4) == ".lz4")
                outputFile = inputFile.substr(0, inputFile.size() - 4);
            else { fprintf(stderr, "Error: Input file doesn't have .lz4 extension\n"); return false; }
        } else {
            outputFile = inputFile + ".lz4";
        }

        if (!testMode && !forceOverwrite && !stdoutMode && stat(outputFile.c_str(), &st) == 0) {
            fprintf(stderr, "Error: Output file already exists: %s\n", outputFile.c_str());
            fprintf(stderr, "Use -f to force overwrite\n");
            return false;
        }

        if (forceOverwrite && !stdoutMode && !testMode) {
            tempOutputFile = outputFile + ".tmp";
            VLOG(VERBOSE, "Using temporary file for safe overwrite: %s\n", tempOutputFile.c_str());
        }

        return true;
    }

    void printShortHelp() {
        std::cout << R"(gzl4 )" << VERSION << R"( - Multi-Backend (GPU, CPU, and Hybrid) LZ4 Compression Tool

Usage: gzl4 [OPTION]... [FILE ...]

Common options:
  -c            write to stdout (single file only)
  -d            decompress
  -f            force overwrite
  -h            show this help (use --help for full details)
  -k            keep original files
      --progress show progress when piped (pipe mode defaults to quiet)
  -q            quiet mode
  -t            test integrity
  -v            verbose (-vv, -vvv for more)
  -z            force compression (even if .lz4)
  -1 to -9      compression level (default: -1)
  -10 to -12    LZ4 High Compression (CPUs only)
  -V            show version (use --version for full details)

Program name behavior:
  ungzl4        Auto-enables decompression mode (-d implied)

Examples:
  gzl4 file.tar              # compress to file.tar.lz4
  gzl4 a.tar b.tar c.tar     # compress multiple files in place
  gzl4 -d file.tar.lz4       # decompress to file.tar
  gzl4 -d *.lz4              # decompress multiple files
  gzl4 file.tar.lz4          # auto-detects decompression
  gzl4 -z file.tar.lz4       # compress again to file.tar.lz4.lz4
  cat file | gzl4 > out.lz4  # pipe mode
  ungzl4 file.tar.lz4        # decompress (same as gzl4 -d)

For complete documentation: --help; version history: --change-log
)" << std::endl;
    }

    void printHelp() {
        std::cout << "gzl4 " << VERSION <<
            R"HELP( - Multi-Backend (GPU, CPU, and Hybrid) LZ4 Compression Tool

USAGE
  gzl4 [OPTIONS] [FILE ...]        compress FILE(s) -> FILE.lz4
  gzl4 -d [OPTIONS] FILE.lz4 ...  decompress FILE(s) -> FILE
  gzl4 [OPTIONS] < FILE > OUT      pipe: stdin -> stdout
  tar -I gzl4 -cf a.tar.lz4 dir/ use as tar compressor
  tar -I gzl4 -xf a.tar.lz4      use as tar decompressor

BASIC OPTIONS
  -c, --stdout         write to standard output; keep original files
  -d, --decompress     decompress (default: compress)
  -f, --force          overwrite output if it already exists
  -k, --keep           keep input file after compress/decompress
  -t, --test           verify integrity; no output written
      --list           list information about one or more .lz4 files
  -z, --force-compress force compression even for .lz4 input files
      --content-size   embed original size in LZ4 frame header
      --no-content-size omit original size from LZ4 frame header
      --sync-output    call fsync() before closing the output file
                       (ensures durability on power loss; disabled by default
                        as it causes 4-20x timing variance on consumer NVMe
                        drives due to SLC write-cache flush behaviour)
  -h                   short help
      --help           this help
      --change-log     full version history
  -V, --version        version information

MULTI-FILE
  gzl4 accepts multiple input files on the command line:
    gzl4 a.tar b.tar c.tar        compress each -> a.tar.lz4 b.tar.lz4 c.tar.lz4
    gzl4 -d a.lz4 b.lz4          decompress each in place
    gzl4 -f *.lz4                 decompress all .lz4, overwrite if needed
  Note: -c (--stdout) is not compatible with multiple files.

COMPRESSION LEVELS
  -1 .. -9             LZ4 fast compression (default: -1)
  --best               HC level 12 on CPU workers + 4 MB chunks on GPU
  -10 / -11 / -12      LZ4 HC high compression (CPU workers only)
  --hc-level N         Explicit HC level 1-12

BACKEND SELECTION
  (default)            Hybrid: GPU-priority with CPU fill-in
  --hybrid             Hybrid mode (explicit)
  --gpu-only           GPU only via nvCOMP batched LZ4
  --cpu-only           Multi-threaded CPU only (all cores)
  -T N, --threads N    CPU thread count (default: all cores)

GPU TUNING
  --batch-size N       Chunks per GPU batch (default: auto)
  --streams-per-gpu N  Concurrent streams per GPU (default: auto)
  --no-early-read      Disable read-ahead during GPU init

VERBOSITY AND PROGRESS
  (default)            Level 1: progress bars + completion summary
  -q, --quiet          Level 0: errors only
  -v                   Level 2: per-file stats
  -vv                  Level 3: timing breakdowns
  -vvv                 Level 4: debug / chunk-level detail
  --progress           Force level-1 progress when reading from stdin

PIPE USAGE
  tar -cf - mydir | gzl4 > archive.tar.lz4
  gzl4 -dc archive.tar.lz4 | tar -x
  gzl4 -c bigfile.dat | ssh host "cat > /data/bigfile.dat.lz4"

TAR INTEGRATION
  tar -I gzl4 -cf archive.tar.lz4 mydir/
  tar -I gzl4 -xf archive.tar.lz4
  tar -I "gzl4 --gpu-only"  -cf archive.tar.lz4 mydir/
  tar -I "gzl4 -12"         -cf archive.tar.lz4 mydir/

UNGZL4 AND SYMLINKS
  ln -s gzl4 ungzl4
  ungzl4 archive.tar.lz4            # decompress
  ungzl4 -z file.tar                # compress (overrides "un" prefix)

EXAMPLES
  gzl4 archive.tar                       compress -> archive.tar.lz4
  gzl4 -d archive.tar.lz4               decompress -> archive.tar
  gzl4 -t archive.tar.lz4               test integrity
  gzl4 --list archive.tar.lz4           show frame info
  gzl4 -12 bigfile.dat                  max HC compression (CPU only)
  gzl4 --sync-output -f archive.tar     compress with fsync for durability
  gzl4 --cpu-only -T 16 data.tar        16-thread CPU only
  gzl4 --gpu-only --batch-size 32 d.tar larger GPU batches
  tar -I gzl4 -cf archive.tar.lz4 mydir/
)HELP" << std::endl;
    }

    struct FrameInfo {
        uint64_t compressedBytes;
        uint64_t uncompressedBytes;
        bool hasContentSize, hasContentChecksum, hasBlockChecksum, blockIndependence;
        int  blockMaxSizeId;
    };

    bool scanFrames(int fd, std::vector<FrameInfo>& frames) {
        off_t pos = 0;
        while (true) {
            uint32_t magic = 0;
            ssize_t n = ::pread(fd, &magic, 4, pos);
            if (n == 0) break;
            if (n < 4) return false;
            if (magic != LZ4_MAGIC) break;
            pos += 4;
            uint8_t flg, bd;
            if (::pread(fd, &flg, 1, pos) != 1) return false;
            pos++;
            if (::pread(fd, &bd,  1, pos) != 1) return false;
            FrameInfo fi{};
            fi.blockIndependence  = (flg & 0x20) != 0;
            fi.hasBlockChecksum   = (flg & 0x10) != 0;
            fi.hasContentSize     = (flg & 0x08) != 0;
            fi.hasContentChecksum = (flg & 0x04) != 0;
            fi.blockMaxSizeId     = (bd  >> 4) & 0x07;
            pos++;
            if (fi.hasContentSize) {
                uint64_t cs = 0;
                if (::pread(fd, &cs, 8, pos) != 8) return false;
                fi.uncompressedBytes = cs; pos += 8;
            }
            pos++;
            uint64_t blockBytes = 0;
            while (true) {
                uint32_t bsz = 0;
                if (::pread(fd, &bsz, 4, pos) != 4) return false;
                pos += 4; blockBytes += 4;
                if (bsz == 0) break;
                uint32_t dataSz = bsz & 0x7FFFFFFF;
                pos += dataSz; blockBytes += dataSz;
                if (fi.hasBlockChecksum) { pos += 4; blockBytes += 4; }
            }
            if (fi.hasContentChecksum) { pos += 4; blockBytes += 4; }
            uint64_t headerBytes = 4 + 1 + 1 + 1 + (fi.hasContentSize ? 8 : 0);
            fi.compressedBytes = headerBytes + blockBytes;
            frames.push_back(fi);
        }
        return true;
    }

    static std::string fmtBytes(uint64_t n) {
        if (n == 0) return "N/A";
        const char* units[] = { "", "K", "M", "G", "T" };
        double v = (double)n; int u = 0;
        while (v >= 1000.0 && u < 4) { v /= 1024.0; u++; }
        char buf[32];
        if (u == 0) snprintf(buf, sizeof(buf), "%" PRIu64, n);
        else        snprintf(buf, sizeof(buf), "%.2f%s", v, units[u]);
        return buf;
    }

    static std::string colCell(const char* color, const std::string& text, int width) {
        std::string padded = text;
        if ((int)text.size() < width) padded.append(width - (int)text.size(), ' ');
        if (!g_color || color == nullptr) return padded;
        return std::string(color) + padded + CC_RESET;
    }

    static std::string fmtComp(uint64_t n, int w)              { return colCell(CC_CYAN,   fmtBytes(n), w); }
    static std::string fmtUncomp(uint64_t n, bool known, int w){ return colCell(known ? CC_BGREEN : CC_DIM, known ? fmtBytes(n) : "N/A", w); }
    static std::string fmtRatio(double comp, double uncomp, int w) {
        if (uncomp <= 0) return colCell(CC_DIM, "N/A", w);
        char buf[32]; snprintf(buf, sizeof(buf), "%.3f", comp / uncomp);
        return colCell(CC_BYELLOW, buf, w);
    }
    static std::string blockLabel(int id, bool independent) {
        char buf[8]; snprintf(buf, sizeof(buf), "B%d%c", id, independent ? 'I' : 'D');
        return buf;
    }

    bool listFile(const std::string& path) {
        struct stat st;
        if (stat(path.c_str(), &st) != 0) { fprintf(stderr, "gzl4 --list: cannot stat '%s': %s\n", path.c_str(), strerror(errno)); return false; }
        uint64_t fileBytes = (uint64_t)st.st_size;
        int fd = ::open(path.c_str(), O_RDONLY | O_LARGEFILE);
        if (fd < 0) { fprintf(stderr, "gzl4 --list: cannot open '%s': %s\n", path.c_str(), strerror(errno)); return false; }
        std::vector<FrameInfo> frames;
        bool ok = scanFrames(fd, frames);
        ::close(fd);
        if (!ok || frames.empty()) { fprintf(stderr, "gzl4 --list: '%s': not a valid LZ4 file\n", path.c_str()); return false; }
        uint64_t totalComp = 0, totalUncomp = 0;
        bool allHaveSize = true;
        for (auto& fi : frames) { totalComp += fi.compressedBytes; if (fi.hasContentSize) totalUncomp += fi.uncompressedBytes; else allHaveSize = false; }
        auto& f0 = frames[0];
        std::string fname = path;
        auto slash = fname.rfind('/');
        if (slash != std::string::npos) fname = fname.substr(slash + 1);
        std::string fnameCol = std::string(CC_BWHITE) + fname + CC_RESET;
        if (g_verbosity >= VERBOSE) {
            fprintf(stdout, "%s%-7s  %-8s  %-5s  %-11s  %-12s  %-8s  %s%s\n",
                CC_BOLD, "Frames", "Type", "Block", "Compressed", "Uncompressed", "Ratio", "Filename", CC_RESET);
            for (size_t i = 0; i < frames.size(); i++) {
                auto& fi = frames[i];
                char frameLbl[32]; snprintf(frameLbl, sizeof(frameLbl), "frame%-2zu", i + 1);
                fprintf(stdout, "  %s  %s  %s  %s  %s  %s\n",
                    colCell(CC_DIM, frameLbl, 7).c_str(),
                    colCell(CC_DIM, "LZ4Frame", 8).c_str(),
                    colCell(CC_DIM, blockLabel(fi.blockMaxSizeId, fi.blockIndependence), 5).c_str(),
                    fmtComp(fi.compressedBytes, 11).c_str(),
                    fmtUncomp(fi.uncompressedBytes, fi.hasContentSize, 12).c_str(),
                    fmtRatio((double)fi.compressedBytes, fi.hasContentSize ? (double)fi.uncompressedBytes : 0.0, 8).c_str());
            }
        }
        fprintf(stdout, "%s  %s  %s  %s  %s  %s  %s\n",
            colCell(CC_DIM, std::to_string(frames.size()), 7).c_str(),
            colCell(CC_DIM, "LZ4Frame", 8).c_str(),
            colCell(CC_DIM, blockLabel(f0.blockMaxSizeId, f0.blockIndependence), 5).c_str(),
            fmtComp(fileBytes, 11).c_str(),
            fmtUncomp(totalUncomp, allHaveSize, 12).c_str(),
            fmtRatio((double)fileBytes, allHaveSize ? (double)totalUncomp : 0.0, 8).c_str(),
            fnameCol.c_str());
        return true;
    }

    void printListHeader() {
        fprintf(stdout, "%s%-7s  %-8s  %-5s  %-11s  %-12s  %-8s  %s%s\n",
            CC_BOLD, "Frames", "Type", "Block", "Compressed", "Uncompressed", "Ratio", "Filename", CC_RESET);
    }

    void printVersion() {
        std::cout << "gzl4 " << VERSION <<
            " - GPU, CPU, and Hybrid LZ4 Compression Tool\n"
            "Built with nvCOMP 5.1.x, CUDA 12.8, liblz4" << std::endl;
    }

    void printChangelog() {
        std::cout << "gzl4 " << VERSION << " - Changelog\n\n"
R"CL(Changelog:
  v3.32.2  writev periodic flush to fix decompressor stalls.
           All three decompressors now flush every 64 blocks (~256 MB at
           level 9) inside the inner drain loop rather than accumulating
           all consecutive ready blocks into one giant writev call.
           Prevents multi-second writer stalls when a full GPU batch
           completes simultaneously.

  v3.32.1  writev use-after-free fixes in compression writer.
           Root cause: task = std::move(it->second) destroyed the previous
           task's chunk vectors while metas/iovecs still held pointers into
           them.  Fix: single-task-per-writev (no cross-task coalescing),
           stable headers[] via resize() before any &headers[i] is taken,
           hash work pushed after writev with data kept alive in task.
           Also fixed decompressor clear-before-flush ordering bug.

  v3.32.0  AsyncWriter overhaul + batch size auto-tuning improvements.

           AsyncWriter (compression path):
             writev() coalescing: iovecs point directly at chunk data,
               eliminating the intermediate 256 MB writeBuf memcpy.
               Consecutive ready chunks coalesced into one writev() call.
             unordered_map: O(1) pendingWrites insert/find vs O(log n).
             Lock released before all I/O  no mutex held during writev().
             F_SETPIPE_SZ: pipe buffer enlarged to 1 MB on stdout/pipe.
             Orphaned copy of old AsyncWriter removed (271 lines dead code).

           Batch size auto-tuning:
             Removed GPU-count heuristic (1 GPU→64, ≤4→16, 5+→4).
             VRAM-driven formula: 85% freeVRAM / pipelineDepth / 5×chunk.
             Latency cap: max(4, 512MB / chunkSize)  keeps GPU startup
               fast regardless of level.  Level 9 (4MB) → 128, level 1
               (256KB) → 2048.  Matches empirical optimum automatically.
             File-size cap: no point allocating beyond total work needed.
             pipelineDepth always 3 (matches H100 copy engine count);
               removed spurious 1-GPU→4 special case.
             SM floor removed from refreshGPUMemoryAndBatchSize.

  v3.31.0  CPU compression during GPU slot init + larger input pool.

           Hybrid compressor CPU Phase 1:
             CPU workers start immediately before slotInitThreads.join(),
             pulling directly from asyncReader while totalGpuSlots==0.
             This overlaps GPU cudaMalloc (~2.6s) with CPU compression.
             Phase 2: normal cpuWorkQueue drain + GPU overflow routing.

           GPU slot overflow routing (gpuFreeSlots):
             Dispatcher routes to CPU when totalGpuSlots==0 (GPU init)
             or gpuFreeSlots==0 (all streams in-flight).  gpuFreeSlots
             incremented on slot clear, decremented on batch launch.

           Input pool sized from available RAM:
             60% of free physical RAM (sysinfo freeram, no swap).
             Retry loop halves on cudaHostAlloc failure.

           Default compression level changed from 1 to 9 (4MB chunks).

  v3.30.0  NVML-based GPU load balancing + XXH32 off write path.

           GPU load balancing (hybrid decompressor):
             NVMLHandle: dynamically loads libnvidia-ml.so via dlopen 
               binary runs normally on systems without NVML.
             GPULoadMonitor: background thread polls NVML every 2s,
               computing loadScore = 0.7×smUtil + 0.3×memUtil per GPU.
               Translates to streamPct/batchPct capacity fractions stored
               as atomics on GPUDevice (four tiers: 100/75/50/25/0%).
             Per-GPU work queues: each GPU worker has its own TsQueue.
               Dispatcher routes blocks via pickGPU()  reads two atomics
               per GPU (~8ns for 8 GPUs), picks least-loaded with capacity.
             GPU worker scaling: after cudaSetDevice each worker reads its
               own streamPct/batchPct and scales effectiveStreams, SC, and
               effectiveBatchSz down proportionally.  A GPU at 60% load
               (e.g. LLM training) gets 50% streams/batch  finishes faster,
               stays in sync with idle GPUs, preventing writer stalls.
             CPU steal (Phase 2): now steals from the deepest GPU queue
               rather than always queue[0], equalizing backlog.
             Overloaded GPUs (streamPct=0): worker drains its queue to CPU
               and exits rather than competing with the loaded GPU's tasks.

           XXH32 checksum off write path (all three decompressors):
             GPU-only, hybrid, and CPU decompressors now run checksum in a
             dedicated cksumThread that walks ready[] independently.
             Writer waits via atomic yield-spin (cksumConsumed) before
             clearing each block's buffer  zero sleep, near-zero overhead.
             Checksum and write now run in parallel: hash thread reads the
             same buffer the kernel is writing, effectively free on NVMe.

  v3.29.0  Zero-wait decompression pipeline + progress overhaul.

           GPU/Hybrid decompression now starts immediately with no blocking
           wait for GPU initialization.

           GPU enumeration (enumerateGPUs):
             Uses cudaGetDeviceProperties only  no cudaSetDevice, no context
             creation.  Takes <5ms for 8 GPUs vs ~2300ms previously.  GPU
             workers call cudaMemGetInfo themselves after cudaSetDevice to
             refresh availableMemory with the accurate free VRAM figure.

           Pre-decompression reader:
             Starts reading the compressed file before GPU enumeration, so
             compressed data is already queued when the decompressor starts.
             decompressFile() is called immediately after enumerateGPUs().

           Hybrid CPU Phase 1 / Phase 2:
             Phase 1: while GPUs are still in cudaMalloc, CPU workers service
               gpuWorkQueue directly via blocking pop()  no polling, no timers.
               GPU workers signal gpusReady after slot allocation; Phase 1 exits
               cleanly on the next iteration check.
             Phase 2: normal protocol  CPUs handle cpuWorkQueue; steal from
               gpuWorkQueue only when depth exceeds gpuStealThreshold (2 full
               batches per GPU).

           Pass-through block accounting:
             Hybrid: uncompressed blocks credited to cpuBlocks.
             GPU-only: uncompressed blocks credited to gpuBlocks.

           Progress display:
             Initializing / Reading / Decompressing / Writing phases shown.
             Decompressing: shows GPU: and CPU: byte counters throughout.
             Transitions to Writing: when decompression done, writer draining.
             All trailing-space clears replaced with CC_EL (\\033[K).
             g_progressActive: VLOG clears the progress line before printing
               so log messages appear cleanly between bar updates.
             Progress bars visible at -v and -vv, not only at default level.
             "Using temporary file" message moved to -v level.

           initializeGPUs() parallelizes per-GPU CUDA context creation 
           all GPUs init concurrently instead of serially.

  v3.28.2  Read-ahead pipeline: readers and CPU workers start immediately,
           before GPU initialization.  No blocking anywhere for capacity.

           Philosophy: read everything the OS will give you at full speed.
           The only thing that stops a reader is OOM (caught via bad_alloc
           and converted to a clean error), never a backpressure gate.

           GPU-only decompressor:
             Reader thread now launches before GPU worker threads.  By the
             time cudaMalloc/cudaStreamCreate complete, the blockQueue
             already has compressed blocks waiting.  The blocking HWM gate
             on blockQueue (blockQueueCV.wait) has been removed entirely 
             blocks push unconditionally at full disk/pipe read speed.

           Hybrid decompressor:
             Launch order is now: progress thread → dispatcher (reader) →
             CPU workers → GPU workers.  CPU workers drain the cpuWorkQueue
             (zero-threshold blocks) immediately while GPU workers are still
             initializing.  The gpuQueueCap size-check and spill-to-CPU-on-
             full logic have been removed  the dispatcher always pushes to
             the GPU queue; only the zero-threshold heuristic routes to CPU.

           Hybrid compressor:
             Same gpuQueueCap size-check and CPU spill-on-full removed from
             the dispatcher.  All non-zero chunks go straight to GPU queue.

           CPU decompressor:
             RAW_HWM constant and rawCV.wait(HWM) gate removed from the
             reader thread.  Blocks push unconditionally.

           TsQueue:
             push_wait() removed  zero call sites after the above changes.

           OOM protection:
             All unbounded reader pushes wrapped in try/catch(bad_alloc)
             which sets the appropriate error flag and breaks the read loop,
             allowing in-flight work to drain and the pipeline to shut down
             cleanly rather than crashing with an unhandled exception.

  v3.28.1  Bugfix: --cpu-only compression hung after "Decompression complete:".

           Three bugs in CPUCompressionPool combined to cause a deadlock:

           1. finish() set shouldStop, which is also the workers' exit signal.
              waitForResult()'s predicate woke on shouldStop even when the
              target chunk hadn't been posted yet.  The drain loop exited
              early, the pool went out of scope, stop() joined the still-
              running workers  but asyncWriter was waiting for chunks that
              were never enqueued, so it hung indefinitely.
              Fix: finish() now sets allJobsSubmitted (new flag) instead of
              shouldStop.  Workers keep running until the job queue is empty;
              only stop()/destructor sets shouldStop to signal workers to exit.

           2. workers called resultCV.notify_all() before decrementing
              activeJobs, so waitForResult's "all done" predicate always saw
              activeJobs > 0 for the last job and went back to sleep, never
              to wake again.
              Fix: activeJobs-- now happens before resultCV.notify_all().

           3. waitForResult's predicate held resultMutex and then tried to
              acquire jobMutex  a lock-order inversion.
              Fix: predicate checks activeJobs atomically first (no lock),
              then acquires jobMutex only after confirming activeJobs == 0.

  v3.28.0  Eliminated all polling and timed waits from hot paths.

           Every wait in the compression and decompression pipelines is now
           event-driven via condition variables.  No thread ever spins or
           sleeps waiting for work  it blocks and wakes exactly when data
           arrives.  The only remaining sleep_for calls are in progress
           display threads (150ms/200ms screen refresh rate), which are
           intentionally timer-driven.

           TsQueue: split pop() into two methods.
             pop()         fully blocking; wakes on push() or close().
             pop_for(ms)   timed variant; retained as a method but has
                            zero call sites (available for future use).
           All consumers updated from pop_for() to blocking pop().

           AsyncWriter::hashLoop: replaced 50ms poll loop with blocking
           pop() that wakes exactly when a HashWork item is enqueued.

           CPUCompressionPool:
             waitForResult()  removed timeout parameter; now uses a plain
               resultCV.wait() that blocks until the specific chunk index
               is posted by a worker.
             finish()  new method called by the reader thread after all
               jobs are submitted.  Notifies both jobCV (wakes idle workers)
               and resultCV (wakes the drain loop) so waitForResult() exits
               cleanly after the last result without a spurious timeout.

           compressFileCPU main loop redesigned as a true pipeline:
             Before: single thread alternated between a blocking getChunk()
               read phase and a 5ms-timeout waitForResult() drain phase.
               While blocked in the read phase, finished compression results
               piled up in the pool undelivered to the writer.
             After: readerSubmitThread reads and submits chunks concurrently
               with a drain loop on the main thread that blocks on
               waitForResult() and enqueues results to the writer immediately.
               Reader, CPU compressors, and writer all run in true parallel.

           Hybrid compressor GPU/CPU workers: all pop_for() calls replaced
           with blocking pop().  close() already called notify_all() so
           workers wake immediately when the dispatcher closes the queues.

           Hybrid decompressor GPU/CPU workers: same pop_for() → pop()
           conversion.  The GPU batch-fill inner loop and CPU worker both
           now block until a block arrives or the queue closes.

           All three decompressor writer loops: resultCV.wait_for(5ms)
           replaced with resultCV.wait().  The predicate covers all
           termination conditions (decompError, readDone && blocksDone >=
           totalBlocks) so the 5ms timeout was purely defensive and added
           latency between the last block arriving and the loop exiting.

  v3.27.2  SIMD-accelerated isAllZeros() for AVX2 and SSE2.

           The scalar isAllZeros() implementation (8 bytes/iteration) becomes
           a measurable bottleneck at large scale.  For a 500 GB file with 50%
           zero content, scanning all zero blocks takes ~250s at scalar speed,
           which partially offsets the write() time saved by sparse holes.

           Three-tier implementation selected at compile time:

           AVX2 (32 bytes/iteration, ~8x scalar):
             ORs 32-byte vectors into an accumulator, tests with
             _mm256_testz_si256 every 128 bytes.  Two accumulators allow
             the CPU to issue two 256-bit OR ops per cycle.  Enabled when
             __AVX2__ is defined (set automatically by -march=native on
             any modern x86-64 CPU including the RTX 5090 host).
             Scan time for 256 KB zero block: ~62 µs.

           SSE2 (16 bytes/iteration, ~4x scalar):
             Same OR-accumulate pattern using __m128i.  Uses
             _mm_movemask_epi8(_mm_cmpeq_epi8(acc, zero)) since SSE2
             lacks the testz instruction added in SSE4.1.
             Scan time for 256 KB zero block: ~125 µs.

           Scalar fallback (8 bytes/iteration):
             Original implementation, used on non-x86 or when SIMD is
             explicitly disabled.
             Scan time for 256 KB zero block: ~500 µs.

           Correctness is unchanged: isAllZeros() always checks actual
           decompressed bytes.  The <2% compressed-size routing heuristic
           is a performance hint only and never gates the sparse lseek.
           A block of all 0x01 bytes compresses similarly to all 0x00 bytes
           but isAllZeros() correctly returns false for it regardless.

           The SIMD includes (<immintrin.h>, <emmintrin.h>) are at file
           scope guarded by the same __AVX2__/__SSE2__ macros.  No CMake
           changes needed: -march=native in the existing compile flags sets
           these macros automatically based on the build host's CPU.

  v3.27.1  Hybrid decompressor: route tiny compressed blocks to CPU.

           In the hybrid decompressor dispatcher, any compressed block
           smaller than 2% of chunkSize is sent to a CPU worker instead
           of the GPU queue.  A fully-zero LZ4 block compresses to 12-20
           bytes regardless of chunk size, so this threshold reliably catches
           zero blocks (and near-zero blocks) while leaving all real data on
           the GPU path.

           Without this change, zero blocks still went through the full GPU
           pipeline  H→D transfer, nvCOMP kernel, D→H transfer  before the
           writer could call isAllZeros() and lseek() past them.  The PCIe
           round-trip dominated, giving hybrid roughly the same decompress
           throughput as gpu-only on zero-heavy data despite the sparse-write
           optimisation being in place.

           With this change, zero blocks skip the GPU entirely and decompress
           locally on a CPU core in microseconds.  The writer receives the
           result almost immediately and lseeks past it.  Hybrid decompress
           on zero-heavy data should now approach cpu-only speeds (~4.9 GiB/s
           observed vs ~3.5 GiB/s with gpu-only).

           False positives (a genuinely non-zero block that compressed to
           under 2%) are handled correctly  they decompress on CPU instead
           of GPU, producing identical output with slightly different
           throughput routing.  No correctness risk.

  v3.27.0  Sparse-file decompression + zero-chunk compression routing.

           Decompression (all three backends  CPU, GPU, hybrid):
           After each block is decompressed, isAllZeros() scans the output.
           If the block is entirely zeros, lseek(SEEK_CUR) punches a sparse
           hole in the output file instead of calling write().  On ext4, xfs,
           btrfs, and other sparse-capable filesystems the OS records no
           physical blocks for the hole; reads return zeros automatically.
           ftruncate() is called before close() to extend the file to its
           full uncompressed size in case the last block(s) were holes
           (lseek moves the cursor but does not set the file's official size).
           canSparse is set to false for stdout, pipes, and test mode so the
           optimization never fires where it cannot apply.  lseek failure
           falls back to a normal write with a warning message.
           -v output reports "Sparse holes: X.X GB skipped (Y.Y%)" when the
           optimization fires so the savings are visible.

           Compression (hybrid backend dispatcher):
           Before routing each input chunk to the GPU or CPU work queue, the
           dispatcher calls isAllZeros().  Zero chunks bypass the GPU queue
           entirely and go straight to CPU workers.  LZ4 compresses a 4 MB
           zero block in microseconds on a single CPU core (one hash-table
           entry, one copy command); sending it to the GPU wastes a full
           H→D + kernel + D→H PCIe round-trip for no gain.

           isAllZeros() helper:
           Aligns to 8-byte boundary then scans 8 bytes per iteration with
           an early exit on the first non-zero word.  For real data (almost
           never all zeros) the cost is one or two iterations.  For a full
           4 MB zero block the scan takes ~500 µs, well under the 4 MB
           write() or PCIe transfer it replaces.

  v3.26.7  Default: skip fsync() on output files for consistent performance.

           Consumer NVMe drives have an SLC write cache that fills during a
           write and then flushes to slower TLC/QLC NAND. When fsync() is
           called at the end of a run it blocks until the drive finishes that
           flush. Because the flush takes about as long as the previous run,
           alternating runs see 4-20x timing variance (e.g. 4.6 s / 20 s /
           4.7 s / 19.8 s). The effect is consistent, reproducible, and has
           nothing to do with CPU, GPU, or compression performance.

           The output was already written atomically (to a .tmp file that is
           renamed on success), so the file is always either complete or absent
           after a crash regardless of fsync(). The durability guarantee from
           fsync() -- that data survives a power failure between write() and
           close() -- is only needed for mission-critical archival workflows.

           New option: --sync-output calls fsync() before closing the output
           file, restoring the old behaviour for users who need that guarantee.
           Documented in --help under BASIC OPTIONS.

  v3.26.6  Default compression level changed from -9 to -1; --best now
           implements true maximum compression.

  v3.26.5  Feature: multi-file command line (compress/decompress N files).

  v3.26.4  Feature: --list: display .lz4 frame metadata.

  v3.26.3  Feature: --content-size / --no-content-size.
  v3.26.2  Version bump.

  v3.26.1  Bugfix: test suite fixes + informational-command exit-code fix
           + -t exits 1 on checksum mismatch + empty-file decompress hang fix.

  v3.26.0  Build + UX overhaul + decompressor segfault fix + GPU-only perf fix.

  v3.25.0  Build: resolved all compiler warnings from v3.24.24.
  v3.24.24 Feature: tar -I "./build/gzl4" compatibility.
  v3.24.23 UX: added --progress flag.
  v3.24.22 UX: default to quiet mode (-q) when input is a pipe/stdin.
  v3.24.21 Bugfix: stdin compression stuttered at the end.
  v3.24.20 Perf: stdin pipe compression was ~3x slower than direct file.
  v3.24.19 Bugfix: stdin compression produced 0-byte output.
  v3.24.18 Bugfix: stdout compression produced a 23-byte file (header only).
  v3.24.17 Bugfix: stdin compression failed with "Cannot stat input file: -".
  v3.24.16 UX: full color treatment for all three compression paths.
  v3.24.15 Bugfix: 5 remaining -Wformat-extra-args.
  v3.24.14 Bugfix: 17 -Wformat-extra-args warnings.
  v3.24.13 UX: three final polish items.
  v3.24.12 UX: three-phase progress display for all three decompressors.
  v3.24.11 Bugfix: progress display appeared frozen then jumped.
  v3.24.10 Bugfix: deadlock/hang in --gpu-only and --hybrid decompression.
  v3.24.9  Perf: parallel XXH32 hash thread in AsyncWriter.
  v3.24.8  Reverted v3.24.7 original-data copy elimination.
  v3.24.4  Perf: two-phase D->H transfer in GPU-only compression worker.
  v3.24.0  Perf: lock-free result store for both decompressors.
  v3.23.3  Perf: eliminated all polling sleeps from hot paths.
  v3.23.2  Perf: two fixes for decompressFileGPU.
  v3.23.1  Perf: decompressFileGPU dedicated reader thread.
  v3.23.0  Perf: pinned host output staging for decompressors.
  v3.22.0  Hybrid decompression fixes.
  v3.21.0  True hybrid decompression.
  v3.19.x  Multiple fixes and features.
  v3.18.0  LZ4-compatible -z flag.
  v3.17.x  Verbosity and quiet mode improvements.
  v3.14.0  Hybrid mode v3: GPU-priority dispatcher.
  v3.13.x  HC compression levels -10/-11/-12.
  v3.12.x  Hybrid rewrite.
  v3.11.x  GPU decompression via nvCOMP.
  v3.10.x  Command-line tuning params.
  v3.9.x   GPU worker threads, pinned input pool.
  v3.7.0   CPU-only mode, hybrid mode.
  v3.3.0   Async reader with readahead.
  v3.0.0   True parallel multi-GPU processing.
)CL" << std::endl;
    }

    bool run(int argc, char* argv[]) {
        setupSignalHandlers();

        if (argc == 1) {
            if (isatty(STDOUT_FILENO)) { printHelp(); return false; }
            inputFile = "-"; outputFile = "-"; stdoutMode = true; keepOriginal = true;
        }

        preprocessArgv(argc, argv);

        if (!parseArguments(argc, argv)) {
            return earlyExit;
        }

        if (inputFile == "-" && g_verbosity == NORMAL && !forceProgress) {
            g_verbosity = QUIET;
        }
        if (!contentSizeExplicit && inputFile == "-") {
            storeContentSize = false;
        }

        setChunkSizeFromLevel();

        if (!listMode) {
            if (backendMode != BackendMode::CPU_ONLY) {
                if (decompress && inputFile != "-") {
                    // ── Decompression fast path ───────────────────────────────
                    // 1. Start pre-reader immediately (reads compressed file)
                    // 2. Enumerate GPUs quickly (no context creation, <10ms)
                    // 3. Call decompressFile() right away  the dispatcher and
                    //    CPU workers start before GPU workers finish cudaMalloc.
                    //    The pre-reader's reading progress is shown until the
                    //    decompressor's own progress thread takes over.
                    int preFd = ::open(inputFile.c_str(), O_RDONLY | O_LARGEFILE);
                    if (preFd >= 0) {
                        posix_fadvise(preFd, 0, 0, POSIX_FADV_SEQUENTIAL);
                        posix_fadvise(preFd, 0, 0, POSIX_FADV_WILLNEED);
                        uint8_t hdrBuf[32];
                        ssize_t hdrRead = ::read(preFd, hdrBuf, 32);
                        if (hdrRead >= 15) {
                            std::vector<uint8_t> hdrBytes(hdrBuf, hdrBuf + hdrRead);
                            struct stat pst;
                            size_t cfsz = (fstat(preFd, &pst) == 0) ? (size_t)pst.st_size : 0;
                            LZ4Frame::FrameDescriptor pdesc;
                            size_t hdrConsumed = 0;
                            {
                                std::string hs((char*)hdrBuf, hdrRead);
                                std::istringstream hss(hs, std::ios::binary);
                                if (LZ4Frame::readFrameHeader(hss, pdesc))
                                    hdrConsumed = hss.tellg();
                            }
                            if (hdrConsumed > 0) {
                                lseek(preFd, (off_t)hdrConsumed, SEEK_SET);
                                preDecompReader.start(preFd, false, cfsz, hdrBytes, hdrConsumed);
                                preDecompReaderStarted = true;
                                VLOG(VERBOSE, "Pre-read started: reading %.2f GB compressed file\n",
                                     cfsz / (1024.0*1024.0*1024.0));
                            } else {
                                close(preFd);
                            }
                        } else {
                            close(preFd);
                        }
                    }

                    // GPU enumeration: cudaGetDeviceProperties only, no context
                    // creation.  Typically <5ms.  Decompressor starts immediately after.
                    // If fast enumeration fails (can happen in some driver environments
                    // where cudaGetDeviceCount needs a context first), fall back to the
                    // full initializeGPUs() which does proper context creation.
                    if (!enumerateGPUs()) {
                        VLOG(VERBOSE, "Fast GPU enumeration failed  falling back to full init\n");
                        gpus.clear();
                        if (!initializeGPUs()) {
                            fprintf(stderr, "Warning: GPU initialization failed, falling back to CPU-only mode\n");
                            backendMode = BackendMode::CPU_ONLY;
                            if (preDecompReaderStarted) preDecompReader.stop();
                        }
                    }
                    // decompressFile() called below  CPUs begin decompressing
                    // from the pre-read queue immediately on entry.

                } else {
                    // ── Compression or stdin decompression: full GPU init ─────
                    if (!decompress && inputFile != "-") {
                        // Compression early reader (unchanged)
                        struct stat st;
                        if (stat(inputFile.c_str(), &st) == 0 && st.st_size > 0) {
                            size_t fSize   = (size_t)st.st_size;
                            size_t nChunks = (fSize + chunkSize - 1) / chunkSize;
                            VLOG(VERBOSE, "Early reader: starting %.2f GB file read-ahead "
                                 "(%zu chunks, unlimited queue) while GPUs initialise\n",
                                 fSize / (1024.0*1024.0*1024.0), nChunks);
                            earlyReader.start(inputFile, chunkSize, SIZE_MAX);
                            earlyReaderStarted = true;
                        } else if (inputFile == "-") {
                            VLOG(VERBOSE, "Early reader: buffering stdin while GPUs initialise\n");
                            earlyReader.start(inputFile, chunkSize, 256);
                            earlyReaderStarted = true;
                        }
                    }
                    auto gpu_init_t0 = std::chrono::high_resolution_clock::now();
                    VLOG(DEBUG, "[run] calling initializeGPUs\n");
                    if (!initializeGPUs()) {
                        fprintf(stderr, "Warning: GPU initialization failed, falling back to CPU-only mode\n");
                        backendMode = BackendMode::CPU_ONLY;
                    }
                    VLOG(DEBUG, "[run] initializeGPUs done (%.1f ms)\n",
                         std::chrono::duration<double>(
                             std::chrono::high_resolution_clock::now() - gpu_init_t0).count() * 1000.0);
                }
            } else {
                VLOG(VERBOSE, "CPU-only mode: skipping GPU initialization\n");
            }

            if (!decompress && backendMode != BackendMode::CPU_ONLY && !gpus.empty()) {
                // Estimate numChunks from file size if known (stdin → 0 = unknown)
                size_t estChunks = 0;
                if (inputFile != "-") {
                    struct stat fst;
                    if (stat(inputFile.c_str(), &fst) == 0 && fst.st_size > 0)
                        estChunks = ((size_t)fst.st_size + chunkSize - 1) / chunkSize;
                }
                batchSize = calculateBatchSize(gpus[0].availableMemory, estChunks);
                VLOG(VERBOSE, "Batch/stream config: %d streams/GPU × %zu chunks/slot = "
                     "%.1f GB VRAM/slot  (%.1f MB chunks, %zu GPUs)\n",
                     gpus[0].pipelineDepth, batchSize,
                     (batchSize * chunkSize * 5) / (1024.0*1024.0*1024.0),
                     chunkSize / (1024.0*1024.0),
                     gpus.size());
            }
        }

        bool success;
        if (listMode) {
            printListHeader();
            success = true;
            for (auto& path : listFileArgs) { if (!listFile(path)) success = false; }
        } else {
            std::vector<std::string> allFiles;
            allFiles.push_back(inputFile);
            for (auto& fi : extraInputFiles) allFiles.push_back(fi);

            const bool decompressFlag  = decompress;
            const bool forceOverwriteF = forceOverwrite;
            const bool keepOriginalF   = keepOriginal;
            const bool testModeF       = testMode;
            const bool stdoutModeF     = stdoutMode;

            success = true;
            for (size_t fi = 0; fi < allFiles.size(); fi++) {
                inputFile      = allFiles[fi];
                if (fi > 0) { outputFile = ""; tempOutputFile = ""; earlyReaderStarted = false; preDecompReaderStarted = false; }
                decompress     = decompressFlag;
                forceOverwrite = forceOverwriteF;
                keepOriginal   = keepOriginalF;
                testMode       = testModeF;
                stdoutMode     = stdoutModeF;

                if (!contentSizeExplicit) storeContentSize = (inputFile != "-");

                if (fi > 0) {
                    struct stat st;
                    if (stat(inputFile.c_str(), &st) != 0) {
                        fprintf(stderr, "Error: Cannot stat '%s': %s\n", inputFile.c_str(), strerror(errno));
                        success = false; continue;
                    }
                    bool hasLz4Ext = (inputFile.size() > 4 && inputFile.substr(inputFile.size()-4) == ".lz4");
                    if (!forceMode) decompress = hasLz4Ext;
                    if (stdoutMode) {
                        outputFile = "-";
                    } else if (decompress) {
                        if (!hasLz4Ext) { fprintf(stderr, "Error: '%s' has no .lz4 extension; skipping\n", inputFile.c_str()); success = false; continue; }
                        outputFile = inputFile.substr(0, inputFile.size()-4);
                    } else {
                        outputFile = inputFile + ".lz4";
                    }
                    if (!testMode && !forceOverwrite && !stdoutMode && stat(outputFile.c_str(), &st) == 0) {
                        fprintf(stderr, "Error: Output already exists: %s (use -f to overwrite)\n", outputFile.c_str());
                        success = false; continue;
                    }
                    if (forceOverwrite && !stdoutMode && !testMode) tempOutputFile = outputFile + ".tmp";
                }

                VLOG(DEBUG, "[run] calling %s for %s\n",
                     decompress ? "decompressFile" : "compressFile",
                     inputFile.c_str());
                bool fileOk = decompress ? decompressFile() : compressFile();
                if (fileOk && !tempOutputFile.empty()) fileOk = renameTempToFinal();
                if (!fileOk) { cleanupTempFile(); success = false; }
            }
        }

        for (size_t i = 0; i < gpus.size(); i++) {
            cudaSetDevice(gpus[i].deviceId);
            cudaDeviceSynchronize();
        }

        return success;
    }
};

// Initialize static member
GZL4Compressor* GZL4Compressor::g_instance = nullptr;

/*
 * Main entry point
 */
int main(int argc, char* argv[]) {
    detectColor();
    try {
        GZL4Compressor compressor;
        bool result = compressor.run(argc, argv);
        return result ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal error: %s\n", e.what());
        return EXIT_FAILURE;
    } catch (...) {
        fprintf(stderr, "Fatal error: Unknown exception\n");
        return EXIT_FAILURE;
    }
}
