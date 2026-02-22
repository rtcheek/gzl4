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

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
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
#include <fcntl.h>
#include <getopt.h>
#include <signal.h>

// Configuration constants
constexpr const char* VERSION = "3.23.0";

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

// Macro for verbosity-aware output
#define VLOG(level, ...) do { \
    if (g_verbosity >= level) { \
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
    
    bool writeFrameHeader(std::ostream& out, size_t contentSize, size_t chunkSizeBytes) {
        // Write magic number
        writeU32(out, LZ4_MAGIC);
        
        // Build frame descriptor
        FrameDescriptor desc;
        desc.contentSize = contentSize;
        desc.hasContentSize = true;
        desc.hasContentChecksum = true;
        desc.hasBlockChecksum = false;
        desc.blockIndependence = true;
        desc.blockMaxSize = calculateBlockMaxSize(chunkSizeBytes);  // Calculate from actual chunk size!
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

    GPUDevice(int id) : deviceId(id), pipelineDepth(3), optimalBatch(64), smCount(1) {}

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
    void* h_input_ptrs_pinned  = nullptr;     // pinned host copy of d_input_ptrs array
    void* h_input_sizes_pinned = nullptr;     // pinned host copy of d_input_sizes array
    void* h_output_ptrs_pinned = nullptr;     // pinned host copy of d_output_ptrs array
};

// ── Pre-allocated GPU compression slot ──────────────────────────────────────
// All device and pinned-host memory is allocated once at startup.
// Zero cudaMalloc / cudaFree calls during compression hot path.

/*
 * PinnedInputPool  pre-allocated pool of pinned host memory slots.
 *
 * One big cudaHostAlloc is divided into N equal slots.  The reader acquires
 * slots and reads directly into them; the GPU's copy engine then transfers
 * from pinned memory to device via DMA without CPU involvement.  Slots are
 * released back to the pool as soon as the GPU worker no longer needs the
 * original data, allowing the reader to run non-stop without any artificial
 * queue cap.
 */
class PinnedInputPool {
public:
    // Reference-counted handle to one pool slot.
    // Destructor automatically releases the slot back to the pool.
    struct Handle {
        uint8_t*        data     = nullptr;  // pinned pointer into pool base
        size_t          size     = 0;        // valid bytes for this chunk
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
            fprintf(stderr, "PinnedInputPool: cudaHostAlloc(%zu × %zu MB) failed: %s\n",
                    nSlots, slotSize>>20, cudaGetErrorString(err));
            return false;
        }
        freeList_.resize(nSlots);
        std::iota(freeList_.begin(), freeList_.end(), 0);
        VLOG(DEBUG, "PinnedInputPool: %zu × %.1f MB = %.1f GB pinned\n",
             nSlots, slotSize/(1024.0*1024.0),
             (nSlots*slotSize)/(1024.0*1024.0*1024.0));
        return true;
    }

    void destroy() {
        if (base_) { cudaFreeHost(base_); base_ = nullptr; }
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

/*
 * Asynchronous Reader with Advanced I/O
 * Reads file chunks in background thread while GPUs initialize and compress.
 *
 * When a PinnedInputPool is provided (via startPooled), the reader acquires
 * pinned slots and reads directly into them  zero extra copies, and the GPU
 * can DMA from pinned memory without CPU involvement.  The reader only blocks
 * when all pool slots are in use (natural backpressure from GPU throughput),
 * never from an arbitrary queue cap.
 *
 * Without a pool (start()), falls back to heap-allocated vectors as before.
 */

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
    // and NEVER change after init, so we never copy them again.
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

    cudaStream_t    stream   = 0;
    bool            ready    = false;

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
    // Returns false when closed and empty (or on timeout)
    bool pop(T& out, int timeoutMs = 50) {
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


class AsyncReader {
public:
    struct ReadChunk {
        size_t chunkIndex = 0;
        size_t size       = 0;

        // Storage: either a pinned pool handle OR a heap vector.
        // GPU worker uses data() regardless.
        PinnedInputPool::Handle poolHandle;   // valid when using pool
        std::vector<uint8_t>    heapData;     // used in fallback mode

        uint8_t*       data()       { return poolHandle.valid() ? poolHandle.data : heapData.data(); }
        const uint8_t* data() const { return poolHandle.valid() ? poolHandle.data : heapData.data(); }
    };

private:
    std::thread  readerThread;

    // Ready-to-process queue (small  just coordination, not storage)
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

    void readerLoop() {
        size_t chunkIndex = 0;
        size_t totalRead  = 0;
        auto   t0         = std::chrono::high_resolution_clock::now();

        while (totalRead < fileSize && !shouldStop.load()) {
            size_t toRead = std::min(chunkSize, fileSize - totalRead);
            ReadChunk chunk;
            chunk.chunkIndex = chunkIndex;
            chunk.size       = toRead;

            if (pool_) {
                // ── Pooled path: acquire pinned slot, read directly into it ──
                // Blocks only if all slots are in use (GPU-paced backpressure).
                chunk.poolHandle = pool_->acquire();
                if (!chunk.poolHandle.valid()) break;  // shutdown
                chunk.poolHandle.size     = toRead;
                chunk.poolHandle.chunkIdx = chunkIndex;

                auto rs = std::chrono::high_resolution_clock::now();
                ssize_t n = ::read(inputFd, chunk.poolHandle.data, toRead);
                auto re = std::chrono::high_resolution_clock::now();
                if (n != (ssize_t)toRead) {
                    fprintf(stderr, "Reader: read error chunk %zu: %s\n",
                            chunkIndex, strerror(errno));
                    break;
                }
                totalReadTime = totalReadTime.load() +
                    std::chrono::duration<double>(re - rs).count();
            } else {
                // ── Fallback: heap allocation with queue-depth cap ────────────
                {
                    std::unique_lock<std::mutex> lk(queueMutex);
                    queueCV.wait(lk, [this]{
                        return readQueue.size() < maxQueuedChunks || shouldStop.load();
                    });
                    if (shouldStop.load()) break;
                }
                chunk.heapData.resize(toRead);

                auto rs = std::chrono::high_resolution_clock::now();
                ssize_t n = ::read(inputFd, chunk.heapData.data(), toRead);
                auto re = std::chrono::high_resolution_clock::now();
                if (n != (ssize_t)toRead) {
                    fprintf(stderr, "Reader: read error chunk %zu: %s\n",
                            chunkIndex, strerror(errno));
                    break;
                }
                totalReadTime = totalReadTime.load() +
                    std::chrono::duration<double>(re - rs).count();
            }

            totalRead  += toRead;
            bytesRead  += toRead;
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
            // Stdin pipe  file size unknown, can't pre-advise
            inputFd  = STDIN_FILENO;
            fileSize = 0;
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
        if (inputFd >= 0) { close(inputFd); inputFd = -1; }
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
 * Key design: all formatted block data (header + payload) is accumulated into
 * a large staging buffer before being flushed to disk in one big write() call.
 * This collapses thousands of small 2-syscall-per-chunk writes into a handful
 * of large sequential writes, saturating the NVMe write path.
 */
class AsyncWriter {
private:
    // ── Staging write buffer ──────────────────────────────────────────────────
    // 256 MB is large enough to hold ~65 chunks at 4 MB each and keeps the
    // number of write() syscalls under 40 for an 8 GB file.
    static constexpr size_t WRITE_BUF_SIZE = 256ULL * 1024 * 1024;

    std::vector<uint8_t> writeBuf;   // allocated once at start()
    size_t               writeBufUsed = 0;

    // Append bytes to staging buffer, flushing when full
    void bufAppend(const void* data, size_t len) {
        const uint8_t* src = reinterpret_cast<const uint8_t*>(data);
        while (len > 0) {
            size_t space = WRITE_BUF_SIZE - writeBufUsed;
            size_t copy  = std::min(len, space);
            memcpy(writeBuf.data() + writeBufUsed, src, copy);
            writeBufUsed += copy;
            src          += copy;
            len          -= copy;
            if (writeBufUsed == WRITE_BUF_SIZE) bufFlush();
        }
    }

    void bufFlushU32(uint32_t v) {
        uint8_t buf[4] = { uint8_t(v), uint8_t(v>>8), uint8_t(v>>16), uint8_t(v>>24) };
        bufAppend(buf, 4);
    }

    void bufFlush() {
        if (writeBufUsed == 0) return;
        auto writeStart = std::chrono::high_resolution_clock::now();
        ssize_t written = ::write(outputFd, writeBuf.data(), writeBufUsed);
        auto writeEnd   = std::chrono::high_resolution_clock::now();
        if (written != (ssize_t)writeBufUsed)
            fprintf(stderr, "Write error: %s\n", strerror(errno));
        // Tell kernel to drop these pages  we'll never re-read the output
        off_t pos = lseek(outputFd, 0, SEEK_CUR);
        if (pos >= (off_t)writeBufUsed)
            posix_fadvise(outputFd, pos - writeBufUsed, writeBufUsed, POSIX_FADV_DONTNEED);
        bytesWritten += writeBufUsed;
        totalWriteTime = totalWriteTime.load() +
            std::chrono::duration<double>(writeEnd - writeStart).count();
        writeBufUsed = 0;
    }

    // ── Per-chunk write task ──────────────────────────────────────────────────
    struct WriteTask {
        size_t chunkIndex;
        std::vector<std::vector<uint8_t>> compressedChunks;
        std::vector<std::vector<uint8_t>> originalChunks;
        std::vector<size_t> chunkIndices;
        std::vector<size_t> originalSizes;
    };

    std::thread writerThread;
    std::map<size_t, WriteTask> pendingWrites;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool>   shouldStop{false};
    std::atomic<size_t> bytesWritten{0};
    std::atomic<double> totalWriteTime{0.0};
    std::atomic<size_t> nextChunkToWrite{0};
    std::atomic<bool>   writerDone{false};
    std::atomic<size_t> totalExpectedChunks{SIZE_MAX};  // set before workers launch

    int         outputFd   = -1;
    std::string outputFile;
    XXH::State* xxhState   = nullptr;

    void writerLoop() {
        VLOG(DEBUG, "Writer thread started\n");
        auto threadStart = std::chrono::high_resolution_clock::now();

        while (true) {
            WriteTask task;
            bool hasTask = false;

            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCV.wait(lock, [this] {
                    return pendingWrites.count(nextChunkToWrite.load()) > 0
                           || shouldStop.load();
                });
                // Exit when: explicit stop() OR we've written every expected chunk
                bool allDone = nextChunkToWrite.load() >= totalExpectedChunks.load();
                if (pendingWrites.empty() && (shouldStop.load() || allDone)) break;
                auto it = pendingWrites.find(nextChunkToWrite.load());
                if (it != pendingWrites.end()) {
                    task    = std::move(it->second);
                    pendingWrites.erase(it);
                    hasTask = true;
                }
            }

            while (hasTask) {
                writeTask(task);
                nextChunkToWrite += task.chunkIndices.size();

                // Greedily drain any already-available consecutive chunks
                // without releasing/re-acquiring the lock between each one.
                // This avoids condvar overhead when the map is pre-populated.
                std::unique_lock<std::mutex> lock(queueMutex);
                auto it = pendingWrites.find(nextChunkToWrite.load());
                if (it != pendingWrites.end()) {
                    task = std::move(it->second);
                    pendingWrites.erase(it);
                    // hasTask stays true  continue draining
                } else {
                    hasTask = false;
                    lock.unlock();
                    queueCV.notify_all();
                }
            }
        }

        bufFlush();   // flush any remaining data
        writerDone.store(true);

        auto threadEnd = std::chrono::high_resolution_clock::now();
        double total   = std::chrono::duration<double>(threadEnd - threadStart).count();
        VLOG(VERBOSE, "Writer thread finished: wrote %zu bytes in %.2fs (%.2fs actual I/O, %.2fs waiting)\n",
             bytesWritten.load(), total, totalWriteTime.load(),
             total - totalWriteTime.load());
    }

    void writeTask(const WriteTask& task) {
        for (size_t i = 0; i < task.originalChunks.size(); i++) {
            size_t origSize = task.originalSizes[i];
            if (xxhState)
                xxhState->update(task.originalChunks[i].data(), origSize);

            bool hasCompressed = i < task.compressedChunks.size()
                                 && !task.compressedChunks[i].empty();
            if (hasCompressed && task.compressedChunks[i].size() < origSize) {
                uint32_t compSz = (uint32_t)task.compressedChunks[i].size();
                bufFlushU32(compSz);
                bufAppend(task.compressedChunks[i].data(), compSz);
            } else {
                uint32_t blockHdr = (uint32_t)origSize | 0x80000000u;
                bufFlushU32(blockHdr);
                bufAppend(task.originalChunks[i].data(), origSize);
            }
        }
    }
    
public:
    AsyncWriter() = default;
    
    ~AsyncWriter() {
        stop();
    }
    
    bool start(const std::string& filename, XXH::State* xxh) {
        outputFile = filename;
        xxhState = xxh;

        // Allocate write staging buffer once
        writeBuf.resize(WRITE_BUF_SIZE);
        writeBufUsed = 0;

        if (filename == "-") {
            outputFd = STDOUT_FILENO;
            VLOG(VERBOSE, "AsyncWriter: writing blocks to stdout\n");
        } else {
            // Open file with O_APPEND since header was already written by main thread
            outputFd = open(filename.c_str(), O_WRONLY | O_APPEND, 0644);
            if (outputFd < 0) {
                fprintf(stderr, "Error opening output file %s: %s\n", 
                        filename.c_str(), strerror(errno));
                return false;
            }
            posix_fadvise(outputFd, 0, 0, POSIX_FADV_SEQUENTIAL);
            VLOG(VERBOSE, "AsyncWriter: opened %s for appending blocks\n", filename.c_str());
        }
        
        shouldStop.store(false);
        writerThread = std::thread(&AsyncWriter::writerLoop, this);
        
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

    // Batch-insert all per-chunk tasks from one GPU batch under a single lock
    // acquisition.  Reduces mutex contention from O(batchSize) to O(1) per batch.
    void enqueueBatch(std::vector<std::vector<uint8_t>>& compressedChunks,
                      std::vector<std::vector<uint8_t>>& originalChunks,
                      const std::vector<size_t>&         chunkIndices,
                      const std::vector<size_t>&         originalSizes) {
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            for (size_t i = 0; i < chunkIndices.size(); i++) {
                WriteTask task;
                task.chunkIndex = chunkIndices[i];
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
        
        if (writerThread.joinable()) {
            writerThread.join();
        }
        
        if (outputFd >= 0 && outputFd != STDOUT_FILENO) {
            fsync(outputFd);
            close(outputFd);
            outputFd = -1;
        }
    }
    
    size_t getBytesWritten() const {
        return bytesWritten.load();
    }
    
    double getWriteTime() const {
        return totalWriteTime.load();
    }
    
    size_t getNextChunkToWrite() const { return nextChunkToWrite.load(); }
    bool   isDone()             const { return writerDone.load(); }
    void   setTotalChunks(size_t n)  { totalExpectedChunks.store(n); }

    size_t getQueueDepth() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(queueMutex));
        return pendingWrites.size();
    }
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
    std::atomic<size_t> activeJobs{0};
    std::atomic<size_t> totalJobsProcessed{0};
    
    size_t numThreads;
    
    // Worker thread function
    void workerThread() {
        while (true) {
            CompressJob job;
            
            // Get job from queue
            {
                std::unique_lock<std::mutex> lock(jobMutex);
                jobCV.wait(lock, [this] {
                    return !jobQueue.empty() || shouldStop.load();
                });
                
                if (shouldStop.load() && jobQueue.empty()) {
                    break;
                }
                
                if (jobQueue.empty()) continue;
                
                job = std::move(jobQueue.front());
                jobQueue.pop();
                activeJobs++;
            }
            
            // Compress using standard LZ4
            CompressResult result;
            result.chunkIndex = job.chunkIndex;
            result.originalSize = job.inputData.size();
            result.originalData = job.inputData;  // Always keep original
            result.compressedData.resize(job.maxOutputSize);
            
            // HC level > 0: use LZ4_compress_HC for levels -10 to -12
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
                // Compressed version is smaller - use it
                result.compressedData.resize(compSize);
                result.success = true;
            } else {
                // Original is smaller (or compression failed) - use original
                result.compressedData.clear();
                result.success = false;
            }
            
            // Store result
            {
                size_t storedIdx = result.chunkIndex;
                std::lock_guard<std::mutex> lock(resultMutex);
                completedJobs[storedIdx] = std::move(result);
                VLOG(DEBUG, "CPU worker: stored chunk %zu (pool now has %zu results)\n",
                     storedIdx, completedJobs.size());
            }
            resultCV.notify_all();
            
            activeJobs--;
            totalJobsProcessed++;
        }
    }
    
public:
    CPUCompressionPool(size_t threads = 0) {
        if (threads == 0) {
            numThreads = std::thread::hardware_concurrency();
            if (numThreads == 0) numThreads = 4;  // Fallback
        } else {
            numThreads = threads;
        }
        
        VLOG(VERBOSE, "CPU compression pool: %zu threads\n", numThreads);
        
        // Start worker threads
        for (size_t i = 0; i < numThreads; i++) {
            workers.emplace_back(&CPUCompressionPool::workerThread, this);
        }
    }
    
    ~CPUCompressionPool() {
        stop();
    }
    
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
    
    bool waitForResult(size_t chunkIndex, CompressResult& result, int timeoutMs = 100) {
        std::unique_lock<std::mutex> lock(resultMutex);
        
        bool found = resultCV.wait_for(lock, std::chrono::milliseconds(timeoutMs), [this, chunkIndex] {
            return completedJobs.count(chunkIndex) > 0;
        });
        
        if (found) {
            result = std::move(completedJobs[chunkIndex]);
            completedJobs.erase(chunkIndex);
            return true;
        }
        return false;
    }
    
    size_t getQueueDepth() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(jobMutex));
        return jobQueue.size();
    }
    
    size_t getActiveJobs() const {
        return activeJobs.load();
    }
    
    size_t getTotalProcessed() const {
        return totalJobsProcessed.load();
    }
    
    size_t getCompletedCount() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(resultMutex));
        return completedJobs.size();
    }
    
    // Returns the smallest chunk index currently in completedJobs, or SIZE_MAX if empty
    size_t getSmallestCompletedIndex() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(resultMutex));
        if (completedJobs.empty()) return SIZE_MAX;
        return completedJobs.begin()->first;
    }
    
    // Pull any one completed result, regardless of index. Returns false if none ready.
    bool drainOne(CompressResult& result) {
        std::lock_guard<std::mutex> lock(resultMutex);
        if (completedJobs.empty()) return false;
        auto it = completedJobs.begin();
        result = std::move(it->second);
        completedJobs.erase(it);
        return true;
    }
    
    void stop() {
        shouldStop.store(true);
        jobCV.notify_all();
        
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers.clear();
    }
};

/*
 * Main application class
 */
class GZL4Compressor {
private:
    std::vector<GPUDevice> gpus;
    size_t chunkSize;
    size_t batchSize;
    int compressionLevel;
    bool decompress;
    bool keepOriginal;
    bool forceOverwrite;
    bool forceMode;         // -z: force compression mode (ignore .lz4 extension auto-detection)
    bool stdoutMode;
    bool testMode;
    
    // Signal handler for cleanup
    static void signalHandler(int signum) {
        (void)signum;  // Suppress unused parameter warning
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
    
    // Get the actual file path to write to (temp if set, otherwise final)
    const char* getActualOutputPath() const {
        return tempOutputFile.empty() ? outputFile.c_str() : tempOutputFile.c_str();
    }
    int  hcLevel;           // 0=fast LZ4, 1-12=HC level (for -10 to -12)
    BackendMode backendMode;
    size_t cpuThreads;
    std::string inputFile;
    std::string outputFile;
    std::string tempOutputFile;  // .tmp file when using -f to overwrite

    // Static instance for signal handler
    static GZL4Compressor* g_instance;
    
    // Tunable GPU parameters (set via command line)
    size_t slotCapacity;      // chunks per GPU slot (--slot-capacity)
    size_t pipelineDepth;     // slots per GPU (--pipeline-depth)
    bool   disableEarlyRead;  // skip early reader (--no-early-read)

    // Started before GPU init so reads overlap with CUDA context creation
    AsyncReader      earlyReader;
    PinnedInputPool  inputPool;    // pinned slots shared between reader + GPU workers
    
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
        , hcLevel(0)
        , backendMode(BackendMode::HYBRID)  // Default to hybrid mode
        , cpuThreads(CPU_THREADS_AUTO)      // Auto-detect
        , slotCapacity(8)                   // "batch size" or "chunks per batch" in UI
        , pipelineDepth(0)                  // 0 = auto-tune, >0 = user override
        , disableEarlyRead(false)
    {}
    
    ~GZL4Compressor() {}
    
    /*
     * Initialize and enumerate all available CUDA GPUs
     */
    bool initializeGPUs() {
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
        
        // Enumerate and initialize each GPU
        for (int i = 0; i < deviceCount; i++) {
            GPUDevice gpu(i);
            
            // Try to set device - skip if fails
            err = cudaSetDevice(i);
            if (err != cudaSuccess) {
                VLOG(VERBOSE, "Skipping GPU %d - failed to set device: %s\n", 
                     i, cudaGetErrorString(err));
                cudaGetLastError(); // Clear error
                continue;
            }
            
            // Try to get properties - skip if fails
            err = cudaGetDeviceProperties(&gpu.properties, i);
            if (err != cudaSuccess) {
                VLOG(VERBOSE, "Skipping GPU %d - failed to get properties: %s\n",
                     i, cudaGetErrorString(err));
                cudaGetLastError(); // Clear error
                continue;
            }
            
            // Check compute capability (need at least 3.5 for nvCOMP)
            int computeCapability = gpu.properties.major * 10 + gpu.properties.minor;
            if (computeCapability < 35) {
                VLOG(VERBOSE, "Skipping GPU %d (%s) - compute capability %d.%d too low (need >= 3.5)\n",
                     i, gpu.properties.name, gpu.properties.major, gpu.properties.minor);
                continue;
            }
            
            // Get memory information - skip if fails
            size_t freeMem, totalMem;
            err = cudaMemGetInfo(&freeMem, &totalMem);
            if (err != cudaSuccess) {
                VLOG(VERBOSE, "Skipping GPU %d (%s) - failed to get memory info: %s\n",
                     i, gpu.properties.name, cudaGetErrorString(err));
                cudaGetLastError(); // Clear error
                continue;
            }
            
            gpu.availableMemory = static_cast<size_t>(freeMem * GPU_MEM_SAFETY_FACTOR);
            gpu.totalMemory = totalMem;
            gpu.smCount     = gpu.properties.multiProcessorCount;

            // asyncEngineCount = number of DMA copy engines (typically 2 on server GPUs:
            // Pipeline depth: temporarily set to 1 during enumeration.
            // After enumeration completes, auto-tune based on GPU count unless user specified.
            gpu.pipelineDepth = (pipelineDepth > 0) ? pipelineDepth : 1;

            VLOG(VERBOSE, "GPU%d: %s  %.1f GB VRAM  %d SMs  %d copy engine%s  pipeline=%d\n",
                 i, gpu.properties.name,
                 totalMem / (1024.0*1024.0*1024.0),
                 (int)gpu.smCount,
                 gpu.properties.asyncEngineCount,
                 gpu.properties.asyncEngineCount == 1 ? "" : "s",
                 gpu.pipelineDepth);

            // Try to create exactly pipelineDepth CUDA streams
            gpu.streams.resize(gpu.pipelineDepth);
            bool streamsOk = true;
            for (int s = 0; s < gpu.pipelineDepth; s++) {
                err = cudaStreamCreate(&gpu.streams[s]);
                if (err != cudaSuccess) {
                    VLOG(VERBOSE, "  Failed to create stream %d: %s\n",
                         s, cudaGetErrorString(err));
                    for (int cleanup = 0; cleanup < s; cleanup++)
                        cudaStreamDestroy(gpu.streams[cleanup]);
                    cudaGetLastError();
                    streamsOk = false;
                    break;
                }
            }
            if (!streamsOk) {
                VLOG(VERBOSE, "Skipping GPU%d - stream creation failed\n", i);
                continue;
            }
            VLOG(DEBUG, "  Created %d pipeline streams\n", gpu.pipelineDepth);
            
            gpus.push_back(std::move(gpu));
        }
        
        // Auto-tune pipeline depth AND batch size based on GPU count (if user didn't specify)
        // Empirical findings:
        // - 1 GPU (RTX 5090): batch 68, streams 4 → 1208 MB/s
        // - 8 GPUs (H100s): batch 3, streams 3 → best (avoids PCIe flooding)
        if (pipelineDepth <= 0 && !gpus.empty()) {
            int autoDepth = 1;
            if (gpus.size() == 1) {
                autoDepth = 4;  // Single GPU: 4 streams for overlap
            } else if (gpus.size() == 2) {
                autoDepth = 3;  // 2 GPUs: moderate overlap
            } else if (gpus.size() <= 4) {
                autoDepth = 3;  // 3-4 GPUs: 3 streams each
            } else {
                autoDepth = 3;  // 5+ GPUs: 3 streams (avoid PCIe saturation)
            }
            
            // Update all GPUs and recreate streams
            for (auto& gpu : gpus) {
                // Destroy old streams
                for (auto& stream : gpu.streams)
                    cudaStreamDestroy(stream);
                
                // Set new depth and create streams
                gpu.pipelineDepth = autoDepth;
                gpu.streams.resize(autoDepth);
                for (int s = 0; s < autoDepth; s++)
                    cudaStreamCreate(&gpu.streams[s]);
                
                VLOG(DEBUG, "GPU%d: auto-tuned pipeline depth to %d (based on %zu GPUs total)\n",
                     gpu.deviceId, autoDepth, gpus.size());
            }
        }
        
        // Auto-tune batch size based on GPU count (if user didn't specify --batch-size)
        // More GPUs = smaller batches to avoid PCIe flooding and bursty writer pattern
        if (slotCapacity == 8 && !gpus.empty()) {  // 8 is the default
            if (gpus.size() == 1) {
                slotCapacity = 64;  // Single GPU: large batches (empirically ~68 optimal)
                VLOG(DEBUG, "Auto-tuned batch size to %zu for single GPU\n", slotCapacity);
            } else if (gpus.size() <= 4) {
                slotCapacity = 16;  // 2-4 GPUs: medium batches
                VLOG(DEBUG, "Auto-tuned batch size to %zu for %zu GPUs\n", 
                     slotCapacity, gpus.size());
            } else {
                slotCapacity = 4;   // 5+ GPUs: tiny batches to avoid PCIe saturation
                VLOG(DEBUG, "Auto-tuned batch size to %zu for %zu GPUs (prevents PCIe flooding)\n",
                     slotCapacity, gpus.size());
            }
        }
        
        if (gpus.empty()) {
            fprintf(stderr, "Error: No suitable GPUs found (all GPUs busy or insufficient memory)\n");
            return false;
        }
        
        VLOG(VERBOSE, "Initialized %zu GPU(s) for processing\n", gpus.size());
        return true;
    }
    
    /*
     * Calculate optimal chunk size based on compression level
     */
    void setChunkSizeFromLevel() {
        // Map -1..-9 to chunk sizes
        // HC levels are set via --hc-level N or -10/-11/-12 (converted to --hc-level by preprocessor)
        if (compressionLevel >= 10) {
            // Legacy: -10, -11, -12 as compressionLevel (shouldn't happen anymore)
            chunkSize = CHUNK_SIZE_LEVEL_9;  // 4MB max for HC
            static const int hcMap[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8, 12};
            if (hcLevel == 0) {  // only set if not already set by --hc-level
                hcLevel = hcMap[std::min(compressionLevel, 12)];
            }
        } else {
            // compressionLevel 1-9: map to chunk sizes
            // Don't reset hcLevel if it was set by --hc-level
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
     * Returns how many chunks can fit in available GPU memory
     * With 4MB chunks (LZ4 limit), we can fit more chunks per batch
     */
    size_t calculateBatchSize(size_t gpuMemory) {
        // Conservative init-time estimate (used for startup message only;
        // refreshGPUMemoryAndBatchSize() does the real calculation at runtime).
        size_t memPerChunk = chunkSize * 5;
        size_t usable      = static_cast<size_t>(gpuMemory * 0.90);
        size_t batchSize   = usable / memPerChunk;
        batchSize = std::max(size_t(4), std::min(size_t(8192), batchSize));
        VLOG(DEBUG, "Init-time batch estimate: %zu chunks/slot (%.1f GB/slot)\n",
             batchSize, (batchSize * memPerChunk) / (1024.0*1024.0*1024.0));
        return batchSize;
    }
    
    /*
     * Dynamically compute how many chunks to pack into one pipeline slot.
     *
     * Goals:
     *   1. Fill enough work that the GPU SMs stay busy for >>1ms per batch
     *      (avoids kernel launch overhead dominating).
     *   2. Leave room for the other (pipelineDepth-1) concurrent slots.
     *   3. Never exceed available VRAM.
     *
     * nvCOMP batched LZ4 memory per chunk (empirical + conservative):
     *   input  1× chunkSize   (pinned H→D copy)
     *   output 1.1× chunkSize (LZ4 worst-case expand)
     *   temp   2×  chunkSize  (nvCOMP internal scratch)
     *   total ≈ 4.2×; we use 5× for safety.
     */
    size_t refreshGPUMemoryAndBatchSize(GPUDevice& gpu) {
        cudaSetDevice(gpu.deviceId);

        size_t freeMem, totalMem;
        if (cudaMemGetInfo(&freeMem, &totalMem) != cudaSuccess) return 0;
        gpu.availableMemory = freeMem;

        const size_t memPerChunk    = chunkSize * 5;
        const int    slots          = std::max(1, gpu.pipelineDepth);

        // Divide VRAM across pipeline slots so all can be in-flight at once.
        // Keep 10% headroom for CUDA runtime bookkeeping.
        size_t perSlotMem = static_cast<size_t>(freeMem * 0.90) / slots;

        if (perSlotMem < memPerChunk) {
            VLOG(DEBUG, "GPU%d: %.1f GB free / %d slots -> only %.0f MB/slot, need %.0f MB\n",
                 gpu.deviceId, freeMem/(1024.0*1024.0*1024.0), slots,
                 perSlotMem/(1024.0*1024.0), memPerChunk/(1024.0*1024.0));
            return 0;
        }

        // VRAM-based ceiling
        size_t vramBatch = perSlotMem / memPerChunk;

        // SM-based floor: target at least 4 chunks per SM so the GPU has
        // enough independent work to fill its wavefronts.
        size_t smFloor = gpu.smCount * 4;

        // Final batch size: at least smFloor, at most vramBatch, hard cap 8192.
        size_t batch = std::max(smFloor, std::min(vramBatch, size_t(8192)));

        // Store for monitoring
        gpu.optimalBatch = batch;

        VLOG(DEBUG, "GPU%d: %.1f GB free / %d slots -> %zu chunks/slot "
             "(VRAM ceiling %zu, SM floor %zu, %zu SMs)\n",
             gpu.deviceId, freeMem/(1024.0*1024.0*1024.0),
             slots, batch, vramBatch, smFloor, gpu.smCount);

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

        cudaGetLastError(); // Clear any pending errors

        // ── 1. Allocate pinned host staging buffers + device input buffers ──
        // Use cudaMemcpyAsync so H→D copies are queued into the stream
        // and never block the CPU.
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
            // Async copy - queued into stream, returns immediately
            CUDA_CHECK_MSG(cudaMemcpyAsync(state.d_inputs[i], inputs[i].data(), size,
                                           cudaMemcpyHostToDevice, stream),
                           "Failed to async copy input to device");
            h_input_ptrs[i] = state.d_inputs[i];
        }

        // ── 2. Device pointer/size arrays (via pinned host staging) ──────────
        // Use cudaHostAlloc (pinned) so cudaMemcpyAsync source stays valid
        // until the stream completes - stack vectors would be freed too early.
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

        // ── 3. Temp buffer (sync query, async is fine for the alloc itself) ─
        nvcompBatchedLZ4CompressOpts_t opts = nvcompBatchedLZ4CompressDefaultOpts;
        size_t temp_bytes = 0;
        NVCOMP_CHECK(nvcompBatchedLZ4CompressGetTempSizeSync(
            state.d_input_ptrs, state.d_input_sizes,
            state.batch_size, max_chunk_size, opts,
            &temp_bytes, total_input_size, stream));

        VLOG(DEBUG, "  Temp buffer size: %.2f MB\n", temp_bytes / (1024.0*1024.0));
        CUDA_CHECK_MSG(cudaMalloc(&state.d_temp, temp_bytes),
                       "Failed to allocate temp buffer");

        // ── 4. Output buffers ─────────────────────────────────────────────
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

        // ── 5. Launch compression kernel ──────────────────────────────────
        // All prior async copies are ordered before this in the same stream.
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
        // Sync only this stream, not the whole device
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Check statuses
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

        // Get output sizes
        std::vector<size_t> output_sizes(state.batch_size);
        CUDA_CHECK_MSG(cudaMemcpy(output_sizes.data(), state.d_output_sizes,
                                  state.batch_size * sizeof(size_t),
                                  cudaMemcpyDeviceToHost),
                       "Failed to copy output sizes");

        // Copy output data D→H
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
        
        cudaGetLastError(); // Clear errors
        
        // Allocate and copy input buffers
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
        
        // Allocate device arrays
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
        
        // Get max sizes
        size_t max_uncompressed = *std::max_element(uncompressed_sizes.begin(),
                                                     uncompressed_sizes.end());
        size_t total_uncompressed = 0;
        for (size_t size : uncompressed_sizes) {
            total_uncompressed += size;
        }
        
        // Get temp buffer size
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
        
        // Allocate buffers
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
        
        // Launch decompression
        VLOG(DEBUG, "  Launching decompression with batch_size=%zu\n", state.batch_size);
        NVCOMP_CHECK(nvcompBatchedLZ4DecompressAsync(
            state.d_input_ptrs,
            state.d_input_sizes,
            state.d_output_sizes,
            state.d_actual_output_sizes,
            state.batch_size,  // Using actual batch size!
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
        
        // Check statuses
        std::vector<nvcompStatus_t> statuses(state.batch_size);
        CUDA_CHECK_MSG(cudaMemcpy(statuses.data(), state.d_statuses,
                                 state.batch_size * sizeof(nvcompStatus_t),
                                 cudaMemcpyDeviceToHost),
                      "Failed to copy statuses");
        
        for (size_t i = 0; i < state.batch_size; i++) {
            if (statuses[i] != nvcompSuccess) {
                fprintf(stderr, "Decompression failed for chunk %zu with status %d\n",
                        i, static_cast<int>(statuses[i]));
                // Cleanup
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
        
        // Get actual decompressed sizes from nvCOMP
        std::vector<size_t> actualSizes(state.batch_size);
        CUDA_CHECK_MSG(cudaMemcpy(actualSizes.data(), state.d_actual_output_sizes,
                                 state.batch_size * sizeof(size_t),
                                 cudaMemcpyDeviceToHost),
                      "Failed to copy actual output sizes");
        
        // Copy output data using ACTUAL sizes, not estimates
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
        
        // Cleanup
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
     * Dynamically adjust stream count based on GPU utilization
     */
    void optimizeStreamCount() {
        static auto lastCheck = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - lastCheck).count() < 2.0) return;
        lastCheck = now;

        for (auto& gpu : gpus) {
            cudaSetDevice(gpu.deviceId);

            size_t busySlots = 0;
            for (auto& stream : gpu.streams)
                if (cudaStreamQuery(stream) == cudaErrorNotReady) busySlots++;

            float util = gpu.streams.empty() ? 0.0f :
                         (float)busySlots / gpu.streams.size();

            VLOG(VERBOSE, "GPU%d: %zu/%zu slots active (%.0f%%)  optBatch=%zu\n",
                 gpu.deviceId, busySlots, gpu.streams.size(),
                 util * 100, gpu.optimalBatch);

            // If all slots are busy and we haven't reached the hardware-detected
            // pipeline depth cap (2× pipelineDepth for extra double-buffering), add one.
            size_t maxSlots = static_cast<size_t>(gpu.pipelineDepth) * 2;
            if (util >= 1.0f && gpu.streams.size() < maxSlots) {
                cudaStream_t s;
                if (cudaStreamCreate(&s) == cudaSuccess) {
                    gpu.streams.push_back(s);
                    VLOG(VERBOSE, "GPU%d: saturated - grew to %zu slots (hw pipeline=%d)\n",
                         gpu.deviceId, gpu.streams.size(), gpu.pipelineDepth);
                }
            }
        }
    }
    
    /*
     * Compress a file using CPU-only multi-threaded compression
     */
    bool compressFileCPU() {
        VLOG(NORMAL, "Compressing (CPU-only): %s -> %s\n",
                inputFile.c_str(), outputFile.c_str());
        
        double timeReading = 0, timeCompressing = 0;
        
        // Get file size
        struct stat st;
        if (stat(inputFile.c_str(), &st) != 0) {
            fprintf(stderr, "Error: Cannot stat input file: %s\n", inputFile.c_str());
            return false;
        }
        size_t fileSize = st.st_size;
        
        VLOG(VERBOSE, "Input file size: %.2f MB\n", fileSize / (1024.0 * 1024.0));
        
        size_t numChunks = (fileSize + chunkSize - 1) / chunkSize;
        VLOG(VERBOSE, "Processing %zu chunk(s) of size %.2f MB\n", 
             numChunks, chunkSize / (1024.0 * 1024.0));
        
        // Determine CPU thread count
        size_t effectiveThreads = cpuThreads;
        if (effectiveThreads == 0) {
            effectiveThreads = std::thread::hardware_concurrency();
            if (effectiveThreads == 0) effectiveThreads = 4;
            // Cap at 64 threads by default (avoid excessive context switching)
            if (effectiveThreads > 64) effectiveThreads = 64;
        }
        VLOG(VERBOSE, "  %zu worker threads, chunk size %zu KB\n",
             effectiveThreads, chunkSize / 1024);
        
        // Start async reader
        AsyncReader asyncReader;
        size_t maxReadQueue = std::min(size_t(64), numChunks);
        if (!asyncReader.start(inputFile, chunkSize, maxReadQueue)) {
            fprintf(stderr, "Error: Failed to start async reader\n");
            return false;
        }
        
        // Write LZ4 frame header
        std::vector<uint8_t> headerBuffer;
        {
            std::ostringstream headerStream(std::ios::binary);
            if (!LZ4Frame::writeFrameHeader(headerStream, fileSize, chunkSize)) {
                fprintf(stderr, "Error: Failed to write LZ4 frame header\n");
                return false;
            }
            std::string headerStr = headerStream.str();
            headerBuffer.assign(headerStr.begin(), headerStr.end());
        }
        
        // Open output and write header (stdout or file)
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
        
        // Initialize content checksum
        XXH::State xxhState(XXH32_SEED);
        
        // Start async writer
        AsyncWriter asyncWriter;
        if (!asyncWriter.start(stdoutMode ? "-" : getActualOutputPath(), &xxhState)) {
            fprintf(stderr, "Error: Failed to start async writer\n");
            return false;
        }
        
        // Initialize CPU compression pool
        CPUCompressionPool cpuPool(effectiveThreads);
        
        size_t nextChunkToWrite = 0;
        size_t totalCompressed = 0;
        size_t chunksSubmitted = 0;
        size_t chunksExpanded = 0;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Main compression loop
        while (nextChunkToWrite < numChunks) {
            
            // PHASE 1: Read chunks and submit to CPU pool
            auto readStart = std::chrono::high_resolution_clock::now();
            
            AsyncReader::ReadChunk chunk;
            while (chunksSubmitted < numChunks && asyncReader.getChunk(chunk)) {
                // Submit to CPU pool
                cpuPool.submitJob(chunk.chunkIndex, std::move(chunk.heapData), hcLevel);
                chunksSubmitted++;
                
                VLOG(DEBUG, "Submitted chunk %zu to CPU pool (queue: %zu)\n",
                     chunk.chunkIndex, cpuPool.getQueueDepth());
                
                // Don't overwhelm the pool - keep queue reasonable
                if (cpuPool.getQueueDepth() > effectiveThreads * 4) {
                    break;
                }
            }
            
            auto readEnd = std::chrono::high_resolution_clock::now();
            timeReading += std::chrono::duration<double>(readEnd - readStart).count();
            
            // PHASE 2: Collect results and enqueue for writing
            auto compressStart = std::chrono::high_resolution_clock::now();
            
            CPUCompressionPool::CompressResult result;
            while (cpuPool.getResult(nextChunkToWrite, result)) {
                // Always pass both compressed and original data to writeTask
                // writeTask uses originalChunks for checksum and uncompressible fallback
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
                
                // Enqueue for async writing
                asyncWriter.enqueue(
                    nextChunkToWrite,
                    std::move(compressedChunks),
                    std::move(originalChunks),
                    {nextChunkToWrite},
                    {result.originalSize}
                );
                
                nextChunkToWrite++;
                
                VLOG(DEBUG, "Processed chunk %zu (writer queue: %zu)\n",
                     nextChunkToWrite - 1, asyncWriter.getQueueDepth());
            }
            
            auto compressEnd = std::chrono::high_resolution_clock::now();
            timeCompressing += std::chrono::duration<double>(compressEnd - compressStart).count();
            
            // Progress - show bytes processed
            if (g_verbosity == NORMAL && numChunks > 10) {
                size_t bytesProcessed = nextChunkToWrite * chunkSize;
                if (bytesProcessed > fileSize) bytesProcessed = fileSize;
                int progress = (100 * nextChunkToWrite) / numChunks;
                std::string cpuBytes = formatBytes(bytesProcessed);
                fprintf(stderr, "\rCompressing: %3d%%  CPU: %s%s", 
                        progress, cpuBytes.c_str(), "          ");  // padding
                fflush(stderr);
            }
            
            // Small sleep if waiting for results
            if (nextChunkToWrite < chunksSubmitted) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }

// Commenting out the following line to allow the writer to overwrite the compression progress.	
//        if (g_verbosity == NORMAL && numChunks > 10) { fprintf(stderr, "\n"); }
        
        // Wait for writer to finish with progress display
        {
            std::atomic<bool> stopProgress{false};
            std::thread progressThread;
            if (g_verbosity == NORMAL && numChunks > 10) {
                progressThread = std::thread([&]() {
                    while (!stopProgress.load()) {
                        size_t w = asyncWriter.getNextChunkToWrite();
                        size_t bytesWritten = w * chunkSize;
                        if (bytesWritten > fileSize) bytesWritten = fileSize;
                        std::string written = formatBytes(bytesWritten);
                        std::string total = formatBytes(fileSize);
                        VLOG(NORMAL, "\rWriting: %3d%%  [%s/%s to disk]%s",
                                (int)(100 * w / numChunks), written.c_str(), total.c_str(),
                                "          ");
                        fflush(stderr);
                        std::this_thread::sleep_for(std::chrono::milliseconds(150));
                    }
                    std::string total = formatBytes(fileSize);
                    VLOG(NORMAL, "\rWriting: 100%%  [%s/%s to disk]  \n",
                            total.c_str(), total.c_str());
                });
            }
            
            VLOG(VERBOSE, "Waiting for async writer to complete...\n");
            asyncWriter.stop();
            
            stopProgress.store(true);
            if (progressThread.joinable()) progressThread.join();
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

                if (!stdoutMode) { fsync(footerFd); close(footerFd); }
                VLOG(VERBOSE, "Computed content checksum: 0x%08X\n", contentChecksum);
            }
        }
        
        // Stop async reader
        asyncReader.stop();
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        double totalBytesWritten = asyncWriter.getBytesWritten();
        double ratio = 100.0 * totalBytesWritten / fileSize;
        double throughputMBps = (fileSize / (1024.0 * 1024.0)) / (duration.count() / 1000.0);
        
        double cpuReadTime  = asyncReader.getReadTime();
        double cpuWriteTime = asyncWriter.getWriteTime();

        std::string inputSize = formatBytes(fileSize);
        std::string outputSize = formatBytes((size_t)totalBytesWritten);
        VLOG(NORMAL, "Compression complete (CPU-only): %s -> %s (%.2f%%) in %.2f s\n",
                inputSize.c_str(), outputSize.c_str(), ratio, duration.count() / 1000.0);
        VLOG(VERBOSE, "Throughput: %.2f MB/s\n", throughputMBps);
        VLOG(VERBOSE, "  Read:    %.2f s  |  CPU compress (%zu threads): %.2f s  |  Write: %.2f s\n",
             cpuReadTime, effectiveThreads, timeCompressing, cpuWriteTime);
        VLOG(VERBOSE, "  Uncompressed blocks: %zu / %zu (%.1f%%)\n",
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
        // Print per-GPU hardware summary
        for (auto& g : gpus)
            VLOG(VERBOSE, "  GPU%d: %s  %.1f GB  %zu SMs  %d pipeline slots\n",
                 g.deviceId, g.properties.name,
                 g.totalMemory/(1024.0*1024.0*1024.0), g.smCount, g.pipelineDepth);
        VLOG(NORMAL, "Compressing (GPU-only, %zu GPU%s): %s -> %s\n",
                gpus.size(), gpus.size()==1?"":"s",
                inputFile.c_str(), outputFile.c_str());
        
        // Get file size
        struct stat st;
        if (stat(inputFile.c_str(), &st) != 0) {
            fprintf(stderr, "Error: Cannot stat input file: %s\n", inputFile.c_str());
            return false;
        }
        size_t fileSize = st.st_size;
        
        VLOG(VERBOSE, "Input file size: %.2f MB\n", 
             fileSize / (1024.0 * 1024.0));
        
        // Calculate chunks
        size_t numChunks = (fileSize + chunkSize - 1) / chunkSize;
        VLOG(VERBOSE, "Processing %zu chunk(s) of size %.2f MB\n",
             numChunks, chunkSize / (1024.0 * 1024.0));

        // ── Async reader setup ────────────────────────────────────────────────
        // Prefer the pre-warmed reader started before GPU init in run().
        // Fall back to starting fresh here (CPU-only fallback, test mode, etc.)
        size_t totalPipelineSlots = 0;
        for (auto& g : gpus) totalPipelineSlots += g.pipelineDepth;
        size_t estBatch    = gpus.empty() ? 64 :
            std::max(size_t(64), static_cast<size_t>(
                gpus[0].availableMemory * 0.90 / gpus[0].pipelineDepth / (chunkSize * 5)));
        size_t maxReadQueue = std::min(numChunks, std::max(size_t(256), totalPipelineSlots * estBatch));

        AsyncReader localReader;
        AsyncReader* asyncReaderPtr = nullptr;

        if (earlyReader.getFileSize() > 0) {
            // Use the pre-warmed reader as-is (heap mode)  never restart it.
            // Restarting would re-read from byte 0 while old chunks are still
            // queued, causing duplicate chunk indices and writer deadlock.
            asyncReaderPtr = &earlyReader;
            VLOG(VERBOSE, "Pre-warmed reader: %.2f MB buffered so far\n",
                 earlyReader.getBytesRead() / (1024.0*1024.0));
        } else {
            // Fresh start  use pool mode for zero-copy DMA transfers
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

        // ── Slot capacity tuning ──────────────────────────────────────────────
        // Larger batches = fewer total batches = fewer sequential ordering gaps
        // where the writer waits for "the next chunk" to arrive. With 7 GPUs
        // racing through batches, 32 chunks/slot gives ~63 total batches vs
        // 252 at capacity=8, cutting writer gap-wait time by ~3-4 seconds.
        // 28 slots × 32 chunks = 896 chunks in-flight (3.5 GB across 7 GPUs).
        const size_t SLOT_CAPACITY = slotCapacity;  // from --slot-capacity flag

        // ── Init pinned input pool (only if no early reader) ──────────────────
        // Skip pool if early reader is active (heap mode)  the pool would sit
        // unused while taking 2-3s to cudaHostAlloc 7GB of pinned memory.
        // Pool is only beneficial for fresh readers (decompression, batch mode).
        if (!inputPool.numSlots() && earlyReader.getFileSize() == 0) {
            size_t poolSlots = std::min(numChunks,
                std::max(size_t(64), 2 * totalPipelineSlots * slotCapacity));
            if (!inputPool.init(poolSlots, chunkSize)) {
                fprintf(stderr, "Warning: pinned pool alloc failed, using heap\n");
            } else {
                VLOG(VERBOSE, "PinnedInputPool: %zu slots × %.0f MB = %.1f GB\n",
                     poolSlots, chunkSize/1024.0/1024.0,
                     poolSlots*chunkSize/1024.0/1024.0/1024.0);
            }
        }
        nvcompBatchedLZ4CompressOpts_t nvOpts = nvcompBatchedLZ4CompressDefaultOpts;
        size_t maxOutPerChunk = 0;
        nvcompBatchedLZ4CompressGetMaxOutputChunkSize(chunkSize, nvOpts, &maxOutPerChunk);

        // Compute sharedTempBytes once on GPU0
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

        // ── Start slot init as background threads NOW ─────────────────────────
        // These run concurrently with header write + asyncWriter.start() below,
        // hiding as much of the 3+ second cudaHostAlloc wall as possible.
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
                    else    { fprintf(stderr, "GPU%d: failed to init slot %d\n", gpus[g].deviceId, si);
                              gpuInitOk[g] = false; break; }
                }
            });
        }

        // Write LZ4 frame header to temporary buffer
        std::vector<uint8_t> headerBuffer;
        {
            // Create a temporary string stream to build header
            std::ostringstream headerStream(std::ios::binary);
            if (!LZ4Frame::writeFrameHeader(headerStream, fileSize, chunkSize)) {
                fprintf(stderr, "Error: Failed to write LZ4 frame header\n");
                return false;
            }
            std::string headerStr = headerStream.str();
            headerBuffer.assign(headerStr.begin(), headerStr.end());
        }
        
        // Open output file and write header synchronously
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
        
        // Initialize content checksum with streaming state
        XXH::State xxhState(XXH32_SEED);
        
        // ── Join slot init threads (started before header write above) ─────────
        for (auto& t : slotInitThreads) t.join();

        bool slotsOk = true;
        for (bool ok : gpuInitOk) if (!ok) { slotsOk = false; break; }

        double initMs = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - initSlotT0).count();
        VLOG(VERBOSE, "Slot init: %.2f s (parallel, overlapped with file setup)\n",
             initMs / 1000.0);

        // Start async writer NOW  after slots are ready so writer thread
        // doesn't sit idle waiting for the first chunk during slot init.
        AsyncWriter asyncWriter;
        if (!asyncWriter.start(getActualOutputPath(), &xxhState)) {
            fprintf(stderr, "Error: Failed to start async writer\n");
            freeAllSlots(); asyncReader.stop(); return false;
        }

        if (!slotsOk) { freeAllSlots(); asyncWriter.stop(); asyncReader.stop(); return false; }

        size_t totalSlots = 0;
        for (auto& gSlots : gpuSlots) totalSlots += gSlots.size();
        VLOG(VERBOSE, "GPU pipeline ready: %zu GPUs x %d slots = %zu total  cap=%zu chunks/slot\n",
             gpus.size(), gpus[0].pipelineDepth, totalSlots, SLOT_CAPACITY);

        std::atomic<size_t> batchesLaunched{0};
        std::atomic<bool>   workerAbort{false};

        // ── Per-GPU worker thread function ────────────────────────────────────
        // Each thread owns its GPU's slots, competes with other threads for
        // chunks from the shared AsyncReader, and submits results DIRECTLY to
        // asyncWriter (which is thread-safe).  No intermediate TsQueue needed.
        std::atomic<size_t> chunksSubmitted{0};

        auto gpuWorker = [&](size_t gpuIdx) {
            GPUDevice& gpu = gpus[gpuIdx];
            cudaSetDevice(gpu.deviceId);
            std::vector<PreallocSlot>& slots = gpuSlots[gpuIdx];
            const int nSlots = (int)slots.size();

            // Initialize all slots as not having pending work
            for (auto& slot : slots) {
                slot.hasPending = false;
            }

            while (!workerAbort.load()) {
                bool anyActivity = false;

                // ── Poll all slots: collect completed batches (non-blocking) ──────
                for (int si = 0; si < nSlots; si++) {
                    PreallocSlot& slot = slots[si];
                    
                    if (slot.hasPending) {
                        // Non-blocking check if this slot's stream is done
                        cudaError_t status = cudaStreamQuery(slot.stream);
                        
                        if (status == cudaSuccess) {
                            // Stream completed! Collect results
                            anyActivity = true;
                            
                            std::vector<std::vector<uint8_t>> cChunks, oChunks;
                            cChunks.reserve(slot.batchSize);
                            oChunks.reserve(slot.batchSize);
                            
                            for (size_t i = 0; i < slot.batchSize; i++) {
                                size_t outSz  = slot.h_oSizes[i];
                                size_t origSz = slot.origSizes[i];

                                // Get pointer to original data (pooled or heap)
                                const uint8_t* origPtr = slot.origHandles.size() > i
                                    ? slot.origHandles[i].data
                                    : slot.origData[i].data();

                                if (outSz < origSz) {
                                    // Compressed version is smaller - use it
                                    std::vector<uint8_t> buf(outSz);
                                    memcpy(buf.data(), slot.h_output + i * slot.outStride, outSz);
                                    cChunks.push_back(std::move(buf));
                                } else {
                                    cChunks.push_back({});   // store as uncompressed (better)
                                }
                                // Always copy original  releases pool slot immediately
                                oChunks.emplace_back(origPtr, origPtr + origSz);
                            }
                            // Release pool handles now  reader can refill those slots
                            slot.origHandles.clear();
                            slot.origData.clear();

                            asyncWriter.enqueueBatch(cChunks, oChunks,
                                                     slot.indices, slot.origSizes);
                            chunksSubmitted += slot.batchSize;
                            slot.hasPending = false;
                        } else if (status != cudaErrorNotReady) {
                            // Actual error (not just "not ready")
                            fprintf(stderr, "GPU%d stream error: %s\n", 
                                    gpu.deviceId, cudaGetErrorString(status));
                            workerAbort.store(true);
                            break;
                        }
                    }
                }

                // ── Launch new batches on any free slots ──────────────────────────
                for (int si = 0; si < nSlots && !asyncReader.isFinished(); si++) {
                    PreallocSlot& slot = slots[si];
                    
                    if (slot.hasPending) continue;  // Slot still busy
                    
                    // Slot is free - try to fill and launch
                    slot.indices.clear();
                    slot.origSizes.clear();
                    slot.origData.clear();
                    slot.origHandles.clear();
                    slot.batchSize = 0;

                    while (slot.batchSize < slot.capacity) {
                        AsyncReader::ReadChunk chunk;
                        if (!asyncReader.getChunk(chunk)) break;
                        // H→D: DMA from pinned slot (or memcpy from heap in fallback)
                        cudaMemcpyAsync(
                            slot.d_input + slot.batchSize * slot.chunkStride,
                            chunk.data(), chunk.size,
                            cudaMemcpyHostToDevice, slot.stream);
                        slot.h_iSizes[slot.batchSize] = chunk.size;
                        slot.indices.push_back(chunk.chunkIndex);
                        slot.origSizes.push_back(chunk.size);
                        // Hold the input data until stream completes
                        if (chunk.poolHandle.valid())
                            slot.origHandles.push_back(std::move(chunk.poolHandle));
                        else
                            slot.origData.push_back(std::move(chunk.heapData));
                        slot.batchSize++;
                    }

                    if (slot.batchSize == 0) continue;  // No chunks available
                    
                    anyActivity = true;

                    // H→D: input sizes (tiny  just sizeof(size_t)*batchSize bytes)
                    cudaMemcpyAsync(slot.d_iSizes, slot.h_iSizes,
                                    slot.batchSize * sizeof(size_t),
                                    cudaMemcpyHostToDevice, slot.stream);

                    // ── Launch: nvCOMP compression (after all H→D copies in stream) ─
                    nvcompStatus_t nErr = nvcompBatchedLZ4CompressAsync(
                        slot.d_iPtrs, slot.d_iSizes, slot.chunkStride,
                        slot.batchSize, slot.d_temp, slot.tempBytes,
                        slot.d_oPtrs, slot.d_oSizes, nvOpts, slot.d_stats, slot.stream);
                    if (nErr != nvcompSuccess) {
                        fprintf(stderr, "GPU%d slot: nvcomp error %d\n", gpu.deviceId, (int)nErr);
                        workerAbort.store(true); 
                        break;
                    }
                    batchesLaunched++;

                    // ── D→H: queue result copies (execute after kernel) ─
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
                
                // If no activity this iteration, sleep briefly to avoid busy-waiting
                if (!anyActivity) {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
                
                // Exit when reader finished AND all slots are idle
                if (asyncReader.isFinished()) {
                    bool allIdle = true;
                    for (const auto& s : slots) {
                        if (s.hasPending) {
                            allIdle = false;
                            break;
                        }
                    }
                    if (allIdle) break;
                }
            }
        };

        // Tell writer exactly how many chunks are coming so it self-exits
        // the moment the last chunk is written, without waiting for stop().
        asyncWriter.setTotalChunks(numChunks);

        // ── Launch one worker thread per GPU ─────────────────────────────────
        auto startTime = std::chrono::high_resolution_clock::now();
        std::atomic<int> activeWorkers{(int)gpus.size()};
        std::vector<std::thread> workers;
        for (size_t g = 0; g < gpus.size(); g++) {
            workers.emplace_back([&, g]() {
                gpuWorker(g);
                --activeWorkers;  // signal this GPU is done
            });
        }

        // ── Main thread: progress display while GPU workers compress ─────────
        // Workers submit directly to asyncWriter  main thread just shows status.
        while (activeWorkers.load() > 0) {
            if (g_verbosity == NORMAL && numChunks > 10) {
                size_t submitted = chunksSubmitted.load();
                size_t bytesProcessed = submitted * chunkSize;
                if (bytesProcessed > fileSize) bytesProcessed = fileSize;
                int progress = (int)(100 * submitted / numChunks);
                std::string gpuBytes = formatBytes(bytesProcessed);
                fprintf(stderr, "\rCompressing: %3d%%  GPU: %s%s", 
                        progress, gpuBytes.c_str(), "          ");
                fflush(stderr);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        // Join all workers
        for (auto& w : workers) if (w.joinable()) w.join();

        // ── Writer drain with live progress ───────────────────────────────────
        // asyncWriter.stop() blocks until the writer thread finishes.
        // With setTotalChunks() the writer self-exits as soon as the last chunk
        // is written  stop() returns immediately rather than after freeAllSlots().
        // We free slots AFTER stop() so GPU memory is held no longer than needed.
        {
            std::atomic<bool> stopProgress{false};
            std::thread progressThread;
            if (g_verbosity == NORMAL && numChunks > 10) {
                progressThread = std::thread([&]() {
                    while (!stopProgress.load()) {
                        size_t w = asyncWriter.getNextChunkToWrite();
                        size_t bytesWritten = w * chunkSize;
                        if (bytesWritten > fileSize) bytesWritten = fileSize;
                        std::string written = formatBytes(bytesWritten);
                        std::string total = formatBytes(fileSize);
                        VLOG(NORMAL, "\rWriting: %3d%%  [%s/%s to disk]%s",
                                (int)(100 * w / numChunks), written.c_str(), total.c_str(),
                                "          ");
                        fflush(stderr);
                        std::this_thread::sleep_for(std::chrono::milliseconds(150));
                    }
                    // Final update
                    std::string total = formatBytes(fileSize);
                    VLOG(NORMAL, "\rWriting: 100%%  [%s/%s to disk]  \n",
                            total.c_str(), total.c_str());
                });
            }

            VLOG(VERBOSE, "Waiting for async writer to complete...\n");
            asyncWriter.stop();   // fast: writer self-exited when last chunk was written

            stopProgress.store(true);
            if (progressThread.joinable()) progressThread.join();
        }

        freeAllSlots();   // GPU memory freed after writer done (parallel with footer write)
        
        // Now append footer synchronously (end mark + checksum)
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

            if (!stdoutMode) { fsync(footerFd); close(footerFd); }
            VLOG(VERBOSE, "Computed content checksum: 0x%08X (from %zu bytes)\n",
                 contentChecksum, xxhState.totalLen);
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

// Avoiding the extra blank line before Compression complete output by comentting the following line.	
//        if (g_verbosity == NORMAL && numChunks > 10) { fprintf(stderr, "\n"); }
        
        // Stop async reader
        asyncReader.stop();
        
        double totalBytesWritten = asyncWriter.getBytesWritten();
        double ratio = 100.0 * totalBytesWritten / fileSize;
        double throughputMBps = (fileSize / (1024.0 * 1024.0)) / (duration.count() / 1000.0);
        
        double asyncReadTime  = asyncReader.getReadTime();
        double asyncWriteTime = asyncWriter.getWriteTime();

        size_t finalSlots = 0;
        for (auto& gSlots : gpuSlots) finalSlots += gSlots.size();
        std::string inputSize = formatBytes(fileSize);
        std::string outputSize = formatBytes((size_t)totalBytesWritten);
        VLOG(NORMAL, "Compression complete (GPU-only, %zu GPU%s / %zu pipeline slots): %s -> %s (%.2f%%) in %.2f s\n",
                gpus.size(), gpus.size() == 1 ? "" : "s", finalSlots,
                inputSize.c_str(), outputSize.c_str(), ratio, duration.count() / 1000.0);
        VLOG(VERBOSE, "Throughput: %.2f MB/s\n", throughputMBps);
        VLOG(VERBOSE, "  Read: %.2f s  |  Write: %.2f s\n",
             asyncReadTime, asyncWriteTime);
        VLOG(VERBOSE, "  Batches launched: %zu  avg %.1f chunks/batch  %zu GPU%s / %zu slot%s\n",
             batchesLaunched.load(), (double)numChunks / std::max(size_t(1), batchesLaunched.load()),
             gpus.size(), gpus.size() == 1 ? "" : "s",
             finalSlots, finalSlots == 1 ? "" : "s");
        
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
            case BackendMode::CPU_ONLY:
                return compressFileCPU();
            
            case BackendMode::GPU_ONLY:
                return compressFileGPU();
            
            case BackendMode::HYBRID:
                return compressFileHybrid();
            
            default:
                fprintf(stderr, "Error: Unknown backend mode\n");
                return false;
        }
    }
    
    /*
     * Compress using CPU + GPU simultaneously (hybrid mode, v3 - GPU priority).
     *
     * Architecture:
     *   Dispatcher thread pulls from AsyncReader:
     *     - Try to push to GPU work queue first (limited capacity)
     *     - If GPU queue full → push to CPU work queue
     *   
     *   GPU workers pull from GPU queue, process batches
     *   CPU workers pull from CPU queue, process single chunks
     *   
     *   Both write directly to AsyncWriter (thread-safe)
     *
     * This ensures GPUs always get chunks first; CPUs only work when GPUs
     * are saturated. As soon as a GPU slot frees up, it pulls the next chunk.
     */
    bool compressFileHybrid() {
        struct stat st;
        if (stat(inputFile.c_str(), &st) != 0) {
            fprintf(stderr, "Error: Cannot stat input file: %s\n", inputFile.c_str());
            return false;
        }
        size_t fileSize  = st.st_size;
        size_t numChunks = (fileSize + chunkSize - 1) / chunkSize;

        size_t effectiveThreads = cpuThreads ? cpuThreads
                                             : std::thread::hardware_concurrency();
        if (!effectiveThreads) effectiveThreads = 4;

        VLOG(NORMAL, "Compressing (Hybrid, %zu thread%s + %zu GPU%s): %s -> %s\n",
                effectiveThreads, effectiveThreads==1?"":"s",
                gpus.size(),      gpus.size()==1?"":"s",
                inputFile.c_str(), outputFile.c_str());

        // ── Reader setup ───────────────────────────────────────────────────────
        size_t totalPipelineSlots = 0;
        for (auto& g : gpus) totalPipelineSlots += g.pipelineDepth;
        size_t estBatch = gpus.empty() ? 64 :
            std::max(size_t(64), static_cast<size_t>(
                gpus[0].availableMemory * 0.90 / gpus[0].pipelineDepth / (chunkSize * 5)));
        size_t maxReadQueue = std::min(numChunks,
                              std::max(size_t(256), totalPipelineSlots * estBatch));

        AsyncReader  localReader;
        AsyncReader* asyncReaderPtr = nullptr;
        if (earlyReader.getFileSize() > 0) {
            asyncReaderPtr = &earlyReader;
        } else {
            bool started = inputPool.numSlots()
                ? localReader.startPooled(inputFile, chunkSize, &inputPool)
                : localReader.start(inputFile, chunkSize, maxReadQueue);
            if (!started) { fprintf(stderr,"Error: reader start failed\n"); return false; }
            asyncReaderPtr = &localReader;
        }
        AsyncReader& asyncReader = *asyncReaderPtr;

        // ── GPU slot setup (same as compressFileGPU) ──────────────────────────
        const size_t SLOT_CAPACITY = slotCapacity;

        if (!inputPool.numSlots() && earlyReader.getFileSize() == 0) {
            size_t poolSlots = std::min(numChunks,
                std::max(size_t(64), 2 * totalPipelineSlots * slotCapacity));
            if (!inputPool.init(poolSlots, chunkSize))
                fprintf(stderr, "Warning: pinned pool alloc failed, using heap\n");
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
            if (!LZ4Frame::writeFrameHeader(hs, fileSize, chunkSize)) {
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
        for (auto& t : slotInitThreads) t.join();

        bool slotsOk = true;
        for (bool ok : gpuInitOk) if (!ok) { slotsOk = false; break; }

        AsyncWriter asyncWriter;
        if (!asyncWriter.start(getActualOutputPath(), &xxhState)) {
            fprintf(stderr,"Error: Failed to start async writer\n");
            freeAllSlots(); asyncReader.stop(); return false;
        }
        if (!slotsOk) {
            fprintf(stderr,"GPU slot init failed, falling back to CPU-only\n");
            freeAllSlots(); asyncWriter.stop(); asyncReader.stop();
            return compressFileCPU();
        }

        asyncWriter.setTotalChunks(numChunks);
        auto startTime = std::chrono::high_resolution_clock::now();
        std::atomic<size_t> chunksSubmitted{0};
        std::atomic<bool>   workerAbort{false};
        std::atomic<size_t> gpuChunkCount{0};
        std::atomic<size_t> cpuChunkCount{0};

        // ── Work queues: GPU priority, CPU fallback ───────────────────────────
        // GPU queue capacity = total slots * batch size (so all GPU slots can fill)
        size_t gpuQueueCap = totalPipelineSlots * SLOT_CAPACITY * 2;  // 2x for overlap
        TsQueue<AsyncReader::ReadChunk> gpuWorkQueue;
        TsQueue<AsyncReader::ReadChunk> cpuWorkQueue;

        // ── Dispatcher: feeds GPU queue first, CPU queue when GPU saturated ───
        std::atomic<bool> dispatcherDone{false};
        std::thread dispatcherThread([&]() {
            while (true) {
                AsyncReader::ReadChunk chunk;
                if (!asyncReader.getChunk(chunk)) break;

                // Try GPU queue first (non-blocking check)
                if (gpuWorkQueue.size() < gpuQueueCap) {
                    gpuWorkQueue.push(std::move(chunk));
                } else {
                    // GPU queue full, send to CPU
                    cpuWorkQueue.push(std::move(chunk));
                }
            }
            gpuWorkQueue.close();
            cpuWorkQueue.close();
            dispatcherDone.store(true);
        });

        // ── GPU worker (pulls from GPU queue) ─────────────────────────────────
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

                // Collect and submit previous results
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

                // Pull chunks from GPU queue
                slot.indices.clear();
                slot.origSizes.clear();
                slot.origData.clear();
                slot.origHandles.clear();
                slot.batchSize = 0;

                while (slot.batchSize < slot.capacity) {
                    AsyncReader::ReadChunk chunk;
                    if (!gpuWorkQueue.pop(chunk, 5)) {  // 5ms timeout
                        if (gpuWorkQueue.isClosed()) break;
                        continue;
                    }
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

                if (slot.batchSize == 0) break;  // queue closed and empty

                // Launch GPU compression
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

            // Drain remaining slots
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

        // ── CPU worker (pulls from CPU queue) ─────────────────────────────────
        auto cpuWorker = [&]() {
            while (!workerAbort.load()) {
                AsyncReader::ReadChunk chunk;
                if (!cpuWorkQueue.pop(chunk, 50)) {  // 50ms timeout
                    if (cpuWorkQueue.isClosed()) break;
                    continue;
                }

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
            }
        };

        // ── Launch all workers ────────────────────────────────────────────────
        std::vector<std::thread> allWorkers;
        allWorkers.reserve(gpus.size() + effectiveThreads);
        for (size_t g = 0; g < gpus.size(); g++)
            allWorkers.emplace_back(gpuWorker, g);
        for (size_t t = 0; t < effectiveThreads; t++)
            allWorkers.emplace_back(cpuWorker);

        // Progress display
        while (chunksSubmitted.load() < numChunks) {
            if (g_verbosity == NORMAL && numChunks > 10) {
                size_t done = asyncWriter.getNextChunkToWrite();
                size_t gpuChunks = gpuChunkCount.load();
                size_t cpuChunks = cpuChunkCount.load();
                size_t gpuBytes = gpuChunks * chunkSize;
                size_t cpuBytes = cpuChunks * chunkSize;
                std::string gpuStr = formatBytes(gpuBytes);
                std::string cpuStr = formatBytes(cpuBytes);
                fprintf(stderr, "\rCompressing: %3zu%%  GPU: %s  CPU: %s%s",
                        (100 * done) / numChunks,
                        gpuStr.c_str(), cpuStr.c_str(), "          ");
                fflush(stderr);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        dispatcherThread.join();
        for (auto& t : allWorkers) t.join();

// Removing this to allow the writer to overwrite the Compression progress line.
//        if (g_verbosity == NORMAL && numChunks > 10) fprintf(stderr, "\n");
        
        // Wait for writer to finish with progress display
        {
            std::atomic<bool> stopProgress{false};
            std::thread progressThread;
            if (g_verbosity == NORMAL && numChunks > 10) {
                progressThread = std::thread([&]() {
                    while (!stopProgress.load()) {
                        size_t w = asyncWriter.getNextChunkToWrite();
                        size_t bytesWritten = w * chunkSize;
                        if (bytesWritten > fileSize) bytesWritten = fileSize;
                        std::string written = formatBytes(bytesWritten);
                        std::string total = formatBytes(fileSize);
                        VLOG(NORMAL, "\rWriting: %3d%%  [%s/%s to disk]%s",
                                (int)(100 * w / numChunks), written.c_str(), total.c_str(),
                                "          ");
                        fflush(stderr);
                        std::this_thread::sleep_for(std::chrono::milliseconds(150));
                    }
                    std::string total = formatBytes(fileSize);
                    VLOG(NORMAL, "\rWriting: 100%%  [%s/%s to disk]  \n",
                            total.c_str(), total.c_str());
                });
            }
            
            VLOG(VERBOSE, "Waiting for async writer to complete...\n");
            asyncWriter.stop();
            
            stopProgress.store(true);
            if (progressThread.joinable()) progressThread.join();
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
                if (!stdoutMode) { fsync(fd); close(fd); }
                VLOG(VERBOSE, "Content checksum: 0x%08X\n", cs);
            }
        }

        asyncReader.stop();
        freeAllSlots();

        auto endTime  = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        double bw = asyncWriter.getBytesWritten();
        double mbps = (fileSize/(1024.0*1024.0)) / (duration.count()/1000.0);

        std::string inputSize = formatBytes(fileSize);
        std::string outputSize = formatBytes((size_t)bw);
        VLOG(NORMAL, "Compression complete (Hybrid, %zu thread%s + %zu GPU%s): "
                "%s -> %s (%.2f%%) in %.2f s\n",
                effectiveThreads, effectiveThreads==1?"":"s",
                gpus.size(),      gpus.size()==1?"":"s",
                inputSize.c_str(), outputSize.c_str(),
                100.0*bw/fileSize, duration.count()/1000.0);
        VLOG(VERBOSE, "Throughput: %.2f MB/s\n", mbps);
        VLOG(VERBOSE, "  GPU: %zu chunks (%.1f%%)  CPU: %zu chunks (%.1f%%)\n",
             gpuChunkCount.load(), 100.0*gpuChunkCount.load()/numChunks,
             cpuChunkCount.load(), 100.0*cpuChunkCount.load()/numChunks);
        VLOG(VERBOSE, "  Write: %.2f s\n", asyncWriter.getWriteTime());

        if (!keepOriginal && !stdoutMode) {
            if (unlink(inputFile.c_str()) != 0)
                fprintf(stderr, "Warning: Could not remove input file: %s\n",
                        inputFile.c_str());
        }

        return true;
    }
    /*
     * Decompress a single LZ4 block on GPU using the batched API with batch_size=1.
     *
     * nvCOMP has no non-batched decompression API  the batched API IS the API.
     * We call it with a single-element batch (batch_size=1).
     *
     * This will return nvcompErrorInvalidValue (Error 12) for blocks compressed
     * by LZ4_compress_default / LZ4_compress_HC (CPU path), because nvCOMP's
     * batched decompressor requires blocks in its own internal format.
     * That's expected  the caller falls back to LZ4_decompress_safe() for those.
     *
     * Blocks compressed by nvCOMP (GPU path) will decompress successfully here.
     *
     * Returns true = GPU decompression succeeded, outData filled.
     * Returns false = try LZ4_decompress_safe() on CPU instead.
     */
    bool decompressBlockGPU(const uint8_t* compData, size_t compSize,
                            uint8_t* outData, size_t outSize,
                            GPUDevice& gpu, cudaStream_t& stream,
                            size_t& actualSize) {  // ← actual bytes written
        //
        // nvCOMP's batched LZ4 API decompresses both nvCOMP-format blocks (produced
        // by the GPU compression path) and raw LZ4 blocks (produced by LZ4_compress_default
        // / LZ4_compress_HC on the CPU path) without issue.  We do NOT pre-filter by
        // format  just send everything to the GPU and let nvCOMP handle it.
        //
        // The original hang was caused by:
        //   1. nvcompBatchedLZ4DecompressGetTempSizeSync  blocking indefinitely on
        //      some blocks in nvCOMP 5.1.x (now removed; fixed temp size used instead)
        //   2. cudaStreamSynchronize  safe to use since nvCOMP handles all LZ4 formats
        //
        // Callers fall back to LZ4_decompress_safe only when this function returns false
        // (genuine GPU errors: OOM, device lost, etc.).
        //
        if (cudaSetDevice(gpu.deviceId) != cudaSuccess) return false;

        void*   d_comp       = nullptr;
        void*   d_decomp     = nullptr;
        void**  d_inPtrs     = nullptr;
        void**  d_outPtrs    = nullptr;
        size_t* d_inSizes    = nullptr;
        size_t* d_outBufSizes= nullptr;
        size_t* d_actualSizes= nullptr;
        nvcompStatus_t* d_statuses = nullptr;
        void*   d_temp       = nullptr;

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

        // H→D: compressed payload + metadata arrays
        const void* h_inPtr  = d_comp;
        void*       h_outPtr = d_decomp;
        cudaMemcpy(d_comp,        compData,    compSize,       cudaMemcpyHostToDevice);
        cudaMemcpy(d_inPtrs,      &h_inPtr,    sizeof(void*),  cudaMemcpyHostToDevice);
        cudaMemcpy(d_outPtrs,     &h_outPtr,   sizeof(void*),  cudaMemcpyHostToDevice);
        cudaMemcpy(d_inSizes,     &compSize,   sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_outBufSizes, &outSize,    sizeof(size_t), cudaMemcpyHostToDevice);

        // Use a fixed temp buffer (outSize bytes) rather than calling
        // nvcompBatchedLZ4DecompressGetTempSizeSync, which blocked indefinitely
        // on certain blocks in nvCOMP 5.1.x.
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

    /*
     * Decompress a file using GPU acceleration (non-batched per-block API).
     *
     * Uses the batched nvCOMP API with batch_size=1.
     * nvCOMP blocks decompress on GPU; CPU-compressed blocks fall back to
     * LZ4_decompress_safe() automatically (Error 12 triggers the fallback).
     */
    bool decompressFileGPU() {
        if (testMode) {
            VLOG(NORMAL, "Testing (GPU+fallback, %zu GPU%s): %s\n",
                    gpus.size(), gpus.size() == 1 ? "" : "s",
                    inputFile.c_str());
        } else {
            VLOG(NORMAL, "Decompressing (GPU+fallback, %zu GPU%s): %s -> %s\n",
                    gpus.size(), gpus.size() == 1 ? "" : "s",
                    inputFile.c_str(), outputFile.c_str());
        }
        
        
        // Open input file with direct I/O
        int inputFd = open(inputFile.c_str(), O_RDONLY);
        if (inputFd < 0) {
            fprintf(stderr, "Error: Cannot open input file: %s\n", inputFile.c_str());
            return false;
        }
        
        // Read-ahead hints for kernel I/O scheduler
        posix_fadvise(inputFd, 0, 0, POSIX_FADV_SEQUENTIAL);
        posix_fadvise(inputFd, 0, 0, POSIX_FADV_WILLNEED);

        // Read LZ4 frame header (first 32 bytes covers all header variants)
        uint8_t headerBuf[32];
        ssize_t headerRead = ::read(inputFd, headerBuf, 32);
        if (headerRead < 15) {
            fprintf(stderr, "Error: File too small to be valid LZ4\n");
            close(inputFd);
            return false;
        }
        
        // Parse header
        LZ4Frame::FrameDescriptor desc;
        size_t headerBytes = 0;
        {
            std::string headerStr((char*)headerBuf, headerRead);
            std::istringstream headerStream(headerStr, std::ios::binary);
            if (!LZ4Frame::readFrameHeader(headerStream, desc)) {
                fprintf(stderr, "Error: Failed to read LZ4 frame header\n");
                close(inputFd);
                return false;
            }
            // How many bytes did the parser actually consume?
            headerBytes = headerStream.tellg();
        }
        
        // Seek fd back to the byte right after the header
        if (lseek(inputFd, (off_t)headerBytes, SEEK_SET) == (off_t)-1) {
            fprintf(stderr, "Error: Failed to seek past header: %s\n", strerror(errno));
            close(inputFd);
            return false;
        }
        VLOG(DEBUG, "LZ4 header consumed %zu bytes, seeking fd to byte %zu\n",
             headerBytes, headerBytes);
        
        size_t originalFileSize = desc.contentSize;
        size_t chunkSize = static_cast<size_t>(1) << (8 + 2 * desc.blockMaxSize);
        size_t estimatedBlocks = (originalFileSize + chunkSize - 1) / chunkSize;
        
        VLOG(VERBOSE, "  %.2f MB  |  block size %zu KB  |  ~%zu blocks\n",
             originalFileSize / (1024.0*1024.0), chunkSize / 1024, estimatedBlocks);
        
        XXH::State xxhState(XXH32_SEED);
        
        // Open output file or setup null output for test mode
        int outputFd = -1;
        if (!testMode) {
            if (stdoutMode) {
                outputFd = STDOUT_FILENO;
            } else {
                outputFd = open(getActualOutputPath(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
                if (outputFd < 0) {
                    fprintf(stderr, "Error: Cannot create output file: %s\n", getActualOutputPath());
                    close(inputFd);
                    return false;
                }
                posix_fadvise(outputFd, 0, 0, POSIX_FADV_SEQUENTIAL);
            }
        }
        
        // ── Per-block result structure ────────────────────────────────────────
        // Each block decompresses independently  we store results in a map
        // keyed by block index so the writer can flush in order regardless of
        // which GPU/CPU finished first.
        struct DecompBlock {
            std::vector<uint8_t> data;  // decompressed bytes
            bool gpuPath;               // true=GPU, false=CPU fallback
        };

        std::mutex                  resultMutex;
        std::map<size_t, DecompBlock> results;    // blockIdx → decompressed
        std::atomic<size_t>         blocksQueued{0};
        std::atomic<size_t>         blocksDone{0};
        std::atomic<bool>           readDone{false};
        std::atomic<bool>           decompError{false};
        std::atomic<size_t>         gpuBlocks{0}, cpuFallbackBlocks{0};

        // ── Block queue for GPU workers ────────────────────────────────────────
        struct RawBlock {
            size_t idx;
            std::vector<uint8_t> compData; // empty = was uncompressed (pass-through)
            std::vector<uint8_t> rawData;  // only set for uncompressed blocks
            size_t origSize;               // expected decompressed size
        };
        std::queue<RawBlock>    blockQueue;
        std::mutex              blockQueueMutex;
        std::condition_variable blockQueueCV;

        // ── GPU decompressor: N_STREAMS pipeline, slotCapacity blocks per stream ─
        //
        // Design rationale:
        //   - nvcompBatchedLZ4DecompressAsync with batch_size > 1 requires ALL blocks
        //     in the batch to be the same LZ4 format (nvCOMP-internal vs raw LZ4).
        //     Hybrid-compressed files mix both formats, so per-block status checking
        //     and CPU fallback is required.  --batch-size controls the batch size;
        //     default is 1 (safe), higher values let you test Error 12 behaviour.
        //   - N_STREAMS independent streams keep the GPU busy across H→D, compute,
        //     and D→H phases  each stream processes one batch simultaneously.
        //   - Pre-allocated device buffers (allocated once, reused per batch) avoid
        //     per-batch cudaMalloc/cudaFree overhead.
        //   - The worker thread accumulates slotCapacity blocks, dispatches the batch
        //     to the next stream in the ring, then moves on.  Results are collected
        //     when the slot is next reused (or at drain time), so N_STREAMS batches
        //     are always in flight simultaneously.
        //
        // Per-stream VRAM at 4 MB chunk, slotCapacity=8:
        //   comp   :  4 MB × 8 = 32 MB
        //   decomp :  4 MB × 8 = 32 MB
        //   temp   :  4 MB × 8 = 32 MB
        //   Total  : ~96 MB/stream × 32 streams ≈ 3 GB  (scale with SC and N_STREAMS)
        //
        // Tune with --streams-per-gpu N and --batch-size SC.

        // N_STREAMS: number of concurrent GPU decompression streams.
        // Controlled by --streams-per-gpu.  If the user didn't explicitly set it
        // (pipelineDepth was auto-tuned for compression to 4), use a decompression-
        // specific default: 32 for single GPU, 16 for 2-4 GPUs, 8 for 5+ GPUs.
        // More streams = more blocks in flight simultaneously = better GPU utilisation.
        // Each stream uses ~12 MB VRAM (at 4 MB chunk size), so 32 streams = ~384 MB.
        size_t N_STREAMS;
        if (pipelineDepth > 0) {
            // User explicitly set --streams-per-gpu  honour it exactly
            N_STREAMS = (size_t)pipelineDepth;
        } else if (gpus.size() == 1) {
            N_STREAMS = 32;
        } else if (gpus.size() <= 4) {
            N_STREAMS = 16;
        } else {
            N_STREAMS = 8;
        }
        VLOG(VERBOSE, "GPU decompressor: %zu streams per GPU%s, batch-size %zu\n",
             N_STREAMS, pipelineDepth > 0 ? " (user-specified)" : " (auto)",
             slotCapacity);

        // Per-stream slot: device memory for slotCapacity in-flight blocks.
        // d_inPtr[i]  → d_comp   + i * maxCompSize   (pre-filled, never changes)
        // d_outPtr[i] → d_decomp + i * chunkSize     (pre-filled, never changes)
        // Per-batch: only d_inSize[i] changes (actual compressed size per block).
        struct DecompSlot {
            cudaStream_t    stream         = nullptr;
            uint8_t*        d_comp         = nullptr;   // SC * maxCompSize
            uint8_t*        d_decomp       = nullptr;   // SC * chunkSize  (device)
            void*           d_temp         = nullptr;   // SC * tempBytes
            void**          d_inPtr        = nullptr;   // [SC] ptrs into d_comp
            void**          d_outPtr       = nullptr;   // [SC] ptrs into d_decomp
            size_t*         d_inSize       = nullptr;   // [SC] written each batch
            size_t*         d_outBufSize   = nullptr;   // [SC] pre-filled = chunkSize
            size_t*         d_actualSize   = nullptr;   // [SC] written by nvCOMP
            nvcompStatus_t* d_status       = nullptr;   // [SC] written by nvCOMP
            size_t*         h_inSizes      = nullptr;   // pinned [SC] staging
            size_t*         h_actualSize   = nullptr;   // pinned [SC]
            nvcompStatus_t* h_status       = nullptr;   // pinned [SC]
            uint8_t*        h_decomp       = nullptr;   // pinned mirror of d_decomp
            bool            inFlight       = false;
            size_t          batchCount     = 0;         // blocks dispatched this round
            std::vector<size_t> blockIdxs;              // [batchCount]
            size_t          origSize       = 0;
        };

        auto gpuWorker = [&](size_t gpuIdx) {
            GPUDevice& gpu = gpus[gpuIdx];
            cudaSetDevice(gpu.deviceId);

            const size_t maxCompSize = chunkSize + (chunkSize / 255) + 16;
            const size_t tempBytes   = chunkSize;  // conservative nvCOMP scratch per slot
            nvcompBatchedLZ4DecompressOpts_t opts = nvcompBatchedLZ4DecompressDefaultOpts;

            // ── Allocate N_STREAMS slots, each holding slotCapacity blocks ────
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

                // Pre-fill invariant pointer arrays (never change between batches):
                //   d_inPtr[i]      → d_comp   + i * maxCompSize
                //   d_outPtr[i]     → d_decomp + i * chunkSize
                //   d_outBufSize[i] = chunkSize
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
            if (allocatedSlots < N_STREAMS) {
                VLOG(VERBOSE, "GPU%zu: allocated %zu/%zu decompression slots (VRAM limited)\n",
                     gpuIdx, allocatedSlots, N_STREAMS);
            } else {
                VLOG(VERBOSE, "GPU%zu: %zu streams × %zu blocks/batch, ~%.0f MB VRAM\n",
                     gpuIdx, N_STREAMS, SC,
                     N_STREAMS * SC * (maxCompSize + chunkSize + tempBytes) / (1024.0*1024.0));
            }

            {
                // ── Main dispatch loop ────────────────────────────────────────
                // Ring of N_STREAMS in-flight slots.  Each slot now holds up to
                // slotCapacity (SC) blocks in a single nvCOMP batched call.
                // Before reusing a slot we synchronize it and collect results;
                // any block with a non-success nvCOMP status (including Error 12 /
                // nvcompErrorInvalidValue) falls back to LZ4_decompress_safe.

                size_t nextSlot = 0;

                // Per-slot, per-block storage of compressed data.
                // Must stay alive until collectSlot() has finished the D→H copy.
                std::vector<std::vector<std::vector<uint8_t>>>
                    slotCompData(allocatedSlots, std::vector<std::vector<uint8_t>>(SC));

                // ── collectSlot ────────────────────────────────────────────────
                // Synchronize stream, then for each block in the batch:
                //   - If nvCOMP succeeded: copy decompressed data from device.
                //   - If nvCOMP failed (any status, including Error 12):
                //       report the specific error and fall back to CPU.
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
                            // GPU failed  report error then CPU fallback
                            if (st == (nvcompStatus_t)12 /*nvcompErrorInvalidValue*/) {
                                VLOG(VERBOSE,
                                     "Block %zu: nvcompErrorInvalidValue (Error 12, "
                                     "raw LZ4 format), CPU fallback\n",
                                     sl.blockIdxs[j]);
                            } else {
                                VLOG(VERBOSE,
                                     "Block %zu: nvCOMP status %d, CPU fallback\n",
                                     sl.blockIdxs[j], (int)st);
                            }
                            out.data.resize(sl.origSize);
                            int r = LZ4_decompress_safe(
                                reinterpret_cast<const char*>(slotCompData[si][j].data()),
                                reinterpret_cast<char*>(out.data.data()),
                                (int)slotCompData[si][j].size(), (int)sl.origSize);
                            if (r < 0) {
                                fprintf(stderr,
                                        "Error: block %zu GPU st=%d CPU fallback also failed\n",
                                        sl.blockIdxs[j], (int)st);
                                decompError = true;
                                return false;
                            }
                            out.data.resize(r);
                            out.gpuPath = false;
                            cpuFallbackBlocks++;
                        } else {
                            // GPU succeeded  data already in pinned h_decomp
                            out.data.resize(actualOut);
                            memcpy(out.data.data(),
                                   sl.h_decomp + j * chunkSize,
                                   actualOut);
                            out.gpuPath = true;
                            gpuBlocks++;
                            VLOG(VERY_VERBOSE, "Block %zu: GPU ok (%zu bytes)\n",
                                 sl.blockIdxs[j], actualOut);
                        }
                        slotCompData[si][j].clear();
                        {
                            std::lock_guard<std::mutex> lk(resultMutex);
                            results[sl.blockIdxs[j]] = std::move(out);
                        }
                        blocksDone++;
                    }
                    return true;
                };

                // ── dispatchSlot ───────────────────────────────────────────────
                // Copy a batch of compressed blocks to device and launch nvCOMP.
                // The pointer arrays (d_inPtr, d_outPtr, d_outBufSize) were pre-filled
                // at allocation time and are not modified here.
                auto dispatchSlot = [&](size_t si,
                                        std::vector<RawBlock>& batch) {
                    DecompSlot& sl   = slots[si];
                    sl.batchCount    = batch.size();
                    sl.origSize      = batch[0].origSize;
                    sl.blockIdxs.resize(sl.batchCount);

                    for (size_t j = 0; j < sl.batchCount; j++) {
                        sl.blockIdxs[j]    = batch[j].idx;
                        size_t csz         = batch[j].compData.size();
                        sl.h_inSizes[j]    = csz;

                        // Keep compData alive until collectSlot runs
                        slotCompData[si][j] = std::move(batch[j].compData);

                        // H→D: compressed data into its stride slot
                        cudaMemcpyAsync(sl.d_comp + j * maxCompSize,
                                        slotCompData[si][j].data(), csz,
                                        cudaMemcpyHostToDevice, sl.stream);
                    }
                    // H→D: input sizes array (pinned staging → device)
                    cudaMemcpyAsync(sl.d_inSize, sl.h_inSizes,
                                    sl.batchCount * sizeof(size_t),
                                    cudaMemcpyHostToDevice, sl.stream);

                    // Launch batched decompression
                    nvcompStatus_t apiSt = nvcompBatchedLZ4DecompressAsync(
                        (const void* const*)sl.d_inPtr,
                        sl.d_inSize, sl.d_outBufSize, sl.d_actualSize,
                        sl.batchCount,
                        sl.d_temp, SC * tempBytes,
                        (void* const*)sl.d_outPtr,
                        opts, sl.d_status, sl.stream);

                    if (apiSt != nvcompSuccess) {
                        // API-level failure  propagate to all blocks in batch
                        VLOG(VERBOSE,
                             "Slot %zu: nvcompBatchedLZ4DecompressAsync API error "
                             "status=%d (batch=%zu), all blocks will CPU-fallback\n",
                             si, (int)apiSt, sl.batchCount);
                        for (size_t j = 0; j < sl.batchCount; j++)
                            sl.h_status[j] = apiSt;
                        cudaStreamSynchronize(sl.stream);
                    } else {
                        // Async D→H of per-block metadata
                        cudaMemcpyAsync(sl.h_actualSize, sl.d_actualSize,
                                        sl.batchCount * sizeof(size_t),
                                        cudaMemcpyDeviceToHost, sl.stream);
                        cudaMemcpyAsync(sl.h_status, sl.d_status,
                                        sl.batchCount * sizeof(nvcompStatus_t),
                                        cudaMemcpyDeviceToHost, sl.stream);
                        // Async D→H of decompressed data  overlaps with next batch fill
                        cudaMemcpyAsync(sl.h_decomp, sl.d_decomp,
                                        sl.batchCount * chunkSize,
                                        cudaMemcpyDeviceToHost, sl.stream);
                    }
                    sl.inFlight = true;
                };

                // ── Main loop: accumulate SC compressed blocks then dispatch ──
                while (!decompError) {
                    std::vector<RawBlock> batch;
                    batch.reserve(SC);
                    bool reachedEOF = false;

                    // Gather up to SC compressed blocks; handle pass-throughs inline
                    while (batch.size() < SC && !decompError) {
                        RawBlock block;
                        {
                            std::unique_lock<std::mutex> lk(blockQueueMutex);
                            blockQueueCV.wait(lk, [&]{
                                return !blockQueue.empty()
                                    || (readDone && blockQueue.empty());
                            });
                            if (blockQueue.empty()) { reachedEOF = true; break; }
                            block = std::move(blockQueue.front());
                            blockQueue.pop();
                        }
                        blockQueueCV.notify_all();

                        if (block.compData.empty()) {
                            // Pass-through (uncompressed block)  route directly
                            DecompBlock out;
                            out.data    = std::move(block.rawData);
                            out.gpuPath = false;
                            {
                                std::lock_guard<std::mutex> lk(resultMutex);
                                results[block.idx] = std::move(out);
                            }
                            blocksDone++;
                            continue;
                        }
                        batch.push_back(std::move(block));
                    }

                    if (batch.empty()) break;   // EOF with no more compressed blocks

                    // Collect (sync) the slot we're about to reuse
                    if (!collectSlot(nextSlot)) break;

                    // Dispatch the batch
                    dispatchSlot(nextSlot, batch);
                    nextSlot = (nextSlot + 1) % allocatedSlots;

                    if (reachedEOF) break;
                }

                // Collect all remaining in-flight slots
                for (size_t i = 0; i < allocatedSlots && !decompError; i++) {
                    size_t si = (nextSlot + i) % allocatedSlots;
                    collectSlot(si);
                }
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

        // ── Launch GPU worker threads ──────────────────────────────────────────
        std::vector<std::thread> workerThreads;
        workerThreads.reserve(gpus.size());
        for (size_t g = 0; g < gpus.size(); g++)
            workerThreads.emplace_back(gpuWorker, g);

        // ── Reader: parse LZ4 frame blocks and enqueue ─────────────────────────
        size_t nextBlockToRead  = 0;
        size_t nextBlockToWrite = 0;
        size_t totalBytesWritten = 0;
        auto   startTime = std::chrono::high_resolution_clock::now();

        while (!decompError) {
            uint32_t blockSize32 = 0;
            ssize_t  nr = ::read(inputFd, &blockSize32, 4);
            if (nr == 0) break;  // clean EOF
            if (nr != 4) {
                fprintf(stderr, "Error: truncated block-size field at block %zu\n",
                        nextBlockToRead);
                decompError = true; break;
            }

            // LZ4 end mark
            if (blockSize32 == 0) { estimatedBlocks = nextBlockToRead; break; }

            bool isUncomp = (blockSize32 & 0x80000000u) != 0;
            uint32_t blockSize = blockSize32 & 0x7FFFFFFFu;

            if (blockSize > 128u * 1024 * 1024) {
                fprintf(stderr, "Error: implausibly large block %u at block %zu\n",
                        blockSize, nextBlockToRead);
                decompError = true; break;
            }

            std::vector<uint8_t> raw(blockSize);
            if (::read(inputFd, raw.data(), blockSize) != (ssize_t)blockSize) {
                fprintf(stderr, "Error: truncated block data at block %zu\n",
                        nextBlockToRead);
                decompError = true; break;
            }

            VLOG(DEBUG, "Block %zu: size=%u isUncomp=%d\n",
                 nextBlockToRead, blockSize, (int)isUncomp);

            RawBlock rb;
            rb.idx      = nextBlockToRead++;
            rb.origSize = chunkSize;
            if (isUncomp) {
                rb.rawData  = std::move(raw);  // pass-through
                // compData stays empty  worker will see this and skip GPU
            } else {
                rb.compData = std::move(raw);
            }
            blocksQueued++;
            VLOG(VERY_VERBOSE, "Reader: queued block %zu  compSize=%u  isUncomp=%d\n",
                 rb.idx, blockSize, (int)isUncomp);

            {
                std::lock_guard<std::mutex> lk(blockQueueMutex);
                blockQueue.push(std::move(rb));
            }
            blockQueueCV.notify_one();

            // ── Writer: flush completed sequential blocks while reading ─────────
            // Interleave writing with reading so memory doesn't pile up.
            while (!decompError) {
                std::lock_guard<std::mutex> lk(resultMutex);
                auto it = results.find(nextBlockToWrite);
                if (it == results.end()) break;

                auto& blk = it->second;
                xxhState.update(blk.data.data(), blk.data.size());
                if (outputFd >= 0) {
                    if (::write(outputFd, blk.data.data(), blk.data.size())
                            != (ssize_t)blk.data.size()) {
                        fprintf(stderr, "Error: write failed at block %zu\n",
                                nextBlockToWrite);
                        decompError = true;
                    }
                }
                totalBytesWritten += blk.data.size();
                results.erase(it);
                nextBlockToWrite++;
            }

            // Progress display  show bytes written (output side, not read side)
            if (g_verbosity == NORMAL && estimatedBlocks > 10) {
                size_t pct = originalFileSize > 0
                    ? totalBytesWritten * 100 / originalFileSize : 0;
                std::string ws = formatBytes(totalBytesWritten);
                fprintf(stderr, "\r%s: %3zu%%  %s%s",
                        testMode ? "Testing" : "Decompressing",
                        pct, ws.c_str(), "          ");
                fflush(stderr);
            }
        }

        // ── Signal workers that reading is done, then join ─────────────────────
        readDone = true;
        blockQueueCV.notify_all();
        for (auto& t : workerThreads) t.join();

        // ── Final writer flush for any remaining in-order results ──────────────
        while (!decompError) {
            std::lock_guard<std::mutex> lk(resultMutex);
            auto it = results.find(nextBlockToWrite);
            if (it == results.end()) break;
            auto& blk = it->second;
            xxhState.update(blk.data.data(), blk.data.size());
            if (outputFd >= 0) {
                if (::write(outputFd, blk.data.data(), blk.data.size())
                        != (ssize_t)blk.data.size())
                    fprintf(stderr, "Warning: write error in final flush\n");
            }
            totalBytesWritten += blk.data.size();
            results.erase(it);
            nextBlockToWrite++;
            if (!testMode && g_verbosity == NORMAL && estimatedBlocks > 10) {
                size_t pct = originalFileSize > 0
                    ? totalBytesWritten * 100 / originalFileSize : 0;
                std::string ws = formatBytes(totalBytesWritten);
                std::string ts = formatBytes(originalFileSize);
                fprintf(stderr, "\rWriting: %3zu%%  [%s / %s]%s",
                        pct, ws.c_str(), ts.c_str(), "          ");
                fflush(stderr);
            }
        }

        auto endTime  = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        // Force a final 100% update, then let the completion \r overwrite it
        if (g_verbosity == NORMAL && estimatedBlocks > 10) {
            std::string ts = formatBytes(originalFileSize);
            if (testMode)
                fprintf(stderr, "\rTesting: 100%%  %s%s", ts.c_str(), "          ");
            else
                fprintf(stderr, "\rWriting: 100%%  [%s / %s]%s",
                        ts.c_str(), ts.c_str(), "          ");
            fflush(stderr);
        }
        
        // ── Verify content checksum ───────────────────────────────────────────
        // LZ4 frame layout: ... [block data] [end-mark = 0x00000000] [checksum 4B]
        // The read loop breaks immediately after consuming the 0x00000000 end-mark,
        // so the cursor is already positioned at the checksum  just read forward.
        uint32_t computedChecksum = xxhState.digest();
        uint32_t storedChecksum   = 0;
        bool     checksumOk       = false;
        if (::read(inputFd, &storedChecksum, 4) == 4) {
            // LZ4 stores checksum little-endian  already correct on x86
            checksumOk = (computedChecksum == storedChecksum);
            if (!checksumOk) {
                fprintf(stderr, "Warning: Checksum mismatch  file may be corrupted!\n");
                fprintf(stderr, "  Stored:   0x%08X\n", storedChecksum);
                fprintf(stderr, "  Computed: 0x%08X\n", computedChecksum);
            }
        } else {
            fprintf(stderr, "Warning: Could not read stored checksum (truncated file?)\n");
        }

        close(inputFd);
        if (outputFd >= 0 && outputFd != STDOUT_FILENO) { fsync(outputFd); close(outputFd); }

        if (testMode) {
            struct stat st;
            size_t compressedSize = (stat(inputFile.c_str(), &st) == 0) ? st.st_size : 0;
            double ratio = compressedSize > 0 ? (100.0 * compressedSize / totalBytesWritten) : 0.0;
            std::string outputSize = formatBytes(totalBytesWritten);
            
            VLOG(NORMAL, "\rTest complete (GPU+fallback, %zu GPU%s): %s in %.2f s%s\n",
                    gpus.size(), gpus.size() == 1 ? "" : "s",
                    outputSize.c_str(), duration.count() / 1000.0,
                    "          ");
            VLOG(NORMAL, checksumOk ? "Test OK: %s\n" : "Test FAILED: %s (checksum mismatch)\n",
                    inputFile.c_str());
            VLOG(VERBOSE, "  Compressed:   %.2f MB\n", compressedSize / (1024.0*1024.0));
            VLOG(VERBOSE, "  Uncompressed: %.2f MB  (ratio %.2f%%)\n",
                 totalBytesWritten / (1024.0*1024.0), ratio);
            VLOG(VERBOSE, "  Time: %.2f s  Throughput: %.2f MB/s\n",
                 duration.count() / 1000.0,
                 (totalBytesWritten / (1024.0*1024.0)) / (duration.count() / 1000.0));
            VLOG(VERBOSE, "  GPU blocks: %zu  CPU-fallback blocks: %zu\n",
                 gpuBlocks.load(), cpuFallbackBlocks.load());
        } else {
            double mbps = (totalBytesWritten / (1024.0*1024.0)) / (duration.count() / 1000.0);
            std::string outputSize = formatBytes(totalBytesWritten);
            VLOG(NORMAL, "\rDecompression complete (GPU+fallback, %zu GPU%s): "
                    "%s in %.2f s%s\n",
                    gpus.size(), gpus.size() == 1 ? "" : "s",
                    outputSize.c_str(), duration.count() / 1000.0,
                    "          ");
            VLOG(VERBOSE, "Throughput: %.2f MB/s\n", mbps);
            VLOG(VERBOSE, "  GPU blocks: %zu  CPU-fallback: %zu  pass-through: %zu\n",
                 gpuBlocks.load(), cpuFallbackBlocks.load(),
                 nextBlockToWrite - gpuBlocks.load() - cpuFallbackBlocks.load());
        }
        
        // Remove compressed file if not keeping
        if (!keepOriginal && !stdoutMode && !testMode) {
            if (unlink(inputFile.c_str()) != 0) {
                fprintf(stderr, "Warning: Could not remove compressed file: %s\n",
                        inputFile.c_str());
            }
        }
        
        return true;
    }
    
    /*
     * Decompress file  dispatches to the right backend.
     *
     * --gpu-only:  GPU workers with CPU fallback per-block (Error 12 fixed)
     * --hybrid:    GPU workers + CPU thread pool, dynamic load balancing
     * --cpu-only:  CPU thread pool only (LZ4_decompress_safe, always works)
     * default:     same as --cpu-only (safest, always correct)
     */
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

    /*
     * Hybrid decompression  GPU workers + CPU thread pool working together.
     *
     * Architecture (mirrors hybrid compression):
     *
     *   Dispatcher thread:
     *     Parses LZ4 frame blocks from disk and routes each RawBlock:
     *        GPU queue first  (capacity = N_STREAMS_H × SC × nGPUs × 2)
     *        CPU queue when GPU queue is at capacity (GPU-priority policy)
     *     Pass-through (uncompressed) blocks bypass both queues and go
     *     directly into the result map.
     *
     *   GPU worker threads (one per GPU):
     *     Same N_STREAMS_H slot-ring + slotCapacity (SC) batch as GPU-only.
     *     nvCOMP handles both nvCOMP-format and raw-LZ4 blocks natively.
     *     Any per-block nvCOMP error falls back to LZ4_decompress_safe inline.
     *
     *   CPU worker threads (effectiveThreads total):
     *     Each pulls one block at a time from the CPU queue and calls
     *     LZ4_decompress_safe  handles overflow when GPUs are saturated.
     *
     *   Both paths post into the shared ordered results map; the main thread
     *   drains it in block-index order for writing and checksum.
     */
    bool decompressFileHybrid() {
        // Open input file
        int inputFd = ::open(inputFile.c_str(), O_RDONLY | O_LARGEFILE);
        if (inputFd < 0) {
            fprintf(stderr, "Error opening input file: %s\n", strerror(errno));
            return false;
        }
        posix_fadvise(inputFd, 0, 0, POSIX_FADV_SEQUENTIAL);
        posix_fadvise(inputFd, 0, 0, POSIX_FADV_WILLNEED);

        // Parse LZ4 frame header
        uint8_t headerBuf[32];
        ssize_t headerRead = ::read(inputFd, headerBuf, 32);
        if (headerRead < 15) {
            fprintf(stderr, "Error: File too small to be valid LZ4\n");
            close(inputFd); return false;
        }
        LZ4Frame::FrameDescriptor desc;
        size_t headerBytes = 0;
        {
            std::string hs((char*)headerBuf, headerRead);
            std::istringstream hstream(hs, std::ios::binary);
            if (!LZ4Frame::readFrameHeader(hstream, desc)) {
                fprintf(stderr, "Error: Failed to read LZ4 frame header\n");
                close(inputFd); return false;
            }
            headerBytes = hstream.tellg();
        }
        lseek(inputFd, (off_t)headerBytes, SEEK_SET);

        size_t originalFileSize = desc.contentSize;
        size_t chunkSize        = static_cast<size_t>(1) << (8 + 2 * desc.blockMaxSize);
        size_t estimatedBlocks  = (originalFileSize + chunkSize - 1) / chunkSize;

        // Determine effective CPU thread count
        size_t effectiveThreads = cpuThreads ? cpuThreads
                                             : std::thread::hardware_concurrency();
        if (!effectiveThreads) effectiveThreads = 4;

        if (testMode) {
            VLOG(NORMAL, "Testing (hybrid, %zu GPU%s + %zu thread%s): %s\n",
                    gpus.size(),       gpus.size()       == 1 ? "" : "s",
                    effectiveThreads,  effectiveThreads  == 1 ? "" : "s",
                    inputFile.c_str());
        } else {
            VLOG(NORMAL, "Decompressing (hybrid, %zu GPU%s + %zu thread%s): %s -> %s\n",
                    gpus.size(),       gpus.size()       == 1 ? "" : "s",
                    effectiveThreads,  effectiveThreads  == 1 ? "" : "s",
                    inputFile.c_str(), outputFile.c_str());
        }
        VLOG(VERBOSE, "  %.2f MB  |  block size %zu KB  |  ~%zu blocks\n",
             originalFileSize / (1024.0*1024.0), chunkSize/1024, estimatedBlocks);

        // Open output
        int outputFd = -1;
        if (!testMode) {
            if (stdoutMode) {
                outputFd = STDOUT_FILENO;
            } else {
                outputFd = open(getActualOutputPath(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
                if (outputFd < 0) {
                    fprintf(stderr, "Error: Cannot create output file: %s\n", getActualOutputPath());
                    close(inputFd); return false;
                }
            }
        }

        XXH::State xxhState(XXH32_SEED);

        // ── Shared types ──────────────────────────────────────────────────────
        struct DecompBlock { std::vector<uint8_t> data; };
        struct RawBlock {
            size_t               idx;
            std::vector<uint8_t> compData;  // non-empty = compressed block
            std::vector<uint8_t> rawData;   // non-empty = uncompressed pass-through
            size_t               origSize;
        };

        // ── Shared state ──────────────────────────────────────────────────────
        std::mutex                    resultMutex;
        std::map<size_t, DecompBlock> results;
        std::atomic<bool>             decompError{false};
        std::atomic<size_t>           gpuBlocks{0}, cpuBlocks{0};
        std::atomic<size_t>           blocksDone{0};

        // ── VRAM-aware stream count ───────────────────────────────────────────
        // Each stream needs SC * (maxCompSize + chunkSize + tempBytes) bytes.
        // maxCompSize ≈ chunkSize + chunkSize/255 + 16 ≈ chunkSize * 1.004
        // tempBytes   = chunkSize  (conservative scratch)
        // → perStream ≈ SC * 3 * chunkSize  (use exactly for the cap calc)
        const size_t SC_h          = slotCapacity;
        const size_t maxComp_h     = chunkSize + (chunkSize / 255) + 16;
        const size_t perStreamVRAM = SC_h * (maxComp_h + chunkSize + chunkSize);

        size_t N_STREAMS_H;
        if (pipelineDepth > 0) {
            N_STREAMS_H = (size_t)pipelineDepth;
            VLOG(DEBUG, "N_STREAMS_H = %zu (user --streams-per-gpu)\n", N_STREAMS_H);
        } else {
            // Use at most 50% of free VRAM on GPU 0 for decompression slots.
            // gpus[0].availableMemory is already the free bytes (from cudaMemGetInfo).
            size_t freeVRAM    = gpus[0].availableMemory;
            size_t target      = freeVRAM / 2;          // 50% cap
            size_t autoStreams = target / perStreamVRAM;
            if (autoStreams < 1)  autoStreams = 1;
            if (autoStreams > 32) autoStreams = 32;      // sanity ceiling
            N_STREAMS_H = autoStreams;
            VLOG(DEBUG, "N_STREAMS_H auto: freeVRAM=%.1f GB  perStream=%.0f MB  "
                 "target=%.1f GB  -> %zu streams\n",
                 freeVRAM / (1024.0*1024.0*1024.0),
                 perStreamVRAM / (1024.0*1024.0),
                 target / (1024.0*1024.0*1024.0),
                 N_STREAMS_H);
        }
        VLOG(VERBOSE, "Hybrid decompressor: %zu streams/GPU%s, batch-size %zu, "
             "%zu CPU thread%s  (~%.0f MB VRAM for GPU slots)\n",
             N_STREAMS_H, pipelineDepth > 0 ? " (user-specified)" : " (auto)",
             SC_h, effectiveThreads, effectiveThreads == 1 ? "" : "s",
             N_STREAMS_H * perStreamVRAM / (1024.0*1024.0));

        // GPU queue capacity: enough to keep all streams filled 2× over.
        // CPU queue is unbounded  only receives overflow.
        const size_t gpuQueueCap = N_STREAMS_H * SC_h * gpus.size() * 2;
        VLOG(DEBUG, "gpuQueueCap = %zu blocks\n", gpuQueueCap);
        TsQueue<RawBlock> gpuWorkQueue;
        TsQueue<RawBlock> cpuWorkQueue;

        // ── Dispatcher thread: reads blocks from disk, routes GPU-first ─────────
        std::atomic<bool>   dispatcherDone{false};
        std::atomic<size_t> totalBlocks{0};   // set once when end-mark is seen
        std::thread dispatcherThread([&]() {
            VLOG(DEBUG, "Dispatcher: started\n");
            size_t blockIdx = 0;
            size_t nGpu = 0, nCpu = 0, nPass = 0;
            while (!decompError) {
                uint32_t bs32 = 0;
                ssize_t nr = ::read(inputFd, &bs32, 4);
                if (nr == 0 || bs32 == 0) {
                    VLOG(DEBUG, "Dispatcher: end-mark/EOF at block %zu "
                         "(nr=%zd bs32=%u)\n", blockIdx, nr, bs32);
                    break;
                }
                if (nr != 4) {
                    fprintf(stderr, "Dispatcher: short read at block %zu\n", blockIdx);
                    decompError = true; break;
                }

                bool isUncomp = (bs32 & 0x80000000u) != 0;
                uint32_t bs   =  bs32 & 0x7FFFFFFFu;
                if (bs > 128u * 1024 * 1024) {
                    fprintf(stderr, "Error: implausibly large block %u at block %zu\n",
                            bs, blockIdx);
                    decompError = true; break;
                }

                std::vector<uint8_t> raw(bs);
                if (::read(inputFd, raw.data(), bs) != (ssize_t)bs) {
                    fprintf(stderr, "Error: truncated block data at block %zu\n", blockIdx);
                    decompError = true; break;
                }

                RawBlock rb;
                rb.idx      = blockIdx;
                rb.origSize = chunkSize;

                if (isUncomp) {
                    // Pass-through: write directly into result map
                    DecompBlock out;
                    out.data = std::move(raw);
                    {
                        std::lock_guard<std::mutex> lk(resultMutex);
                        results[rb.idx] = std::move(out);
                    }
                    blocksDone++;
                    nPass++;
                    VLOG(DEBUG, "Dispatcher: block %zu pass-through (%u B)\n",
                         blockIdx, bs);
                } else {
                    rb.compData = std::move(raw);
                    if (gpuWorkQueue.size() < gpuQueueCap) {
                        gpuWorkQueue.push(std::move(rb));
                        nGpu++;
                        VLOG(DEBUG, "Dispatcher: block %zu -> gpuQ (%zu/%zu)\n",
                             blockIdx, gpuWorkQueue.size(), gpuQueueCap);
                    } else {
                        cpuWorkQueue.push(std::move(rb));
                        nCpu++;
                        VLOG(DEBUG, "Dispatcher: block %zu -> cpuQ (gpuQ full)\n",
                             blockIdx);
                    }
                }
                blockIdx++;
            }
            totalBlocks.store(blockIdx);
            gpuWorkQueue.close();
            cpuWorkQueue.close();
            dispatcherDone.store(true);
            VLOG(DEBUG, "Dispatcher: done. total=%zu  gpu=%zu cpu=%zu pass=%zu  "
                 "queues closed\n", blockIdx, nGpu, nCpu, nPass);
        });

        // ── GPU worker lambda (one thread per GPU) ────────────────────────────
        auto gpuWorker = [&](size_t gpuIdx) {
            GPUDevice& gpu = gpus[gpuIdx];
            cudaSetDevice(gpu.deviceId);
            VLOG(DEBUG, "GPU%zu worker: started (deviceId=%d)\n",
                 gpuIdx, gpu.deviceId);

            const size_t SC          = SC_h;
            const size_t maxCompSize = maxComp_h;
            const size_t tempBytes   = chunkSize;
            nvcompBatchedLZ4DecompressOpts_t opts = nvcompBatchedLZ4DecompressDefaultOpts;

            struct HDecompSlot {
                cudaStream_t    stream       = nullptr;
                uint8_t*        d_comp       = nullptr;
                uint8_t*        d_decomp     = nullptr;  // device output
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
                uint8_t*        h_decomp     = nullptr;  // pinned host mirror of d_decomp
                bool            inFlight     = false;
                size_t          batchCount   = 0;
                std::vector<size_t> blockIdxs;
                size_t          origSize     = 0;
            };

            // Allocate slots one at a time; stop when VRAM runs out.
            // Each fully-allocated slot is pushed onto `slots` before we try
            // the next one, so partial failures only waste one attempt.
            std::vector<HDecompSlot> slots;
            slots.reserve(N_STREAMS_H);

            for (size_t si = 0; si < N_STREAMS_H; si++) {
                HDecompSlot sl;
                cudaError_t e;
                bool ok = true;
                auto tryAlloc = [&](cudaError_t err, const char* what) {
                    if (err != cudaSuccess) {
                        VLOG(VERBOSE, "GPU%zu slot %zu: %s failed (%s)  "
                             "stopping at %zu slots\n",
                             gpuIdx, si, what, cudaGetErrorString(err),
                             slots.size());
                        cudaGetLastError();   // clear sticky error
                        ok = false;
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
                    // Free whatever was partially allocated for this slot
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

                // Pre-fill invariant pointer arrays
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
                VLOG(DEBUG, "GPU%zu: slot %zu ok  VRAM %.0f MB  pinned %.0f MB\n",
                     gpuIdx, si,
                     slots.size() * perStreamVRAM / (1024.0*1024.0),
                     slots.size() * SC * chunkSize / (1024.0*1024.0));
            }

            const size_t allocatedSlots = slots.size();
            VLOG(VERBOSE, "Hybrid GPU%zu: %zu/%zu streams × %zu blocks, "
                 "~%.0f MB VRAM + %.0f MB pinned host\n",
                 gpuIdx, allocatedSlots, N_STREAMS_H, SC,
                 allocatedSlots * perStreamVRAM / (1024.0*1024.0),
                 allocatedSlots * SC * chunkSize / (1024.0*1024.0));

            if (allocatedSlots == 0) {
                fprintf(stderr,
                        "GPU%zu: zero slots allocated  re-routing GPU queue to CPU\n",
                        gpuIdx);
                // Drain the GPU queue into the CPU queue so work is not lost
                RawBlock block;
                size_t rerouted = 0;
                while (gpuWorkQueue.pop(block, 10))
                    { cpuWorkQueue.push(std::move(block)); rerouted++; }
                VLOG(DEBUG, "GPU%zu: re-routed %zu blocks -> cpuQ\n", gpuIdx, rerouted);
                return;
            }

            // Per-slot compressed data kept alive until D→H finishes
            std::vector<std::vector<std::vector<uint8_t>>>
                slotCompData(allocatedSlots, std::vector<std::vector<uint8_t>>(SC));
            size_t nextSlot   = 0;
            size_t blocksProc = 0;

            // ── collectSlot ───────────────────────────────────────────────────
            auto collectSlot = [&](size_t si) -> bool {
                HDecompSlot& sl = slots[si];
                if (!sl.inFlight) return true;
                VLOG(DEBUG, "GPU%zu: collectSlot %zu (batch=%zu)\n",
                     gpuIdx, si, sl.batchCount);
                cudaStreamSynchronize(sl.stream);
                sl.inFlight = false;
                for (size_t j = 0; j < sl.batchCount; j++) {
                    nvcompStatus_t st = sl.h_status[j];
                    size_t actualOut  = sl.h_actualSize[j];
                    DecompBlock out;
                    if (st != nvcompSuccess || actualOut == 0) {
                        VLOG(VERBOSE, "GPU%zu block %zu: nvCOMP st=%d, "
                             "inline CPU fallback\n",
                             gpuIdx, sl.blockIdxs[j], (int)st);
                        out.data.resize(sl.origSize);
                        int r = LZ4_decompress_safe(
                            reinterpret_cast<const char*>(slotCompData[si][j].data()),
                            reinterpret_cast<char*>(out.data.data()),
                            (int)slotCompData[si][j].size(), (int)sl.origSize);
                        if (r < 0) {
                            fprintf(stderr,
                                    "Error: GPU%zu block %zu st=%d CPU fallback failed\n",
                                    gpuIdx, sl.blockIdxs[j], (int)st);
                            decompError = true; return false;
                        }
                        out.data.resize(r);
                        cpuBlocks++;
                    } else {
                        // D→H already completed async on the stream before
                        // cudaStreamSynchronize returned  just memcpy from pinned.
                        out.data.resize(actualOut);
                        memcpy(out.data.data(),
                               sl.h_decomp + j * chunkSize,
                               actualOut);
                        gpuBlocks++;
                        VLOG(DEBUG, "GPU%zu block %zu: ok (%zu B)\n",
                             gpuIdx, sl.blockIdxs[j], actualOut);
                    }
                    slotCompData[si][j].clear();
                    { std::lock_guard<std::mutex> lk(resultMutex);
                      results[sl.blockIdxs[j]] = std::move(out); }
                    blocksDone++;
                    blocksProc++;
                }
                return true;
            };

            // ── dispatchSlot ──────────────────────────────────────────────────
            auto dispatchSlot = [&](size_t si, std::vector<RawBlock>& batch) {
                HDecompSlot& sl = slots[si];
                sl.batchCount  = batch.size();
                sl.origSize    = batch[0].origSize;
                sl.blockIdxs.resize(sl.batchCount);
                VLOG(DEBUG, "GPU%zu: dispatchSlot %zu batch=%zu [%zu..%zu]\n",
                     gpuIdx, si, sl.batchCount,
                     batch.front().idx, batch.back().idx);
                for (size_t j = 0; j < sl.batchCount; j++) {
                    sl.blockIdxs[j]     = batch[j].idx;
                    size_t csz          = batch[j].compData.size();
                    sl.h_inSizes[j]     = csz;
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
                    VLOG(VERBOSE, "GPU%zu slot %zu: API err %d, all -> CPU fallback\n",
                         gpuIdx, si, (int)apiSt);
                    for (size_t j = 0; j < sl.batchCount; j++) sl.h_status[j] = apiSt;
                    cudaStreamSynchronize(sl.stream);
                } else {
                    // Async D→H: per-block metadata
                    cudaMemcpyAsync(sl.h_actualSize, sl.d_actualSize,
                                    sl.batchCount * sizeof(size_t),
                                    cudaMemcpyDeviceToHost, sl.stream);
                    cudaMemcpyAsync(sl.h_status, sl.d_status,
                                    sl.batchCount * sizeof(nvcompStatus_t),
                                    cudaMemcpyDeviceToHost, sl.stream);
                    // Async D→H: decompressed data for the whole batch.
                    // Transfers all SC*chunkSize bytes; collectSlot uses
                    // h_actualSize[j] to know how many bytes per block are valid.
                    // This races ahead on the stream while the GPU worker is
                    // busy filling the next batch, so by the time collectSlot
                    // calls cudaStreamSynchronize the data is already in host RAM.
                    cudaMemcpyAsync(sl.h_decomp, sl.d_decomp,
                                    sl.batchCount * chunkSize,
                                    cudaMemcpyDeviceToHost, sl.stream);
                }
                sl.inFlight = true;
            };

            // ── Main dispatch loop ────────────────────────────────────────────
            VLOG(DEBUG, "GPU%zu: dispatch loop start (%zu slots, SC=%zu)\n",
                 gpuIdx, allocatedSlots, SC);
            bool queueDone = false;
            while (!decompError && !queueDone) {
                std::vector<RawBlock> batch;
                batch.reserve(SC);
                while ((int)batch.size() < (int)SC && !decompError) {
                    RawBlock block;
                    if (!gpuWorkQueue.pop(block, 10)) {
                        if (gpuWorkQueue.isClosed()) {
                            VLOG(DEBUG, "GPU%zu: gpuQ closed, partial batch=%zu\n",
                                 gpuIdx, batch.size());
                            queueDone = true; break;
                        }
                        continue;
                    }
                    batch.push_back(std::move(block));
                }
                if (batch.empty()) break;
                if (!collectSlot(nextSlot)) break;
                dispatchSlot(nextSlot, batch);
                nextSlot = (nextSlot + 1) % allocatedSlots;
            }

            // Drain all in-flight slots
            VLOG(DEBUG, "GPU%zu: draining in-flight slots\n", gpuIdx);
            for (size_t i = 0; i < allocatedSlots && !decompError; i++)
                collectSlot((nextSlot + i) % allocatedSlots);
            VLOG(DEBUG, "GPU%zu worker: done (blocksProc=%zu)\n", gpuIdx, blocksProc);

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
        };  // end gpuWorker lambda

        // ── CPU worker lambda: pulls overflow blocks from cpuWorkQueue ────────
        auto cpuWorker = [&](size_t tidx) {
            VLOG(DEBUG, "CPU worker %zu: started\n", tidx);
            size_t processed = 0;
            while (!decompError) {
                RawBlock block;
                if (!cpuWorkQueue.pop(block, 50)) {
                    if (cpuWorkQueue.isClosed()) break;
                    continue;
                }
                DecompBlock out;
                out.data.resize(block.origSize);
                int r = LZ4_decompress_safe(
                    reinterpret_cast<const char*>(block.compData.data()),
                    reinterpret_cast<char*>(out.data.data()),
                    (int)block.compData.size(),
                    (int)block.origSize);
                if (r < 0) {
                    fprintf(stderr,
                            "Error: CPU worker %zu failed block %zu "
                            "(compSz=%zu origSz=%zu)\n",
                            tidx, block.idx,
                            block.compData.size(), block.origSize);
                    decompError = true; break;
                }
                out.data.resize(r);
                VLOG(DEBUG, "CPU worker %zu: block %zu ok (%d B)\n",
                     tidx, block.idx, r);
                { std::lock_guard<std::mutex> lk(resultMutex);
                  results[block.idx] = std::move(out); }
                cpuBlocks++;
                blocksDone++;
                processed++;
            }
            VLOG(DEBUG, "CPU worker %zu: done (processed=%zu)\n", tidx, processed);
        };  // end cpuWorker lambda

        // ── Launch all workers ────────────────────────────────────────────────
        auto startTime = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> allWorkers;
        allWorkers.reserve(gpus.size() + effectiveThreads);
        for (size_t g = 0; g < gpus.size(); g++)
            allWorkers.emplace_back(gpuWorker, g);
        for (size_t t = 0; t < effectiveThreads; t++)
            allWorkers.emplace_back(cpuWorker, t);
        VLOG(DEBUG, "Launched %zu GPU + %zu CPU workers\n",
             gpus.size(), effectiveThreads);

        // ── Main thread: drain ordered results and write ──────────────────────
        // Termination: dispatcherDone=true, totalBlocks known, blocksDone==totalBlocks.
        // Using blocksDone (not nextBlockToWrite) avoids the pass-through count race.
        size_t nextBlockToWrite  = 0;
        size_t totalBytesWritten = 0;
        size_t dbgIter           = 0;

        while (true) {
            bool flushedAny = false;
            while (!decompError) {
                std::lock_guard<std::mutex> lk(resultMutex);
                auto it = results.find(nextBlockToWrite);
                if (it == results.end()) break;
                auto& blk = it->second;
                xxhState.update(blk.data.data(), blk.data.size());
                if (outputFd >= 0 &&
                    ::write(outputFd, blk.data.data(), blk.data.size())
                        != (ssize_t)blk.data.size()) {
                    fprintf(stderr, "Warning: write error at block %zu\n",
                            nextBlockToWrite);
                    decompError = true;
                }
                totalBytesWritten += blk.data.size();
                results.erase(it);
                nextBlockToWrite++;
                flushedAny = true;
            }

            if (decompError) break;

            size_t tb = totalBlocks.load();
            size_t bd = blocksDone.load();
            if (dispatcherDone.load() && tb > 0 && bd >= tb) {
                VLOG(DEBUG, "Main: termination  totalBlocks=%zu blocksDone=%zu "
                     "written=%zu\n", tb, bd, nextBlockToWrite);
                break;
            }

            if (!flushedAny)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));

            // DEBUG heartbeat every ~2 s
            if (++dbgIter % 2000 == 0) {
                VLOG(DEBUG, "Main heartbeat: written=%zu  gpuQ=%zu cpuQ=%zu  "
                     "results=%zu  blocksDone=%zu/%zu  dispDone=%d\n",
                     nextBlockToWrite,
                     gpuWorkQueue.size(), cpuWorkQueue.size(),
                     results.size(), bd, tb,
                     (int)dispatcherDone.load());
            }

            if (g_verbosity == NORMAL && estimatedBlocks > 10) {
                size_t pct = originalFileSize > 0
                    ? totalBytesWritten * 100 / originalFileSize : 0;
                std::string gpuStr = formatBytes(gpuBlocks.load() * chunkSize);
                std::string cpuStr = formatBytes(cpuBlocks.load() * chunkSize);
                fprintf(stderr, "\r%s: %3zu%%  GPU: %s  CPU: %s%s",
                        testMode ? "Testing" : "Decompressing",
                        pct, gpuStr.c_str(), cpuStr.c_str(), "          ");
                fflush(stderr);
            }
        }

        VLOG(DEBUG, "Main: joining %zu workers\n", allWorkers.size() + 1);
        if (dispatcherThread.joinable()) dispatcherThread.join();
        for (auto& t : allWorkers) t.join();
        VLOG(DEBUG, "Main: all joined\n");

        // Final drain
        while (!decompError) {
            std::lock_guard<std::mutex> lk(resultMutex);
            auto it = results.find(nextBlockToWrite);
            if (it == results.end()) break;
            auto& blk = it->second;
            xxhState.update(blk.data.data(), blk.data.size());
            if (outputFd >= 0 &&
                ::write(outputFd, blk.data.data(), blk.data.size())
                    != (ssize_t)blk.data.size())
                fprintf(stderr, "Warning: write error in final flush block %zu\n",
                        nextBlockToWrite);
            totalBytesWritten += blk.data.size();
            results.erase(it);
            nextBlockToWrite++;
        }

        auto endTime  = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        if (g_verbosity == NORMAL && estimatedBlocks > 10) {
            std::string gpuStr = formatBytes(gpuBlocks.load() * chunkSize);
            std::string cpuStr = formatBytes(cpuBlocks.load() * chunkSize);
            fprintf(stderr, "\r%s: 100%%  GPU: %s  CPU: %s%s\n",
                    testMode ? "Testing" : "Decompressing",
                    gpuStr.c_str(), cpuStr.c_str(), "          ");
            fflush(stderr);
        }

        // ── Verify content checksum ───────────────────────────────────────────
        uint32_t computedCS = xxhState.digest();
        uint32_t storedCS   = 0;
        bool     csOk       = false;
        if (::read(inputFd, &storedCS, 4) == 4) {
            csOk = (computedCS == storedCS);
            if (!csOk)
                fprintf(stderr,
                        "Warning: checksum mismatch  stored 0x%08X computed 0x%08X\n",
                        storedCS, computedCS);
        } else {
            fprintf(stderr, "Warning: could not read stored checksum\n");
        }
        close(inputFd);
        if (outputFd >= 0 && outputFd != STDOUT_FILENO) { fsync(outputFd); close(outputFd); }

        double mbps = totalBytesWritten > 0 && duration.count() > 0
            ? (totalBytesWritten / (1024.0*1024.0)) / (duration.count() / 1000.0) : 0.0;
        std::string outputSize = formatBytes(totalBytesWritten);
        size_t passthroughBlocks = nextBlockToWrite
            - gpuBlocks.load() - cpuBlocks.load();

        if (testMode) {
            VLOG(NORMAL, "\rTest complete (hybrid, %zu GPU%s + %zu thread%s): "
                    "%s in %.2f s%s\n",
                    gpus.size(),       gpus.size()       == 1 ? "" : "s",
                    effectiveThreads,  effectiveThreads  == 1 ? "" : "s",
                    outputSize.c_str(), duration.count() / 1000.0, "          ");
            VLOG(NORMAL, csOk ? "Test OK: %s\n" : "Test FAILED: %s (checksum mismatch)\n",
                    inputFile.c_str());
        } else {
            VLOG(NORMAL, "\rDecompression complete (hybrid, %zu GPU%s + %zu thread%s): "
                    "%s in %.2f s%s\n",
                    gpus.size(),       gpus.size()       == 1 ? "" : "s",
                    effectiveThreads,  effectiveThreads  == 1 ? "" : "s",
                    outputSize.c_str(), duration.count() / 1000.0, "          ");
        }
        VLOG(VERBOSE, "  GPU: %zu blocks (%.1f%%)  CPU: %zu blocks (%.1f%%)"
             "  pass-through: %zu  throughput: %.2f MB/s\n",
             gpuBlocks.load(),
             nextBlockToWrite > 0 ? 100.0 * gpuBlocks.load()  / nextBlockToWrite : 0.0,
             cpuBlocks.load(),
             nextBlockToWrite > 0 ? 100.0 * cpuBlocks.load()  / nextBlockToWrite : 0.0,
             passthroughBlocks, mbps);

        if (!keepOriginal && !stdoutMode && !testMode)
            unlink(inputFile.c_str());

        return !decompError;
    }
    
    /*
     * CPU-based decompressor using LZ4_decompress_safe.
     * Works for files compressed by --cpu-only, --gpu-only, or --hybrid,
     * since all compressed blocks are standard raw LZ4 block format.
     * Uses a thread pool for parallel decompression.
     */
    bool decompressFileCPU() {
        // Open input file
        int inputFd = ::open(inputFile.c_str(), O_RDONLY | O_LARGEFILE);
        if (inputFd < 0) {
            fprintf(stderr, "Error opening input file: %s\n", strerror(errno));
            return false;
        }

        // Hint kernel to read ahead aggressively - biggest win for throughput
        posix_fadvise(inputFd, 0, 0, POSIX_FADV_SEQUENTIAL);
        posix_fadvise(inputFd, 0, 0, POSIX_FADV_WILLNEED);

        // Read and parse LZ4 frame header
        uint8_t headerBuf[32];
        ssize_t headerRead = ::read(inputFd, headerBuf, 32);
        if (headerRead < 15) {
            fprintf(stderr, "Error: File too small to be valid LZ4\n");
            close(inputFd); return false;
        }
        LZ4Frame::FrameDescriptor desc;
        size_t headerBytes = 0;
        {
            std::string hs((char*)headerBuf, headerRead);
            std::istringstream hstream(hs, std::ios::binary);
            if (!LZ4Frame::readFrameHeader(hstream, desc)) {
                fprintf(stderr, "Error: Failed to read LZ4 frame header\n");
                close(inputFd); return false;
            }
            headerBytes = hstream.tellg();
        }
        if (lseek(inputFd, (off_t)headerBytes, SEEK_SET) == (off_t)-1) {
            fprintf(stderr, "Error seeking past header: %s\n", strerror(errno));
            close(inputFd); return false;
        }

        size_t originalFileSize = desc.contentSize;
        size_t chunkSize = static_cast<size_t>(1) << (8 + 2 * desc.blockMaxSize);
        size_t estimatedBlocks = originalFileSize > 0 ?
            (originalFileSize + chunkSize - 1) / chunkSize : 0;

        size_t numWorkers = (cpuThreads > 0) ? cpuThreads :
                            std::thread::hardware_concurrency();
        if (numWorkers == 0) numWorkers = 4;

        if (testMode) {
            VLOG(NORMAL, "Testing (CPU, %zu thread%s): %s\n",
                    numWorkers, numWorkers == 1 ? "" : "s",
                    inputFile.c_str());
        } else {
            VLOG(NORMAL, "Decompressing (CPU, %zu thread%s): %s -> %s\n",
                    numWorkers, numWorkers == 1 ? "" : "s",
                    inputFile.c_str(), outputFile.c_str());
        }
        VLOG(VERBOSE, "  %.2f MB source  |  block size %zu KB  |  ~%zu blocks\n",
             originalFileSize/(1024.0*1024.0), chunkSize/1024, estimatedBlocks);
        VLOG(VERBOSE, "CPU decompression: %zu worker threads\n", numWorkers);

        // Open output file (or null for test mode)
        int outputFd = -1;
        if (!testMode) {
            if (stdoutMode) {
                outputFd = STDOUT_FILENO;
            } else {
                outputFd = ::open(getActualOutputPath(),
                                  O_WRONLY | O_CREAT | O_TRUNC | O_LARGEFILE, 0644);
                if (outputFd < 0) {
                    fprintf(stderr, "Error opening output '%s': %s\n",
                            getActualOutputPath(), strerror(errno));
                    close(inputFd); return false;
                }
            }
        }

        // ── shared block queue (reader → workers) ──────────────────────
        struct RawBlock {
            size_t   blockIdx;
            bool     isUncompressed;
            std::vector<uint8_t> data;
        };
        std::queue<RawBlock>    rawQueue;
        std::mutex              rawMutex;
        std::condition_variable rawCV;
        std::atomic<bool>       readerDone{false};
        std::atomic<bool>       readError{false};
        const size_t            RAW_HWM = numWorkers * 8;  // high-water-mark

        // ── result map (workers → writer) ─────────────────────────────
        struct DecompResult {
            size_t   blockIdx;
            bool     ok;
            std::vector<uint8_t> data;
        };
        std::map<size_t, DecompResult> resultMap;
        std::mutex                     resultMutex;
        std::condition_variable        resultCV;

        // ── writer state ───────────────────────────────────────────────
        std::atomic<bool>   writeError{false};
        std::atomic<size_t> blocksSubmitted{0};  // total blocks sent to workers
        XXH::State          xxhState(XXH32_SEED);
        std::atomic<size_t> totalBytesWritten{0};

        auto startTime = std::chrono::high_resolution_clock::now();

        // ── READER THREAD: reads all blocks from disk, feeds rawQueue ──
        std::thread readerThread([&]() {
            size_t blockIdx = 0;
            while (true) {
                uint32_t rawSz;
                ssize_t n = ::read(inputFd, &rawSz, 4);
                if (n == 0 || rawSz == 0) break;        // EOF or end-mark
                if (n != 4) { readError.store(true); break; }

                bool isUncomp = (rawSz & 0x80000000) != 0;
                uint32_t bsz  =  rawSz & 0x7FFFFFFF;
                if (bsz > 256*1024*1024) {
                    fprintf(stderr, "Implausible blockSize=%u at block %zu\n",
                            bsz, blockIdx);
                    readError.store(true); break;
                }

                RawBlock blk;
                blk.blockIdx      = blockIdx++;
                blk.isUncompressed = isUncomp;
                blk.data.resize(bsz);
                n = ::read(inputFd, blk.data.data(), bsz);
                if (n != (ssize_t)bsz) {
                    fprintf(stderr, "Short read block %zu: wanted %u got %zd\n",
                            blk.blockIdx, bsz, n);
                    readError.store(true); break;
                }

                // Back-pressure: wait if workers are falling behind
                {
                    std::unique_lock<std::mutex> lk(rawMutex);
                    rawCV.wait(lk, [&]{ return rawQueue.size() < RAW_HWM
                                               || readError.load()
                                               || writeError.load(); });
                    if (readError.load() || writeError.load()) break;

                    if (isUncomp) {
                        // Short-circuit: uncompressed blocks go straight to
                        // resultMap  no decompression needed, skip the workers.
                        DecompResult res;
                        res.blockIdx = blk.blockIdx;
                        res.ok       = true;
                        res.data     = std::move(blk.data);
                        {
                            std::lock_guard<std::mutex> rlk(resultMutex);
                            resultMap[res.blockIdx] = std::move(res);
                        }
                        resultCV.notify_one();
                    } else {
                        rawQueue.push(std::move(blk));
                    }
                    blocksSubmitted++;
                }
                rawCV.notify_all();
            }
            readerDone.store(true);
            rawCV.notify_all();  // wake workers so they can drain and exit
        });

        // ── WORKER THREADS: decompress blocks, post to resultMap ──────
        std::atomic<bool> workerStop{false};
        std::vector<std::thread> workers;
        for (size_t t = 0; t < numWorkers; t++) {
            workers.emplace_back([&]() {
                while (true) {
                    RawBlock blk;
                    {
                        std::unique_lock<std::mutex> lk(rawMutex);
                        rawCV.wait(lk, [&]{
                            return !rawQueue.empty()
                                || (readerDone.load() && rawQueue.empty())
                                || readError.load() || writeError.load();
                        });
                        if (rawQueue.empty()) break;  // done or error
                        blk = std::move(rawQueue.front());
                        rawQueue.pop();
                    }
                    rawCV.notify_all();  // wake reader (queue has space again)

                    DecompResult res;
                    res.blockIdx = blk.blockIdx;
                    res.ok       = true;

                    if (blk.isUncompressed) {
                        res.data = std::move(blk.data);
                    } else {
                        res.data.resize(chunkSize);
                        int dsz = LZ4_decompress_safe(
                            (const char*)blk.data.data(),
                            (char*)res.data.data(),
                            (int)blk.data.size(),
                            (int)chunkSize
                        );
                        if (dsz < 0) {
                            fprintf(stderr,
                                "LZ4_decompress_safe failed block %zu "
                                "(compSz=%zu ret=%d)\n",
                                blk.blockIdx, blk.data.size(), dsz);
                            res.ok = false;
                        } else {
                            res.data.resize(dsz);
                        }
                    }

                    {
                        std::lock_guard<std::mutex> lk(resultMutex);
                        resultMap[res.blockIdx] = std::move(res);
                    }
                    resultCV.notify_one();
                }
            });
        }

        // ── WRITER LOOP (main thread): drain resultMap in order ────────
        bool ok = true;
        size_t nextBlockToWrite = 0;
        while (ok && !writeError.load()) {
            std::unique_lock<std::mutex> lk(resultMutex);
            resultCV.wait_for(lk, std::chrono::milliseconds(20), [&]{
                return resultMap.count(nextBlockToWrite) > 0
                    || readError.load() || writeError.load();
            });

            // Drain all consecutive results
            while (resultMap.count(nextBlockToWrite) > 0) {
                DecompResult res = std::move(resultMap[nextBlockToWrite]);
                resultMap.erase(nextBlockToWrite);
                lk.unlock();

                if (!res.ok) { writeError.store(true); ok = false; break; }

                xxhState.update(res.data.data(), res.data.size());
                if (outputFd >= 0) {
                    ssize_t written = ::write(outputFd, res.data.data(), res.data.size());
                    if (written != (ssize_t)res.data.size()) {
                        fprintf(stderr, "Error writing decompressed data\n");
                        writeError.store(true); ok = false; break;
                    }
                }
                totalBytesWritten += res.data.size();
                nextBlockToWrite++;

                if (g_verbosity == NORMAL && estimatedBlocks > 10) {
                    size_t denom = std::max(estimatedBlocks, nextBlockToWrite);
                    std::string written = formatBytes(totalBytesWritten);
                    fprintf(stderr, "\r%s: %3zu%%  %s%s",
                            testMode ? "Testing" : "Decompressing",
                            (100 * nextBlockToWrite) / denom, written.c_str(),
                            "          ");
                    fflush(stderr);
                }
                lk.lock();
            }

            // Done when reader finished AND we've written every submitted block
            if (ok && readError.load()) { ok = false; break; }
            if (ok && readerDone.load()
                   && nextBlockToWrite >= blocksSubmitted.load()
                   && resultMap.empty()) {
                lk.unlock();
                break;
            }
        }

        // Shutdown
        workerStop.store(true);
        rawCV.notify_all();
        resultCV.notify_all();

        readerThread.join();
        for (auto& w : workers) w.join();

        close(inputFd);
        if (outputFd >= 0 && outputFd != STDOUT_FILENO) { fsync(outputFd); close(outputFd); }

        auto elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - startTime).count();

        if (ok) {
            uint32_t checksum = xxhState.digest();
            std::string outputSize = formatBytes(totalBytesWritten.load());
            
            // Overwrite progress line with completion message
            fprintf(stderr, "\r%s (CPU, %zu thread%s): %s in %.2f s%s\n",
                    testMode ? "Test complete" : "Decompression complete",
                    numWorkers, numWorkers==1?"":"s", outputSize.c_str(), elapsed,
                    "          ");  // Clear any progress debris
            
            if (testMode) {
                VLOG(NORMAL, "Test OK: %s\n", inputFile.c_str());
            }
            
            VLOG(VERBOSE, "Throughput: %.2f MB/s\n", 
                 (totalBytesWritten.load() / (1024.0*1024.0)) / elapsed);
            VLOG(VERBOSE, "  Checksum: 0x%08X\n", checksum);
        }
        return ok;
    }
    
    /*
     * Pre-process argv to support multi-digit compression level options.
     * Converts -10, -11, -12 into --hc-level N equivalents since getopt
     * only supports single-character short options.
     */
    void preprocessArgv(int& argc, char**& argv) {
        static std::vector<char*> newArgv;
        static std::vector<std::string> allocatedStrings;
        
        newArgv.clear();
        allocatedStrings.clear();
        newArgv.push_back(argv[0]);  // program name
        
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            
            // Convert -10, -11, -12 to --hc-level N
            if (arg == "-10") {
                allocatedStrings.push_back("--hc-level");
                allocatedStrings.push_back("4");
                newArgv.push_back(const_cast<char*>(allocatedStrings[allocatedStrings.size()-2].c_str()));
                newArgv.push_back(const_cast<char*>(allocatedStrings[allocatedStrings.size()-1].c_str()));
            } else if (arg == "-11") {
                allocatedStrings.push_back("--hc-level");
                allocatedStrings.push_back("8");
                newArgv.push_back(const_cast<char*>(allocatedStrings[allocatedStrings.size()-2].c_str()));
                newArgv.push_back(const_cast<char*>(allocatedStrings[allocatedStrings.size()-1].c_str()));
            } else if (arg == "-12") {
                allocatedStrings.push_back("--hc-level");
                allocatedStrings.push_back("12");
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
     * Format bytes as human-readable string (like ls -h)
     * Returns: "123 B", "45.2 KB", "3.7 MB", "1.2 GB", etc.
     */
    static std::string formatBytes(size_t bytes) {
        const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        int unitIdx = 0;
        double size = (double)bytes;
        
        while (size >= 1024.0 && unitIdx < 4) {
            size /= 1024.0;
            unitIdx++;
        }
        
        char buf[32];
        if (unitIdx == 0) {
            snprintf(buf, sizeof(buf), "%zu B", bytes);
        } else if (size >= 100.0) {
            snprintf(buf, sizeof(buf), "%.0f %s", size, units[unitIdx]);
        } else if (size >= 10.0) {
            snprintf(buf, sizeof(buf), "%.1f %s", size, units[unitIdx]);
        } else {
            snprintf(buf, sizeof(buf), "%.2f %s", size, units[unitIdx]);
        }
        return std::string(buf);
    }

    /*
     * Parse command line arguments
     */
    bool parseArguments(int argc, char* argv[]) {
        // Check if program name starts with "un" (e.g., ungzl4)
        // If so, auto-enable decompression mode (lz4-compatible behavior)
        if (argc > 0 && argv[0] != nullptr) {
            const char* progName = strrchr(argv[0], '/');
            progName = progName ? progName + 1 : argv[0];  // Get basename
            if (strlen(progName) >= 2 && progName[0] == 'u' && progName[1] == 'n') {
                decompress = true;
                VLOG(VERBOSE, "Auto-enabled decompression mode (program name: %s)\n", progName);
            }
        }
        
        const char* short_opts = "cdfhkqT:tvVzZ123456789";
        const struct option long_opts[] = {
            {"stdout", no_argument, nullptr, 'c'},
            {"to-stdout", no_argument, nullptr, 'c'},
            {"decompress", no_argument, nullptr, 'd'},
            {"uncompress", no_argument, nullptr, 'd'},
            {"force", no_argument, nullptr, 'f'},
            {"help", no_argument, nullptr, 2000},  // --help shows full help
            {"keep", no_argument, nullptr, 'k'},
            {"quiet", no_argument, nullptr, 'q'},
            {"threads", required_argument, nullptr, 'T'},
            {"test", no_argument, nullptr, 't'},
            {"verbose", no_argument, nullptr, 'v'},
            {"version", no_argument, nullptr, 2001},  // --version shows full version
            {"fast", no_argument, nullptr, '1'},
            {"best", no_argument, nullptr, '9'},
            {"cpu-only", no_argument, nullptr, 1001},
            {"gpu-only", no_argument, nullptr, 1002},
            {"hybrid", no_argument, nullptr, 1003},
            {"slot-capacity", required_argument, nullptr, 1004},
            {"batch-size", required_argument, nullptr, 1004},  // clearer alias
            {"chunks-per-batch", required_argument, nullptr, 1004},  // even clearer
            {"pipeline-depth", required_argument, nullptr, 1005},
            {"streams-per-gpu", required_argument, nullptr, 1005},  // clearer alias
            {"slots-per-gpu", required_argument, nullptr, 1005},  // alternate alias
            {"no-early-read", no_argument, nullptr, 1006},
            {"force-compress", no_argument, nullptr, 'z'},
            {"hc-level", required_argument, nullptr, 1007},
            {nullptr, 0, nullptr, 0}
        };
        
        int opt;
        int option_index = 0;
        while ((opt = getopt_long(argc, argv, short_opts, long_opts, &option_index)) != -1) {
            switch (opt) {
                case 'c':
                    stdoutMode = true;
                    keepOriginal = true;
                    break;
                case 'd':
                    decompress = true;
                    break;
                case 'f':
                    forceOverwrite = true;
                    break;
                case 'h':
                    printShortHelp();
                    return false;
                case 'k':
                    keepOriginal = true;
                    break;
                case 't':
                    testMode = true;
                    decompress = true;
                    break;
                case 'q':
                    g_verbosity = QUIET;
                    break;
                case 'v':
                    g_verbosity++;
                    break;
                case 'V':
                    printVersion();
                    return false;
                case 'T':
                    {
                        char* endptr;
                        long threads = strtol(optarg, &endptr, 10);
                        if (*endptr != '\0' || threads < 1 || threads > 1024) {
                            fprintf(stderr, "Error: Invalid thread count: %s\n", optarg);
                            return false;
                        }
                        cpuThreads = threads;
                        VLOG(DEBUG, "CPU threads set to %zu\n", cpuThreads);
                    }
                    break;
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                case '8':
                case '9':
                    compressionLevel = opt - '0';
                    VLOG(DEBUG, "Compression level %d specified\n", compressionLevel);
                    break;
                case 1001:  // --cpu-only
                    backendMode = BackendMode::CPU_ONLY;
                    VLOG(DEBUG, "Backend mode: CPU-only\n");
                    break;
                case 1002:  // --gpu-only
                    backendMode = BackendMode::GPU_ONLY;
                    VLOG(DEBUG, "Backend mode: GPU-only\n");
                    break;
                case 1003:  // --hybrid
                    backendMode = BackendMode::HYBRID;
                    VLOG(DEBUG, "Backend mode: Hybrid\n");
                    break;
                case 1004:  // --slot-capacity / --batch-size
                    {
                        char* endptr;
                        long cap = strtol(optarg, &endptr, 10);
                        if (*endptr != '\0' || cap < 1 || cap > 1024) {
                            fprintf(stderr, "Error: --slot-capacity must be 1-1024\n");
                            return false;
                        }
                        slotCapacity = cap;
                        VLOG(DEBUG, "Batch size (chunks per batch) set to %zu\n", slotCapacity);
                    }
                    break;
                case 1005:  // --pipeline-depth
                    {
                        char* endptr;
                        long depth = strtol(optarg, &endptr, 10);
                        if (*endptr != '\0' || depth < 1 || depth > 128) {
                            fprintf(stderr, "Error: --pipeline-depth must be 1-128\n");
                            return false;
                        }
                        pipelineDepth = depth;
                        VLOG(DEBUG, "Streams per GPU set to %zu\n", pipelineDepth);
                    }
                    break;
                case 1006:  // --no-early-read
                    disableEarlyRead = true;
                    VLOG(DEBUG, "Early reader disabled\n");
                    break;
                case 'z':   // -z: force compression mode (ignore .lz4 extension)
                case 'Z':
                    forceMode = true;
                    VLOG(DEBUG, "Force compression mode enabled\n");
                    break;
                case 1007:  // --hc-level N: explicitly set HC level 1-12
                    {
                        char* endptr;
                        long hlv = strtol(optarg, &endptr, 10);
                        if (*endptr != '\0' || hlv < 1 || hlv > 12) {
                            fprintf(stderr, "Error: --hc-level must be 1-12\n");
                            return false;
                        }
                        hcLevel = (int)hlv;
                    }
                    break;
                case 2000:  // --help: show full help
                    printHelp();
                    return false;
                case 2001:  // --version: show full version
                    printVersion();
                    return false;
                default:
                    fprintf(stderr, "Try 'gzl4 --help' for more information.\n");
                    return false;
            }
        }
        
        // Get input file  "-" or missing with piped stdin = read from stdin
        if (optind < argc) {
            inputFile = argv[optind];
        } else if (!isatty(STDIN_FILENO)) {
            // stdin is a pipe/redirect  use it as input
            inputFile  = "-";
            stdoutMode = true;   // piped input implies piped output
            keepOriginal = true; // nothing to delete
        } else {
            fprintf(stderr, "Error: No input file specified\n");
            fprintf(stderr, "Try 'gzl4 --help' for more information.\n");
            return false;
        }

        // If stdin, skip all file-existence checks
        if (inputFile == "-") {
            if (outputFile.empty()) outputFile = "-";
            return true;
        }

        // Verify input file exists
        struct stat st;
        if (stat(inputFile.c_str(), &st) != 0) {
            fprintf(stderr, "Error: Cannot access input file: %s\n", inputFile.c_str());
            return false;
        }
        
        if (!S_ISREG(st.st_mode)) {
            fprintf(stderr, "Error: Input is not a regular file: %s\n", inputFile.c_str());
            return false;
        }
        
        // Auto-detect decompression mode based on .lz4 extension (lz4-compatible behavior)
        // User can override with -d (decompress) or -z (force compress)
        bool hasLz4Extension = (inputFile.size() > 4 && 
                                 inputFile.substr(inputFile.size() - 4) == ".lz4");
        
        if (!decompress && !forceMode && hasLz4Extension) {
            // Auto-switch to decompression mode
            decompress = true;
            VLOG(VERBOSE, "Auto-detected decompression mode (input has .lz4 extension)\n");
        } else if (!decompress && forceMode && hasLz4Extension) {
            // User wants to compress a .lz4 file - warn but allow it
            fprintf(stderr, "Warning: Compressing .lz4 file (use -z to override auto-detection)\n");
            fprintf(stderr, "         Output will be: %s.lz4\n", inputFile.c_str());
        } else if (!decompress && !forceMode && !hasLz4Extension) {
            // Normal compression of non-.lz4 file
            VLOG(DEBUG, "Compression mode\n");
        }
        
        // Determine output file name
        if (stdoutMode) {
            outputFile = "-";
        } else if (decompress) {
            if (inputFile.size() > 4 && inputFile.substr(inputFile.size() - 4) == ".lz4") {
                outputFile = inputFile.substr(0, inputFile.size() - 4);
            } else {
                fprintf(stderr, "Error: Input file doesn't have .lz4 extension\n");
                return false;
            }
        } else {
            outputFile = inputFile + ".lz4";
        }
        
        // -t (test) mode: never writes output, so skip all output file checks
        if (!testMode && !forceOverwrite && !stdoutMode && stat(outputFile.c_str(), &st) == 0) {
            fprintf(stderr, "Error: Output file already exists: %s\n", outputFile.c_str());
            fprintf(stderr, "Use -f to force overwrite\n");
            return false;
        }
        
        // When using -f to overwrite, use a .tmp file for safety
        // On successful completion, we'll rename .tmp to the final name
        if (forceOverwrite && !stdoutMode && !testMode) {
            tempOutputFile = outputFile + ".tmp";
            VLOG(NORMAL, "Using temporary file for safe overwrite: %s\n", tempOutputFile.c_str());
        }
        
        return true;
    }
    
    /*
     * Print short help message (-h)
     */
    void printShortHelp() {
        std::cout << R"(gzl4 )" << VERSION << R"( - Multi-Backend LZ4 Compression Tool

Usage: gzl4 [OPTION]... [FILE]

Common options:
  -c            write to stdout
  -d            decompress
  -f            force overwrite
  -h            show this help (use --help for full details)
  -k            keep original files
  -q            quiet mode
  -t            test integrity
  -v            verbose (-vv, -vvv for more)
  -z            force compression (even if .lz4)
  -1 to -9      compression level (default: -9)
  -10 to -12    LZ4 High Compression (CPUs only)
  -V            show version (use --version for full details)

Program name behavior:
  ungzl4        Auto-enables decompression mode (-d implied)
                Use -z to force compression despite "un" prefix

Examples:
  gzl4 file.tar              # compress to file.tar.lz4
  gzl4 -d file.tar.lz4       # decompress to file.tar
  gzl4 file.tar.lz4          # auto-detects decompression
  gzl4 -z file.tar.lz4       # compress again to file.tar.lz4.lz4
  cat file | gzl4 > out.lz4  # pipe mode
  ungzl4 file.tar.lz4        # decompress (same as gzl4 -d)

For complete documentation, use --help
)";
    }
    
    /*
     * Print full help message (--help)
     */
    void printHelp() {
        std::cout << "gzl4 " << VERSION << 
            R"( - Multi-Backend (GPU, CPU, and Hybrid) LZ4 Compression Tool

Usage: gzl4 [OPTION]... [FILE]

Options:
  -c, --stdout         write to standard output, keep original files
  -d, --decompress     decompress (default is to compress)
  -f, --force          force overwrite of output file
  -h                   display short help (-h)
      --help           display this complete help and exit
  -k, --keep           keep (don't delete) input files
  -q, --quiet          quiet mode: only errors are output (verbosity level 0)
  -t, --test           test compressed file integrity
  -z, --force-compress force compression mode (ignore .lz4 auto-detection)
                       Note: gzl4 auto-detects decompression for .lz4 files
                       Use -z to compress .lz4 files (creates .lz4.lz4)
  -T N, --threads N    CPU thread count (default: auto-detect all cores)
		       only relevant with CPU and Hybrid backend selection.
  -v                   verbose output: -v (level 2), -vv (level 3),
                                       -vvv (level 4/debug)
                       Default is level 1 (progress + completion messages)

Compression levels:
  -1 .. -9             LZ4 fast compression (default: -9 = 4MB chunks)
                         -1: 256 KB chunks  (fastest)
                         -5: 2 MB chunks    (default chunk size)
                         -9: 4 MB chunks    (best ratio, LZ4 frame limit)
      --fast           alias for -1
      --best           alias for -9
  
  -10, -11, -12        LZ4 HC (high compression) - slower, better ratio
		       Supported only on CPU. If run in --hybrid mode,
		       chunks handled by GPU use LZ4 fast compression
                         -10: HC level 4   (moderate)
                         -11: HC level 8   (strong)
                         -12: HC level 12  (maximum)
      --hc-level N     Explicit HC level 1-12 (e.g., --hc-level 6)

  Note: GPU backend always uses LZ4 fast compression (nvCOMP limitation).
        HC levels (-10 to -12 or --hc-level) use CPU workers only.
  
  -V, --version        display version information and exit

Backend Selection:
      --cpu-only       multi-threaded CPU (all cores, LZ4_compress_default)
      --gpu-only       GPU-only via nvCOMP batched LZ4
      --hybrid         GPU + CPU with dynamic load balancing (DEFAULT)

GPU Tuning (for --gpu-only and --hybrid modes):
      --batch-size N            Chunks per batch (default: auto,
                                range: 1-1024)
                                Auto-tuned: 1 GPU=64, 2-4 GPUs=16,
                                            5+ GPUs=4
                                (Smaller batches avoid PCIe flooding)
      --chunks-per-batch N      (alias for --batch-size)
      --slot-capacity N         (legacy name for --batch-size)
                                Larger = fewer batches, less overhead,
                                         more sequential gaps
                                Smaller = more batches, more overhead,
                                          fewer sequential gaps
      --streams-per-gpu N       Concurrent streams per GPU (default: auto,
                                range: 1-128)
                                Compression auto: 1 GPU=4, 2-4 GPUs=3, 5+=3
                                Decompression auto: 1 GPU=32, 2-4=16, 5+=8
      --slots-per-gpu N         (alias for --streams-per-gpu)
      --pipeline-depth N        (legacy name for --streams-per-gpu)
                                Higher = more GPU utilization,
                                         more disorder for writer
                                Lower = less GPU utilization,
                                        less disorder for writer
                                Enables overlap with non-blocking workers
      --no-early-read           Disable file read-ahead during GPU init
                                (Reduces memory, may hurt throughput)

Compression backends:
  cpu-only  Multi-threaded LZ4_compress_default/HC; async I/O pipeline;
            posix_fadvise readahead; LZ4 HC levels 1-12 (-10/-11/-12)
  gpu-only  nvCOMP batched LZ4 fast; per-GPU workers with slot rotation;
            pinned memory pool for zero-copy DMA; configurable batch/streams;
            async reader/writer overlap I/O with GPU (600+ MB/s)
  hybrid    GPU-priority scheduler: feeds GPU queue first, CPU when GPUs
            saturated or not yet initialized; GPUs handle 70-90% of chunks, 
	    CPUs fill gaps; workers submit directly to thread-safe AsyncWriter

Decompression:
  Hybrid GPU + CPU: nvCOMP batched API tries each block; CUDA Error 12
  triggers automatic CPU fallback via LZ4_decompress_safe. Handles
  mixed-format files (GPU + CPU compressed blocks). Reports GPU/CPU
  block counts. Uncompressed blocks bypass workers (zero-copy fast).
  Multi-threaded with parallel worker pool and sequential writer.

Performance notes:
  - All modes use async reader + async writer to overlap I/O + compute
  - POSIX_FADV_SEQUENTIAL + WILLNEED hints on all input file descriptors
  - Decompression: uncompressed blocks bypass workers (zero-copy)

Architecture:
  3-stage pipeline:  AsyncReader -> workers (GPU/CPU) -> AsyncWriter
  - AsyncReader:     Pre-reads file during GPU init (overlaps 3s CUDA setup)
  - PinnedInputPool: Zero-copy DMA to GPU (7GB pool, 64+ slots)
  - GPU workers:     Per-GPU thread with slot rotation (pipeline depth set)
  - CPU workers:     Thread pool pulls from work queue (default: all cores)
  - AsyncWriter:     Thread-safe out-of-order with sequential reordering
  
  posix_fadvise(SEQUENTIAL|WILLNEED) on all input file descriptors
  GPU pipeline auto-tuned based on GPU count:
  - Streams: 1 GPU=4, 2 GPUs=3, 3-8 GPUs=3 (non-blocking overlap)
  - Batch size: 1 GPU=64 chunks, 2-4 GPUs=16, 5+ GPUs=4 (avoid PCIe flood)
  Empirical: RTX 5090 peaks at 1208 MB/s (batch=68, streams=4)
             8× H100s best with batch=3, streams=3 (avoid bus saturation)
	     1x H100 best with batch=9, streams=5
  Standard LZ4 frame format; fully compatible with lz4 command-line tool

Changelog:
  v3.23.0  Perf #1: pinned host output staging (h_decomp) for both
           decompressFileGPU and decompressFileHybrid. Each slot now allocates
           SC*chunkSize bytes of pinned host memory as a mirror of d_decomp.
           dispatchSlot queues an async D→H of the full batch output on the same
           stream immediately after nvCOMP, so the transfer overlaps with the
           next batch being filled. collectSlot replaces the synchronous
           cudaMemcpy with a plain memcpy from pinned RAM  eliminating the
           CUDA driver's internal bounce-buffer and the blocking D→H stall.
           Both compression paths (compressFileGPU, compressFileHybrid) already
           had this optimization via PreallocSlot.h_output; decompression now
           matches.
  v3.22.0  Hybrid decompression fixes: VRAM-aware stream auto-sizing (50% of
           free VRAM cap prevents OOM hang); goto-free GPU worker with per-slot
           CUDA error logging and graceful re-routing to CPU queue on alloc fail;
           blocksDone-based termination (fixes pass-through count race);
           DEBUG heartbeat in main drain loop; cpuWorker takes tidx arg
  v3.21.0  True hybrid decompression: dispatcher thread routes blocks GPU-first
           via TsQueue; CPU overflow workers (effectiveThreads) run
           LZ4_decompress_safe in parallel  cpuBlocks now reflects real CPU
           work, not just nvCOMP error fallbacks; progress shows true GPU/CPU split
  v3.19.9  CONSISTENT TEST MODE (-t) OUTPUT: all three decompression backends
           now show "Decompressing (test, <backend>): <infile>" with no arrow to
           the output file; "Writing:" drain phase suppressed in test mode;
           garbled residual characters from line-length mismatch fixed; CPU
           opening line now includes thread count matching completion format
  v3.19.8  STDOUT PIPELINE FIXES: all three compressors now write LZ4 header,
           blocks, and footer to stdout in pipe mode; AsyncWriter handles stdout
           correctly (no open("-"), no fsync(stdout), no close(stdout));
           decompressFileCPU guards fsync/close against STDOUT_FILENO; all three
           compressors guard unlink(inputFile) with !stdoutMode; writeBuf
           allocation restored to AsyncWriter::start() (regression from v3.19.5)
  v3.19.7  DECOMPRESSOR WRITE PROGRESS: GPU-only and hybrid now track bytes
           written (output side) rather than blocks read; final drain shows
           "Writing: N%  [X / Y]" for GPU-only and hybrid; hybrid live progress
           shows live GPU/CPU byte split; hybrid completion shows GPU/CPU block
           counts with percentages at normal verbosity (matching compression)
  v3.19.6  DECOMPRESSOR CHUNK SIZE FROM FRAME HEADER: suppressed misleading
           "Compression level N: LZ4 fast" log during decompression (chunk size
           comes from the LZ4 frame header, not the level flag); all three
           decompressors print "N MB | block size N KB | ~N blocks" at -v from
           the parsed frame header; GPU-only decompressor now uses the same
           single-line format as hybrid and CPU
  v3.19.5  BATCHED GPU DECOMPRESSION + --hybrid STREAMS FIX: --batch-size now
           controls nvCOMP batch size during decompression for both --gpu-only
           and --hybrid; each stream slot holds SC blocks per nvCOMP call with
           strided contiguous device buffers; Error 12 / nvcompErrorInvalidValue
           (raw LZ4 format) named in verbose log with per-block CPU fallback;
           --hybrid ported to full N_STREAMS x SC slot-ring (was single-stream,
           silently ignoring --streams-per-gpu); -f rename now silent at normal
           verbosity, shown as "Renaming: src -> dst" at -v and above
  v3.19.4  FAST --gpu-only DECOMPRESSION + --streams-per-gpu TUNING: decompressor
           uses N_STREAMS independent streams (default 32 single GPU, 16 for 2-4
           GPUs, 8 for 5+), each with batch_size=1 per nvCOMP call to avoid
           Error 12 when files contain mixed nvCOMP + raw LZ4 blocks (hybrid);
           --streams-per-gpu N now controls decompression parallelism (previously
           only affected compression); each stream has pre-allocated persistent
           device buffers (~12 MB VRAM/stream at 4 MB chunk); async H→D and
           metadata D→H overlap across streams; CPU fallback only on genuine
           GPU errors; hybrid mode unchanged: CPU-first, GPU for nvCOMP blocks;
           fixed asyncWriter.start() outputFile vs getActualOutputPath() (-f bug)
  v3.19.3  (intermediate  superseded by 3.19.4)
  v3.19.2  CRITICAL BUG FIXES: Fixed GPU/Hybrid decompression stalling in final flush by adding
           missing mutex locks around results.find(); fixed -f temp file error messages to show
           actual path being written (temp file); all file writes now correctly use temp file
           when -f is active; updated output format for better readability
  v3.19.1  SIMPLIFIED VERSION OUTPUT: -V and --version now show same concise output; moved
           detailed backend/architecture info from version to --help for better organization;
           fixed compiler warning (unused parameter); removed outdated decompression note;
           fixed changelog formatting (removed extra newline, corrected indentation for
           v3.4.0-v3.0.0); all --help lines now ≤80 characters for better terminal display
  v3.19.0  SAFE FILE REPLACEMENT & UNGZL4 SUPPORT: -f now uses .tmp files with atomic rename on
           success, protecting original from corruption if interrupted; SIGINT/SIGTERM cleanup temp
           files; program name detection: "ungzl4" auto-enables decompression (-z overrides); two-
           tier help: -h shows brief syntax, --help shows full documentation; -V shows version,
           --version shows full details; all temp file operations reported at normal verbosity
  v3.18.0  LZ4-COMPATIBLE -z FLAG: Fixed -z to work like standard lz4 tool; -z now forces
           compression mode (ignores .lz4 extension auto-detection), allowing .lz4.lz4 files;
           removed incorrect "force compressed output even if larger" behavior; always stores
           smaller version (compressed or original); auto-detects decompression when input ends
           with .lz4; warns when compressing .lz4 files without -z; matches lz4 CLI behavior
  v3.17.1  UNIFIED VERBOSITY SYSTEM: Refactored verbosity into single 5-level system (0-4);
           removed separate g_quiet flag and QLOG macro; -q sets level 0 (errors only), default
           is level 1 (normal progress), -v/-vv/-vvv increment to 2/3/4; simpler, more standard
           architecture; all output uses single VLOG macro; easier to maintain and extend
  v3.17.0  QUIET MODE & PIPE CONVENIENCE: Added -q/--quiet flag to suppress all non-error
           output (useful in scripts/pipes); proper EXIT_SUCCESS/EXIT_FAILURE return codes
           on all paths; no-argument invocation automatically assumes pipe mode (stdin to
           stdout) if stdout is not a tty, otherwise shows help; QLOG macro for conditional
           output based on quiet flag
  v3.16.4  REFINED TEST MODE OUTPUT: Removed "Testing: <filename>" start line; progress
           and completion messages now show "Decompressing (test):" when in test mode;
           cleaner, more consistent output across all decompression modes
  v3.16.3  UNIFIED DECOMPRESSION OUTPUT: All decompression modes now show consistent output;
           progress displays human-readable bytes instead of blocks; completion message
           overwrites progress line using \r; test mode shows "Test OK: filename" on new
           line after completion; removed "-> /dev/null" clutter from test mode output
  v3.16.2  INTELLIGENT AUTO-TUNING: Batch size and stream count now auto-tune based on
           GPU count; single GPU uses batch=64, streams=4 (RTX 5090: 1208 MB/s); 5+ GPUs
           use batch=4, streams=3 (prevents PCIe flooding on multi-GPU); empirical tuning
           based on RTX 5090 and 8× H100 testing; user can still override with flags
  v3.16.1  EXPANDED TUNING RANGES: --batch-size cap raised from 128 to 1024 chunks;
           --streams-per-gpu cap raised from 16 to 128 streams; enables future-proofing
           and fine-tuning for high-memory GPUs and evolving workloads
  v3.16.0  NON-BLOCKING GPU WORKERS: Replaced cudaStreamSynchronize with cudaStreamQuery
           polling; GPU workers never block waiting for streams to complete; continuously
           launch batches on free slots while polling others; provides steady flow of
           results to writer instead of bursty stop-and-go pattern; eliminates writer
           starvation; achieves 1195 MB/s with batch-size 48 + streams 10 on RTX 5090
  v3.15.2  CRITICAL PERFORMANCE FIX: Auto-tune pipeline depth based on GPU count;
           single GPU now uses 6 slots (was 1) to hide sync latency, restoring
           performance from 118 MB/s back to ~800 MB/s; multi-GPU uses 1 slot
  v3.15.1  Unified progress output: writing progress added to CPU-only/Hybrid modes;
           all "Compression complete" and "Decompression complete" messages now use
           human-readable bytes (matching ls -h format)
  v3.15.0  Progress output consistency: "Compressing: N% GPU: X CPU: Y" format across
           all modes; "Writing: N% [X/Y to disk]" for writer phase; decompression
           shows bytes processed; formatBytes() helper for human-readable sizes
  v3.14.0  Hybrid mode v3: GPU-priority dispatcher; separate GPU/CPU work queues;
           GPUs get chunks first, CPUs only when GPUs saturated; all workers
           submit directly to thread-safe AsyncWriter (no central coordinator)
  v3.13.x  HC compression levels: -10/-11/-12 support via argv preprocessor;
           --hc-level 1-12 for explicit control; fixed hcLevel reset bug;
           removed duplicate help text; cleaned up output formatting
  v3.12.x  Hybrid rewrite: unified worker model (same slot machinery as gpu-only);
           removed old scheduler-based architecture; direct writer submission
  v3.11.x  GPU decompression via nvCOMP batched API (batch_size=1 for single blocks);
           hybrid decompression with automatic CPU fallback on Error 12;
           pipe support (stdin/stdout via "-"); output message standardization;
           actualSize parameter fix; checksum seek position fix
  v3.10.x  Command-line tuning params: --batch-size (chunks/batch, default 8);
           --streams-per-gpu (pipeline depth, optimal=1); performance testing
           optimal config: cap=8, depth=1, 601 MB/s on 8GB test file
  v3.9.x   GPU worker threads with per-slot rotation; parallel slot initialization;
           early reader startup (overlap with GPU init); writer greedy drain;
           PinnedInputPool for zero-copy DMA; double-read deadlock fix
  v3.8.0   Unified output across all modes; fadvise on all code paths;
           decompression reader thread + parallel worker pool;
           uncompressed-block zero-copy fast path in decompressor;
           progress lines padded to prevent collision with error text
  v3.7.x   Hybrid stall fix (per-chunk GPU batch entries);
           CPU decompressor replaces GPU decompressor (handles mixed-format
           blocks from hybrid files); LZ4 header seek fix
  v3.7.0   CPU-only mode: thread pool, async I/O, configurable threads;
           hybrid mode: CPU+GPU simultaneous compression
  v3.6.0   Parallel decompression; direct I/O syscalls; multi-GPU batches
  v3.5.0   Dynamic stream scaling (128->1024/GPU); test mode (-t)
  v3.4.0   Out-of-order async writer
  v3.3.0   Async reader with posix_fadvise readahead
  v3.2.0   Async writer thread
  v3.0.0   True parallel multi-GPU processing
  
File format:
  Standard LZ4 frame format (.lz4 extension).
  Output is compatible with the lz4 command-line tool:
    lz4 -d file.tar.lz4
    unlz4 file.tar.lz4

Examples:
  gzl4 archive.tar              compress (hybrid, all GPUs + CPUs)
  gzl4 -d archive.tar.lz4       decompress
  gzl4 -t archive.tar.lz4       verify integrity without writing output
  gzl4 --cpu-only -T 32 f.dat   CPU-only with 32 threads
  gzl4 --gpu-only large.bin     GPU-only compression
  gzl4 --gpu-only --batch-size 32 data.tar        larger batches (32 chunks, compress)
  gzl4 --gpu-only --batch-size 4 data.tar         smaller batches (4 chunks, compress)
  gzl4 --gpu-only --streams-per-gpu 2 data.tar    2 concurrent streams/GPU (compress)
  gzl4 -d --gpu-only --batch-size 8 file.lz4      decompress with 8-block nvCOMP batches
  gzl4 -d --gpu-only --streams-per-gpu 64 f.lz4   64 concurrent decompression streams
  gzl4 -vv -k archive.tar       verbose, keep original

Pipe examples:
  cat file.tar | gzl4 -c > file.tar.lz4   compress via pipe (stdout)
  gzl4 -c file.tar | ssh host "cat > out.lz4"  compress to remote
  gzl4 -dc file.tar.lz4 | tar -x          decompress to stdout, pipe to tar
  gzl4 -c - < file.tar > file.tar.lz4     explicit stdin with "-"
)" << std::endl;
    }

    /*
     * Print version information (-V and --version)
     */
    void printVersion() {
        std::cout << "gzl4 " << VERSION << 
            R"( - GPU, CPU, and Hybrid LZ4 Compression Tool
Built with nvCOMP 5.1.x, CUDA 12.8, liblz4
)" << std::endl;
    }

    /*
     * Main processing entry point
     */
    bool run(int argc, char* argv[]) {
        // Setup signal handlers for cleanup
        setupSignalHandlers();
        
        // If no arguments provided, assume pipe mode (stdin -> stdout)
        // but only if stdout is not a terminal
        if (argc == 1) {
            if (isatty(STDOUT_FILENO)) {
                // stdout is a terminal - show help instead
                printHelp();
                return false;
            }
            // Assume pipe mode: compress stdin to stdout
            inputFile = "-";
            outputFile = "-";
            stdoutMode = true;
            keepOriginal = true;
        }
        
        // Pre-process argv to convert -10, -11, -12 into --hc-level N
        preprocessArgv(argc, argv);
        
        // Parse command line
        if (!parseArguments(argc, argv)) {
            return false;
        }

        // Chunk size is derived from the compression level  only meaningful when
        // compressing.  During decompression the real chunk size is read from the
        // LZ4 frame header; we still call this to set a sane default for the early
        // reader / pool sizing before the file is opened.
        setChunkSizeFromLevel();

        // ── Start reading as early as possible ────────────────────────────────
        // GPU context creation takes 1-4 s per machine. Start reading during
        // that window so the disk is never idle while we wait for CUDA.
        if (!decompress && backendMode != BackendMode::CPU_ONLY && !disableEarlyRead) {
            // Stat the file to know how large a queue to pre-fill
            struct stat st;
            if (stat(inputFile.c_str(), &st) == 0 && st.st_size > 0) {
                size_t fSize     = (size_t)st.st_size;
                size_t nChunks   = (fSize + chunkSize - 1) / chunkSize;
                // Queue the entire file  no artificial cap. If malloc fails,
                // we get a clean error. On modern servers (64GB+ RAM), capping
                // an 8GB file at 2009 chunks just creates 5s of idle waiting.
                // The reader will malloc chunks until OOM or EOF, whichever comes first.
                VLOG(VERBOSE, "Early reader: starting %.2f GB file read-ahead "
                     "(%zu chunks, unlimited queue) while GPUs initialise\n",
                     fSize / (1024.0*1024.0*1024.0), nChunks);
                earlyReader.start(inputFile, chunkSize, SIZE_MAX);
            }
        }

        // ── Initialize GPUs in parallel with early I/O ────────────────────────
        if (backendMode != BackendMode::CPU_ONLY) {
            if (!initializeGPUs()) {
                fprintf(stderr, "Warning: GPU initialization failed, falling back to CPU-only mode\n");
                backendMode = BackendMode::CPU_ONLY;
            }
        } else {
            VLOG(VERBOSE, "CPU-only mode: skipping GPU initialization\n");
        }

        // Calculate batch size for first GPU (only if using GPUs, compression only 
        // decompression batch size is slotCapacity, set directly from --batch-size)
        if (!decompress && backendMode != BackendMode::CPU_ONLY && !gpus.empty()) {
            batchSize = calculateBatchSize(gpus[0].availableMemory);
            VLOG(VERBOSE, "Initial batch estimate: %zu chunks/slot (%.1f GB/slot at 5x chunk overhead)\n",
                 batchSize, (batchSize * chunkSize * 5) / (1024.0*1024.0*1024.0));
        }

        // Perform operation
        bool success;
        if (decompress) {
            success = decompressFile();
        } else {
            success = compressFile();
        }

        // Synchronize all GPUs
        for (size_t i = 0; i < gpus.size(); i++) {
            cudaSetDevice(gpus[i].deviceId);
            cudaDeviceSynchronize();
        }

        // On success, rename temp file to final name (if using temp file)
        if (success && !tempOutputFile.empty()) {
            success = renameTempToFinal();
        }
        
        // On failure, cleanup temp file
        if (!success) {
            cleanupTempFile();
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
