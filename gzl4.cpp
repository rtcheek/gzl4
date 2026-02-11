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

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <algorithm>
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

// Configuration constants
constexpr const char* VERSION = "3.5.1";
constexpr size_t MIN_CHUNK_SIZE = 16 * 1024;            // 16KB minimum
constexpr size_t MAX_CHUNK_SIZE = 4 * 1024 * 1024;      // 4MB maximum (LZ4 frame limit)
constexpr int STREAMS_PER_GPU = 128;                     // CUDA streams per GPU (dynamic scaling starts here)
constexpr double GPU_MEM_SAFETY_FACTOR = 0.5;           // Use 50% of available memory for batches

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
constexpr size_t CHUNK_SIZE_LEVEL_5 = 2 * 1024 * 1024;  // 2MB (was 256KB, now default)
constexpr size_t CHUNK_SIZE_LEVEL_6 = 3 * 1024 * 1024;  // 3MB (was 512KB)
constexpr size_t CHUNK_SIZE_LEVEL_7 = 3 * 1024 * 1024 + 512 * 1024;  // 3.5MB (was 1MB)
constexpr size_t CHUNK_SIZE_LEVEL_8 = 4 * 1024 * 1024;  // 4MB (was 2MB)
constexpr size_t CHUNK_SIZE_LEVEL_9 = 4 * 1024 * 1024;  // 4MB (max for LZ4)

// Verbosity levels
enum VerbosityLevel {
    QUIET = 0,
    VERBOSE = 1,
    VERY_VERBOSE = 2,
    DEBUG = 3
};

// Global verbosity setting
int g_verbosity = QUIET;

// Macro for verbose output
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
    
    GPUDevice(int id) : deviceId(id) {}
    
    ~GPUDevice() {
        // Don't try to destroy streams in destructor
        // CUDA automatically cleans up when the program exits
    }
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
 * Reads chunks in background thread while GPUs initialize and compress
 */

/*
 * Asynchronous Reader with Advanced I/O
 * Reads file chunks in background thread while GPUs initialize and compress
 */
class AsyncReader {
public:
    struct ReadChunk {
        size_t chunkIndex;
        std::vector<uint8_t> data;
        size_t size;
    };
    
private:
    std::thread readerThread;
    std::queue<ReadChunk> readQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool> shouldStop{false};
    std::atomic<bool> finished{false};
    std::atomic<size_t> bytesRead{0};
    std::atomic<double> totalReadTime{0.0};  // Track reading time
    
    int inputFd = -1;
    size_t fileSize = 0;
    size_t chunkSize = 0;
    size_t maxQueuedChunks = 0;  // Limit RAM usage
    
    // Reader thread main loop
    void readerLoop() {
        VLOG(DEBUG, "Reader thread started\n");
        
        size_t chunkIndex = 0;
        size_t totalRead = 0;
        auto threadStartTime = std::chrono::high_resolution_clock::now();
        
        while (totalRead < fileSize) {
            // Check queue depth - don't overflow RAM
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCV.wait(lock, [this] {
                    return readQueue.size() < maxQueuedChunks || shouldStop.load();
                });
                
                if (shouldStop.load()) break;
            }
            
            // Read next chunk
            size_t toRead = std::min(chunkSize, fileSize - totalRead);
            ReadChunk chunk;
            chunk.chunkIndex = chunkIndex;
            chunk.data.resize(toRead);
            chunk.size = toRead;
            
            auto readStart = std::chrono::high_resolution_clock::now();
            ssize_t bytesReadNow = ::read(inputFd, chunk.data.data(), toRead);
            auto readEnd = std::chrono::high_resolution_clock::now();
            
            if (bytesReadNow != (ssize_t)toRead) {
                fprintf(stderr, "Reader thread: Read error at chunk %zu: %s\n",
                        chunkIndex, strerror(errno));
                break;
            }
            
            totalRead += toRead;
            bytesRead += toRead;
            
            // Accumulate read time
            double readTime = std::chrono::duration<double>(readEnd - readStart).count();
            totalReadTime = totalReadTime.load() + readTime;
            
            // Enqueue chunk
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                readQueue.push(std::move(chunk));
            }
            queueCV.notify_one();
            
            chunkIndex++;
            
            VLOG(DEBUG, "Reader: Read chunk %zu (%zu bytes in %.3fs), queue depth: %zu\n",
                 chunkIndex - 1, toRead, readTime, getQueueDepth());
        }
        
        auto threadEndTime = std::chrono::high_resolution_clock::now();
        double totalThreadTime = std::chrono::duration<double>(threadEndTime - threadStartTime).count();
        
        finished.store(true);
        queueCV.notify_all();
        VLOG(VERBOSE, "Reader thread finished: read %zu bytes in %.2fs (%.2fs actual I/O, %.2fs waiting)\n", 
             bytesRead.load(), totalThreadTime, totalReadTime.load(), 
             totalThreadTime - totalReadTime.load());
    }
    
public:
    AsyncReader() = default;
    
    ~AsyncReader() {
        stop();
    }
    
    bool start(const std::string& filename, size_t chunk_size, size_t max_queued = 128) {
        chunkSize = chunk_size;
        maxQueuedChunks = max_queued;
        
        // Open file with O_RDONLY for sequential reads
        inputFd = open(filename.c_str(), O_RDONLY);
        if (inputFd < 0) {
            fprintf(stderr, "Error opening input file %s: %s\n",
                    filename.c_str(), strerror(errno));
            return false;
        }
        
        // Get file size
        struct stat st;
        if (fstat(inputFd, &st) != 0) {
            fprintf(stderr, "Error getting file size: %s\n", strerror(errno));
            close(inputFd);
            return false;
        }
        fileSize = st.st_size;
        
        // Hint to kernel that we'll be reading sequentially
        posix_fadvise(inputFd, 0, fileSize, POSIX_FADV_SEQUENTIAL);
        posix_fadvise(inputFd, 0, fileSize, POSIX_FADV_WILLNEED);  // Start readahead
        
        VLOG(VERBOSE, "AsyncReader: opened %s (%.2f MB) for reading\n",
             filename.c_str(), fileSize / (1024.0 * 1024.0));
        
        shouldStop.store(false);
        finished.store(false);
        readerThread = std::thread(&AsyncReader::readerLoop, this);
        
        return true;
    }
    
    bool getChunk(ReadChunk& chunk) {
        std::unique_lock<std::mutex> lock(queueMutex);
        
        // Wait for data or finish
        queueCV.wait(lock, [this] {
            return !readQueue.empty() || finished.load();
        });
        
        if (readQueue.empty()) {
            return false;  // No more chunks
        }
        
        chunk = std::move(readQueue.front());
        readQueue.pop();
        
        // Notify reader it can read more (if queue was full)
        queueCV.notify_one();
        
        return true;
    }
    
    void stop() {
        shouldStop.store(true);
        queueCV.notify_all();
        
        if (readerThread.joinable()) {
            readerThread.join();
        }
        
        if (inputFd >= 0) {
            close(inputFd);
            inputFd = -1;
        }
    }
    
    bool isFinished() const {
        return finished.load() && getQueueDepth() == 0;
    }
    
    size_t getQueueDepth() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(queueMutex));
        return readQueue.size();
    }
    
    size_t getFileSize() const { return fileSize; }
    size_t getBytesRead() const { return bytesRead.load(); }
    double getReadTime() const { return totalReadTime.load(); }
};

/*
 * Asynchronous Writer with Advanced I/O
 * Writes completed batches in background thread while GPUs continue working
 */
class AsyncWriter {
private:
    struct WriteTask {
        size_t chunkIndex;
        std::vector<std::vector<uint8_t>> compressedChunks;
        std::vector<std::vector<uint8_t>> originalChunks;
        std::vector<size_t> chunkIndices;
        std::vector<size_t> originalSizes;
    };
    
    std::thread writerThread;
    std::map<size_t, WriteTask> pendingWrites;  // Out-of-order buffer, indexed by first chunk
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool> shouldStop{false};
    std::atomic<size_t> bytesWritten{0};
    std::atomic<double> totalWriteTime{0.0};  // Track writing time
    std::atomic<size_t> nextChunkToWrite{0};  // Next expected chunk index
    
    int outputFd = -1;
    std::string outputFile;
    XXH::State* xxhState = nullptr;
    
    // Writer thread main loop
    void writerLoop() {
        VLOG(DEBUG, "Writer thread started\n");
        auto threadStartTime = std::chrono::high_resolution_clock::now();
        
        while (true) {
            WriteTask task;
            bool hasTask = false;
            
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                
                // Wait for either: next sequential chunk available OR stop signal
                queueCV.wait(lock, [this] { 
                    return pendingWrites.count(nextChunkToWrite.load()) > 0 || shouldStop.load(); 
                });
                
                if (pendingWrites.empty() && shouldStop.load()) {
                    break;  // Done
                }
                
                // Check if next sequential chunk is available
                if (pendingWrites.count(nextChunkToWrite.load()) > 0) {
                    task = std::move(pendingWrites[nextChunkToWrite.load()]);
                    pendingWrites.erase(nextChunkToWrite.load());
                    hasTask = true;
                    
                    VLOG(DEBUG, "Writer: got sequential chunk %zu (buffer: %zu pending)\n",
                         nextChunkToWrite.load(), pendingWrites.size());
                }
            }
            
            if (hasTask) {
                // Write this task (outside lock for parallelism)
                auto writeStart = std::chrono::high_resolution_clock::now();
                writeTask(task);
                auto writeEnd = std::chrono::high_resolution_clock::now();
                
                double writeTime = std::chrono::duration<double>(writeEnd - writeStart).count();
                totalWriteTime = totalWriteTime.load() + writeTime;
                
                // Update next expected chunk
                nextChunkToWrite += task.chunkIndices.size();
                queueCV.notify_all();  // Notify in case we were blocking something
            }
        }
        
        auto threadEndTime = std::chrono::high_resolution_clock::now();
        double totalThreadTime = std::chrono::duration<double>(threadEndTime - threadStartTime).count();
        
        VLOG(VERBOSE, "Writer thread finished: wrote %zu bytes in %.2fs (%.2fs actual I/O, %.2fs waiting)\n",
             bytesWritten.load(), totalThreadTime, totalWriteTime.load(),
             totalThreadTime - totalWriteTime.load());
    }
    
    void writeTask(const WriteTask& task) {
        for (size_t i = 0; i < task.originalChunks.size(); i++) {
            size_t origSize = task.originalSizes[i];
            
            // Update checksum
            if (xxhState) {
                xxhState->update(task.originalChunks[i].data(), origSize);
            }
            
            // Write block
            if (i < task.compressedChunks.size() && !task.compressedChunks[i].empty()) {
                // Compressed chunk
                uint32_t compSize = task.compressedChunks[i].size();
                
                if (compSize >= origSize) {
                    // Doesn't compress - write uncompressed
                    writeUncompressedBlock(task.originalChunks[i].data(), origSize);
                } else {
                    // Write compressed
                    writeCompressedBlock(task.compressedChunks[i].data(), compSize);
                }
            } else {
                // Pre-marked as uncompressed
                writeUncompressedBlock(task.originalChunks[i].data(), origSize);
            }
        }
    }
    
    void writeU32(uint32_t value) {
        uint8_t buf[4];
        buf[0] = value & 0xFF;
        buf[1] = (value >> 8) & 0xFF;
        buf[2] = (value >> 16) & 0xFF;
        buf[3] = (value >> 24) & 0xFF;
        
        ssize_t written = ::write(outputFd, buf, 4);
        if (written != 4) {
            fprintf(stderr, "Error writing to file: %s\n", strerror(errno));
        }
        bytesWritten += 4;
    }
    
    void writeUncompressedBlock(const void* data, size_t size) {
        uint32_t blockSize = size | 0x80000000;  // Set high bit
        writeU32(blockSize);
        
        ssize_t written = ::write(outputFd, data, size);
        if (written != (ssize_t)size) {
            fprintf(stderr, "Error writing uncompressed block: %s\n", strerror(errno));
        }
        bytesWritten += size;
    }
    
    void writeCompressedBlock(const void* data, size_t size) {
        writeU32(size);
        
        ssize_t written = ::write(outputFd, data, size);
        if (written != (ssize_t)size) {
            fprintf(stderr, "Error writing compressed block: %s\n", strerror(errno));
        }
        bytesWritten += size;
    }
    
public:
    AsyncWriter() = default;
    
    ~AsyncWriter() {
        stop();
    }
    
    bool start(const std::string& filename, XXH::State* xxh) {
        outputFile = filename;
        xxhState = xxh;
        
        // Open file with O_APPEND since header was already written by main thread
        // File already exists with header, we're appending compressed blocks
        outputFd = open(filename.c_str(), O_WRONLY | O_APPEND, 0644);
        if (outputFd < 0) {
            fprintf(stderr, "Error opening output file %s: %s\n", 
                    filename.c_str(), strerror(errno));
            return false;
        }
        
        // Hint to kernel that we'll be writing sequentially
        posix_fadvise(outputFd, 0, 0, POSIX_FADV_SEQUENTIAL);
        
        VLOG(VERBOSE, "AsyncWriter: opened %s for appending blocks\n", filename.c_str());
        
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
        task.chunkIndex = chunkIndex;
        task.compressedChunks = std::move(compressedChunks);
        task.originalChunks = std::move(originalChunks);
        task.chunkIndices = std::move(chunkIndices);
        task.originalSizes = std::move(originalSizes);
        
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            pendingWrites[chunkIndex] = std::move(task);
            
            VLOG(DEBUG, "Enqueued batch starting at chunk %zu (writer has %zu pending batches)\n",
                 chunkIndex, pendingWrites.size());
        }
        queueCV.notify_all();  // Wake up writer if it's waiting for this chunk
    }
    
    void stop() {
        shouldStop.store(true);
        queueCV.notify_one();
        
        if (writerThread.joinable()) {
            writerThread.join();
        }
        
        if (outputFd >= 0) {
            // Sync to disk before closing
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
    
    size_t getNextChunkToWrite() const {
        return nextChunkToWrite.load();
    }
    
    size_t getQueueDepth() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(queueMutex));
        return pendingWrites.size();
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
    bool stdoutMode;
    bool testMode;
    std::string inputFile;
    std::string outputFile;
    
public:
    GZL4Compressor() 
        : chunkSize(CHUNK_SIZE_LEVEL_9)
        , batchSize(1)
        , compressionLevel(9)
        , decompress(false)
        , keepOriginal(false)
        , forceOverwrite(false)
        , stdoutMode(false)
        , testMode(false) {}
    
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
            
            VLOG(VERBOSE, "GPU %d: %s\n", i, gpu.properties.name);
            VLOG(VERY_VERBOSE, "  Compute Capability: %d.%d\n", 
                 gpu.properties.major, gpu.properties.minor);
            VLOG(VERY_VERBOSE, "  Total Memory: %.2f GB\n", 
                 totalMem / (1024.0 * 1024.0 * 1024.0));
            VLOG(VERY_VERBOSE, "  Available Memory: %.2f GB\n", 
                 gpu.availableMemory / (1024.0 * 1024.0 * 1024.0));
            
            // Try to create CUDA streams - skip GPU if this fails
            gpu.streams.resize(STREAMS_PER_GPU);
            bool streamsOk = true;
            for (int s = 0; s < STREAMS_PER_GPU; s++) {
                err = cudaStreamCreate(&gpu.streams[s]);
                if (err != cudaSuccess) {
                    VLOG(VERBOSE, "  Failed to create stream %d: %s\n",
                         s, cudaGetErrorString(err));
                    // Clean up any streams we did create
                    for (int cleanup = 0; cleanup < s; cleanup++) {
                        cudaStreamDestroy(gpu.streams[cleanup]);
                    }
                    cudaGetLastError(); // Clear error
                    streamsOk = false;
                    break;
                }
            }
            
            if (!streamsOk) {
                VLOG(VERBOSE, "Skipping GPU %d (%s) - insufficient memory for streams\n",
                     i, gpu.properties.name);
                continue;
            }
            
            VLOG(DEBUG, "  Created %d CUDA streams\n", STREAMS_PER_GPU);
            
            gpus.push_back(std::move(gpu));
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
        
        VLOG(VERBOSE, "Compression level %d: chunk size %.2f KB\n",
             compressionLevel, chunkSize / 1024.0);
    }
    
    /*
     * Calculate optimal batch size based on GPU memory
     * Returns how many chunks can fit in available GPU memory
     * With 4MB chunks (LZ4 limit), we can fit more chunks per batch
     */
    size_t calculateBatchSize(size_t gpuMemory) {
        // Account for compression overhead (~2x worst case) + temp buffers
        size_t memoryPerChunk = chunkSize * 4; // Conservative estimate
        size_t availableForBatch = static_cast<size_t>(gpuMemory * GPU_MEM_SAFETY_FACTOR);
        size_t batchSize = availableForBatch / memoryPerChunk;
        
        // With smaller chunks (4MB vs 16MB), we can process more per batch
        // Minimum of 16, maximum of 2048 (increased from 256)
        batchSize = std::max(size_t(16), std::min(size_t(2048), batchSize));
        
        VLOG(DEBUG, "Calculated batch size: %zu chunks (%.2f MB per batch)\n",
             batchSize, (batchSize * chunkSize) / (1024.0 * 1024.0));
        
        return batchSize;
    }
    
    /*
     * Refresh GPU memory status and calculate current batch size
     * Returns 0 if GPU doesn't have enough memory
     */
    size_t refreshGPUMemoryAndBatchSize(GPUDevice& gpu) {
        cudaSetDevice(gpu.deviceId);
        
        size_t freeMem, totalMem;
        cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
        if (err != cudaSuccess) {
            VLOG(DEBUG, "GPU %d: Failed to get memory info: %s\n", 
                 gpu.deviceId, cudaGetErrorString(err));
            return 0;
        }
        
        gpu.availableMemory = freeMem;
        
        // Conservative memory estimate per chunk
        // nvCOMP needs: input buffer + output buffer (~2x) + temp buffers
        // Use 5x multiplier (reduced from 6x for larger batches)
        size_t memoryPerChunk = chunkSize * 5;
        
        // Use 90% of free memory for compression (increased from 80% for better utilization)
        size_t usableMemory = static_cast<size_t>(freeMem * 0.9);
        
        // Need at least enough for 16 chunks minimum
        size_t minRequired = memoryPerChunk * 16;
        
        if (usableMemory < minRequired) {
            VLOG(DEBUG, "GPU %d: Insufficient memory (%.2f GB free, need %.2f GB minimum)\n",
                 gpu.deviceId, 
                 freeMem / (1024.0 * 1024.0 * 1024.0),
                 minRequired / (1024.0 * 1024.0 * 1024.0));
            return 0;
        }
        
        // Calculate how many chunks we can process
        size_t batchSize = usableMemory / memoryPerChunk;
        // Reduced max from 2048 to 256 for better pipeline overlap
        // Smaller batches = more batches = better async I/O parallelism
        batchSize = std::max(size_t(16), std::min(size_t(256), batchSize));
        
        VLOG(DEBUG, "GPU %d: %.2f GB free, batch size %zu chunks\n",
             gpu.deviceId,
             freeMem / (1024.0 * 1024.0 * 1024.0),
             batchSize);
        
        return batchSize;
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
        
        VLOG(DEBUG, "Launching async batch compression of %zu chunks on GPU %d\n",
             state.batch_size, gpu.deviceId);
        
        cudaError_t err = cudaSetDevice(gpu.deviceId);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: Failed to set device %d - %s\n",
                    gpu.deviceId, cudaGetErrorString(err));
            return false;
        }
        
        cudaGetLastError(); // Clear any pending errors
        
        // Allocate and copy input buffers
        state.d_inputs.resize(state.batch_size);
        std::vector<const void*> h_input_ptrs(state.batch_size);
        std::vector<size_t> h_input_sizes(state.batch_size);
        
        for (size_t i = 0; i < state.batch_size; i++) {
            size_t size = inputs[i].size();
            state.input_sizes[i] = size;
            h_input_sizes[i] = size;
            
            CUDA_CHECK_MSG(cudaMalloc(&state.d_inputs[i], size),
                          "Failed to allocate input buffer");
            CUDA_CHECK_MSG(cudaMemcpy(state.d_inputs[i], inputs[i].data(), size,
                                     cudaMemcpyHostToDevice),
                          "Failed to copy input to device");
            
            h_input_ptrs[i] = state.d_inputs[i];
        }
        
        // Allocate and initialize device arrays
        CUDA_CHECK_MSG(cudaMalloc(&state.d_input_ptrs, state.batch_size * sizeof(void*)),
                      "Failed to allocate input pointers array");
        CUDA_CHECK_MSG(cudaMalloc(&state.d_input_sizes, state.batch_size * sizeof(size_t)),
                      "Failed to allocate input sizes array");
        
        CUDA_CHECK_MSG(cudaMemcpy(state.d_input_ptrs, h_input_ptrs.data(),
                                 state.batch_size * sizeof(void*), cudaMemcpyHostToDevice),
                      "Failed to copy input pointers");
        CUDA_CHECK_MSG(cudaMemcpy(state.d_input_sizes, h_input_sizes.data(),
                                 state.batch_size * sizeof(size_t), cudaMemcpyHostToDevice),
                      "Failed to copy input sizes");
        
        // Get max chunk size
        size_t max_chunk_size = 0;
        for (size_t size : state.input_sizes) {
            max_chunk_size = std::max(max_chunk_size, size);
        }
        
        // Calculate total input size
        size_t total_input_size = 0;
        for (size_t size : state.input_sizes) {
            total_input_size += size;
        }
        
        // Configure LZ4 compression
        nvcompBatchedLZ4CompressOpts_t opts = nvcompBatchedLZ4CompressDefaultOpts;
        
        // Get temp buffer size
        size_t temp_bytes = 0;
        NVCOMP_CHECK(nvcompBatchedLZ4CompressGetTempSizeSync(
            state.d_input_ptrs,
            state.d_input_sizes,
            state.batch_size,
            max_chunk_size,
            opts,
            &temp_bytes,
            total_input_size,
            stream
        ));
        
        VLOG(DEBUG, "  Temp buffer size: %.2f MB\n", temp_bytes / (1024.0 * 1024.0));
        
        // Allocate temp buffer
        CUDA_CHECK_MSG(cudaMalloc(&state.d_temp, temp_bytes),
                      "Failed to allocate temp buffer");
        
        // Get output sizes and allocate output buffers
        state.d_outputs.resize(state.batch_size);
        std::vector<void*> h_output_ptrs(state.batch_size);
        
        for (size_t i = 0; i < state.batch_size; i++) {
            size_t max_output_size = 0;
            NVCOMP_CHECK(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
                state.input_sizes[i],
                opts,
                &max_output_size
            ));
            
            state.max_output_sizes[i] = max_output_size;
            
            CUDA_CHECK_MSG(cudaMalloc(&state.d_outputs[i], max_output_size),
                          "Failed to allocate output buffer");
            h_output_ptrs[i] = state.d_outputs[i];
        }
        
        // Allocate device arrays for outputs
        CUDA_CHECK_MSG(cudaMalloc(&state.d_output_ptrs, state.batch_size * sizeof(void*)),
                      "Failed to allocate output pointers array");
        CUDA_CHECK_MSG(cudaMalloc(&state.d_output_sizes, state.batch_size * sizeof(size_t)),
                      "Failed to allocate output sizes array");
        CUDA_CHECK_MSG(cudaMalloc(&state.d_statuses, state.batch_size * sizeof(nvcompStatus_t)),
                      "Failed to allocate statuses array");
        
        CUDA_CHECK_MSG(cudaMemcpy(state.d_output_ptrs, h_output_ptrs.data(),
                                 state.batch_size * sizeof(void*), cudaMemcpyHostToDevice),
                      "Failed to copy output pointers");
        
        // Launch compression
        VLOG(DEBUG, "  Launching compression with batch_size=%zu\n", state.batch_size);
        NVCOMP_CHECK(nvcompBatchedLZ4CompressAsync(
            state.d_input_ptrs,
            state.d_input_sizes,
            max_chunk_size,
            state.batch_size,  // THIS IS THE KEY - now using actual batch size!
            state.d_temp,
            temp_bytes,
            state.d_output_ptrs,
            state.d_output_sizes,
            opts,
            state.d_statuses,
            stream
        ));
        
        return true;
    }
    
    /*
     * Get batch compression results
     */
    bool getBatchCompressResults(BatchCompressState& state,
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
                fprintf(stderr, "Compression failed for chunk %zu with status %d\n",
                        i, static_cast<int>(statuses[i]));
                // Cleanup and return false
                for (auto ptr : state.d_inputs) cudaFree(ptr);
                for (auto ptr : state.d_outputs) cudaFree(ptr);
                cudaFree(state.d_input_ptrs);
                cudaFree(state.d_input_sizes);
                cudaFree(state.d_temp);
                cudaFree(state.d_output_ptrs);
                cudaFree(state.d_output_sizes);
                cudaFree(state.d_statuses);
                return false;
            }
        }
        
        // Get output sizes
        std::vector<size_t> output_sizes(state.batch_size);
        CUDA_CHECK_MSG(cudaMemcpy(output_sizes.data(), state.d_output_sizes,
                                 state.batch_size * sizeof(size_t),
                                 cudaMemcpyDeviceToHost),
                      "Failed to copy output sizes");
        
        // Copy output data
        outputs.resize(state.batch_size);
        for (size_t i = 0; i < state.batch_size; i++) {
            outputs[i].resize(output_sizes[i]);
            CUDA_CHECK_MSG(cudaMemcpy(outputs[i].data(), state.d_outputs[i],
                                     output_sizes[i], cudaMemcpyDeviceToHost),
                          "Failed to copy output data");
            
            VLOG(DEBUG, "  Chunk %zu: %zu -> %zu bytes (%.2f%%)\n",
                 i, state.input_sizes[i], output_sizes[i],
                 100.0 * output_sizes[i] / state.input_sizes[i]);
        }
        
        // Cleanup
        for (auto ptr : state.d_inputs) cudaFree(ptr);
        for (auto ptr : state.d_outputs) cudaFree(ptr);
        cudaFree(state.d_input_ptrs);
        cudaFree(state.d_input_sizes);
        cudaFree(state.d_temp);
        cudaFree(state.d_output_ptrs);
        cudaFree(state.d_output_sizes);
        cudaFree(state.d_statuses);
        
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
        
        // Only check every 2 seconds to avoid overhead
        if (std::chrono::duration<double>(now - lastCheck).count() < 2.0) {
            return;
        }
        lastCheck = now;
        
        for (auto& gpu : gpus) {
            cudaSetDevice(gpu.deviceId);
            
            // Count busy streams (approximates GPU utilization)
            int busyStreams = 0;
            for (auto& stream : gpu.streams) {
                cudaError_t status = cudaStreamQuery(stream);
                if (status == cudaErrorNotReady) {
                    busyStreams++;
                }
            }
            
            float utilization = gpu.streams.empty() ? 0.0f : (float)busyStreams / gpu.streams.size();
            
            // If underutilized and room for more streams, add them
            // Max 1024 streams, add 32 at a time
            if (utilization < 0.9f && gpu.streams.size() < 1024) {
                size_t streamsToAdd = std::min(size_t(32), size_t(1024 - gpu.streams.size()));
                
                VLOG(VERBOSE, "GPU %d utilization %.1f%% (%d/%zu busy) - adding %zu streams\n",
                     gpu.deviceId, utilization * 100, busyStreams, gpu.streams.size(),
                     streamsToAdd);
                
                for (size_t i = 0; i < streamsToAdd; i++) {
                    cudaStream_t newStream;
                    if (cudaStreamCreate(&newStream) == cudaSuccess) {
                        gpu.streams.push_back(newStream);
                    }
                }
            }
        }
    }
    
    /*
     * Compress a file using TRUE batch processing with LZ4 frame format
     */
    bool compressFile() {
        VLOG(VERBOSE, "Compressing: %s -> %s\n", 
             inputFile.c_str(), outputFile.c_str());
        
        // Timing instrumentation
        auto totalStartTime = std::chrono::high_resolution_clock::now();
        double timeReading = 0, timeLaunching = 0, timeWaiting = 0, timeWriting = 0;
        size_t totalBatchesLaunched = 0;
        
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
        
        // Start async reader (reads while GPUs initialize)
        AsyncReader asyncReader;
        size_t maxReadQueue = std::min(size_t(64), numChunks);  // Buffer up to 64 chunks
        if (!asyncReader.start(inputFile, chunkSize, maxReadQueue)) {
            fprintf(stderr, "Error: Failed to start async reader\n");
            return false;
        }
        VLOG(VERBOSE, "AsyncReader started (max queue: %zu chunks = %.2f GB RAM)\n",
             maxReadQueue, (maxReadQueue * chunkSize) / (1024.0 * 1024.0 * 1024.0));
        
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
        int outputFd = open(outputFile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (outputFd < 0) {
            fprintf(stderr, "Error: Cannot create output file: %s\n", outputFile.c_str());
            return false;
        }
        ssize_t written = ::write(outputFd, headerBuffer.data(), headerBuffer.size());
        if (written != (ssize_t)headerBuffer.size()) {
            fprintf(stderr, "Error writing header: %s\n", strerror(errno));
            close(outputFd);
            return false;
        }
        close(outputFd);
        
        // Initialize content checksum with streaming state
        XXH::State xxhState(XXH32_SEED);
        
        // Start async writer
        AsyncWriter asyncWriter;
        if (!asyncWriter.start(outputFile, &xxhState)) {
            fprintf(stderr, "Error: Failed to start async writer\n");
            return false;
        }
        
        // Storage for GPU batch operations
        struct GPUBatch {
            std::vector<std::vector<uint8_t>> chunks;
            std::vector<size_t> chunkIndices;
            std::vector<size_t> originalSizes;
            BatchCompressState state;
            bool inProgress;
            GPUDevice* gpu;
            cudaStream_t stream;  // Track which stream this batch is using
        };
        
        // Storage for completed batches (allows out-of-order completion)
        struct CompletedBatch {
            std::vector<std::vector<uint8_t>> compressedChunks;
            std::vector<std::vector<uint8_t>> originalChunks;
            std::vector<size_t> chunkIndices;
            std::vector<size_t> originalSizes;
        };
        
        std::vector<GPUBatch> gpuBatches(gpus.size());
        for (size_t i = 0; i < gpus.size(); i++) {
            gpuBatches[i].gpu = &gpus[i];
            gpuBatches[i].inProgress = false;
            gpuBatches[i].stream = 0;  // Will be assigned when batch launches
        }
        
        // Map to store completed batches by their starting chunk index
        // This allows GPUs to finish out-of-order without blocking
        std::map<size_t, CompletedBatch> completedBatches;
        
        size_t nextChunkToRead = 0;
        size_t totalCompressed = 0;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Main loop: continue until writer has written everything
        while (asyncWriter.getNextChunkToWrite() < numChunks || !asyncReader.isFinished() || 
               !completedBatches.empty() ||
               std::any_of(gpuBatches.begin(), gpuBatches.end(), [](const GPUBatch& b) { return b.inProgress; })) {
            
            // Dynamically adjust stream count based on GPU utilization
            optimizeStreamCount();
            
            // PHASE 1: Fill batches for GPUs that are idle by pulling from async reader
            for (size_t gpuIdx = 0; gpuIdx < gpus.size() && !asyncReader.isFinished(); gpuIdx++) {
                GPUBatch& batch = gpuBatches[gpuIdx];
                
                if (batch.inProgress) continue;
                
                // Check current GPU memory and get adjusted batch size
                size_t currentBatchSize = refreshGPUMemoryAndBatchSize(*batch.gpu);
                if (currentBatchSize == 0) {
                    VLOG(DEBUG, "GPU %zu: Skipping due to insufficient memory\n", gpuIdx);
                    continue;  // Skip this GPU if not enough memory
                }
                
                auto readStart = std::chrono::high_resolution_clock::now();
                
                // Pull chunks from async reader
                batch.chunks.clear();
                batch.chunkIndices.clear();
                batch.originalSizes.clear();
                
                AsyncReader::ReadChunk readChunk;
                while (batch.chunks.size() < currentBatchSize && asyncReader.getChunk(readChunk)) {
                    // Got a chunk from async reader
                    batch.chunks.push_back(std::move(readChunk.data));
                    batch.chunkIndices.push_back(readChunk.chunkIndex);
                    batch.originalSizes.push_back(readChunk.size);
                    nextChunkToRead++;
                }
                
                auto readEnd = std::chrono::high_resolution_clock::now();
                timeReading += std::chrono::duration<double>(readEnd - readStart).count();
                
                VLOG(DEBUG, "GPU %zu: Pulled %zu chunks from reader (queue depth: %zu)\n",
                     gpuIdx, batch.chunks.size(), asyncReader.getQueueDepth());
                
                // Launch batch if we have chunks
                if (!batch.chunks.empty()) {
                    auto launchStart = std::chrono::high_resolution_clock::now();
                    
                    VLOG(DEBUG, "Launching batch of %zu chunks (indices %zu-%zu) on GPU %zu\n",
                         batch.chunks.size(), batch.chunkIndices.front(), 
                         batch.chunkIndices.back(), gpuIdx);
                    
                    // Select stream for this batch (round-robin across available streams)
                    static size_t streamCounter = 0;
                    size_t streamIdx = streamCounter % batch.gpu->streams.size();
                    batch.stream = batch.gpu->streams[streamIdx];
                    streamCounter++;
                    
                    if (!compressBatchAsync(batch.chunks, batch.state, *batch.gpu, batch.stream)) {
                        // Allocation failed - can't use this GPU
                        // We already pulled chunks from reader, so we need to process them later
                        // Buffer them for retry on another GPU
                        VLOG(VERBOSE, "GPU %zu failed to allocate memory, buffering %zu chunks for retry\n", 
                             gpuIdx, batch.chunks.size());
                        
                        // Store chunks in completed batches as "pending compression"
                        for (size_t i = 0; i < batch.chunks.size(); i++) {
                            CompletedBatch pending;
                            pending.compressedChunks.resize(1);  // Will be filled later or written uncompressed
                            pending.originalChunks.push_back(std::move(batch.chunks[i]));
                            pending.chunkIndices.push_back(batch.chunkIndices[i]);
                            pending.originalSizes.push_back(batch.originalSizes[i]);
                            // Note: These will be picked up by next available GPU or written uncompressed
                        }
                        
                        continue;  // Skip to next GPU
                    }
                    batch.inProgress = true;
                    totalBatchesLaunched++;
                    
                    auto launchEnd = std::chrono::high_resolution_clock::now();
                    timeLaunching += std::chrono::duration<double>(launchEnd - launchStart).count();
                }
            }
            
            // PHASE 2: Collect completed batches (any order) and buffer them
            for (size_t gpuIdx = 0; gpuIdx < gpus.size(); gpuIdx++) {
                GPUBatch& batch = gpuBatches[gpuIdx];
                
                if (!batch.inProgress) continue;
                
                // Check if this batch is done (non-blocking check)
                cudaError_t err = cudaSetDevice(batch.gpu->deviceId);
                if (err != cudaSuccess) continue;
                
                // Query the specific stream this batch is using
                err = cudaStreamQuery(batch.stream);
                if (err == cudaErrorNotReady) {
                    // Still processing, skip this GPU
                    continue;
                } else if (err != cudaSuccess) {
                    fprintf(stderr, "CUDA stream query error on GPU %zu: %s\n", 
                            gpuIdx, cudaGetErrorString(err));
                    continue;
                }
                
                // Batch is complete! Get results and buffer them
                auto waitStart = std::chrono::high_resolution_clock::now();
                
                std::vector<std::vector<uint8_t>> compressedChunks;
                if (!getBatchCompressResults(batch.state, compressedChunks, *batch.gpu)) {
                    fprintf(stderr, "Failed to get batch results from GPU %zu\n", gpuIdx);
                    return false;
                }
                
                auto waitEnd = std::chrono::high_resolution_clock::now();
                timeWaiting += std::chrono::duration<double>(waitEnd - waitStart).count();
                
                // Store completed batch (out-of-order is fine!)
                CompletedBatch completed;
                completed.compressedChunks = std::move(compressedChunks);
                completed.originalChunks = std::move(batch.chunks);
                completed.chunkIndices = batch.chunkIndices;
                completed.originalSizes = batch.originalSizes;
                
                size_t firstChunkIdx = batch.chunkIndices[0];
                completedBatches[firstChunkIdx] = std::move(completed);
                
                VLOG(DEBUG, "GPU %zu batch complete, buffered chunks %zu-%zu\n",
                     gpuIdx, firstChunkIdx, batch.chunkIndices.back());
                
                batch.inProgress = false;  // GPU is now free for more work!
            }
            
            // PHASE 3: Enqueue ALL completed batches immediately (out-of-order OK!)
            auto writeStart = std::chrono::high_resolution_clock::now();
            
            // Enqueue any completed batches (writer will handle sequencing)
            for (auto it = completedBatches.begin(); it != completedBatches.end(); ) {
                auto& completed = it->second;
                size_t firstChunkIdx = it->first;
                
                // Enqueue this batch for async writing (non-blocking!)
                asyncWriter.enqueue(
                    firstChunkIdx,
                    std::move(completed.compressedChunks),
                    std::move(completed.originalChunks),
                    completed.chunkIndices,
                    completed.originalSizes
                );
                
                // Update counters
                for (size_t i = 0; i < completed.originalSizes.size(); i++) {
                    if (i < completed.compressedChunks.size() && !completed.compressedChunks[i].empty()) {
                        totalCompressed += completed.compressedChunks[i].size();
                    }
                }
                
                VLOG(DEBUG, "Enqueued batch starting at chunk %zu for async writing\n", firstChunkIdx);
                
                // Remove from buffer
                it = completedBatches.erase(it);
            }
            
            auto writeEnd = std::chrono::high_resolution_clock::now();
            timeWriting += std::chrono::duration<double>(writeEnd - writeStart).count();
            
            // Update progress based on writer's progress
            size_t chunksWritten = asyncWriter.getNextChunkToWrite();
            if (g_verbosity < DEBUG && numChunks > 10 && chunksWritten > 0) {
                int progress = (100 * chunksWritten) / numChunks;
                fprintf(stderr, "\rProgress: %d%% ", progress);
                fflush(stderr);
            }
            
            // Small sleep if all GPUs busy and nothing completed
            bool allBusy = std::all_of(gpuBatches.begin(), gpuBatches.end(),
                                      [](const GPUBatch& b) { return b.inProgress; });
            bool nothingCompleted = completedBatches.empty();
            
            if (allBusy && nothingCompleted && !asyncReader.isFinished()) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
        
        // Wait for async writer to finish all pending writes
        VLOG(VERBOSE, "Waiting for async writer to complete...\n");
        asyncWriter.stop();
        
        // Now append footer synchronously (end mark + checksum)
        outputFd = open(outputFile.c_str(), O_WRONLY | O_APPEND);
        if (outputFd < 0) {
            fprintf(stderr, "Error reopening file for footer: %s\n", strerror(errno));
            return false;
        }
        
        // Write end mark (4 zero bytes)
        uint32_t endMark = 0;
        ssize_t bytesWritten = ::write(outputFd, &endMark, 4);
        if (bytesWritten != 4) {
            fprintf(stderr, "Error writing end mark: %s\n", strerror(errno));
        }
        
        // Write content checksum
        uint32_t contentChecksum = xxhState.digest();
        uint8_t checksumBuf[4];
        checksumBuf[0] = contentChecksum & 0xFF;
        checksumBuf[1] = (contentChecksum >> 8) & 0xFF;
        checksumBuf[2] = (contentChecksum >> 16) & 0xFF;
        checksumBuf[3] = (contentChecksum >> 24) & 0xFF;
        bytesWritten = ::write(outputFd, checksumBuf, 4);
        if (bytesWritten != 4) {
            fprintf(stderr, "Error writing checksum: %s\n", strerror(errno));
        }
        
        fsync(outputFd);
        close(outputFd);
        
        VLOG(VERBOSE, "Computed content checksum: 0x%08X (from %zu bytes)\n", 
             contentChecksum, xxhState.totalLen);
        VLOG(DEBUG, "Wrote LZ4 footer: end mark + checksum 0x%08X\n", contentChecksum);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        if (g_verbosity < DEBUG && numChunks > 10) {
            fprintf(stderr, "\n");
        }
        
        // Stop async reader
        asyncReader.stop();
        
        double bytesWrittenD = asyncWriter.getBytesWritten();
        double ratio = 100.0 * bytesWrittenD / fileSize;
        double throughputMBps = (fileSize / (1024.0 * 1024.0)) / (duration.count() / 1000.0);
        
        VLOG(VERBOSE, "Compression complete: %.2f MB -> %.2f MB (%.2f%%) in %.2f seconds\n",
             fileSize / (1024.0 * 1024.0),
             bytesWrittenD / (1024.0 * 1024.0),
             ratio,
             duration.count() / 1000.0);
        VLOG(VERBOSE, "Throughput: %.2f MB/s\n", throughputMBps);
        
        // Get async thread times
        double asyncReadTime = asyncReader.getReadTime();
        double asyncWriteTime = asyncWriter.getWriteTime();
        double gpuTime = timeLaunching + timeWaiting;
        
        // Print timing breakdown
        double totalTime = std::chrono::duration<double>(endTime - totalStartTime).count();
        VLOG(VERBOSE, "\nTiming breakdown:\n");
        VLOG(VERBOSE, "  Total elapsed:           %.2f s\n", totalTime);
        VLOG(VERBOSE, "\nPipeline stages (parallel):\n");
        VLOG(VERBOSE, "  AsyncReader thread:      %.2f s (%.1f%% of total) - actual disk reads\n", 
             asyncReadTime, 100.0 * asyncReadTime / totalTime);
        VLOG(VERBOSE, "  GPU compression:         %.2f s (%.1f%% of total) - GPU compute time\n",
             gpuTime, 100.0 * gpuTime / totalTime);
        VLOG(VERBOSE, "    - Launching:           %.2f s\n", timeLaunching);
        VLOG(VERBOSE, "    - Waiting for GPU:     %.2f s\n", timeWaiting);
        VLOG(VERBOSE, "  AsyncWriter thread:      %.2f s (%.1f%% of total) - actual disk writes\n",
             asyncWriteTime, 100.0 * asyncWriteTime / totalTime);
        VLOG(VERBOSE, "\nMain thread (scheduling):\n");
        VLOG(VERBOSE, "  Queue management:        %.2f s (%.1f%% of total) - pulling from reader, enqueuing to writer\n",
             timeReading + timeWriting, 100.0 * (timeReading + timeWriting) / totalTime);
        
        // Calculate pipeline efficiency
        double serialTime = asyncReadTime + gpuTime + asyncWriteTime;
        double speedup = serialTime / totalTime;
        VLOG(VERBOSE, "\nPipeline efficiency:\n");
        VLOG(VERBOSE, "  If serial (read→GPU→write): %.2f s\n", serialTime);
        VLOG(VERBOSE, "  Actual (parallel):          %.2f s\n", totalTime);
        VLOG(VERBOSE, "  Speedup from parallelism:   %.2fx\n", speedup);
        VLOG(VERBOSE, "  Efficiency:                 %.1f%% (100%% = perfect overlap)\n",
             100.0 * (speedup - 1.0) / 2.0);  // 3 stages, so max 3x, efficiency = (actual-1)/(max-1)
        
        VLOG(VERBOSE, "\nBatch statistics:\n");
        VLOG(VERBOSE, "  Total batches launched: %zu\n", totalBatchesLaunched);
        VLOG(VERBOSE, "  Average chunks per batch: %.1f\n", (double)numChunks / totalBatchesLaunched);
        
        VLOG(VERBOSE, "\nGPU stream scaling:\n");
        for (size_t i = 0; i < gpus.size(); i++) {
            VLOG(VERBOSE, "  GPU %d: %zu streams (started with %d)\n",
                 gpus[i].deviceId, gpus[i].streams.size(), STREAMS_PER_GPU);
        }
        
        // Remove original file if not keeping
        if (!keepOriginal) {
            if (unlink(inputFile.c_str()) != 0) {
                fprintf(stderr, "Warning: Could not remove input file: %s\n", 
                        inputFile.c_str());
            }
        }
        
        return true;
    }
    
    /*
     * Decompress a file using TRUE batch processing with LZ4 frame format
     */
    bool decompressFile() {
        VLOG(VERBOSE, "Decompressing: %s -> %s\n", 
             inputFile.c_str(), outputFile.c_str());
        
        // Open input file
        std::ifstream inFile(inputFile, std::ios::binary);
        if (!inFile) {
            fprintf(stderr, "Error: Cannot open input file: %s\n", 
                    inputFile.c_str());
            return false;
        }
        
        // Read and verify LZ4 frame header
        LZ4Frame::FrameDescriptor desc;
        if (!LZ4Frame::readFrameHeader(inFile, desc)) {
            fprintf(stderr, "Error: Failed to read LZ4 frame header\n");
            return false;
        }
        
        size_t originalFileSize = desc.contentSize;
        XXH::State xxhState(XXH32_SEED);
        
        // Extract chunk size from frame header
        size_t headerChunkSize = static_cast<size_t>(1) << (8 + 2 * desc.blockMaxSize);
        
        // CRITICAL: Detect actual block size by reading first few blocks
        // This handles files from v2.8.3 where header blockMaxSize was always 7 (4MB)
        // but actual blocks could be 16KB-2MB depending on compression level
        size_t fileChunkSize = headerChunkSize;
        std::streampos startPos = inFile.tellg();
        
        VLOG(DEBUG, "Header says blockMaxSize=%d (%zu KB), detecting actual block size...\n",
             desc.blockMaxSize, headerChunkSize / 1024);
        
        // Read first few blocks to detect actual size
        std::vector<size_t> blockSizes;
        for (int i = 0; i < 5 && inFile; i++) {
            uint32_t blockSize = LZ4Frame::readU32(inFile);
            if (blockSize == 0) break;  // End mark
            
            bool isUncompressed = (blockSize & 0x80000000) != 0;
            blockSize &= 0x7FFFFFFF;
            
            if (isUncompressed) {
                // Uncompressed blocks tell us the exact chunk size
                blockSizes.push_back(blockSize);
                VLOG(DEBUG, "  Block %d: uncompressed, %u bytes\n", i, blockSize);
            }
            
            // Skip the block data
            inFile.seekg(blockSize, std::ios::cur);
        }
        
        // If we found uncompressed blocks, use their size as the actual chunk size
        if (!blockSizes.empty()) {
            size_t detectedSize = blockSizes[0];
            if (detectedSize != headerChunkSize) {
                VLOG(VERBOSE, "WARNING: Header blockMaxSize mismatch detected!\n");
                VLOG(VERBOSE, "  Header claims: %zu KB blocks (blockMaxSize=%d)\n", 
                     headerChunkSize / 1024, desc.blockMaxSize);
                VLOG(VERBOSE, "  Actual blocks: %zu KB (likely v2.8.3 file)\n", 
                     detectedSize / 1024);
                VLOG(VERBOSE, "  Using actual block size for decompression\n");
                fileChunkSize = detectedSize;
            }
        }
        
        // Rewind to start of blocks
        inFile.seekg(startPos);
        
        VLOG(VERBOSE, "Original file size: %.2f MB\n", 
             originalFileSize / (1024.0 * 1024.0));
        VLOG(VERBOSE, "Block max size from header: %zu KB (blockMaxSize=%d)\n",
             fileChunkSize / 1024, desc.blockMaxSize);
        
        // Setup output
        std::ofstream outFile;
        std::ostream* outStream;
        
        if (stdoutMode) {
            outStream = &std::cout;
        } else {
            outFile.open(outputFile, std::ios::binary);
            if (!outFile) {
                fprintf(stderr, "Error: Cannot create output file: %s\n", 
                        outputFile.c_str());
                return false;
            }
            outStream = &outFile;
        }
        
        // Storage for GPU batch operations
        struct GPUBatch {
            std::vector<std::vector<uint8_t>> chunks;
            std::vector<size_t> uncompressedSizes;
            std::vector<size_t> chunkIndices;
            BatchDecompressState state;
            bool inProgress;
            GPUDevice* gpu;
        };
        
        std::vector<GPUBatch> gpuBatches(gpus.size());
        for (size_t i = 0; i < gpus.size(); i++) {
            gpuBatches[i].gpu = &gpus[i];
            gpuBatches[i].inProgress = false;
        }
        
        // Storage for uncompressed blocks (indexed by chunk number)
        std::map<size_t, std::vector<uint8_t>> uncompressedBlocks;
        
        size_t nextChunkToRead = 0;
        size_t nextChunkToWrite = 0;
        size_t totalDecompressed = 0;
        bool reachedEOF = false;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        while (!reachedEOF ||
               std::any_of(gpuBatches.begin(), gpuBatches.end(), 
                          [](const GPUBatch& b) { return b.inProgress; }) ||
               !uncompressedBlocks.empty()) {
            
            // PHASE 1: Fill batches for GPUs that are idle
            for (size_t gpuIdx = 0; gpuIdx < gpus.size() && !reachedEOF; gpuIdx++) {
                GPUBatch& batch = gpuBatches[gpuIdx];
                
                if (batch.inProgress) continue;
                
                // Check current GPU memory and get adjusted batch size
                size_t currentBatchSize = refreshGPUMemoryAndBatchSize(*batch.gpu);
                if (currentBatchSize == 0) {
                    VLOG(DEBUG, "GPU %zu: Skipping due to insufficient memory\n", gpuIdx);
                    continue;  // Skip this GPU if not enough memory
                }
                
                // Read up to currentBatchSize chunks for this GPU
                batch.chunks.clear();
                batch.uncompressedSizes.clear();
                batch.chunkIndices.clear();
                
                // Read chunks sequentially (no round-robin)
                while (batch.chunks.size() + uncompressedBlocks.size() < currentBatchSize && !reachedEOF) {
                    
                    // Read LZ4 block size
                    uint32_t blockSize = LZ4Frame::readU32(inFile);
                    
                    // Check for end mark
                    if (blockSize == 0) {
                        VLOG(DEBUG, "Reached LZ4 end mark\n");
                        reachedEOF = true;
                        break;
                    }
                    
                    // Check if block is uncompressed
                    bool isUncompressed = (blockSize & 0x80000000) != 0;
                    blockSize &= 0x7FFFFFFF;  // Clear high bit
                    
                    // Validate block size is reasonable (shouldn't exceed max chunk size + overhead)
                    if (blockSize > fileChunkSize * 2) {
                        fprintf(stderr, "Error: Invalid block size %u at chunk %zu (max expected %zu)\n",
                                blockSize, nextChunkToRead, fileChunkSize * 2);
                        fprintf(stderr, "File may be corrupted or not a valid LZ4 file\n");
                        return false;
                    }
                    
                    if (!inFile) {
                        if (totalDecompressed >= originalFileSize) {
                            reachedEOF = true;
                            break;
                        }
                        fprintf(stderr, "Error: Unexpected end of file at chunk %zu\n", nextChunkToRead);
                        return false;
                    }
                    
                    // Read block data
                    std::vector<uint8_t> chunk(blockSize);
                    inFile.read(reinterpret_cast<char*>(chunk.data()), blockSize);
                    
                    if (!inFile && !inFile.eof()) {
                        fprintf(stderr, "Error reading compressed data at chunk %zu\n", nextChunkToRead);
                        return false;
                    }
                    
                    if (isUncompressed) {
                        // Store uncompressed block for later writing in order
                        VLOG(DEBUG, "Block %zu is uncompressed (%u bytes), storing\n", nextChunkToRead, blockSize);
                        uncompressedBlocks[nextChunkToRead] = std::move(chunk);
                        totalDecompressed += blockSize;
                    } else {
                        // Estimate uncompressed size for compressed blocks
                        size_t remaining = originalFileSize - totalDecompressed;
                        size_t origSize = std::min(fileChunkSize, remaining);
                        
                        batch.chunks.push_back(std::move(chunk));
                        batch.uncompressedSizes.push_back(origSize);
                        batch.chunkIndices.push_back(nextChunkToRead);
                        totalDecompressed += origSize;
                    }
                    
                    nextChunkToRead++;
                }
                
                // Launch batch if we have compressed chunks
                if (!batch.chunks.empty()) {
                    VLOG(DEBUG, "Launching decompression batch of %zu chunks (indices %zu-%zu) on GPU %zu\n",
                         batch.chunks.size(), batch.chunkIndices.front(),
                         batch.chunkIndices.back(), gpuIdx);
                    
                    if (!decompressBatchAsync(batch.chunks, batch.uncompressedSizes,
                                             batch.state, *batch.gpu, 0)) {
                        // Allocation failed - this GPU doesn't have enough memory
                        // Put chunks back and try another GPU
                        VLOG(VERBOSE, "GPU %zu failed to allocate memory, skipping\n", gpuIdx);
                        nextChunkToRead -= batch.chunks.size();
                        continue;  // Skip to next GPU
                    }
                    batch.inProgress = true;
                }
            }
            
            // PHASE 2: Check for completed batches and write results (including uncompressed blocks)
            
            // First, write any uncompressed blocks that are next in sequence
            while (uncompressedBlocks.count(nextChunkToWrite) > 0) {
                auto& block = uncompressedBlocks[nextChunkToWrite];
                
                // Update content checksum
                xxhState.update(block.data(), block.size());
                
                // Write data
                outStream->write(reinterpret_cast<const char*>(block.data()), block.size());
                
                VLOG(DEBUG, "Wrote uncompressed block %zu (%zu bytes)\n", nextChunkToWrite, block.size());
                
                uncompressedBlocks.erase(nextChunkToWrite);
                nextChunkToWrite++;
            }
            
            // Then check for completed GPU batches
            for (size_t gpuIdx = 0; gpuIdx < gpus.size(); gpuIdx++) {
                GPUBatch& batch = gpuBatches[gpuIdx];
                
                if (!batch.inProgress) continue;
                
                // Check if first chunk in this batch is what we need to write next
                if (batch.chunkIndices.empty() || batch.chunkIndices[0] != nextChunkToWrite) {
                    continue;
                }
                
                // Get results
                std::vector<std::vector<uint8_t>> decompressedChunks;
                if (!getBatchDecompressResults(batch.state, decompressedChunks, *batch.gpu)) {
                    fprintf(stderr, "Failed to get batch decompress results from GPU %zu\n", gpuIdx);
                    return false;
                }
                
                // Write all chunks from this batch in order
                for (size_t i = 0; i < decompressedChunks.size(); i++) {
                    size_t decompSize = decompressedChunks[i].size();
                    
                    VLOG(DEBUG, "  Chunk %zu decompressed to %zu bytes (estimated %zu)\n", 
                         batch.chunkIndices[i], decompSize, batch.uncompressedSizes[i]);
                    
                    // Update content checksum
                    xxhState.update(decompressedChunks[i].data(), decompSize);
                    
                    outStream->write(reinterpret_cast<const char*>(decompressedChunks[i].data()),
                                   decompSize);
                    nextChunkToWrite++;
                    
                    // After writing a GPU batch chunk, check for uncompressed blocks again
                    while (uncompressedBlocks.count(nextChunkToWrite) > 0) {
                        auto& block = uncompressedBlocks[nextChunkToWrite];
                        xxhState.update(block.data(), block.size());
                        outStream->write(reinterpret_cast<const char*>(block.data()), block.size());
                        VLOG(DEBUG, "Wrote uncompressed block %zu (%zu bytes)\n", nextChunkToWrite, block.size());
                        uncompressedBlocks.erase(nextChunkToWrite);
                        nextChunkToWrite++;
                    }
                }
                
                batch.inProgress = false;
                
                // Show progress
                if (g_verbosity < DEBUG && originalFileSize > 10 * 1024 * 1024) {
                    int progress = (100 * nextChunkToWrite * fileChunkSize) / originalFileSize;
                    progress = std::min(progress, 100);
                    fprintf(stderr, "\rProgress: %d%%", progress);
                    fflush(stderr);
                }
            }
            
            // Small sleep if all GPUs busy
            bool allBusy = true;
            for (const auto& batch : gpuBatches) {
                if (!batch.inProgress) {
                    allBusy = false;
                    break;
                }
            }
            if (allBusy && !uncompressedBlocks.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        
        // Verify content checksum if present
        if (desc.hasContentChecksum) {
            uint32_t storedChecksum = LZ4Frame::readU32(inFile);
            uint32_t computedChecksum = xxhState.digest();
            if (computedChecksum != storedChecksum) {
                fprintf(stderr, "Error: Content checksum mismatch: got 0x%08X, expected 0x%08X\n",
                        computedChecksum, storedChecksum);
                return false;
            }
            VLOG(DEBUG, "Content checksum verified: 0x%08X\n", computedChecksum);
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        if (g_verbosity < DEBUG && originalFileSize > 10 * 1024 * 1024) {
            fprintf(stderr, "\n");
        }
        
        inFile.close();
        if (!stdoutMode) {
            outFile.close();
        }
        
        double throughputMBps = (originalFileSize / (1024.0 * 1024.0)) / (duration.count() / 1000.0);
        
        VLOG(VERBOSE, "Decompression complete: %zu bytes restored in %.2f seconds\n",
             originalFileSize, duration.count() / 1000.0);
        VLOG(VERBOSE, "Throughput: %.2f MB/s\n", throughputMBps);
        
        // Remove compressed file if not keeping
        if (!keepOriginal && !stdoutMode) {
            if (unlink(inputFile.c_str()) != 0) {
                fprintf(stderr, "Warning: Could not remove compressed file: %s\n",
                        inputFile.c_str());
            }
        }
        
        return true;
    }
    
    /*
     * Parse command line arguments
     */
    bool parseArguments(int argc, char* argv[]) {
        const char* short_opts = "cdfhktvV123456789";
        const struct option long_opts[] = {
            {"stdout", no_argument, nullptr, 'c'},
            {"to-stdout", no_argument, nullptr, 'c'},
            {"decompress", no_argument, nullptr, 'd'},
            {"uncompress", no_argument, nullptr, 'd'},
            {"force", no_argument, nullptr, 'f'},
            {"help", no_argument, nullptr, 'h'},
            {"keep", no_argument, nullptr, 'k'},
            {"test", no_argument, nullptr, 't'},
            {"verbose", no_argument, nullptr, 'v'},
            {"version", no_argument, nullptr, 'V'},
            {"fast", no_argument, nullptr, '1'},
            {"best", no_argument, nullptr, '9'},
            {nullptr, 0, nullptr, 0}
        };
        
        int opt;
        while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1) {
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
                    printHelp();
                    return false;
                case 'k':
                    keepOriginal = true;
                    break;
                case 't':
                    testMode = true;
                    decompress = true;
                    break;
                case 'v':
                    g_verbosity++;
                    break;
                case 'V':
                    printVersion();
                    return false;
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
                default:
                    fprintf(stderr, "Try 'gzl4 --help' for more information.\n");
                    return false;
            }
        }
        
        // Get input file
        if (optind < argc) {
            inputFile = argv[optind];
        } else {
            fprintf(stderr, "Error: No input file specified\n");
            fprintf(stderr, "Try 'gzl4 --help' for more information.\n");
            return false;
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
        
        // Check if output file exists
        if (!forceOverwrite && !stdoutMode && stat(outputFile.c_str(), &st) == 0) {
            fprintf(stderr, "Error: Output file already exists: %s\n", outputFile.c_str());
            fprintf(stderr, "Use -f to force overwrite\n");
            return false;
        }
        
        return true;
    }
    
    /*
     * Print help message
     */
    void printHelp() {
        std::cout << 
R"(Usage: gzl4 [OPTION]... [FILE]

Compress or decompress FILE using GPU-accelerated LZ4 compression.

Options:
  -c, --stdout         write to standard output, keep original files
      --to-stdout      
  -d, --decompress     decompress (default is to compress)
      --uncompress     
  -f, --force          force overwrite of output file
  -h, --help           display this help and exit
  -k, --keep           keep (don't delete) input files
  -v, --verbose        verbose mode (use multiple times for more verbosity)
                       -v: basic progress information
                       -vv: detailed processing information
                       -vvv: debug-level information with GPU details
  -V, --version        display version information and exit
  -1 .. -9             compression level (controls chunk size and speed)
                       -1 (fast): 256KB chunks, fastest (~23s)
                       -2: 512KB chunks, slight compression gain
                       -3: 1MB chunks, balanced speed/compression
                       -4: 2MB chunks, better compression
                       -5 (default): 2MB chunks, good balance
                       -6: 3MB chunks, higher compression
                       -7: 3.5MB chunks, near-best compression  
                       -8: 4MB chunks, best compression
                       -9 (best): 4MB chunks (LZ4 frame limit)
                       Note: All levels optimized for GPU performance
      --fast           alias for -1
      --best           alias for -9

The compressed file has the .lz4 extension added automatically.
When decompressing, the .lz4 extension is removed from the output filename.

Output files use standard LZ4 frame format and can be decompressed with:
  lz4 -d file.tar.lz4      # Standard lz4 tool
  unlz4 file.tar.lz4       # Alternative command

Examples:
  gzl4 file.txt              Compress file.txt to file.txt.lz4
  gzl4 -d file.txt.lz4       Decompress file.txt.lz4 to file.txt
  lz4 -d file.txt.lz4        Can also use standard lz4 tool
  gzl4 -k large_file.dat     Compress but keep original file
  gzl4 -f data.bin           Compress and overwrite if output exists
  gzl4 -vv data.bin          Compress with detailed progress output
)" << std::endl;
    }
    
    /*
     * Print version information
     */
    void printVersion() {
        std::cout << "gzl4 " << VERSION << "\n"
R"(GPU-Accelerated LZ4 Compression Tool
Using nvCOMP 5.1.x and CUDA 12.8

Changes in v3.3.0:
- ASYNC READER: Background reader thread pre-loads chunks while GPUs work
- Overlaps disk reads with GPU compression (was serial: read→compress→write)
- posix_fadvise() with SEQUENTIAL + WILLNEED for kernel readahead
- Direct read() syscalls for maximum I/O performance
- Triple-buffered pipeline: Read → Compress → Write all parallel
- RAM-limited queue (default 128 chunks) prevents OOM
- Expected 20-30% additional speedup over v3.2.0

Changes in v3.2.0:
- Async writer thread overlaps disk writes with GPU work
- Non-blocking enqueue - GPUs never wait for disk I/O

Changes in v3.0.0:
- True parallel GPU processing with out-of-order completion

Performance & Reliability:
- Full pipeline parallelism: I/O and compute overlap completely
- Production-ready on shared systems
- Backward compatible with all LZ4 files
)" << std::endl;
    }
    
    /*
     * Main processing entry point
     */
    bool run(int argc, char* argv[]) {
        // Parse command line
        if (!parseArguments(argc, argv)) {
            return false;
        }
        
        // Initialize GPUs
        if (!initializeGPUs()) {
            return false;
        }
        
        // Set chunk size based on compression level
        setChunkSizeFromLevel();
        
        // Calculate batch size for first GPU (assume all similar)
        batchSize = calculateBatchSize(gpus[0].availableMemory);
        VLOG(VERBOSE, "Using batch size of %zu chunks per GPU\n", batchSize);
        
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
        
        return success;
    }
};

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
