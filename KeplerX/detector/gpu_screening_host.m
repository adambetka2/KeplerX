// Metal host for gpu_screening.metal: window and fold pipelines per duration count, a merge, one blocking call.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
#define MAX_DURATIONS 16
#define FOLD_DURATIONS 16
static id<MTLComputePipelineState> windowers[MAX_DURATIONS], folders[FOLD_DURATIONS], merger;

static id<MTLComputePipelineState> specialised(id<MTLLibrary> library, NSString *name, int m,
                                               NSError **error) {
    MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
    [values setConstantValue:&m type:MTLDataTypeInt atIndex:0];
    id<MTLFunction> function = [library newFunctionWithName:name constantValues:values error:error];
    return function ? [device newComputePipelineStateWithFunction:function error:error] : nil;
}

typedef struct {
    int32_t n, table_size;
    float grid, edge, n_trans;
    int32_t n_dur;
    float threshold;
} Params;

int v3_gpu_open(const char *source, char *message, int length) {
    @autoreleasepool {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            snprintf(message, length, "no Metal device");
            return 1;
        }
        MTLCompileOptions *options = [MTLCompileOptions new];
        // Double-float arithmetic needs IEEE float behaviour.
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError *error = nil;
        id<MTLLibrary> library =
            [device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                 options:options
                                   error:&error];
        if (!library) {
            snprintf(message, length, "%s", [[error description] UTF8String]);
            return 2;
        }
        for (int m = 1; m <= MAX_DURATIONS; m++) {
            windowers[m - 1] = specialised(library, @"window_fold", m, &error);
            if (m <= FOLD_DURATIONS) folders[m - 1] = specialised(library, @"fold", m, &error);
            if (!windowers[m - 1] || (m <= FOLD_DURATIONS && !folders[m - 1])) {
                snprintf(message, length, "%s", [[error description] UTF8String]);
                return 3;
            }
        }
        merger = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"merge_partials"]
                                                       error:&error];
        if (!merger) {
            snprintf(message, length, "%s", [[error description] UTF8String]);
            return 3;
        }
        queue = [device newCommandQueue];
        snprintf(message, length, "%s", [[device name] UTF8String]);
        return 0;
    }
}

// Threadgroup limit of the window_fold (fold: the fold) pipeline for `durations`.
int v3_gpu_max_threads(int durations, int fold) {
    int limit = fold ? FOLD_DURATIONS : MAX_DURATIONS;
    id<MTLComputePipelineState> pipeline =
        durations < 1 || durations > limit ? nil
        : fold                            ? folders[durations - 1]
                                          : windowers[durations - 1];
    return pipeline ? (int)[pipeline maxTotalThreadsPerThreadgroup] : 0;
}

static id<MTLBuffer> copy(const void *data, size_t bytes) {
    if (!bytes) return [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
    return [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
}

// A pass's cadence inputs are copied once per inputs key and reused while it repeats.
static int64_t bound_key = -1;
static size_t bound_bytes[4];
static id<MTLBuffer> bound[4];

static void bind_inputs(int64_t key, const void *const data[4], const size_t bytes[4]) {
    int same = key >= 0 && key == bound_key;
    for (int i = 0; same && i < 4; i++) same = bound[i] && bound_bytes[i] == bytes[i];
    if (same) return;
    for (int i = 0; i < 4; i++) {
        bound[i] = copy(data[i], bytes[i]);
        bound_bytes[i] = bytes[i];
    }
    bound_key = key;
    for (int i = 0; i < 4; i++)
        if (!bound[i]) bound_key = -1;
}

int v3_gpu_screen(int64_t inputs_key, int32_t n, const float *times, int32_t table_size,
                  const int32_t *table,
                  float grid, int32_t n_dur, const void *pairs, int64_t pairs_bytes,
                  const float *factors,
                  int32_t n_periods, const void *periods, int32_t period_bytes, int32_t n_tiles,
                  const void *tiles, int32_t tile_bytes, int32_t n_classes,
                  const int32_t *class_start, const int32_t *class_threads,
                  const int32_t *class_durations, const int32_t *class_ints,
                  int32_t n_groups, const void *groups,
                  int32_t group_bytes, int32_t n_group_classes, const int32_t *group_start,
                  const int32_t *group_durations, int32_t fold_threads, int32_t n_partials,
                  int32_t n_cells, const int32_t *first, const int32_t *length, float *cells,
                  int32_t *flags, float edge, float n_trans, float threshold, double *seconds) {
    static NSObject *guard;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ guard = [NSObject new]; });
    @synchronized(guard) {
    @autoreleasepool {
        if (!windowers[0]) return 1;
        // An oversized threadgroup would be skipped without an error.
        for (int32_t c = 0; c < n_classes; c++) {
            if (class_durations[c] < 1 || class_durations[c] > MAX_DURATIONS) return 4;
            id<MTLComputePipelineState> pipeline = windowers[class_durations[c] - 1];
            if (class_threads[c] < 1 ||
                class_threads[c] > (int32_t)[pipeline maxTotalThreadsPerThreadgroup])
                return 5;
        }
        for (int32_t c = 0; c < n_group_classes; c++) {
            if (group_durations[c] < 1 || group_durations[c] > FOLD_DURATIONS) return 4;
            id<MTLComputePipelineState> pipeline = folders[group_durations[c] - 1];
            if (fold_threads < 1 ||
                fold_threads > (int32_t)[pipeline maxTotalThreadsPerThreadgroup])
                return 5;
        }
        const void *data[4] = {times, table, pairs, factors};
        const size_t bytes[4] = {(size_t)n * 8, (size_t)table_size * 4, (size_t)pairs_bytes,
                                 (size_t)n_dur * 4};
        bind_inputs(inputs_key, data, bytes);
        id<MTLBuffer> bt = bound[0], bb = bound[1], bn = bound[2], bs = bound[3];
        id<MTLBuffer> bp = copy(periods, (size_t)n_periods * period_bytes);
        id<MTLBuffer> bg = copy(tiles, (size_t)n_tiles * tile_bytes);
        id<MTLBuffer> bo = copy(groups, (size_t)n_groups * group_bytes);
        id<MTLBuffer> bf = copy(first, (size_t)n_cells * 4), bl = copy(length, (size_t)n_cells * 4);
        id<MTLBuffer> bq = [device newBufferWithLength:(size_t)(n_partials ? n_partials : 1) * 16
                                               options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bc = [device newBufferWithLength:(size_t)(n_cells ? n_cells : 1) * 16
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> bx = [device newBufferWithLength:(size_t)n_periods * 4
                                               options:MTLResourceStorageModeShared];
        if (!bt || !bb || !bn || !bs || !bp || !bg || !bo || !bf || !bl || !bq || !bc || !bx)
            return 2;
        memset([bx contents], 0, (size_t)n_periods * 4);
        Params params = {n, table_size, grid, edge, n_trans, n_dur, threshold};
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setBuffer:bt offset:0 atIndex:0];
        [encoder setBuffer:bb offset:0 atIndex:1];
        [encoder setBuffer:bn offset:0 atIndex:2];
        [encoder setBuffer:bp offset:0 atIndex:5];
        [encoder setBuffer:bq offset:0 atIndex:6];
        [encoder setBuffer:bx offset:0 atIndex:7];
        [encoder setBytes:&params length:sizeof(params) atIndex:8];
        [encoder setBuffer:bs offset:0 atIndex:9];
        for (int32_t c = 0; c < n_classes; c++) {
            int32_t begin = class_start[c], end = class_start[c + 1];
            if (end <= begin) continue;
            [encoder setComputePipelineState:windowers[class_durations[c] - 1]];
            [encoder setBuffer:bg offset:(size_t)begin * tile_bytes atIndex:4];
            [encoder setThreadgroupMemoryLength:(NSUInteger)class_ints[c] * 4 atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(end - begin, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(class_threads[c], 1, 1)];
        }
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:merger];
        [encoder setBuffer:bq offset:0 atIndex:0];
        [encoder setBuffer:bf offset:0 atIndex:1];
        [encoder setBuffer:bl offset:0 atIndex:2];
        [encoder setBuffer:bc offset:0 atIndex:3];
        [encoder setBytes:&n_cells length:4 atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(n_cells ? n_cells : 1, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        // Fold groups write their cells after the merge; buffers 5, 7, 8 and 9 keep the tiles' binding.
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setBuffer:bt offset:0 atIndex:0];
        [encoder setBuffer:bn offset:0 atIndex:2];
        [encoder setBuffer:bc offset:0 atIndex:10];
        for (int32_t c = 0; c < n_group_classes; c++) {
            int32_t begin = group_start[c], end = group_start[c + 1];
            if (end <= begin) continue;
            [encoder setComputePipelineState:folders[group_durations[c] - 1]];
            [encoder setBuffer:bo offset:(size_t)begin * group_bytes atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(end - begin, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(fold_threads, 1, 1)];
        }
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.error) return 3;
        *seconds = command.GPUEndTime - command.GPUStartTime;
        memcpy(cells, [bc contents], (size_t)n_cells * 16);
        memcpy(flags, [bx contents], (size_t)n_periods * 4);
        return 0;
    }
    }
}
