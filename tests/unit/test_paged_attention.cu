#include "paged_kv_cache.cuh"
#include "flash_attention.cuh"
#include "cuda_utils.cuh"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  %-50s", name); fflush(stdout)
#define PASS() do { printf("[PASS]\n"); ++tests_passed; } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); ++tests_failed; } while(0)

// Reference naive attention for correctness checking
void naive_attention_ref(const float* Q, const float* K, const float* V,
                         float* O, int seq_len, int head_dim, bool causal) {
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int i = 0; i < seq_len; ++i) {
        float max_val = -1e30f;
        for (int j = 0; j <= (causal ? i : seq_len - 1); ++j) {
            float dot = 0.f;
            for (int d = 0; d < head_dim; ++d)
                dot += Q[i * head_dim + d] * K[j * head_dim + d];
            dot *= scale;
            if (dot > max_val) max_val = dot;
        }

        float sum = 0.f;
        std::vector<float> weights(seq_len, 0.f);
        for (int j = 0; j <= (causal ? i : seq_len - 1); ++j) {
            float dot = 0.f;
            for (int d = 0; d < head_dim; ++d)
                dot += Q[i * head_dim + d] * K[j * head_dim + d];
            weights[j] = expf(dot * scale - max_val);
            sum += weights[j];
        }

        for (int d = 0; d < head_dim; ++d) {
            float val = 0.f;
            for (int j = 0; j <= (causal ? i : seq_len - 1); ++j) {
                val += (weights[j] / sum) * V[j * head_dim + d];
            }
            O[i * head_dim + d] = val;
        }
    }
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

void test_paged_kv_cache_lifecycle() {
    TEST("PagedKVCache allocate/extend/free");

    ModelConfig cfg = ModelConfig::gpt2();
    int block_size = 16;
    BlockAllocator alloc(256, block_size, cfg.n_layers, cfg.actual_kv_heads(),
                         cfg.head_dim(), KVQuantType::NONE);
    PagedKVCache cache(alloc, cfg);

    // Allocate for prefill
    bool ok = cache.allocate_for_prefill(/*seq_id=*/0, /*prompt_len=*/48);
    assert(ok);
    assert(cache.has_sequence(0));
    assert(cache.sequence_length(0) == 48);

    // 48 tokens / 16 block_size = 3 blocks per layer
    auto& state = cache.get_state(0);
    assert((int)state.block_tables[0].size() == 3);

    // Extend by one token (still fits in last block)
    ok = cache.extend_one_token(0);
    assert(ok);
    assert(cache.sequence_length(0) == 49);

    // Extend to boundary: 49->64 (fills block 3, needs block 4 at token 65)
    for (int i = 0; i < 15; ++i) cache.extend_one_token(0);
    assert(cache.sequence_length(0) == 64);
    assert((int)cache.get_state(0).block_tables[0].size() == 4);

    // One more should allocate a new block
    cache.extend_one_token(0);
    assert(cache.sequence_length(0) == 65);
    assert((int)cache.get_state(0).block_tables[0].size() == 5);

    // Free
    int free_before = alloc.num_free();
    cache.free_sequence(0);
    int free_after = alloc.num_free();
    assert(free_after > free_before);
    assert(!cache.has_sequence(0));

    PASS();
}

void test_paged_kv_cache_fork() {
    TEST("PagedKVCache fork sequence");

    ModelConfig cfg;
    cfg.n_layers = 2;
    cfg.n_heads = 4;
    cfg.d_model = 256;

    int block_size = 16;
    BlockAllocator alloc(128, block_size, cfg.n_layers, cfg.actual_kv_heads(),
                         cfg.head_dim(), KVQuantType::NONE);
    PagedKVCache cache(alloc, cfg);

    cache.allocate_for_prefill(0, 32);  // 2 blocks per layer

    // Fork
    bool ok = cache.fork_sequence(0, 1);
    assert(ok);
    assert(cache.has_sequence(1));
    assert(cache.sequence_length(1) == 32);

    // Shared blocks should have refcount = 2
    auto& s0 = cache.get_state(0);
    auto& s1 = cache.get_state(1);
    assert(s0.block_tables[0][0] == s1.block_tables[0][0]);
    assert(alloc.ref_count(s0.block_tables[0][0]) == 2);

    // Free one sequence — shared blocks should remain
    cache.free_sequence(1);
    assert(alloc.ref_count(s0.block_tables[0][0]) == 1);

    cache.free_sequence(0);

    PASS();
}

void test_flash_attention_correctness() {
    TEST("Standard FlashAttention vs naive reference");

    int batch = 1, heads = 2, seq = 64, hd = 64;
    int total = batch * heads * seq * hd;

    // Allocate host data
    std::vector<float> h_Q(total), h_K(total), h_V(total);
    std::vector<float> h_O(total, 0.f), h_ref(total, 0.f);

    srand(42);
    for (int i = 0; i < total; ++i) {
        h_Q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        h_K[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        h_V[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
    }

    // GPU computation
    DeviceBuffer<float> d_Q(total), d_K(total), d_V(total), d_O(total);
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    CudaStream stream;
    flash_attention::forward(d_Q, d_K, d_V, d_O, nullptr,
                             batch, heads, seq, hd, true, stream);
    stream.sync();

    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, total * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            int offset = (b * heads + h) * seq * hd;
            naive_attention_ref(h_Q.data() + offset,
                               h_K.data() + offset,
                               h_V.data() + offset,
                               h_ref.data() + offset,
                               seq, hd, true);
        }
    }

    // Compare
    float max_err = 0.f;
    for (int i = 0; i < total; ++i) {
        float err = fabsf(h_O[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }

    if (max_err < 1e-2f) {
        printf("(max_err=%.6f) ", max_err);
        PASS();
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "max_err=%.6f > 1e-2", max_err);
        FAIL(msg);
    }
}

void test_paged_attention_decode() {
    TEST("Paged attention decode kernel basic");

    // Minimal test: 1 seq, 1 head, short context
    ModelConfig cfg;
    cfg.n_layers = 1;
    cfg.n_heads = 1;
    cfg.d_model = 64;
    int block_size = 16;
    int context_len = 32;

    BlockAllocator alloc(64, block_size, cfg.n_layers, cfg.actual_kv_heads(),
                         cfg.head_dim(), KVQuantType::NONE);
    PagedKVCache cache(alloc, cfg);

    bool ok = cache.allocate_for_prefill(0, context_len);
    assert(ok);

    // Write known K/V data into cache blocks
    // For simplicity, write identity-like patterns
    auto& state = cache.get_state(0);
    for (auto& table : state.block_tables) {
        for (int b = 0; b < (int)table.size(); ++b) {
            PhysicalBlockIdx phys = table[b];
            void* ptr = alloc.block_ptr(phys);
            // Fill with known values
            int tokens_in_block = std::min(block_size, context_len - b * block_size);
            std::vector<float> data(block_size * cfg.head_dim() * 2, 0.1f);
            CUDA_CHECK(cudaMemcpy(ptr, data.data(),
                                  data.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
    }

    // Prepare query
    DeviceBuffer<float> d_Q(cfg.head_dim());
    DeviceBuffer<float> d_O(cfg.head_dim());
    std::vector<float> h_Q(cfg.head_dim(), 0.5f);
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), cfg.head_dim() * sizeof(float),
                          cudaMemcpyHostToDevice));

    CudaStream stream;
    auto params = cache.prepare_batch({0}, 0, stream);

    paged_attention::decode(d_Q, d_O, params, cfg.n_heads, 0, stream);
    stream.sync();

    // Read back output
    std::vector<float> h_O(cfg.head_dim());
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, cfg.head_dim() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Output should be finite and non-zero (since we wrote non-zero K/V)
    bool all_finite = true;
    bool any_nonzero = false;
    for (float v : h_O) {
        if (!std::isfinite(v)) all_finite = false;
        if (fabsf(v) > 1e-10f) any_nonzero = true;
    }

    if (all_finite && any_nonzero) {
        PASS();
    } else {
        FAIL("Output not finite or all zero");
    }

    cache.free_sequence(0);
}

void test_prepare_batch_multiseq() {
    TEST("Prepare batch with multiple sequences");

    ModelConfig cfg;
    cfg.n_layers = 2;
    cfg.n_heads = 4;
    cfg.d_model = 256;
    int block_size = 16;

    BlockAllocator alloc(128, block_size, cfg.n_layers, cfg.actual_kv_heads(),
                         cfg.head_dim(), KVQuantType::NONE);
    PagedKVCache cache(alloc, cfg);

    // Different length sequences
    cache.allocate_for_prefill(0, 24);
    cache.allocate_for_prefill(1, 48);
    cache.allocate_for_prefill(2, 16);

    CudaStream stream;
    auto params = cache.prepare_batch({0, 1, 2}, 0, stream);
    stream.sync();

    assert(params.num_seqs == 3);
    assert(params.max_blocks_per_seq >= 3);  // seq 1 needs 3 blocks
    assert(params.block_size == block_size);
    assert(params.kv_pool != nullptr);

    cache.free_sequence(0);
    cache.free_sequence(1);
    cache.free_sequence(2);

    PASS();
}

int main() {
    printf("═══ Paged Attention Tests ═══\n\n");

    test_paged_kv_cache_lifecycle();
    test_paged_kv_cache_fork();
    test_flash_attention_correctness();
    test_paged_attention_decode();
    test_prepare_batch_multiseq();

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
