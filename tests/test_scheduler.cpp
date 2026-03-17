#include "scheduler.hpp"
#include <cassert>
#include <cstdio>
#include <memory>
#include <numeric>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  %-50s", name); fflush(stdout)
#define PASS() do { printf("[PASS]\n"); ++tests_passed; } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); ++tests_failed; } while(0)

static std::shared_ptr<SequenceGroup> make_request(int id, int prompt_len) {
    auto sg = std::make_shared<SequenceGroup>();
    sg->request_id = id;
    sg->is_prefill = true;

    SequenceData seq;
    seq.prompt_tokens.resize(prompt_len);
    std::iota(seq.prompt_tokens.begin(), seq.prompt_tokens.end(), id * 1000);
    seq.status = SequenceStatus::WAITING;
    sg->sequences.push_back(std::move(seq));

    sg->sampling.max_tokens = 10;
    return sg;
}

// ---------------------------------------------------------------------------
//  Scheduler unit tests
// ---------------------------------------------------------------------------

void test_add_and_schedule_single() {
    TEST("Schedule single request");

    ModelConfig mcfg;
    mcfg.n_layers = 2;
    mcfg.n_heads = 4;
    mcfg.d_model = 256;

    SchedulerConfig scfg;
    scfg.max_num_seqs = 8;
    scfg.max_num_tokens = 1024;

    BlockAllocator alloc(256, 16, mcfg.n_layers, mcfg.actual_kv_heads(),
                         mcfg.head_dim(), KVQuantType::NONE);
    PagedKVCache kv_cache(alloc, mcfg);
    PrefixCache prefix_cache(alloc, 16, mcfg.n_layers);

    Scheduler sched(scfg, alloc, kv_cache, prefix_cache);

    sched.add_request(make_request(0, 32));
    assert(sched.num_waiting() == 1);

    auto output = sched.schedule();
    assert((int)output.scheduled_prefills.size() == 1);
    assert(output.num_prefill_tokens == 32);
    assert(sched.num_running() == 1);
    assert(sched.num_waiting() == 0);

    PASS();
}

void test_multiple_requests_batched() {
    TEST("Multiple requests batched together");

    ModelConfig mcfg;
    mcfg.n_layers = 1;
    mcfg.n_heads = 2;
    mcfg.d_model = 128;

    SchedulerConfig scfg;
    scfg.max_num_seqs = 16;
    scfg.max_num_tokens = 2048;

    BlockAllocator alloc(512, 16, mcfg.n_layers, mcfg.actual_kv_heads(),
                         mcfg.head_dim(), KVQuantType::NONE);
    PagedKVCache kv_cache(alloc, mcfg);
    PrefixCache prefix_cache(alloc, 16, mcfg.n_layers);

    Scheduler sched(scfg, alloc, kv_cache, prefix_cache);

    // Add 4 requests
    for (int i = 0; i < 4; ++i) {
        sched.add_request(make_request(i, 64));
    }

    auto output = sched.schedule();
    assert((int)output.scheduled_prefills.size() == 4);
    assert(output.num_prefill_tokens == 256);  // 4 * 64

    PASS();
}

void test_token_budget_limit() {
    TEST("Token budget limits batch size");

    ModelConfig mcfg;
    mcfg.n_layers = 1;
    mcfg.n_heads = 1;
    mcfg.d_model = 64;

    SchedulerConfig scfg;
    scfg.max_num_seqs = 100;
    scfg.max_num_tokens = 100;  // very limited
    scfg.max_prefill_tokens = 100;

    BlockAllocator alloc(256, 16, mcfg.n_layers, mcfg.actual_kv_heads(),
                         mcfg.head_dim(), KVQuantType::NONE);
    PagedKVCache kv_cache(alloc, mcfg);
    PrefixCache prefix_cache(alloc, 16, mcfg.n_layers);

    Scheduler sched(scfg, alloc, kv_cache, prefix_cache);

    // Add 3 requests of 64 tokens each (total 192 > budget 100)
    for (int i = 0; i < 3; ++i) {
        sched.add_request(make_request(i, 64));
    }

    auto output = sched.schedule();
    // Should only schedule 1 (64 <= 100, but 128 > 100)
    assert((int)output.scheduled_prefills.size() == 1);
    assert(sched.num_waiting() == 2);

    PASS();
}

void test_decode_after_prefill() {
    TEST("Decode phase after prefill");

    ModelConfig mcfg;
    mcfg.n_layers = 1;
    mcfg.n_heads = 2;
    mcfg.d_model = 128;

    SchedulerConfig scfg;
    scfg.max_num_seqs = 8;
    scfg.max_num_tokens = 1024;

    BlockAllocator alloc(256, 16, mcfg.n_layers, mcfg.actual_kv_heads(),
                         mcfg.head_dim(), KVQuantType::NONE);
    PagedKVCache kv_cache(alloc, mcfg);
    PrefixCache prefix_cache(alloc, 16, mcfg.n_layers);

    Scheduler sched(scfg, alloc, kv_cache, prefix_cache);

    sched.add_request(make_request(0, 32));

    // Step 1: prefill
    auto out1 = sched.schedule();
    assert(!out1.scheduled_prefills.empty());
    assert(out1.scheduled_decodes.empty());

    // Simulate prefill completing and generating a token
    sched.post_step({out1.scheduled_prefills[0]}, {42});

    // Step 2: should now be in decode mode
    auto out2 = sched.schedule();
    assert(out2.scheduled_prefills.empty());
    assert(!out2.scheduled_decodes.empty());

    PASS();
}

void test_finish_on_eos() {
    TEST("Sequence finishes on EOS token");

    ModelConfig mcfg;
    mcfg.n_layers = 1;
    mcfg.n_heads = 1;
    mcfg.d_model = 64;

    SchedulerConfig scfg;
    scfg.max_num_seqs = 4;
    scfg.max_num_tokens = 512;

    BlockAllocator alloc(128, 16, mcfg.n_layers, mcfg.actual_kv_heads(),
                         mcfg.head_dim(), KVQuantType::NONE);
    PagedKVCache kv_cache(alloc, mcfg);
    PrefixCache prefix_cache(alloc, 16, mcfg.n_layers);

    Scheduler sched(scfg, alloc, kv_cache, prefix_cache);

    auto req = make_request(0, 16);
    req->sampling.stop_token_ids = {50256};
    sched.add_request(req);

    // Prefill
    auto out1 = sched.schedule();
    sched.post_step({out1.scheduled_prefills[0]}, {100});

    // Decode with EOS token
    auto out2 = sched.schedule();
    sched.post_step({out2.scheduled_decodes[0]}, {50256});

    // Should be finished now
    assert(sched.num_running() == 0);
    assert(!sched.has_pending());

    PASS();
}

void test_finish_on_max_tokens() {
    TEST("Sequence finishes on max_tokens");

    ModelConfig mcfg;
    mcfg.n_layers = 1;
    mcfg.n_heads = 1;
    mcfg.d_model = 64;

    SchedulerConfig scfg;
    scfg.max_num_seqs = 4;
    scfg.max_num_tokens = 512;

    BlockAllocator alloc(128, 16, mcfg.n_layers, mcfg.actual_kv_heads(),
                         mcfg.head_dim(), KVQuantType::NONE);
    PagedKVCache kv_cache(alloc, mcfg);
    PrefixCache prefix_cache(alloc, 16, mcfg.n_layers);

    Scheduler sched(scfg, alloc, kv_cache, prefix_cache);

    auto req = make_request(0, 16);
    req->sampling.max_tokens = 3;
    sched.add_request(req);

    // Prefill
    auto out1 = sched.schedule();
    sched.post_step({out1.scheduled_prefills[0]}, {1});

    // Decode steps
    for (int i = 0; i < 2; ++i) {
        auto out = sched.schedule();
        if (out.scheduled_decodes.empty()) break;
        sched.post_step({out.scheduled_decodes[0]}, {2});
    }

    assert(sched.num_running() == 0);

    PASS();
}

void test_abort_request() {
    TEST("Abort request removes from queue");

    ModelConfig mcfg;
    mcfg.n_layers = 1;
    mcfg.n_heads = 1;
    mcfg.d_model = 64;

    SchedulerConfig scfg;
    BlockAllocator alloc(64, 16, mcfg.n_layers, mcfg.actual_kv_heads(),
                         mcfg.head_dim(), KVQuantType::NONE);
    PagedKVCache kv_cache(alloc, mcfg);
    PrefixCache prefix_cache(alloc, 16, mcfg.n_layers);

    Scheduler sched(scfg, alloc, kv_cache, prefix_cache);

    sched.add_request(make_request(0, 16));
    sched.add_request(make_request(1, 16));
    assert(sched.num_waiting() == 2);

    sched.abort_request(0);
    assert(sched.num_waiting() == 1);

    PASS();
}

int main() {
    printf("═══ Scheduler Tests ═══\n\n");

    test_add_and_schedule_single();
    test_multiple_requests_batched();
    test_token_budget_limit();
    test_decode_after_prefill();
    test_finish_on_eos();
    test_finish_on_max_tokens();
    test_abort_request();

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
