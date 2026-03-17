#include "block_allocator.hpp"
#include <cassert>
#include <cstdio>
#include <set>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    printf("  %-50s", name); \
    fflush(stdout);

#define PASS() \
    do { printf("[PASS]\n"); ++tests_passed; } while(0)

#define FAIL(msg) \
    do { printf("[FAIL] %s\n", msg); ++tests_failed; } while(0)

// ---------------------------------------------------------------------------
//  Block allocator unit tests
// ---------------------------------------------------------------------------

void test_basic_alloc_free() {
    TEST("Basic allocate and free");

    BlockAllocator alloc(64, 16, 2, 4, 64, KVQuantType::NONE);
    assert(alloc.num_total() == 64);
    assert(alloc.num_free() == 64);

    auto b0 = alloc.allocate();
    assert(b0 != kNullBlock);
    assert(alloc.num_free() == 63);
    assert(alloc.ref_count(b0) == 1);

    alloc.free(b0);
    assert(alloc.num_free() == 64);

    PASS();
}

void test_bulk_allocation() {
    TEST("Bulk allocate_n");

    BlockAllocator alloc(32, 16, 1, 1, 64, KVQuantType::NONE);

    auto blocks = alloc.allocate_n(10);
    assert((int)blocks.size() == 10);
    assert(alloc.num_free() == 22);

    // Verify unique indices
    std::set<PhysicalBlockIdx> unique(blocks.begin(), blocks.end());
    assert((int)unique.size() == 10);

    alloc.free_all(blocks);
    assert(alloc.num_free() == 32);

    PASS();
}

void test_oom_handling() {
    TEST("OOM returns kNullBlock / empty vector");

    BlockAllocator alloc(4, 16, 1, 1, 64, KVQuantType::NONE);

    // Allocate all blocks
    std::vector<PhysicalBlockIdx> all;
    for (int i = 0; i < 4; ++i) {
        auto b = alloc.allocate();
        assert(b != kNullBlock);
        all.push_back(b);
    }
    assert(alloc.num_free() == 0);

    // Next allocation should fail
    auto b = alloc.allocate();
    assert(b == kNullBlock);

    auto batch = alloc.allocate_n(2);
    assert(batch.empty());

    alloc.free_all(all);
    assert(alloc.num_free() == 4);

    PASS();
}

void test_ref_counting() {
    TEST("Reference counting");

    BlockAllocator alloc(8, 16, 1, 1, 64, KVQuantType::NONE);

    auto b = alloc.allocate();
    assert(alloc.ref_count(b) == 1);

    alloc.add_ref(b);
    assert(alloc.ref_count(b) == 2);
    assert(alloc.num_free() == 7);  // still 7 free, not freed yet

    alloc.free(b);
    assert(alloc.ref_count(b) == 1);
    assert(alloc.num_free() == 7);  // still held by one ref

    alloc.free(b);
    assert(alloc.num_free() == 8);  // now freed

    PASS();
}

void test_cow_copy() {
    TEST("Copy-on-write");

    BlockAllocator alloc(8, 16, 1, 1, 64, KVQuantType::NONE);
    CudaStream stream;

    auto b = alloc.allocate();
    assert(alloc.ref_count(b) == 1);

    // CoW with refcount=1 should return same block
    auto b2 = alloc.cow_copy(b, stream);
    assert(b2 == b);
    assert(alloc.num_free() == 7);

    // Add ref, then CoW should create a new block
    alloc.add_ref(b);
    assert(alloc.ref_count(b) == 2);

    auto b3 = alloc.cow_copy(b, stream);
    stream.sync();

    assert(b3 != b);
    assert(b3 != kNullBlock);
    assert(alloc.ref_count(b) == 1);   // original decremented
    assert(alloc.ref_count(b3) == 1);  // new block

    alloc.free(b);
    alloc.free(b3);
    assert(alloc.num_free() == 8);

    PASS();
}

void test_utilization() {
    TEST("Utilization tracking");

    BlockAllocator alloc(10, 16, 1, 1, 64, KVQuantType::NONE);

    assert(alloc.utilization() == 0.f);

    auto blocks = alloc.allocate_n(5);
    float u = alloc.utilization();
    assert(u > 0.49f && u < 0.51f);  // ~0.5

    alloc.free_all(blocks);
    assert(alloc.utilization() == 0.f);

    PASS();
}

void test_block_ptr() {
    TEST("Block pointer computation");

    BlockAllocator alloc(4, 16, 2, 4, 64, KVQuantType::NONE);

    auto b0 = alloc.allocate();
    auto b1 = alloc.allocate();

    void* p0 = alloc.block_ptr(b0);
    void* p1 = alloc.block_ptr(b1);

    // Blocks should be at different addresses
    assert(p0 != p1);

    // Layer/head offsets should be within block
    void* p0_l0_h0_k = alloc.block_ptr(b0, 0, 0, false);
    void* p0_l0_h0_v = alloc.block_ptr(b0, 0, 0, true);
    assert(p0_l0_h0_k == p0);
    assert(p0_l0_h0_v != p0_l0_h0_k);

    alloc.free(b0);
    alloc.free(b1);

    PASS();
}

void test_null_block_handling() {
    TEST("Null block operations are safe");

    BlockAllocator alloc(4, 16, 1, 1, 64, KVQuantType::NONE);

    // free(kNullBlock) should be no-op
    alloc.free(kNullBlock);
    assert(alloc.num_free() == 4);

    // ref_count(kNullBlock) should return 0
    assert(alloc.ref_count(kNullBlock) == 0);

    PASS();
}

int main() {
    printf("═══ Block Allocator Tests ═══\n\n");

    test_basic_alloc_free();
    test_bulk_allocation();
    test_oom_handling();
    test_ref_counting();
    test_cow_copy();
    test_utilization();
    test_block_ptr();
    test_null_block_handling();

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
