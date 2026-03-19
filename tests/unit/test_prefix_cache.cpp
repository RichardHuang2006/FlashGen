#include "prefix_cache.hpp"
#include <cassert>
#include <cstdio>
#include <numeric>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  %-50s", name); fflush(stdout)
#define PASS() do { printf("[PASS]\n"); ++tests_passed; } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); ++tests_failed; } while(0)

// ---------------------------------------------------------------------------
//  Prefix cache unit tests
// ---------------------------------------------------------------------------

void test_empty_match() {
    TEST("Empty cache returns no match");

    BlockAllocator alloc(64, 16, 2, 4, 64, KVQuantType::NONE);
    PrefixCache cache(alloc, 16, 2);

    std::vector<int> tokens(48);
    std::iota(tokens.begin(), tokens.end(), 0);

    auto result = cache.match_prefix(tokens);
    assert(result.tokens_matched == 0);
    assert(result.block_indices[0].empty());

    PASS();
}

void test_insert_and_match() {
    TEST("Insert prefix then match");

    BlockAllocator alloc(128, 16, 2, 4, 64, KVQuantType::NONE);
    PrefixCache cache(alloc, 16, 2);

    // Create a 48-token prefix (3 blocks)
    std::vector<int> tokens(48);
    std::iota(tokens.begin(), tokens.end(), 100);

    // Simulate block tables after prefill: 3 blocks per layer
    std::vector<std::vector<PhysicalBlockIdx>> block_tables(2);
    for (int l = 0; l < 2; ++l) {
        auto blocks = alloc.allocate_n(3);
        block_tables[l] = blocks;
    }

    cache.insert(tokens, block_tables);

    // Match with same prefix + different suffix
    std::vector<int> query(64);
    std::iota(query.begin(), query.begin() + 48, 100);  // same prefix
    std::iota(query.begin() + 48, query.end(), 999);    // different suffix

    auto result = cache.match_prefix(query);
    assert(result.tokens_matched == 48);
    assert((int)result.block_indices[0].size() == 3);
    assert((int)result.block_indices[1].size() == 3);

    // Release matched references
    cache.release(query, result.tokens_matched);

    // Clean up
    for (auto& table : block_tables) alloc.free_all(table);

    PASS();
}

void test_partial_match() {
    TEST("Partial prefix match");

    BlockAllocator alloc(128, 16, 1, 1, 64, KVQuantType::NONE);
    PrefixCache cache(alloc, 16, 1);

    // Insert 32-token prefix (2 blocks)
    std::vector<int> tokens(32);
    std::iota(tokens.begin(), tokens.end(), 0);

    std::vector<std::vector<PhysicalBlockIdx>> tables(1);
    tables[0] = alloc.allocate_n(2);
    cache.insert(tokens, tables);

    // Query with 16-token common prefix, then diverges
    std::vector<int> query(32);
    std::iota(query.begin(), query.begin() + 16, 0);   // matches block 0
    std::iota(query.begin() + 16, query.end(), 500);    // diverges at block 1

    auto result = cache.match_prefix(query);
    assert(result.tokens_matched == 16);  // only first block matches
    assert((int)result.block_indices[0].size() == 1);

    cache.release(query, result.tokens_matched);
    alloc.free_all(tables[0]);

    PASS();
}

void test_no_match_different_tokens() {
    TEST("No match for completely different tokens");

    BlockAllocator alloc(64, 16, 1, 1, 64, KVQuantType::NONE);
    PrefixCache cache(alloc, 16, 1);

    std::vector<int> tokens(32);
    std::iota(tokens.begin(), tokens.end(), 0);

    std::vector<std::vector<PhysicalBlockIdx>> tables(1);
    tables[0] = alloc.allocate_n(2);
    cache.insert(tokens, tables);

    // Completely different query
    std::vector<int> query(32);
    std::iota(query.begin(), query.end(), 1000);

    auto result = cache.match_prefix(query);
    assert(result.tokens_matched == 0);

    alloc.free_all(tables[0]);

    PASS();
}

void test_disabled_cache() {
    TEST("Disabled cache returns no match");

    BlockAllocator alloc(64, 16, 1, 1, 64, KVQuantType::NONE);
    PrefixCache cache(alloc, 16, 1);
    cache.set_enabled(false);

    std::vector<int> tokens(16);
    std::iota(tokens.begin(), tokens.end(), 0);

    std::vector<std::vector<PhysicalBlockIdx>> tables(1);
    tables[0] = alloc.allocate_n(1);

    cache.insert(tokens, tables);  // should be ignored

    auto result = cache.match_prefix(tokens);
    assert(result.tokens_matched == 0);

    alloc.free_all(tables[0]);

    PASS();
}

void test_multiple_prefixes() {
    TEST("Multiple distinct prefixes");

    BlockAllocator alloc(256, 16, 1, 1, 64, KVQuantType::NONE);
    PrefixCache cache(alloc, 16, 1);

    // Insert three different 32-token prefixes
    for (int p = 0; p < 3; ++p) {
        std::vector<int> tokens(32);
        std::iota(tokens.begin(), tokens.end(), p * 1000);

        std::vector<std::vector<PhysicalBlockIdx>> tables(1);
        tables[0] = alloc.allocate_n(2);
        cache.insert(tokens, tables);
    }

    // Match each one
    for (int p = 0; p < 3; ++p) {
        std::vector<int> query(48);
        std::iota(query.begin(), query.begin() + 32, p * 1000);  // known prefix
        std::iota(query.begin() + 32, query.end(), 9000);         // unique suffix

        auto result = cache.match_prefix(query);
        assert(result.tokens_matched == 32);
        cache.release(query, result.tokens_matched);
    }

    PASS();
}

void test_short_tokens_no_match() {
    TEST("Tokens shorter than block_size yield no match");

    BlockAllocator alloc(32, 16, 1, 1, 64, KVQuantType::NONE);
    PrefixCache cache(alloc, 16, 1);

    std::vector<int> tokens(10);  // less than block_size
    std::iota(tokens.begin(), tokens.end(), 0);

    auto result = cache.match_prefix(tokens);
    assert(result.tokens_matched == 0);

    PASS();
}

int main() {
    printf("═══ Prefix Cache Tests ═══\n\n");

    test_empty_match();
    test_insert_and_match();
    test_partial_match();
    test_no_match_different_tokens();
    test_disabled_cache();
    test_multiple_prefixes();
    test_short_tokens_no_match();

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
