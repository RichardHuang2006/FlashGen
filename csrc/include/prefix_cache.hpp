#pragma once

#include "block_allocator.hpp"

#include <chrono>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
//  Prefix cache — radix-tree-based KV cache reuse
//
//  Common prompt prefixes (system prompts, few-shot examples) generate
//  identical KV cache blocks. The prefix cache stores a radix tree
//  indexed by token sequences where each node represents one block's
//  worth of tokens (block_size tokens per edge).
//
//  On a new request:
//    1. match_prefix() traverses the tree to find the longest matching
//       prefix, returning the physical block indices for reuse.
//    2. Only the unmatched suffix needs prefill computation.
//    3. After prefill, insert() registers the new blocks for future reuse.
//
//  Memory pressure is handled by LRU eviction of unreferenced prefix nodes.
// ---------------------------------------------------------------------------

struct RadixEdge {
    std::vector<int>                tokens;         // block_size tokens per edge
    std::vector<PhysicalBlockIdx>   layer_blocks;   // one block per layer
};

struct RadixNode {
    std::vector<std::pair<RadixEdge, std::unique_ptr<RadixNode>>> children;
    int      ref_count   = 0;         // active sequences using this prefix
    int64_t  last_access = 0;         // monotonic clock for LRU

    RadixNode* find_child(const int* tokens, int count) {
        for (auto& [edge, child] : children) {
            if ((int)edge.tokens.size() == count &&
                std::equal(edge.tokens.begin(), edge.tokens.end(), tokens))
                return child.get();
        }
        return nullptr;
    }
};

struct PrefixMatchResult {
    int tokens_matched = 0;
    // block_indices[layer][logical_block] = physical block idx
    std::vector<std::vector<PhysicalBlockIdx>> block_indices;
};

class PrefixCache {
public:
    PrefixCache(BlockAllocator& allocator, int block_size, int num_layers);

    /// Find the longest cached prefix for the given token sequence.
    /// Increments refcounts on matched blocks so they won't be evicted.
    PrefixMatchResult match_prefix(const std::vector<int>& token_ids);

    /// Register blocks after prefill so future requests can reuse them.
    /// token_ids: full prompt token sequence.
    /// block_tables: [num_layers][num_logical_blocks] physical block indices.
    void insert(const std::vector<int>& token_ids,
                const std::vector<std::vector<PhysicalBlockIdx>>& block_tables);

    /// Release references obtained from match_prefix().
    void release(const std::vector<int>& token_ids, int tokens_matched);

    /// Evict least-recently-used entries to free at least `blocks_needed` blocks.
    /// Returns actual number of blocks freed.
    int evict_lru(int blocks_needed);

    /// Number of prefix entries (nodes with blocks).
    size_t size() const;

    /// Whether prefix caching is effectively enabled.
    bool enabled() const { return enabled_; }

    void set_enabled(bool e) { enabled_ = e; }

private:
    BlockAllocator&                allocator_;
    int                            block_size_;
    int                            num_layers_;
    bool                           enabled_ = true;
    std::unique_ptr<RadixNode>     root_;
    int64_t                        clock_ = 0;
    mutable std::mutex             mu_;

    // LRU list of evictable nodes (leaf-first)
    std::list<RadixNode*>          lru_list_;
    std::unordered_map<RadixNode*, std::list<RadixNode*>::iterator> lru_map_;

    void touch(RadixNode* node);
    void collect_evictable(RadixNode* node, std::vector<RadixNode*>& out);
};
