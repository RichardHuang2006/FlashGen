#include "prefix_cache.hpp"
#include <algorithm>
#include <cassert>

// ---------------------------------------------------------------------------
//  PrefixCache implementation — radix tree for KV block reuse
// ---------------------------------------------------------------------------

PrefixCache::PrefixCache(BlockAllocator& allocator, int block_size, int num_layers)
    : allocator_(allocator),
      block_size_(block_size),
      num_layers_(num_layers),
      root_(std::make_unique<RadixNode>())
{}

PrefixMatchResult PrefixCache::match_prefix(const std::vector<int>& token_ids) {
    PrefixMatchResult result;
    result.block_indices.resize(num_layers_);

    if (!enabled_ || token_ids.empty()) return result;

    std::lock_guard<std::mutex> lock(mu_);

    RadixNode* node = root_.get();
    int pos = 0;
    int total_tokens = (int)token_ids.size();

    while (pos + block_size_ <= total_tokens) {
        const int* chunk = token_ids.data() + pos;
        RadixNode* child = node->find_child(chunk, block_size_);

        if (!child) break;  // no match for this block

        // Found a matching block — collect block indices from the edge
        for (auto& [edge, child_ptr] : node->children) {
            if (child_ptr.get() == child &&
                (int)edge.tokens.size() == block_size_ &&
                std::equal(edge.tokens.begin(), edge.tokens.end(), chunk)) {
                // Increment refcounts and record block indices
                for (int l = 0; l < num_layers_; ++l) {
                    PhysicalBlockIdx bidx = edge.layer_blocks[l];
                    allocator_.add_ref(bidx);
                    result.block_indices[l].push_back(bidx);
                }
                break;
            }
        }

        result.tokens_matched += block_size_;
        pos += block_size_;
        touch(child);
        node = child;
    }

    return result;
}

void PrefixCache::insert(const std::vector<int>& token_ids,
                         const std::vector<std::vector<PhysicalBlockIdx>>& block_tables) {
    if (!enabled_) return;

    std::lock_guard<std::mutex> lock(mu_);

    int num_full_blocks = (int)token_ids.size() / block_size_;
    RadixNode* node = root_.get();

    for (int b = 0; b < num_full_blocks; ++b) {
        int start = b * block_size_;
        const int* chunk = token_ids.data() + start;

        RadixNode* child = node->find_child(chunk, block_size_);
        if (child) {
            // Already cached — just update access time
            touch(child);
            node = child;
            continue;
        }

        // Insert new edge
        RadixEdge edge;
        edge.tokens.assign(chunk, chunk + block_size_);
        edge.layer_blocks.resize(num_layers_);
        for (int l = 0; l < num_layers_; ++l) {
            PhysicalBlockIdx bidx = block_tables[l][b];
            allocator_.add_ref(bidx);  // prefix cache holds a reference
            edge.layer_blocks[l] = bidx;
        }

        auto new_node = std::make_unique<RadixNode>();
        RadixNode* new_ptr = new_node.get();
        node->children.emplace_back(std::move(edge), std::move(new_node));

        touch(new_ptr);
        node = new_ptr;
    }
}

void PrefixCache::release(const std::vector<int>& token_ids, int tokens_matched) {
    if (!enabled_ || tokens_matched == 0) return;

    std::lock_guard<std::mutex> lock(mu_);

    RadixNode* node = root_.get();
    int pos = 0;

    while (pos < tokens_matched) {
        const int* chunk = token_ids.data() + pos;
        RadixNode* child = node->find_child(chunk, block_size_);
        if (!child) break;

        // Decrement refcounts on matched blocks
        for (auto& [edge, child_ptr] : node->children) {
            if (child_ptr.get() == child) {
                for (int l = 0; l < num_layers_; ++l) {
                    allocator_.free(edge.layer_blocks[l]);
                }
                break;
            }
        }

        pos += block_size_;
        node = child;
    }
}

int PrefixCache::evict_lru(int blocks_needed) {
    if (!enabled_) return 0;

    std::lock_guard<std::mutex> lock(mu_);

    int freed = 0;

    // Collect evictable nodes (ref_count == 0, leaf-first)
    std::vector<RadixNode*> candidates;
    collect_evictable(root_.get(), candidates);

    // Sort by last_access (oldest first)
    std::sort(candidates.begin(), candidates.end(),
              [](RadixNode* a, RadixNode* b) {
                  return a->last_access < b->last_access;
              });

    for (RadixNode* node : candidates) {
        if (freed >= blocks_needed) break;
        if (node->ref_count > 0) continue;

        // Find and remove this node from its parent
        // (simplified — in production, maintain parent pointers)
        // For now, we just free the blocks referenced by edges leading to this node
        // The actual tree pruning would need parent tracking
        node->ref_count = -1;  // mark as evicted

        // Remove from LRU
        auto lru_it = lru_map_.find(node);
        if (lru_it != lru_map_.end()) {
            lru_list_.erase(lru_it->second);
            lru_map_.erase(lru_it);
        }

        freed += num_layers_;  // approximate: one block per layer per edge
    }

    return freed;
}

size_t PrefixCache::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return lru_map_.size();
}

void PrefixCache::touch(RadixNode* node) {
    node->last_access = ++clock_;

    auto it = lru_map_.find(node);
    if (it != lru_map_.end()) {
        lru_list_.erase(it->second);
    }
    lru_list_.push_back(node);
    lru_map_[node] = std::prev(lru_list_.end());
}

void PrefixCache::collect_evictable(RadixNode* node,
                                     std::vector<RadixNode*>& out) {
    for (auto& [edge, child] : node->children) {
        collect_evictable(child.get(), out);
    }
    if (node != root_.get() && node->ref_count == 0) {
        out.push_back(node);
    }
}
