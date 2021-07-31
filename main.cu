#include <iostream>
#include "add_vectors/baseline/add_vectors.h"
#include "add_vectors/pinned_memory/add_vec_pinned_mem.h"
#include "add_vectors/unified_memory/unified_memory.h"

int main() {
    add_vectors();
    add_vec_pinned_mem();
    add_vec_unified_memory_baseline();
    add_vec_unified_memory_prefetch();
    std::cout << "--- END ---" << std::endl;
    return 0;
}
