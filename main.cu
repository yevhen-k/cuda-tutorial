#include <iostream>
#include "add_vectors/baseline/add_vectors.h"
#include "add_vectors/pinned_memory/add_vec_pinned_mem.h"

int main() {
    add_vectors();
    add_vec_pinned_mem();
    std::cout << "--- END ---" << std::endl;
    return 0;
}
