#include <iostream>
#include "debugbreak.h"

int main() {
    std::cout << "Before the breakpoint\n";
    std::cout << "After the breakpoint\n"; // This line won't be reached until the breakpoint is handled
    return 0;
}