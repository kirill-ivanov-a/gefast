find .. \( -path "../include/*" -o -path "../src/*" -o -path "../tests/*" \) -a \( -path "*.h" -o -path "*.cpp" \) -print | xargs clang-format -i
