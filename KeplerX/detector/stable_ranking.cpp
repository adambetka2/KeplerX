// Sort values with their positions; the position breaks ties exactly as a stable sort.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

struct Record { double value; int32_t index; };

extern "C" int rank_pairs(int64_t n, const double *values, int32_t *rank, double *ordered) {
    if (n < 0 || n > INT32_MAX) return 1;
    Record *work = static_cast<Record *>(std::malloc((n ? n : 1) * sizeof(Record)));
    if (!work) return 2;
    for (int64_t i = 0; i < n; ++i) {
        if (!std::isfinite(values[i])) { std::free(work); return 1; }
        work[i] = {values[i], static_cast<int32_t>(i)};
    }
    std::sort(work, work + n, [](const Record &a, const Record &b) {
        return a.value < b.value || (!(b.value < a.value) && a.index < b.index);
    });
    for (int64_t i = 0; i < n; ++i) {
        rank[work[i].index] = static_cast<int32_t>(i);
        ordered[i] = work[i].value;
    }
    std::free(work);
    return 0;
}
