#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <chrono>

namespace py = pybind11;

namespace {

constexpr double kKeyScale = 1e8;  // matches rounding to 8 decimals in Python

struct VectorHash {
    std::size_t operator()(const std::vector<long long>& v) const noexcept {
        // FNV-1a 64-bit hash
        std::size_t hash = 1469598103934665603ull;
        for (long long x : v) {
            hash ^= static_cast<std::size_t>(x);
            hash *= 1099511628211ull;
        }
        return hash;
    }
};

inline std::vector<long long> make_key(const std::vector<double>& nums) {
    std::vector<long long> key;
    key.reserve(nums.size());
    for (double x : nums) {
        // Round to 8 decimals and convert to integer key
        long long k = llround(x * kKeyScale);
        key.push_back(k);
    }
    std::sort(key.begin(), key.end());
    return key;
}

struct DfsSolver {
    double target;
    double tol;
    bool use_deadline{false};
    std::chrono::steady_clock::time_point deadline;
    long long max_expansions{-1};
    long long expansions{0};

    // Memo: key -> (can_merge, actions)
    std::unordered_map<std::vector<long long>, std::pair<bool, long long>, VectorHash> memo;

    std::pair<bool, long long> dfs_keyed(const std::vector<long long>& key) {
        auto it = memo.find(key);
        if (it != memo.end()) return it->second;

        // Reconstruct values from key for operations? Not needed; we only need values for computations.
        // But key loses exact double values. We therefore carry doubles alongside and only key the memo with rounded values.
        // We'll reconstruct a vector<double> from the key by dividing by kKeyScale.
        std::vector<double> state;
        state.reserve(key.size());
        for (long long k : key) {
            state.push_back(static_cast<double>(k) / kKeyScale);
        }

        auto res = dfs_values(state);
        memo.emplace(key, res);
        return res;
    }

    std::pair<bool, long long> dfs_values(const std::vector<double>& state) {
        const int n = static_cast<int>(state.size());
        if (n == 0) return {false, 0};
        if (n == 1) {
            bool ok = std::fabs(state[0] - target) < tol;
            return {ok, 0};
        }

        long long total_actions = 0;
        if (use_deadline && std::chrono::steady_clock::now() > deadline) {
            return {false, total_actions};
        }
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const double a = state[i];
                const double b = state[j];
                std::vector<double> rest;
                rest.reserve(n - 1);
                for (int k = 0; k < n; ++k) {
                    if (k != i && k != j) rest.push_back(state[k]);
                }

                // Generate candidates in the same order as Python
                std::vector<double> candidates;
                candidates.reserve(6);
                candidates.push_back(a + b);
                candidates.push_back(a * b);
                candidates.push_back(a - b);
                candidates.push_back(b - a);
                if (std::fabs(b) > tol) candidates.push_back(a / b);
                if (std::fabs(a) > tol) candidates.push_back(b / a);

                for (double c : candidates) {
                    std::vector<double> next_state = rest;
                    next_state.push_back(c);
                    std::vector<long long> next_key = make_key(next_state);
                    auto sub = dfs_keyed(next_key);
                    total_actions += sub.second + 1;  // count this expansion
                    if (max_expansions >= 0) {
                        expansions += 1;
                        if (expansions >= max_expansions) {
                            return {false, total_actions};
                        }
                    }
                    if (use_deadline && std::chrono::steady_clock::now() > deadline) {
                        return {false, total_actions};
                    }
                    if (sub.first) {
                        return {true, total_actions};
                    }
                }
            }
        }
        return {false, total_actions};
    }
};

std::pair<bool, long long> can_merge_to_target_actions(const std::vector<double>& values,
                                                       double target,
                                                       double tol) {
    if (values.empty()) return {false, 0};
    DfsSolver solver;
    solver.target = target;
    solver.tol = tol;
    std::vector<long long> key = make_key(values);
    return solver.dfs_keyed(key);
}

std::pair<bool, long long> can_merge_to_target_actions_budget(const std::vector<double>& values,
                                                              double target,
                                                              double tol,
                                                              double time_budget_ms,
                                                              long long max_expansions) {
    if (values.empty()) return {false, 0};
    DfsSolver solver;
    solver.target = target;
    solver.tol = tol;
    solver.max_expansions = max_expansions;
    if (time_budget_ms > 0) {
        solver.use_deadline = true;
        solver.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(static_cast<long long>(time_budget_ms));
    }
    std::vector<long long> key = make_key(values);
    return solver.dfs_keyed(key);
}

}  // namespace

PYBIND11_MODULE(_merge_search, m) {
    m.doc() = "C++ accelerated DFS for countdown merge-to-target with action counting";
    // Unified API: single function with optional timeout and expansion cap
    m.def("can_merge_to_target",
          &can_merge_to_target_actions_budget,
          py::arg("values"), py::arg("target"), py::arg("tol") = 1e-6,
          py::arg("time_budget_ms") = -1.0, py::arg("max_expansions") = -1,
          "Return (can_merge, actions) with optional time budget (ms) and expansion cap.");
}


