#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t result = (call);                                             \
        if (result != hipSuccess) {                                              \
            std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(result)); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

__global__ void compare_shuffles(std::uint32_t* divergent,
                                 std::uint32_t* convergent) {
    const unsigned lane = threadIdx.x;
    const unsigned value = lane + 1;

    if ((lane & 1u) == 0) {
        const unsigned odd_value = __shfl_down(value, 1, 64);
        divergent[lane / 2] = odd_value;
    }

    const unsigned odd_value = __shfl_down(value, 1, 64);
    if ((lane & 1u) == 0) {
        convergent[lane / 2] = odd_value;
    }
}

int main() {
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    if (std::string(properties.gcnArchName).rfind("gfx950", 0) != 0) {
        std::fprintf(stderr, "requires gfx950, found %s\n", properties.gcnArchName);
        return 2;
    }

    std::uint32_t *divergent_device = nullptr, *convergent_device = nullptr;
    HIP_CHECK(hipMalloc(&divergent_device, 32 * sizeof(std::uint32_t)));
    HIP_CHECK(hipMalloc(&convergent_device, 32 * sizeof(std::uint32_t)));
    HIP_CHECK(hipMemset(divergent_device, 0xff, 32 * sizeof(std::uint32_t)));
    HIP_CHECK(hipMemset(convergent_device, 0xff, 32 * sizeof(std::uint32_t)));
    compare_shuffles<<<1, 64>>>(divergent_device, convergent_device);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<std::uint32_t> divergent(32), convergent(32);
    HIP_CHECK(hipMemcpy(divergent.data(), divergent_device,
                        32 * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(convergent.data(), convergent_device,
                        32 * sizeof(std::uint32_t), hipMemcpyDeviceToHost));

    unsigned divergent_mismatches = 0;
    for (unsigned pair = 0; pair < 32; ++pair) {
        const unsigned expected = pair * 2 + 2;
        if (convergent[pair] != expected) {
            std::fprintf(stderr, "repaired shuffle wrong at pair %u: %u != %u\n",
                         pair, convergent[pair], expected);
            return 3;
        }
        if (divergent[pair] != expected) {
            ++divergent_mismatches;
            std::printf("pair=%u divergent=%u expected=%u\n", pair,
                        divergent[pair], expected);
        }
    }
    std::printf("arch=%s divergent_mismatches=%u/32 repaired_mismatches=0/32\n",
                properties.gcnArchName, divergent_mismatches);
    HIP_CHECK(hipFree(divergent_device));
    HIP_CHECK(hipFree(convergent_device));
    return 0;
}
