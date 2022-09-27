/*
Copyright (c) 2013, NVIDIA Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once
#include <trove/array.h>
#include <trove/detail/dismember.h>

namespace trove {
namespace detail {

template<int s>
struct shuffle {
    __device__
    static void impl(array<int, s>& d, const int& i) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
        d.head = __shfl_sync(WARP_CONVERGED, d.head, i);
#else
        d.head = __shfl(d.head, i);
#endif
        shuffle<s-1>::impl(d.tail, i);
    }
};

template<>
struct shuffle<1> {
    __device__
    static void impl(array<int, 1>& d, const int& i) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
        d.head = __shfl_sync(WARP_CONVERGED, d.head, i);
#else
        d.head = __shfl(d.head, i);
#endif
    }
};

#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
template<int s>
struct shuffle_sync {
    __device__
    static void impl(unsigned mask, array<int, s>& d, const int& i) {
        d.head = __shfl_sync(mask, d.head, i);
        shuffle_sync<s-1>::impl(mask, d.tail, i);
    }
};

template<>
struct shuffle_sync<1> {
    __device__
    static void impl(unsigned mask, array<int, 1>& d, const int& i) {
        d.head = __shfl_sync(mask, d.head, i);
    }
};
#endif

}
}

template<typename T>
__device__
T __shfl(const T& t, const int& i) {
    typedef trove::array<int,
                         trove::detail::aliased_size<T, int>::value>
        lysed_array;
    lysed_array lysed = trove::detail::lyse<int>(t);
    trove::detail::shuffle<lysed_array::size>
      ::impl(lysed, i);
    return trove::detail::fuse<T>(lysed);
}

#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
template<typename T>
__device__
T __shfl_sync(unsigned mask, const T& t, const int& i) {
    typedef trove::array<int,
                         trove::detail::aliased_size<T, int>::value>
        lysed_array;
    lysed_array lysed = trove::detail::lyse<int>(t);
    trove::detail::shuffle_sync<lysed_array::size>
      ::impl(mask, lysed, i);
    return trove::detail::fuse<T>(lysed);
}
#endif
