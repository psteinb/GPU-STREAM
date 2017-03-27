// Copyright (c) 2015-16 Peter Steinbach, MPI CBG Scientific Computing Facility
//
// For full license terms please see the LICENSE file distributed with this
// source code
#include "HCStream.h"

#include <codecvt>
#include <vector>
#include <locale>
#include <numeric>

static constexpr int TBSIZE = 1024;

std::string getDeviceName(const hc::accelerator& _acc)
{
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  std::string value = converter.to_bytes(_acc.get_description());
  return value;
}

void listDevices(void)
{
  // Get number of devices
  std::vector<hc::accelerator> accs = hc::accelerator::get_all();

  // Print device names
  if (accs.empty())
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < accs.size(); i++)
    {
      std::cout << i << ": " << getDeviceName(accs[i]) << std::endl;
    }
    std::cout << std::endl;
  }
}


template <class T>
HCStream<T>::HCStream(unsigned int array_sz, int device_index)
    : d_a(array_sz), d_b(array_sz), d_c(array_sz)
{
    // The array size must be divisible by TBSIZE for kernel launches
    if (array_sz % TBSIZE != 0)
    {
        std::stringstream ss;
        ss << "Array size must be a multiple of " << TBSIZE;
        throw std::runtime_error(ss.str());
    }
    // Set device

    std::vector<hc::accelerator> accs = hc::accelerator::get_all();
    auto current = accs[device_index];

    hc::accelerator::set_default(current.get_device_path());

    std::cout << "Using HC device " << getDeviceName(current) << std::endl;
}

template <class T>
void HCStream<T>::init_arrays(T _a, T _b, T _c)
{
    auto& view_a = this->d_a;
    auto& view_b = this->d_b;
    auto& view_c = this->d_c;

    hc::parallel_for_each(
        view_a.get_extent(),
        [=, &view_a, &view_b, &view_c](hc::index<1> idx) [[hc]] {
        view_a[idx] = _a;
        view_b[idx] = _b;
        view_c[idx] = _c;
    });

    try {
        view_a.get_accelerator_view().wait();
    }
    catch (std::exception& e) {
        std::cout << __FILE__ << ":" << __LINE__ << e.what() << std::endl;

        throw;
    }
}

template <class T>
void HCStream<T>::read_arrays(
    std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
    a = d_a;
    b = d_b;
    c = d_c;
}

template <class T>
void HCStream<T>::copy()
{
    const auto& view_a = this->d_a;
    auto& view_c = this->d_c;

    try {
        hc::parallel_for_each(
            view_a.get_extent(), [&](hc::index<1> idx) [[hc]] {
            view_c[idx] = view_a[idx];
        });
        view_c.get_accelerator_view().wait();
    }
    catch (std::exception& e) {
        std::cerr << __FILE__ << ":" << __LINE__ << "\t" << e.what() << std::endl;

        throw e;
    }
}

template <class T>
void HCStream<T>::mul()
{
    static constexpr T scalar = 0.3;

    auto& view_b = this->d_b;
    const auto& view_c = this->d_c;

    try {
        hc::parallel_for_each(
            view_b.get_extent(), [=, &view_b, &view_c](hc::index<1> idx) [[hc]] {
            view_b[idx] = scalar * view_c[idx];
        });

        view_b.get_accelerator_view().wait();
    }
    catch (std::exception& e) {
        std::cerr << __FILE__ << ":" << __LINE__ << "\t" << e.what() << std::endl;
        throw;
    }
}

template <class T>
void HCStream<T>::add()
{
    const auto& view_a = this->d_a;
    const auto& view_b = this->d_b;
    auto& view_c = this->d_c;

    try {
        hc::parallel_for_each(
            view_c.get_extent(), [&](hc::index<1> idx) [[hc]] {
            view_c[idx] = view_a[idx] + view_b[idx];
        });
        view_c.get_accelerator_view().wait();
    }
    catch (std::exception& e) {
        std::cerr << __FILE__ << ":" << __LINE__ << "\t" << e.what() << std::endl;

        throw e;
    }
}

template <class T>
void HCStream<T>::triad()
{
    static constexpr T scalar = 0.3;

    auto& view_a = this->d_a;
    const auto& view_b = this->d_b;
    const auto& view_c = this->d_c;

    try {
        hc::parallel_for_each(
            view_a.get_extent(),
            [=, &view_a, &view_b, &view_c](hc::index<1> idx) [[hc]] {
            view_a[idx] = view_b[idx] + scalar * view_c[idx];
        });
        view_a.get_accelerator_view().wait();
    }
    catch (std::exception& e) {
        std::cerr << __FILE__ << ":" << __LINE__ << "\t" << e.what() << std::endl;

        throw e;
    }
}

template <class T>
T HCStream<T>::dot_impl()
{
    //implementation adapted from
    //https://ampbook.codeplex.com/SourceControl/latest
    // ->Samples/CaseStudies/Reduction
    // ->CascadingReduction.h

    static constexpr std::size_t n_tiles = 64;

    const auto& view_a = this->d_a;
    const auto& view_b = this->d_b;

    auto ex = view_a.get_extent();
    const auto tiled_ex = hc::extent<1>(n_tiles * TBSIZE).tile(TBSIZE);
    const auto domain_sz = tiled_ex.size();

    hc::array<T, 1> partial(n_tiles);

    hc::parallel_for_each(tiled_ex,
                          [=,
                           &view_a,
                           &view_b,
                           &partial](const hc::tiled_index<1>& tidx) [[hc]] {
        // Alex note: you could consider this alternative, which I find more
        //            straightforward to teach, and which should come at no
        //            performance penalty.
        auto gidx = tidx.global[0];
        T r = T{0}; // Assumes reduction op is addition.
        while (gidx < view_a.get_extent().size()) {
            r += view_a[gidx] * view_b[gidx];
            gidx += domain_sz;
        }

        tile_static T tileData[TBSIZE];
        tileData[tidx.local[0]] = r;

        tidx.barrier.wait_with_tile_static_memory_fence();

        for (auto h = TBSIZE / 2; h; h /= 2) {
            if (tidx.local[0] < h) {
                tileData[tidx.local[0]] += tileData[tidx.local[0] + h];
            }
            tidx.barrier.wait_with_tile_static_memory_fence();
        }

        if (tidx.global == tidx.tile_origin) partial[tidx.tile] = tileData[0];

//        auto i = (tidx.tile[0] * 2 * TBSIZE) + tidx.local[0];
//        auto stride = TBSIZE * 2 * n_tiles;
//
//        //  Load and add many elements, rather than just two
//        T sum = 0;
//        do
//        {
//            T n = view_a[i] * view_b[i];
//            T f = view_a[i + TBSIZE] * view_b[i + TBSIZE];
//
//            sum += n + f;
//            i += stride;
//        }
//        while (i < view_a.get_extent().size());
//
//        tile_static T tileData[TBSIZE];
//        tileData[tidx.local[0]] = sum;
//
//        tidx.barrier.wait_with_tile_static_memory_fence();
//
//        //  Reduce values for data on this tile
//        for (auto h = (TBSIZE / 2); h > 0; stride /= 2)
//        {
//            //  Remember that this is a branch within a loop and all threads
//            //  will have to execute this but only threads with a tid < stride
//            //  will do useful work.
//
//            if (tidx.local[0] < h)
//                tileData[tidx.local[0]] += tileData[tidx.local[0] + h];
//
//            tidx.barrier.wait_with_tile_static_memory_fence();
//        }
//
//        //  Write the result for this tile back to global memory
//        if (tidx.local[0] == 0)
//            partial[tidx.tile[0]] = tileData[tidx.local[0]];
    });

    try {
        partial.get_accelerator_view().wait();
    }
    catch (std::exception& e) {
        std::cerr << __FILE__ << ":" << __LINE__ << "\t" << e.what() << std::endl;
        throw;
    }

    std::vector<T> h_partial = partial;
    T result = std::accumulate(h_partial.begin(), h_partial.end(), 0.);

    return result;
}

template class HCStream<float>;
template class HCStream<double>;
