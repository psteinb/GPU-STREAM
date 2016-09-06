
// Copyright (c) 2015-16 Tom Deakin (1), Simon McIntosh-Smith (1), Peter Steinbach (2)
// (1) University of Bristol HPC, (2) MPI CBG (Dresden, Germany) 
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <codecvt>
#include <vector>
#include <locale>
#include <cstdio>

#include "HCCStream.h"


#define TBSIZE 1024


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
HCCStream<T>::HCCStream(const unsigned int ARRAY_SIZE, const int device_index):
  array_size(ARRAY_SIZE),
  accelerator(hc::accelerator::get_all()[device_index]),
  d_a(ARRAY_SIZE,accelerator.get_default_view()),
  d_b(ARRAY_SIZE,accelerator.get_default_view()),
  d_c(ARRAY_SIZE,accelerator.get_default_view())
{

  // The array size must be divisible by TBSIZE for kernel launches
  if (ARRAY_SIZE % TBSIZE != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of " << TBSIZE;
    throw std::runtime_error(ss.str());
  }

  // // Set device
  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  auto current = accs[device_index];
  
  std::cout << "Using HCC device " << getDeviceName(current) << std::endl;

  // Check buffers fit on the device
  // TODO: unclear how to do that!!
}


template <class T>
HCCStream<T>::~HCCStream()
{
}

template <class T>
void HCCStream<T>::write_arrays(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c)
{
  hc::copy(a.cbegin(),a.cend(),d_a);
  hc::copy(b.cbegin(),b.cend(),d_b);
  hc::copy(c.cbegin(),c.cend(),d_c);
}

template <class T>
void HCCStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  // Copy device memory to host
  hc::copy(d_a,a.begin());
  hc::copy(d_b,b.begin());
  hc::copy(d_c,c.begin());
}

template <class T>
void HCCStream<T>::copy()
{
  hc::array<T> &d_a = this->d_a;
  hc::array<T> &d_c = this->d_c;

  try{
  // launch a GPU kernel to compute the saxpy in parallel 
    hc::completion_future future_kernel = hc::parallel_for_each(accelerator.get_default_view()
                , hc::extent<1>(array_size)
								, [&](hc::index<1> i) __attribute((hc)) {

								 d_c[i] = d_a[i];
								});
    future_kernel.wait();
  }
  catch(std::exception& e){
    std::cout << e.what() << std::endl;
    throw;
  }
}

template <class T>
void HCCStream<T>::mul()
{
  hc::array<T> &d_b = this->d_b;
  hc::array<T> &d_c = this->d_c;

  const T scalar = 0.3;
  try{
  // launch a GPU kernel to compute the saxpy in parallel 
    hc::completion_future future_kernel = hc::parallel_for_each(accelerator.get_default_view()
                , hc::extent<1>(array_size)
								, [&](hc::index<1> i) __attribute((hc)) {
								  d_b[i] = scalar*d_c[i];
								});
    future_kernel.wait();
  }
  catch(std::exception& e){
    std::cout << e.what() << std::endl;
    throw;
  }
}

template <class T>
void HCCStream<T>::add()
{
  hc::array<T> &d_a = this->d_a;
  hc::array<T> &d_b = this->d_b;
  hc::array<T> &d_c = this->d_c;

  try{
    // launch a GPU kernel to compute the saxpy in parallel 
    hc::completion_future future_kernel = hc::parallel_for_each(accelerator.get_default_view()
                , hc::extent<1>(array_size)
								, [&](hc::index<1> i) __attribute((hc)) {
								  d_c[i] = d_a[i]+d_b[i];
								});
    future_kernel.wait();
  }
  catch(std::exception& e){
    std::cout << e.what() << std::endl;
    throw;
  }
}

template <class T>
void HCCStream<T>::triad()
{
  hc::array<T> &d_a = this->d_a;
  hc::array<T> &d_b = this->d_b;
  hc::array<T> &d_c = this->d_c;

  const T scalar = 0.3;
  try{
    // launch a GPU kernel to compute the saxpy in parallel 
    hc::completion_future future_kernel = hc::parallel_for_each(accelerator.get_default_view()
                , hc::extent<1>(array_size)
								, [&](hc::index<1> i) __attribute((hc)) {
								  d_a[i] = d_b[i] + scalar*d_c[i];
								});
    future_kernel.wait();
  }
  catch(std::exception& e){
    std::cout << e.what() << std::endl;
    throw;
  }
}



template class HCCStream<float>;
template class HCCStream<double>;
