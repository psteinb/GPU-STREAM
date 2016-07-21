
// Copyright (c) 2015-16 Tom Deakin (1), Simon McIntosh-Smith (1), Peter Steinbach (2)
// (1) University of Bristol HPC, (2) MPI CBG (Dresden, Germany) 
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>
#include <sstream>

#include "Stream.h"
#include "hc.hpp"

#define IMPLEMENTATION_STRING "HCC"

template <class T>
class HCCStream : public Stream<T>
{
protected:
  // Size of arrays
  unsigned int array_size;

  // Device side pointers to arrays
  hc::array<T,1> d_a;
  hc::array<T,1> d_b;
  hc::array<T,1> d_c;


public:

  HCCStream(const unsigned int ARRAY_SIZE, const int device_index = 0);
  ~HCCStream();

  virtual void copy() override;
  virtual void add() override;
  virtual void mul() override;
  virtual void triad() override;

  virtual void write_arrays(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c) override;
  virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};
