#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "grid_domain.h"

class Simulator {
public:
  Simulator();
  ~Simulator();

private:
  std::unique_ptr<Grid> grid_;
};

#endif