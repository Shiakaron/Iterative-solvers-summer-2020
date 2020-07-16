#include <iostream>
#include <string>
#include <random>
#include <fstream>
#include <cmath>
#include <limits>

#include "C:\Users\savva\Documents\GitHub\NewtonKrylov\types.h"
#include "C:\Users\savva\Documents\GitHub\NewtonKrylov\lgmres.h"
#include "C:\Users\savva\Documents\GitHub\NewtonKrylov\newton_krylov.h"

using namespace std;

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat;
typedef Eigen::Triplet<double> T;
