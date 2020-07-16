#include "Header.h"

int d = 2; // domain size
int N = 5; // points in each direction
double h = (double)d / N; // mesh spacing
double k = 0.2; // time step
int Tf = 500; // final time
int nn = N * N;

// pde parameters
double r = 0.01;
double g = 1;

Vec Uo; // this is U[0 - 1] which is set equal to U[0] for simplicity
Vec UoUo; // UoUo[i] = Uo[i]*Uo[i]
Vec UoUoUo;
SpMat L(nn, nn);

Vec residual(Vec u){
	Vec x;
	Vec uu = u.cwiseProduct(u);
	// (u - Uo) / k - (L @ u + g * uu - np.multiply(u, uu) + L @ Uo + g * UoUo - UoUoUo) / 2
	x = L * u;
	x += L * Uo;
	x += g * uu;
	x += g * UoUo;
	x -= u.cwiseProduct(uu);
	x -= UoUoUo;
	x /= -2;
	x += (u - Uo) / k;
	return x;
}

int main(int argc, char** argv) {
	// random initial condition
	Vec U = Vec::Random(nn); // this is uniform probability but it should be fine 

	// Form the discreet Laplacian
	double e = 1 / h / h;
	SpMat f(N, N);
	SpMat Lap(nn, nn);
	for (int i = 0; i < N-1; ++i) {
		f.insert(i, i) = -4*e;
		f.insert(i + 1, i) = e;
		f.insert(i, i + 1) = e;
	}
	f.insert(0, N - 1) = e;
	f.insert(N - 1, 0) = e;
	f.insert(N - 1, N - 1) = -4*e;	
	double val;
	int row, col;
	for (int k = 0; k < f.outerSize(); ++k) {
		for (SpMat::InnerIterator it(f, k); it; ++it)
		{

			val = it.value();
			row = it.row();
			col = it.col();
			for (int i = 0; i < N; ++i) {
				Lap.insert(row+i*N, col+i*N) = val;
			}
		}
	}
	for (int i = 0; i < nn-N; ++i) {
		Lap.insert(i, i + N) = e;
		Lap.insert(i + N, i) = e;
	}
	for (int i = 0; i < N; ++i) {
		Lap.insert(i, i + nn - N) = e;
		Lap.insert(i + nn - N, i) = e;
	}
	
	// form the discreet linear operator
	SpMat Ident(nn, nn);
	for (int i = 0; i < nn;++i) {
		Ident.insert(i, i) = 1;
	}
	L = -(Lap * Lap).pruned();
	L -= 2 * Lap;
	L += (r - 1) * Ident;
	L.makeCompressed();
	
	// prepare file to output data
	ofstream myfile;
	string folder = "."; // folder path
	string filename = "output.txt"; // file name
	string path = folder + "\\" + filename; // full path for file
	myfile.open(path);
	if (!myfile.is_open()) {
		throw "File not opened with path: " + path + "\nPlease fix path";
	}

	int Nsteps = ceil(Tf / k); // ESTIMATE NUMBER OF TIME STEPS REQUIRED
	for (int s = 0; s < Nsteps; ++s) {
		// HERE U is U[s] and Uo is U[s - 1]
		// BEFORE OVERWRITING U, STORE U[s] to Uo for the next iteration
		Uo = U;
		UoUo = Uo.cwiseProduct(Uo);
		UoUoUo = Uo.cwiseProduct(UoUo);

		cout << s + 1 << endl;
		float a = std::numeric_limits<float>::infinity();
		// NEWTON KRYLOV
		U = nonlin_solve(residual, Uo, 6e-6,a,a,a);

		
	}
	
	return 0;
}
