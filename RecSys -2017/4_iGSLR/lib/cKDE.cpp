#include <iostream>
#include <stdio.h>
#include <math.h>
#include <Python.h>

using namespace std;

double const CONS = 1.0 / sqrt(2 * M_PI);
double const SQRT2 = sqrt(2.0);

double* new_doubleArray(int len) {
	double* d = new double[len];
	return d;
}

void set_doubleItem(double* d, int i, double v) {
	d[i] = v;
} 

double dist(double lat1, double long1, double lat2, double long2) {
	if (fabs(lat1 - lat2) < 1e-6 && fabs(long1 - long2) < 1e-6) return 0.0;
	double degree_to_radians = M_PI / 180.0;
	double phi1 = (90.0 - lat1) * degree_to_radians;
	double phi2 = (90.0 - lat2) * degree_to_radians;
	double theta1 = long1 * degree_to_radians;
	double theta2 = long2 * degree_to_radians;
	double cosine = sin(phi1) * sin(phi2) * cos(theta1 - theta2) + cos(phi1) * cos(phi2);
	double arc = acos(cosine);
	return arc * 6371;
}

double K(double x) {
	return CONS * exp(-(x * x));
}

double f(double dij, double *d, double h, int len_d) {
	double res = 0.0;
	double h_judge = h * 4.0, h_use = h * SQRT2;
	for (int i = 0; i < len_d; ++i) {
		if (fabs(dij - d[i]) < h_judge)
			res += K((dij - d[i]) / h_use);
	}
	return res / len_d / h;
}

double prob(double lat1, double long1, double lat2, double long2, double *d, int len_d, double h) {
	double dij = dist(lat1, long1, lat2, long2);
	return f(dij, d, h, len_d);
}
