/* File : cKDE.i */
%module cKDE
%{
/* Put headers and other declarations here */
extern double dist(double lat1, double long1, double lat2, double long2);
extern double K(double x);
extern double prob(double lat1, double long1, double lat2, double long2, double* d, int len_d, double h);
extern double* new_doubleArray(int len);
extern void set_doubleItem(double* d, int i, double v);
%}

extern double dist(double lat1, double long1, double lat2, double long2);
extern double K(double x);
extern double prob(double lat1, double long1, double lat2, double long2, double* d, int len_d, double h);
extern double* new_doubleArray(int len);
extern void set_doubleItem(double* d, int i, double v);