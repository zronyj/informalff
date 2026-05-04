# Makefile for Fortran subroutines
# Regular: gfortran -o lib_geometries.o -fbounds-check -shared -fPIC -O3 lib_geometries.f90
# Debug: gfortran -o lib_geometries.o -fbounds-check -shared -fPIC -Og lib_geometries.f90

lib_geometries.so: lib_geometries.f90
	gfortran -o lib_geometries.so -fbounds-check -shared -fPIC -O3 lib_geometries.f90