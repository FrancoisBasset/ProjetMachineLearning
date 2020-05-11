// Console.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
//

#include <iostream>
#include <Eigen/Dense>

#include "../Dll/Source.h";

using Eigen::MatrixXd;


int main()
{
	std::cout << my_add(1, 2) << std::endl;
	
	/*Eigen::
	MatrixXd m(2, 2);
	m(0, 0) = 3;
	m(1, 0) = 2.5;
	m(0, 1) = -1;
	m(1, 1) = m(1, 0) + m(0, 1);
	std::cout << m << std::endl;*/


}