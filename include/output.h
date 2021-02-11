/*
 *   AI implementation of handwritten digit recognition written in C with GSL_BLAS.
 * 
 *   Author: Aleksander Szpakiewicz-Szatan
 * 
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * 
 *   Contact: aleksander.szpakiewicz-szatan.dokt(a)pw.edu.pl
 */

#ifndef _OUTPUT_H_
#define _OUTPUT_H_

#include <stdint.h>
#include <gsl/gsl_blas.h>

void printDigit(uint8_t* trainingData,int32_t height,int32_t width,int32_t id,uint8_t digit);
void printOther(uint8_t max,gsl_matrix* what,char* text);
void printProbabilities(uint8_t* labels,gsl_matrix* probabilities,uint32_t id,int detected);
void printLayers(gsl_matrix** layers,uint8_t numberOfLayers,uint32_t* numberOfLayersPoints);
void printWeights(gsl_matrix** weights,uint8_t numberOfLayers);
void printBiases(gsl_matrix** biases,uint8_t numberOfLayers);

void gnuNotice();

#endif
