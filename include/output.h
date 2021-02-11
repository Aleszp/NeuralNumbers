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

#endif
