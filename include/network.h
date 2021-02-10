#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <gsl/gsl_blas.h>

#define HIDDEN 256

enum NETWORK{NOT_DIGIT=255};

int testImage(int32_t id,uint8_t* labels,gsl_matrix* probabilities);
gsl_matrix** prepareLayers(uint8_t numberOfLayers,uint32_t* numberOfLayersPoints);
gsl_matrix** prepareWeights(uint8_t numberOfLayers,uint32_t* numberOfLayersPoints);
void unloadLayers(gsl_matrix** layers,uint8_t numberOfLayers);
void unloadWeights(gsl_matrix** weights,uint8_t numberOfLayers);
double calculateLoss(gsl_matrix* probabilities,uint8_t* label, int32_t id);
void forwardPass(int32_t id,uint8_t* data,gsl_matrix** layers,gsl_matrix** weights);
void backwardPass(int32_t id,uint8_t* data,uint8_t* labels,gsl_matrix** layers,gsl_matrix** weights,double rate,gsl_matrix* delta1,gsl_matrix* error1,gsl_matrix* realProbabilities);

inline double randomUniform(double from, double to){return from+(to-from)*((double) rand () / RAND_MAX);}
inline double sigmoid(double x){return 1.0/(1.0+exp(-x));}	//sigmoid
inline double deSigmoid(double x){return x*(1.0-x);}		//sigmoid^-1

void softmax(gsl_matrix* in,gsl_matrix* out);
void deSoftmax(gsl_matrix* in,gsl_matrix* out);//softmax^-1

#endif
