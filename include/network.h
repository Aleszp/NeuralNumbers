#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <gsl/gsl_blas.h>

#define HIDDEN 256

enum NETWORK{NOT_DIGIT=255};
enum ACTIVATION{SIGMOID,SOFTMAX};

int testImage(int32_t id,uint8_t* labels,gsl_matrix* probabilities);
gsl_matrix** prepareLayers(uint8_t numberOfLayers,uint32_t* numberOfLayersPoints);
gsl_matrix** prepareWeights(uint8_t numberOfLayers,uint32_t* numberOfLayersPoints);
void unloadLayers(gsl_matrix** layers,uint8_t numberOfLayers);
void unloadWeights(gsl_matrix** weights,uint8_t numberOfLayers);
double calculateLoss(gsl_matrix* probabilities,uint8_t* label, int32_t id);
void forwardPass(int32_t id,uint8_t* data,gsl_matrix** layers,gsl_matrix** weights,gsl_matrix** biases,uint8_t numberOfLayers,uint8_t* activations);
void backwardPass(int32_t id,uint8_t* data,uint8_t* labels,gsl_matrix** layers,gsl_matrix** weights,gsl_matrix** biases,double rate,gsl_matrix** dWeights,gsl_matrix** dLayers,gsl_matrix** dBiases,uint8_t numberOfLayers,uint8_t* activations);

inline double randomUniform(double from, double to){return from+(to-from)*((double) rand () / RAND_MAX);}
inline double _sigmoid(double x){return 1.0/(1.0+exp(-x));}	
inline double _deSigmoid(double x){return x*(1.0-x);}		//sigmoid^-1

void activate(gsl_matrix* in,gsl_matrix* out,uint8_t activation);
void deActivate(gsl_matrix* in,gsl_matrix* out,uint8_t activation);

void softmax(gsl_matrix* in,gsl_matrix* out);
void deSoftmax(gsl_matrix* in,gsl_matrix* out);			//softmax^-1

void sigmoid(gsl_matrix* in,gsl_matrix* out);
void deSigmoid(gsl_matrix* in,gsl_matrix* out);			//sigmoid^-1

#endif
