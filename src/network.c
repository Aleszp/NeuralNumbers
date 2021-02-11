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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_blas.h>

#include "network.h"
#include "output.h"


int testImage(int32_t id,uint8_t* labels,gsl_matrix* probabilities)
{
	double maxProb=0.0;
	double tmp=0.0;
	uint8_t	label=10;
	for(uint8_t jj=0;jj<10;jj++)
	{
		tmp=gsl_matrix_get(probabilities,0,jj);
		if(tmp>maxProb)
		{
			maxProb=tmp;
			label=jj;
		}
	}
	if(label==labels[id])
		return -1;
	return label;
}

gsl_matrix** prepareLayers(uint8_t numberOfLayers,uint32_t*	numberOfLayersPoints)
{
	gsl_matrix** layers=(gsl_matrix**)malloc(sizeof(gsl_matrix*)*numberOfLayers);
	for(uint8_t ii=0;ii<numberOfLayers;ii++)
	{
		layers[ii]=gsl_matrix_calloc(1,numberOfLayersPoints[ii]);
	}
	return layers;
}

gsl_matrix** prepareWeights(uint8_t numberOfLayers,uint32_t* numberOfLayersPoints)
{
	gsl_matrix** weights=(gsl_matrix**)malloc(sizeof(gsl_matrix*)*numberOfLayers);
	for(uint8_t ii=0;ii<numberOfLayers-1;ii++)
	{
		weights[ii]=gsl_matrix_alloc(numberOfLayersPoints[ii],numberOfLayersPoints[ii+1]);
		for(uint32_t jj=0;jj<numberOfLayersPoints[ii+1];jj++)
		{
			for(uint32_t kk=0;kk<numberOfLayersPoints[ii];kk++)
			{
				gsl_matrix_set(weights[ii],kk,jj,randomUniform(-0.01,0.01));
			}
		}
	}
	return weights;
}

void unloadLayers(gsl_matrix** layers,uint8_t numberOfLayers)
{
	for(uint8_t ii=0;ii<numberOfLayers;ii++)
	{
		gsl_matrix_free(layers[ii]);
	}
	free(layers);
}

void unloadWeights(gsl_matrix** weights,uint8_t numberOfLayers)
{
	for(uint8_t ii=0;ii<numberOfLayers-1;ii++)
	{
		gsl_matrix_free(weights[ii]);
	}
	free(weights);
}

double calculateLoss(gsl_matrix* probabilities,uint8_t* labels, int32_t id)
{
	double loss=0.0;
	double tmp=0.0;
	const double epsilon=1e-9;
	
	uint8_t label=labels[id];
	for(uint8_t jj=0;jj<10;jj++)
	{
		if(label==jj)
		{
			tmp=gsl_matrix_get(probabilities,0,jj);
			loss-=log(tmp>0?tmp:epsilon);				//add small epsilon as failsafe for 0
		}
	}
	
	return loss;
}

void forwardPass(int32_t id,uint8_t* data,gsl_matrix** layers,gsl_matrix** weights,gsl_matrix** biases,uint8_t numberOfLayers,uint8_t* activations)
{
	uint currentLayer=0;
	int32_t offset=id*((layers[currentLayer]->size2));
	for(int ii=0;ii<((layers[currentLayer]->size2));ii++)
	{
		//fprintf(stderr,"layers[0](0,%u) out of (0,%lu)\n",ii,layers[0]->size2);
		gsl_matrix_set(layers[currentLayer],0,ii,((double)data[ii+offset])/255.0);
	}
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,1.0,layers[currentLayer],weights[currentLayer],0.0, layers[currentLayer+1]);	
	currentLayer++;
	gsl_matrix_add(layers[currentLayer],biases[currentLayer]);
	activate(layers[currentLayer],layers[currentLayer],activations[currentLayer-1]);
	
	/*gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,1.0,layers[currentLayer],weights[currentLayer],0.0, layers[currentLayer+1]);	
	currentLayer++;
	gsl_matrix_add(layers[currentLayer],biases[currentLayer]);
	activate(layers[currentLayer],layers[currentLayer],activations[currentLayer-1]);*/
	
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,1.0,layers[currentLayer],weights[currentLayer],0.0, layers[currentLayer+1]);
	currentLayer++;
	gsl_matrix_add(layers[currentLayer],biases[currentLayer]);
	activate(layers[currentLayer],layers[currentLayer],activations[currentLayer-1]);
}

void backwardPass(int32_t id,uint8_t* data,uint8_t* labels,gsl_matrix** layers,gsl_matrix** weights,gsl_matrix** biases,double rate,gsl_matrix** dWeights,gsl_matrix** dLayers,gsl_matrix** dBiases,uint8_t numberOfLayers,uint8_t* activations)
{
	uint8_t currentLayer=numberOfLayers-1;
	for(uint8_t jj=0;jj<10;jj++)
	{
		gsl_matrix_set(dLayers[currentLayer],0,jj,jj==labels[id]?-1.0:0.0);
	}
	//fprintf(stderr,"dLayers[1]:(%lu,%lu)+=layers[1]:(%lu,%lu)\n",dLayers[1]->size1,dLayers[1]->size2,layers[1]->size1,layers[1]->size2);
	
	//deActivate(layers[currentLayer],layers[currentLayer],activations[currentLayer-1]);
	gsl_matrix_add(dLayers[currentLayer],layers[currentLayer]);
	
	gsl_matrix_scale(dLayers[currentLayer],rate);
	gsl_matrix_sub(biases[currentLayer],dLayers[currentLayer]);
	currentLayer--;
	gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,layers[currentLayer],dLayers[currentLayer+1],0.0, dWeights[currentLayer]);
	
	//fprintf(stderr,"dLayers[%u](%lu,%lu) x weights[%u](%lu,%lu)=dLayers[%u](%lu,%lu)\n",currentLayer+1,layers[currentLayer+1]->size1,layers[currentLayer+1]->size2,currentLayer,weights[currentLayer]->size1,weights[currentLayer]->size2,currentLayer,dLayers[currentLayer]->size1,dLayers[currentLayer]->size2);
	/*deActivate(layers[currentLayer+1],layers[currentLayer+1],activations[currentLayer]);
	gsl_blas_dgemm(CblasNoTrans,CblasTrans,1.0,dLayers[currentLayer+1],weights[currentLayer],0.0, dLayers[currentLayer]);
	gsl_matrix_sub(weights[currentLayer],dWeights[currentLayer]);
	
	gsl_matrix_scale(dLayers[currentLayer],rate);
	gsl_matrix_sub(biases[currentLayer],dLayers[currentLayer]);
	currentLayer--;
	gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,layers[currentLayer],dLayers[currentLayer+1],0.0, dWeights[currentLayer]);*/
	
	//fprintf(stderr,"dLayers[%u](%lu,%lu) x weights[%u](%lu,%lu)=dLayers[%u](%lu,%lu)\n",currentLayer+1,layers[currentLayer+1]->size1,layers[currentLayer+1]->size2,currentLayer,weights[currentLayer]->size1,weights[currentLayer]->size2,currentLayer,dLayers[currentLayer]->size1,dLayers[currentLayer]->size2);
	deActivate(layers[currentLayer+1],layers[currentLayer+1],activations[currentLayer]);
	gsl_blas_dgemm(CblasNoTrans,CblasTrans,1.0,dLayers[currentLayer+1],weights[currentLayer],0.0, dLayers[currentLayer]);
	gsl_matrix_sub(weights[currentLayer],dWeights[currentLayer]);
	
	gsl_matrix_scale(dLayers[currentLayer],rate);
	gsl_matrix_sub(biases[currentLayer],dLayers[currentLayer]);
	
	currentLayer--;
	deActivate(layers[currentLayer],layers[currentLayer],activations[currentLayer-1]);
	gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,layers[currentLayer],dLayers[currentLayer+1],0.0, dWeights[currentLayer]);
	gsl_matrix_sub(weights[currentLayer],dWeights[currentLayer]);
}

void softmax(gsl_matrix* in,gsl_matrix* out)
{
	double tmp=0.0,tmp2=0.0;
	double sum=0.0;
	for(uint16_t ii=0;ii<in->size2;ii++)
	{
		tmp=gsl_matrix_get(in,0,ii);
		
		tmp2=exp(tmp);
		if(isinf(tmp2))
		{
			tmp2=100.0;	//clip
			fprintf(stderr,"Clipping for ii=%u, tmp=%lf\n",ii,tmp);
		}
		sum+=tmp2;
		gsl_matrix_set(out,0,ii,tmp2);
		//fprintf(stderr,"Softmax: %u/%lu,tmp=%lf,exp(tmp)=%lf,sum=%lf\n",ii+1,in->size2,tmp,tmp2,sum);
	}
	gsl_matrix_scale(out,1.0/sum);
}

void deSoftmax(gsl_matrix* in,gsl_matrix* out)
{
	gsl_matrix* tmpMatrix=gsl_matrix_calloc(in->size1,in->size2);
	double tmp=0.0;
	double sum=0.0;
	for(uint16_t ii=0;ii<in->size2;ii++)
	{
		tmp=gsl_matrix_get(in,0,ii);
		tmp=exp(tmp);
		sum+=tmp;
		for(uint16_t jj=0;jj<in->size2;jj++)
		{
			if(jj==ii)
			{
				gsl_matrix_set(out,0,jj,tmp);
			}
			gsl_matrix_set(tmpMatrix,0,jj,-tmp*tmp);
		}
		gsl_matrix_scale(out,1.0/sum);
		gsl_matrix_scale(tmpMatrix,1.0/(sum*sum));
		gsl_matrix_add(out,tmpMatrix);
	}
	gsl_matrix_free(tmpMatrix);
}

void sigmoid(gsl_matrix* in,gsl_matrix* out)
{
	for(int ii=0;ii<in->size2;ii++)
	{
		gsl_matrix_set(out,0,ii,_sigmoid(gsl_matrix_get(in,0,ii)));
	}
}
void deSigmoid(gsl_matrix* in,gsl_matrix* out)			//sigmoid^-1
{
	for(int ii=0;ii<in->size2;ii++)
	{
		gsl_matrix_set(out,0,ii,_deSigmoid(gsl_matrix_get(in,0,ii)));
	}
}

void activate(gsl_matrix* in,gsl_matrix* out,uint8_t activation)
{
	switch(activation)
	{
		case(SIGMOID):
			sigmoid(in,out);
			break;
		case(SOFTMAX):
			softmax(in,out);
			break;
		default:
			break;//linear
	}
}

void deActivate(gsl_matrix* in,gsl_matrix* out,uint8_t activation)
{
	switch(activation)
	{
		case(SIGMOID):
			deSigmoid(in,out);
			break;
		case(SOFTMAX):
			deSoftmax(in,out);
			break;
		default:
			break;//linear
	}
}
	
