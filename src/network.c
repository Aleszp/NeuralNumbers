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
		return 1;
	return 0;
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

void forwardPass(int32_t id,uint8_t* data,gsl_matrix** layers,gsl_matrix** weights)
{
	uint currentLayer=0;
	int32_t offset=id*((layers[currentLayer]->size2)-1);
	for(int ii=0;ii<((layers[currentLayer]->size2-1));ii++)
	{
		//fprintf(stderr,"layers[0](0,%u) out of (0,%lu)\n",ii,layers[0]->size2);
		gsl_matrix_set(layers[currentLayer],0,ii,((double)data[ii+offset])/255.0);
	}
	//fprintf(stderr,"Bias, layers[0](0,%lu) out of (0,%lu)\n",layers[0]->size2,layers[0]->size2);
	gsl_matrix_set(layers[currentLayer],0,layers[currentLayer]->size2-1,1.0);	//bias
	
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,1.0,layers[currentLayer],weights[currentLayer],0.0, layers[currentLayer+1]);
	currentLayer++;
	sigmoid(layers[currentLayer],layers[currentLayer]);
	
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,1.0,layers[currentLayer],weights[currentLayer],0.0, layers[currentLayer+1]);
	currentLayer++;
	softmax(layers[currentLayer],layers[currentLayer]);
}

void backwardPass(int32_t id,uint8_t* data,uint8_t* labels,gsl_matrix** layers,gsl_matrix** weights,double rate,gsl_matrix** dWeights,gsl_matrix** dLayers)
{
	uint currentLayer=2;
	for(uint8_t jj=0;jj<10;jj++)
	{
		gsl_matrix_set(dLayers[currentLayer],0,jj,jj==labels[id]?-1.0:0.0);
	}
	//fprintf(stderr,"dLayers[1]:(%lu,%lu)+=layers[1]:(%lu,%lu)\n",dLayers[1]->size1,dLayers[1]->size2,layers[1]->size1,layers[1]->size2);
	gsl_matrix_add(dLayers[currentLayer],layers[currentLayer]);
		
	currentLayer--;
	deSigmoid(layers[currentLayer],layers[currentLayer]);
	//fprintf(stderr,"layers[0]:(%lu,%lu), dLayers[1]:(%lu,%lu)=dWeights[0]:(%lu,%lu)\n",layers[0]->size1,layers[0]->size2,dLayers[1]->size1,dLayers[1]->size2,dWeights[0]->size1,dWeights[0]->size2);
	gsl_blas_dgemm(CblasTrans,CblasNoTrans,rate,layers[currentLayer],dLayers[currentLayer+1],0.0, dWeights[currentLayer]);

	//fprintf(stderr,"weights[1]:(%lu,%lu)-=dWeights[1]:(%lu,%lu)\n",weights[1]->size1,weights[1]->size2,dWeights[1]->size1,dWeights[1]->size2);
	gsl_matrix_sub(weights[currentLayer],dWeights[currentLayer]);
	
	//fprintf(stderr,"dLayers[2](%lu,%lu) x weights[1](%lu,%lu)=dLayers[1](%lu,%lu)=\n",dLayers[2]->size1,dLayers[2]->size2,weights[1]->size1,weights[1]->size2,dLayers[1]->size1,dLayers[1]->size2);
	gsl_blas_dgemm(CblasNoTrans,CblasTrans,1.0,dLayers[currentLayer+1],weights[currentLayer],0.0, dLayers[currentLayer]);
	currentLayer--;
	gsl_blas_dgemm(CblasTrans,CblasNoTrans,rate,layers[currentLayer],dLayers[currentLayer+1],0.0, dWeights[currentLayer]);

	gsl_matrix_sub(weights[currentLayer], dWeights[currentLayer]);
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

	
