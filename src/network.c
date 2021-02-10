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

void forwardPass(int32_t id,uint8_t* data,gsl_matrix** layers,gsl_matrix** weights,gsl_matrix* probabilities)
{
	int32_t offset=id*((layers[0]->size2)-1);
	for(int ii=0;ii<((layers[0]->size2-1));ii++)
	{
		//fprintf(stderr,"layers[0](0,%u) out of (0,%lu)\n",ii,layers[0]->size2);
		gsl_matrix_set(layers[0],0,ii,((double)data[ii+offset])/255.0);
	}
	//fprintf(stderr,"Bias, layers[0](0,%lu) out of (0,%lu)\n",layers[0]->size2,layers[0]->size2);
	gsl_matrix_set(layers[0],0,layers[0]->size2-1,1.0);	//bias
	
	//find what image is most similiar to
	//multiply input layer by matrix
	//fprintf(stderr,"layers[0]:(%lu,%lu), layers[1]:(%lu,%lu)=probabilities:(%lu,%lu)\n",layers[0]->size1,layers[0]->size2,layers[1]->size1,layers[1]->size2,probabilities->size1,probabilities->size2);
	
	
	//fprintf(stderr,"layers[0]:(%lu,%lu), layers[1]:(%lu,%lu)=C:(%lu,%lu)\n",layers[0]->size1,layers[0]->size2,layers[1]->size1,layers[1]->size2,C->size1,C->size2);
	
	//Normalize
	//softmax(C,C);
	/*for(int ii=0;ii<HIDDEN;ii++)
	{
		gsl_matrix_set(C,0,ii,sigmoid(gsl_matrix_get(C,0,ii)));
	}*/
	//multiply intermediate layer by matrix and return probabilities in output layer by matrix
	//fprintf(stderr,"C:(%lu,%lu), layers[2]:(%lu,%lu)=probabilities:(%lu,%lu)\n",C->size1,C->size2,layers[2]->size1,layers[2]->size2,probabilities->size1,probabilities->size2);
	//gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,1.0, C,layers[2],0.0, probabilities);
	
	//Normalize
	/*for(int jj=0;jj<10;jj++)
	{
		gsl_matrix_set(probabilities,0,jj,sigmoid(gsl_matrix_get(probabilities,0,jj)));
	}*/
	//printOther(10,probabilities,"Probab");
	
	
	//Normalize
	/*for(int ii=0;ii<HIDDEN;ii++)
	{
		gsl_matrix_set(C,0,ii,sigmoid(gsl_matrix_get(C,0,ii)));
	}*/
	
	//fprintf(stderr,"layers[0]:(%lu,%lu), weights[0]:(%lu,%lu)=probabilities:(%lu,%lu)\n",layers[0]->size1,layers[0]->size2,weights[0]->size1,weights[0]->size2,probabilities->size1,probabilities->size2);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,1.0,layers[0],weights[0],0.0, probabilities);
	
	softmax(probabilities,probabilities);
	//printOther(10,probabilities,"nProbab");
	
}

void backwardPass(int32_t id,uint8_t* data,uint8_t* labels,gsl_matrix** layers,gsl_matrix** weights,gsl_matrix* probabilities,double rate,gsl_matrix* delta1,gsl_matrix* error1,gsl_matrix* realProbabilities)
{
	for(uint8_t jj=0;jj<10;jj++)
	{
		gsl_matrix_set(realProbabilities,0,jj,jj==labels[id]?-1.0:0.0);
	}
	gsl_matrix_add(realProbabilities,probabilities);
		
	//fprintf(stderr,"layers[0]:(%lu,%lu), realProbabilities:(%lu,%lu)=delta1:(%lu,%lu)\n",layers[0]->size1,layers[0]->size2,realProbabilities->size1,realProbabilities->size2,delta1->size1,delta1->size2);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans,rate,layers[0],realProbabilities,0.0, delta1);

	//fprintf(stderr,"weights[0]:(%lu,%lu)-=delta1:(%lu,%lu)\n",weights[0]->size1,weights[0]->size2,delta1->size1,delta1->size2);
	gsl_matrix_sub(weights[0], delta1);
	
	/*for(uint8_t jj=0;jj<10;jj++)
	{
		tmp=gsl_matrix_get(probabilities,0,jj);
		//loss-=(((labels[id]==jj)?1.0:0.0)*log(tmp));
		
		//gsl_matrix_set(error2,0,jj,-(((labels[id]==jj)?1.0:-1.0/9.0)*log(tmp)));
		//gsl_matrix_set(error2,0,jj,labels[id]==jj?(tmp-1.0):(tmp));
		gsl_matrix_set(delta1,0,jj,deSigmoid(tmp));
	}*/
	//deSoftmax(probabilities,delta1);
	//printOther(10,probabilities,"dProbab");
	//printOther(10,delta2,"Delta2");
	//printOther(10,error2,"Error2");
	//gsl_matrix_mul_elements(delta2, error2);
	//gsl_matrix_scale(delta1,-loss);
	//printOther(10,delta2,"Delta2");	
	
	//fprintf(stderr,"C:(%lu,%lu)*delta2:(%lu,%lu)=A:(%lu,%lu)\n",C->size1,C->size2,delta2->size1,delta2->size2,A->size1,A->size2);
	//gsl_blas_dgemm(CblasTrans, CblasNoTrans,rate,layers[0],delta1,0.0, A);
	
	//fprintf(stderr,"delta2:(%lu,%lu)*layers[2]:(%lu,%lu)=error1:(%lu,%lu)\n",delta2->size1,delta2->size2,layers[2]->size1,layers[2]->size2,error1->size1,error1->size2);
	//gsl_blas_dgemm(CblasNoTrans, CblasTrans,1.0,delta2,layers[2],0.0,error1);
		
	/*for(uint8_t jj=0;jj<10;jj++)
	{
		tmp=gsl_matrix_get(C,0,jj);
		gsl_matrix_set(delta1,0,jj,deSigmoid(tmp));
	}*/
	//deSoftmax(C,delta1);
	//fprintf(stderr,"delta1:(%lu,%lu)*error1:(%lu,%lu)\n",delta1->size1,delta1->size2,error1->size1,error1->size2);
	//gsl_matrix_mul_elements(delta1, error1);
	
	//gsl_matrix* B=gsl_matrix_alloc(dataSize,HIDDEN);
	//fprintf(stderr,"layers[0]:(%lu,%lu)*delta1:(%lu,%lu)=B:(%lu,%lu)\n",layers[0]->size1,layers[0]->size2,delta1->size1,delta1->size2,B->size1,B->size2);
	//gsl_blas_dgemm(CblasTrans, CblasNoTrans,rate,layers[0],delta1,0.0, B);
	
	//fprintf(stderr,"layer[1]:(%lu,%lu)+B:(%lu,%lu)\n",layers[1]->size1,layers[1]->size2,B->size1,B->size2);
	//gsl_matrix_scale(B,rate);
	//gsl_matrix_add(layers[1], B);
	
	//fprintf(stderr,"layer[2]:(%lu,%lu)+A:(%lu,%lu)\n",layers[2]->size1,layers[2]->size2,A->size1,A->size2);
	//gsl_matrix_scale(A,rate);
	//gsl_matrix_add(layers[1], A);
	//gsl_vector_free(temporaryRow);
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
		//fprintf(stderr,"dSoftmax: %u/%lu,tmp=%lf,sum=%lf\n",ii,in->size2,tmp,sum);
	}
	//gsl_matrix_scale(out,1.0/sum);
	gsl_matrix_free(tmpMatrix);
}

