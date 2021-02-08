#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_blas.h>

#include "network.h"
#include "output.h"


void testImage(int32_t id,uint8_t* data, int32_t dataSize,uint8_t* labels,gsl_matrix** layers,gsl_matrix* probabilities)
{
	int32_t offset=id*dataSize;
	
	gsl_matrix* A=gsl_matrix_alloc(dataSize,1); 
	
	for(int ii=0;ii<dataSize;ii++)
	{
		//A->data[ii]=
		gsl_matrix_set(A,ii,1,((double) data[ii+offset])/255.0);
	}
	
	gsl_matrix* C=gsl_matrix_alloc(dataSize,256);
	
	
	
	//find what image is most similiar to
	//multiply input layer by matrix
	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,1.0, A, layers[0],0.0, C);
	//multiply intermediate layer by matrix and return probabilities in output layer by matrix
	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,1.0, C, layers[1],0.0, probabilities);

	gsl_matrix_free(A);
	gsl_matrix_free(C);
}

gsl_matrix** prepareLayers(uint8_t numberOfLayers,uint32_t*	numberOfLayersPoints)
{
	gsl_matrix** layers=(gsl_matrix**)malloc(sizeof(gsl_matrix*)*numberOfLayers);
	for(uint8_t ii=0;ii<numberOfLayers;ii++)
	{
		layers[ii]=gsl_matrix_alloc(ii>0?numberOfLayersPoints[ii-1]:1,numberOfLayersPoints[ii]);
		
		if(ii==0)
		{
			for(uint32_t jj=0;jj<numberOfLayersPoints[ii];jj++)
			{
				gsl_matrix_set(layers[ii],0,jj,randomUniform(-0.01,0.01));
				
			}
		}
		else
		{
			for(uint32_t jj=0;jj<numberOfLayersPoints[ii];jj++)
			{
				for(uint32_t kk=0;kk<numberOfLayersPoints[ii-1];kk++)
				{
					gsl_matrix_set(layers[ii],kk,jj,randomUniform(-0.01,0.01));
				}
			}
		}
	}
	return layers;
}

void unloadLayers(gsl_matrix** layers,uint8_t numberOfLayers)
{
	for(uint8_t ii=0;ii<numberOfLayers;ii++)
	{
		gsl_matrix_free(layers[ii]);
	}
	free(layers);
}

double calculateLoss(gsl_matrix* probabilities,uint8_t* labels, int32_t id)
{
	double loss=0.0;
	double yy,yy2;
	uint8_t label=labels[id];
	for(uint8_t jj=0;jj<10;jj++)
	{
		yy=gsl_matrix_get(probabilities,0,jj);
		yy2=(label==jj)?1.0:0.0;
		loss-=(yy2*log(yy));
	}
	
	return loss;
}

void forwardPass(int32_t id,uint8_t* data, int32_t dataSize,gsl_matrix** layers,gsl_matrix* probabilities,gsl_matrix* C)
{
	int32_t offset=id*dataSize;
	for(int ii=0;ii<dataSize;ii++)
	{
		gsl_matrix_set(layers[0],0,ii,((double)data[ii+offset])/255.0);
	}
	
	
	//find what image is most similiar to
	//multiply input layer by matrix
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,1.0,layers[0],layers[1],0.0, C);
	//fprintf(stderr,"layers[0]:(%lu,%lu), layers[1]:(%lu,%lu)=C:(%lu,%lu)\n",layers[0]->size1,layers[0]->size2,layers[1]->size1,layers[1]->size2,C->size1,C->size2);
	
	//Normalize
	for(int ii=0;ii<256;ii++)
	{
		gsl_matrix_set(C,0,ii,activation(gsl_matrix_get(C,0,ii)));
	}
	//multiply intermediate layer by matrix and return probabilities in output layer by matrix
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,1.0, C,layers[2],0.0, probabilities);
	
	//Normalize
	for(int jj=0;jj<10;jj++)
	{
		gsl_matrix_set(probabilities,0,jj,activation(gsl_matrix_get(probabilities,0,jj)));
	}
	
	
}

void backwardPass(int32_t id,uint8_t* data,uint8_t* labels,int32_t dataSize,gsl_matrix** layers,gsl_matrix* probabilities,double rate,gsl_matrix* delta1,gsl_matrix* delta2,gsl_matrix* error1,gsl_matrix* error2,gsl_matrix* A,gsl_matrix* B,gsl_matrix* C)
{
	//gsl_matrix* delta2=gsl_matrix_alloc(1,10);
	//gsl_matrix* error2=gsl_matrix_alloc(1,10);
	double tmp=0.0;
	//double loss=0.0;
	for(uint8_t jj=0;jj<10;jj++)
	{
		tmp=gsl_matrix_get(probabilities,0,jj);
		//loss-=(((labels[id]==jj)?1.0:0.0)*log(tmp));
		
		gsl_matrix_set(error2,0,jj,-(((labels[id]==jj)?1.0:-1.0/9.0)*log(tmp)));
		//gsl_matrix_set(error2,0,jj,labels[id]==jj?(tmp-1.0):(tmp));
		gsl_matrix_set(delta2,0,jj,deActivation(tmp));
	}
	
	//printOther(10,probabilities,"Probab");
	//printOther(10,delta2,"Delta2");
	//printOther(10,error2,"Error2");
	gsl_matrix_mul_elements(delta2, error2);
	//gsl_matrix_scale(delta2,loss);
	//printOther(10,delta2,"Delta2");	
	
	
	//gsl_matrix* A=gsl_matrix_alloc(256,10);
	//fprintf(stderr,"C:(%lu,%lu)*delta2:(%lu,%lu)=A:(%lu,%lu)\n",C->size1,C->size2,delta2->size1,delta2->size2,A->size1,A->size2);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans,rate,C,delta2,0.0, A);
	
	//gsl_matrix* error1=gsl_matrix_alloc(1,256);
	//gsl_matrix* delta1=gsl_matrix_alloc(1,256);
	//fprintf(stderr,"delta2:(%lu,%lu)*layers[2]:(%lu,%lu)=error1:(%lu,%lu)\n",delta2->size1,delta2->size2,layers[2]->size1,layers[2]->size2,error1->size1,error1->size2);
	gsl_blas_dgemm(CblasNoTrans, CblasTrans,1.0,delta2,layers[2],0.0,error1);
		
	for(uint8_t jj=0;jj<10;jj++)
	{
		tmp=gsl_matrix_get(C,0,jj);
		gsl_matrix_set(delta1,0,jj,deActivation(tmp));
	}
	
	//fprintf(stderr,"delta1:(%lu,%lu)*error1:(%lu,%lu)\n",delta1->size1,delta1->size2,error1->size1,error1->size2);
	gsl_matrix_mul_elements(delta1, error1);
	
	//gsl_matrix* B=gsl_matrix_alloc(dataSize,256);
	//fprintf(stderr,"layers[0]:(%lu,%lu)*delta1:(%lu,%lu)=B:(%lu,%lu)\n",layers[0]->size1,layers[0]->size2,delta1->size1,delta1->size2,B->size1,B->size2);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans,rate,layers[0],delta1,0.0, B);
	
	//fprintf(stderr,"layer[1]:(%lu,%lu)+B:(%lu,%lu)\n",layers[1]->size1,layers[1]->size2,B->size1,B->size2);
	//gsl_matrix_scale(B,rate);
	gsl_matrix_add(layers[1], B);
	
	//fprintf(stderr,"layer[2]:(%lu,%lu)+A:(%lu,%lu)\n",layers[2]->size1,layers[2]->size2,A->size1,A->size2);
	//gsl_matrix_scale(A,rate);
	gsl_matrix_add(layers[2], A);
	
	/*gsl_matrix_free(A);
	gsl_matrix_free(B);
	gsl_matrix_free(error1);
	gsl_matrix_free(error2);
	gsl_matrix_free(delta1);
	gsl_matrix_free(delta2);*/
}

