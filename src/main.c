/*
 * AI implementation of handwritten digit recognition written in C with GSL_BLAS.
 * 
 * Author: Aleksander Szpakiewicz-Szatan
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "input.h"
#include "output.h"
#include "network.h"

int main(int argc, char** argv)
{
	srand (time (NULL));	//Prepare random number generator
	double rate=1.0;	//Rate of error correction
	if(argc!=5)
	{
		fprintf(stderr,"Wrong number of parameters.\n");
		return ARGUMENTERROR;
	}
	uint32_t count=0,count2=0,height=0,width=0;
	uint8_t* trainingData=loadData(argv[1],&count,&height,&width);	//Load training data
	uint8_t* trainingLabels=loadLabels(argv[2],&count2);			//Load labels for training data
	if(count!=count2)												//Check if number of labels fits number of images
	{
		fprintf(stderr,"Missmatched number of counts, training data: %i, labels: %i.\n",count, count2);
		return GENERALERROR;
	}
	//Test if image was loaded correctrly, display with one-bit resolution in console
	//int32_t ii=1;	//id of tested image [0;count)
	//printDigit(trainingData,height,width,ii,trainingLabels[ii]);
	
	double loss=0.0;
	double lossSingle=0.0;
	//double loss_accepted=0.1;//((double)round(((double)count)*0.01));
	
	uint8_t	numberOfLayers=3;
	uint32_t numberOfLayersPoints[]={width*height,HIDDEN,10};
	
	gsl_matrix** layers=prepareLayers(numberOfLayers,numberOfLayersPoints);
	//printLayers(layers,numberOfLayers,numberOfLayersPoints);
	gsl_matrix* probabilities=gsl_matrix_alloc(1,10);
	
	//int32_t id=0;
	gsl_matrix* C=gsl_matrix_calloc(1,HIDDEN);
	//forwardPass(id,trainingData,height*width,layers,probabilities,C);
	//fprintf(stdout,"Test loss function for %i: %lf;\n",trainingLabels[0],calculateLoss(&probabilities,trainingLabels,0));
	//printProbabilities(probabilities,0);
	gsl_matrix* delta1=gsl_matrix_alloc(1,HIDDEN);
	gsl_matrix* delta2=gsl_matrix_alloc(1,10);
	gsl_matrix* error1=gsl_matrix_alloc(1,HIDDEN);
	gsl_matrix* error2=gsl_matrix_alloc(1,10);
	gsl_matrix* A=gsl_matrix_alloc(HIDDEN,10);
	gsl_matrix* B=gsl_matrix_alloc(height*width,HIDDEN);
	
	uint8_t max=6;
	//train data
	for(uint8_t	jj=0;jj<max;jj++)
	{
		loss=0.0;
		for(uint32_t ii=0;ii<count;ii++)
		{
			//for(uint8_t	jj=0;jj<max;jj++)
			{
				forwardPass(ii,trainingData,height*width,layers,probabilities,C);
				backwardPass(ii,trainingData,trainingLabels,height*width,layers,probabilities,rate,delta1,delta2,error1,error2,A,B,C);
				printProbabilities(trainingLabels,probabilities,ii);
				lossSingle=calculateLoss(probabilities,trainingLabels,ii);
				loss+=lossSingle;
				//forwardPass(ii,trainingData,height*width,layers,probabilities,C);
				//printProbabilities(trainingLabels,probabilities,ii);
			}
			//fprintf(stdout,"\n");
			
			if(jj==(max-1))
			{
				lossSingle=calculateLoss(probabilities,trainingLabels,ii);
				fprintf(stdout,"Loss function calculation %lf.\n\n",lossSingle);
			}
			if((ii%(count/20))==0)
				fprintf(stderr,"*");
				
			//break;
		}
		//break;
		loss/=(double)count;
		fprintf(stderr,"\n");
		fprintf(stderr,"Pass %u/%u finished. Average loss per digit: %lf.\n",jj+1,max,loss);
		fprintf(stdout,"Pass %u/%u finished. Average loss per digit: %lf.\n",jj+1,max,loss);
		fprintf(stdout,"\n");
	}
	
	gsl_matrix_free(A);
	gsl_matrix_free(B);
	gsl_matrix_free(error1);
	gsl_matrix_free(error2);
	gsl_matrix_free(delta1);
	gsl_matrix_free(delta2);
	
	/*do
	{
		
		backwardPass(id,trainingData,trainingLabels,height*width,layers,probabilities,rate,C);
		forwardPass(id,trainingData,height*width,layers,probabilities,C);
		printProbabilities(probabilities,0);
		loss=calculateLoss(&probabilities,trainingLabels,0);
		fprintf(stdout,"Loss function calculation %lf with limit of %lf.\n",loss,loss_accepted);
		//break;	//break loop until implementation is finished
	}
	while(loss>loss_accepted);	//train data as long as loss is higher than accepted value (i.e. 0.1%)*/
	//fprintf(stdout,"Loss function calculation %lf with limit of %lf.\n",loss,loss_accepted);
	//printLayers(layers,numberOfLayers,numberOfLayersPoints);
	/*uint32_t count3=0,count4=0,height2=0,width2=0;
	uint8_t* testData=loadData(argv[3],&count3,&height2,&width2);	//Load training data
	uint8_t* testLabels=loadLabels(argv[4],&count4);			//Load labels for training data
	if(count3!=count4)												//Check if number of labels fits number of images
	{
		fprintf(stderr,"Missmatched number of counts, test data: %i, labels: %i.\n",count3, count4);
		return GENERALERROR;
	}
	if(height!=height2 || width!=width2)												//Check if test images dimensions fit training images dimensions
	{
		fprintf(stderr,"Missmatched dimensions. Training data %i x %i, test data %i x %i.\n",width, height, width2, height2);
		return GENERALERROR;
	}*/
	
	//test loss function:
	/*{
		uint8_t testLabels[]={1,7};
		gsl_matrix* testProbabilities[2];
		double tmp[]={0.1,0.8,0.3,0.4,0.1,0.1,0.2,0.1,0.4,0.1};
		gsl_matrix* testProbabilities0=gsl_matrix_calloc(1,10);
		gsl_matrix* testProbabilities1=gsl_matrix_calloc(1,10);
		testProbabilities[0]=testProbabilities0;
		testProbabilities[1]=testProbabilities1;
		fprintf(stdout,"SIZE1: %lu, SIZE2: %lu, TDA: %lu\n",testProbabilities1->size1,testProbabilities1->size2,testProbabilities1->tda);
		for(int ii=0;ii<10;ii++)
		{
			gsl_matrix_set(testProbabilities[0],0,ii,tmp[ii]);
			gsl_matrix_set(testProbabilities[1],0,ii,tmp[ii]);
		}
		
									
		fprintf(stdout,"Test loss function for %i: %lf;\n",testLabels[0],calculateLoss(testProbabilities[0],testLabels,0));
		fprintf(stdout,"Test loss function for %i: %lf;\n",testLabels[1],calculateLoss(testProbabilities[1],testLabels,1));
		
		gsl_matrix_free(testProbabilities0);
		gsl_matrix_free(testProbabilities1);
	}*/
	
	//Free memory
	free(trainingData);
	free(trainingLabels);
	//free(testData);
	//free(testLabels);
	
	gsl_matrix_free(probabilities);
	gsl_matrix_free(C);
	unloadLayers(layers,numberOfLayers);
	return OK;
}
