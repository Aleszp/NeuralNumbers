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
	double rate=0.1;	//Rate of error correction
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
		
	uint8_t	numberOfLayers=2;
	uint32_t numberOfLayersPoints[]={width*height+1,10};//HIDDEN,10};
	
	gsl_matrix** layers=prepareLayers(numberOfLayers,numberOfLayersPoints);
	
	gsl_matrix* probabilities=gsl_matrix_alloc(1,numberOfLayersPoints[1]);
		
	//allocate memory for backward pass temporary matrices to reduce allocation/deallocation of memory	
	//gsl_matrix* delta1=gsl_matrix_alloc(numberOfLayersPoints[1],numberOfLayersPoints[2]);
	//gsl_matrix* error1=gsl_matrix_alloc(1,numberOfLayersPoints[2]);
	
	gsl_matrix* delta1=gsl_matrix_alloc(numberOfLayersPoints[0],numberOfLayersPoints[1]);
	gsl_matrix* error1=gsl_matrix_alloc(1,numberOfLayersPoints[1]);
	
	gsl_matrix* realProbabilities=gsl_matrix_alloc(1,numberOfLayersPoints[1]);
	
	uint8_t max=3;
	uint32_t correct=0;
	
	//train network
	fprintf(stdout,"Training network with %u passes of %u digits:\n",max,count);
	fprintf(stderr,"Training network with %u passes of %u digits:\n",max,count);
	for(uint8_t	jj=0;jj<max;jj++)
	{
		correct=0;
		loss=0.0;
		for(uint32_t ii=0;ii<count;ii++)
		{
			forwardPass(ii,trainingData,numberOfLayersPoints[0]-1,layers,probabilities);
			printProbabilities(trainingLabels,probabilities,ii);
			backwardPass(ii,trainingData,trainingLabels,numberOfLayersPoints[0],layers,probabilities,rate,delta1,error1,realProbabilities);
			lossSingle=calculateLoss(probabilities,trainingLabels,ii);
			loss+=lossSingle;
			correct+=testImage(ii,trainingLabels,probabilities);
			
			lossSingle=calculateLoss(probabilities,trainingLabels,ii);
			fprintf(stdout,"Loss function calculation %lf.\n\n",lossSingle);
			
			if((ii%(count/20))==0)
				fprintf(stderr,"*");
		}
		fprintf(stderr,"\n");
		
		loss/=(double)count;
		
		fprintf(stderr,"Pass %u/%u finished. Correct %lf %% (%u out of %u). Average loss per digit: %lf.\n",jj+1,max,(100.0*(double)correct)/((double)count),correct,count,loss);
		fprintf(stdout,"Pass %u/%u finished. Correct %lf %% (%u out of %u). Average loss per digit: %lf.\n",jj+1,max,(100.0*(double)correct)/((double)count),correct,count,loss);
		fprintf(stdout,"\n");
	}
	correct=0;
	
	//load test data
	uint32_t count3=0,count4=0,height2=0,width2=0;
	uint8_t* testData=loadData(argv[3],&count3,&height2,&width2);	//Load test data
	uint8_t* testLabels=loadLabels(argv[4],&count4);				//Load labels for test data
	if(count3!=count4)												//Check if number of labels fits number of images
	{
		fprintf(stderr,"Missmatched number of counts, test data: %i, labels: %i.\n",count3, count4);
		return GENERALERROR;
	}
	if(height!=height2 || width!=width2)							//Check if test images dimensions fit training images dimensions
	{
		fprintf(stderr,"Missmatched dimensions. Training data %i x %i, test data %i x %i.\n",width, height, width2, height2);
		return GENERALERROR;
	}
	
	//test network
	loss=0.0;
	fprintf(stdout,"Testing network with %u new digits:\n",count3);
	fprintf(stderr,"Testing network with %u new digits:\n",count3);
	for(uint32_t ii=0;ii<count3;ii++)
	{
		forwardPass(ii,testData,numberOfLayersPoints[0]-1,layers,probabilities);
		printProbabilities(testLabels,probabilities,ii);
		correct+=testImage(ii,testLabels,probabilities);
		lossSingle=calculateLoss(probabilities,testLabels,ii);
		loss+=lossSingle;
		if((ii%(count3/20))==0)
			fprintf(stderr,"*");
	}
	fprintf(stderr,"\n");
	loss/=(double)count3;
		
	fprintf(stderr,"Test digits compared. Correct %lf %% (%u out of %u). Average loss per digit: %lf.\n",(100.0*(double)correct)/((double)count3),correct,count3,loss);
	fprintf(stdout,"Test digits compared. Correct %lf %% (%u out of %u). Average loss per digit: %lf.\n",(100.0*(double)correct)/((double)count3),correct,count3,loss);
	fprintf(stdout,"\n");
	
	//show layers weights
	printLayers2(layers,numberOfLayers,numberOfLayersPoints);
	
	//Free memory
	free(trainingData);
	free(trainingLabels);
	free(testData);
	free(testLabels);
	
	gsl_matrix_free(probabilities);
	gsl_matrix_free(realProbabilities);
	gsl_matrix_free(delta1);
	gsl_matrix_free(error1);
	//gsl_matrix_free(delta2);
	//gsl_matrix_free(error2);
	
	unloadLayers(layers,numberOfLayers);
	return OK;
}
