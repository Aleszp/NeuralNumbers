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
#include <time.h>

#include "input.h"
#include "output.h"
#include "network.h"

int main(int argc, char** argv)
{
	gnuNotice();
	srand (time (NULL));	//Prepare random number generator
	double rate=0.2;		//Rate of error correction
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
		
	uint8_t	numberOfLayers=3;
	uint32_t numberOfLayersPoints[]={width*height,width*height/2,10};
	uint8_t	activations[]={SIGMOID,SOFTMAX};
	
	gsl_matrix** layers=prepareLayers(numberOfLayers,numberOfLayersPoints);
	gsl_matrix** biases=prepareLayers(numberOfLayers,numberOfLayersPoints);
	gsl_matrix** weights=prepareWeights(numberOfLayers,numberOfLayersPoints);
		
	//allocate memory for backward pass temporary matrices to reduce allocation/deallocation of memory	
	gsl_matrix** dLayers=prepareLayers(numberOfLayers,numberOfLayersPoints);	
	gsl_matrix** dBiases=prepareLayers(numberOfLayers,numberOfLayersPoints);
	gsl_matrix** dWeights=prepareWeights(numberOfLayers,numberOfLayersPoints);
	
	const uint8_t max=3;		//set maximum number of iterations
	uint32_t correct=0;			//count how many digits were recognised correctly
	uint32_t isCorrect=0;
	
	//train network
	fprintf(stdout,"Training network with %u passes of %u digits:\n",max,count);
	fprintf(stderr,"Training network with %u passes of %u digits:\n",max,count);
	for(uint8_t	jj=0;jj<max;jj++)
	{
		correct=0;
		loss=0.0;
		//count*=max;
		for(uint32_t ii=0;ii<count;ii++)
		{
			//for(uint8_t	jj=0;jj<max;jj++)
			{
				forwardPass(ii,trainingData,layers,weights,biases,numberOfLayers,activations);
				lossSingle=calculateLoss(layers[numberOfLayers-1],trainingLabels,ii);
				loss+=lossSingle;
				isCorrect=testImage(ii,trainingLabels,layers[numberOfLayers-1]);
				printProbabilities(trainingLabels,layers[numberOfLayers-1],ii,isCorrect);
				fprintf(stdout,"loss=%0.4lf.\n",lossSingle);
				
				backwardPass(ii,trainingData,trainingLabels,layers,weights,biases,rate,dWeights,dLayers,dBiases,numberOfLayers,activations);
				
				if(isCorrect!=-1)
				{
					//printDigit(trainingData,height,width,ii/3,trainingLabels[ii]);
				}
				else
				{
					correct++;
				}
			}
			//fprintf(stdout,"Loss function calculation %lf.\n",lossSingle);
			
			if((ii%(count/20))==0)
				fprintf(stderr,"*");
		}
		fprintf(stderr,"\n");
		
		loss/=(double)count;
		rate/=2.0;
		fprintf(stderr,"Pass %u/%u finished. Correct %lf %% (%u out of %u). Average loss per digit: %lf.\n",jj+1,max,(100.0*(double)correct)/((double)(count)),correct,count,loss);
		fprintf(stdout,"Pass %u/%u finished. Correct %lf %% (%u out of %u). Average loss per digit: %lf.\n",jj+1,max,(100.0*(double)correct)/((double)(count)),correct,count,loss);
		fprintf(stdout,"\n");
		//break;
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
		forwardPass(ii,testData,layers,weights,biases,numberOfLayers,activations);
		
		isCorrect=testImage(ii,testLabels,layers[numberOfLayers-1]);
		printProbabilities(testLabels,layers[numberOfLayers-1],ii,isCorrect);
			
		lossSingle=calculateLoss(layers[numberOfLayers-1],testLabels,ii);
		loss+=lossSingle;
		fprintf(stdout,"loss=%0.4lf.\n",lossSingle);
		if(isCorrect!=-1)
		{
			printDigit(testData,height,width,ii,testLabels[ii]);
		}
		else
		{
			correct++;
		}
		if((ii%(count3/20))==0)
			fprintf(stderr,"*");
	}
	fprintf(stderr,"\n");
	loss/=(double)count3;
		
	fprintf(stderr,"Test digits compared. Correct %lf %% (%u out of %u). Average loss per digit: %lf.\n",(100.0*(double)correct)/((double)count3),correct,count3,loss);
	fprintf(stdout,"Test digits compared. Correct %lf %% (%u out of %u). Average loss per digit: %lf.\n",(100.0*(double)correct)/((double)count3),correct,count3,loss);
	fprintf(stdout,"\n");
	
	//show layers weights
	printWeights(weights,numberOfLayers);
	printBiases(biases,numberOfLayers);
	
	//Free memory
	free(trainingData);
	free(trainingLabels);
	free(testData);
	free(testLabels);
	
	unloadLayers(layers,numberOfLayers);
	unloadLayers(dLayers,numberOfLayers);
	
	unloadLayers(biases,numberOfLayers);
	unloadLayers(dBiases,numberOfLayers);
	
	unloadWeights(weights,numberOfLayers);
	unloadWeights(dWeights,numberOfLayers);
	
	return OK;
}
