#include <stdlib.h>
#include <stdio.h>
#include "output.h"

void printDigit(uint8_t* trainingData,int32_t height,int32_t width,int32_t id,uint8_t digit)
{
	uint8_t* pointer=trainingData+height*width*id;	 //create pointer to beginning of this particular digit
	
	for(int32_t yy=0;yy<height;yy++)
	{
		char digits[]="00";
		digits[0]='0'+digit;
		digits[1]='0'+digit;
		for(int32_t xx=0;xx<width;xx++)
		{
			//fprintf(stdout,"%02x",trainingData[xx+width*yy]);
			fprintf(stdout,"%s",pointer[xx+width*yy]>127?digits:"  ");
		}
		fprintf(stdout,"\n");
	}
}

void printProbabilities(uint8_t* labels,gsl_matrix* probabilities,uint32_t id)
{
	//fprintf(stderr,"Probabilities(%lu,%lu):\n",probabilities->size1,probabilities->size2);
	double sum=0.0;
	double tmp=0.0;
	fprintf(stdout,"Label: %u, probabilities: ",labels[id]);
	for(uint8_t jj=0;jj<10;jj++)
	{
		tmp=gsl_matrix_get(probabilities,0,jj);
		fprintf(stdout,"%0.4lf,",tmp);
		sum+=tmp;
	}
	fprintf(stdout,"sum=%0.2lf, ",sum);
}

void printOther(uint8_t max,gsl_matrix* what,char* text)
{
	//fprintf(stderr,"Probabilities(%lu,%lu):\n",probabilities->size1,probabilities->size2);
	
	fprintf(stdout,"%s: ",text);
	for(uint8_t jj=0;jj<max;jj++)
	{
		fprintf(stdout,"%0.5lf,",gsl_matrix_get(what,0,jj));
	}
	fprintf(stdout,"\n");
}

void printLayers(gsl_matrix** layers,uint8_t numberOfLayers,uint32_t* numberOfLayersPoints)
{
	fprintf(stdout,"\n");
	for(uint8_t ii=1;ii<numberOfLayers;ii++)
	{
		if(ii==0)
		{
			fprintf(stdout,"Layer %u ^T (%u,%u):\n",ii,1,numberOfLayersPoints[ii]);
			for(uint32_t jj=0;jj<numberOfLayersPoints[ii];jj++)
			{
				fprintf(stdout,"%0.5lf;",gsl_matrix_get(layers[ii],0,jj));
			}
			fprintf(stdout,"\n");
		}
		else
		{
			fprintf(stdout,"Layer %u (%u,%u):\n",ii,numberOfLayersPoints[ii-1],numberOfLayersPoints[ii]);
			
			for(uint32_t jj=0;jj<numberOfLayersPoints[ii];jj++)
			{
				for(uint32_t kk=0;kk<numberOfLayersPoints[ii-1];kk++)
				{
					fprintf(stdout,"%0.5lf;",gsl_matrix_get(layers[ii],kk,jj));
				}
				fprintf(stdout,"\n");
			}
			fprintf(stdout,"\n");
		}
	}
	fprintf(stdout,"\n");
}

void printWeights(gsl_matrix** weights,uint8_t numberOfLayers)
{
	fprintf(stdout,"\n");
	for(uint8_t ii=0;ii<numberOfLayers-1;ii++)
	{
		fprintf(stdout,"Weight %u (%lu,%lu):\n",ii,weights[ii]->size1,weights[ii]->size2);
			
		for(uint32_t jj=0;jj<weights[ii]->size2;jj++)
		{
			for(uint32_t kk=0;kk<weights[ii]->size1;kk++)
			{
				fprintf(stdout,"%0.5lf;",gsl_matrix_get(weights[ii],kk,jj));
				if((kk%28==27 && ii==0)||(kk%16==15 && ii==1))
				{
					fprintf(stdout,"\n");
				}
			}
			fprintf(stdout,"\n\n");
		}
		fprintf(stdout,"\n\n\n");
	}
	fprintf(stdout,"\n");
}
