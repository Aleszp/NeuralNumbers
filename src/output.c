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
			fprintf(stdout,"%s",pointer[xx+width*yy]>63?digits:"  ");
		}
		fprintf(stdout,"\n");
	}
}

void printProbabilities(uint8_t* labels,gsl_matrix* probabilities,uint32_t id,int detected)
{
	//fprintf(stderr,"Probabilities(%lu,%lu):\n",probabilities->size1,probabilities->size2);
	double sum=0.0;
	double tmp=0.0;
	fprintf(stdout,"Label: %u, Detec: %i, prob: ",labels[id],detected==-1?labels[id]:detected);
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

void printBiases(gsl_matrix** biases,uint8_t numberOfLayers)
{
	fprintf(stdout,"\n");
	for(uint8_t ii=0;ii<numberOfLayers;ii++)
	{
		fprintf(stdout,"Bias^T %u (%lu,%lu)^T:\n",ii,biases[ii]->size1,biases[ii]->size2);
			
		for(uint32_t jj=0;jj<biases[ii]->size1;jj++)
		{
			for(uint32_t kk=0;kk<biases[ii]->size2;kk++)
			{
				fprintf(stdout,"%0.5lf;",gsl_matrix_get(biases[ii],jj,kk));
			}
			fprintf(stdout,"\n\n");
		}
		fprintf(stdout,"\n\n\n");
	}
	fprintf(stdout,"\n");
}

void gnuNotice()
{
	fprintf(stdout,"NeuralNumbers  Copyright (C) 2021  Aleksander Szpakiewicz-Szatan\nThis program comes with ABSOLUTELY NO WARRANTY.\nThis is free software, and you are welcome to redistribute it\nunder certain conditions.\n");
	fprintf(stdout,"For more information please read LICENSE file supplied\nor read <https://www.gnu.org/licenses/gpl-3.0.html>\n\n");
}
