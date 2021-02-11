/*
 *   AI implementation of handwritten digit recognition written in C with GSL_BLAS.
 * 
 *   Author: Aleksander Szpakiewicz-Szatan (c) 2021
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
#include "input.h"

uint8_t* loadData(char* path,uint32_t* ucount,uint32_t* uheight,uint32_t* uwidth)
{
	int32_t height[1];
	int32_t width[1];
	int32_t count[1];
	
	FILE* input=fopen(path,"rb");
	if(!input)
	{
		fprintf(stderr,"Could not open file: %s\n",path);
		exit(PATHERROR);
	}
	
	int32_t buffer32;
	
	fread(&buffer32,sizeof(uint32_t),1,input);
	buffer32=fixEndianness(buffer32);
	
	if(buffer32!=0x00000803)
	{
		fprintf(stderr,"Wrong type of file, magic number is: 0x%08X, but should be 0x%08X.\n", buffer32, 0x00000803);
		exit(FILEERROR);
	}
	
	fread(&buffer32,sizeof(uint32_t),1,input);
	*count=fixEndianness(buffer32);
	
	fread(&buffer32,sizeof(uint32_t),1,input);
	*width=fixEndianness(buffer32);
	
	fread(&buffer32,sizeof(uint32_t),1,input);
	*height=fixEndianness(buffer32);
	
	int32_t size=(*count)*(*width)*(*height);
	
	fprintf(stdout,"Counted %i items of %i x %i, which multiplied returned %i.\n", *count, *width, *height, size);

	if(size<0 || size<*count)
	{
		fprintf(stderr,"Too big file, 32bit integer overflow.\n Counted %i items of %i x %i, which multiplied returned %i.", *count, *width, *height, size);
		exit(SIZEERROR);
	}
	
	uint8_t* database=(uint8_t*)calloc(sizeof(uint8_t),size);
	int32_t test;
	test=fread(database,sizeof(uint8_t),size,input);
	fprintf(stdout,"Loaded %i bytes.\n",test);
	fclose(input);
	
	*uwidth=(uint32_t)(*width);
	*uheight=(uint32_t)(*height);
	*ucount=(uint32_t)(*count);
	
	return database;
}

uint8_t* loadLabels(char* path,uint32_t* ucount)
{
	int32_t count[1];
	FILE* input=fopen(path,"rb");
	if(!input)
	{
		fprintf(stderr,"Could not open file: %s\n",path);
		exit(PATHERROR);
	}
	
	int32_t buffer32;
	
	fread(&buffer32,sizeof(uint32_t),1,input);
	buffer32=fixEndianness(buffer32);
	
	if(buffer32!=0x00000801)
	{
		fprintf(stderr,"Wrong type of file, magic number is: 0x%08X, but should be 0x%08X.\n", buffer32, 0x00000801);
		exit(FILEERROR);
	}
	
	fread(&buffer32,sizeof(uint32_t),1,input);
	*count=fixEndianness(buffer32);
	
	fprintf(stdout,"Counted %i items.\n", *count);

	if(*count<0)
	{
		fprintf(stderr,"Too big file, 32bit integer overflow.\n Counted %i items.", *count);
		exit(SIZEERROR);
	}
	
	uint8_t* database=(uint8_t*)calloc(sizeof(uint8_t),*count);
	int32_t test;
	test=fread(database,sizeof(uint8_t),*count,input);
	fprintf(stdout,"Loaded %i bytes.\n",test);
	fclose(input);
	
	*ucount=(uint32_t)(*count);
	
	return database;
}

int32_t fixEndianness(int32_t origBuffer32)
{
	int32_t extraBuffer32=origBuffer32>>16;
	origBuffer32<<=16;
	origBuffer32|=(0xFF00&(extraBuffer32<<8))|(0x00FF&(extraBuffer32>>8));
	return origBuffer32;
}
