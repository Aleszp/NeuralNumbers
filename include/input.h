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

#ifndef _INPUT_H_
#define _INPUT_H_

#include <stdint.h>

enum ERRCODES{OK=0,GENERALERROR,ARGUMENTERROR,PATHERROR,FILEERROR,SIZEERROR};

uint8_t* loadData(char* path,uint32_t* ucount,uint32_t* uheight,uint32_t* uwidth);
uint8_t* loadLabels(char* path,uint32_t* ucount);
int32_t fixEndianness(int32_t origBuffer32);

#endif
