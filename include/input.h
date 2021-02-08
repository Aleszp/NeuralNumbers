#ifndef _INPUT_H_
#define _INPUT_H_

#include <stdint.h>

enum ERRCODES{OK=0,GENERALERROR,ARGUMENTERROR,PATHERROR,FILEERROR,SIZEERROR};

uint8_t* loadData(char* path,uint32_t* ucount,uint32_t* uheight,uint32_t* uwidth);
uint8_t* loadLabels(char* path,uint32_t* ucount);
int32_t fixEndianness(int32_t origBuffer32);

#endif
