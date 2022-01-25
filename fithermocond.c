/*
curve fitting calculation of thermal_conductivity as a function of temperature
*/
#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include <float.h>

/* generate random number between -0.005 and 0.005 */
#define RNG_UNIFORM() (rand()/(double)RAND_MAX - 0.5)/10

 double pi=3.14159265;
 double kb=1.38e-23;
 double hb=1.05e-34;
 double hok=7.641812E-12; //get the value of ratio hb/kb;

 float v0=3700.0, v1;
 double  L0=1.054e-5, L1;
 double  A0=2.79e-43, A1;
 double  B0=5.38e-18, B1;
 float  D0=370.0, D1;

 int DSize = 0;
 char * line = NULL;
 size_t len = 0;
 ssize_t read;

/*
experimental thermoconductivity data,
X0=temperature in kelvin,Y0=thermocond in W/K*m
*/
double X0[300],Y0[300];

 double thermocond(double T, float v,  double L,  double A,  double B, float D)
{
   int x, m=1000;
   double w;
   double  Sum=0.0e0, Integral, thermal_conductivity;
   double  numerator, denominator, Integrand;
   double  n1, n2, d1, d2, K;
   double  w_range = D/hok - 0.0;
   double  dw = w_range/m;

   for ( x=1; x<=m; ++x)
    {w=x*dw;
     n1=pow(w, 4);
     n2=exp(hok*w/T);
     d1=pow(exp(hok*w/T)-1,2);
     d2=v/L+A*pow(w,4)+B*w*w*T*exp(-D/(3*T));

     numerator= n1*n2;
     denominator= d1*d2;
     Integrand =numerator/denominator;
    //   printf("T=%3.0f, n=%Le, d=%Le, Integrand=%Le \n", T, numerator, denominator, Integrand);
     Sum = Sum + Integrand;
     }

    Integral = Sum*dw;
    K= hok*hb/(2*pi*pi*v*T*T);
    thermal_conductivity = K*Integral;
  //printf("T=%f, thermocond=%lf\n", T, thermal_conductivity);
  return thermal_conductivity;
}


/*squared difference between exper.data and simul.data*/
 double  diff(float v,  double L,  double A,  double B, float D)
{  int i;
   double sdiff=0.0, tot=0.0, sumY0=0.0;
   for( i=0; i<DSize; ++i)
    {
    sdiff=pow((thermocond(X0[i], v, L, A, B, D)-Y0[i]),2);
    tot =tot + sdiff;
    sumY0 += Y0[i];
    }
 //  printf("sdiff=%Lf, v=%f, L=%Le, A=%Le, B=%Le, D=%f\n", tot, v, L, A, B, D);
   return tot/sumY0;
}

#include <errno.h>
#include <stdint.h>

// if typedef doesn't exist (msvc, blah)
typedef intptr_t ssize_t;

ssize_t getline(char **lineptr, size_t *n, FILE *stream) {
    size_t pos;
    int c;

    if (lineptr == NULL || stream == NULL || n == NULL) {
        errno = EINVAL;
        return -1;
    }

    c = getc(stream);
    if (c == EOF) {  return -1;  }

    if (*lineptr == NULL) {
        *lineptr = malloc(128);
        if (*lineptr == NULL) { return -1; }
        *n = 128;
    }

    pos = 0;
    while(c != EOF) {
        if (pos + 1 >= *n) {
            size_t new_size = *n + (*n >> 2);
            if (new_size < 128) {
                new_size = 128;
            }
            char *new_ptr = realloc(*lineptr, new_size);
            if (new_ptr == NULL) { return -1; }
            *n = new_size;
            *lineptr = new_ptr;
        }

        ((unsigned char *)(*lineptr))[pos ++] = c;
        if (c == '\n') {
            break;
        }
        c = getc(stream);
    }

    (*lineptr)[pos] = '\0';
    return pos;
}

int main(int argc, char **argv)
{
 int i=0, j=0;
 double tot0=0.0, tot1=0.0;

 if (argc != 2)
    { printf("Usage: fithermocond inputfile\n"); }

 FILE *myFile;
 myFile = fopen(argv[1], "r");

 if (myFile == NULL)
        exit(EXIT_FAILURE);
 while ((read = getline(&line, &len, myFile)) != -1) {
        printf("Retrieved line of length %zu:  ", read);
        printf("%s \n", line);
        DSize++;
       }

 rewind(myFile);
 //printf("\n size=%d\n",size);
  for (i = 0; i < DSize; ++i)
  {
      fscanf(myFile, "%lf%lf", &X0[i], &Y0[i]);
      printf("Input %d: %lf %lf\n",i, X0[i], Y0[i]);
   //   printf("\n DSize=%d\n",DSize);
  }

  FILE *fp = fopen("result.txt", "w");
  if (fp == NULL)
     { printf("Error opening OUTPUT file!\n");
       exit(1);
     }

 do{
   tot0=diff(v0, L0, A0, B0, D0);

   v1=v0*(1+RNG_UNIFORM());
   L1=L0*(1+RNG_UNIFORM());
   A1=A0*(1+RNG_UNIFORM());
   B1=B0*(1+RNG_UNIFORM());
   D1=D0*(1+RNG_UNIFORM());
   tot1=diff(v1, L1, A1, B1, D1);

   if (tot1<tot0)
     {v0=v1; L0=L1; A0=A1; B0=B1; D0=D1;
      j++;
      printf("Cycle%d (err=%Lf): v=%f, L=%Le, A=%Le, B=%Le, D=%f\n", j, tot1, v0, L0, A0, B0, D0);
     }

   }while(fabsl(tot0-tot1)>1e-5 || tot1 >0.05);


for (i = 0; i<DSize; i=i+1){
   //   printf("DSize=%d",DSize);
      fprintf(fp, "%lf %lf\n", X0[i], thermocond(X0[i], v0, L0, A0, B0, D0));
   //   printf("Output%d: %lf %lf\n", i, X0[i], thermocond(X0[i], v0, L0, A0, B0, D0));
    }

  fclose(myFile);
  fclose(fp);
  if (line)
     free(line);

 printf("Results: v=%f, L=%Le, A=%Le, B=%Le, D=%f\n", v0, L0, A0, B0, D0);
 printf("Simulated curve is listed in result.txt !");
  exit(EXIT_SUCCESS);
//return 0;
}

