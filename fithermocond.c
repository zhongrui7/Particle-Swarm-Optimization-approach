  /*
   * Implementing Particle Swarm Optimization (PSO) in the C language
   * Thermoconductivity-curve-fitting-in-Particle-Swarm-Optimization-approach
   * fithermocond.c in main
   * reference:
   * J. Callaway, Phys. Rev. 113, 1046 (1959). 
   * J. Callaway, Phys. Rev. 120, 1149 (1960).
   */

  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h>
  #include <math.h>
  #include <time.h>
  #include <omp.h>

  #define c1 1.495 //The acceleration factor is generally obtained from a large number of experiments
  #define c2 1.495
  #define maxgen  256  // number of iterations
  #define sizepop 128 // population size
  #define dim 5 // the dimension of the particle
  #define popmin 0 // Individual minimum value
  #define PI 3.1415926 
  #define RNG_UNIFORM() ((double)rand())/RAND_MAX-0.5  // random number in the range -0.5~+0.5

   double pop[sizepop][dim]; // define population array
   double V[sizepop][dim]; // Define the population velocity array
   double fitness[sizepop]; // Define the fitness array of the population
   double result[maxgen];  //Define an array to store the optimal value of the population for each iteration
   double pbest[sizepop][dim];  // location of individual extremes
   double gbest[dim]; //the location of the group extremum
   double fitnesspbest[sizepop]; //The value of individual extreme fitness
   double fitnessgbest; // group extreme fitness value
   double genbest[maxgen][dim]; //Each generation of optimal value-valued particles

   double kb=1.38e-23;   // boltzmann constant
   double hb=1.05e-34;   // Plank constant divided by 2π; the quantization of angular momentum
   double hok=7.641812E-12; //get the value of ratio hb/kb;

   int DSize = 0;
   char * line = NULL;
   size_t len = 0;
   ssize_t read;

   double X0[300],Y0[300];

 double thermocond(double T,  double v,  double L,  double A,  double B,  double D)
{
   int x, m=512;
   double w;
    double  Sum=0.0, Integral, thermal_conductivity;
    double  numerator, denominator, Integrand;
    double n1, n2, d1,d2;
    double  w_range = ( double) D/hok-0.0;
    double  dw =  ( double) w_range/m;

    /*
    int threadCount = 4;
    #pragma omp parallel num_threads(threadCount)
    #pragma omp parallel for reduction(+: Sum)
    */
    for (x=1; x<=m; ++x)
     { w=x*dw;
      n1=pow(w, 4);
      n2=exp(hok*w/T);
      d1=pow(exp(hok*w/T)-1,2);
      d2=v/L+A*pow(w,4)+B*w*w*T*exp(-D/(3*T));

      numerator= n1*n2;
      denominator= d1*d2;
      Integrand =numerator/denominator;
       //printf("T=%3.0f, n=%Le, d=%Le, Integrand=%Le \n", iT, numerator, denominator, Integrand);
      Sum += Integrand;
      }

     Integral = Sum*dw;
  //  printf("T=%3.0f K, Integr=%5.4Le\n",T,Integral);

     double   K= hok*hb/(2*PI*PI*v*T*T);
     thermal_conductivity = K*Integral;

  return thermal_conductivity;
}

/*
experimental thermoconductivity data,
X0=temperature in kelvin, Y0=thermocond in W/K*m

//fitness function
/* squared difference between exp. and simul.*/
double func( double * arr)
{  int i;
   double v=arr[0], D=arr[4];
   double L=arr[1], A=arr[2], B=arr[3];
   double sdiff=0.0, tot=0.0;
   for( i=0; i<DSize; i++)
    {
    sdiff=pow((thermocond(X0[i], v, L, A, B, D)-Y0[i]),2);
    tot =tot + sdiff;
    }
  // printf("sdiff=%Lf, v=%f, L=%Le, A=%Le, B=%Le, D=%f\n", tot, v, L, A, B, D);
   return tot;
}

  // Population initialization
  void pop_init(void)
  {
      for(int i=0;i<sizepop;i++)
      {
       pop[i][0]= 2700.0*(1+RNG_UNIFORM()); //v0=2700.0
       pop[i][1]= 1.054e-5*(1+RNG_UNIFORM()); //L0=1.054e-5
       pop[i][2]= 2.79e-43*(1+RNG_UNIFORM()); //A0=2.79e-43,
       pop[i][3]= 5.38e-18*(1+RNG_UNIFORM()); //B0=5.38e-18
       pop[i][4]= 387.0*(1+RNG_UNIFORM()); //D0=287.0,

          for(int j=0;j<dim;j++)
          {
            V[i][j]=pop[i][j]/100;
          }
          fitness[i] = func(pop[i]); //Calculate the fitness function value
      }
  }

  /*min() function definition*/ 
  double * min(double * fit, int size)
  {
      int index = 0; // initialization sequence number
      double min = *fit; // Initialize the smallest value as the first element of the fit array
      static double best_fit_index[2];
      for(int i=1;i<size;i++)
      {
          if(*(fit+i) < min)
              min = *(fit+i);
              index = i;
      }

      best_fit_index[0] = index;
      best_fit_index[1] = min;
      return best_fit_index;
    }

  /*** iterative optimization ***/ 
  void PSO_func(void)
  {
      pop_init();
      double * best_fit_index; // Used to store group extrema and its position (serial number)
      best_fit_index = min(fitness,sizepop); //find group extrema
      int index = (int)(*best_fit_index);
      // group extreme position
      for(int j=0;j<dim;j++)
      {
          gbest[j] = pop[index][j];
      }
      // individual extreme position
      for(int i=0;i<sizepop;i++)
      {
          for(int j=0;j<dim;j++)
          {
            pbest[i][j] = pop[i][j];
          }
      }
      // Individual extreme fitness value
      for(int i=0;i<sizepop;i++)
      {
          fitnesspbest[i] = fitness[i];
      }
      //group extreme fitness value
      double bestfitness = *(best_fit_index+1);
      fitnessgbest = bestfitness;

     //iterative optimization
     for(int i=0;i<maxgen;i++)
     {
       //float w=0.9-0.5*(i/maxgen)*(i/maxgen);
         for(int j=0;j<sizepop;j++)
         {
             //Velocity ​​update and particle update
             for(int k=0;k<dim;k++)
             {
                // velocity update
                 double rand1 = (double)rand()/RAND_MAX; //random number between 0 and 1
                 double rand2 = (double)rand()/RAND_MAX;
                 V[j][k] = 0.9*V[j][k] + c1*rand1*(pbest[j][k]-pop[j][k]) + c2*rand2*(gbest[k]-pop[j][k]);
                  // particle update
                 pop[j][k] = pop[j][k] + V[j][k];
                 if(pop[j][k] < popmin)
                     pop[j][k] = popmin;
            }
           fitness[j] = func(pop[j]); //The fitness value of the new particle
         }

         for(int j=0;j<sizepop;j++)
         {
             // Individual extreme value update
            if(fitness[j] < fitnesspbest[j])
             {
                 for(int k=0;k<dim;k++)
                 {
                     pbest[j][k] = pop[j][k];
                 }
                 fitnesspbest[j] = fitness[j];
             }
             // Population extrema update
            if(fitness[j] < fitnessgbest)
             {
                 for(int k=0;k<dim;k++)
                     gbest[k] = pop[j][k];
                 fitnessgbest = fitness[j];
             }
         }

         for(int k=0;k<dim;k++)
         {
             genbest[i][k] = gbest[k]; // The optimal value of each generation is the record of the particle position
          }
         result[i] = fitnessgbest; // The optimal value of each generation is recorded to the array
        printf("Cycle[%d]Error=%f\n",i, result[i]/DSize*100);
     }
 }

#include <errno.h>
#include <stdint.h>

/*if typedef doesn't exist (msvc, blah)*/ 
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

 //***** The main function *****// 
int main(int argc, char **argv)
 {
 int i=0;
 if (argc != 2)
    { printf("Usage: fithermocond inputfile\n"); }

 FILE *myFile;
 myFile = fopen(argv[1], "r");

 if (myFile == NULL)
        exit(EXIT_FAILURE);
 while ((read = getline(&line, &len, myFile)) != -1) {
        printf("Retrieved line of length %zu:  ", DSize);
        printf("%s \n", line);
        DSize++;
       }

 rewind(myFile);

 
  for (i = 0; i < DSize; ++i)
  {
      fscanf(myFile, "%lf%lf", &X0[i], &Y0[i]);
      printf("Input %d: %lf %lf\n",i, X0[i], Y0[i]);
  }

  FILE *fp = fopen("fitresult.txt", "w");
  if (fp == NULL)
     { printf("Error opening OUTPUT file!\n");
       exit(1);
     }

     clock_t start,finish; //the start and end time of the procedure
     start = clock(); //initialize the start time
     srand((unsigned)time(NULL)); // initialize the random number seeds
     PSO_func();

     double * best_arr;
     best_arr = min(result,maxgen);
     int best_gen_number = *best_arr; // the index number of the optimal value
     double best = *(best_arr+1); //the optimal value

  for (i = 0; i<DSize; i=i+1){
      fprintf(fp, "%lf %lf\n", X0[i], thermocond(X0[i], genbest[best_gen_number][0],genbest[best_gen_number][1], genbest[best_gen_number][2], genbest[best_gen_number][3], genbest[best_gen_number][4]));
      printf("Output%d: %lf %lf\n", i, X0[i], thermocond(X0[i], genbest[best_gen_number][0],genbest[best_gen_number][1], genbest[best_gen_number][2], genbest[best_gen_number][3], genbest[best_gen_number][4]));
      }

    printf("After iterating %d times, the optimal value is obtained at the %dth time, and the optimal value is:%f.\n",maxgen,best_gen_number+1,best);
    printf("The position where the optimal value is obtained is:\nv=%.5e, \nL=%.5e, \nA=%.5e, \nB=%.5e, \nD=%.5e\n",genbest[best_gen_number][0],genbest[best_gen_number][1], genbest[best_gen_number][2], genbest[best_gen_number][3], genbest[best_gen_number][4]);
    finish = clock(); //End time
    double duration = (double)(finish - start)/CLOCKS_PER_SEC; // program running time
    printf("Program running time:%lf seconds\n",duration);

  fclose(myFile);
  fclose(fp);
  if (line)
     free(line);
  exit(EXIT_SUCCESS);
 }

