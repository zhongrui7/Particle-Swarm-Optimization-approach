/* Parameter extraction of solar cells using particle swarm optimization
 * Implementing Particle Swarm Optimization (PSO) using C language with OpenMP parallelization
 * 2025-11-11
 * J, current density (ampere/cm2)
 * JL, photogenerated current density (ampere/cm2)
 * I0, reverse saturation current density (ampere/cm2)
 * Rs, specific series resistance (Ω·cm2)
 * Rsh, specific shunt resistance (Ω·cm2)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define c1 1.495 // The acceleration factor is generally obtained from a large number of experiments
#define c2 1.495
#define maxgen 4096  // number of iterations
#define sizepop 1024 // population size
#define dim 5 // the dimension of the particle
#define popmin 0 // Individual minimum value
#define RNG_UNIFORM(seed) ((double)rand_r(&seed))/RAND_MAX-0.5  // random number in the range -0.5~+0.5

double pop[sizepop][dim]; // define population array
double V[sizepop][dim]; // Define the population velocity array
double fitness[sizepop]; // Define the fitness array of the population
double result[maxgen];  // Define an array to store the optimal value of the population for each iteration
double pbest[sizepop][dim];  // location of individual extremes
double gbest[dim]; // the location of the group extremum
double fitnesspbest[sizepop]; // The value of individual extreme fitness
double fitnessgbest; // group extreme fitness value
double genbest[maxgen][dim]; // Each generation of optimal value-valued particles

double k = 1.38e-23;
double q = 1.6e-19;
double T = 300; /* temperature in Kelvin */
double Vt = 0.025875; /* Vt=kT/q, for example, Vt(300K)=0.0259eV */
double A = 1.0; /* Solar cell area in cm^2 */

double Jph0=1.2e-3, Rs0=0.1, Rp0=2345; /* initial common parameters for both single/double-diode */
double Js0=1.2e-5, n0=1; /* initial single-diode parameters */
double Js10=1.2e-7, Js20=1.2e-7, n1=1, n2=2; /* initial double-diode parameters */
double V0[300], I0[300]; /* experimental I-V curve data */
double Jsc=0, Voc=0, FF=0, Pmax=0, Vm=0, Jm=0, eff=0;

int DSize = 0;
char * line = NULL;
char model, CurrentUnit;
size_t len = 0;
ssize_t read;

/* solar cell I-V function - single diode model */
double IL1(double V0, double I0, double Jph, double Js, double Rs, double Rp, double n)
{
    return (Jph - Js*(exp((V0+I0*Rs)/(n*Vt)) -1)-(V0+I0*Rs)/Rp);
}

/* solar cell I-V function - double diode model */
double IL2(double V0, double I0, double Jph, double Js1, double Js2, double Rs, double Rp)
{
    return (Jph - Js1*(exp((V0+I0*Rs)/(n1*Vt)) -1)-Js2*(exp((V0+I0*Rs)/(n2*Vt)) -1)-(V0+I0*Rs)/Rp);
}

/* fitness function squared difference between exp. and simul. */
double func1(double * arr)
{
    int i;
    double Jph=arr[0], Js=arr[1], Rs=arr[2], Rp=arr[3], n=arr[4];
    double erf=0.0, tot=0.0;

    for(i=0; i<DSize; ++i)
    {
        erf = IL1(V0[i], I0[i], Jph, Js, Rs, Rp, n) - I0[i];
        tot = tot + erf*erf;
    }

    return tot/DSize;
}

double func2(double * arr)
{
    int i;
    double Jph=arr[0], Js1=arr[1], Js2=arr[2], Rs=arr[3], Rp=arr[4];
    double erf=0.0, tot=0.0;

    for(i=0; i<DSize; ++i)
    {
        erf = IL2(V0[i], I0[i], Jph, Js1, Js2, Rs, Rp) - I0[i];
        tot = tot + erf*erf;
    }

    return tot/DSize;
}

/* Population initialization */
void pop_init(void)
{
    #pragma omp parallel for
    for(int i=0; i<sizepop; i++)
    {
        unsigned int seed = time(NULL) ^ (omp_get_thread_num() + i); // Unique seed per thread
        switch(model)
        {
        case '1':
            pop[i][0]= Jph0*(1+RNG_UNIFORM(seed)); // Jph
            pop[i][1]= Js0*(1+RNG_UNIFORM(seed)); // Js
            pop[i][2]= Rs0*(1+RNG_UNIFORM(seed)); // Rs
            pop[i][3]= Rp0*(1+RNG_UNIFORM(seed)); // Rp
            pop[i][4]= n0*(1+RNG_UNIFORM(seed)); // n

            for(int j=0; j<dim; j++)
            {
                V[i][j]=pop[i][j]/10;
            }
            fitness[i] = func1(pop[i]);
            break;

        case '2':
            pop[i][0]= Jph0*(1+RNG_UNIFORM(seed)); // Jph
            pop[i][1]= Js10*(1+RNG_UNIFORM(seed)); // Js1
            pop[i][2]= Js20*(1+RNG_UNIFORM(seed)); // Js2
            pop[i][3]= Rs0*(1+RNG_UNIFORM(seed)); // Rs
            pop[i][4]= Rp0*(1+RNG_UNIFORM(seed)); // Rp

            for(int j=0; j<dim; j++)
            {
                V[i][j]=pop[i][j]/10;
            }
            fitness[i] = func2(pop[i]);
            break;
        }
    }
}

/* min() program function definition */
double * min(double * fit, int size)
{
    int index = 0;
    double min = *fit;
    static double best_fit_index[2];
    for(int i=1; i<size; i++)
    {
        if(*(fit+i) < min)
        {
            min = *(fit+i);
            index = i;
        }
    }

    best_fit_index[0] = index;
    best_fit_index[1] = min;
    return best_fit_index;
}

/* iterative optimization */
void PSO_func(void)
{
    pop_init();
    double * best_fit_index;
    best_fit_index = min(fitness, sizepop);
    int index = (int)(*best_fit_index);

    for(int j=0; j<dim; j++)
    {
        gbest[j] = pop[index][j];
    }

    #pragma omp parallel for
    for(int i=0; i<sizepop; i++)
    {
        for(int j=0; j<dim; j++)
        {
            pbest[i][j] = pop[i][j];
        }
        fitnesspbest[i] = fitness[i];
    }

    double bestfitness = *(best_fit_index+1);
    fitnessgbest = bestfitness;

    for(int i=0; i<maxgen; i++)
    {
        #pragma omp parallel for
        for(int j=0; j<sizepop; j++)
        {
            unsigned int seed = time(NULL) ^ (omp_get_thread_num() + j + i); // Unique seed
            for(int k=0; k<dim; k++)
            {
                double rand1 = (double)rand_r(&seed)/RAND_MAX;
                double rand2 = (double)rand_r(&seed)/RAND_MAX;
                V[j][k] = 0.9*V[j][k] + c1*rand1*(pbest[j][k]-pop[j][k]) + c2*rand2*(gbest[k]-pop[j][k]);
                pop[j][k] = pop[j][k] + V[j][k];
                if(pop[j][k] < popmin)
                    pop[j][k] = popmin;
            }
            pop[j][0]= (Jsc==0)? 0 : pop[j][0];
            if(model=='1')
            {
                do{pop[j][4]= n0*(1+RNG_UNIFORM(seed)/50);}
                while(pop[j][4]<0||pop[j][4]>3);
            }
            fitness[j] = (model=='1')? func1(pop[j]): func2(pop[j]);
        }

        double local_gbest[dim];
        double local_fitnessgbest = fitnessgbest;
        int update_needed = 0;

        #pragma omp parallel for
        for(int j=0; j<sizepop; j++)
        {
            if(fitness[j] < fitnesspbest[j])
            {
                for(int k=0; k<dim; k++)
                {
                    pbest[j][k] = pop[j][k];
                }
                fitnesspbest[j] = fitness[j];
            }

            if(fitness[j] < local_fitnessgbest)
            {
                #pragma omp critical
                {
                    if(fitness[j] < fitnessgbest)
                    {
                        fitnessgbest = fitness[j];
                        for(int k=0; k<dim; k++)
                        {
                            gbest[k] = pop[j][k];
                        }
                    }
                }
            }
        }

        for(int k=0; k<dim; k++)
        {
            genbest[i][k] = gbest[k];
        }
        result[i] = fitnessgbest;
        if(i%64==0) printf("*");
    }
}

#include <errno.h>
#include <stdint.h>
typedef intptr_t ssize_t;

ssize_t getline(char **lineptr, size_t *n, FILE *stream) {
    size_t pos;
    int c;

    if (lineptr == NULL || stream == NULL || n == NULL) {
        errno = EINVAL;
        return -1;
    }

    c = getc(stream);
    if (c == EOF) { return -1; }

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

void swap(double* xp, double* yp)
{
    double temp = *xp;
    *xp = *yp;
    *yp = temp;
}

void SortV(double arrV[], double arrI[], int n)
{
    int i, j, min_idx;
    for (i = 0; i < n - 1; i++) {
        min_idx = i;
        for (j = i + 1; j < n; j++)
            if (arrV[j] < arrV[min_idx])
                min_idx = j;
        swap(&arrV[min_idx], &arrV[i]);
        swap(&arrI[min_idx], &arrI[i]);
    }
}

void PV_perf()
{
    int i=0, s=0, p=0, m=0;
    double Vtmp=fabs(V0[0]), e1, e2;
    double Itmp=fabs(I0[0]);

    SortV(V0, I0, DSize);

    if ((V0[1]*I0[1]) > 0 && I0[0]+I0[1]<0)
        for(i=0; i < DSize; ++i)
        {
            I0[i]=-I0[i];
        }

    for(i=0; i < DSize; ++i)
    {
        if(fabs(V0[i]) < Vtmp)
        {
            Vtmp = fabs(V0[i]);
            p=i;
        }
        if(fabs(I0[i]) < Itmp)
        {
            Itmp=fabs(I0[i]);
            s=i;
        }
        if ((V0[i]*I0[i]) > Pmax && V0[i]>0 && I0[i]>0)
        {
            Pmax=V0[i]*I0[i];
            m=i;
        }
    }
    Vm=V0[m];
    Jm=I0[m];

    Jsc= (V0[p]==0)? fabs(I0[p]) : I0[p-1]-V0[p-1]*(I0[p+1]-I0[p-1])/(V0[p+1]-V0[p-1]);
    Voc= (I0[s]==0)? fabs(V0[s]) : V0[s-1]-I0[s-1]*(V0[s+1]-V0[s-1])/(I0[s+1]-I0[s-1]);

    printf("\t Voc=V0[%d]=%f, Jsc=I0[%d]=%f \n", s, Voc, p, Jsc);
    if(Jsc!=0 && Voc!=0)
    {
        FF = Pmax/(Voc*Jsc);
        eff = Pmax*1000;
        printf("\t Voc=%7.3f[V], Jsc=%7.4f[mA/cm^2], FF=%5.2f, effi=%7.3f%% \n", Voc, Jsc*1000, FF, eff);
        printf("\t Pmax=%7.3f[mW/cm^2], Vm=%7.3f[V], Jm=%7.3f[mA/cm^2] \n", Pmax*1000, Vm, 1000*Jm);

        Rs0 = (fabs((V0[s+1]-V0[s-2])/(I0[s+1]-I0[s-2]))+fabs((V0[s+2]-V0[s-1])/(I0[s+2]-I0[s-1])))/2;
        Rp0 = (fabs((V0[p+1]-V0[p-2])/(I0[p+1]-I0[p-2]))+fabs((V0[p+2]-V0[p-1])/(I0[p+2]-I0[p-1])))/2;
        Jph0= Jsc;
        switch(model)
        {
        case '1':
            n0 = 1.0;
            Js0 = (Jsc*(Rp0-Rs0) - Voc)/(Rp0*(exp(Voc/(n0*Vt))-exp(Rs0*Jsc/(n0*Vt))));
            printf("Initial guess:\n\t Jph0=%.3e[mA/cm^2], Js0=%.3e[mA/cm^2],\n\t Rs0=%7.3f[Ohm.cm^2], Rp0=%9.3f[Ohm.cm^2], n0=%3.2f\n", Jph0*1000, Js0*1000, Rs0, Rp0, n0);
            break;

        case '2':
            Js10 = (Jsc*(Rp0-Rs0) - Voc)/(Rp0*(exp(Voc/(n1*Vt))-exp(Rs0*Jsc/(n1*Vt))));
            Js20 = Jsc/(exp(Voc*q/(k*T))-1);
            printf("Initial guess:\n\t Jph0=%.3e[mA/cm^2], Js1=%.3e[mA/cm^2], Js2=%.3e[mA/cm^2],\n\t Rs0=%7.3f[Ohm.cm^2, Rp0=%9.3f[Ohm.cm^2]\n", Jph0*1000, Js10*1000, Js20*1000, Rs0, Rp0);
            break;
        }
    }
    else
    {
        printf("\n\t $$$ The input IV curve is a DARK current! $$$\n");
        Jph0 = 0;
        Js0 = -((I0[(int)((DSize + p)/2)])/exp(V0[(int)((DSize + p)/2)]/Vt -1)+I0[DSize]/exp(V0[DSize]/Vt -1))/2;
        n0 = 1;
        Rs0 = (fabs((V0[s+1]-V0[s-2])/(I0[s+1]-I0[s-2]))+fabs((V0[s+2]-V0[s-1])/(I0[s+2]-I0[s-1])))/2;
        Rp0 = (fabs((V0[p+1]-V0[p-2])/(I0[p+1]-I0[p-2]))+fabs((V0[p+2]-V0[p-1])/(I0[p+2]-I0[p-1])))/2;
        Js10 = Js0;
        Js20 = Js0;
    }
}

int main(int argc, char **argv)
{
    int i=0, j=0, k=0;
    char chr;
    double * best_arr;
    int bgn; // best_gen_number
    double best;

    printf("Parameter extraction of solar cells based on a single/double diode model using Particle Swarm Optimization\n");
    printf("the input IV curve data must be a 2-column ASCII file (V I) \n");

    if(argc != 2)
    {
        printf("Usage: pso_PVIV_FIT IV-file\n");
        return 0;
    }

    printf("Current Density unit in the IV file (select 1 or 2): \n");
    printf("\t 1: A/cm^2 \n ");
    printf("\t 2: mA/cm^2 \n ");
    do{scanf("%s",&CurrentUnit);}while(CurrentUnit!='1'&&CurrentUnit!='2');

    printf("Input Solar Cell Size and Temperature: [default: 1.0cm^2, 300K] \n");
    scanf("%f, %f", &A, &T);
    Vt = 0.025875*T/300;

    FILE *myFile;
    myFile = fopen(argv[1], "r");
    char *out=malloc(strlen(argv[1]) + 9);
    FILE *fp;

    if(myFile == NULL)
        exit(EXIT_FAILURE);
    while((read = getline(&line, &len, myFile)) != -1) {
        DSize++;
    }

    rewind(myFile);

    for(i = 0; i < DSize; ++i)
    {
        fscanf(myFile, "%lf%lf", &V0[i], &I0[i]);
        I0[i]=I0[i]/A;
        if(CurrentUnit=='2') I0[i]=I0[i]/1000;
    }

    printf("Which model should be used for the IV curve fitting (enter 1 or 2)?:\n ");
    printf("\t 1: Single-diode Model\n ");
    printf("\t 2: Double-diode Model\n ");
    scanf(" %c",&model);
    switch(model)
    {
    case '1':
        printf("Single-diode model is selected:\n\t IL1=Jph - Js*(exp((V0+I0*Rs)/(n*Vt)) -1) - (V0+I0*Rs)/Rp \n");
        break;
    case '2':
        printf("Double-diode model is selected:\n\t IL2=Jph - Js1*(exp((V0+I0*Rs)/(n1*Vt)) -1) - Js2*(exp((V0+I0*Rs)/(n2*Vt)) -1) - (V0+I0*Rs)/Rp \n");
        printf("diode ideality factors n1, and n2, \n\t are fixed to 1 and 2 to represent the diffusion and recombination current terms, respectively!\n");
        break;
    default:
        printf("Error! this model is not available, please enter either 1 or 2, Bye! \n");
        return 0;
    }

    PV_perf();

    clock_t start, finish;
    start = clock();
    srand((unsigned)time(NULL));

    do{
        printf("PSO is working on Model %c, please wait : \n", model);
        PSO_func();
        k++;
        best_arr = min(result, maxgen);
        bgn = *best_arr;  // best_gen_number
        best = *(best_arr+1);
        printf("\n After iterating %d times, the fitness value is: %.5e.\n", bgn, best);

        switch(model)
        {
        case '1':
            Jph0 = genbest[bgn][0];
            Js0 = genbest[bgn][1];
            Rs0 = genbest[bgn][2];
            Rp0 = genbest[bgn][3];
            n0 = genbest[bgn][4];
            printf("\n Single-diode model PSO fitting results:\n");
            printf("\t Jph=%.3e[mA/cm^2], Js=%e[mA/cm^2],\n\t Rs=%.3e[Ohm/cm^2], Rp=%.3e[Ohm/cm^2], n=%5.3f\n",
                   1000*genbest[bgn][0], 1000*genbest[bgn][1], genbest[bgn][2], genbest[bgn][3], genbest[bgn][4]);
            break;

        case '2':
            Jph0 = genbest[bgn][0];
            Js10 = genbest[bgn][1];
            Js20 = genbest[bgn][2];
            Rs0 = genbest[bgn][3];
            Rp0 = genbest[bgn][4];
            printf("\n Double-diode model PSO fitting results:\n");
            printf("\t Jph=%.3e[mA/cm^2], Js1=%.3e[mA/cm^2], Js2=%.e[mA/cm^2],\n\t Rs=%7.3f[Ohm/cm^2], Rp=%9.3f[Ohm/cm^2] \n",
                   1000*genbest[bgn][0], 1000*genbest[bgn][1], genbest[bgn][2], genbest[bgn][3], genbest[bgn][4]);
            break;
        }

        printf(" Continue the PSO fitting? [y/n]");
        scanf(" %c",&chr);
    }while(chr=='Y'||chr=='y');

    finish = clock();
    double duration = (double)(finish - start)/CLOCKS_PER_SEC;
    printf("\n #- Program running time: %lf seconds -#", duration);

    memset(out, '\0', sizeof(out));
    strncat(out, argv[1], 6);
    strcat(out, model=='1'? "_fit1.dat" : "_fit2.dat");
    strcat(out, "\0");
    fp = fopen(out, "w");
    if (fp == NULL)
    {
        printf("Error opening OUTPUT file %s!\n", out);
        exit(1);
    }
    (CurrentUnit=='2')?fprintf(fp, "%s %s %s\n", "Bias[V]", "Jexp[mA/cm^2]", "Jfit[mA/cm^2]"):fprintf(fp, "%s %s %s\n", "Bias[V]", "Jexp[A/cm2]", "Jfit[A/cm2]");

    switch(model)
    {
    case '1':
        for (i = 0; i<DSize; i=i+1)
        {
            if(CurrentUnit=='2')
                fprintf(fp, "%lf %lf %lf\n", V0[i], 1000*I0[i], 1000*IL1(V0[i], I0[i], genbest[bgn][0], genbest[bgn][1], genbest[bgn][2], genbest[bgn][3], genbest[bgn][4]));
            else
                fprintf(fp, "%lf %lf %lf\n", V0[i], I0[i], IL1(V0[i], I0[i], genbest[bgn][0], genbest[bgn][1], genbest[bgn][2], genbest[bgn][3], genbest[bgn][4]));
        }
        break;

    case '2':
        for (i = 0; i<DSize; i=i+1)
        {
            if(CurrentUnit=='2')
                fprintf(fp, "%lf %lf %lf\n", V0[i], 1000*I0[i], 1000*IL2(V0[i], I0[i], genbest[bgn][0], genbest[bgn][1], genbest[bgn][2], genbest[bgn][3], genbest[bgn][4]));
            else
                fprintf(fp, "%lf %lf %lf\n", V0[i], I0[i], IL2(V0[i], I0[i], genbest[bgn][0], genbest[bgn][1], genbest[bgn][2], genbest[bgn][3], genbest[bgn][4]));
        }
        break;
    }

    printf("\n\t Simulated curve is saved in \' %s \'!\n", out);

    fclose(myFile);
    fclose(fp);

    if (line)
        free(line);
    exit(EXIT_SUCCESS);
}
