/* MPI program that uses a monte carlo method to compute the value of PI */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <mpi.h>
#include <time.h>
#define USE_MPI   
#define PI 3.14159265358979323
#define DEBUG 0

/* Define the max number of accurate values until termination*/
#define MAX 10

double get_eps(void);

int main(int argc, char *argv[])
{
    double x, y;
    int i;
    int count= 0, mycount; /* # of points in the 1st quadrant of unit circle */
    double z;
    double pi = 0.0;
    int myid, numprocs, proc;
    MPI_Status status;
    int master = 0;
    int tag = 123;
    long int myiter = 1;
    int done = 0;
    long int iterval = 1000000; /* how many points per iteration to increase */
    long int niter = iterval;
   
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);

    srand48(time(NULL)*myid);
    double epsilon;
    // if(myid==0)epsilon=get_eps();

   
    if (argc <= 1) {
       fprintf(stderr,"Usage: monte_pi_mpi epsilon(0.0001)\n");
       MPI_Finalize();
       exit(-1);
    }

    epsilon = atof(argv[1]);

    while(done != MAX){

        mycount = 0;
        myiter= niter/numprocs;

        /* initialize random numbers */
        for (i = 0; i < myiter; i++) {
            x = (double)drand48();
            y = (double)drand48();
            z = x*x + y*y;
           if (z <= 1) mycount++; 
        }
   
        if (myid == 0) { /* if I am the master process gather results from others */
            
            count = mycount;
            
            for (proc = 1; proc<numprocs; proc++) {
                MPI_Recv(&mycount,1,MPI_INT,proc,tag,MPI_COMM_WORLD,&status);
                count += mycount;        
            }

            pi = (double)count/(myiter*numprocs)*4; /* 4 quadrants of circle */
      
            if (DEBUG)
                printf("procs= %d, trials= %ld, estimate= %2.10f, PI= %2.10f, error= %2.10f \n", numprocs,myiter*numprocs,pi, PI,fabs(pi-PI));
    
            if (fabs(pi - PI) <= (epsilon * fabs(pi))) {
                printf("\n# (%d) Accuracy Met: iters = %ld, PI= %2.10f, pi= %2.10f, error= %2.10f, eps= %2.10f\n", done,numprocs*myiter,PI,pi,fabs(pi-PI),epsilon); 
                done++;
            }
            /* Tell everyone we are done*/
            MPI_Bcast(&done,1,MPI_INT,myid,MPI_COMM_WORLD);
        } else { /* for all the slave processes send results to the master */
        
            /* printf("Processor %d sending results= %d to master process\n",myid,mycount); */
            MPI_Send(&mycount,1,MPI_INT,master,tag,MPI_COMM_WORLD);

            /* Have we received a done command */
            MPI_Bcast(&done,1,MPI_INT,0,MPI_COMM_WORLD);
        }
    }   

    MPI_Finalize(); 
  
    return 0;
}

/* get machine epsilon */
double get_eps(){
    double xxeps=1.0;
    int i;
    for(i=1; i<55; i++) {
        /* we know about 52 bits */
        xxeps=xxeps/2.0;
        if(1.0-xxeps == 1.0) break;
    }
    xxeps=2.0*xxeps;
    printf("type double eps=%24.17E \n", xxeps);
    return xxeps;
}


