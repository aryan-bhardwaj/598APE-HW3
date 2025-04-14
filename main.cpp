#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <stdalign.h>
#include <immintrin.h>

// Compute elapsed time in seconds.
float tdiff(struct timeval *start, struct timeval *end)
{
   return (end->tv_sec - start->tv_sec) + 1e-6 * (end->tv_usec - start->tv_usec);
}

// Global seed for the random number generator.
unsigned long long seed = 100;

// Generate a 64-bit random number using bitwise operations.
unsigned inline long long randomU64()
{
   seed ^= (seed << 21);
   seed ^= (seed >> 35);
   seed ^= (seed << 4);
   return seed;
}

// Produce a random double in the range [0, 1].
double inline randomDouble()
{
   unsigned long long next = randomU64();
   next >>= (64 - 26);
   unsigned long long next2 = randomU64();
   next2 >>= (64 - 26);
   return ((next << 27) + next2) / (double)(1LL << 53);
}

typedef struct alignas(64)
{
   double *mass;
   double *x;
   double *y;
   double *vx;
   double *vy;
} PlanetStruct;

// Global simulation parameters.
int nplanets;
int timesteps;
double dt;

// Global pointers for the double-buffered structures.
PlanetStruct *planets;
PlanetStruct *buffer;

// Function to allocate memory for a PlanetStruct instance.
void initPlanetStruct(PlanetStruct *planet, int nplanets)
{
   planet->mass = (double *)malloc(sizeof(double) * nplanets);
   planet->x = (double *)malloc(sizeof(double) * nplanets);
   planet->y = (double *)malloc(sizeof(double) * nplanets);
   planet->vx = (double *)malloc(sizeof(double) * nplanets);
   planet->vy = (double *)malloc(sizeof(double) * nplanets);
}

// Function to free memory allocated for a PlanetStruct instance.
void freePlanetStruct(PlanetStruct *planet)
{
   free(planet->mass);
   free(planet->x);
   free(planet->y);
   free(planet->vx);
   free(planet->vy);
}

void swapBuffers()
{
   double *temp;
   temp = planets->mass;
   planets->mass = buffer->mass;
   buffer->mass = temp;
   temp = planets->x;
   planets->x = buffer->x;
   buffer->x = temp;
   temp = planets->y;
   planets->y = buffer->y;
   buffer->y = temp;
   temp = planets->vx;
   planets->vx = buffer->vx;
   buffer->vx = temp;
   temp = planets->vy;
   planets->vy = buffer->vy;
   buffer->vy = temp;
}

// The simulate() function performs the simulation over the given number of timesteps.
void simulate()
{
   for (int t = 0; t < timesteps; t++)
   {
      // Copy the current state from planets to buffer.
      #pragma omp parallel for if (nplanets > 50)
      for (int i = 0; i < nplanets; i++)
      {
         buffer->mass[i] = planets->mass[i];
         buffer->x[i] = planets->x[i];
         buffer->y[i] = planets->y[i];
         buffer->vx[i] = planets->vx[i];
         buffer->vy[i] = planets->vy[i];
      }

      #pragma omp parallel for if (nplanets > 50)
      for (int i = 0; i < nplanets; i++)
      {
         __m256d vx_accum = _mm256_setzero_pd();
         __m256d vy_accum = _mm256_setzero_pd();
         __m256d xi = _mm256_set1_pd(planets->x[i]);
         __m256d yi = _mm256_set1_pd(planets->y[i]);
         __m256d mi = _mm256_set1_pd(planets->mass[i]);
         __m256d dt_vec = _mm256_set1_pd(dt);
         __m256d softening = _mm256_set1_pd(0.0001);

         int j;
         for (j = 0; j <= nplanets - 4; j += 4)
         {
            __m256d xj = _mm256_loadu_pd(&planets->x[j]);
            __m256d yj = _mm256_loadu_pd(&planets->y[j]);
            __m256d mj = _mm256_loadu_pd(&planets->mass[j]);

            __m256d dx = _mm256_sub_pd(xj, xi);
            __m256d dy = _mm256_sub_pd(yj, yi);
            __m256d dx2 = _mm256_mul_pd(dx, dx);
            __m256d dy2 = _mm256_mul_pd(dy, dy);
            __m256d distSqr = _mm256_add_pd(_mm256_add_pd(dx2, dy2), softening);
            __m256d sqrtDist = _mm256_sqrt_pd(distSqr);
            __m256d mprod = _mm256_mul_pd(mi, mj);
            __m256d invDist = _mm256_div_pd(mprod, sqrtDist);
            __m256d invDist2 = _mm256_mul_pd(invDist, invDist);
            __m256d invDist3 = _mm256_mul_pd(invDist2, invDist);
            __m256d factor = _mm256_mul_pd(dt_vec, invDist3);
            vx_accum = _mm256_add_pd(vx_accum, _mm256_mul_pd(dx, factor));
            vy_accum = _mm256_add_pd(vy_accum, _mm256_mul_pd(dy, factor));
         }

         double vx_sum = 0.0, vy_sum = 0.0;
         double temp[4];
         _mm256_storeu_pd(temp, vx_accum);
         vx_sum = temp[0] + temp[1] + temp[2] + temp[3];
         _mm256_storeu_pd(temp, vy_accum);
         vy_sum = temp[0] + temp[1] + temp[2] + temp[3];

         buffer->vx[i] += vx_sum;
         buffer->vy[i] += vy_sum;
         buffer->x[i] += dt * buffer->vx[i];
         buffer->y[i] += dt * buffer->vy[i];
      }

      // Swap the data in the global structures by swapping all array pointers.
      swapBuffers();
   }
}

// Main function.
int main(int argc, const char **argv)
{
   if (argc < 3)
   {
      printf("Usage: %s <nplanets> <timesteps>\n", argv[0]);
      return 1;
   }
   nplanets = atoi(argv[1]);
   if (nplanets % 4 != 0)
   {
      printf("nplanets must be a multiple of 4\n");
      return 1;
   }

   timesteps = atoi(argv[2]);
   dt = 0.001;

   // Allocate the global double-buffered structures.
   planets = (PlanetStruct *)malloc(sizeof(PlanetStruct));
   buffer = (PlanetStruct *)malloc(sizeof(PlanetStruct));
   initPlanetStruct(planets, nplanets);
   initPlanetStruct(buffer, nplanets);

   // Initialize the planet data.
   for (int i = 0; i < nplanets; i++)
   {
      planets->mass[i] = randomDouble() * 10 + 0.2;
      double scale = 100 * pow(1 + nplanets, 0.4);
      planets->x[i] = (randomDouble() - 0.5) * scale;
      planets->y[i] = (randomDouble() - 0.5) * scale;
      planets->vx[i] = randomDouble() * 5 - 2.5;
      planets->vy[i] = randomDouble() * 5 - 2.5;
   }

   struct timeval start, end;
   gettimeofday(&start, NULL);

   // Run the simulation using the global variables.
   simulate();

   gettimeofday(&end, NULL);
   printf("Total time to run simulation %0.6f seconds, final location %f %f\n",
          tdiff(&start, &end), planets->x[nplanets - 1], planets->y[nplanets - 1]);

   // Free allocated memory.
   freePlanetStruct(planets);
   freePlanetStruct(buffer);
   free(planets);
   free(buffer);

   return 0;
}
