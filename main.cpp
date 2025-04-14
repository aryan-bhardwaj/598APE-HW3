#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

// Compute elapsed time in seconds.
float tdiff(struct timeval *start, struct timeval *end) {
   return (end->tv_sec - start->tv_sec) + 1e-6 * (end->tv_usec - start->tv_usec);
}

// Structure representing a planet.
struct Planet {
   double mass;
   double x;
   double y;
   double vx;
   double vy;
};

// Global seed for the random number generator.
unsigned long long seed = 100;

// Generate a 64-bit random number using bitwise operations.
unsigned long long randomU64() {
   seed ^= (seed << 21);
   seed ^= (seed >> 35);
   seed ^= (seed << 4);
   return seed;
}

// Produce a random double in the range [0, 1].
double randomDouble() {
   unsigned long long next = randomU64();
   next >>= (64 - 26);
   unsigned long long next2 = randomU64();
   next2 >>= (64 - 26);
   return ((next << 27) + next2) / (double)(1LL << 53);
}

// Global simulation parameters.
int nplanets;
int timesteps;
double dt;
double G;

// Function to perform one simulation time step.
// The new state is computed from 'current' and stored in 'next'.
void step(struct Planet* current, struct Planet* next) {
   // Copy the current state into the next buffer.
   #pragma omp parallel for
   for (int i = 0; i < nplanets; i++) {
      next[i].mass = current[i].mass;
      next[i].x = current[i].x;
      next[i].y = current[i].y;
      next[i].vx = current[i].vx;
      next[i].vy = current[i].vy;
   }

   // Update velocities based on the gravitational-like interactions.
   #pragma omp parallel for
   for (int i = 0; i < nplanets; i++) {
      for (int j = 0; j < nplanets; j++) {
         double dx = current[j].x - current[i].x;
         double dy = current[j].y - current[i].y;
         double distSqr = dx * dx + dy * dy + 0.0001;  // Softening term
         double invDist = current[i].mass * current[j].mass / sqrt(distSqr);
         double invDist3 = invDist * invDist * invDist;
         next[i].vx += dt * dx * invDist3;
         next[i].vy += dt * dy * invDist3;
      }
      // Update positions based on the updated velocities.
      next[i].x += dt * next[i].vx;
      next[i].y += dt * next[i].vy;
   }
}

int main(int argc, const char** argv) {
   if (argc < 3) {
      printf("Usage: %s <nplanets> <timesteps>\n", argv[0]);
      return 1;
   }
   nplanets = atoi(argv[1]);
   timesteps = atoi(argv[2]);
   dt = 0.001;

   // Allocate two buffers for double buffering.
   struct Planet* planets = (struct Planet*)malloc(sizeof(struct Planet) * nplanets);
   struct Planet* buffer  = (struct Planet*)malloc(sizeof(struct Planet) * nplanets);

   // Initialize planet values.
   for (int i = 0; i < nplanets; i++) {
      planets[i].mass = randomDouble() * 10 + 0.2;
      planets[i].x = (randomDouble() - 0.5) * 100 * pow(1 + nplanets, 0.4);
      planets[i].y = (randomDouble() - 0.5) * 100 * pow(1 + nplanets, 0.4);
      planets[i].vx = randomDouble() * 5 - 2.5;
      planets[i].vy = randomDouble() * 5 - 2.5;
   }

   struct timeval start, end;
   gettimeofday(&start, NULL);

   // Main simulation loop using double buffering.
   // 'planets' holds the current state, while 'buffer' will be used to store the new state.
   for (int t = 0; t < timesteps; t++) {
      step(planets, buffer);

      // Swap the pointers.
      struct Planet* temp = planets;
      planets = buffer;
      buffer = temp;
   }

   gettimeofday(&end, NULL);
   printf("Total time to run simulation: %0.6f seconds, final location of planet %d: %f %f\n",
         tdiff(&start, &end), nplanets - 1, planets[nplanets - 1].x, planets[nplanets - 1].y);

   // Free the allocated memory.
   free(planets);
   free(buffer);
   return 0;
}
