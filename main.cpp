#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

// Compute elapsed time in seconds.
float tdiff(struct timeval *start, struct timeval *end) {
    return (end->tv_sec - start->tv_sec) + 1e-6 * (end->tv_usec - start->tv_usec);
}

// Global seed for the random number generator.
unsigned long long seed = 100;

// Generate a 64-bit random number using bitwise operations.
unsigned inline long long randomU64() {
    seed ^= (seed << 21);
    seed ^= (seed >> 35);
    seed ^= (seed << 4);
    return seed;
}

// Produce a random double in the range [0, 1].
double inline randomDouble() {
    unsigned long long next = randomU64();
    next >>= (64 - 26);
    unsigned long long next2 = randomU64();
    next2 >>= (64 - 26);
    return ((next << 27) + next2) / (double)(1LL << 53);
}

// Structure representing planet data using structure of arrays (SoA)
typedef struct  {
    double *mass;
    double *x;
    double *y;
    double *vx;
    double *vy;
} PlanetStruct;

// Function to allocate memory for a PlanetStruct instance.
void initPlanetStruct(PlanetStruct *planet, int nplanets) {
    planet->mass = (double *)malloc(sizeof(double) * nplanets);
    planet->x    = (double *)malloc(sizeof(double) * nplanets);
    planet->y    = (double *)malloc(sizeof(double) * nplanets);
    planet->vx   = (double *)malloc(sizeof(double) * nplanets);
    planet->vy   = (double *)malloc(sizeof(double) * nplanets);
}

// Function to free memory allocated for a PlanetStruct instance.
void freePlanetStruct(PlanetStruct *planet) {
    free(planet->mass);
    free(planet->x);
    free(planet->y);
    free(planet->vx);
    free(planet->vy);
}

// Global simulation parameters.
int nplanets;
int timesteps;
double dt;
double G;  // Defined but not used in the force calculation.

// The step function updates the simulation state
// It reads from "current" and writes into "next"
void step(PlanetStruct *current, PlanetStruct *next) {
    // Copy the current state into the next buffer.
    #pragma omp parallel for
    for (int i = 0; i < nplanets; i++) {
        next->mass[i] = current->mass[i];
        next->x[i]    = current->x[i];
        next->y[i]    = current->y[i];
        next->vx[i]   = current->vx[i];
        next->vy[i]   = current->vy[i];
    }

    // Update velocities based on the gravitational-like interactions.
    #pragma omp parallel for
    for (int i = 0; i < nplanets; i++) {
        for (int j = 0; j < nplanets; j++) {
            double dx = current->x[j] - current->x[i];
            double dy = current->y[j] - current->y[i];
            double distSqr = dx * dx + dy * dy + 0.0001;  // Softening term to avoid singularity
            double invDist = current->mass[i] * current->mass[j] / sqrt(distSqr);
            double invDist3 = invDist * invDist * invDist;
            next->vx[i] += dt * dx * invDist3;
            next->vy[i] += dt * dy * invDist3;
        }
        // Update positions based on updated velocities.
        next->x[i] += dt * next->vx[i];
        next->y[i] += dt * next->vy[i];
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
    G = 6.6743; // Gravity constant (not used in force calculation here).

    // Allocate two PlanetStruct structures for double-buffering.
    PlanetStruct *planets = (PlanetStruct *)malloc(sizeof(PlanetStruct));
    PlanetStruct *buffer  = (PlanetStruct *)malloc(sizeof(PlanetStruct));
    initPlanetStruct(planets, nplanets);
    initPlanetStruct(buffer, nplanets);

    // Initialize the planet data.
    for (int i = 0; i < nplanets; i++) {
        planets->mass[i] = randomDouble() * 10 + 0.2;
        double scale = 100 * pow(1 + nplanets, 0.4);
        planets->x[i]    = (randomDouble() - 0.5) * scale;
        planets->y[i]    = (randomDouble() - 0.5) * scale;
        planets->vx[i]   = randomDouble() * 5 - 2.5;
        planets->vy[i]   = randomDouble() * 5 - 2.5;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Main simulation loop using double buffering.
    for (int t = 0; t < timesteps; t++) {
        step(planets, buffer);

        // Swap the pointers of current and buffer.
        PlanetStruct *temp = planets;
        planets = buffer;
        buffer = temp;
    }

    gettimeofday(&end, NULL);
    printf("Total time to run simulation %0.6f seconds, final location %f %f\n", tdiff(&start, &end), planets->x[nplanets-1], planets->y[nplanets-1]);

    // Free allocated memory.
    freePlanetStruct(planets);
    freePlanetStruct(buffer);
    free(planets);
    free(buffer);

    return 0;
}
