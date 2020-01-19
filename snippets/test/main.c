#include <math.h>
#include <stdio.h>

void add_arrays(float*, float*, float*, int);

#define SIZE 1000

int main() {
    float a[SIZE], b[SIZE], c[SIZE];

    for (int i = 0; i < SIZE; i++) {
        a[i] = sinf(i)*sinf(i);
        b[i] = cosf(i)*cosf(i);
    }

    add_arrays(a, b, c, SIZE);

    float sum = 0;
    for (int i = 0; i < SIZE; i++) {
        sum += c[i];
    }

    printf("final sum: %f\n", sum);
}
