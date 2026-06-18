#!/bin/bash
export LC_NUMERIC=C

echo "Generowanie œcie¿ki Gamma -> M -> K -> Gamma (91 punktów)..."
awk 'BEGIN {
    print "91"
    w = 0.010989
    for(i=0; i<30; i++) {
        kx = (3.62759873 / 30) * i; ky = 0.0
        printf "%.8f  %.8f  0.00000000  %.6f\n", kx, ky, w
    }
    for(i=0; i<30; i++) {
        kx = 3.62759873; ky = (2.09439510 / 30) * i
        printf "%.8f  %.8f  0.00000000  %.6f\n", kx, ky, w
    }
    for(i=0; i<=30; i++) {
        kx = 3.62759873 - (3.62759873 / 30) * i
        ky = 2.09439510 - (2.09439510 / 30) * i
        printf "%.8f  %.8f  0.00000000  %.6f\n", kx, ky, w
    }
}' > band.kpts
echo "Plik band.kpts utworzony pomyœlnie."

# Generujemy te¿ pomocnicz¹ siatkê do kroku SCF
echo "Generowanie pomocniczej siatki SCF (12x12)..."
awk 'BEGIN {
    N=12; print N*N
    g1x=3.62759873; g1y=6.28318531; g2x=3.62759873; g2y=-6.28318531
    w=1.0/(N*N)
    for(i=0; i<N; i++) {
        for(j=0; j<N; j++) {
            printf "%.8f  %.8f  0.00000000  %.8f\n", (i/N)*g1x+(j/N)*g2x, (i/N)*g1y+(j/N)*g2y, w
        }
    }
}' > scf.kpts