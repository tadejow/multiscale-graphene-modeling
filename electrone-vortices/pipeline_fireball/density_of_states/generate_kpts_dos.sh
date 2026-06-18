#!/bin/bash
export LC_NUMERIC=C

echo "Generowanie gêstej siatki K-points (72x72) dla pe³nej strefy Brillouina..."
awk 'BEGIN {
    N=72; print N*N
    g1x=3.62759873; g1y=6.28318531; g2x=3.62759873; g2y=-6.28318531
    w=1.0/(N*N)
    for(i=0; i<N; i++) {
        for(j=0; j<N; j++) {
            printf "%.8f  %.8f  0.00000000  %.8f\n", (i/N)*g1x+(j/N)*g2x, (i/N)*g1y+(j/N)*g2y, w
        }
    }
}' > dos_full_bz.kpts
echo "Plik dos_full_bz.kpts utworzony pomyœlnie."