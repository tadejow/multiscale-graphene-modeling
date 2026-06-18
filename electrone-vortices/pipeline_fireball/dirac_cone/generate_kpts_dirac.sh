#!/bin/bash
export LC_NUMERIC=C

echo "Generowanie siatki 2D (41x41) wokó³ punktu K..."
awk 'BEGIN {
    N = 41; print N*N
    Kx = 3.62759873; Ky = 2.09439510; delta = 0.8
    w = 1.0 / (N*N)
    for(i=0; i<N; i++) {
        for(j=0; j<N; j++) {
            kx = Kx - delta + (2.0 * delta * i) / (N - 1)
            ky = Ky - delta + (2.0 * delta * j) / (N - 1)
            printf "%.8f  %.8f  0.00000000  %.8f\n", kx, ky, w
        }
    }
}' > dirac_3d.kpts
echo "Plik dirac_3d.kpts utworzony pomyœlnie."

# Generowanie pomocniczej siatki SCF
awk 'BEGIN {
    N=12; print N*N
    g1x=3.62759873; g1y=6.28318531; g2x=3.62759873; g2y=-6.28318531
    w=1.0/(N*N)
    for(i=0; i<N; i++) { for(j=0; j<N; j++) { printf "%.8f %.8f 0.00000000 %.8f\n", (i/N)*g1x+(j/N)*g2x, (i/N)*g1y+(j/N)*g2y, w } }
}' > scf.kpts