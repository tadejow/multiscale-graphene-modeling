#!/bin/bash

# ==============================================================================
# POBIERAMY PE£N¥ ŒCIE¯KÊ DO OBECNEGO KATALOGU
# ==============================================================================
BASE_DIR=$(pwd)
FIREBALL_EXEC="$BASE_DIR/fireball.x"
FDATA_PATH="/home/em0/Fdata_H-ss_C-spd_Si-spd"

chmod +x $FIREBALL_EXEC

alat=$(LC_NUMERIC=C seq -s ' ' 2.0 0.001 3.0)
RESULTS_DIR="../RESULTS"
mkdir -p $RESULTS_DIR

# Dodano kolumnê F_max do pliku CSV
echo "Lattice,E_tot,E_fermi,F_max" > $RESULTS_DIR/summary.csv

for i in $alat; do
    echo "======================================"
    echo "Rozpoczynam obliczenia dla sieci: $i"

    WORK_DIR="$RESULTS_DIR/$i"
    mkdir -p $WORK_DIR

    cp C1.bas C1.lvs C1.32.kpts dos.optional $WORK_DIR/ 2>/dev/null
    sed "s/AAA/$i/g" fireball.sample > $WORK_DIR/fireball.in
    echo "$FDATA_PATH" > $WORK_DIR/Fdata.optional

    cd $WORK_DIR

    $FIREBALL_EXEC > out_$i.log

    # ================= WYCI¥GANIE DANYCH =================
    ETOT=$(grep "etot/atom" out_$i.log | cut -b50-65 | xargs)
    EFERMI=$(grep "Fermi" out_$i.log | tail -n 1 | awk '{print $NF}')
    
    # Wyci¹ganie wartoœci Fmax (Maksymalna si³a)
    FMAX=$(grep -i "Fmax" out_$i.log | tail -n 1 | awk '{print $NF}')
    if [ -z "$FMAX" ]; then FMAX="N/A"; fi

    # Zapis do CSV
    echo "$i,$ETOT,$EFERMI,$FMAX" >> ../summary.csv

    if [ -f "eigen.dat" ]; then cp eigen.dat ../eigen_$i.dat; fi
    if [ -f "dens_TOT.dat" ]; then cp dens_TOT.dat ../dens_$i.dat; fi

    echo "Zakoñczono: E_tot = $ETOT | E_F = $EFERMI | F_max = $FMAX"

    cd $BASE_DIR
done

echo "Wszystkie obliczenia zakoñczone! Pakujê wyniki..."
cd ..
tar czvf RESULTS.tgz RESULTS/
echo "Gotowe!"