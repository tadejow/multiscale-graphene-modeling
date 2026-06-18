#!/bin/bash
FIREBALL_EXEC="../fireball.x"
FDATA_PATH="/home/em0/Fdata_H-ss_C-spd_Si-spd"

echo "$FDATA_PATH" > Fdata.optional

echo "=== KROK 1: Obliczenia SCF (Relaksacja gêstoœci ³adunku) ==="
cat <<EOF > fireball.in
&OPTION
basisfile = 'C1.bas'
lvsfile = 'C1.lvs'
kptpreference = 'scf.kpts'
nstepf = 1
ifixcharge = 0
iqout = 1
rescal = 2.46
&END
EOF

$FIREBALL_EXEC > out_scf.log
FMAX_SCF=$(grep -i "Fmax" out_scf.log | tail -n 1 | awk '{print $NF}')
echo "Zakoñczono SCF. F_max = $FMAX_SCF. Wygenerowano plik CHARGES."

echo "=== KROK 2: Diagonalizacja Pasm (Zamro¿ony ³adunek) ==="
cat <<EOF > fireball.in
&OPTION
basisfile = 'C1.bas'
lvsfile = 'C1.lvs'
kptpreference = 'band.kpts'
nstepf = 1
ifixcharge = 1
iqout = 1
rescal = 2.46
&END
&OUTPUT
iwrteigen = 1
iwrtdos = 0
&END
EOF

$FIREBALL_EXEC > out_band.log
FMAX_BAND=$(grep -i "Fmax" out_band.log | tail -n 1 | awk '{print $NF}')
echo "Zakoñczono analizê pasmow¹. F_max = $FMAX_BAND."
echo "Wyniki znajduj¹ siê w pliku eigen.dat."