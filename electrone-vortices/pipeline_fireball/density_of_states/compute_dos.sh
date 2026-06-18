#!/bin/bash
FIREBALL_EXEC="../fireball.x"
FDATA_PATH="/home/em0/Fdata_H-ss_C-spd_Si-spd"

echo "$FDATA_PATH" > Fdata.optional

echo "Konfiguracja pliku dos.optional..."
cat <<EOF > dos.optional
1.0
1 2
401
-10.0 0.05
0
0.0  0.0
0.05
EOF

echo "=== Obliczenia Density of States (SCF + DOS Output) ==="
cat <<EOF > fireball.in
&OPTION
basisfile = 'C1.bas'
lvsfile = 'C1.lvs'
kptpreference = 'dos_full_bz.kpts'
nstepf = 1
ifixcharge = 0
iqout = 1
rescal = 2.46
&END
&OUTPUT
iwrtdos = 1
iwrteigen = 0
iwrtewf = 0
&END
EOF

$FIREBALL_EXEC > out_dos.log
FMAX_DOS=$(grep -i "Fmax" out_dos.log | tail -n 1 | awk '{print $NF}')

echo "Zakoñczono analizê DOS. F_max = $FMAX_DOS."
echo "Wyniki znajduj¹ siê w pliku dens_TOT.dat."