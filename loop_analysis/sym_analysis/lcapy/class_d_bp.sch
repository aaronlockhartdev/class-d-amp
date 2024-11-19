Pin 1 0; down
Lf 1 2_1 10e-6; right=1.5
Cf 2_1 0 4.7e-6; down
W? 2_1 2_2; right
Rl 2_2 0 4; down
Rfb 2_2 3_1 680; right=1.5
Rin 3_1 0 1.8e3; down
W? 3_1 3_2; right
Cf1 3_2 0; down
Rf 3_2 4_1; right
Cf2 4_1 0; down
W? 4_1 4_2; right
R? 4_2 5_1; right
W? 2_1 2_4; up
W? 2_4 2_5; right
Rlp 2_5 5_1; down
W? 2_5 2_6; right=1.5
R? 2_6 7_1 Rl1; down
W? 5_1 5_2; right
R? 7_1 5_2 Rl2; down
W? 2_6 2_7; right
C? 2_7 6 Cl1; down
W? 7_1 7_2; right
C? 6 7_2 Cl1; down
C? 7_2 8 Cl2; down
W? 5_2 5_3; right
C? 8 5_3 Cl2; down
W? 0_1 0_2; down
W? 0_2 0_3; down
R? 6 0_1 Rl3; right
R? 7_2 0_2 Rl4; right
R? 8 0_3 Rl5; right
W? 0_2 0; right=0
R? 5_3 9_1; right=2
C? 9_1 0 Cmfb; down
R? 9_1 10; right=1.5
W? 9_1 9_2; up
R? 9_2 11_1; right=1.5
C? 11_1 10 Cmfb; down
W? 11_1 11_2; right
W? 11_2 11_3; down=1.5
W? 0_4 0; left=0
Emfb 11_3 0 opamp 0_4 10 1e8; mirror, scale=0.75
R? 11_3 12_1; right
W? 12_1 12_2; up=0.5
W? 13_1 13_2; up
Ri 12_2 13_2; right
W? 12_2 12_3; up
W? 13_2 13_3; up
Ci 12_3 13_3; right
W? 0_5 0; left=0
Ei 13_1 0 opamp 0_5 12_1 1e8; mirror, scale=0.75
R? 13_1 14_1; right
W? 2_7 2_8; right
R? 2_8 17_1 Rl1; down
R? 17_1 14_1 Rl2; down
W? 2_8 2_9; right
C? 2_9 16 Cl1; down
W? 17_1 17_2; right
C? 16 17_2 Cl1; down
C? 17_2 18 Cl2; down
W? 14_1 14_2; right
C? 18 14_2 Cl2; down
W? 0_6 0_7; down
W? 0_7 0_8; down
W? 0_7 0; right=0
R? 16 0_6 Rl3; right
R? 17_2 0_7 Rl4; right
R? 18 0_8 Rl5; right
W? 4_2 4_3; down
W? 4_3 4_4; right
Rff 4_4 14_1; up
Pout 14_2 0; down
;draw_nodes=connections, autoground=true

