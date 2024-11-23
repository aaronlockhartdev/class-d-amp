Vs 1 0 s; down

Lfilt 1 2_1; right=1.5
Cfilt 2_1 0; down
Rload 2_2 0 4; down

Rfb 2_2 3_1; right=1.5
Rin 3_1 0; down

Cf1 3_2 0; down
Rf 3_2 4_1; right=1.5
Cf2 4_1 0; down

R? 4_2 5_1; right=1.5

C? 10_1 11 Cl1; down
C? 11 12_1 Cl2; down
C? 12_3 13 Cl3; down
C? 13 14_1 Cl4; down

R? 10_10 12_10 Rl1; down
R? 12_11 14_10 Rl2; down

R? 11 0_10 Rlg1; left
R? 12_2 0_11 Rlg2; left
R? 13 0_12 Rlg3; left

Rlbp 10_20 14_20; down

R? 14_10 5_1; down

E? 20_1 0 opamp 0_20 5_2 A; right, mirror
C? 20_2 5_3 Ci; left
R? 20_3 5_4 Ri2; left

R? 20_1 21_1 Ri1; right
E? 30_1 0 opamp 0_30 21_1 A; right, mirror
C? 30_2 21_2 Ci; left
R? 30_3 21_3 Ri2; left

R? 30_1 out; right=1.5

C? 40_1 41 Cl1; down
C? 41 42_1 Cl2; down
C? 42_3 43 Cl3; down
C? 43 44_1 Cl4; down

R? 40_10 42_10 Rl1; down
R? 42_11 44_10 Rl2; down

R? 41 0_40 Rlg1; left
R? 42_2 0_41 Rlg2; left
R? 43 0_42 Rlg3; left

R? 44_1 out; down

R? out 4_ff2; down=1.5

# Wires and open circuit elements
W 2_1 2_2; right
W 3_1 3_2; right
W 4_1 4_2; right

W 12_1 12_2; down=0.5
W 12_2 12_3; down=0.5
W 10_1 10_10; right=0.75
W 12_1 12_10; right=0.75
W 12_3 12_11; right=0.75
W 14_1 14_10; right=0.75

W 10_10 10_20; right=0.75
W 14_10 14_20; right=0.75

W 0_10 0_11; down
W 0_11 0_12; down
W 0_11 0; left=0

W 2_ff2 10_10; down
W 2_ff3 40_1; down

W 2_2 2_ff1; up
W 2_ff1 2_ff2; right=5.5
W 2_ff2 2_ff3; right=9

W 5_1 5_2; right=1.5

W 0_20 0; left=0

W 20_1 20_2; up=1.5
W 5_2 5_3; up

W 20_2 20_3; up=0.75
W 5_3 5_4; up=0.75

W 0_30 0; left=0

W 30_1 30_2; up=1.5
W 21_1 21_2; up

W 30_2 30_3; up=0.75
W 21_2 21_3; up=0.75

W 42_1 42_2; down=0.5
W 42_2 42_3; down=0.5
W 40_1 40_10; right=0.75
W 42_1 42_10; right=0.75
W 42_3 42_11; right=0.75
W 44_1 44_10; right=0.75

W 0_40 0_41; down
W 0_41 0_42; down
W 0_41 0; left=0

W 4_2 4_ff1; down
W 4_ff1 4_ff2; right

# Schematic drawing instructions
;draw_nodes=connections, autoground=true, label_nodes=alpha, label_ids=false
