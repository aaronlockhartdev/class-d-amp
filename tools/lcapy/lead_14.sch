Vs in 0 s; down=1.5
Cl1 1_1 2; right
Cl2 2 3_1; right

Cl3 3_3 4; right
Cl4 4 5_1; right

Rl1 1_2 3_4; right
Rl2 3_5 5_2; right

Rlg1 2 0; down=1.5
Rlg2 3_2 0; down=1.5
Rlg3 4 0; down=1.5

# Wires and open circuit elements
W in 1_1; right
W 3_1 3_2; right=0.5
W 3_2 3_3; right=0.5

W 1_1 1_2; up=0.75
W 3_1 3_4; up=0.75

W 3_3 3_5; up=0.75
W 5_1 5_2; up=0.75

W 5_1 out; right

# Schematic drawing instructions
;draw_nodes=connections, autoground=true, label_nodes=alpha, label_ids=false
