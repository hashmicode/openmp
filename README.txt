# Sequence Alignment Example 
# Univie

On Alma, compile with:
/opt/global/gcc-11.2.0/bin/g++ -O2 -std=c++20 -fopenmp -o gpsa  main.cpp

To Run: 
./gpsa 

To run all configurations: 
sh runall ./gpsa

To run with a specific data set use: ./gpsa --x <sequence1_filename> --y <sequence2_filename>

By default, your program will look for X.txt and Y.txt. 

Here are the available sequences: 

1. X.txt, Y.txt, size: [18481x18961] 
Random, big sequences.

2. X2.txt, Y2.txt, size: [16383x16383] 
Same, but this has the same size dimensions that nicely divide.

3. simple1.txt, simple2.txt, size: [5x6] 
Small sequences that you can use for debugging. 
You can change this file as you please.