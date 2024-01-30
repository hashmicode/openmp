#include <unordered_map>
#include <omp.h>
#include "helpers.hpp"

unsigned long SequenceInfo::gpsa_sequential(float** S, float** SUB, std::unordered_map<char, int>& cmap) {
    unsigned long visited = 0;
    gap_penalty = SUB[0][cmap['*']]; // min score

	// Boundary
    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
		visited++;
	}

    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
		visited++;
	}

	// Main part
	for (unsigned int i = 1; i < rows; i++) {
		for (unsigned int j = 1; j < cols; j++) {
			float match = S[i - 1][j - 1] + SUB[ cmap.at(X[i - 1]) ][ cmap.at(Y[j-1]) ];
			float del = S[i - 1][j] + gap_penalty;
			float insert = S[i][j - 1] + gap_penalty;
			S[i][j] = std::max({match, del, insert});

			visited++;
		}
	}

    return visited;
}
// different grain size 1, 2 and 3
unsigned long SequenceInfo::gpsa_taskloop(float** S, float** SUB, std::unordered_map<char, int> cmap, int grain_size=1) {
    unsigned long visited = 0;
    gap_penalty = SUB[0][cmap['*']]; // min score

    // Boundary
    #pragma omp parallel for reduction(+:visited) schedule(static, grain_size)
    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
        visited++;
    }   
    #pragma omp parallel for reduction(+:visited) schedule(static, grain_size)
    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
        visited++;
    }
    //tried with different blocks 256 512
    int block_size = 128;
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int d = 1; d <= rows + cols - 1; d += block_size) {
                #pragma omp taskloop reduction(+:visited) grainsize(grain_size)
                for (int rowi = std::min(rows - block_size + 1, d); rowi >= std::max(1, d - cols + 1); rowi -= block_size) {
                    int colj = d - rowi + 1;
                    for (int i = rowi; i < std::min(rowi + block_size, rows); i++) {
                        for (int j = colj; j < std::min(colj + block_size, cols); j++) {
                            char newX = X[i - 1];
                            char newY = Y[j - 1];
                            try {
                                    float match = S[i - 1][j - 1] + SUB[cmap.at(newX)][cmap.at(newY)];
                                    float del = S[i - 1][j] + gap_penalty;
			                        float insert = S[i][j - 1] + gap_penalty;
                                    S[i][j] = std::max({match, del, insert});
                                    visited++;
                            } catch (const std::out_of_range& e) {
                            // avoiding out_of_range exception
                            //std::cerr  << keyX << keyY << std::endl;
                            // handling the exception as needed
                            }
                        }
                        
                    }
                }
            }
        }
    } // parallelization finish
    return visited;
}



unsigned long SequenceInfo::gpsa_tasks(float** S, float** SUB, std::unordered_map<char, int> cmap, int grain_size=1) {
	// paralellize the code below using OpenMP tasks 

    unsigned long visited = 0;
    gap_penalty = SUB[0][cmap['*']]; // min score

	// Boundary
    #pragma omp parallel for reduction(+:visited) schedule(static, grain_size) 
    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
		visited++;
	}
    #pragma omp parallel for reduction(+:visited) schedule(static, grain_size)
    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
		visited++;
	}
    //tried with different blocks 256 512
    int block_size = 128; 
	// Main part
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int d = 1; d <= rows + cols - 1; d += block_size) {
                int rowi;
                for (rowi = std::min(rows - block_size + 1, d); rowi >= std::max(1, d - cols + 1); rowi -= block_size) {
                    int colj = d - rowi + 1;
                    unsigned long visit_loc = 0;
                    #pragma omp task default(none) \
                        shared(SUB, cmap, X, Y, S, visited, d, block_size) \
                        firstprivate(visit_loc, rowi, colj) \
                        depend(in : S[rowi + block_size - 1][colj - 1]) \
                        depend(in : S[rowi - 1][colj + block_size - 1]) \
                        depend(out : S[rowi + block_size - 1][colj + block_size - 1])
                    {

                    for (int i = rowi; i < std::min(rowi + block_size, rows); i++) {
                        for (int j = colj; j < std::min(colj + block_size, cols); j++) {
                            char newX = X[i - 1];
                            char newY = Y[j - 1];
                            try {   
                                float match = S[i - 1][j - 1] + SUB[cmap.at(newX)][cmap.at(newY)];
                                float del = S[i - 1][j] + gap_penalty;
                                float insert = S[i][j - 1] + gap_penalty;
                                S[i][j] = std::max({match, del, insert});
                                visit_loc++;
                            } catch (const std::out_of_range& e) {
                                 // avoiding out_of_range exception
                                // Handle the out_of_range exception as needed
                            }
                        }
                    }
                    #pragma omp atomic
                    visited += visit_loc;
                    }
                    
                    
                }
                
            }
        }
    } // parallelization finish

    return visited;
}
