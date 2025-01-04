# MCF_TB_SA Hyper-heuristic - AIM Cousrework
### Haocheng Yuan
---
## Project Description
This project proposes a selection hyper-heuristic using modified choice function and simulated annealing algorithm to solve MIN-SET-COVER problem. The modified choice function was proposed by Drake, Özcan and Burke in 2015. I implemented it and did some adjustion including a tabu-list and tournament selection, and combinedthem with simulated annealing algorithm. Moreover, a reheating strategy is used to present local optimal by applying a ruin-recreate heuristic and increasing the temperature.

## How to run my code
1. Unzip the 20320036_code.zip and open the project with Intellj
2. Set up **Java19** as the project sdk
3. Launch the project from **Runner** class
4. Set trial number, random seed, problem instance, run time in the **Config** class
5. The output data is saved in the **output** folder
6. If you would like to try more instances provided by yourslef, please put the file in **test_instaces** folder and add the file path in **MSC** class
  
  ![src](screenshot.png "src")

## Reference List
J. H. Drake, E. Özcan and E. K. Burke, "A modified choice function hyper-heuristic controlling unary and binary operators," 2015 IEEE Congress on Evolutionary Computation (CEC), Sendai, Japan, 2015, pp. 3389-3396, doi: 10.1109/CEC.2015.7257315.

T. -S. Khoo, B. B. Mohammad, V. -H. Wong, Y. -H. Tay and M. Nair, "A Two-Phase Distributed Ruin-and-Recreate Genetic Algorithm for Solving the Vehicle Routing Problem With Time Windows," in IEEE Access, vol. 8, pp. 169851-169871, 2020, doi: 10.1109/ACCESS.2020.3023741.