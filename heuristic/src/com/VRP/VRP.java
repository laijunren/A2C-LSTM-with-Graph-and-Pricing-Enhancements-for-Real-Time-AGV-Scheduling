package com.VRP;

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Objects;
import java.util.Random;
import java.util.stream.IntStream;

public class VRP {
    protected Random rng;
    private int lrepeats;
    private int mrepeats;
    private double rprop;
    private int populationSize;

    private Order[] orders;
    public Solution[] solutionMemory;
    public double bestEverObjectiveFunction;
    public Solution bestEverSolution;

    private final int[] mutations = new int[]{ 0 };
    private final int[] localSearches = new int[]{ 2, 1 };
    private final int[] ruin_recreate = new int[]{ }; // 5
    private final int[] crossovers = new int[]{ };
    private final int[] others = new int[]{ };

    private TournamentSelection tournamentSelection;

    public VRP(long seed, int[][] orders) {
        this.rng = new Random(seed);

        this.setDepthOfSearch(0.2);
        this.setIntensityOfMutation(0.2);

        loadInstance(orders);
        this.bestEverObjectiveFunction = (int) Double.POSITIVE_INFINITY;
        setMemorySize(2);
    }

    public void loadInstance(int[][] orders) {
        this.orders = new Order[orders.length];
        for (int i = 0; i < orders.length; i++) {
            this.orders[i] = new Order(orders[i][0], "1", "2", orders[i][1], orders[i][2],
                    orders[i][3], orders[i][4], orders[i][5], orders[i][6]);
        }
    }

    /**
     * set solution memory size and initialize all solutions
     * @param memorySize size of solution memory
     */
    public void setMemorySize(int memorySize) {
        this.solutionMemory = new Solution[memorySize];
        for(int i = 0; i < memorySize; i++) {
            initialiseSolution(i);
        }
    }

    /**
     * initialize the solution randomly
     * @param index solution index
     */
    public void initialiseSolution(int index) {
        int[] v = IntStream.range(0, orders.length).toArray();
        solutionMemory[index] = new Solution(v);

        double temp = getFunctionValueWithSurrogate(index);
        if (temp < bestEverObjectiveFunction) {
            this.bestEverObjectiveFunction = temp;
            this.bestEverSolution = solutionMemory[index];
        }
    }

    /**
     * set population size for crossover low level heuristic
     * if you are not using crossover heuristic, it's not necessary to call this function
     * @param populationSize population size
     */
    public void setPopulationSize(int populationSize) {
        this.populationSize = populationSize;
        this.tournamentSelection = new TournamentSelection(this, rng, populationSize);
    }


    /**
     * Objective value function:
     * @param solutionIndex solution index
     * @return objective value
     */
    public double getFunctionValue(int solutionIndex) {
        return getFunctionValueWithSurrogate(solutionIndex);
    }


    /**
     * Objective value function:
     * @param solutionIndex solution index
     * @return objective value
     */
    public double getFunctionValueWithSurrogate(int solutionIndex) {
        Solution solution = solutionMemory[solutionIndex];
        double cost = 0;
//        double battery=80;
        int prevOrderIndex = solution.getVariable(0);
        if (prevOrderIndex < 0) return 0;
        cost += distanceCost(orders[prevOrderIndex]);
        for (int j = 1; j < solution.getLength(); j++) {
            double costPerOrder=0;
            costPerOrder += distanceCost(orders[prevOrderIndex], orders[solution.getVariable(j)]);
            costPerOrder += distanceCost(orders[solution.getVariable(j)]);
            prevOrderIndex = solution.getVariable(j);
            cost+=costPerOrder;
//            battery-=costPerOrder/100;
//            if (battery<20) {
//                cost += (80 - battery) * 60;
//                battery = 80;
//            }
        }
        return cost;
    }

    private double distanceCost(Order o){
        double cost = 0;
        if (o.getStartZ() != o.getEndZ()) cost += 60;
//        if (dic.get(o.getStartNode()+o.getEndNode())!=null)
//            cost += dic.get(o.getStartNode()+o.getEndNode());
//        else {
        cost += ManhattanDistance(o.getStartX(), o.getEndX(), o.getEndX(), o.getEndY());
//        System.out.println(o.getStartNode()+" "+o.getEndNode());
//        }
        return cost;
    }

    private double distanceCost(Order o1, Order o2){
        if (Objects.equals(o1.getEndNode(), o2.getStartNode()))
            return 0;
        else {
            double cost = 0;
            if (o1.getEndZ() != o2.getStartZ()) cost += 60;
//            if (dic.get(o1.getEndNode()+o2.getStartNode())!=null) {
//                cost += dic.get(o1.getEndNode() + o2.getStartNode());
//            } else {
            cost += ManhattanDistance(o1.getEndX(), o1.getEndY(), o2.getStartX(), o2.getStartY());
//            System.out.println(o1.getEndNode()+" "+o2.getStartNode());
//            }
            return cost;
        }
    }

    private double ManhattanDistance(double x1, double y1, double x2, double y2) {
        return Math.abs(x1-x2)+Math.abs(y1-y2);
    }

    /**
     * @param heuristicID integer id represent the heuristic
     * @return dest solution objective value
     */
    public double applyHeuristic(int heuristicID, int sourceSolutionIndex, int destSolutionIndex) {
        solutionMemory[destSolutionIndex] = solutionMemory[sourceSolutionIndex].deepCopy();
        double candidateCost = bestEverObjectiveFunction;
        if (heuristicID == 0) {
            candidateCost = this.applyHeuristic0(destSolutionIndex, true);
        } else if (heuristicID == 1) {
            candidateCost = this.applyHeuristic1(destSolutionIndex, true);
        } else if (heuristicID == 2) {
            candidateCost = this.applyHeuristic2(destSolutionIndex, true);
        } else {
            System.err.println("Heuristic " + heuristicID + "does not exist");
            System.exit(0);
        }
        if (candidateCost < this.bestEverObjectiveFunction) {
            this.bestEverObjectiveFunction = candidateCost;
            this.bestEverSolution = solutionMemory[destSolutionIndex].deepCopy();
        }

        return candidateCost;
    }

    /**
     * @param heuristicID integer id represent the heuristic
     * @return dest solution objective value
     */
    public double applyHeuristicWithSurrogate(int heuristicID, int sourceSolutionIndex, int destSolutionIndex, double[] populationCosts) {
        solutionMemory[destSolutionIndex] = solutionMemory[sourceSolutionIndex].deepCopy();
        double candidateCost = Double.POSITIVE_INFINITY;
//        this.populationCosts = populationCosts;
        if (heuristicID == 0) {
            candidateCost = this.applyHeuristic0(destSolutionIndex, false);
        } else if (heuristicID == 1) {
            candidateCost = this.applyHeuristic1(destSolutionIndex, false);
        } else if (heuristicID == 2) {
            candidateCost = this.applyHeuristic2(destSolutionIndex, false);
        } else {
            System.err.println("Heuristic " + heuristicID + "does not exist");
            System.exit(0);
        }
        return candidateCost;
    }

    /**
     * mutation LLH 0 random bit swap
     */
    private double applyHeuristic0(int solutionIndex, boolean real) {
        Solution solution = solutionMemory[solutionIndex];
        for (int i = 0; i < mrepeats; i++) {
            int l = solution.getLength();
            int index1 = rng.nextInt(l);
            int index2 = rng.nextInt(l);
            while (index1 == index2) {
                index2 = rng.nextInt(l);
            }
            solution.bitswap(index1, index2);
        }

        return real?getFunctionValue(solutionIndex):getFunctionValueWithSurrogate(solutionIndex);
    }

    /**
     * local search LLH 1 Steepest Descent Hill Climbing
     */
    private double applyHeuristic1(int solutionIndex, boolean real) {
        Solution solution = solutionMemory[solutionIndex];
        double tmpEval;
        for (int i = 0; i < lrepeats; i++) {
            boolean improved = false;
            int bestIndex1 = 0;
            int bestIndex2 = 0;
            double bestEval = real?getFunctionValue(solutionIndex):getFunctionValueWithSurrogate(solutionIndex);


            for (int j = 0; j < orders.length; j++) {
                for (int k = j; k < orders.length; k++) {
                    solution.bitswap(j, k);
                    tmpEval = real?getFunctionValue(solutionIndex):getFunctionValueWithSurrogate(solutionIndex);

                    if (tmpEval <= bestEval) {
                        bestEval = tmpEval;
                        bestIndex1 = j;
                        bestIndex2 = k;
                        improved = true;
                    }
                    solution.bitswap(j, k);
                }
            }
            if (improved) {
                solution.bitswap(bestIndex1, bestIndex2);
            }
        }
        return real?getFunctionValue(solutionIndex):getFunctionValueWithSurrogate(solutionIndex);
    }

    /**
     * local search LLH 2 First Bit Hill Climbing
     */
    private double applyHeuristic2(int solutionIndex, boolean real) {
        Solution solution = solutionMemory[solutionIndex];
        double bestEval ,tmpEval;
        for (int i = 0; i < lrepeats; i++) {
            int[] perm = MyUtilities.shuffle(IntStream.range(0, orders.length).toArray(), rng);
            bestEval = real?getFunctionValue(solutionIndex):getFunctionValueWithSurrogate(solutionIndex);

            int counter = 0;
            int cnt=0;
            outerloop:
            for (int j = 0; j < orders.length; j++) {
                for (int k = j; k < orders.length; k++) {
                    solution.bitswap(perm[j], perm[k]);
                    tmpEval = real?getFunctionValue(solutionIndex):getFunctionValueWithSurrogate(solutionIndex);

                    if (tmpEval <= bestEval) {
                        bestEval = tmpEval;
//                        if (counter++ >= 5) break outerloop;
                    } else {
                        solution.bitswap(perm[j], perm[k]);
                    }

                    if (cnt++>orders.length&&real)
//                        return getFunctionValue(solutionIndex);
                        break outerloop;
                }
            }
        }
        return real?getFunctionValue(solutionIndex):getFunctionValueWithSurrogate(solutionIndex);
    }

    /**
     * crossover LLH 7 Order Crossover
     */
    private double applyHeuristic7(int solutionIndex) {
        Solution parent1 = solutionMemory[solutionIndex];
        Solution parent2 = solutionMemory[tournamentSelection.tournamentSelection()];
        Solution child = parent2.deepCopy();

//        for (int i = 0; i < solution_length; i++) {
//            if (rng.nextDouble() < 0.5) {
//                boolean originValue = child.getVariable(i);
//                boolean newValue = parent1.getVariable(i);
//                child.setVariable(i, newValue);
//            }
//        }

        solutionMemory[solutionIndex] = child;
        return getFunctionValue(solutionIndex);
    }


    /**
     * DOS parameter setting method
     * @param depthOfSearch DOS parameter
     */
    public void setDepthOfSearch(double depthOfSearch) {
        if (depthOfSearch <= 0.2) {
            this.lrepeats = 1;
        } else if (depthOfSearch <= 0.4) {
            this.lrepeats = 2;
        } else if (depthOfSearch <= 0.6) {
            this.lrepeats = 3;
        } else if (depthOfSearch <= 0.8) {
            this.lrepeats = 4;
        } else if (depthOfSearch <= 1) {
            this.lrepeats = 5;
        } else {
            this.lrepeats = 6;
        }
    }

    /**
     * IOM parameter setting method
     * @param intensityOfMutation IOM parameter
     */
    public void setIntensityOfMutation(double intensityOfMutation) {
        if (intensityOfMutation <= 0.2) {
            mrepeats = 1;
            rprop = 0.1;
        } else if (intensityOfMutation <= 0.4) {
            mrepeats = 2;
            rprop = 0.2;
        } else if (intensityOfMutation <= 0.6) {
            mrepeats = 3;
            rprop = 0.3;
        } else if (intensityOfMutation <= 0.8) {
            mrepeats = 4;
            rprop = 0.4;
        } else if (intensityOfMutation <= 1) {
            mrepeats = 5;
            rprop = 0.5;
        } else {
            mrepeats = 6;
            rprop = 0.6;
        }
    }

    /**
     * deep copy from src solution to dest solution
     * @param src source solution index
     * @param dest dest solution index
     */
    public void copySolution(int src, int dest) {
        solutionMemory[dest] = solutionMemory[src].deepCopy();
    }

    public double getBestSolutionValue() {
        return bestEverObjectiveFunction;
    }

    public Solution getBestEverSolution() {
        return bestEverSolution;
    }

    public int[] getBestSolutionOrder() {
        int[] orders = new int[this.orders.length];
        for (int i = 0; i < this.orders.length; i++) {
            orders[i] = this.orders[bestEverSolution.variables[i]].getOrderNum();
        }
        return orders;
    }

    public String toString() {
        return "Vehicle Routing Problem";
    }

    /**
     * @param hType HeuristicType
     * @return an array of low level heuristic index in hType
     */
    public int[] getHeuristicsOfType(VRP.HeuristicType hType) {
        switch (hType.ordinal()) {
            case 0:
                return this.mutations;
            case 1:
                return this.crossovers;
            case 2:
                return this.ruin_recreate;
            case 3:
                return this.localSearches;
            case 4:
                return this.others;
            default:
                return null;
        }
    }

    public enum HeuristicType {
        MUTATION,
        CROSSOVER,
        RUIN_RECREATE,
        LOCAL_SEARCH,
        OTHER;

        HeuristicType() {
        }
    }
}
