package com.VNS;

import com.VRP.LowLevelHeuristic;
import com.VRP.MyUtilities;
import com.VRP.Solution;
import com.VRP.VRP;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

public class SVNS {

    protected Random rng;
    private long endTime;
    private final long maxDuration;
    private final VRP problem;
    private final int populationSize = 1;
    double[] populationCosts = new double[populationSize];
    private String data;
    private int[][] initial_solution = null;


    /**
     * @param seed The experimental seed.
     */
    public SVNS(long seed, long maxDuration, VRP problemdomain) {

        this.rng = new Random(seed);
        this.maxDuration = maxDuration;
        this.problem = problemdomain;
    }

    /**
     * run wrapper method to set the start time of running
     */
    public void run() {
        if (this.problem == null) {
            System.err.println("No problem domain has been loaded with loadProblemDomain()");
            System.exit(1);
        }
        this.endTime = System.currentTimeMillis() + maxDuration;
        this.solve(this.problem);
    }

    /**
     * main part of VNS
     */
    public void solve(VRP problem) {
        // step1: initialisation solution memory
        problem.setMemorySize(populationSize + 3);
        problem.setPopulationSize(populationSize);
        for (int i = 0; i < populationSize; i++) {
            populationCosts[i] = problem.getFunctionValue(i);
        }
        int currentIndex = populationSize, candidateIndex = populationSize + 1, surrogateIndex = populationSize + 2;
        if (initial_solution != null) {
            problem.solutionMemory[currentIndex] = new Solution(initial_solution);
        }
        double currentCost = problem.getFunctionValue(currentIndex);
        double candidateCost = problem.getFunctionValue(candidateIndex);
        double surrogateCost = problem.getFunctionValueWithSurrogate(surrogateIndex);

        // step3: pick all low level heuristic your want
        // hs <- { MTN } U { LS }
        int[] hs;
        int[] mtns = problem.getHeuristicsOfType(VRP.HeuristicType.MUTATION);
        int[] lss = problem.getHeuristicsOfType(VRP.HeuristicType.LOCAL_SEARCH);
        hs = IntStream.concat(Arrays.stream(mtns), Arrays.stream(lss)).toArray();

        // set up each heuristic with pairs of IOM/DOS settings:{ h1, 0.2, 0.2 }, { h1, 0.4, 0.4 }, ..., { hn, 1.0, 1.0 }.
        LowLevelHeuristic[] heuristics = new LowLevelHeuristic[hs.length];
        for (int i = 0; i < hs.length; i++) {
            heuristics[i] = new LowLevelHeuristic(hs[i], 0.2, 0.2, System.nanoTime());
        }
        // set the IOM and DOS dependent on the selected heuristic
        problem.setDepthOfSearch(0.2);
        problem.setIntensityOfMutation(0.2);

        StringBuilder sb = new StringBuilder();
        sb.append("start train\n");
        int iter = 0;

        // step4: main loop
        while (!hasTimeExpired()) {
            if (iter++ <= populationSize) {
                int h_ID = 0;
                do {
                    LowLevelHeuristic llh = heuristics[h_ID];
                    candidateCost = problem.applyHeuristic(llh.getHeuristicId(), currentIndex, candidateIndex);

                    // moving acceptance
                    if (candidateCost - currentCost < 0) {
                        problem.copySolution(candidateIndex, currentIndex);
                        currentCost = candidateCost;
                        h_ID = 0;
                    } else {
                        h_ID++;
                    }
                } while (h_ID < heuristics.length);

                // update population
                updatePopulation(currentIndex, currentCost, problem);

                problem.copySolution(currentIndex, surrogateIndex);
                surrogateCost = problem.getFunctionValueWithSurrogate(surrogateIndex);

                // record pairs of objective values
                sb.append("iteration " + iter + ": best cost " + problem.getBestSolutionValue() + "current cost " + currentCost + "\n");
                MyUtilities.saveData("temp_solution.txt", problem.getBestEverSolution().toString());
            } else if (iter % 10 == 0) {
                problem.copySolution(surrogateIndex, candidateIndex);
                candidateCost = problem.getFunctionValue(candidateIndex);
                if (candidateCost < problem.bestEverObjectiveFunction) {
                    problem.bestEverObjectiveFunction = candidateCost;
                    problem.bestEverSolution = problem.solutionMemory[candidateIndex].deepCopy();
                }
                if (candidateCost - currentCost < 0) {
                    problem.copySolution(candidateIndex, currentIndex);
                    currentCost = candidateCost;
                }
                // update population
                updatePopulation(currentIndex, currentCost, problem);
                problem.copySolution(currentIndex, surrogateIndex);
                surrogateCost = problem.getFunctionValueWithSurrogate(surrogateIndex);

                // record pairs of objective values
                sb.append("iteration " + iter + ": best cost " + problem.getBestSolutionValue() + "current cost " + currentCost + "\n");
                MyUtilities.saveData("temp_solution.txt", problem.getBestEverSolution().toString());
//                surrogateCost *= 1.2;
            } else {
                int h_ID = 0;
                do {
                    LowLevelHeuristic llh = heuristics[h_ID];
                    candidateCost = problem.applyHeuristicWithSurrogate(llh.getHeuristicId(), surrogateIndex, candidateIndex, populationCosts);

                    // moving acceptance
                    if (candidateCost - surrogateCost < 0) {
                        problem.copySolution(candidateIndex, surrogateIndex);
                        surrogateCost = candidateCost;
                        h_ID = 0;
                    } else {
                        h_ID++;
                    }
                } while (h_ID < heuristics.length);
            }
        }

        this.data = sb.toString();
    }

    public void setInitialSolution(int[][] solution) {
        int[][] arr = new int[solution.length][];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = solution[i].clone();
        }
        this.initial_solution = arr;
    }

    /**
     * update population if the current individual is better than the worst individual in population
     */
    private void updatePopulation(int currentIndex, double currentCost, VRP problem) {
        double maxCost = 0;
        int maxIndex = 0;
        boolean isImproved = false;
        for (int i = 0; i < populationSize; i++) {
            if (currentCost==populationCosts[i]) return;
            if (currentCost < populationCosts[i]) {
                isImproved = true;
            }
            if (populationCosts[i] > maxCost) {
                maxCost = populationCosts[i];
                maxIndex = i;
            }
        }
        if (isImproved) {
            problem.copySolution(currentIndex, maxIndex);
            populationCosts[maxIndex] = currentCost;
        }
    }

    /**
     * check if reached the maximum duration in seconds
     */
    private boolean hasTimeExpired() {
        return System.currentTimeMillis() > endTime;
    }

    /**
     * return output data in maxNumberOf lines in string
     */
    public String getData() {
        return data;
    }

}
