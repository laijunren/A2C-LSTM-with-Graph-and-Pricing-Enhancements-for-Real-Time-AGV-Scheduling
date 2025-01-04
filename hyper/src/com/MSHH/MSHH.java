package com.MSHH;

import com.VRP.*;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

public class MSHH {

    protected Random rng;
    private final long maxDuration;
    private long endTime;
    private int lastImproveCounter = 0;
    private final VRP problem;
    private final GeometricCooling coolingSchedule;
    private final int populationSize = 5;
    private String data;
    private double bestCost = Double.MAX_VALUE;
    private int[][] initial_solution = null;


    /**
     * @param seed The experimental seed.
     */
    public MSHH(long seed, long maxDuration, VRP problem) {

        this.rng = new Random(seed);
        this.coolingSchedule = new GeometricCooling();
        this.maxDuration = maxDuration;
        this.problem = problem;
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
     * main part of MCF_TB_SA_HH
     * step1: initialize solution memory including
     *      the first generation of population
     *      currentSolution constructed by greedy search algorithm
     *      candidateSolution
     * step2: initialize coolingSchedule with initial temperature
     * step3: pick all low level heuristic your want
     *      and create the Modified Choice Function
     * step4: main loop
     * while (run time duration is not exceeded):
     *      get the heuristic be applied by Modified Choice Function
     *      when the best score get stuck for while
     *          reheat the temperature
     *          reselect a ruin-and-recreate heuristic to recreate the current solution
     *      set the IOM and DOS parameter dependent on the selected heuristic
     *      apply the chosen heuristic, and record the time taken to apply it
     *      updates the heuristic score in the Modified Choice Function
     *      moving acceptance with simulated annealing
     *      update current temperature, reheat when get stuck
     *      update population if the current individual is better than the worst individual in population
     *      record pairs of objective values
     */
    public void solve(VRP problem) {
        // step1: initialisation solution memory
        problem.setMemorySize(populationSize + 2);
        problem.setPopulationSize(populationSize);
        double[] populationCosts = new double[populationSize];      // initialize the population
        for (int i = 0; i < populationSize; i++) {
            populationCosts[i] = problem.getFunctionValue(i);
        }
        int currentIndex = populationSize, candidateIndex = populationSize + 1;
        if (initial_solution != null) {
            problem.solutionMemory[currentIndex] = new Solution(initial_solution);
        }
        double currentCost = problem.getFunctionValue(currentIndex);
        double candidateCost = problem.getFunctionValue(candidateIndex);

        // step2: initialize cooling schedule's temperature
        coolingSchedule.setCurrentTemperature(candidateCost);

        // step3: pick all low level heuristic your want
        // hs <- { MTN } U { LS } U { RR }
        int[] hs;
        int[] mtns = problem.getHeuristicsOfType(VRP.HeuristicType.MUTATION);
        int[] lss = problem.getHeuristicsOfType(VRP.HeuristicType.LOCAL_SEARCH);
        int[] rr = problem.getHeuristicsOfType(VRP.HeuristicType.RUIN_RECREATE);
        int[] csvs = problem.getHeuristicsOfType(VRP.HeuristicType.CROSSOVER);
        hs = IntStream.concat(Arrays.stream(mtns), Arrays.stream(lss)).toArray();
        hs = IntStream.concat(Arrays.stream(hs), Arrays.stream(rr)).toArray();
        hs = IntStream.concat(Arrays.stream(hs), Arrays.stream(csvs)).toArray();

        // set up each heuristic with pairs of IOM/DOS settings:{ h1, 0.2, 0.2 }, { h1, 0.4, 0.4 }, ..., { hn, 1.0, 1.0 }.
        LowLevelHeuristic[] heuristics = new LowLevelHeuristic[hs.length * 5];
        for (int i = 0; i < hs.length; i++) {
            for (int j = 1; j <= 5; j++) {
                heuristics[i*5+j-1] = new LowLevelHeuristic(hs[i], 0.2*j, 0.2*j, System.nanoTime());
            }
        }

        // step3: initialise the Modified Choice Function with the tabu size for local search heuristic
        ModifiedChoiceFunction mcf = new ModifiedChoiceFunction(heuristics, lss.length, rng);

        StringBuilder sb = new StringBuilder();
        sb.append("start train\n");
        int iter = 0;

        // step4: main loop
        while (!hasTimeExpired()) {
            // get the heuristic be applied by Modified Choice Function
            LowLevelHeuristic llh = mcf.selectHeuristicToApply();
            if (iter++ % 3 == 0)
                llh = mcf.selectHeuristicToApply(VRP.HeuristicType.LOCAL_SEARCH, problem);

            // reheating step: ruin-and-recreate current solution when get stuck for while
            if (isNeedReheating(currentCost)) {
                llh = mcf.selectHeuristicToApply(VRP.HeuristicType.MUTATION, problem);
                coolingSchedule.reheating();
            }

            // set the IOM and DOS dependent on the selected heuristic
            problem.setDepthOfSearch(llh.getDos());
            problem.setIntensityOfMutation(llh.getIom());

            // apply the chosen heuristic, and record the time taken to apply it
            long startTime = System.nanoTime();
            candidateCost = problem.applyHeuristic(llh.getHeuristicId(), currentIndex, candidateIndex);
            long endTime = System.nanoTime();

            // update the heuristic score in the Modified Choice Function
            double delta = candidateCost - currentCost;
            mcf.updateHeuristicData(llh, startTime, endTime - startTime, delta);

            // moving acceptance with simulated annealing
            if (delta < 0 || rng.nextDouble() < Math.exp((-delta)/ coolingSchedule.getCurrentTemperature())) {
                problem.copySolution(candidateIndex, currentIndex);
                currentCost = candidateCost;
            }

            // update current temperature, reheat when get stuck
            coolingSchedule.advanceTemperature(candidateCost);

            // update population
            updatePopulation(currentIndex, currentCost, populationCosts, problem);

            // record pairs of objective values
            sb.append("iteration " + iter + ": best cost " + problem.getBestSolutionValue() + "current cost " + currentCost + "\n");
            MyUtilities.saveData("temp_solution.txt", problem.getBestEverSolution().toString());
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
     * check if reached the maximum duration in seconds
     */
    private boolean hasTimeExpired() {
        return System.currentTimeMillis() > endTime;
    }

    /**
     * check if the best score has stuck for a while and need apply a reheating and ruin-recreate
     */
    private boolean isNeedReheating(double currentCost) {
        if (currentCost < bestCost) {
            bestCost = currentCost;
            lastImproveCounter = 0;
            return false;
        } else if (lastImproveCounter > 3) {
            lastImproveCounter = 0;
            return true;
        }
        return false;
    }

    /**
     * update population if the current individual is better than the worst individual in population
     */
    private void updatePopulation(int currentIndex, double currentCost, double[] populationCosts, VRP problem) {
        double maxCost = 0;
        int maxIndex = 0;
        boolean isImproved = false;
        for (int i = 0; i < populationSize; i++) {
            if (currentCost <= populationCosts[i]) {
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
     * return output data in maxNumberOf lines in string
     */
    public String getData() {
        return data;
    }
}
