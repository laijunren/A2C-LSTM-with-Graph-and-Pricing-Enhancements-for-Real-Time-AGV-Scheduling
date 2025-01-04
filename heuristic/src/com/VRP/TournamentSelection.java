package com.VRP;

import java.util.Random;
import java.util.stream.IntStream;

public class TournamentSelection {
    private final Random rng;
    private final int POPULATION_SIZE;
    private final int tournamentSize;
    private final VRP problem;

    public TournamentSelection(VRP problem, Random rng, int POPULATION_SIZE) {

        this.problem = problem;
        this.rng = rng;
        this.POPULATION_SIZE = POPULATION_SIZE;
        this.tournamentSize = POPULATION_SIZE / 2;
    }

    /**
     * @return The index of the chosen parent solution.
     */
    public int tournamentSelection() {

        int bestIndex = -1;
        double bestFitness = Double.MAX_VALUE;

        int[] indices = MyUtilities.shuffle(IntStream.range(0, POPULATION_SIZE).toArray(), rng);

        for(int i = 0; i < tournamentSize; i++) {
            double fitness = problem.getFunctionValue(indices[i]);

            if(fitness < bestFitness || fitness == bestFitness && rng.nextDouble() < 0.2) {
                bestFitness = fitness;
                bestIndex = indices[i];
            }
        }

        return bestIndex;
    }
}
