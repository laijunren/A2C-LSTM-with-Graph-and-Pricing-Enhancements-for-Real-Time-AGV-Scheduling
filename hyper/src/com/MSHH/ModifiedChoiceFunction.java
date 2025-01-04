package com.MSHH;

import com.VRP.LowLevelHeuristic;
import com.VRP.MyUtilities;
import com.VRP.VRP;

import java.util.LinkedList;
import java.util.Random;
import java.util.stream.IntStream;

public class ModifiedChoiceFunction {

    /**
     * The array of heuristics paired with their low-level configurations.
     */
    private final LowLevelHeuristic[] heuristics;
    private LinkedList<Integer> tabuList;
    private LowLevelHeuristic previousLLH;
    private final Random rng;
    private final int numOfLss;

    /**
     * Global variable phi controlling the importance of improvement versus time since last application.
     */
    private double phi;

    /**
     * @param heuristics The array of heuristics paired with their low-level configurations.
     */
    public ModifiedChoiceFunction(LowLevelHeuristic[] heuristics, Random rng) {

        this.rng = rng;
        this.heuristics = heuristics;
        this.phi = 0.50;
        this.previousLLH = null;
        this.tabuList = new LinkedList<>();
        this.numOfLss = 4;
    }

    /**
     * @param heuristics The array of heuristics paired with their low-level configurations.
     * @param numOfLss The number of local search heuristics as tabu list size.
     */
    public ModifiedChoiceFunction(LowLevelHeuristic[] heuristics, int numOfLss, Random rng) {

        this.rng = rng;
        this.heuristics = heuristics;
        this.phi = 0.50;
        this.previousLLH = null;
        this.tabuList = new LinkedList<>();
        this.numOfLss = numOfLss + 1;
    }

    /**
     * Updates the data associated with the heuristic and updates the global variable <code>phi</code>.
     * See exercise sheet for details regarding the value of phi.
     *
     * @param llh The heuristic to be updated.
     * @param timeApplied Time, in nanoseconds, when the heuristic was applied.
     * @param timeTaken Time taken to apply <code>heuristic</code> in nanoseconds.
     * @param delta Objective value of the candidate solution minus current solution, f(s'_i) - f(s_i).
     */
    public void updateHeuristicData(LowLevelHeuristic llh, long timeApplied, long timeTaken, double delta) {

        // update heuristic data
        llh.setTimeLastApplied(timeApplied);
        llh.setPreviousApplicationDuration(timeTaken);
        llh.setF_delta(delta);

        // update PHI
        if (delta < 0) {
            phi = 0.99;
        } else if (delta > 0) {
            phi = Math.max(0.01, phi-0.01);
        }

        // update previous heuristic data
        if (previousLLH != null) {
            llh.setPreviousApplicationDuration(previousLLH, timeTaken);
            llh.setF_delta(previousLLH, delta);
        }
        previousLLH = llh;

    }

    /**
     * contain a tabu list
     * @return The Heuristic with the highest score.
     */
    public LowLevelHeuristic selectHeuristicToApply() {

        long t = System.nanoTime();
        int best_index = 0;
        double best_score = -Double.MAX_VALUE;

        int[] indices = MyUtilities.shuffle(IntStream.range(0, heuristics.length).toArray(), rng);

        // if more than one heuristic shares the same highest score, one is chosen at random!
        for(int i = 0; i < heuristics.length / 3; i++) {
            if (!tabuList.contains(heuristics[indices[i]].getHeuristicId())) {
                double score = calculateScore(heuristics[indices[i]], t);
                if( score > best_score|| score == best_score && rng.nextDouble() < 0.1) {
                    best_score = score;
                    best_index = indices[i];
                }
            }
        }

        tabuList.addLast(heuristics[best_index].getHeuristicId());
        if (tabuList.size() > numOfLss) {
            tabuList.removeFirst();
        }

//        System.out.println(heuristics[best_index].getHeuristicId() + "  " + best_score);

        return heuristics[best_index];
    }


    /**
     * select a Heuristic with certain type
     */
    public LowLevelHeuristic selectHeuristicToApply(VRP.HeuristicType htype, VRP problem) {

        long t = System.nanoTime();
        int best_index = 0;
        double best_score = -Double.MAX_VALUE;

        int[] indices = MyUtilities.shuffle(IntStream.range(0, heuristics.length).toArray(), rng);
        int[] hs = problem.getHeuristicsOfType(htype);

        // if more than one heuristic shares the same highest score, one is chosen at random!
        for(int i = 0; i < heuristics.length; i++) {
            int id = heuristics[indices[i]].getHeuristicId();
            boolean contain = false;
            for (int h : hs) {
                if (id == h) {
                    contain = true;
                    break;
                }
            }
            if (!contain) continue;

            double score = calculateScore(heuristics[indices[i]], t);
            if( score > best_score|| score == best_score && rng.nextDouble() < 0.1) {
                best_score = score;
                best_index = indices[i];
            }
        }

        return heuristics[best_index];
    }

    /**
     * Calculates the score for the given heuristic at the current time.
     * F_t(h_j) = ϕ_t * f_1(h_j) + ϕ_t * f_2(h_k, h_j) + δ_t * f_3(h_j)
     *
     * @param llh The heuristic to calculate the score of.
     * @param currentTime The current time in nanoseconds.
     * @return The score of the heuristic `h` at the time `currentTime`.
     */
    public double calculateScore(LowLevelHeuristic llh, long currentTime) {

        double f = 0;
        // f1 = phi * (I(h)/Y(h))
        f += phi * (-llh.getF_delta()/llh.getPreviousApplicationDuration()*1e5);
        // f2 = phi * (I(h',h)/Y(h',h))
        if (previousLLH != null) f += phi * (-llh.getF_delta(previousLLH)/llh.getPreviousApplicationDuration(previousLLH)*1e4);
        // f3 = (1-phi) * T(h)
        f += (1 - phi) * (currentTime - llh.getTimeLastApplied()) / 1e5;

        return f;
    }
}
