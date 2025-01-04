package com;

import com.MSHH.MSHH;
import com.VNS.VNS;
import com.VRP.MyUtilities;
import com.VRP.VRP;

import java.util.ArrayList;
import java.util.Objects;

public class Runner {
    final int TOTAL_TRIALS;
    final long[] SEEDS;
    final int INSTANCE_ID;
    final long RUN_TIME;
    final int MAX_NUMBER_OF;
    final String Algorithm_Type;
    int[][] orders = {{1, 121, 100, 1, 80, 180, 2},
            {2, 121, 100, 1, 72, 89, 2},
            {3, 3, 84, 0, 80, 180, 2},
            {4, 121, 100, 1, 72, 89, 2},
            {5, 3, 84, 0, 80, 180, 2},
            {6, 26, 280, 2, 111, 70, 2},
            {7, 121, 100, 1, 111, 70, 2},
            {8, 121, 100, 1, 72, 89, 2},
            {9, 80, 180, 2, 72, 89, 2}};

    public Runner(Config config) {
        this.TOTAL_TRIALS = config.getTotalTrails();
        this.SEEDS = config.getSeeds();
        this.RUN_TIME = config.getRunTime();
        this.INSTANCE_ID = config.getInstanceID();
        this.MAX_NUMBER_OF = config.getMax_Number_Of();
        this.Algorithm_Type = config.getAlgorithm_type();
    }

    public void runTests() {
        ArrayList<String>[] arrayLists = new ArrayList[2];
        arrayLists[0] = new ArrayList<String>();
        arrayLists[0].add("asdf");
        arrayLists[0].add("bsdf");
        arrayLists[0].remove(0);
        int size = arrayLists[0].size();
        int[][] ord = new int[size][];
        ord[0] = new int[7];
        ord[0][0] = 1;
        System.out.println(arrayLists[0].get(0));

        for(int run = 0; run < TOTAL_TRIALS; run++) {
            VRP vrp = new VRP(SEEDS[run], orders);
            if (Objects.equals(Algorithm_Type, "MSHH")) {
                MSHH hh = new MSHH(SEEDS[run], RUN_TIME, vrp);
//                hh.setInitialSolution(arr);
                hh.run();
                MyUtilities.saveData(String.format("d%d_%d_output.txt", INSTANCE_ID, run), hh.getData());
            } else if (Objects.equals(Algorithm_Type, "VNS")) {
                VNS vns = new VNS(SEEDS[run], RUN_TIME, vrp);
//                vns.setInitialSolution(arr);
                vns.run();
                MyUtilities.saveData(String.format("d%d_%d_output.txt", INSTANCE_ID, run), vns.getData());
            }
            System.out.printf("Trial#%d:\n%f\n%s\n", run + 1, vrp.getBestSolutionValue(), vrp.getBestEverSolution());
            MyUtilities.saveData("solution.txt", vrp.getBestEverSolution().toString());
            int[] arr = vrp.getBestSolutionOrder();
        }

    }

    public static void main(String[] args) {
        new Runner(new Config()).runTests();
    }
}
