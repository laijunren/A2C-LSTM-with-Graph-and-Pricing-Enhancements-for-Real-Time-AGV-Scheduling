package com;

import com.MSHH.MSHH;
import com.MSHH.SMSHH;
import com.VNS.SVNS;
import com.VNS.VNS;
import com.VRP.*;

import java.util.Objects;

public class Runner {
    final int TOTAL_TRIALS;
    final long[] SEEDS;
    final int INSTANCE_ID;
    final long RUN_TIME;
    final int MAX_NUMBER_OF;
    final String Algorithm_Type;
    int[][] arr = {{5, 18, 26, 34, 40, 48, 62, 71, 76, 85, 97, 107, 118, 127, 133, 141},
            {4, 11, 17, 23, 32, 44, 51, 56, 67, 72, 80, 83, 94, 100, 110, 116, 125, 136, 144},
            {8, 13, 21, 35, 45, 49, 59, 74, 87, 93, 106, 114, 121, 128, 138, 146},
            {2, 12, 24, 37, 42, 55, 60, 73, 90, 101, 109, 120, 130, 135, 143},
            {6, 14, 25, 41, 54, 61, 65, 75, 84, 98, 103, 111, 119, 126, 137, 145},
            {3, 10, 20, 29, 39, 52, 64, 69, 79, 91, 104, 115, 124, 134, 147},
            {7, 9, 19, 28, 30, 38, 47, 53, 58, 70, 78, 88, 92, 102, 113, 123, 132, 142, 148},
            {1, 16, 27, 33, 43, 50, 63, 66, 77, 81, 86, 95, 99, 108, 112, 122, 131, 139},
            {0, 15, 22, 31, 36, 46, 57, 68, 82, 89, 96, 105, 117, 129, 140, 149}};

    public Runner(Config config) {
        this.TOTAL_TRIALS = config.getTotalTrails();
        this.SEEDS = config.getSeeds();
        this.RUN_TIME = config.getRunTime();
        this.INSTANCE_ID = config.getInstanceID();
        this.MAX_NUMBER_OF = config.getMax_Number_Of();
        this.Algorithm_Type = config.getAlgorithm_type();
    }

    public void runTests() {

        for(int run = 0; run < TOTAL_TRIALS; run++) {
            VRP vrp = new VRP(9, SEEDS[run], INSTANCE_ID);
            if (Objects.equals(Algorithm_Type, "MSHH")) {
                SMSHH hh = new SMSHH(SEEDS[run], RUN_TIME, vrp);
//                hh.setInitialSolution(arr);
                hh.run();
                MyUtilities.saveData(String.format("d%d_%d_output.txt", INSTANCE_ID, run), hh.getData());
            } else if (Objects.equals(Algorithm_Type, "VNS")) {
                SVNS vns = new SVNS(SEEDS[run], RUN_TIME, vrp);
//                vns.setInitialSolution(arr);
                vns.run();
                MyUtilities.saveData(String.format("d%d_%d_output.txt", INSTANCE_ID, run), vns.getData());
            }
            System.out.printf("Trial#%d:\n%f\n%s\n", run + 1, vrp.getBestSolutionValue(), vrp.getBestEverSolution());
            MyUtilities.saveData("solution.txt", vrp.getBestEverSolution().toString());
        }

    }

    public static void main(String[] args) {
        new Runner(new Config()).runTests();
    }
}
