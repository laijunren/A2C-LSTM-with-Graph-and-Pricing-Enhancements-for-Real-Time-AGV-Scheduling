package com;

import java.util.Random;

public class Config {

    /**
     * The experimental seed, set as the date this coursework was submitted.
     */
    private final long PARENT_SEED = 20230307;
    private final long[] SEEDS;

    /**
     * algorithm type = MSHH/VNS
     */
    private final String algorithm_type = "MSHH";

    /**
     * permitted total trials = 5
     */
    private final int TOTAL_TRIALS = 3;

    /**
     * permitted run time = 3600
     */
    private final double RUN_TIME = 0.05;

    /**
     * permitted instance ID's: {1, 2, 3}
     */
    private final int INSTANCE_ID = 0;

    /**
     * maxNumberOf indicating how many pairs of objective values
     * you will be recording when you run your algorithm for multiple trials for output
     */
    private final int MAX_NUMBER_OF = 100;


    public Config() {
        /*
         * Generation of SEED values
         */
        Random random = new Random(PARENT_SEED);
        SEEDS = new long[TOTAL_TRIALS];

        for(int i = 0; i < TOTAL_TRIALS; i++) {
            SEEDS[i] = random.nextLong();
        }

    }

    public String getAlgorithm_type() {
        return algorithm_type;
    }

    public int getTotalTrails() {

        return TOTAL_TRIALS;
    }

    public long[] getSeeds() {

        return SEEDS;
    }

    public int getInstanceID() {

        return INSTANCE_ID;
    }

    public long getRunTime() {
        return (long) (RUN_TIME * 1000);
    }

    public int getMax_Number_Of() {
        return MAX_NUMBER_OF;
    }
}
