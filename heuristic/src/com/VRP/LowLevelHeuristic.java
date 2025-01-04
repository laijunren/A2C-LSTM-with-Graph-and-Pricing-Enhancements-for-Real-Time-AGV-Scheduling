package com.VRP;

import java.util.LinkedHashMap;

public class LowLevelHeuristic {
    private double iom, dos;
    private final int heuristicId;
    private long timeLastApplied;
    private long previousApplicationDuration;
    private double f_delta;
    private LinkedHashMap<LowLevelHeuristic,Long> previousApplicationDurations;
    private LinkedHashMap<LowLevelHeuristic,Double> f_deltas;

    public LowLevelHeuristic(int heuristicId, double iom, double dos, long startTimeNano) {
        this.heuristicId = heuristicId;
        this.iom = iom;
        this.dos = dos;
        this.timeLastApplied = startTimeNano;
        this.f_delta = -Double.MAX_VALUE;
        this.previousApplicationDuration = 0;
        this.previousApplicationDurations = new LinkedHashMap<>();
        this.f_deltas = new LinkedHashMap<>();
    }

    /**
     * get id of this heuristic
     */
    public int getHeuristicId() {
        return heuristicId;
    }

    /**
     * get DOS parameter of this heuristic
     */
    public double getDos() {
        return dos;
    }

    /**
     * get IOM parameter of this heuristic
     */
    public double getIom() {
        return iom;
    }

    /**
     * get the last time since this heuristic has applied
     */
    public long getTimeLastApplied() {
        return timeLastApplied;
    }

    /**
     * set the last time since this heuristic applied
     */
    public void setTimeLastApplied(long timeLastApplied) {
        this.timeLastApplied = timeLastApplied;
    }

    /**
     * get the delta value of last time when this heuristic has applied
     * return -infinity if it is the first call
     */
    public double getF_delta() {
        return f_delta;
    }

    /**
     * get the delta value of last time when this heuristic has applied right after specific previousLLH
     * return -infinity if it is the first call
     * @param previousLLH previous heuristic
     */
    public double getF_delta(LowLevelHeuristic previousLLH) {
        if (f_deltas.get(previousLLH) == null)
            return -Double.MAX_VALUE;
        return f_deltas.get(previousLLH);
    }

    /**
     * set the delta value after this heuristic has applied
     * @param f_delta delta value
     */
    public void setF_delta(double f_delta) {
        this.f_delta = f_delta;
    }

    /**
     * set the delta value after this heuristic has applied right after specific previousLLH
     * @param previousLLH previous heuristic
     * @param f_delta delta value
     */
    public void setF_delta(LowLevelHeuristic previousLLH, double f_delta) {
        f_deltas.put(previousLLH, f_delta);
    }

    /**
     * get the Application Duration of last time when this heuristic has applied
     * return 0 if it is the first call
     */
    public long getPreviousApplicationDuration() {
        return previousApplicationDuration;
    }

    /**
     * get the Application Duration of last time when this heuristic has applied right after specific previousLLH
     * return 0 if it is the first call
     */
    public long getPreviousApplicationDuration(LowLevelHeuristic previousLLH) {
        if (previousApplicationDurations.get(previousLLH) == null)
            return 0;
        return previousApplicationDurations.get(previousLLH);
    }

    public void setPreviousApplicationDuration(long previousApplicationDuration) {
        this.previousApplicationDuration = previousApplicationDuration;
    }

    public void setPreviousApplicationDuration(LowLevelHeuristic previousLLH, long previousApplicationDuration) {
        previousApplicationDurations.put(previousLLH, previousApplicationDuration);
    }
}
