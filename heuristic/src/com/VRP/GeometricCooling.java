package com.VRP;

public class GeometricCooling {

    private double initialTemperature;
    private double currentTemperature;

    /**
     * The $\alpha$ parameter of the cooling schedule.
     */
    private final double dAlpha = 0.995d;

    /**
     *  set to 5% of the initial solution cost for initialTemperature
     */
    private final double c = 0.02d;


    public void setCurrentTemperature(double initialSolutionFitness) {
        this.initialTemperature = c * initialSolutionFitness;
        this.currentTemperature = c * initialSolutionFitness;
    }

    public double getCurrentTemperature() {

        return this.currentTemperature;
    }

    /**
     * update temperature: T_{i + 1} = alpha * T_i
     */
    public void advanceTemperature(double candidateCost) {
        currentTemperature = currentTemperature * dAlpha;
    }

    /**
     * reheat if get stuck: T_i = T_0
     */
    public void reheating() {
        currentTemperature = initialTemperature;
    }

}
