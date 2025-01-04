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

    private Order[] orders;
    private HashMap<String, Double> dic;
    private int agvNum = 9;
    private int populationSize = 5;
    public Solution[] solutionMemory;
    public double[] populationCosts;
    public double bestEverObjectiveFunction;
    public Solution bestEverSolution;

    private final int[] mutations = new int[]{ 0, 1, 6 };
    private final int[] localSearches = new int[]{ 4 };
    private final int[] ruin_recreate = new int[]{ }; // 5
    private final int[] crossovers = new int[]{ };
    private final int[] others = new int[]{ };

    private TournamentSelection tournamentSelection;

    private static final String[] instances = new String[] {
            "test_instances/d1_82.txt",
            "test_instances/d2_150.txt",
            "test_instances/d3_243.txt",
            "test_instances/d4_181.txt",
            "test_instances/d5_164.txt"
    };

    public VRP(int agvNum, long seed, int instanceID) {
        this.agvNum = agvNum;
        this.rng = new Random(seed);

        this.setDepthOfSearch(0.2);
        this.setIntensityOfMutation(0.2);

        loadInstance(instanceID - 1);
        this.bestEverObjectiveFunction = (int) Double.POSITIVE_INFINITY;
        setMemorySize(2);
    }

    /**
     * load a problem instance with instance id
     * @param index instance index
     */
    public void loadInstance(int index) {
        BufferedReader buffread;
        try {
            FileReader read = new FileReader(instances[index]);
            buffread = new BufferedReader(read);
            readInInstance(buffread);
        } catch (FileNotFoundException a) {
            try {
                InputStream fis = getClass().getResourceAsStream(instances[index]);
                assert fis != null;
                InputStreamReader reader = new InputStreamReader(fis);
                buffread = new BufferedReader(reader);
                readInInstance(buffread);
            } catch (NullPointerException n) {
                n.printStackTrace();
                System.err.println("cannot find file " + instances[index]);
                System.exit(-1);
            }
        }
    }

    /**
     * read a instance data from system file
     * @param buffread file reader of instance data file
     */
    private void readInInstance(BufferedReader buffread) {
        try {
            String readline;
            readline = buffread.readLine();
            orders = new Order[Integer.parseInt(readline)];
            for (int i = 0; i < orders.length; i++) {
                readline = buffread.readLine();
                String[] str = readline.split(" ");
                orders[i] = new Order(Integer.parseInt(str[0]), str[1], str[2], Integer.parseInt(str[3]), Integer.parseInt(str[4]),
                        Integer.parseInt(str[5]), Integer.parseInt(str[6]), Integer.parseInt(str[7]), Integer.parseInt(str[8]));
            }
            readline = buffread.readLine();
            dic = new HashMap<>();
            int l = Integer.parseInt(readline);
            for (int i = 0; i < l; i++) {
                readline = buffread.readLine();
                String[] str = readline.split("\t");
                dic.put(str[1]+str[0], Double.parseDouble(str[2]));
                dic.put(str[0]+str[1], Double.parseDouble(str[2]));
            }
        } catch (IOException b) {
            System.err.println(b.getMessage());
            System.exit(0);
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
        int[][] v = new int[agvNum][];
        int l = orders.length / agvNum;
        int j = 0;
        for (int i = 0; i < agvNum-1; i++) {
            v[i] = IntStream.range(j, j+l).toArray();
            j += l;
        }
        v[agvNum-1] = IntStream.range(j, orders.length).toArray();
        solutionMemory[index] = new Solution(v);

        double temp = getFunctionValue(index);
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
        double timeCost = 0;
        for (int i = 0; i < agvNum; i++) {
            double tempCost = getFunctionValueOfSingleAGV(solutionIndex, i);
            timeCost = Math.max(timeCost, tempCost);
        }
        try {
            Thread.sleep(1);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        return timeCost;
    }


    /**
     * Objective value function:
     * @param solutionIndex solution index
     * @return objective value
     */
    public double getFunctionValueWithSurrogate(int solutionIndex) {
        double timeCost = 0;

//        double[] diff = new double[populationSize];
//        double sum = 0;
//        for (int i = 0; i < populationSize; i++) {
//            diff[i] = 1/EuclideanDistance(i, solutionIndex);
//            sum += diff[i];
//        }
//        for (int i = 0; i < populationSize; i++) {
//            timeCost+=populationCosts[i]*diff[i]/sum;
//        }

        for (int i = 0; i < agvNum; i++) {
            double tempCost = getFunctionValueOfSingleAGV(solutionIndex, i);
            timeCost = Math.max(timeCost, tempCost);
        }
//        timeCost = (rng.nextDouble()*0.2+1)*timeCost;
        return timeCost;
    }

    public double getFunctionValueOfSingleAGV(int solutionIndex, int agvIndex) {
        Solution solution = solutionMemory[solutionIndex];
        double cost = 0;
        double battery=80;
        int prevOrderIndex = solution.getVariable(agvIndex, 0);
        if (prevOrderIndex < 0) return 0;
        cost += distanceCost(orders[prevOrderIndex]);
        for (int j = 1; j < solution.getLengthOfSingleAGV(agvIndex); j++) {
            double costPerOrder=0;
            costPerOrder += distanceCost(orders[prevOrderIndex], orders[solution.getVariable(agvIndex, j)]);
            costPerOrder += distanceCost(orders[solution.getVariable(agvIndex, j)]);
            prevOrderIndex = solution.getVariable(agvIndex, j);
            cost+=costPerOrder;
            battery-=costPerOrder/100;
            if (battery<20) {
                cost += (80 - battery) * 60;
                battery = 80;
            }
        }
        return cost;
    }

    private double distanceCost(Order o){
        double cost = 0;
        if (o.getStartZ() != o.getEndZ()) cost += 60;
        if (dic.get(o.getStartNode()+o.getEndNode())!=null)
            cost += dic.get(o.getStartNode()+o.getEndNode());
        else {
            cost += ManhattanDistance(o.getStartX(), o.getEndX(), o.getEndX(), o.getEndY());
            System.out.println(o.getStartNode()+" "+o.getEndNode());
        }
        return cost;
    }

    private double distanceCost(Order o1, Order o2){
        if (Objects.equals(o1.getEndNode(), o2.getStartNode()))
            return 0;
        else {
            double cost = 0;
            if (o1.getEndZ() != o2.getStartZ()) cost += 60;
            if (dic.get(o1.getEndNode()+o2.getStartNode())!=null) {
                cost += dic.get(o1.getEndNode() + o2.getStartNode());
            } else {
                cost += ManhattanDistance(o1.getEndX(), o1.getEndY(), o2.getStartX(), o2.getStartY());
                System.out.println(o1.getEndNode()+" "+o2.getStartNode());
            }
            return cost;
        }
    }

    private double ManhattanDistance(double x1, double y1, double x2, double y2) {
        return Math.abs(x1-x2)+Math.abs(y1-y2);
    }

    private double EuclideanDistance(int solutionIndex1, int solutionIndex2) {
        double x = 0;
        for (int i = 0; i < agvNum; i++) {
            int[] a = solutionMemory[solutionIndex1].variables[i];
            int[] b = solutionMemory[solutionIndex2].variables[i];
            if (a.length > b.length) {
                for (int j = 0; j < b.length; j++) {
                    if (a[j] != b[j]) x++;
                }
                for (int j = b.length; j < a.length; j++) {
                    x++;
                }
            } else {
                for (int j = 0; j < a.length; j++) {
                    if (a[j] != b[j]) x++;
                }
                for (int j = a.length; j < b.length; j++) {
                    x++;
                }
            }
        }
        return Math.sqrt(x);
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
        } else if (heuristicID == 3) {
            candidateCost = this.applyHeuristic3(destSolutionIndex, true);
        } else if (heuristicID == 4) {
            candidateCost = this.applyHeuristic4(destSolutionIndex, true);
        } else if (heuristicID == 5) {
            candidateCost = this.applyHeuristic5(destSolutionIndex, true);
        } else if (heuristicID == 6) {
            candidateCost = this.applyHeuristic6(destSolutionIndex, true);
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
        this.populationCosts = populationCosts;
        if (heuristicID == 0) {
            candidateCost = this.applyHeuristic0(destSolutionIndex, false);
        } else if (heuristicID == 1) {
            candidateCost = this.applyHeuristic1(destSolutionIndex, false);
        } else if (heuristicID == 2) {
            candidateCost = this.applyHeuristic2(destSolutionIndex, false);
        } else if (heuristicID == 3) {
            candidateCost = this.applyHeuristic3(destSolutionIndex, false);
        } else if (heuristicID == 4) {
            candidateCost = this.applyHeuristic4(destSolutionIndex, false);
        } else if (heuristicID == 5) {
            candidateCost = this.applyHeuristic5(destSolutionIndex, false);
        } else if (heuristicID == 6) {
            candidateCost = this.applyHeuristic6(destSolutionIndex, false);
        } else {
            System.err.println("Heuristic " + heuristicID + "does not exist");
            System.exit(0);
        }
        return candidateCost;
    }


    /**
     * mutation LLH 0 random bit insert
     */
    private double applyHeuristic0(int solutionIndex, boolean real) {
        Solution solution = solutionMemory[solutionIndex];
        for (int i = 0; i < mrepeats; i++) {
            int agv1 = rng.nextInt(agvNum);
            int agv2 = rng.nextInt(agvNum);
            while (agv1 == agv2 || solution.getLengthOfSingleAGV(agv2) <= 1) {
                agv1 = rng.nextInt(agvNum);
                agv2 = rng.nextInt(agvNum);
            }
            int index1 = rng.nextInt(solution.getLengthOfSingleAGV(agv1));
            int index2 = rng.nextInt(solution.getLengthOfSingleAGV(agv2));

            solution.bitinsert(agv1, index1, agv2, index2);
        }

        return real?getFunctionValue(solutionIndex):getFunctionValueWithSurrogate(solutionIndex);
    }

    /**
     * mutation LLH 1 inner random bit swap
     */
    private double applyHeuristic1(int solutionIndex, boolean real) {
        Solution solution = solutionMemory[solutionIndex];
        for (int i = 0; i < mrepeats; i++) {
            int agv = rng.nextInt(agvNum);
            while (solution.getLengthOfSingleAGV(agv) <= 1) {
                agv = rng.nextInt(agvNum);
            }
            int l = solution.getLengthOfSingleAGV(agv);
            int index1 = rng.nextInt(l);
            int index2 = rng.nextInt(l);
            while (index1 == index2) {
                index2 = rng.nextInt(l);
            }
            solution.bitswap(agv, index1, index2);
        }

        return real?getFunctionValue(solutionIndex):getFunctionValueWithSurrogate(solutionIndex);
    }

    /**
     * mutation LLH 2 outer random bit swap
     */
    private double applyHeuristic2(int solutionIndex, boolean real) {
        Solution solution = solutionMemory[solutionIndex];
        for (int i = 0; i < mrepeats; i++) {
            int agv1 = rng.nextInt(agvNum);
            int agv2 = rng.nextInt(agvNum);
            while (agv1 == agv2) {
                agv2 = rng.nextInt(agvNum);
            }
            int index1 = rng.nextInt(solution.getLengthOfSingleAGV(agv1));
            int index2 = rng.nextInt(solution.getLengthOfSingleAGV(agv2));
            solution.bitswap(agv1, index1, agv2, index2);
        }

        return real?getFunctionValue(solutionIndex):getFunctionValueWithSurrogate(solutionIndex);
    }


    /**
     * local search LLH 3 Steepest Descent Hill Climbing
     */
    private double applyHeuristic3(int solutionIndex, boolean real) {
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
     * local search LLH 4 First Bit Hill Climbing
     */
    private double applyHeuristic4(int solutionIndex, boolean real) {
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
     * ruin_recreate LLH 5 random ruin and recreate bit
     */
    private double applyHeuristic5(int solutionIndex, boolean real) {
        Solution solution = solutionMemory[solutionIndex];
        int[] indices = MyUtilities.shuffle(IntStream.range(0, orders.length).toArray(), rng);
        int end = (int) rprop * orders.length;

        for(int i = 0; i < end; i++) {
            solution.bitinsert(indices[i], indices[rng.nextInt(orders.length)]);
        }

        return real?getFunctionValue(solutionIndex):getFunctionValueWithSurrogate(solutionIndex);
    }

    /**
     * mutation LLH 6 best bit insert
     */
    private double applyHeuristic6(int solutionIndex, boolean real) {
        Solution solution = solutionMemory[solutionIndex];
        for (int i = 0; i < mrepeats; i++) {
            int agv1 = 0, agv2 = 0;
            double min = Double.POSITIVE_INFINITY, max = 0;
            for (int j = 0; j < agvNum; j++) {
                double temp = getFunctionValueOfSingleAGV(solutionIndex, j);
                if (temp < min) {
                    min = temp;
                    agv1 = j;
                }
                if (temp > max) {
                    max = temp;
                    agv2 = j;
                }
            }
            if (solution.getLengthOfSingleAGV(agv2) <= 1)
                System.out.println("error best insert");
            int index1 = rng.nextInt(solution.getLengthOfSingleAGV(agv1));
            int index2 = rng.nextInt(solution.getLengthOfSingleAGV(agv2));

            solution.bitinsert(agv1, index1, agv2, index2);
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
