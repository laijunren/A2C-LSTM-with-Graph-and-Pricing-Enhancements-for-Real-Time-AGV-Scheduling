package com.VRP;


public class Solution {
    public int[][] variables;

    public Solution(int[][] variables) {
        this.variables = variables;
    }

    public void bitinsert(int agvIndex1, int bitIndex1, int agvIndex2, int bitIndex2) {
        int[] arr = new int[variables[agvIndex1].length+1];
        int temp = getVariable(agvIndex2, bitIndex2);
        int i = 0;
        int j = 0;
        while (i < arr.length) {
            if (bitIndex1 == i)
                arr[i++] = temp;
            else
                arr[i++] = variables[agvIndex1][j++];
        }
        variables[agvIndex1] = arr;
        bitremove(agvIndex2, bitIndex2);
    }

    private void bitremove(int agvIndex, int bitIndex) {
        int[] arr = new int[variables[agvIndex].length-1];
        int i = 0;
        int j = 0;
        while (j < arr.length) {
            if (bitIndex == i)
                i++;
            else
                arr[j++] = variables[agvIndex][i++];
        }
        variables[agvIndex] = arr;
    }

    public void bitinsert(int generalIndex1, int generalIndex2) {
        int agv1=0, agv2=0, index1=0, index2=0;
        for (int i = 0; i < variables.length; i++) {
            for (int j = 0; j < variables[i].length; j++) {
                if (generalIndex1-- == 0) {
                    agv1 = i;
                    index1 = j;
                }
                if (generalIndex2-- == 0) {
                    agv2 = i;
                    index2 = j;
                }
            }
        }
        if (getLengthOfSingleAGV(agv2) > 1)
            bitinsert(agv1, index1, agv2, index2);
    }

    /**
     * swap the value of two bits permanently
     */
    public void bitswap(int agvIndex, int bitIndex1, int bitIndex2) {
        int order_no = variables[agvIndex][bitIndex1];
        variables[agvIndex][bitIndex1] = variables[agvIndex][bitIndex2];
        variables[agvIndex][bitIndex2] = order_no;
    }

    /**
     * swap the value of two bits permanently
     */
    public void bitswap(int agv1, int bitIndex1, int agv2, int bitIndex2) {
        int order_no = variables[agv1][bitIndex1];
        variables[agv1][bitIndex1] = variables[agv2][bitIndex2];
        variables[agv2][bitIndex2] = order_no;
    }

    public void bitswap(int generalIndex1, int generalIndex2) {
        int agv1=0, agv2=0, index1=0, index2=0;
        for (int i = 0; i < variables.length; i++) {
            for (int j = 0; j < variables[i].length; j++) {
                if (generalIndex1-- == 0) {
                    agv1 = i;
                    index1 = j;
                }
                if (generalIndex2-- == 0) {
                    agv2 = i;
                    index2 = j;
                }
            }
        }
        int temp = getVariable(agv1, index1);
        setVariable(agv1, index1, getVariable(agv2, index2));
        setVariable(agv2, index2, temp);
    }

    /**
     * @return a new deep copy of this solution
     */
    public Solution deepCopy() {
        int[][] variables = new int[this.variables.length][];
//        int[] ages = new int[this.variables.length];
        for (int i = 0; i < variables.length; i++) {
            variables[i] = this.variables[i].clone();
//            ages[i] = this.ages[i];
        }
        return new Solution(variables);
    }

    /**
     * set a certain bit with certain value
     */
    public void setVariable(int agvIndex, int variableIndex, int b) {
        this.variables[agvIndex][variableIndex] = b;
//        this.ages[variableIndex] = 0;
    }

    public int getVariable(int agvIndex, int variableIndex) {
        if (agvIndex > variables.length)
            return -1;
        if (variableIndex > variables[agvIndex].length)
            return -1;
        return this.variables[agvIndex][variableIndex];
    }

    public int getLengthOfSingleAGV(int agvIndex) {
        return this.variables[agvIndex].length;
    }

    public int getAGVIndex(int generalIndex) {
        for (int i = 0; i < variables.length; i++) {
            for (int j : variables[i]) {
                if (generalIndex-- == 0) return i;
            }
        }
        System.out.println("error: general index out of range");
        return 0;
    }

    public int[][] toIntArray() {
        int[][] variables = new int[this.variables.length][];
        for (int i = 0; i < variables.length; i++) {
            variables[i] = this.variables[i].clone();
        }
        return variables;
    }


    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int[] variable : variables) {
            for (int i : variable) {
                sb.append(i);
                sb.append(" ");
            }
            sb.append("\n");
        }
        return sb.toString();
    }
}
