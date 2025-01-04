package com.VRP;


public class Solution {
    public int[] variables;

    public Solution(int[] variables) {
        this.variables = variables;
    }

    /**
     * swap the value of two bits permanently
     */
    public void bitswap(int bitIndex1, int bitIndex2) {
        int order_no = variables[bitIndex1];
        variables[bitIndex1] = variables[bitIndex2];
        variables[bitIndex2] = order_no;
    }


    /**
     * @return a new deep copy of this solution
     */
    public Solution deepCopy() {
        int[] variables = this.variables.clone();
//        int[] ages = this.ages.clone();
        return new Solution(variables);
    }

    public int getVariable(int variableIndex) {
        if (variableIndex > variables.length)
            return -1;
        return this.variables[variableIndex];
    }

    public int getLength() {
        return this.variables.length;
    }

    public int[] toIntArray() {
        return this.variables.clone();
    }


    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int variable : variables) {
                sb.append(variable);
                sb.append(" ");
        }
        sb.append("\n");
        return sb.toString();
    }
}
