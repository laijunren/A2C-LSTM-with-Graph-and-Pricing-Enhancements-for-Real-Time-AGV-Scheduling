package com.VRP;

public class Order {
    private int orderNum;
    private String startNode;
    private String endNode;
    private int startX;
    private int startY;
    private int startZ;
    private int endX;
    private int endY;
    private int endZ;

    public Order(int orderNum, String startNode, String endNode, int startZ, int endZ){
        this.orderNum = orderNum;
        this.startNode = startNode;
        this.endNode = endNode;
        this.startZ = startZ;
        this.endZ = endZ;
    }

    public Order(int orderNum, String startNode, String endNode, int startX, int startY, int startZ, int endX, int endY, int endZ){
        this.orderNum = orderNum;
        this.startNode = startNode;
        this.endNode = endNode;
        this.startX = startX;
        this.startY = startY;
        this.startZ = startZ;
        this.endX = endX;
        this.endY = endY;
        this.endZ = endZ;
    }

    public String getStartNode() {
        return startNode;
    }

    public String getEndNode() {
        return endNode;
    }

    public int getStartX() {
        return startX;
    }

    public int getStartY() {
        return startY;
    }

    public int getEndX() {
        return endX;
    }

    public int getEndY() {
        return endY;
    }

    public int getStartZ() {
        return startZ;
    }

    public int getEndZ() {
        return endZ;
    }
}
