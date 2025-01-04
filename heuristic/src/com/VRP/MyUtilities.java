package com.VRP;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

public class MyUtilities {

    /**
     * Shuffles a list of int's using a predefined random number generator!
     *
     * @param array The input array
     * @param random The random generator to use for shuffling
     * @return A new array containing shuffled elements of the input array
     */
    public static int[] shuffle(int[] array, Random random) {

        int[] shuffledArray = new int[array.length];
        System.arraycopy(array, 0, shuffledArray,0, array.length);

        for(int i = 0; i < shuffledArray.length; i++) {

            //swap ith index with random index
            int index = random.nextInt(shuffledArray.length);
            int temp = shuffledArray[i];
            shuffledArray[i] = shuffledArray[index];
            shuffledArray[index] = temp;
        }

        return shuffledArray;
    }

    /**
     * Save data into system files
     * @param filePath The output file name
     * @param data The data in string type
     */
    public static void saveData(String filePath, String data) {
        Path path = Paths.get("output/" + filePath);
        if(!Files.exists(path)) {
            try {
                Files.createFile(path);
                Files.write(path, data.getBytes());
            } catch (IOException e) {
                System.err.println("Could not create file at " + path.toAbsolutePath());
                System.err.println("Printing data to screen instead...");
                System.out.println(data);
            }
        } else {
            try {
                Files.write(path, data.getBytes());
            } catch (IOException e) {
                System.err.println("Could not open file at " + path.toAbsolutePath());
                System.err.println("Printing data to screen instead...");
                System.out.println(data);
            }
        }
    }

}
