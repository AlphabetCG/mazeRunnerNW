package com.nw.maze;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class MazeData {
	
	private int N, M;
	private int entranceX, entranceY;
	private int exitX, exitY;
	private char[][] maze;
	public boolean[][] path;
	public boolean[][] visited;
	public boolean[][] result;
	public int[][] weight;
	public static final char WALL ='#';
	public static final char ROAD = ' ';
	
	public MazeData(String fileName) {
		Scanner scanner = null;
		try {
			File file = new File(fileName);
			FileInputStream fis = new FileInputStream(file);
			scanner = new Scanner(new BufferedInputStream(fis), "UTF-8");

			ArrayList<String> lines = new ArrayList<>();
			while (scanner.hasNextLine()) {
				lines.add(scanner.nextLine());
			}
			if (lines.isEmpty()) {
				throw new IllegalArgumentException("Maze file is empty: " + fileName);
			}

			String first = lines.get(0).trim();
			if (first.matches("^\\d+\\s+\\d+$")) {
				// Old format: first line has N M, followed by N lines of raw characters
				String[] nm = first.split("\\s+");
				N = Integer.parseInt(nm[0]);
				M = Integer.parseInt(nm[1]);
				maze = new char[N][M];
				path = new boolean[N][M];
				visited = new boolean[N][M];
				result = new boolean[N][M];
				weight = new int[N][M];

				this.exitX = N - 2;
				this.exitY = M - 1;

				for (int i = 0; i < N; i++) {
					String line = lines.get(i + 1);
					for (int j = 0; j < M; j++) {
						maze[i][j] = line.charAt(j);
					}
				}

				// Detect entrance (left edge) and exit (right edge) if present
				for (int i = 0; i < N; i++) {
					if (maze[i][0] == ROAD) { entranceX = i; entranceY = 0; break; }
				}
				for (int i = 0; i < N; i++) {
					if (maze[i][M - 1] == ROAD) { exitX = i; exitY = M - 1; break; }
				}

				// Initialize default weights: walls=-1, roads in [1,9]
				Random rand = new Random(42);
				for (int i = 0; i < N; i++) {
					for (int j = 0; j < M; j++) {
						if (maze[i][j] == ROAD) {
							weight[i][j] = 1 + rand.nextInt(9);
						} else {
							weight[i][j] = -1;
						}
					}
				}
			} else {
				// New weighted format: no header, tokens per cell (#, S, G, or "number")
				N = lines.size();
				M = countCells(lines.get(0));
				maze = new char[N][M];
				path = new boolean[N][M];
				visited = new boolean[N][M];
				result = new boolean[N][M];
				weight = new int[N][M];

				// Defaults in case S/G not present
				entranceX = 1; entranceY = 1;
				exitX = N - 2; exitY = M - 2;

				for (int i = 0; i < N; i++) {
					parseWeightedLineInto(i, lines.get(i));
				}
			}
		} catch (Exception e) {
			throw new RuntimeException("Failed to load maze file: " + fileName, e);
		} finally {
			if (scanner != null) {
				scanner.close();
			}
		}
	}

	// Count number of cells in a weighted-format line
	private int countCells(String line) {
		int i = 0; int count = 0; int len = line.length();
		while (i < len) {
			char c = line.charAt(i);
			if (c == '#') { count++; i++; }
			else if (c == 'S' || c == 'G') { count++; i++; }
			else if (c == '"') {
				// consume until next quote
				i++; while (i < len && line.charAt(i) != '"') i++; if (i < len && line.charAt(i) == '"') i++;
				count++;
			} else {
				// skip any separators/spaces just in case
				i++;
			}
		}
		return count;
	}

	// Parse one weighted-format line into row 'row'
	private void parseWeightedLineInto(int row, String line) {
		int i = 0; int col = 0; int len = line.length();
		while (i < len && col < M) {
			char c = line.charAt(i);
			if (c == '#') {
				maze[row][col] = WALL;
				weight[row][col] = -1;
				i++; col++;
			} else if (c == 'S') {
				maze[row][col] = ROAD;
				weight[row][col] = 1;
				entranceX = row; entranceY = col;
				i++; col++;
			} else if (c == 'G') {
				maze[row][col] = ROAD;
				weight[row][col] = 1;
				exitX = row; exitY = col;
				i++; col++;
			} else if (c == '"') {
				int start = ++i;
				while (i < len && line.charAt(i) != '"') i++;
				String num = line.substring(start, Math.min(i, len));
				try {
					int w = Integer.parseInt(num);
					maze[row][col] = ROAD;
					weight[row][col] = w;
				} catch (NumberFormatException ex) {
					maze[row][col] = ROAD;
					weight[row][col] = 1;
				}
				if (i < len && line.charAt(i) == '"') i++;
				col++;
			} else {
				// skip unexpected char
				i++;
			}
		}
	}
	
	public boolean inArea(int x, int y) {
		return x >= 0 && x < N && y >=0 && y < M;
	}
	
	public char getMazeChar(int i, int j) {
		return maze[i][j];
	}

	public int getExitX() {
		return exitX;
	}

	public int getExitY() {
		return exitY;
	}

	public int getEntranceX() {
		return entranceX;
	}

	public int getEntranceY() {
		return entranceY;
	}

	public int N() {
		return N;
	}

	public int M() {
		return M;
	}
	
	

}
