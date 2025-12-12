package com.nw.maze;

import java.util.Comparator;
import java.util.PriorityQueue;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

// import org.springframework.CollectionUtils;

public class Main {

    private static final int directions[][] = { { -1, 0 }, { 0, 1 }, { 1, 0 }, { 0, -1 } };
    // private static final String FILE_NAME = "src/com/nw/maze/maze_101_101.txt";
    private static final String FILE_NAME = "m100_100.txt";
    private static final int DEFAULT_BLOCK_SIZE = 20;
    private static final int MIN_BLOCK_SIZE = 4;   // Smallest visible cell
    private static final int MAX_BLOCK_SIZE = 80;  // Cap so small mazes don't blow up

    MazeFrame frame;
    MazeData data;
    private volatile boolean renderEnabled = true;

    public void initFrame() {
        data = new MazeData(FILE_NAME);
        // collect maze files from current directory
        String[] mazeFiles = listMazeFiles();
        // compute block size so the maze fits in the user screen
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        int screenW = (int) screen.getWidth();
        int screenH = (int) screen.getHeight();
        // leave some space for OS chrome / controls
        int maxSceneW = Math.max(300, screenW - 150);
        int maxSceneH = Math.max(200, screenH - 150);
        int blockSize = DEFAULT_BLOCK_SIZE;
        // try to keep DEFAULT_BLOCK_SIZE but shrink until maze fits
        while (blockSize > MIN_BLOCK_SIZE && (blockSize * data.M() > maxSceneW || blockSize * data.N() > maxSceneH)) {
            blockSize--;
        }
        if (blockSize < MIN_BLOCK_SIZE) blockSize = MIN_BLOCK_SIZE;
        if (blockSize > MAX_BLOCK_SIZE) blockSize = MAX_BLOCK_SIZE;

        int frameW = Math.min(blockSize * data.M(), maxSceneW);
        int frameH = Math.min(blockSize * data.N(), maxSceneH);
        frame = new MazeFrame("Maze Solver", frameW, frameH, mazeFiles);
        frame.setResizable(true);
        frame.setLocationRelativeTo(null);
        frame.setControlListener(new MazeFrame.ControlListener() {
            @Override
            public void onRunRequested(String algorithmName) {
                frame.setControlsEnabled(false);
                new Thread(() -> {
                    try {
                        runWithAlgorithm(algorithmName);
                    } finally {
                        // Re-enable controls on EDT after run completes
                        javax.swing.SwingUtilities.invokeLater(() -> frame.setControlsEnabled(true));
                    }
                }, "maze-runner").start();
            }
            @Override
            public void onResetRequested() { resetState(); }
            @Override
            public void onMazeSelected(String mazeFile) {
                // Load maze on EDT to avoid concurrency issues with renderer
                javax.swing.SwingUtilities.invokeLater(() -> reloadMaze(mazeFile));
            }
        });
        frame.setSelectedMaze(FILE_NAME);
        frame.render(data);
        // Wait for user to press Run; no auto-execution
    }

    private String[] listMazeFiles() {
        java.io.File dir = new java.io.File(".");
        java.io.File[] files = dir.listFiles((d, name) -> name.toLowerCase().endsWith(".txt"));
        if (files == null) return new String[] { FILE_NAME };
        java.util.Arrays.sort(files, java.util.Comparator.comparing(java.io.File::getName));
        String[] names = new String[files.length];
        for (int i = 0; i < files.length; i++) names[i] = files[i].getName();
        return names;
    }

    private void reloadMaze(String fileName) {
        try {
            // Replace data with new maze, recompute frame size based on screen and maze size
            data = new MazeData(fileName);
            // compute block size & frame size same as init
            Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
            int screenW = (int) screen.getWidth();
            int screenH = (int) screen.getHeight();
            int maxSceneW = Math.max(300, screenW - 150);
            int maxSceneH = Math.max(200, screenH - 150);
            int blockSize = DEFAULT_BLOCK_SIZE;
            while (blockSize > MIN_BLOCK_SIZE && (blockSize * data.M() > maxSceneW || blockSize * data.N() > maxSceneH)) blockSize--;
            if (blockSize < MIN_BLOCK_SIZE) blockSize = MIN_BLOCK_SIZE;
            if (blockSize > MAX_BLOCK_SIZE) blockSize = MAX_BLOCK_SIZE;
            int frameW = Math.min(blockSize * data.M(), maxSceneW);
            int frameH = Math.min(blockSize * data.N(), maxSceneH);
            frame.setCanvasSize(frameW, frameH);
            frame.setTitle("Maze Solver - " + fileName);
            resetState();
            frame.render(data);
        } catch (Exception e) {
            System.err.println("Failed to load maze: " + fileName + " -> " + e.getMessage());
        }
    }

    private void resetState() {
        for (int i = 0; i < data.N(); i++) {
            for (int j = 0; j < data.M(); j++) {
                data.visited[i][j] = false;
                data.path[i][j] = false;
                data.result[i][j] = false;
            }
        }
        frame.setTitle("Maze Solver - reset");
        javax.swing.SwingUtilities.invokeLater(() -> frame.render(data));
    }

    private void runWithAlgorithm(String algo) {
        // Reset state arrays
        for (int i = 0; i < data.N(); i++) {
            for (int j = 0; j < data.M(); j++) {
                data.visited[i][j] = false;
                data.path[i][j] = false;
                data.result[i][j] = false;
            }
        }

        switch (algo) {
            case "BFS":
                runBFS();
                return;
            case "A*":
                runAStar();
                return;
            case "Genetic":
                runGeneticStub();
                return;
            case "Dijkstra":
            default:
                runDijkstra();
        }
    }

    private void runDijkstra() {
        // Dijkstra's algorithm on grid with per-cell weights
        int rows = data.N();
        int cols = data.M();
        int[][] dist = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) dist[i][j] = Integer.MAX_VALUE;
        }

        Node start = new Node(data.getEntranceX(), data.getEntranceY(), 0, null);
        if (data.inArea(start.x, start.y)) {
            dist[start.x][start.y] = 0;
        }

        PriorityQueue<Node> pq = new PriorityQueue<>(Comparator.comparingInt(n -> n.cost));
        pq.add(start);

        boolean isSolved = false;
        int visitedCount = 0;
        int visitedWeightSum = 0;
        long t0 = System.nanoTime();
        Node endNode = null;

        while (!pq.isEmpty()) {
            Node cur = pq.poll();
            if (data.visited[cur.x][cur.y]) continue; // finalized already
            data.visited[cur.x][cur.y] = true;
            if (data.weight != null) {
                int w = data.weight[cur.x][cur.y];
                if (w > 0) visitedWeightSum += w; else visitedWeightSum += 1;
            } else {
                visitedWeightSum += 1;
            }
            visitedCount++;

            setData(cur.x, cur.y, true); // visualize exploration

            if (cur.x == data.getExitX() && cur.y == data.getExitY()) {
                isSolved = true;
                endNode = cur;
                break;
            }

            for (int[] d : directions) {
                int nx = cur.x + d[0];
                int ny = cur.y + d[1];
                if (!data.inArea(nx, ny)) continue;
                if (data.getMazeChar(nx, ny) != MazeData.ROAD) continue; // skip walls
                if (data.visited[nx][ny]) continue;
                int stepCost = 1;
                if (data.weight != null) {
                    int w = data.weight[nx][ny];
                    stepCost = (w > 0 ? w : 1);
                }
                int newCost = (cur.cost == Integer.MAX_VALUE ? Integer.MAX_VALUE : cur.cost + stepCost);
                if (newCost < dist[nx][ny]) {
                    dist[nx][ny] = newCost;
                    pq.add(new Node(nx, ny, newCost, cur));
                }
            }
        }

        
        long t1 = System.nanoTime();

        if (isSolved && endNode != null) {
            int steps = findPath(endNode); // mark result path and count steps
            int totalCost = endNode.cost;
            long ms = (t1 - t0) / 1_000_000L;
            updateMetricsEDT(totalCost, steps, visitedCount, ms, "Dijkstra", visitedWeightSum);
        } else {
            updateMetricsEDT(null, null, visitedCount, (System.nanoTime()-t0)/1_000_000L, "Dijkstra", visitedWeightSum);
            System.out.println("The maze has NO solution!");
        }
        setData(-1, -1, false);
    }

    // Repair a child chromosome by appending / replacing the tail with shortest-path moves from its endpoint to goal.
    // Uses BFS from endpoint to goal (no rendering) and caps resulting chromosome to `maxGenomeLength`.
    private int[] repairChild(int[] child, MazeData data, int[][] distMap, int maxGenomeLength) {
        int rows = data.N(), cols = data.M();
        int x = data.getEntranceX(), y = data.getEntranceY();
        // simulate child to endpoint (skip invalid moves)
        for (int i = 0; i < child.length; i++) {
            int dir = Math.floorMod(child[i], 4);
            int nx = x + directions[dir][0], ny = y + directions[dir][1];
            if (!data.inArea(nx, ny) || data.getMazeChar(nx, ny) != MazeData.ROAD) continue;
            x = nx; y = ny;
        }
        int ex = data.getExitX(), ey = data.getExitY();
        // If endpoint can't reach goal, try to find nearest reachable cell (connected to exit)
        if (distMap == null) return child;
        if (distMap[x][y] == Integer.MAX_VALUE) {
            // attempt to BFS until we find a node that has distMap != Integer.MAX_VALUE
            int startIdx2 = x * cols + y;
            int[] prev2 = new int[rows * cols]; java.util.Arrays.fill(prev2, -1);
            java.util.ArrayDeque<Integer> q2 = new java.util.ArrayDeque<>(); q2.add(startIdx2); prev2[startIdx2] = startIdx2;
            int foundIdx = -1;
            int limitNodes = Math.max(1000, rows * cols / 4); // avoid huge searches
            int nodes = 0;
            while (!q2.isEmpty() && nodes < limitNodes) {
                int cur = q2.poll(); nodes++;
                int cx = cur / cols, cy = cur % cols;
                if (distMap[cx][cy] != Integer.MAX_VALUE) { foundIdx = cur; break; }
                for (int di = 0; di < directions.length; di++) {
                    int nx = cx + directions[di][0], ny = cy + directions[di][1];
                    if (!data.inArea(nx, ny)) continue;
                    if (data.getMazeChar(nx, ny) != MazeData.ROAD) continue;
                    int nidx = nx * cols + ny;
                    if (prev2[nidx] != -1) continue;
                    prev2[nidx] = cur; q2.add(nidx);
                }
            }
            if (foundIdx == -1) return child;
            // Reconstruct path from startIdx2 to foundIdx
            java.util.ArrayList<Integer> path2 = new java.util.ArrayList<>(); int cur2 = foundIdx; while (cur2 != startIdx2) { path2.add(cur2); cur2 = prev2[cur2]; if (cur2 == -1) break; }
            java.util.Collections.reverse(path2);
            // convert to direction indices
            int[] dirs2 = new int[path2.size()]; int px = x, py = y; for (int i=0;i<path2.size();i++) { int cidx = path2.get(i); int cx = cidx / cols, cy = cidx % cols; int dx = cx - px, dy = cy - py; int dirIndex=0; for (int k=0;k<directions.length;k++){ if (directions[k][0]==dx && directions[k][1]==dy){ dirIndex = k; break; } } dirs2[i]=dirIndex; px = cx; py = cy; }
            // attach this path and continue: replace tail as below using found path
            int pathLen2 = dirs2.length; int prefixLen2 = Math.max(0, child.length - pathLen2); int newLen2 = prefixLen2 + pathLen2; if (newLen2 > maxGenomeLength) { int allowedPathLen = Math.max(0, maxGenomeLength - prefixLen2); if (allowedPathLen <= 0) return child; int start = pathLen2 - allowedPathLen; int[] ndirs = new int[allowedPathLen]; System.arraycopy(dirs2, start, ndirs, 0, allowedPathLen); dirs2 = ndirs; pathLen2 = allowedPathLen; newLen2 = prefixLen2 + pathLen2; }
            int[] res2 = new int[newLen2]; System.arraycopy(child, 0, res2, 0, prefixLen2); System.arraycopy(dirs2, 0, res2, prefixLen2, pathLen2); child = res2;
            // recompute x,y endpoint by simulating child
            x = data.getEntranceX(); y = data.getEntranceY(); for (int i = 0; i < child.length; i++) { int dir = Math.floorMod(child[i], 4); int nx = x + directions[dir][0], ny = y + directions[dir][1]; if (!data.inArea(nx, ny) || data.getMazeChar(nx, ny) != MazeData.ROAD) continue; x=nx;y=ny; }
        }
        // BFS from endpoint to exit (shortest path)
        int startIdx = x * cols + y;
        int goalIdx = ex * cols + ey;
        int[] prev = new int[rows * cols];
        java.util.Arrays.fill(prev, -1);
        java.util.ArrayDeque<Integer> q = new java.util.ArrayDeque<>();
        q.add(startIdx); prev[startIdx] = startIdx;
        boolean found = false;
        while (!q.isEmpty()) {
            int cur = q.poll();
            if (cur == goalIdx) { found = true; break; }
            int cx = cur / cols, cy = cur % cols;
            for (int di = 0; di < directions.length; di++) {
                int nx = cx + directions[di][0], ny = cy + directions[di][1];
                if (!data.inArea(nx, ny)) continue;
                if (data.getMazeChar(nx, ny) != MazeData.ROAD) continue;
                int nidx = nx * cols + ny;
                if (prev[nidx] != -1) continue;
                prev[nidx] = cur;
                q.add(nidx);
            }
        }
        if (!found) return child;
        // reconstruct path from startIdx to goalIdx
        java.util.ArrayList<Integer> path = new java.util.ArrayList<>();
        int cur = goalIdx;
        while (cur != startIdx) {
            path.add(cur);
            cur = prev[cur];
            if (cur == -1) break;
        }
        java.util.Collections.reverse(path);
        // convert path to direction indices
        int[] dirs = new int[path.size()];
        int px = x, py = y;
        for (int i = 0; i < path.size(); i++) {
            int cidx = path.get(i);
            int cx = cidx / cols, cy = cidx % cols;
            int dx = cx - px, dy = cy - py;
            int dirIndex = 0;
            for (int k = 0; k < directions.length; k++) { if (directions[k][0] == dx && directions[k][1] == dy) { dirIndex = k; break; } }
            dirs[i] = dirIndex;
            px = cx; py = cy;
        }
        // replace tail of child to end with dirs, ensuring not to exceed maxGenomeLength
        int pathLen = dirs.length;
        int prefixLen = Math.max(0, child.length - pathLen);
        int newLen = prefixLen + pathLen;
        if (newLen > maxGenomeLength) {
            // trim path if too long
            int allowedPathLen = Math.max(0, maxGenomeLength - prefixLen);
            if (allowedPathLen <= 0) return child; // no room
            int start = pathLen - allowedPathLen;
            pathLen = allowedPathLen;
            int[] ndirs = new int[pathLen];
            System.arraycopy(dirs, start, ndirs, 0, pathLen);
            dirs = ndirs;
            newLen = prefixLen + pathLen;
        }
        int[] res = new int[newLen];
        System.arraycopy(child, 0, res, 0, prefixLen);
        System.arraycopy(dirs, 0, res, prefixLen, pathLen);
        return res;
    }

    private void runBFS() {
        ArrayDeque<Position> queue = new ArrayDeque<>();
        Position entrance = new Position(data.getEntranceX(), data.getEntranceY(), null);
        queue.add(entrance);
        if (data.inArea(entrance.x, entrance.y)) data.visited[entrance.x][entrance.y] = true;

        boolean isSolved = false;
        int visitedCount = 0;
        int visitedWeightSum = 0;
        long t0 = System.nanoTime();
        Position end = null;

        while (!queue.isEmpty()) {
            Position cur = queue.poll();
            visitedCount++;
            if (data.weight != null) {
                int w = data.weight[cur.x][cur.y];
                if (w > 0) visitedWeightSum += w; else visitedWeightSum += 1;
            } else {
                visitedWeightSum += 1;
            }
            setData(cur.x, cur.y, true);
            if (cur.x == data.getExitX() && cur.y == data.getExitY()) { isSolved = true; end = cur; break; }
            for (int[] d : directions) {
                int nx = cur.x + d[0], ny = cur.y + d[1];
                if (data.inArea(nx, ny) && !data.visited[nx][ny] && data.getMazeChar(nx,ny)==MazeData.ROAD) {
                    data.visited[nx][ny] = true;
                    queue.add(new Position(nx, ny, cur));
                }
            }
        }

        long t1 = System.nanoTime();
        if (isSolved && end != null) {
            int steps = findPath(end);
            long ms = (t1 - t0) / 1_000_000L;
            updateMetricsEDT(null, steps, visitedCount, ms, "BFS", visitedWeightSum);
        } else {
            updateMetricsEDT(null, null, visitedCount, (System.nanoTime()-t0)/1_000_000L, "BFS", visitedWeightSum);
        }
        setData(-1, -1, false);
    }

    private int findPath(Position p) {
        int steps = 0;
        Position cur = p;
        while (cur != null) {
            data.result[cur.x][cur.y] = true;
            cur = cur.prev;
            steps++;
        }
        return steps;
    }

    private void runAStar() {
        int rows = data.N(), cols = data.M();
        int[][] dist = new int[rows][cols];
        for (int i=0;i<rows;i++) for(int j=0;j<cols;j++) dist[i][j]=Integer.MAX_VALUE;
        Node start = new Node(data.getEntranceX(), data.getEntranceY(), 0, null);
        Node goal = new Node(data.getExitX(), data.getExitY(), 0, null);
        dist[start.x][start.y] = 0;

        Comparator<Node> cmp = (a,b) -> Integer.compare(a.cost + heuristic(a, goal), b.cost + heuristic(b, goal));
        PriorityQueue<Node> open = new PriorityQueue<>(cmp);
        open.add(start);

        boolean isSolved=false; int visitedCount=0; int visitedWeightSum=0; long t0=System.nanoTime(); Node end=null;
        while(!open.isEmpty()){
            Node cur = open.poll();
            if (data.visited[cur.x][cur.y]) continue;
            data.visited[cur.x][cur.y] = true; visitedCount++;
            if (data.weight != null) {
                int w = data.weight[cur.x][cur.y];
                if (w > 0) visitedWeightSum += w; else visitedWeightSum += 1;
            } else {
                visitedWeightSum += 1;
            }
            setData(cur.x, cur.y, true);
            if (cur.x==goal.x && cur.y==goal.y){ isSolved=true; end=cur; break; }
            for(int[]d:directions){
                int nx=cur.x+d[0], ny=cur.y+d[1];
                if(!data.inArea(nx,ny) || data.getMazeChar(nx,ny)!=MazeData.ROAD || data.visited[nx][ny]) continue;
                int stepCost = data.weight!=null && data.weight[nx][ny]>0 ? data.weight[nx][ny] : 1;
                int newCost = cur.cost + stepCost;
                if(newCost < dist[nx][ny]){ dist[nx][ny]=newCost; open.add(new Node(nx,ny,newCost,cur)); }
            }
        }
        long t1=System.nanoTime();
        if(isSolved && end!=null){ int steps=findPath(end); long ms=(t1-t0)/1_000_000L; updateMetricsEDT(end.cost, steps, visitedCount, ms, "A*", visitedWeightSum); }
        else { updateMetricsEDT(null, null, visitedCount, (System.nanoTime()-t0)/1_000_000L, "A*", visitedWeightSum); }
        setData(-1,-1,false);
    }

    private int heuristic(Node a, Node goal){
        // Manhattan distance as heuristic (weights ignored, admissible if weights>=1)
        return Math.abs(a.x - goal.x) + Math.abs(a.y - goal.y);
    }

    private void runGeneticStub() {
        // Placeholder for future GA: just show a message
        runGenetic();
    }

    private void runGenetic() {
        // Simple genetic algorithm: evolve sequences of moves with fitness = path cost to reach goal (penalize walls)
        int maxGenerations = 200;
        int populationSize = 200;
        // genome length will be dynamic based on maze size
        Random rnd = new Random(42);

        // Precompute distance map (BFS from goal) once for this run, used by the fitness function.
        final int[][] distMap = computeDistanceMap();
        // avoid frequent rendering during GA runs for performance and thread-safety
            try {
                renderEnabled = false;

        // dynamic genome length bounds: min and max derived from maze size
        int maxGenomeLength = data.N() * data.M(); // upper bound
        int minGenomeLength = Math.max(10, Math.min(100, (data.N() + data.M()) * 2)); // reasonable lower bound
        if (minGenomeLength > maxGenomeLength) minGenomeLength = maxGenomeLength;
        int startX = data.getEntranceX(), startY = data.getEntranceY();
        final int globalDistStart = (data.inArea(startX, startY) && distMap != null) ? distMap[startX][startY] : Integer.MAX_VALUE;
        if (globalDistStart != Integer.MAX_VALUE) {
            int suggested = Math.max(minGenomeLength, globalDistStart * 3);
            maxGenomeLength = Math.min(maxGenomeLength, suggested);
        } else {
            maxGenomeLength = Math.min(maxGenomeLength, 1000);
        }
        // Tune GA parameters for large mazes (scale up population and generations)
        int area = data.N() * data.M();
        if (area > 8000 || globalDistStart > 600) { populationSize = 400; maxGenerations = 600; }
        else if (area > 4000 || globalDistStart > 300) { populationSize = 300; maxGenerations = 500; }
        else if (area > 2000 || globalDistStart > 100) { populationSize = 200; maxGenerations = 400; }

        // Helper to evaluate a genome (add fitness field)
        class EvalResult { int cost; int steps; int visited; int visitedWeight; List<int[]> path; boolean reached; int bfsDistance; double fitness; }
        // Fitness constants
        final double STEP_PENALTY = 1.0;
        final double COST_PENALTY = 1.0;
        // penalty/fitness tuning constants
        final double LOOP_PENALTY_PER_REVISIT = 5.0;
        final double WEIGHT_PENALTY_SCALE = 0.05;
        final int INVALID_MOVE_PENALTY = 5;
        final double MIN_NON_REACHED_FITNESS = -1000000.0;
        // Dynamic reached bonus scaled to the maze (distance from start)
        final double REACHED_BONUS = (globalDistStart != Integer.MAX_VALUE) ? Math.max(2000.0, globalDistStart * (STEP_PENALTY + COST_PENALTY) * 3.0) : 2000.0;
        // Strengthen heuristics for large mazes
        final double VALID_RATIO_WEIGHT = 200.0; // * (steps/denom)
        final double PROGRESS_WEIGHT = 500.0; // * ((distStart - distCur)/distStart)
        // stamp-based seen mark for per-eval usage
        final int[][] seenMark = new int[data.N()][data.M()];
        final int[] stampHolder = new int[] { 1 };
        Function<int[], EvalResult> evaluate = genome -> {
            // manage stamp and possible overflow
            stampHolder[0] = stampHolder[0] + 1;
            if (stampHolder[0] == Integer.MAX_VALUE) {
                for (int i = 0; i < seenMark.length; i++) java.util.Arrays.fill(seenMark[i], 0);
                stampHolder[0] = 1;
            }
            final int stamp = stampHolder[0];
            int sx = startX, sy = startY;
            int x = sx, y = sy;
            int cost = 0, steps = 0, visited = 1, visitedWeightSum = 0;
            int revisits = 0;
            ArrayList<int[]> path = new ArrayList<>();
            path.add(new int[]{x,y});
            seenMark[x][y] = stamp;
            int startW = data.weight!=null?data.weight[x][y]:1; visitedWeightSum += (startW>0?startW:1);
            for (int i=0;i<genome.length;i++) {
                int dirIndex = Math.floorMod(genome[i], 4);
                int[] d = directions[dirIndex];
                int nx = x + d[0], ny = y + d[1];
                if (!data.inArea(nx, ny) || data.getMazeChar(nx,ny)!=MazeData.ROAD) {
                    cost += INVALID_MOVE_PENALTY; // penalty for invalid move
                    continue;
                }
                int stepWeight = (data.weight!=null && data.weight[nx][ny] > 0) ? data.weight[nx][ny] : 1;
                cost += stepWeight;
                steps++;
                if (seenMark[nx][ny] == stamp) {
                    revisits++;
                } else {
                    seenMark[nx][ny] = stamp;
                    visited++;
                    visitedWeightSum += stepWeight;
                }
                x = nx; y = ny;
                path.add(new int[]{x,y});
                if (x==data.getExitX() && y==data.getExitY()) break;
            }
            EvalResult r = new EvalResult(); r.cost=cost; r.steps=steps; r.visited=visited; r.visitedWeight=visitedWeightSum; r.path=path; r.reached=(x==data.getExitX() && y==data.getExitY());
            // Use precomputed distance map; if out-of-area or unreachable, keep Integer.MAX_VALUE
            if (!data.inArea(x, y)) r.bfsDistance = Integer.MAX_VALUE;
            else r.bfsDistance = distMap != null ? distMap[x][y] : bfsDistance(x, y);
            // Compute fitness
            int distStart = globalDistStart;
            int denom = genome.length;
            if (distStart > 0 && distStart != Integer.MAX_VALUE) denom = Math.max(denom, Math.max(1, distStart * 2));
            double validRatioScore = denom > 0 ? (steps / (double) denom) * VALID_RATIO_WEIGHT : 0.0;
            double progressScore = 0.0;
            if (distStart > 0 && distStart != Integer.MAX_VALUE) {
                double distCur = r.bfsDistance == Integer.MAX_VALUE ? distStart : r.bfsDistance;
                // progress is stronger and proportional to progress weight
                progressScore = ((distStart - distCur) / (double) distStart) * PROGRESS_WEIGHT;
                if (progressScore < 0.0) progressScore = 0.0;
            }
            double loopPenalty = revisits * LOOP_PENALTY_PER_REVISIT;
            double weightPenalty = visitedWeightSum * WEIGHT_PENALTY_SCALE;
            if (r.reached) {
                r.fitness = REACHED_BONUS - (r.steps * STEP_PENALTY) - (r.cost * COST_PENALTY);
            } else {
                r.fitness = validRatioScore + progressScore - loopPenalty - weightPenalty;
                if (r.fitness < MIN_NON_REACHED_FITNESS) r.fitness = MIN_NON_REACHED_FITNESS;
            }
            return r;
        };

        // Initialize population
        List<int[]> pop = new ArrayList<>();
        for (int i=0;i<populationSize;i++){
            int len = minGenomeLength + (maxGenomeLength==minGenomeLength ? 0 : rnd.nextInt(maxGenomeLength - minGenomeLength + 1));
            int[] g=new int[len];
            for(int j=0;j<len;j++) g[j]=rnd.nextInt(4);
            pop.add(g);
        }

        double bestScore = Double.NEGATIVE_INFINITY; Integer bestCost = null; int bestSteps=0, bestVisited=0, bestVisitedWeight=0; List<int[]> bestPath=null; String algoName="Genetic";
        long t0 = System.nanoTime();
        final int TOURNAMENT_SIZE = 3; // 3-5 recommended
        final int UI_UPDATE_GENS = 50;
        for (int gen=0; gen<maxGenerations; gen++) {
            // Evaluate
            List<EvalResult> results = new ArrayList<>(populationSize);
            for (int[] g : pop) {
                try {
                    results.add(evaluate.apply(g));
                } catch (Exception ex) {
                    System.err.println("evaluate() threw: " + ex.getMessage());
                    // fallback: create a minimal EvalResult with very low fitness so it's ignored
                    EvalResult errR = new EvalResult(); errR.cost = Integer.MAX_VALUE; errR.steps = 0; errR.visited = 0; errR.visitedWeight = 0; errR.path = new ArrayList<>(); errR.reached = false; errR.bfsDistance = Integer.MAX_VALUE; errR.fitness = -10000.0; results.add(errR);
                }
            }
            // Select top elites by scanning (no full sort) and use tournament selection for parents
            List<int[]> next = new ArrayList<>();
            int eliteCount = Math.max(1, populationSize/5);
            boolean[] chosen = new boolean[populationSize];
            for (int e = 0; e < eliteCount; e++) {
                int bestIdx = -1; double bestVal = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < results.size(); i++) {
                    if (chosen[i]) continue;
                    if (results.get(i).fitness > bestVal) { bestVal = results.get(i).fitness; bestIdx = i; }
                }
                if (bestIdx >= 0) { chosen[bestIdx] = true; next.add(pop.get(bestIdx)); }
            }
            // Track best (scan for max)
            int bestIdx = -1; double curBestVal = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < results.size(); i++) { if (results.get(i).fitness > curBestVal) { curBestVal = results.get(i).fitness; bestIdx = i; } }
            EvalResult br = results.get(bestIdx);
            boolean improved = false;
            if (br.fitness > bestScore) { improved = true; bestScore = br.fitness; bestSteps = br.steps; bestVisited = br.visited; bestVisitedWeight = br.visitedWeight; bestPath = br.path; if (br.reached) { if (bestCost == null || br.cost < bestCost) bestCost = br.cost; } }
            // Crossover + mutation to refill
            while (next.size() < populationSize) {
                int p1idx = -1; double p1best = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < TOURNAMENT_SIZE; k++) {
                    int idx = rnd.nextInt(pop.size());
                    if (results.get(idx).fitness > p1best) { p1idx = idx; p1best = results.get(idx).fitness; }
                }
                int p2idx = -1; double p2best = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < TOURNAMENT_SIZE; k++) {
                    int idx = rnd.nextInt(pop.size());
                    if (results.get(idx).fitness > p2best) { p2idx = idx; p2best = results.get(idx).fitness; }
                }
                int[] p1 = pop.get(p1idx);
                int[] p2 = pop.get(p2idx);
                // child length can vary; average + small noise or random within bounds
                int childLen = minGenomeLength + (maxGenomeLength==minGenomeLength ? 0 : rnd.nextInt(maxGenomeLength - minGenomeLength + 1));
                int[] child = new int[childLen];
                // pick cut points in parents
                int cut1 = rnd.nextInt(p1.length + 1);
                int cut2 = rnd.nextInt(p2.length + 1);
                // copy prefix from parent1
                int copyFromP1 = Math.min(cut1, childLen);
                if (copyFromP1 > 0) System.arraycopy(p1, 0, child, 0, copyFromP1);
                // fill remaining from p2 starting at cut2 with wrap
                int pos = copyFromP1;
                while (pos < childLen) {
                    child[pos] = p2[(cut2 + (pos - copyFromP1)) % p2.length];
                    pos++;
                }
                // mutation (gene replacement / insertion / deletion)
                int numMutations = Math.max(1, child.length / 50);
                for (int m=0;m<numMutations;m++) {
                    double op = rnd.nextDouble();
                    if (op < 0.90) {
                        // replace
                        child[rnd.nextInt(child.length)] = rnd.nextInt(4);
                    } else if (op < 0.96 && child.length < maxGenomeLength) {
                        // insert
                        int at = rnd.nextInt(child.length + 1);
                        int[] tmp = new int[child.length + 1];
                        System.arraycopy(child, 0, tmp, 0, at);
                        tmp[at] = rnd.nextInt(4);
                        System.arraycopy(child, at, tmp, at + 1, child.length - at);
                        child = tmp;
                    } else if (child.length > minGenomeLength) {
                        // delete
                        int at = rnd.nextInt(child.length);
                        int[] tmp = new int[child.length - 1];
                        System.arraycopy(child, 0, tmp, 0, at);
                        System.arraycopy(child, at + 1, tmp, at, child.length - at - 1);
                        child = tmp;
                    }
                }
                 // Repair child to end in a traversable path if possible
                 try {
                     child = repairChild(child, data, distMap, maxGenomeLength);
                 } catch (Exception ex) {
                     System.err.println("repairChild() threw: " + ex.getMessage());
                 }
                 next.add(child);
             }
             pop = next;
             // Occasionally update UI with best metrics (every N gens or when improved)
            if (gen % UI_UPDATE_GENS == 0 || improved) updateMetricsEDT(bestCost, bestSteps, bestVisited, (System.nanoTime()-t0)/1_000_000L, algoName, bestVisitedWeight);
         }
         long t1 = System.nanoTime();
         // Render best path
         resetState();
         if (bestPath != null) {
             for (int[] cell : bestPath) {
                 setData(cell[0], cell[1], true);
             }
         }
        updateMetricsEDT(bestCost, bestSteps, bestVisited, (t1-t0)/1_000_000L, algoName, bestVisitedWeight);
        // re-enable rendering, then render final result
            } finally {
                renderEnabled = true;
                setData(-1, -1, false);
            }
        renderEnabled = true;
        setData(-1, -1, false);
     }

    // BFS distance for fitness calculation
    private int bfsDistance(int sx, int sy) {
        if (!data.inArea(sx, sy)) return Integer.MAX_VALUE;
        int rows = data.N(), cols = data.M();
        boolean[][] seen = new boolean[rows][cols];
        ArrayDeque<int[]> q = new ArrayDeque<>();
        q.add(new int[]{sx, sy, 0});
        seen[sx][sy] = true;
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0], y = cur[1], d = cur[2];
            if (x == data.getExitX() && y == data.getExitY()) return d;
            for (int[] dir : directions) {
                int nx = x + dir[0], ny = y + dir[1];
                if (!data.inArea(nx, ny)) continue;
                if (seen[nx][ny]) continue;
                if (data.getMazeChar(nx, ny) != MazeData.ROAD) continue;
                seen[nx][ny] = true;
                q.add(new int[]{nx, ny, d + 1});
            }
        }
        return Integer.MAX_VALUE;
    }

    private static class Position {
        int x, y; Position prev;
        Position(int x, int y, Position prev){ this.x=x; this.y=y; this.prev=prev; }
    }

    private int findPath(Node p) {
        int steps = 0;
        Node cur = p;
        while (cur != null) {
            data.result[cur.x][cur.y] = true;
            cur = cur.prev;
            steps++;
        }
        return steps;
    }

    private void setData(int x, int y, boolean isPath) {
        if (data.inArea(x, y)) {
            data.path[x][y] = isPath;
        }
        if (!renderEnabled) return;
        // Run the render on EDT, but keep the pause on this background thread
        javax.swing.SwingUtilities.invokeLater(() -> frame.render(data));
        MazeUtil.pause(frame.getDelayMs());
    }

    private void updateMetricsEDT(Integer cost, Integer steps, Integer visited, Long timeMs, String algoName, Integer visitedWeightSum) {
        javax.swing.SwingUtilities.invokeLater(() -> frame.updateMetrics(cost, steps, visited, timeMs, algoName, visitedWeightSum));
    }

    // No Position class needed after switching to Dijkstra's Node representation

    private class Node {
        private int x, y;
        private int cost;
        private Node prev;

        private Node(int x, int y, int cost, Node prev) {
            this.x = x;
            this.y = y;
            this.cost = cost;
            this.prev = prev;
        }
    }

    // Compute distance-from-exit map with BFS (distance in steps, Integer.MAX_VALUE = unreachable).
    private int[][] computeDistanceMap() {
        int rows = data.N(), cols = data.M();
        int[][] dist = new int[rows][cols];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) dist[i][j] = Integer.MAX_VALUE;
        int ex = data.getExitX(), ey = data.getExitY();
        if (!data.inArea(ex, ey)) return dist;
        ArrayDeque<int[]> q = new ArrayDeque<>();
        dist[ex][ey] = 0;
        q.add(new int[]{ex, ey});
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0], y = cur[1], d = dist[x][y];
            for (int[] dir : directions) {
                int nx = x + dir[0], ny = y + dir[1];
                if (!data.inArea(nx, ny)) continue;
                if (dist[nx][ny] != Integer.MAX_VALUE) continue;
                if (data.getMazeChar(nx, ny) != MazeData.ROAD) continue;
                dist[nx][ny] = d + 1;
                q.add(new int[]{nx, ny});
            }
        }
        return dist;
    }

    public static void main(String[] args) {
        new Main().initFrame();
    }

}
