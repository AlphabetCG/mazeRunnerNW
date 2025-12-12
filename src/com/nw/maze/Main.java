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

// Add missing imports for project classes
import com.nw.maze.MazeUtil;
import com.nw.maze.MazeFrame;
import com.nw.maze.MazeData;

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
        // PURE GA: no BFS guidance, no repairChild. Fitness uses valid-move ratio + Manhattan progress.
        final int maxGenerations = 800;
        final int populationSize = 300;
        Random rnd = new Random(42);

        // genome length bounds (safer caps than N*M)
        int rows = data.N(), cols = data.M();
        int sx = data.getEntranceX(), sy = data.getEntranceY();
        int ex = data.getExitX(), ey = data.getExitY();

        // Manhattan start distance
        int tmpDistStartMan = Math.abs(sx - ex) + Math.abs(sy - ey);
        final int distStartMan = (tmpDistStartMan <= 0) ? 1 : tmpDistStartMan;

        int hardCap = 2000; // prevent explosion
        int maxGenomeLength = Math.min(rows * cols, Math.max(100, distStartMan * 6));
        maxGenomeLength = Math.min(maxGenomeLength, hardCap);
        int minGenomeLength = Math.max(20, Math.min(200, (rows + cols))); // avoid too short for big maps
        if (minGenomeLength > maxGenomeLength) minGenomeLength = Math.min(maxGenomeLength, Math.max(10, distStartMan * 2));

        // fitness constants
        final double REACHED_BONUS = 1500.0;
        final double STEP_PENALTY = 2.0;
        final double COST_PENALTY = 0.5;
        final int INVALID_MOVE_PENALTY = 5;
        final double LOOP_PENALTY_PER_REVISIT = 1.2;
        final double WEIGHT_PENALTY_SCALE = 0.05;

        // GA operators
        final int TOURNAMENT_SIZE = 3;
        final int UI_UPDATE_GENS = 50;
        final int DIVERSITY_INJECTION_INTERVAL = 40; // inject random individuals every N generations
        final int DIVERSITY_INJECTION_COUNT = Math.max(1, populationSize / 20);

        // seenMark stamp for fast per-eval visit marking
        final int[][] seenMark = new int[rows][cols];
        final int[] stampHolder = new int[] { 1 };

        class EvalResult { double fitness; int steps; int cost; int visited; int visitedWeight; boolean reached; int manhattan; java.util.List<int[]> path; }

        Function<int[], EvalResult> evaluate = genome -> {
            // stamp
            stampHolder[0] = (stampHolder[0] == Integer.MAX_VALUE - 5) ? 1 : stampHolder[0] + 1;
            int stamp = stampHolder[0];

            int x = sx, y = sy;
            int steps = 0, cost = 0, visited = 1, visitedWeightSum = 0;
            int revisits = 0;
            java.util.ArrayList<int[]> path = new java.util.ArrayList<>();
            path.add(new int[] { x, y });
            seenMark[x][y] = stamp;
            int startW = data.weight != null ? data.weight[x][y] : 1;
            visitedWeightSum += (startW > 0 ? startW : 1);

            for (int i = 0; i < genome.length; i++) {
                int dir = Math.floorMod(genome[i], 4);
                int nx = x + directions[dir][0];
                int ny = y + directions[dir][1];
                if (!data.inArea(nx, ny) || data.getMazeChar(nx, ny) != MazeData.ROAD) {
                    cost += INVALID_MOVE_PENALTY;
                    continue;
                }
                int stepWeight = (data.weight != null && data.weight[nx][ny] > 0) ? data.weight[nx][ny] : 1;
                cost += stepWeight;
                steps++;
                if (seenMark[nx][ny] == stamp) revisits++;
                else { seenMark[nx][ny] = stamp; visited++; visitedWeightSum += stepWeight; }
                x = nx; y = ny;
                path.add(new int[] { x, y });
                if (x == ex && y == ey) break;
            }

            EvalResult r = new EvalResult();
            r.steps = steps; r.cost = cost; r.visited = visited; r.visitedWeight = visitedWeightSum; r.path = path;
            r.reached = (path.size() > 0 && path.get(path.size()-1)[0] == ex && path.get(path.size()-1)[1] == ey);
            r.manhattan = Math.abs(x - ex) + Math.abs(y - ey);

            // compute fitness (pure GA style)
            double validRatioScore = genome.length > 0 ? (steps / (double) genome.length) * 50.0 : 0.0;
            validRatioScore = Math.max(0.0, Math.min(50.0, validRatioScore));

            double progressScore = 0.0;
            if (distStartMan > 0) {
                double distNow = r.manhattan;
                progressScore = ((distStartMan - distNow) / (double) distStartMan) * 50.0;
                if (progressScore < 0.0) progressScore = 0.0;
            }

            double loopPenalty = revisits * LOOP_PENALTY_PER_REVISIT;
            double weightPenalty = visitedWeightSum * WEIGHT_PENALTY_SCALE;

            if (r.reached) {
                r.fitness = REACHED_BONUS - (r.steps * STEP_PENALTY) - (r.cost * COST_PENALTY);
            } else {
                r.fitness = validRatioScore + progressScore - loopPenalty - weightPenalty;
                // clamp to reasonable range
                if (r.fitness < -1000.0) r.fitness = -1000.0;
            }
            return r;
        };

        // Initialize population
        List<int[]> pop = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            int len = minGenomeLength + (maxGenomeLength == minGenomeLength ? 0 : rnd.nextInt(maxGenomeLength - minGenomeLength + 1));
            int[] g = new int[len];
            for (int j = 0; j < len; j++) g[j] = rnd.nextInt(4);
            pop.add(g);
        }

        // GA main loop
        double bestScore = Double.NEGATIVE_INFINITY;
        List<int[]> bestPath = null;
        Integer bestCost = null;
        int bestSteps = 0, bestVisited = 0, bestVisitedWeight = 0;
        long t0 = System.nanoTime();

        for (int gen = 0; gen < maxGenerations; gen++) {
            // Evaluate
            List<EvalResult> results = new ArrayList<>(pop.size());
            for (int[] g : pop) {
                try { results.add(evaluate.apply(g)); }
                catch (Throwable e) {
                    // defensive: print and degrade fitness
                    e.printStackTrace();
                    EvalResult bad = new EvalResult(); bad.fitness = -10000; bad.reached = false; results.add(bad);
                }
            }

            // selection of elites (scan)
            int eliteCount = Math.max(1, populationSize / 10);
            boolean[] picked = new boolean[results.size()];
            List<int[]> next = new ArrayList<>();
            for (int k = 0; k < eliteCount; k++) {
                int bestIdx = -1; double bestVal = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < results.size(); i++) {
                    if (picked[i]) continue;
                    if (results.get(i).fitness > bestVal) { bestVal = results.get(i).fitness; bestIdx = i; }
                }
                if (bestIdx >= 0) { picked[bestIdx] = true; next.add(pop.get(bestIdx)); }
            }

            // track best
            int bestIdx = -1; double curBestVal = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < results.size(); i++) { if (results.get(i).fitness > curBestVal) { curBestVal = results.get(i).fitness; bestIdx = i; } }
            EvalResult br = results.get(bestIdx);
            boolean improved = false;
            if (br.fitness > bestScore) {
                improved = true;
                bestScore = br.fitness;
                bestPath = br.path;
                bestCost = br.cost;
                bestSteps = br.steps;
                bestVisited = br.visited;
                bestVisitedWeight = br.visitedWeight;
            }

            // refill via tournament selection + crossover + mutation (no repair)
            while (next.size() < populationSize) {
                // tournament select p1
                int p1idx = rnd.nextInt(pop.size());
                for (int t = 1; t < TOURNAMENT_SIZE; t++) {
                    int cand = rnd.nextInt(pop.size());
                    if (results.get(cand).fitness > results.get(p1idx).fitness) p1idx = cand;
                }
                // tournament select p2
                int p2idx = rnd.nextInt(pop.size());
                for (int t = 1; t < TOURNAMENT_SIZE; t++) {
                    int cand = rnd.nextInt(pop.size());
                    if (results.get(cand).fitness > results.get(p2idx).fitness) p2idx = cand;
                }
                int[] p1 = pop.get(p1idx), p2 = pop.get(p2idx);

                // child length random
                int childLen = minGenomeLength + (maxGenomeLength == minGenomeLength ? 0 : rnd.nextInt(maxGenomeLength - minGenomeLength + 1));
                int[] child = new int[childLen];
                int cut1 = rnd.nextInt(p1.length + 1);
                int cut2 = rnd.nextInt(p2.length + 1);
                int copyFromP1 = Math.min(cut1, childLen);
                if (copyFromP1 > 0) System.arraycopy(p1, 0, child, 0, copyFromP1);
                int pos = copyFromP1;
                while (pos < childLen) { child[pos] = p2[(cut2 + (pos - copyFromP1)) % p2.length]; pos++; }

                // mutation
                int numMutations = Math.max(1, child.length / 20);
                for (int m = 0; m < numMutations; m++) {
                    double op = rnd.nextDouble();
                    if (op < 0.80) {
                        child[rnd.nextInt(child.length)] = rnd.nextInt(4);
                    } else if (op < 0.90 && child.length < maxGenomeLength) {
                        int at = rnd.nextInt(child.length + 1);
                        int[] tmp = new int[child.length + 1];
                        System.arraycopy(child, 0, tmp, 0, at);
                        tmp[at] = rnd.nextInt(4);
                        System.arraycopy(child, at, tmp, at + 1, child.length - at);
                        child = tmp;
                    } else if (child.length > minGenomeLength) {
                        int at = rnd.nextInt(child.length);
                        int[] tmp = new int[child.length - 1];
                        System.arraycopy(child, 0, tmp, 0, at);
                        System.arraycopy(child, at + 1, tmp, at, child.length - at - 1);
                        child = tmp;
                    }
                }

                next.add(child);
            }

            // diversity injection every N generations
            if (gen > 0 && gen % DIVERSITY_INJECTION_INTERVAL == 0) {
                for (int i = 0; i < DIVERSITY_INJECTION_COUNT && i < next.size(); i++) {
                    int idx = rnd.nextInt(next.size());
                    int len = minGenomeLength + (maxGenomeLength == minGenomeLength ? 0 : rnd.nextInt(maxGenomeLength - minGenomeLength + 1));
                    int[] g = new int[len];
                    for (int j = 0; j < len; j++) g[j] = rnd.nextInt(4);
                    next.set(idx, g);
                }
            }

            pop = next;

            // UI update occasionally (do on EDT)
            if (gen % UI_UPDATE_GENS == 0 || improved) {
                final double reportScore = bestScore;
                final Integer reportCost = bestCost;
                final Integer reportSteps = bestSteps;
                final Integer reportVisited = bestVisited;
                final Integer reportVisitedWeightLocal = bestVisitedWeight;
                final Long reportElapsed = (System.nanoTime() - t0) / 1_000_000L;
                javax.swing.SwingUtilities.invokeLater(() -> {
                    frame.updateMetrics(reportCost, reportSteps, reportVisited, reportElapsed, "Genetic(Pure)", reportVisitedWeightLocal);
                });
            }

            // early exit if perfect (reached and minimal cost)
            if (bestPath != null && bestPath.size() > 0) {
                int[] last = bestPath.get(bestPath.size()-1);
                if (last[0] == ex && last[1] == ey && bestScore >= REACHED_BONUS - (distStartMan * STEP_PENALTY)) {
                    // good enough
                    break;
                }
            }
        } // end generations

        long t1 = System.nanoTime();
        // render best path (on EDT)
        resetState();
        if (bestPath != null) {
            for (int[] cell : bestPath) {
                setData(cell[0], cell[1], true);
            }
        }
        frame.updateMetrics(bestCost, bestSteps, bestVisited, (t1-t0)/1_000_000L, "Genetic(Pure)", bestVisitedWeight);
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
