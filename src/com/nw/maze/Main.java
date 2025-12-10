package com.nw.maze;

import java.util.Comparator;
import java.util.PriorityQueue;

// import org.springframework.CollectionUtils;

public class Main {

    private static final int directions[][] = { { -1, 0 }, { 0, 1 }, { 1, 0 }, { 0, -1 } };
    // private static final String FILE_NAME = "src/com/nw/maze/maze_101_101.txt";
    private static final String FILE_NAME = "m40_40.txt";
    private static final int BLOCK_SIZE = 20;

    MazeFrame frame;
    MazeData data;

    public void initFrame() {
        data = new MazeData(FILE_NAME);
        frame = new MazeFrame("Maze Solver", data.M() * BLOCK_SIZE, data.N() * BLOCK_SIZE);
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
        });
        frame.render(data);
        // Wait for user to press Run; no auto-execution
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
        frame.render(data);
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
            frame.updateMetrics(totalCost, steps, visitedCount, ms, "Dijkstra", visitedWeightSum);
        } else {
            frame.updateMetrics(null, null, visitedCount, (System.nanoTime()-t0)/1_000_000L, "Dijkstra", visitedWeightSum);
            System.out.println("The maze has NO solution!");
        }
        setData(-1, -1, false);
    }

    private void runBFS() {
        java.util.ArrayDeque<Position> queue = new java.util.ArrayDeque<>();
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
            frame.updateMetrics(null, steps, visitedCount, ms, "BFS", visitedWeightSum);
        } else {
            frame.updateMetrics(null, null, visitedCount, (System.nanoTime()-t0)/1_000_000L, "BFS", visitedWeightSum);
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
        if(isSolved && end!=null){ int steps=findPath(end); long ms=(t1-t0)/1_000_000L; frame.updateMetrics(end.cost, steps, visitedCount, ms, "A*", visitedWeightSum); }
        else { frame.updateMetrics(null, null, visitedCount, (System.nanoTime()-t0)/1_000_000L, "A*", visitedWeightSum); }
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
        final int maxGenerations = 200;
        final int populationSize = 100;
        final int genomeLength = data.N() * data.M(); // upper bound on steps
        java.util.Random rnd = new java.util.Random(42);

        // Helper to evaluate a genome
        class EvalResult { int cost; int steps; int visited; int visitedWeight; java.util.List<int[]> path; boolean reached; }
        java.util.function.Function<int[], EvalResult> evaluate = genome -> {
            // reset temp visited
            boolean[][] seen = new boolean[data.N()][data.M()];
            int x = data.getEntranceX(), y = data.getEntranceY();
            int cost = 0, steps = 0, visited = 0, visitedWeightSum = 0;
            java.util.ArrayList<int[]> path = new java.util.ArrayList<>();
            path.add(new int[]{x,y});
            seen[x][y] = true; visited++;
            int startW = data.weight!=null?data.weight[x][y]:1; visitedWeightSum += (startW>0?startW:1);
            for (int i=0;i<genome.length;i++) {
                int[] d = directions[genome[i]%4];
                int nx = x + d[0], ny = y + d[1];
                if (!data.inArea(nx, ny) || data.getMazeChar(nx,ny)!=MazeData.ROAD) {
                    cost += 5; // penalty for invalid move
                    continue;
                }
                x = nx; y = ny; steps++;
                int w = data.weight!=null?data.weight[x][y]:1; int stepCost = w>0?w:1; cost += stepCost;
                if (!seen[x][y]) { seen[x][y]=true; visited++; visitedWeightSum += stepCost; }
                path.add(new int[]{x,y});
                if (x==data.getExitX() && y==data.getExitY()) break;
            }
            EvalResult r = new EvalResult(); r.cost=cost; r.steps=steps; r.visited=visited; r.visitedWeight=visitedWeightSum; r.path=path; r.reached=(x==data.getExitX() && y==data.getExitY()); return r;
        };

        // Initialize population
        java.util.List<int[]> pop = new java.util.ArrayList<>();
        for (int i=0;i<populationSize;i++){ int[] g=new int[genomeLength]; for(int j=0;j<genomeLength;j++) g[j]=rnd.nextInt(4); pop.add(g); }

        int bestCost = Integer.MAX_VALUE, bestSteps=0, bestVisited=0, bestVisitedWeight=0; java.util.List<int[]> bestPath=null; String algoName="Genetic";
        long t0 = System.nanoTime();
        for (int gen=0; gen<maxGenerations; gen++) {
            // Evaluate
            java.util.List<EvalResult> results = new java.util.ArrayList<>(populationSize);
            for (int[] g : pop) results.add(evaluate.apply(g));
            // Select top 20%
            results.sort(java.util.Comparator.comparingInt(r -> r.cost));
            java.util.List<int[]> next = new java.util.ArrayList<>();
            int eliteCount = Math.max(1, populationSize/5);
            for (int i=0;i<eliteCount;i++) next.add(pop.get(i));
            // Track best
            EvalResult br = results.get(0);
            if (br.cost < bestCost) { bestCost=br.cost; bestSteps=br.steps; bestVisited=br.visited; bestVisitedWeight=br.visitedWeight; bestPath=br.path; }
            // Crossover + mutation to refill
            while (next.size() < populationSize) {
                int[] p1 = pop.get(rnd.nextInt(eliteCount));
                int[] p2 = pop.get(rnd.nextInt(eliteCount));
                int[] child = new int[genomeLength];
                int cut = rnd.nextInt(genomeLength);
                System.arraycopy(p1, 0, child, 0, cut);
                System.arraycopy(p2, cut, child, cut, genomeLength-cut);
                // mutation
                for (int m=0;m<genomeLength/20;m++) child[rnd.nextInt(genomeLength)] = rnd.nextInt(4);
                next.add(child);
            }
            pop = next;
            // Occasionally update UI with best metrics
            if (gen % 10 == 0) frame.updateMetrics(bestCost, bestSteps, bestVisited, (System.nanoTime()-t0)/1_000_000L, algoName, bestVisitedWeight);
        }
        long t1 = System.nanoTime();
        // Render best path
        resetState();
        if (bestPath != null) {
            for (int[] cell : bestPath) {
                setData(cell[0], cell[1], true);
            }
        }
        frame.updateMetrics(bestCost, bestSteps, bestVisited, (t1-t0)/1_000_000L, algoName, bestVisitedWeight);
        setData(-1, -1, false);
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
        frame.render(data);
        MazeUtil.pause(frame.getDelayMs());
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

    public static void main(String[] args) {
        new Main().initFrame();
    }

}
