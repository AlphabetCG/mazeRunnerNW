package com.nw.maze;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JButton;
import javax.swing.JSlider;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

public class MazeFrame  extends JFrame{
	
	private int canvasWidth;
	private int canvasHeight;
	
	private MazeData data;

	// Controls
	private JComboBox<String> algorithmBox;
	private JButton runButton;
	private JButton resetButton;
	private JSlider speedSlider;
	private ControlListener controlListener;
	// Metrics labels
	private JLabel costLabel;
	private JLabel stepsLabel;
	private JLabel visitedLabel;
	private JLabel timeLabel;
	private JLabel visitedWeightLabel;
	
	public MazeFrame(String title, int canvasWidth, int canvasHeight) {
		super(title);
		this.canvasWidth = canvasWidth;
		this.canvasHeight = canvasHeight;

		// Build UI
		MazeCanvas canvas = new MazeCanvas();
		JPanel root = new JPanel(new BorderLayout());
		root.add(buildControlPanel(), BorderLayout.NORTH);
		root.add(canvas, BorderLayout.CENTER);
		this.setContentPane(root);
		
		this.pack();
		this.setResizable(false);
        this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        this.setVisible(true);
	}
	
	public void render(MazeData data) {
		this.data = data;
		repaint();
	}

	public void setControlListener(ControlListener listener) {
		this.controlListener = listener;
	}

	public void setControlsEnabled(boolean enabled) {
		if (algorithmBox != null) algorithmBox.setEnabled(enabled);
		if (runButton != null) runButton.setEnabled(enabled);
	}

	private JPanel buildControlPanel() {
		JPanel panel = new JPanel();
		panel.add(new JLabel("Algorithm:"));
		this.algorithmBox = new JComboBox<>(new String[]{
			"Dijkstra", "A*", "BFS", "Genetic"
		});
		panel.add(algorithmBox);
		runButton = new JButton("Run");
		runButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if (controlListener != null) {
					String name = (String) algorithmBox.getSelectedItem();
					controlListener.onRunRequested(name);
				}
			}
		});
		panel.add(runButton);
		resetButton = new JButton("Reset");
		resetButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if (controlListener != null) {
					controlListener.onResetRequested();
				}
			}
		});
		panel.add(resetButton);

		panel.add(new JLabel("Speed:"));
		speedSlider = new JSlider(0, 200, 30); // delay ms
		speedSlider.setMajorTickSpacing(50);
		speedSlider.setMinorTickSpacing(10);
		speedSlider.setPaintTicks(true);
		panel.add(speedSlider);

		// Metrics display
		costLabel = new JLabel("Cost: -");
		stepsLabel = new JLabel("Steps: -");
		visitedLabel = new JLabel("Visited: -");
		timeLabel = new JLabel("Time: -ms");
		visitedWeightLabel = new JLabel("Visited Weight: -");
		panel.add(costLabel);
		panel.add(stepsLabel);
		panel.add(visitedLabel);
		panel.add(timeLabel);
		panel.add(visitedWeightLabel);
		return panel;
	}
	
	public void paint(MazeUtil util) {
		int w = canvasWidth / data.M();
		int h = canvasHeight / data.N();
		for(int i = 0; i < data.N(); i++) {
			for(int j = 0; j < data.M(); j++) {
				if(data.getMazeChar(i, j) == MazeData.WALL) {
					util.setColor(MazeUtil.LightBlue);
				}else {
					util.setColor(MazeUtil.White);
				}
				if(data.path[i][j]) {
					util.setColor(MazeUtil.Yellow);
				}
				if(data.result[i][j]) {
					util.setColor(MazeUtil.Red);
				}
				util.fillRectangle(j * w, i * h, w, h);

				// Draw weight for road cells
				if (data.getMazeChar(i, j) == MazeData.ROAD && data.weight != null && data.weight[i][j] > 0) {
					util.setColor(Color.BLACK);
					util.drawCenteredString(Integer.toString(data.weight[i][j]), j * w, i * h, w, h);
				}
			}
		}
	}
	
	private class MazeCanvas extends JPanel{

		@Override
		protected void paintComponent(Graphics g) {
			super.paintComponent(g);
			MazeUtil util = MazeUtil.getInstance(g);
			if(data != null) {
				MazeFrame.this.paint(util);
			}
		}

		@Override
		public Dimension getPreferredSize() {
			return new Dimension(canvasWidth, canvasHeight);
		}
		
		
	}

	public static interface ControlListener {
		void onRunRequested(String algorithmName);
		void onResetRequested();
	}

	public int getDelayMs() {
		return speedSlider != null ? speedSlider.getValue() : 10;
	}

	public void updateMetrics(Integer cost, Integer steps, Integer visited, Long timeMs, String algoName, Integer visitedWeightSum) {
		if (algoName != null) {
			setTitle("Maze Solver - " + algoName);
		}
		if (costLabel != null) costLabel.setText("Cost: " + (cost != null ? cost : "-"));
		if (stepsLabel != null) stepsLabel.setText("Steps: " + (steps != null ? steps : "-"));
		if (visitedLabel != null) visitedLabel.setText("Visited: " + (visited != null ? visited : "-"));
		if (timeLabel != null) timeLabel.setText("Time: " + (timeMs != null ? timeMs : "-") + "ms");
		if (visitedWeightLabel != null) visitedWeightLabel.setText("Visited Weight: " + (visitedWeightSum != null ? visitedWeightSum : "-"));
	}
}
