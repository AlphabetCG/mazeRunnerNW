package com.nw.maze;

import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Rectangle2D;

public class MazeUtil {

	public static final Color Red = new Color(0xF44336);
	public static final Color LightBlue = new Color(0x03A9F4);
	public static final Color Yellow = new Color(0xFFEB3B);
	public static final Color White = new Color(0xFFFFFF);
	
	private Graphics2D g2d;

	private MazeUtil(Graphics2D g2d) {
		this.g2d = g2d;
	}
	
	public static MazeUtil getInstance(Graphics g) {
		return new MazeUtil((Graphics2D) g);
	}
	
	public static void pause(long time) {
		try {
			Thread.sleep(time);
		}catch (InterruptedException e) {
            e.printStackTrace();
        }
	}
	
	public void setColor(Color color) {
		this.g2d.setColor(color);
	}
	
	public void fillRectangle(int x, int y, int w, int h) {
		Rectangle2D rectangle = new Rectangle2D.Double(x, y, w, h);
		g2d.fill(rectangle);
	}

	public void drawCenteredString(String text, int x, int y, int w, int h) {
		if (text == null) return;
		int baseSize = Math.min(w, h);
		float fontSize = Math.max(10f, baseSize * 0.5f);
		Font font = g2d.getFont().deriveFont(Font.PLAIN, fontSize);
		g2d.setFont(font);
		FontMetrics fm = g2d.getFontMetrics(font);
		int tx = x + (w - fm.stringWidth(text)) / 2;
		int ty = y + (h - fm.getHeight()) / 2 + fm.getAscent();
		g2d.drawString(text, tx, ty);
	}

}
