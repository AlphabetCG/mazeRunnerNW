@echo off
javac -d bin -sourcepath src .\src\com\nw\maze\*.java
java -cp bin com.nw.maze.Main