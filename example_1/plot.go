package main

import (
	"fmt"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"image/color"
	"log"
	"os"
	"path/filepath"
)

// Function to create or recreate the folder
func CreateOrEmptyOutFolder(folder string) {

	// Check if folder exists
	if _, err := os.Stat(folder); !os.IsNotExist(err) {
		// If it exists, remove it
		err := os.RemoveAll(folder)
		if err != nil {
			panic(err)
		}
	}

	// Create the folder
	err := os.Mkdir(folder, 0755)
	if err != nil {
		panic(err)
	}
}

// Function to plot the points and hyperplane using image/color
func PlotEpoch(folder string, epoch int, weights []float64, bias float64, inputs [][]float64, labels []int) {
	// Ensure the folder is created or recreated

	p := plot.New()

	p.Title.Text = fmt.Sprintf("Epoch %d - Perceptron Decision Boundary", epoch+1)
	p.X.Label.Text = "x1"
	p.Y.Label.Text = "x2"

	// Plot the OR dataset points
	pts := make(plotter.XYs, len(inputs))
	ptsRed := make(plotter.XYs, 0)
	ptsBlue := make(plotter.XYs, 0)

	for i := range inputs {
		pts[i].X = inputs[i][0]
		pts[i].Y = inputs[i][1]
		if labels[i] == 1 {
			ptsBlue = append(ptsBlue, pts[i])
		} else {
			ptsRed = append(ptsRed, pts[i])
		}
	}

	// Create scatter plot for red and blue points
	scatterRed, _ := plotter.NewScatter(ptsRed)
	scatterRed.GlyphStyle.Color = color.RGBA{R: 255, A: 255} // Red for class 0

	scatterBlue, _ := plotter.NewScatter(ptsBlue)
	scatterBlue.GlyphStyle.Color = color.RGBA{B: 255, A: 255} // Blue for class 1

	// Plot the decision boundary (hyperplane)
	line := plotter.NewFunction(func(x float64) float64 {
		return -(weights[0]*x + bias) / weights[1]
	})
	log.Print(line)
	line.Color = color.RGBA{G: 255, A: 255} // Green for hyperplane

	// Add scatter plot and line
	p.Add(scatterRed, scatterBlue, line)

	// Save the plot to the "plots" folder
	filePath := filepath.Join(folder, fmt.Sprintf("epoch_%03d.png", epoch+1))
	if err := p.Save(4*vg.Inch, 4*vg.Inch, filePath); err != nil {
		panic(err)
	}
}
